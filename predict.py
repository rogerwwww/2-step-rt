from logging import basicConfig, INFO, info, warning
import pandas as pd
import numpy as np
import pickle
import json
import re
from tap import Tap
from typing import List, Optional, Literal, Tuple, Union
import pickle
import io
import torch
import yaml
import sys
from time import time

from utils import Data
from mapping import LADModel

class DataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes' and not torch.cuda.is_available():
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def load_model(path: str, all_in_one:bool=False):
    path = path + '.pt' if not path.endswith('pt') else path
    if (torch.cuda.is_available()):
        model = torch.load(path, weights_only=False)
    else:
        model = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        if hasattr(model, 'ranknet_encoder'):
            model.ranknet_encoder.embedding.gnn.device = torch.device('cpu')
            model.ranknet_encoder.embedding.gnn.encoder[0].device = torch.device('cpu')
        try:
            model.encoder.encoder[0].device = torch.device('cpu')
        except:
            pass
    if (all_in_one):
        return model
    path = re.sub(r'_ep\d+(\.pt)?$', '', re.sub(r'\.pt$', '', path)) # for ep_save
    data = DataUnpickler(open(f'{path}_data.pkl', 'rb')).load()
    config = json.load(open(f'{path}_config.json'))
    return model, data, config

class PredictArgs(Tap):
    input_compounds: str                 # TSV file with `smiles` and `rt` columns
    input_metadata: Optional[str] = None # YAML for original metadata-based models
    method: Optional[str] = None         # CN/HP/CP/HN/C18P for a packaged specialized model
    model: str                           # model to load
    gpu: bool = False                    # whether to use GPU for predictions
    output_roi: bool = False             # include Retention Order Index (ROI) in output
    output_anchors: bool = False         # include anchors in output
    out: Optional[str] = None            # where to write the output (TSV format). If not specified, output will be written to screen.
    batch_size: int = 256                # adjust according to available VRAM
    repo_root_folder: str = '../RepoRT/' # location of RepoRT, needed for HSM/Tanaka database
    verbose: bool = False                # more info on what is being done internally

if __name__ == '__main__':
    args = PredictArgs().parse_args()
    if (args.verbose):
        basicConfig(level=INFO)
    if (args.gpu):
        torch.set_default_device('cuda')

    # load model
    info('load model...')
    model = load_model(args.model, all_in_one=True)
    if 'specialized' in getattr(model, 'extra_storage', {}):
        from specialized import METHODS, structure_identity, predict_frame, set_model_device
        if args.method not in METHODS or args.input_metadata is not None:
            raise ValueError('Specialized prediction requires --method CN|HP|CP|HN|C18P and no --input_metadata')
        device = torch.device('cuda' if args.gpu else 'cpu')
        if args.gpu and not torch.cuda.is_available():
            raise ValueError('--gpu requested but CUDA is unavailable')
        set_model_device(model, device)
        frame = pd.read_csv(args.input_compounds, sep='\t')
        smiles_column = 'smiles' if 'smiles' in frame else 'smiles.std' if 'smiles.std' in frame else None
        if smiles_column is None:
            raise ValueError('Input compounds require a smiles or smiles.std column')
        original_columns = frame.columns.tolist()
        # Validate structures without deduplicating or reordering the query rows.
        identities = {s: structure_identity(s)[0] for s in frame[smiles_column].unique()}
        query = frame.copy()
        query['smiles'] = query[smiles_column].map(identities)
        query['dataset_id'] = args.method
        result = predict_frame(model, query, batch_size=args.batch_size)
        frame['rt_pred'] = result.rt_pred.to_numpy()
        if args.output_roi:
            frame['roi'] = result.roi.to_numpy()
        output_columns = original_columns + (['roi'] if args.output_roi and 'roi' not in original_columns else [])
        if 'rt_pred' not in output_columns:
            output_columns.append('rt_pred')
        frame[output_columns].to_csv(args.out if args.out else sys.stdout, sep='\t', index=False)
        sys.exit(0)
    if args.method is not None or args.input_metadata is None:
        raise ValueError('Original models require --input_metadata and do not accept --method')
    data_args = model.extra_storage['data_args']
    data_args['repo_root_folder'] =  args.repo_root_folder
    sysfeature_scaler = model.extra_storage['sysfeature_scaler']

    info('load input data...')
    d = Data(**data_args)
    metadata = yaml.load(open(args.input_metadata), yaml.SafeLoader)
    # flatten metadata
    [metadata] = pd.json_normalize(metadata, sep='.').to_dict(orient='records')
    if (    # column not specified and neither HSM/Tanaka parameters
            ('column.name' not in metadata and
             any(f'column.{field}' not in metadata for field in d.hsm_fields + d.tanaka_fields))
            # or pH not specified
            or not any(ph_column in metadata for ph_column in
                       ['eluent.A.pH', 'eluent.B.pH'])):
       warning(f'Important metadata is missing. Required: 1. Column name or manual HSM/Tanaka parameters, 2. pH. '
               f'Provided: {", ".join(metadata)}. '
               'This metadata is necessary; only the `setupagnostic` and `nocolumn` model will work. '
               'For these models the predictions will necessarily be worse!')
    original_input_columns = open(args.input_compounds).readlines()[0].strip().split('\t')
    d.add_external_data(args.input_compounds, metadata=metadata,
                        remove_nan_rts=False, tab_mode=True,
                        isomeric=True, split_type='evaluate')

    info('computing features')
    d.compute_features(mode=None, add_descs=False)

    info('computing graphs')
    d.compute_graphs()
    info('(fake) splitting data')
    d.split_data((0, 0))
    if (sysfeature_scaler is not None):
        info('standardize data')
        d.standardize(other_descriptor_scaler=None, other_sysfeature_scaler=sysfeature_scaler,
                      can_create_new_scaler=False)
    ((train_graphs, train_x, train_sys, train_y),
     (val_graphs, val_x, val_sys, val_y),
     (test_graphs, test_x, test_sys, test_y)) = d.get_split_data()
    X = np.concatenate((train_x, test_x, val_x)).astype(np.float32)
    X_sys = np.concatenate((train_sys, test_sys, val_sys)).astype(np.float32)
    Y = np.concatenate((train_y, test_y, val_y))
    info(f'done preprocessing. predicting ROIs...')
    t0 = time()
    graphs = np.concatenate((train_graphs, test_graphs, val_graphs))
    if (hasattr(model, 'add_sys_features') and model.add_sys_features):
        from utils_newbg import sysfeature_graph
        info('add system features to graphs')
        smiles_list = d.df.iloc[np.concatenate((d.train_indices, d.test_indices, d.val_indices))]['smiles'].tolist()
        assert len(graphs) == len(smiles_list)
        from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim
        if (model.add_sys_features_mode == 'bond'):
            set_extra_bond_fdim(train_sys.shape[1])
        elif (model.add_sys_features_mode == 'atom'):
            set_extra_atom_fdim(train_sys.shape[1])
        for i in range(len(graphs)):
            graphs[i] = sysfeature_graph(smiles_list[i], graphs[i], X_sys[i],
                                         bond_or_atom=model.add_sys_features_mode)
    t0_actual = time()
    preds = model.predict(graphs, X, X_sys, batch_size=args.batch_size,
                          ret_features=False, prog_bar=args.verbose)
    t1 = time() - t0
    t1_actual = time() - t0_actual
    info(f'done predicting ROIs, took {t1:.1}({t1_actual:.1})s ({t1/len(preds):.2}s per instance [{t1_actual/len(preds):.2}s], #={len(preds)}). predicting retention times...')
    d.df['roi'] = preds[np.arange(len(d.df.rt))[ # restore correct order
        np.argsort(np.concatenate([d.train_indices, d.test_indices, d.val_indices]))]]
    # predict retention times right here
    d.df['roi2'] = d.df.roi ** 2 # for LAD model
    # anchors are all data points with annotated retention time, discarding the void volume
    data_anchors = d.df.loc[d.df.rt > metadata['column.t0']]
    data_to_predict = d.df.loc[pd.isna(d.df.rt)].copy()
    info(f'building mapping using {len(data_anchors)} anchors, predicting {len(data_to_predict)} retention times...')
    t0 = time()
    mapping_model = LADModel(data_anchors, ols_after=True, ols_discard_if_negative=True, ols_drop_mode='2*median')
    import pickle
    with open(args.input_compounds.replace('.tsv', '_model.pkl'), 'wb') as out:
        pickle.dump(mapping_model, out)
    data_to_predict['rt_pred'] = mapping_model.get_mapping(data_to_predict.roi)
    t1 = time() - t0
    info(f'took {t1:.1}s ({len(data_to_predict)/t1:.2}s per instance)')
    if (args.output_roi):
        original_input_columns += ['roi']
    out_df = data_to_predict[
        # [c for c in data_to_predict.columns if any(['smiles' in c, 'inchi' in c.lower(), 'name' in c, c.startswith('rt_pred'), c.startswith('id')])]
        original_input_columns + ['rt_pred']
    ]
    if (args.output_anchors):
        out_df = pd.concat([data_anchors[original_input_columns], out_df])
    if (args.out is None):
        info(f'showing output.')
        out_df.to_csv(sys.stdout, sep='\t')
    else:
        info(f'saving to {args.out}.')
        out_df.to_csv(args.out, sep='\t')
