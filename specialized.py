"""Shared data and inference contract for the five-method RT model."""
import hashlib

import numpy as np
import pandas as pd
from rdkit import Chem

METHODS = {
    'CN': ('negative', 'C18'),
    'HP': ('positive', 'HILIC'),
    'CP': ('positive', 'C8'),
    'HN': ('negative', 'HILIC'),
    'C18P': ('positive', 'C18'),
}
FEATURE_NAMES = ['mode_positive', 'mode_negative', 'method_C8', 'method_C18', 'method_HILIC']
PARTITIONS = ['train', 'test', 'val']


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def encode_methods(methods, method_mapping=None):
    method_mapping = METHODS if method_mapping is None else method_mapping
    rows = []
    for code in methods:
        if code not in method_mapping:
            raise ValueError(f'Unknown method {code!r}; expected one of {list(method_mapping)}')
        mode, method = method_mapping[code]
        if mode not in {'positive', 'negative'} or method not in {'C8', 'C18', 'HILIC'}:
            raise ValueError(f'Invalid method metadata for {code!r}: {method_mapping[code]!r}')
        rows.append([mode == 'positive', mode == 'negative', method == 'C8',
                     method == 'C18', method == 'HILIC'])
    return np.asarray(rows, dtype=np.float32).reshape(-1, len(FEATURE_NAMES))


def structure_identity(smiles):
    """Keep isomeric structures; connectivity alone determines partition membership."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError('missing_smiles')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('invalid_smiles')
    key = Chem.MolToInchiKey(mol)
    if not key or len(key.split('-')[0]) != 14:
        raise ValueError('inchikey_generation_failed')
    return Chem.MolToSmiles(mol, isomericSmiles=True), key


def assign_partitions(frame, reference, unmatched_policy='error'):
    if reference.inchikey2d.duplicated().any():
        raise ValueError('Reference mapping must contain exactly one row per 2D InChIKey')
    if not set(reference.split_type).issubset(PARTITIONS):
        raise ValueError('Unknown reference partition')
    mapping = reference.set_index('inchikey2d').split_type
    result = frame.copy()
    result['split_type'] = result.inchikey2d.map(mapping)
    missing = result.split_type.isna()
    if missing.any() and unmatched_policy != 'train':
        raise ValueError(f'{result.loc[missing, "inchikey2d"].nunique()} keys lack reference assignments')
    result.loc[missing, 'split_type'] = 'train'
    result['assignment_source'] = np.where(missing, 'rt_only_training', 'molgpu_split_1')
    assert_disjoint(result)
    return result


def assert_disjoint(frame):
    if frame.inchikey2d.isna().any() or not frame.inchikey2d.str.fullmatch('[A-Z]{14}').all():
        raise ValueError('Missing or malformed 2D InChIKey')
    if not set(frame.split_type).issubset(PARTITIONS):
        raise ValueError('Unknown partition')
    if (frame.groupby('inchikey2d').split_type.nunique() > 1).any():
        raise ValueError('2D InChIKey overlap between partitions')


def fit_rt_scalers(frame):
    scalers = {}
    train = frame.loc[frame.split_type == 'train']
    for method in METHODS:
        values = train.loc[train.dataset_id == method, 'rt'].to_numpy(dtype=float)
        if len(values) < 3 or not np.isfinite(values).all():
            raise ValueError(f'{method}: need at least three finite training RTs')
        lo, hi = float(values.min()), float(values.max())
        if hi <= lo:
            raise ValueError(f'{method}: training RT range is zero')
        scalers[method] = {'min': lo, 'max': hi, 'range': hi - lo,
                           'fit_partition': 'train', 'n_rows': len(values)}
    return scalers


def scale_rt(values, scaler):
    return (np.asarray(values, dtype=float) - scaler['min']) / scaler['range']


def inverse_rt(values, scaler):
    return np.asarray(values, dtype=float) * scaler['range'] + scaler['min']


def add_scaled_rt(frame, scalers):
    result = frame.copy()
    result['rt_normalized'] = np.nan
    for method in METHODS:
        mask = result.dataset_id == method
        result.loc[mask, 'rt_normalized'] = scale_rt(result.loc[mask, 'rt'], scalers[method])
    return result


def validate_prepared(frame, scalers, method_mapping=None):
    required = {'smiles', 'inchikey2d', 'dataset_id', 'rt', 'rt_normalized', 'split_type',
                'assignment_source'}
    if not required.issubset(frame):
        raise ValueError(f'Missing prepared columns: {sorted(required - set(frame))}')
    assert_disjoint(frame)
    if not set(frame.dataset_id).issubset(METHODS):
        raise ValueError('Unknown method in prepared data')
    if not np.isfinite(frame.rt).all() or (frame.rt <= 0).any():
        raise ValueError('Prepared RTs must be finite and positive')
    additions = frame.assignment_source == 'rt_only_training'
    if not (frame.loc[additions, 'split_type'] == 'train').all():
        raise ValueError('RT-only additions must belong to training')
    if not np.allclose(frame[FEATURE_NAMES].to_numpy(), encode_methods(frame.dataset_id, method_mapping)):
        raise ValueError('Prepared method features disagree with method codes')
    expected = add_scaled_rt(frame, scalers)
    if not np.allclose(frame.rt_normalized, expected.rt_normalized):
        raise ValueError('Prepared RT scaling disagrees with stored scalers')
    fitted = fit_rt_scalers(frame)
    for method in METHODS:
        if (fitted[method]['n_rows'] != scalers[method]['n_rows']
                or scalers[method]['fit_partition'] != 'train'
                or not np.allclose([fitted[method][k] for k in ['min', 'max', 'range']],
                                   [scalers[method][k] for k in ['min', 'max', 'range']],
                                   rtol=1e-12, atol=1e-12)):
            raise ValueError(f'{method}: scaler must be fitted on training rows only')


def graph_inputs(frame, cache=None, method_mapping=None):
    from dmpnn_graph import dmpnn_graph
    cache = {} if cache is None else cache
    for smiles in frame.smiles.unique():
        if smiles not in cache:
            cache[smiles] = dmpnn_graph(smiles)
    graphs = np.empty(len(frame), dtype=object)
    graphs[:] = [cache[s] for s in frame.smiles]
    return graphs, np.empty((len(frame), 0), dtype=np.float32), encode_methods(frame.dataset_id, method_mapping)


def set_model_device(model, device):
    import torch
    device = torch.device(device)
    model.to(device)
    model.encoder.device = device
    for encoder in model.encoder.encoder:
        encoder.device = device
    return model


def predict_frame(model, frame, batch_size=256, cache=None):
    """Use stored training calibration; input RT annotations never refit the mapping."""
    storage = getattr(model, 'extra_storage', {}).get('specialized')
    if storage is None or storage['feature_names'] != FEATURE_NAMES:
        raise ValueError('Not a compatible specialized model')
    result = frame.copy()
    if result.empty:
        result['roi'] = pd.Series(dtype=float)
        result['rt_pred'] = pd.Series(dtype=float)
        return result
    # Frozen checkpoints retain their training encoding, including legacy CN=C8.
    roi = model.predict(*graph_inputs(result, cache, storage['methods']), batch_size=batch_size)
    result['roi'] = roi
    result['rt_pred'] = np.nan
    for method in result.dataset_id.unique():
        mask = result.dataset_id == method
        coefficients = storage['mappings'][method]['coefficients']
        values = result.loc[mask, 'roi'].to_numpy()
        normalized = coefficients[0] + coefficients[1] * values + coefficients[2] * values ** 2
        result.loc[mask, 'rt_pred'] = inverse_rt(normalized, storage['rt_scalers'][method])
    return result
