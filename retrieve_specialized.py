"""Same-formula PubChem retrieval using a frozen specialized RT model."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import gc
import hashlib
import json
import multiprocessing
import pickle
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import torch
import torch.nn.functional as F

from dmpnn_graph import dmpnn_graph, dmpnn_batch
from predict import load_model
from specialized import encode_methods, inverse_rt, set_model_device, sha256
from train_specialized import load_prepared


def molecular_formula(smiles):
    """Match ms-pred's uncharged_formula: strip charge suffix, keep H count."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid query SMILES')
    return re.split(r'[+-]', CalcMolFormula(mol))[0]


def build_candidate(item):
    formula, smiles, supplied_key = item
    RDLogger.DisableLog('rdApp.*')
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, 'invalid_smiles'
        if re.split(r'[+-]', CalcMolFormula(mol))[0] != formula:
            return None, 'formula_mismatch'
        key = Chem.MolToInchiKey(mol)
        if not key or len(key.split('-')[0]) != 14:
            return None, 'invalid_inchikey'
        if key[:14] != supplied_key[:14]:
            return None, 'mapping_key_mismatch'
        return (smiles, key[:14], dmpnn_graph(mol)), None
    except Exception as exc:
        return None, type(exc).__name__


def build_candidate_batch(tasks):
    """Construct tensors in workers to avoid transferring nested feature lists."""
    torch.set_num_threads(1)
    valid, exclusions = [], {}
    for task in tasks:
        candidate, error = build_candidate(task)
        if error:
            exclusions[error] = exclusions.get(error, 0) + 1
        else:
            valid.append(candidate)
    if not valid:
        return [], None, exclusions
    batch = dmpnn_batch([c[2] for c in valid])
    # The encoder consumes only tensors/scopes; retaining original graphs would
    # redundantly serialize every Python feature list alongside those tensors.
    batch.mol_graphs = []
    return [(c[0], c[1]) for c in valid], batch, exclusions


def candidate_batches(pool, formula, entries, batch_size, workers):
    tasks = ([(formula, s, key) for s, key in entries[start:start + batch_size]]
             for start in range(0, len(entries), batch_size))
    pending = deque()
    for _ in range(2 * workers):
        batch = next(tasks, None)
        if batch is not None:
            pending.append(pool.submit(build_candidate_batch, batch))
    while pending:
        result = pending.popleft().result()
        batch = next(tasks, None)
        if batch is not None:
            pending.append(pool.submit(build_candidate_batch, batch))
        yield result


def predict_methods(model, graphs, methods):
    """Encode a structure once; apply the unchanged model heads per method."""
    if model.add_sys_features or model.include_special_atom_features or model.extra_features_dim:
        raise ValueError('Shared encoder inference requires the specialized zero-descriptor model')
    model.eval()
    device = next(model.parameters()).device
    storage = model.extra_storage['specialized']
    output = {}
    prepared = graphs if hasattr(graphs, 'a_scope') else dmpnn_batch(graphs)
    n_structures = len(prepared.a_scope)
    with torch.no_grad():
        embedding = model.encoder([prepared])
        extra = embedding.new_empty((n_structures, 0))
        for method in methods:
            sysf = torch.as_tensor(encode_methods([method] * n_structures, storage['methods']), device=device)
            enc = embedding
            if not model.no_sys_layers:
                if model.sys_blowup:
                    sysf = F.relu(model.sys_blowup_layer(sysf))
                pv = torch.cat([enc, extra, sysf], 1)
                for layer in model.hidden_pv:
                    pv = F.relu(layer(pv))
                pv = model.dropout_pv(pv)
                enc = torch.cat([enc, extra, pv], 1) if model.res_conn_enc else torch.cat([extra, pv], 1)
            else:
                enc = torch.cat([extra, enc], 1)
            for layer in model.hidden:
                enc = F.relu(layer(enc))
            roi = model.ident(model.dropout_rank(enc)).flatten()
            if not model.no_sigmoid_roi:
                roi = torch.sigmoid(roi)
            roi = roi.cpu().numpy()
            coef = storage['mappings'][method]['coefficients']
            rt = inverse_rt(coef[0] + coef[1] * roi + coef[2] * roi ** 2,
                            storage['rt_scalers'][method])
            if not np.isfinite(rt).all():
                raise ValueError('Nonfinite candidate predictions')
            output[method] = rt
    return output


def rank_query(keys, predictions, true_key, observed_rt, top_ks):
    """Collapse 2D identities by their best structure error, then rank error/key."""
    keys = np.asarray(keys)
    errors = np.abs(np.asarray(predictions) - observed_rt)
    grouped = pd.Series(errors).groupby(keys, sort=True).min()
    n = len(grouped)
    base = {'n_candidates': n, 'truth_present': true_key in grouped.index}
    if not base['truth_present']:
        return {**base, 'rank': None, 'rank_optimistic': None, 'rank_pessimistic': None,
                'truth_error_minutes': None,
                **{f'top_{k}': False for k in top_ks},
                **{f'random_top_{k}': 0. for k in top_ks}}
    score = grouped[true_key]
    lower = int((grouped < score).sum())
    equal = grouped == score
    rank = 1 + lower + int((equal & (grouped.index < true_key)).sum())
    return {**base, 'rank': rank, 'rank_optimistic': lower + 1,
            'rank_pessimistic': lower + int(equal.sum()), 'truth_error_minutes': float(score),
            **{f'top_{k}': rank <= k for k in top_ks},
            **{f'random_top_{k}': min(k, n) / n for k in top_ks}}


def retrieval_metrics(frame, top_ks):
    present = frame.loc[frame.truth_present]
    def rates(x):
        return {f'top_{k}': float(x[f'top_{k}'].mean()) if len(x) else None for k in top_ks}
    keys = frame.groupby('inchikey2d')[[f'top_{k}' for k in top_ks]].mean()
    return {'rows': len(frame), 'keys': int(frame.inchikey2d.nunique()),
            'covered_rows': len(present), 'coverage': len(present) / len(frame),
            'all_queries': rates(frame), 'covered_queries': rates(present),
            'equal_weight_per_key': {f'top_{k}': float(keys[f'top_{k}'].mean()) for k in top_ks},
            'random_all_queries': {f'top_{k}': float(frame[f'random_top_{k}'].mean()) for k in top_ks},
            'optimistic_tie_bounds': {f'top_{k}': float((frame.rank_optimistic <= k).mean()) for k in top_ks},
            'pessimistic_tie_bounds': {f'top_{k}': float((frame.rank_pessimistic <= k).mean()) for k in top_ks},
            'candidate_count': {k: float(v) for k, v in frame.n_candidates.describe().items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data_dir', default='data/specialized')
    parser.add_argument('--partition', choices=['test', 'val'], default='test')
    parser.add_argument('--pubchem_map', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--top_ks', type=int, nargs='+', default=[1, 5, 10, 20, 50, 100])
    parser.add_argument('--probe_formulas', type=int, default=0,
                        help='Positive values make a explicitly labelled subset probe')
    parser.add_argument('--extract_only', action='store_true')
    parser.add_argument('--formula_subset', help='Previously extracted subset from this evaluator')
    parser.add_argument('--shard_index', type=int, default=0)
    parser.add_argument('--shard_count', type=int, default=1)
    args = parser.parse_args()
    if min(args.workers, args.batch_size, args.threads, *args.top_ks) <= 0 or args.probe_formulas < 0:
        raise ValueError('Sizes must be positive')
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError('Invalid formula shard')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.set_num_threads(args.threads)
    device = 'cuda' if args.gpu else 'cpu'
    if args.gpu and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    model = load_model(args.model, all_in_one=True)
    set_model_device(model, device)
    storage = model.extra_storage['specialized']
    if storage['smoke_only'] or storage['data_audit_sha256'] != sha256(Path(args.data_dir) / 'audit.json'):
        raise ValueError('Require full model and matching dataset provenance')
    all_rows, _, _ = load_prepared(args.data_dir, storage['methods'])
    frame = all_rows.loc[all_rows.split_type == args.partition].copy()
    if set(frame.inchikey2d) & set(storage['training_keys']):
        raise ValueError('Test/train 2D-key overlap')
    smiles_formula = {s: molecular_formula(s) for s in frame.smiles.unique()}
    frame['formula'] = frame.smiles.map(smiles_formula)
    formulas = sorted(frame.formula.unique())
    if args.probe_formulas:
        formulas = formulas[:args.probe_formulas]
        frame = frame.loc[frame.formula.isin(formulas)].copy()
    formulas = formulas[args.shard_index::args.shard_count]
    frame = frame.loc[frame.formula.isin(formulas)].copy()
    map_path = Path(args.pubchem_map)
    map_stat = map_path.stat()
    config = {**vars(args), 'model_sha256': sha256(args.model),
              'data_audit_sha256': storage['data_audit_sha256'],
              'pubchem_map_bytes': map_stat.st_size, 'pubchem_map_mtime_ns': map_stat.st_mtime_ns,
              'identity': '2D InChIKey, minimum RT error over structures sharing a key',
              'ranking': 'absolute RT error in minutes; exact ties broken by 2D key alphabetically',
              'formula': 'RDKit CalcMolFormula, charge suffix stripped to match ms-pred map',
              'no_truth_injection': True, 'no_candidate_cap': True,
              'probe_only': bool(args.probe_formulas), 'n_formulas': len(formulas)}
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Output directory belongs to a different evaluation')
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    # A cached subset avoids repeatedly materializing the full 8.7 GB pickle.
    subset_path = Path(args.formula_subset) if args.formula_subset else out / 'formula_subset.p'
    if subset_path.exists():
        if args.formula_subset:
            extract_config = json.loads((subset_path.parent / 'config.json').read_text())
            for field in ['pubchem_map', 'pubchem_map_bytes', 'pubchem_map_mtime_ns',
                          'model_sha256', 'data_audit_sha256', 'partition']:
                if extract_config[field] != config[field]:
                    raise ValueError(f'Subset provenance mismatch: {field}')
            (out / 'pubchem_map_sha256.txt').write_text(
                (subset_path.parent / 'pubchem_map_sha256.txt').read_text())
        with subset_path.open('rb') as f:
            cached_mapping = pickle.load(f)
        mapping = {formula: cached_mapping[formula] for formula in formulas}
        del cached_mapping
    else:
        print('Loading full PubChem map', map_path, flush=True)
        digest = hashlib.sha256()
        with map_path.open('rb') as f:
            while block := f.read(16 * 1024 * 1024):
                digest.update(block)
        (out / 'pubchem_map_sha256.txt').write_text(digest.hexdigest() + '\n')
        with map_path.open('rb') as f:
            full_mapping = pickle.load(f)
        mapping = {formula: sorted(full_mapping.get(formula, [])) for formula in formulas}
        del full_mapping
        gc.collect()
        tmp = subset_path.with_suffix('.tmp')
        with tmp.open('wb') as f:
            pickle.dump(mapping, f, protocol=4)
        tmp.replace(subset_path)
    if map_path.stat().st_size != map_stat.st_size or map_path.stat().st_mtime_ns != map_stat.st_mtime_ns:
        raise ValueError('PubChem map changed during extraction')
    print(json.dumps({'formulas': len(formulas), 'raw_candidates': sum(map(len, mapping.values())),
                      'missing_formulas': sum(not v for v in mapping.values())}), flush=True)
    if args.extract_only:
        inventory = {'formulas': len(formulas), 'raw_candidates': sum(map(len, mapping.values())),
                     'missing_formulas': sum(not v for v in mapping.values()),
                     'elapsed_seconds': time.time() - started,
                     'formula_counts': {f: len(v) for f, v in mapping.items()}}
        (out / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
        return
    pieces, audits = [], []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        for index, formula in enumerate(formulas, 1):
            result_path = out / 'formula_results' / f'{formula}.tsv'
            audit_path = out / 'formula_results' / f'{formula}.json'
            result_path.parent.mkdir(exist_ok=True)
            if result_path.exists() and audit_path.exists():
                pieces.append(pd.read_csv(result_path, sep='\t'))
                audits.append(json.loads(audit_path.read_text()))
                continue
            queries = frame.loc[frame.formula == formula]
            methods = sorted(queries.dataset_id.unique())
            entries = mapping[formula]
            keys, smiles = [], []
            predictions = {m: [] for m in methods}
            exclusions = {}
            for valid, graphs, excluded in candidate_batches(
                    pool, formula, entries, args.batch_size, args.workers):
                for reason, count in excluded.items():
                    exclusions[reason] = exclusions.get(reason, 0) + count
                if not valid:
                    continue
                smiles.extend(c[0] for c in valid)
                keys.extend(c[1] for c in valid)
                batch = predict_methods(model, graphs, methods)
                for m in methods:
                    predictions[m].extend(batch[m])
            pred_frame = pd.DataFrame({'smiles': smiles, 'inchikey2d': keys, **predictions})
            pred_path = out / 'candidate_predictions' / f'{formula}.tsv.gz'
            pred_path.parent.mkdir(exist_ok=True)
            pred_frame.to_csv(pred_path, sep='\t', index=False, compression='gzip')
            rows = []
            for _, query in queries.iterrows():
                rank = rank_query(keys, predictions[query.dataset_id], query.inchikey2d,
                                  query.rt, args.top_ks)
                rows.append({**query.to_dict(), **rank})
            result = pd.DataFrame(rows)
            tmp = result_path.with_suffix('.tmp')
            result.to_csv(tmp, sep='\t', index=False)
            tmp.replace(result_path)
            audit = {'formula': formula, 'raw_entries': len(entries), 'scored_entries': len(keys),
                     'unique_2d_keys': len(set(keys)), 'exclusions': exclusions,
                     'queries': len(result), 'covered_queries': int(result.truth_present.sum())}
            audit_path.write_text(json.dumps(audit, indent=2) + '\n')
            pieces.append(result)
            audits.append(audit)
            print(json.dumps({'completed_formulas': index, 'total_formulas': len(formulas),
                              **audit, 'elapsed_seconds': time.time() - started}), flush=True)
    ranked = pd.concat(pieces, ignore_index=True)
    ranked.to_csv(out / 'query_ranks.tsv', sep='\t', index=False)
    pd.DataFrame(audits).to_json(out / 'candidate_audit.json', orient='records', indent=2)
    summary = {'probe_only': bool(args.probe_formulas), 'partition': args.partition,
               'shard_index': args.shard_index, 'shard_count': args.shard_count,
               'elapsed_seconds': time.time() - started, 'n_formulas': len(formulas),
               'overall': retrieval_metrics(ranked, args.top_ks),
               'by_method': {m: retrieval_metrics(g, args.top_ks)
                             for m, g in ranked.groupby('dataset_id')},
               'raw_candidate_entries': sum(a['raw_entries'] for a in audits),
               'scored_candidate_entries': sum(a['scored_entries'] for a in audits),
               'pubchem_map_sha256': (out / 'pubchem_map_sha256.txt').read_text().strip()}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
