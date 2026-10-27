"""Prepare workbook RTs using the immutable molgpu MS/MS partition assignments."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shlex
import subprocess

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from specialized import (METHODS, FEATURE_NAMES, PARTITIONS, sha256, structure_identity,
                         encode_methods, assign_partitions, fit_rt_scalers, add_scaled_rt,
                         validate_prepared)

REFERENCE_EXPORT = r'''
import hashlib, json, sys
from pathlib import Path
import pandas as pd
p = Path(sys.argv[1])
def digest(f):
    h = hashlib.sha256()
    with open(f, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
paths = [p / 'labels.tsv', p / 'splits/split_1.tsv']
before = {str(f): digest(f) for f in paths}
labels = pd.read_csv(paths[0], sep='\t', usecols=['spec', 'inchikey'], dtype=str)
splits = pd.read_csv(paths[1], sep='\t', dtype=str)
joined = labels.merge(splits, on='spec', how='left', validate='one_to_one')
if joined.Fold_0.isna().any() or len(splits) != len(labels):
    raise ValueError('Reference split does not cover the labels exactly')
joined['inchikey2d'] = joined.inchikey.str.split('-').str[0]
missing = int(joined.inchikey2d.isna().sum())
joined = joined.dropna(subset=['inchikey2d'])
if not joined.inchikey2d.str.fullmatch('[A-Z]{14}').all():
    raise ValueError('Malformed reference keys')
if (joined.groupby('inchikey2d').Fold_0.nunique() > 1).any():
    raise ValueError('Reference has overlapping 2D InChIKeys')
if not set(joined.Fold_0).issubset({'train', 'val', 'test'}):
    raise ValueError('Unknown partition in Fold_0')
mapping = joined.drop_duplicates('inchikey2d').sort_values('inchikey2d')
after = {str(f): digest(f) for f in paths}
if before != after:
    raise ValueError('Reference files changed during export')
print(json.dumps({'source_checksums': before, 'fold': 'Fold_0',
    'missing_key_labels': missing, 'n_labels': len(labels),
    'mapping': mapping[['inchikey2d', 'Fold_0']].values.tolist()}))
'''


def load_reference(output_dir, host, root, python, refresh=False):
    mapping_path = output_dir / 'reference_keys.tsv'
    meta_path = output_dir / 'reference_source.json'
    if mapping_path.exists() and meta_path.exists() and not refresh:
        metadata = json.loads(meta_path.read_text())
        if metadata['mapping_sha256'] != sha256(mapping_path):
            raise ValueError('Local reference snapshot checksum mismatch')
        if metadata['host'] != host or metadata['root'] != root:
            raise ValueError('Snapshot source differs; use --refresh_reference explicitly')
        return pd.read_csv(mapping_path, sep='\t', dtype=str), metadata
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
               python, '-c', shlex.quote(REFERENCE_EXPORT), shlex.quote(root)]
    print(f'Snapshotting Fold_0 from {host}:{root}', flush=True)
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=180)
    payload = json.loads(result.stdout)
    reference = pd.DataFrame(payload.pop('mapping'), columns=['inchikey2d', 'split_type'])
    reference.to_csv(mapping_path, sep='\t', index=False)
    metadata = payload | {'host': host, 'root': root, 'mapping_sha256': sha256(mapping_path)}
    meta_path.write_text(json.dumps(metadata, indent=2) + '\n')
    return reference, metadata


def read_workbooks(input_dir):
    rows, excluded, inputs, cache, seen = [], [], [], {}, set()
    for path in sorted(input_dir.glob('*.xlsx')):
        match = re.search(r'_(CN|HP|CP|HN|C18P)$', path.stem)
        if not match:
            continue
        method = match.group(1)
        if method in seen:
            raise ValueError(f'Multiple workbooks for {method}')
        seen.add(method)
        book = pd.ExcelFile(path, engine='openpyxl')
        candidates = []
        for sheet in book.sheet_names:
            preview = pd.read_excel(book, sheet_name=sheet, header=None, nrows=10)
            for index, values in preview.iterrows():
                if {'Broad Smiles', 'RT', 'compound_library_id'}.issubset(set(values.dropna())):
                    candidates.append((sheet, int(index)))
        if len(candidates) != 1:
            raise ValueError(f'{path}: expected one recognizable data sheet, found {candidates}')
        sheet, header = candidates[0]
        frame = pd.read_excel(book, sheet_name=sheet, header=header)
        inputs.append({'file': str(path), 'sha256': sha256(path), 'sheet': sheet,
                       'header_row': header + 1, 'rows': len(frame)})
        for index, row in frame.iterrows():
            source = {'dataset_id': method, 'source_file': path.name,
                      'source_sheet': sheet, 'source_row': int(index) + header + 2,
                      'compound_library_id': row.get('compound_library_id'),
                      'name': row.get('Catalog Name'), 'source_smiles': row.get('Broad Smiles'),
                      'source_inchi': row.get('Broad InChI'),
                      'source_scaled_rt': row.get('scaled_RT'), 'source_rt': row.get('RT')}
            rt = pd.to_numeric(row.get('RT'), errors='coerce')
            if pd.isna(rt) or not np.isfinite(rt) or rt <= 0:
                excluded.append(source | {'reason': 'missing_or_invalid_rt'})
                continue
            smiles = row.get('Broad Smiles')
            cache_key = smiles if isinstance(smiles, str) else None
            if cache_key not in cache:
                try:
                    cache[cache_key] = structure_identity(smiles)
                except ValueError as error:
                    cache[cache_key] = str(error)
            identity = cache[cache_key]
            if isinstance(identity, str):
                excluded.append(source | {'reason': identity})
                continue
            canonical, key = identity
            inchi = row.get('Broad InChI')
            inchi_key = Chem.InchiToInchiKey(inchi) if isinstance(inchi, str) and inchi else ''
            rows.append(source | {'id': f'{method}_row{source["source_row"]}',
                'smiles': canonical, 'inchikey': key, 'inchikey2d': key[:14], 'rt': float(rt),
                'source_inchi_key': inchi_key,
                'inchi_disagrees': bool(inchi_key and inchi_key[:14] != key[:14])})
        book.close()
        print(f'Read {path.name}: {len(frame)} rows', flush=True)
    if seen != set(METHODS):
        raise ValueError(f'Missing workbooks: {sorted(set(METHODS) - seen)}')
    accepted = pd.DataFrame(rows)
    duplicates = accepted.duplicated(['dataset_id', 'smiles', 'rt'])
    excluded.extend(accepted.loc[duplicates].assign(reason='exact_duplicate').to_dict('records'))
    return accepted.loc[~duplicates].reset_index(drop=True), pd.DataFrame(excluded), inputs


def prepare(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference, provenance = load_reference(output, args.reference_host, args.reference_root,
                                           args.reference_python, args.refresh_reference)
    frame, excluded, inputs = read_workbooks(Path(args.input_dir))
    missing = frame.loc[~frame.inchikey2d.isin(reference.inchikey2d)].copy()
    missing.to_csv(output / 'unmatched_rows.tsv', sep='\t', index=False)
    excluded.to_csv(output / 'excluded_rows.tsv', sep='\t', index=False)
    if args.expected_unmatched_keys is not None and missing.inchikey2d.nunique() != args.expected_unmatched_keys:
        raise ValueError(f'Expected {args.expected_unmatched_keys} unmatched keys; found '
                         f'{missing.inchikey2d.nunique()}. Inspect unmatched_rows.tsv before proceeding.')
    frame = assign_partitions(frame, reference, args.unmatched_policy)
    encoded = encode_methods(frame.dataset_id)
    for index, name in enumerate(FEATURE_NAMES):
        frame[name] = encoded[:, index].astype(int)
    scalers = fit_rt_scalers(frame)
    frame = add_scaled_rt(frame, scalers)
    validate_prepared(frame, scalers)
    artifacts = []
    for name, subset in [('all', frame)] + [(s, frame.loc[frame.split_type == s]) for s in PARTITIONS]:
        path = output / f'{name}.tsv'
        subset.to_csv(path, sep='\t', index=False)
        artifacts.append(path)
    manifest = frame[['inchikey2d', 'split_type', 'assignment_source']].drop_duplicates().sort_values('inchikey2d')
    manifest.to_csv(output / 'key_assignments.tsv', sep='\t', index=False)
    (output / 'rt_scalers.json').write_text(json.dumps(scalers, indent=2) + '\n')
    counts = {split: {'rows': int((frame.split_type == split).sum()),
                      'keys': int(frame.loc[frame.split_type == split, 'inchikey2d'].nunique())}
              for split in PARTITIONS}
    per_method = {method: {split: {'rows': len(g := frame.loc[(frame.dataset_id == method) & (frame.split_type == split)]),
                                  'keys': int(g.inchikey2d.nunique())} for split in PARTITIONS}
                  for method in METHODS}
    keys = {split: set(frame.loc[frame.split_type == split, 'inchikey2d']) for split in PARTITIONS}
    overlap = {f'{a}/{b}': len(keys[a] & keys[b]) for a, b in [('train', 'test'), ('train', 'val'), ('test', 'val')]}
    audit = {'reference': provenance, 'inputs': inputs, 'feature_names': FEATURE_NAMES,
             'methods': METHODS, 'unmatched_policy': args.unmatched_policy,
             'rt_only_training_keys': int(missing.inchikey2d.nunique()),
             'rt_only_training_rows': len(missing), 'partitions': counts,
             'per_method': per_method, 'overlap': overlap,
             'excluded_reasons': dict(Counter(excluded.reason)),
             'inchi_disagreement_rows': int(frame.inchi_disagrees.sum())}
    for name in ['key_assignments.tsv', 'reference_keys.tsv', 'reference_source.json',
                 'rt_scalers.json', 'excluded_rows.tsv', 'unmatched_rows.tsv']:
        artifacts.append(output / name)
    audit['artifact_checksums'] = {path.name: sha256(path) for path in artifacts}
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'partitions': counts, 'overlap': overlap,
                      'rt_only_training_keys': audit['rt_only_training_keys'],
                      'excluded_reasons': audit['excluded_reasons']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', default='data')
    parser.add_argument('--output_dir', default='data/specialized')
    parser.add_argument('--reference_host', default='molgpu03.mit.edu')
    parser.add_argument('--reference_root', default='/home/runzhong/ms-pred/data/spec_datasets/all_msms_20260826')
    parser.add_argument('--reference_python', default='/home/runzhong/mambaforge/envs/ms-gen-torch24/bin/python')
    parser.add_argument('--refresh_reference', action='store_true')
    parser.add_argument('--unmatched_policy', choices=['error', 'train'], default='error')
    parser.add_argument('--expected_unmatched_keys', type=int)
    args = parser.parse_args()
    rdBase.DisableLog('rdApp.warning')
    prepare(args)


if __name__ == '__main__':
    main()
