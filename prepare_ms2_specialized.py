"""Prepare a new training snapshot with MS2-supported RT observations only."""
import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

from specialized import (METHODS, FEATURE_NAMES, PARTITIONS, sha256, assert_disjoint,
                         fit_rt_scalers, add_scaled_rt, validate_prepared)
from train_specialized import load_prepared


def select_supported(frame, contributors, latest_reference):
    """Require the original method/catalog record and its 2D identity, never RT error."""
    if contributors.raw_source_spec.duplicated().any():
        raise ValueError('MS2 raw contributor identifiers must be unique')
    if latest_reference.inchikey2d.duplicated().any():
        raise ValueError('MS2 reference must have one partition per 2D key')
    mapping = latest_reference.set_index('inchikey2d').Fold_0
    expected = frame.inchikey2d.map(mapping)
    known = expected.notna()
    if not (frame.loc[known, 'split_type'] == expected[known]).all():
        raise ValueError('MS2 reference changes an existing RT partition')
    raw_ids = 'broad_' + frame.dataset_id + '_' + frame.compound_library_id
    evidence = contributors.set_index('raw_source_spec')
    keep = frame.inchikey2d == raw_ids.map(evidence.inchikey2d)
    selected = frame.loc[keep].copy()
    selected['ms2_raw_source_spec'] = raw_ids.loc[keep]
    selected['ms2_group_spec'] = raw_ids.loc[keep].map(evidence.spec)
    if not selected.inchikey2d.isin(mapping.index).all():
        raise ValueError('Retained MS2 evidence lacks split provenance')
    assert_disjoint(selected)
    return selected


def prepare(source_dir, reference_dir, selection_summary, output_dir):
    source, reference, output = map(Path, [source_dir, reference_dir, output_dir])
    if output.exists():
        raise ValueError('Output exists; choose a new directory to preserve previous snapshots')
    frame, old_scalers, old_audit = load_prepared(source)
    summary = json.loads(Path(selection_summary).read_text())
    export = json.loads((reference / 'export.json').read_text())
    for name, checksum in export['export_sha256'].items():
        if sha256(reference / name) != checksum:
            raise ValueError(f'MS2 reference checksum mismatch: {name}')
    if export['source_sha256'] != summary['reference_source']['source_sha256']:
        raise ValueError('Reference differs from the evaluated cleaned cohort')
    contributors = pd.read_csv(reference / 'broad_contributors.tsv', sep='\t')
    latest_reference = pd.read_csv(reference / 'reference_keys.tsv', sep='\t')
    selected = select_supported(frame, contributors, latest_reference)
    expected_ids = set(pd.read_csv(Path(selection_summary).parent / 'filtered_rt/all.tsv',
                                  sep='\t', usecols=['id']).id)
    if set(selected.id) != expected_ids or selected.id.duplicated().any():
        raise ValueError('Selected observations differ from the evaluated cleaned cohort')
    for split in PARTITIONS:
        count = summary['cohort_counts'][split]['clean_record_2d']
        actual = selected.loc[selected.split_type == split]
        if len(actual) != count['rows'] or actual.inchikey2d.nunique() != count['keys']:
            raise ValueError(f'{split}: cleaned cohort counts differ')
    # Fit new bounds on retained training rows, never on the old or held-out cohort.
    scalers = fit_rt_scalers(selected)
    selected = add_scaled_rt(selected, scalers)
    validate_prepared(selected, scalers)
    preserved = [c for c in frame if c != 'rt_normalized']
    pd.testing.assert_frame_equal(selected[preserved], frame.loc[selected.index, preserved], check_exact=True)
    output.mkdir(parents=True)
    artifacts = []
    for name, group in [('all', selected)] + [(s, selected.loc[selected.split_type == s]) for s in PARTITIONS]:
        path = output / f'{name}.tsv'
        group.to_csv(path, sep='\t', index=False)
        artifacts.append(path)
    manifest = selected[['inchikey2d', 'split_type', 'assignment_source']].drop_duplicates().sort_values('inchikey2d')
    manifest.to_csv(output / 'key_assignments.tsv', sep='\t', index=False)
    (output / 'rt_scalers.json').write_text(json.dumps(scalers, indent=2) + '\n')
    for name in ['reference_keys.tsv', 'reference_source.json']:
        shutil.copy2(source / name, output / name)
    for name in ['broad_contributors.tsv', 'reference_keys.tsv', 'export.json']:
        target = output / ('ms2_' + name)
        shutil.copy2(reference / name, target)
        artifacts.append(target)
    excluded = frame.loc[~frame.id.isin(selected.id)].copy()
    support_audit = pd.read_csv(Path(selection_summary).parent / 'rt_support_audit.tsv', sep='\t', usecols=['id', 'ms2_status'])
    excluded = excluded.merge(support_audit, on='id', validate='one_to_one')
    excluded.to_csv(output / 'ms2_excluded_rows.tsv', sep='\t', index=False)
    counts = {s: {'rows': int((selected.split_type == s).sum()),
                  'keys': int(selected.loc[selected.split_type == s].inchikey2d.nunique())} for s in PARTITIONS}
    per_method = {m: {s: {'rows': len(g := selected.loc[(selected.dataset_id == m) & (selected.split_type == s)]),
                          'keys': int(g.inchikey2d.nunique())} for s in PARTITIONS} for m in METHODS}
    keys = {s: set(selected.loc[selected.split_type == s].inchikey2d) for s in PARTITIONS}
    overlap = {f'{a}/{b}': len(keys[a] & keys[b]) for a, b in [('train','test'),('train','val'),('test','val')]}
    additions = selected.assignment_source == 'rt_only_training'
    audit = {'reference': old_audit['reference'], 'inputs': old_audit['inputs'],
             'feature_names': FEATURE_NAMES, 'methods': METHODS,
             'unmatched_policy': 'error', 'rt_only_training_keys': int(selected.loc[additions].inchikey2d.nunique()),
             'rt_only_training_rows': int(additions.sum()), 'partitions': counts,
             'per_method': per_method, 'overlap': overlap,
             'inchi_disagreement_rows': int(selected.inchi_disagrees.sum()),
             'selection': {'policy': summary['primary_policy'],
                'evidence_limitation': summary['evidence_limitation'],
                'ms2_reference': export, 'filter_independent_of_rt_error': True,
                'parent_rt_audit_sha256': sha256(source / 'audit.json'),
                'parent_rt_all_sha256': sha256(source / 'all.tsv'),
                'selection_summary_sha256': sha256(selection_summary),
                'excluded_rows': len(excluded),
                'excluded_by_status': excluded.ms2_status.value_counts().to_dict(),
                'rt_only_additions': summary['rt_only_training_additions'],
                'split_assignments_preserved': True,
                'scaler_fit_partition': 'retained_train_only'}}
    for name in ['key_assignments.tsv', 'rt_scalers.json', 'reference_keys.tsv', 'reference_source.json', 'ms2_excluded_rows.tsv']:
        artifacts.append(output / name)
    audit['artifact_checksums'] = {p.name: sha256(p) for p in artifacts}
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    load_prepared(output)
    validation = {'partitions': counts, 'overlap': overlap, 'all_existing_assignments_preserved': True,
                  'only_retained_training_rows_fit_scalers': True,
                  'scalers': scalers, 'parent_scalers': old_scalers,
                  'original_rt_identity_and_features_unchanged': True,
                  'same_evaluation_cohort_as_previous_filtered_report': True,
                  'prepared_audit_sha256': sha256(output / 'audit.json')}
    (output / 'preparation_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(validation, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_dir', default='data/specialized_c18_corrected_20261016')
    parser.add_argument('--reference_dir', default='reports/ms2_cleaned_rt_20261023/reference')
    parser.add_argument('--selection_summary', default='reports/ms2_cleaned_rt_20261023/summary.json')
    parser.add_argument('--output_dir', default='data/specialized_ms2_clean_20261023')
    args = parser.parse_args()
    prepare(args.source_dir, args.reference_dir, args.selection_summary, args.output_dir)


if __name__ == '__main__':
    main()
