import pandas as pd
import pytest

from prepare_ms2_specialized import select_supported


def inputs():
    frame = pd.DataFrame({
        'id': ['cn', 'c18p', 'identity_conflict', 'hn', 'different_catalog'],
        'dataset_id': ['CN', 'C18P', 'HP', 'HN', 'CP'],
        'compound_library_id': ['CL1', 'CL1', 'CL2', 'CL3', 'CL9'],
        'inchikey2d': ['A'*14, 'A'*14, 'B'*14, 'D'*14, 'A'*14],
        'split_type': ['test', 'test', 'val', 'train', 'test'],
        'rt': [1., 2., 3., 4., 5.]})
    evidence = pd.DataFrame({
        'raw_source_spec': ['broad_CN_CL1', 'broad_HP_CL2', 'broad_HN_CL3', 'broad_CP_CL8'],
        'inchikey2d': ['A'*14, 'C'*14, 'D'*14, 'A'*14],
        'spec': ['group1', 'group2', 'group3', 'group4']})
    reference = pd.DataFrame({'inchikey2d': ['A'*14, 'B'*14, 'C'*14, 'D'*14],
                              'Fold_0': ['test', 'val', 'train', 'train']})
    return frame, evidence, reference


def test_support_requires_original_method_catalog_and_structure_and_ignores_rt():
    frame, evidence, reference = inputs()
    actual = select_supported(frame, evidence, reference)
    assert actual.id.tolist() == ['cn', 'hn']
    assert actual.split_type.tolist() == ['test', 'train']
    changed_rt = frame.assign(rt=frame.rt * 100, rt_pred=-999.)
    assert select_supported(changed_rt, evidence, reference).id.tolist() == ['cn', 'hn']


def test_ambiguous_contributors_and_changed_partitions_are_rejected():
    frame, evidence, reference = inputs()
    duplicate = pd.concat([evidence, evidence.iloc[[0]].assign(inchikey2d='B'*14)])
    with pytest.raises(ValueError, match='identifiers must be unique'):
        select_supported(frame, duplicate, reference)
    reference.loc[0, 'Fold_0'] = 'train'
    with pytest.raises(ValueError, match='changes an existing RT partition'):
        select_supported(frame, evidence, reference)
