"""Tests of compound isolation, training-only fitting, and model integration."""
from itertools import combinations
import random

import numpy as np
import pandas as pd
import pytest
import torch

from specialized import (METHODS, FEATURE_NAMES, encode_methods, structure_identity,
                         assign_partitions, assert_disjoint, fit_rt_scalers, scale_rt,
                         inverse_rt, add_scaled_rt, predict_frame)
from train_specialized import calibrate, validation_loss, evaluation_metrics, make_rank_dataset
from utils_newbg import RankDataset
from utils import Data


def test_method_encoding_and_unknown_rejection():
    np.testing.assert_array_equal(encode_methods(['CN', 'HP', 'CP', 'HN', 'C18P']),
        [[0, 1, 0, 1, 0], [1, 0, 0, 0, 1], [1, 0, 1, 0, 0],
         [0, 1, 0, 0, 1], [1, 0, 0, 1, 0]])
    with pytest.raises(ValueError, match='Unknown method'):
        encode_methods(['C18N'])


def test_stereoisomers_and_cross_method_measurements_follow_reference():
    identities = [structure_identity(s) for s in ['C[C@H](O)C(=O)O', 'C[C@@H](O)C(=O)O']]
    assert identities[0][1] != identities[1][1]
    key = identities[0][1][:14]
    assert key == identities[1][1][:14]
    frame = pd.DataFrame({'inchikey2d': [key, key, key, 'AAAAAAAAAAAAAA'],
                          'dataset_id': ['CN', 'HP', 'HN', 'CP']})
    reference = pd.DataFrame({'inchikey2d': [key], 'split_type': ['test']})
    a = assign_partitions(frame, reference, 'train')
    b = assign_partitions(frame.sample(frac=1, random_state=3), reference, 'train').sort_index()
    pd.testing.assert_frame_equal(a, b)
    assert a.split_type.tolist() == ['test', 'test', 'test', 'train']
    assert a.assignment_source.tolist() == ['molgpu_split_1'] * 3 + ['rt_only_training']
    with pytest.raises(ValueError, match='lack reference'):
        assign_partitions(frame, reference)
    overlapping = a.copy()
    overlapping.loc[0, 'split_type'] = 'val'
    with pytest.raises(ValueError, match='overlap'):
        assert_disjoint(overlapping)


def test_conflicting_reference_is_rejected():
    reference = pd.DataFrame({'inchikey2d': ['AAAAAAAAAAAAAA'] * 2,
                              'split_type': ['train', 'test']})
    with pytest.raises(ValueError, match='exactly one row'):
        assign_partitions(pd.DataFrame({'inchikey2d': ['AAAAAAAAAAAAAA']}), reference)


def make_scaling_frame():
    return pd.DataFrame([{'dataset_id': method, 'rt': rt, 'split_type': split}
        for method in METHODS for rt, split in [(2., 'train'), (4., 'train'), (6., 'train'),
                                                (1., 'val'), (10., 'test')]])


def test_scaling_fits_training_only_and_never_clips():
    frame = make_scaling_frame()
    scalers = fit_rt_scalers(frame)
    assert all(s['min'] == 2 and s['max'] == 6 for s in scalers.values())
    scaled = add_scaled_rt(frame, scalers)
    assert set(scaled.loc[scaled.split_type == 'val', 'rt_normalized']) == {-0.25}
    assert set(scaled.loc[scaled.split_type == 'test', 'rt_normalized']) == {2.0}
    np.testing.assert_allclose(inverse_rt(scale_rt([1, 3, 10], scalers['CN']), scalers['CN']), [1, 3, 10])
    frame.loc[frame.split_type != 'train', 'rt'] *= 100
    assert fit_rt_scalers(frame) == scalers
    frame.loc[(frame.dataset_id == 'CN') & (frame.split_type == 'train'), 'rt'] = 2
    with pytest.raises(ValueError, match='range is zero'):
        fit_rt_scalers(frame)


def test_uniform_pairs_are_bounded_unique_and_cover_triangle():
    expected = set(combinations(range(13), 2))
    random.seed(42)
    pairs = list(RankDataset.dataset_pair_it(list(range(13)), max_num_pairs=1000, uniform=True))
    assert {(a, b) for a, b, _ in pairs} == expected
    assert len(pairs) == len(expected)
    random.seed(42)
    small = list(RankDataset.dataset_pair_it(list(range(13)), max_num_pairs=17, uniform=True))
    assert len(small) == len({(a, b) for a, b, _ in small}) == 17
    random.seed(42)
    assert small == list(RankDataset.dataset_pair_it(list(range(13)), max_num_pairs=17, uniform=True))
    large = list(RankDataset.dataset_pair_it(list(range(20_000)), max_num_pairs=5000, uniform=True))
    assert len({(a, b) for a, b, _ in large}) == 5000
    assert all(0 <= a < b < 20_000 for a, b, _ in large)


def test_range_scaled_pair_weights_keep_raw_epsilon():
    # Identical relative differences on different minute ranges get identical weights.
    def dataset(span, discard=False):
        return RankDataset(x_mols=['a', 'b'], x_extra=np.zeros((2, 0)),
            x_sys=encode_methods(['CN', 'CN']), x_ids=['a', 'b'], y=np.array([1., 1. + span / 10]),
            dataset_info=['CN', 'CN'], dynamic_weights=False, use_group_weights=False,
            y_neg=True, weight_mid=0.05, rt_ranges={'CN': span},
            epsilon=10 / 60, discard_smaller_than_epsilon=discard)
    np.testing.assert_allclose(dataset(1).weights, dataset(10).weights)
    assert len(dataset(1, True)) == 0  # 0.1 minute is below 10 seconds.
    assert len(dataset(10, True)) == 1 # 1 minute is above 10 seconds.


@pytest.mark.parametrize('difference', [10 / 60, 0.5, 1.0])
def test_paper_pair_weights_use_minutes_and_original_cutoff(monkeypatch, difference):
    monkeypatch.setattr('train_specialized.graph_inputs', lambda frame, cache: (
        frame.smiles.tolist(), np.zeros((len(frame), 0)), encode_methods(frame.dataset_id)))
    frame = pd.DataFrame({'smiles': ['CC', 'CCC'], 'rt': [1., 1. + difference],
                          'dataset_id': ['CN', 'CN']})
    expected = 1 / (1 + np.exp(-20 * (difference - 0.75)))
    for span in [14., 23.]:
        scalers = {'CN': {'range': span}}
        paper = make_rank_dataset(frame, scalers, {}, 10, 'paper')
        if expected < 1e-4:
            assert len(paper) == 0
        else:
            assert len(paper) == 1
            np.testing.assert_allclose(paper.weights, [expected])
        baseline = make_rank_dataset(frame, scalers, {}, 10)
        np.testing.assert_allclose(baseline.weights,
            [1 / (1 + np.exp(-20 * (difference / span - 0.05)))])


def test_external_loader_keeps_method_measurements_and_voids(tmp_path):
    path = tmp_path / 'input.tsv'
    pd.DataFrame({'smiles': ['CC', 'CC', 'CCC', 'CCCC'], 'rt': [1., 1., 2., 3.],
                  'dataset_id': ['CN', 'HP', 'CN', 'HP'], 'column.t0': [0.1, 0.2, 0.1, 0.2]}).to_csv(path, sep='\t', index=False)
    data = Data(use_system_information=False, metadata_void_rt=True)
    data.add_external_data(str(path))
    assert len(data.df) == 4
    assert data.void_info['CN'] == 0.2
    assert data.void_info['HP'] == 0.4


@pytest.mark.parametrize('cn_family', ['C8', 'C18'])
def test_packaged_prediction_uses_saved_mapping_not_query_rts(cn_family):
    class Model:
        extra_storage = {'specialized': {'feature_names': FEATURE_NAMES,
            'methods': {**METHODS, 'CN': ('negative', cn_family)},
            'mappings': {'CN': {'coefficients': [0., 1., 0.]}},
            'rt_scalers': {'CN': {'min': 2., 'range': 4.}}}}

        def predict(self, *args, **kwargs):
            np.testing.assert_array_equal(args[2], encode_methods(['CN'] * 2,
                self.extra_storage['specialized']['methods']))
            return np.array([-0.25, 2.])

    frame = pd.DataFrame({'smiles': ['CC', 'CCC'], 'dataset_id': ['CN'] * 2, 'rt': [100., 200.]})
    result = predict_frame(Model(), frame)
    np.testing.assert_allclose(result.rt_pred, [1., 10.])
    assert result.index.equals(frame.index)


def test_epoch_validation_loss_is_sample_weighted_and_independent():
    class Model:
        def eval(self):
            pass

        def loss_step(self, x, y, weights, loss_function):
            losses = torch.tensor(x, dtype=torch.float32)
            return losses.mean(), losses

    batches = [([1., 1., 1.], None, None, None), ([5.], None, None, None)]
    assert validation_loss(Model(), batches, None) == 2.
    assert validation_loss(Model(), [([0.5], None, None, None)], None) == 0.5


def test_calibration_uses_normalized_training_anchors_and_rejects_heldout():
    class Model:
        def predict(self, *args, **kwargs):
            return np.tile([0., 0.5, 1.], 5)

    frame = make_scaling_frame()
    scalers = fit_rt_scalers(frame)
    train = add_scaled_rt(frame.loc[frame.split_type == 'train'].copy(), scalers)
    train['smiles'] = ['CC', 'CCC', 'CCCC'] * 5
    mappings = calibrate(Model(), train, scalers, {}, 16)
    for method in METHODS:
        np.testing.assert_allclose(mappings[method]['coefficients'], [0., 1., 0.], atol=1e-10)
        assert mappings[method]['n_anchors'] == 3
        assert mappings[method]['fit_partition'] == 'train'
    train.loc[train.index[0], 'split_type'] = 'test'
    with pytest.raises(ValueError, match='exclusively from training'):
        calibrate(Model(), train, scalers, {}, 16)


def test_order_metrics_exclude_near_ties_and_do_not_compare_methods():
    frame = pd.DataFrame([{'dataset_id': method, 'inchikey2d': str(i), 'rt': rt,
                           'rt_pred': rt + 1, 'roi': roi}
        for method in METHODS for i, rt, roi in [(0, 1., 0.), (1, 1.1, 5.), (2, 2., 1.)]])
    metrics = evaluation_metrics(frame)
    assert metrics['aggregate']['mae_minutes'] == 1.
    assert metrics['aggregate']['comparable_pairs'] == 10
    assert metrics['aggregate']['retention_order_accuracy'] == 0.5
