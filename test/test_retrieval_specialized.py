import numpy as np
import pandas as pd
import pytest
import torch
from rdkit import Chem

from dmpnn_graph import dmpnn_graph
from mpnranker2 import MPNranker
from retrieve_specialized import (molecular_formula, rank_query, retrieval_metrics,
                                  predict_methods, build_candidate_batch)
from specialized import FEATURE_NAMES, METHODS, predict_frame, set_model_device


def test_retrieval_collapses_connectivity_without_injecting_truth():
    keys = ['AAAAAAAAAAAAAA', 'BBBBBBBBBBBBBB', 'BBBBBBBBBBBBBB', 'CCCCCCCCCCCCCC']
    preds = [3., 1., 2., 4.]
    result = rank_query(keys, preds, keys[1], 2.1, [1, 5])
    assert result['n_candidates'] == 3 and result['rank'] == 1
    assert result['top_1'] and result['top_5']
    missing = rank_query(keys, preds, 'DDDDDDDDDDDDDD', 2.1, [1, 5])
    assert not missing['truth_present'] and missing['rank'] is None
    assert not missing['top_5'] and missing['random_top_5'] == 0
    empty = rank_query([], [], keys[0], 2.1, [1])
    assert empty['n_candidates'] == 0 and not empty['top_1']


def test_tie_bounds_and_missing_truth_denominators():
    keys = ['AAAAAAAAAAAAAA', 'BBBBBBBBBBBBBB']
    hit = rank_query(keys, [1., 3.], keys[1], 2., [1, 5])
    assert hit['rank'] == 2 and hit['rank_optimistic'] == 1 and hit['rank_pessimistic'] == 2
    missing = rank_query(keys, [1., 3.], 'CCCCCCCCCCCCCC', 2., [1, 5])
    frame = pd.DataFrame([{'inchikey2d': keys[1], **hit},
                          {'inchikey2d': 'CCCCCCCCCCCCCC', **missing}])
    metrics = retrieval_metrics(frame, [1, 5])
    assert metrics['coverage'] == .5
    assert metrics['all_queries']['top_5'] == .5
    assert metrics['covered_queries']['top_5'] == 1.
    assert metrics['optimistic_tie_bounds']['top_1'] == .5
    assert metrics['pessimistic_tie_bounds']['top_1'] == 0.


def test_formula_matches_mapping_charge_convention():
    assert molecular_formula('C[NH3+]') == 'CH6N'
    assert molecular_formula('CN') == 'CH5N'


@pytest.mark.parametrize('cn_family', ['C8', 'C18'])
def test_shared_encoder_matches_standard_inference_for_every_method(cn_family):
    torch.set_num_threads(1)
    torch.manual_seed(13)
    model = MPNranker(encoder_size=32, sys_features_dim=5,
                      hidden_units=[16, 8], hidden_units_pv=[16, 16],
                      res_conn_enc=False, no_sigmoid_roi=True)
    set_model_device(model, 'cpu')
    model.extra_storage = {'specialized': {
        'feature_names': FEATURE_NAMES,
        'methods': {**METHODS, 'CN': ('negative', cn_family)},
        'mappings': {m: {'coefficients': [.1, .5, .01]} for m in METHODS},
        'rt_scalers': {m: {'min': 1., 'range': 10.} for m in METHODS}}}
    smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
    fast = predict_methods(model, [dmpnn_graph(s) for s in smiles], list(METHODS))
    for m in METHODS:
        frame = pd.DataFrame({'smiles': smiles, 'dataset_id': m})
        standard = predict_frame(model, frame, batch_size=3)
        np.testing.assert_allclose(fast[m], standard.rt_pred, rtol=1e-6, atol=1e-6)


def test_worker_tensor_batch_matches_original_graph_batch():
    model = MPNranker(encoder_size=32, sys_features_dim=5,
                      hidden_units=[16, 8], hidden_units_pv=[16, 16],
                      res_conn_enc=False, no_sigmoid_roi=True)
    set_model_device(model, 'cpu')
    model.extra_storage = {'specialized': {
        'methods': METHODS,
        'mappings': {m: {'coefficients': [.1, .5, .01]} for m in METHODS},
        'rt_scalers': {m: {'min': 1., 'range': 10.} for m in METHODS}}}
    smiles = ['CCO', 'CC(=O)O', 'O', 'CCO']
    tasks = [(molecular_formula(s), s, Chem.MolToInchiKey(Chem.MolFromSmiles(s))) for s in smiles]
    identities, batch, errors = build_candidate_batch(tasks)
    assert not errors and len(identities) == len(smiles)
    expected = predict_methods(model, [dmpnn_graph(s) for s in smiles], list(METHODS))
    actual = predict_methods(model, batch, list(METHODS))
    for method in METHODS:
        np.testing.assert_allclose(actual[method], expected[method], rtol=1e-7, atol=1e-7)
