"""Train the existing two-step architecture on prepared five-method data."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dmpnn_graph import dmpnn_batch
from mapping import LADModel
from mpnranker2 import MPNranker, custom_collate
from sampling import CustomWeightedRandomSampler, calc_sampling_weights
from specialized import (METHODS, FEATURE_NAMES, PARTITIONS, sha256, fit_rt_scalers,
                         add_scaled_rt, validate_prepared, graph_inputs, predict_frame,
                         set_model_device, scale_rt)
from utils_newbg import RankDataset


def load_prepared(directory, method_mapping=None):
    directory = Path(directory)
    audit = json.loads((directory / 'audit.json').read_text())
    for name, checksum in audit['artifact_checksums'].items():
        if sha256(directory / name) != checksum:
            raise ValueError(f'Prepared artifact checksum mismatch: {name}')
    frame = pd.read_csv(directory / 'all.tsv', sep='\t')
    scalers = json.loads((directory / 'rt_scalers.json').read_text())
    expected_methods = METHODS if method_mapping is None else method_mapping
    if {k: list(v) for k, v in audit['methods'].items()} != {k: list(v) for k, v in expected_methods.items()}:
        raise ValueError('Prepared method metadata disagrees with requested encoding; reprepare the data')
    validate_prepared(frame, scalers, expected_methods)
    reference = pd.read_csv(directory / 'reference_keys.tsv', sep='\t', dtype=str)
    mapping = reference.set_index('inchikey2d').split_type
    expected = frame.inchikey2d.map(mapping)
    known = expected.notna()
    if not (frame.loc[known, 'split_type'] == expected[known]).all():
        raise ValueError('Prepared partitions disagree with molgpu assignments')
    if not (frame.loc[known, 'assignment_source'] == 'molgpu_split_1').all():
        raise ValueError('Incorrect reference provenance')
    if not (frame.loc[~known, 'assignment_source'] == 'rt_only_training').all():
        raise ValueError('Missing-key additions lack provenance')
    for split in PARTITIONS:
        actual = frame.loc[frame.split_type == split]
        if len(actual) != audit['partitions'][split]['rows'] or actual.inchikey2d.nunique() != audit['partitions'][split]['keys']:
            raise ValueError(f'{split}: audit counts disagree')
    return frame, scalers, audit


def make_rank_dataset(frame, scalers, cache, max_pairs, pair_weight_mode='normalized'):
    if pair_weight_mode == 'paper':
        weight_mid, rt_ranges = 0.75, None
    elif pair_weight_mode == 'normalized':
        weight_mid = 0.05
        rt_ranges = {method: scaler['range'] for method, scaler in scalers.items()}
    else:
        raise ValueError(f'Unknown pair weight mode: {pair_weight_mode}')
    return RankDataset(*graph_inputs(frame, cache), x_ids=frame.smiles.tolist(),
                       y=frame.rt.to_numpy(), dataset_info=frame.dataset_id.tolist(),
                       x_sys_global_num=len(FEATURE_NAMES),
                       void_info={method: 0.0 for method in METHODS},
                       no_inter_pairs=True, use_group_weights=False,
                       dynamic_weights=False, y_neg=True, epsilon=10 / 60,
                       weight_mid=weight_mid, weight_steepness=20,
                       max_num_pairs=max_pairs, uniform_pair_sampling=True,
                       rt_ranges=rt_ranges)


def validation_loss(model, loader, loss_function):
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for x, y, weights, _ in loader:
            losses = model.loss_step(x, y, weights, loss_function)[1]
            total += losses.sum().item()
            count += len(losses)
    if not count:
        raise ValueError('Validation has no usable pairs')
    return total / count


def calibrate(model, train, scalers, cache, batch_size):
    # Only this explicit training frame can supply anchor RTs.
    if train.empty or not (train.split_type == 'train').all():
        raise ValueError('Calibration anchors must come exclusively from training')
    anchors = train.copy()
    anchors['roi'] = model.predict(*graph_inputs(train, cache), batch_size=batch_size)
    mappings = {}
    for method in METHODS:
        subset = anchors.loc[anchors.dataset_id == method].copy()
        if len(subset) < 3 or subset.roi.nunique() < 3:
            raise ValueError(f'{method}: insufficient distinct training anchors/ROIs')
        subset['rt'] = scale_rt(subset.rt, scalers[method])
        mapping = LADModel(subset, ols_after=True, ols_discard_if_negative=True,
                           ols_drop_mode='2*median')
        coefficients = np.asarray(mapping.coefficients, dtype=float)
        if coefficients.shape != (3,) or not np.isfinite(coefficients).all():
            raise ValueError(f'{method}: invalid calibration coefficients')
        mappings[method] = {'coefficients': coefficients.tolist(), 'fit_partition': 'train',
                            'n_anchors': len(subset), 'ols_used': not mapping.no_ols,
                            'ols_rejection': mapping.no_ols_why}
    return mappings


def order_accuracy(frame, epsilon=10 / 60):
    """Strict order accuracy outside the raw-minute tolerance, including ROI ties as errors."""
    rt, roi = frame.rt.to_numpy(), frame.roi.to_numpy()
    correct = total = 0
    # Vectorized rows keep memory linear in the number of held-out observations.
    for i in range(len(frame) - 1):
        delta = rt[i + 1:] - rt[i]
        comparable = np.abs(delta) >= epsilon
        total += int(comparable.sum())
        correct += int(((roi[i + 1:] - roi[i]) * delta > 0)[comparable].sum())
    return {'accuracy': correct / total if total else None, 'correct_pairs': correct,
            'comparable_pairs': total, 'epsilon_minutes': epsilon}


def evaluation_metrics(predictions):
    per_method = {}
    for method in METHODS:
        group = predictions.loc[predictions.dataset_id == method]
        if group.empty:
            raise ValueError(f'{method}: missing held-out observations')
        error = group.rt_pred.to_numpy() - group.rt.to_numpy()
        per_method[method] = {'rows': len(group), 'keys': int(group.inchikey2d.nunique()),
            'mae_minutes': float(np.abs(error).mean()),
            'median_ae_minutes': float(np.median(np.abs(error))),
            'rmse_minutes': float(np.sqrt(np.square(error).mean())),
            'retention_order': order_accuracy(group)}
    error = predictions.rt_pred.to_numpy() - predictions.rt.to_numpy()
    pairs = sum(value['retention_order']['comparable_pairs'] for value in per_method.values())
    correct = sum(value['retention_order']['correct_pairs'] for value in per_method.values())
    return {'per_method': per_method, 'aggregate': {
        'rows': len(predictions), 'keys': int(predictions.inchikey2d.nunique()),
        'mae_minutes': float(np.abs(error).mean()),
        'median_ae_minutes': float(np.median(np.abs(error))),
        'rmse_minutes': float(np.sqrt(np.square(error).mean())),
        'macro_mae_minutes': float(np.mean([v['mae_minutes'] for v in per_method.values()])),
        'retention_order_accuracy': correct / pairs if pairs else None,
        'comparable_pairs': pairs}}


def evaluate(model, frame, output, batch_size, cache=None):
    storage = model.extra_storage['specialized']
    if set(frame.inchikey2d) & set(storage['training_keys']):
        raise ValueError('Evaluation contains training 2D InChIKeys')
    predictions = predict_frame(model, frame, batch_size, cache)
    if not np.isfinite(predictions[['roi', 'rt_pred']].to_numpy()).all():
        raise ValueError('Nonfinite held-out predictions')
    predictions.to_csv(output.with_suffix('.tsv'), sep='\t', index=False)
    metrics = evaluation_metrics(predictions)
    output.with_suffix('.json').write_text(json.dumps(metrics, indent=2) + '\n')
    return metrics


def run(args):
    if args.smoke:
        args.epochs, args.batch_size, args.sampling_count = 1, 16, 128
        args.max_train_pairs, args.max_val_pairs = 256, 128
        args.encoder_size, args.sizes, args.sizes_sys = 64, [32, 16], [32, 32]
    if min(args.epochs, args.batch_size, args.sampling_count,
           args.max_train_pairs, args.max_val_pairs, args.encoder_size) <= 0:
        raise ValueError('Training sizes and counts must be positive')
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'best.pt').exists() or (output / 'model.pt').exists():
        raise ValueError('Output already contains a model; choose a new --out_dir')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.gpu:
        if not torch.cuda.is_available():
            raise ValueError('--gpu requested but CUDA is unavailable')
        torch.cuda.manual_seed_all(args.seed)
        torch.set_default_device('cuda')
    frame, scalers, audit = load_prepared(args.data_dir)
    if args.smoke:
        groups = []
        for (method, split), group in frame.groupby(['dataset_id', 'split_type'], sort=True):
            groups.append(group.sample(n=min(len(group), 32 if split == 'train' else 16),
                                       random_state=args.seed))
        frame = pd.concat(groups, ignore_index=True)
        scalers = fit_rt_scalers(frame)
        frame = add_scaled_rt(frame, scalers)
        validate_prepared(frame, scalers)
        frame.to_csv(output / 'smoke_input.tsv', sep='\t', index=False)
    partitions = {split: frame.loc[frame.split_type == split].reset_index(drop=True) for split in PARTITIONS}
    for split, subset in partitions.items():
        if set(subset.dataset_id) != set(METHODS):
            raise ValueError(f'{split}: all five methods must have observations')
    cache = {}
    print('Building bounded training/validation pairs', flush=True)
    train = make_rank_dataset(partitions['train'], scalers, cache, args.max_train_pairs,
                              args.pair_weight_mode)
    # Keep the validation pair pool independent of training pair-cap choices.
    random.seed(args.seed)
    val = make_rank_dataset(partitions['val'], scalers, cache, args.max_val_pairs,
                            args.pair_weight_mode)
    if not len(train) or not len(val):
        raise ValueError('No usable training or validation pairs')
    custom_collate.graph_batch = dmpnn_batch
    sampling_weights = calc_sampling_weights(train, method='pairs')
    sampler = CustomWeightedRandomSampler(sampling_weights, args.sampling_count, replacement=True)
    generator = torch.Generator(device='cuda' if args.gpu else 'cpu').manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler,
                              collate_fn=custom_collate, generator=generator)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                            collate_fn=custom_collate, generator=generator)
    model = MPNranker(sys_features_dim=5, encoder_size=args.encoder_size,
                      hidden_units=args.sizes, hidden_units_pv=args.sizes_sys,
                      res_conn_enc=False, no_sigmoid_roi=True)
    set_model_device(model, 'cuda' if args.gpu else 'cpu')
    metadata = {'version': 1, 'feature_names': FEATURE_NAMES, 'methods': METHODS,
                'rt_scalers': scalers, 'config': vars(args).copy(),
                'training_keys': sorted(set(partitions['train'].inchikey2d)),
                'data_audit_sha256': sha256(Path(args.data_dir) / 'audit.json'),
                'reference': audit['reference'],
                'pair_counts': {'train': len(train), 'val': len(val)},
                'smoke_only': args.smoke}
    if 'selection' in audit:
        metadata['data_selection'] = audit['selection']
    model.extra_storage = {'specialized': metadata}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = torch.nn.MarginRankingLoss(margin=0.1, reduction='none')
    best_loss, history = float('inf'), []
    started = time.time()
    with SummaryWriter(str(output / 'tensorboard')) as writer:
        for epoch in range(args.epochs):
            model.train()
            total, count = 0.0, 0
            for x, y, weights, _ in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}'):
                optimizer.zero_grad()
                mean_loss, losses = model.loss_step(x, y, weights, loss_function)
                if not torch.isfinite(mean_loss):
                    raise ValueError('Training loss is not finite')
                mean_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                total += losses.detach().sum().item()
                count += len(losses)
            val_loss = validation_loss(model, val_loader, loss_function)
            if not np.isfinite(val_loss):
                raise ValueError('Validation loss is not finite')
            model.max_epoch = epoch + 1
            record = {'epoch': epoch + 1, 'train_loss': total / count, 'val_loss': val_loss}
            history.append(record)
            writer.add_scalar('loss/train', record['train_loss'], epoch + 1)
            writer.add_scalar('loss/validation', val_loss, epoch + 1)
            if val_loss < best_loss:
                best_loss = val_loss
                metadata['best_epoch'] = epoch + 1
                metadata['best_val_loss'] = val_loss
                torch.save(model, output / 'best.pt')
            (output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
            print(json.dumps(record), flush=True)
    # Calibrate and evaluate only the chosen validation checkpoint.
    from predict import load_model
    model = load_model(str(output / 'best.pt'), all_in_one=True)
    set_model_device(model, 'cuda' if args.gpu else 'cpu')
    model.extra_storage['specialized']['mappings'] = calibrate(
        model, partitions['train'], scalers, cache, args.batch_size)
    torch.save(model, output / 'model.pt')
    (output / 'config.json').write_text(json.dumps(model.extra_storage['specialized'], indent=2) + '\n')
    metrics = {split: evaluate(model, partitions[split], output / f'{split}_predictions',
                               args.batch_size, cache) for split in ['val', 'test']}
    summary = {'smoke_only': args.smoke, 'elapsed_seconds': time.time() - started,
               'best_epoch': model.extra_storage['specialized']['best_epoch'],
               'best_val_loss': best_loss,
               'counts': {split: len(subset) for split, subset in partitions.items()},
               'aggregate_metrics': {split: metric['aggregate'] for split, metric in metrics.items()}}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', default='data/specialized')
    parser.add_argument('--out_dir', default='results/specialized')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='One small CPU/GPU validation run; not a scientific benchmark')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--sampling_count', type=int, default=500_000)
    parser.add_argument('--max_train_pairs', type=int, default=500_000)
    parser.add_argument('--max_val_pairs', type=int, default=100_000)
    parser.add_argument('--pair_weight_mode', choices=['normalized', 'paper'], default='normalized',
                        help='paper: sigmoid(20*(absolute RT difference in minutes - 0.75)); '
                             'normalized: original specialized training-range weights')
    parser.add_argument('--encoder_size', type=int, default=512)
    parser.add_argument('--sizes', type=int, nargs='+', default=[256, 64])
    parser.add_argument('--sizes_sys', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
