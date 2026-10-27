"""Evaluate a packaged specialized model on its prepared validation or test partition."""
import argparse
import json
from pathlib import Path

import torch

from predict import load_model
from train_specialized import load_prepared, evaluate
from specialized import sha256, set_model_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data_dir', default='data/specialized')
    parser.add_argument('--partition', choices=['val', 'test'], default='test')
    parser.add_argument('--out', default='results/specialized/evaluation')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.gpu:
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable')
        torch.set_default_device('cuda')
    model = load_model(args.model, all_in_one=True)
    set_model_device(model, 'cuda' if args.gpu else 'cpu')
    storage = model.extra_storage['specialized']
    if storage['smoke_only']:
        raise ValueError('Smoke models must not be evaluated as full-data models; use their smoke reports')
    if storage['data_audit_sha256'] != sha256(Path(args.data_dir) / 'audit.json'):
        raise ValueError('Model and evaluation data have different provenance')
    frame, _, _ = load_prepared(args.data_dir, storage['methods'])
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(model, frame.loc[frame.split_type == args.partition], output, args.batch_size)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
