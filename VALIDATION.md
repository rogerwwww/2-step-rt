# Specialized RT adaptation validation

Verified locally on 2026-10-02 in the isolated `2-step-specialized` environment:
Python 3.12, PyTorch 2.5.1 CPU, Chemprop 1.6.1, RDKit 2025.03.6,
NumPy 1.26.4, and pandas 2.3.3.

## Data and reference verification

The reference was read from molgpu03. `labels.tsv` contains 1,333,625 spectrum
records; the `split_1.tsv` join covers every record and gives one partition per
valid 2D InChIKey. Three reference labels lack keys. The local reference snapshot
contains 297,331 unique connectivity keys.

Source `split_1.tsv` SHA256:
`485c54e9d9f585b9ca3be987bba75b66a4046e5d9a1a6a008e7a1b459c86edba`.
The label checksum, mapping checksum, workbook checksums, and prepared artifact
checksums are saved in [data/specialized/audit.json](data/specialized/audit.json).

| Partition | Unique 2D keys | Measurements |
| --- | ---: | ---: |
| Train | 18,047 | 74,737 |
| Test | 2,171 | 9,232 |
| Validation | 2,007 | 8,536 |

All pairwise key overlaps are zero. All 21,890 matched RT keys retain their
reference partition; all 335 unmatched valid keys are in training, with explicit
RT-only provenance. Those additions account for 460 measurements.

Excluded: 26,370 rows without usable RTs and five Ferroquine measurements with
failed InChIKey generation. Source InChI/SMILES disagreement is exposed in 6,444
accepted rows (C18P 4,155; CN 627; HP 567; CP 552; HN 543). The workbooks were
not edited and source InChIs were not substituted for model structures.

## Initial checks (2026-10-02)

- Eleven focused tests pass: encodings, stereoisomer/method isolation,
  reference conflicts, training-only scaling, unclipped extrapolation,
  deterministic bounded pair sampling (including a 20,000-compound case),
  raw-minute tolerance, external loader handling, stored calibration,
  independent sample-weighted validation loss, training-only anchors, and
  within-method evaluation metrics.
- A bounded one-epoch CPU smoke run completes training, best-checkpoint
  selection, five training-only calibrations, packaging, and held-out prediction.
  It uses 160 training, 80 validation, and 80 test observations.
- The specialized prediction CLI preserves all 413 supplied example rows and
  produces finite ROI/RT outputs. Reloading the package reproduces saved held-out
  RT predictions within numerical tolerance.
- The original setup-agnostic model still completes the original metadata-based
  prediction example.
- Python compilation and `git diff --check` pass.

Those initial smoke metrics validate execution only. See
[SPECIALIZED.md](SPECIALIZED.md) for full training commands. Data snapshots,
models, and runtime validation artifacts referenced here are local and are not
included in Git.

## Completed MS2-supported retraining

The cleaned-data run completed ten epochs and selected epoch 6 using validation
loss. It uses CN = C18 negative and preserves the original molecular split.
The retained train/validation/test partitions have 61,499/7,026/7,584 rows and
16,894/1,907/2,061 unique 2D InChIKeys, with zero pairwise overlap. Scaling and
calibration use retained training observations only.

On the same cleaned test observations, the previous corrected-CN model versus
the retrained model achieved:

| Metric | Previous model | Cleaned-data retraining |
| --- | ---: | ---: |
| MAE (minutes) | 0.684749 | 0.601789 |
| RMSE (minutes) | 1.133302 | 1.008461 |
| Retention-order accuracy | 92.0234% | 92.9004% |

After optimization completed, a CUDA runtime failure interrupted calibration.
Calibration and evaluation were recovered on CPU with unchanged neural weights.
Use the calibrated `model.pt`, rather than the uncalibrated `best.pt`.
