# Five-method retention-time model

The Broad specialized entrypoint reuses the repository's D-MPNN molecular encoder,
system–molecule preference layers, pairwise margin-ranking loss, and LAD/OLS
retention-time mapping. It trains from scratch. It requires no RepoRT metadata.

The five source Excel workbooks, reference snapshots, MS2 selection audit,
specialized model weights, and generated reports are not bundled in Git.
Supply the workbooks under `data/` and access to the reference MS/MS dataset
before preparing a training snapshot. Counts and run paths below document the
local experiments; they are not files shipped with a fresh clone.

## Environment and data preparation

```bash
conda env create -f env_specialized.yaml
conda run -n 2-step-specialized python prepare_specialized.py \
    --input_dir data --output_dir data/specialized \
    --unmatched_policy train --expected_unmatched_keys 335
```

The environment is separate from `ms-gen`. RDKit is pinned to 2025.03.6, the
version used for the structure/coverage audit. The YAML installs CPU PyTorch;
use a CUDA-enabled PyTorch installation on a GPU training machine.

Initial preparation reads `labels.tsv` and `splits/split_1.tsv` at
`molgpu03.mit.edu:/home/runzhong/ms-pred/data/spec_datasets/all_msms_20260826`.
It joins on `spec`, reduces each InChIKey to its first 14 characters, and uses
`Fold_0`. It checks for conflicting partitions before saving a local mapping
snapshot and checksums. Later preparation runs reuse the verified snapshot;
`--refresh_reference` explicitly requests a new snapshot.

Every measured RT row follows that mapping, including stereoisomers and
measurements on other methods. The approved **335 keys absent from the reference
are assigned to training**, representing 460 measurements. The manifest marks
them `rt_only_training`; existing assignments are never overridden. The default
unmatched policy is `error`, so this exception must be explicit. The expected
count argument guards against unexpected input or structure-processing changes.

| Partition | Unique 2D InChIKeys | RT measurements |
| --- | ---: | ---: |
| Train | 18,047 | 74,737 |
| Test | 2,171 | 9,232 |
| Validation | 2,007 | 8,536 |

All three pairwise key overlaps are zero. The proportions follow the existing
split and the training additions; there is no new random 80/10/10 split.

Prepared files under `data/specialized/` include `all.tsv`, `train.tsv`,
`test.tsv`, `val.tsv`, `key_assignments.tsv`, `rt_scalers.json`, the reference
snapshot, and `audit.json`. Source file, sheet, row, library ID, original SMILES,
and InChI are retained for tracing each observation.

Preparation excludes 26,370 rows without usable RTs and five Ferroquine
measurements whose structure cannot generate an InChIKey. It canonicalizes
isomeric SMILES and removes only exact within-method SMILES/RT duplicates.
No filtering based on MS/MS detection, adducts, SNR, or precursor mass is applied
to the RT measurements. `excluded_rows.tsv` records exclusions.

The source InChI connectivity disagrees with SMILES-derived connectivity in
6,444 accepted rows. SMILES determine model inputs and split identities;
`source_inchi_key` and `inchi_disagrees` in the prepared tables expose these
inconsistencies. For the 335 unmatched keys, the pre-QC Broad labels explain
149 through recorded exclusions (formula QC, unsupported adducts, or precursor
mass errors); the other 186 are absent from those pre-QC labels. All 335 use the
approved training exception.

## Features, scaling, and training

Features are binary, in order `[positive, negative, C8, C18, HILIC]`:

| Code | Polarity | Method | Features |
| --- | --- | --- | --- |
| CN | Negative | C18 | `[0,1,0,1,0]` |
| HP | Positive | HILIC | `[1,0,0,0,1]` |
| CP | Positive | C8 | `[1,0,1,0,0]` |
| HN | Negative | HILIC | `[0,1,0,0,1]` |
| C18P | Positive | C18 | `[1,0,0,1,0]` |

CN was corrected from C8 negative to C18 negative on 2026-10-16. The corrected
prepared snapshot is `data/specialized_c18_corrected_20261016/`. Reprepare data
in a new directory for retraining; old prepared data is rejected by the current
trainer. Existing checkpoints keep their saved training-time method encoding
when predicting or evaluating, including the legacy CN=C8 assignment.

The corrected snapshot preserves the original split snapshot and all RT values.
To train it with the latest paper-weight settings, use a new output directory:

```bash
conda run -n 2-step-specialized python train_specialized.py \
    --data_dir data/specialized_c18_corrected_20261016 \
    --out_dir results/specialized_cn_c18 --pair_weight_mode paper
```

Add `--gpu` in the CUDA-enabled remote environment. The full corrected run was
launched on molgpu03 GPU 0 in screen `rt_c18_20261016T200442Z`, under
`/home/runzhong/2-step-RT/launches/20261016T200442Z_cn_c18/`. Its outputs are in
`results/cn_c18/` there; the local launch record and split validation are in
`results/retrain_cn_c18_20261016T200442Z/`.

The supplied mapping takes precedence over workbook title labels. Binary
features are not standardized. Raw `RT` is interpreted as minutes; workbook
`scaled_RT` is retained for provenance but is not a target.

For each method, training-only RT bounds define
`rt_normalized = (rt - min_train_rt) / (max_train_rt - min_train_rt)`.
The model learns order; raw RTs determine pair direction and the 10-second
evaluation tolerance. Pair weights use normalized RT differences, with logistic
midpoint 0.05 and steepness 20. No 30-minute runtime is assumed. Held-out values
outside the training range remain outside `[0,1]`.

For a controlled pair-weight ablation, use `--pair_weight_mode paper`. This uses
the paper's raw-minute weight `sigmoid(20 * (abs(rt1 - rt2) - 0.75))` and the
authors' existing small-weight cutoff. It applies to training and validation
losses; calibration, pair-pool caps, and checkpoint selection stay the same.
The default `normalized` mode preserves the original specialized run. Use a
separate output directory for every ablation.

```bash
conda run -n 2-step-specialized python train_specialized.py \
    --data_dir data/specialized --out_dir results/specialized
```

Defaults are 10 epochs, batch size 512, learning rate 0.0005, encoder size 512,
ranking layers 256/64, system layers 256/256, no encoder residual connection,
and no final ROI sigmoid. Seed 42 controls Python, NumPy, and PyTorch randomness;
CUDA numerical results can still vary by hardware and kernels.

Pairs belong to one method and one partition. Training samples up to 500,000
unique candidate pairs per method without enumerating all combinations and
draws 500,000 pairs per epoch with replacement, balancing the five methods.
Validation uses up to 100,000 candidate pairs per method in fixed order.
Ambiguous repeated-structure pairs are filtered by the existing pair machinery.

The best epoch is selected using the sample-weighted mean validation loss.
There is no exhaustive training-accuracy calculation. After selection, the
trainer fits each method's LAD/OLS mapping using training anchors only, in
normalized RT space, and converts output back to minutes. OLS follows the
existing refinement and nonnegative-coefficient checks.

Outputs include `best.pt` (the selected ROI checkpoint), `model.pt` (the complete
model with training-fitted calibration), configuration, epoch history,
TensorBoard logs, and validation/test predictions and metrics. Calibration is
performed once for the selected checkpoint. Test results do not select the
checkpoint. Existing model directories are not overwritten.

## Prediction and evaluation

### MS2-supported retraining snapshot

`prepare_ms2_specialized.py` creates a separate training-ready snapshot from the
separately supplied MS2 support audit in `reports/ms2_cleaned_rt_20261023/`.
Its `--reference_dir` must contain `broad_contributors.tsv`, `reference_keys.tsv`,
and `export.json`; `--selection_summary` points to `summary.json`, alongside
`filtered_rt/all.tsv` and `rt_support_audit.tsv`. The checksummed audit is an
input to this script, not an export generated by it. It requires the same
original method/catalog record and 2D molecular identity among contributors to
retained Broad spectra in `all_msms_20261014`. It uses the original frozen split;
the latest reference has compatible assignments. The filter never uses RT errors.

```bash
conda run -n 2-step-specialized python prepare_ms2_specialized.py
conda run -n 2-step-specialized python train_specialized.py \
    --data_dir data/specialized_ms2_clean_20261023 \
    --out_dir results/specialized_ms2_clean --pair_weight_mode paper
```

The cleaned snapshot has 61,499 training rows (16,894 keys), 7,026 validation
rows (1,907 keys), and 7,584 test rows (2,061 keys), with zero pairwise overlap.
The 335 RT-only training additions have no compatible MS2 support and are omitted
from this new snapshot. Original workbooks, prepared snapshots and models remain
available. New RT scalers and calibration are fitted using retained training rows
only; existing normalized RTs are recomputed. Checkpoints package the MS2 selection
provenance under `data_selection`.

Add `--gpu` in the verified remote CUDA environment. Use the same cleaned test
observations to compare with the prior frozen model (MAE 0.684748607 minutes).
MS2 QC supports merged-spectrum compatibility; it does not prove every raw
contributor, stereoisomer or RT peak assignment is correct.

The 2026-10-23 full retraining completed ten epochs on molgpu01 GPU 1 in screen
`rt_ms2clean_20261023T154845Z`. Its isolated directory is
`/home/runzhong/2-step-RT/launches/20261023T154845Z_ms2_clean/`, with outputs under
`results/ms2_clean/`. Local launch and validation records are in
`results/retrain_ms2_clean_20261023T154845Z/`. Epoch 6 was selected by validation
loss. The calibrated model achieved test MAE **0.601789 minutes**, RMSE
**1.008461 minutes**, and within-method retention-order accuracy **92.9004%**
on 7,584 observations (2,061 unique 2D keys). The previous corrected-CN model
had MAE 0.684749 minutes on those same observations. The local
`baseline_comparison.json` records the matched comparison.

Use **`model.pt`** for prediction; `best.pt` has no RT calibration yet. Provide
a TSV with a `smiles` column (`smiles.std` is also accepted), and select one method:

```bash
conda run -n 2-step-specialized python predict.py \
    --model results/specialized/model.pt --method CN \
    --input_compounds compounds.tsv --output_roi --out predictions.tsv
```

The specialized path preserves query rows and input columns, appending
`rt_pred` in minutes and optionally `roi`. Input `rt` annotations do not refit
the stored training calibration. Do not provide `--input_metadata` for a
specialized model. Original models retain their metadata-based prediction path.

```bash
conda run -n 2-step-specialized python evaluate_specialized.py \
    --model results/specialized/model.pt --data_dir data/specialized \
    --partition test --out results/specialized/test_evaluation
```

Evaluation checks dataset provenance and rejects any overlap with packaged
training keys. Reports contain per-method and pooled MAE, median absolute error,
and RMSE in minutes, macro-average MAE, and strict retention-order accuracy for
pairs at least 10 seconds apart. ROI ties count as incorrect order. Pooled order
accuracy combines within-method comparisons only.

## Same-formula PubChem retrieval

`retrieve_specialized.py` evaluates a frozen calibrated model against every
candidate in the supplied full PubChem formula mapping. The full mapping on
molgpu is `ms-pred/data/retrieval/pubchem/pubchem_formula_map.p`; the older HDF5
copy and dataset-specific subsets are separate snapshots.

```bash
python retrieve_specialized.py --gpu \
    --model results/full/model.pt --data_dir data/specialized \
    --partition test \
    --pubchem_map /home/runzhong/ms-pred/data/retrieval/pubchem/pubchem_formula_map.p \
    --out_dir results/pubchem_rt_retrieval
```

Query formulas come from the prepared source SMILES. Formula computation matches
the mapping generator: RDKit `CalcMolFormula` with the charge suffix removed,
retaining its hydrogen count. Candidates are checked against their formula and
stored connectivity key. No ground-truth candidate is appended and there is no
candidate cap or sampling. Invalid entries and missing formulas are audited.

For each experimental measurement, rank candidates by
`abs(predicted_RT_minutes - observed_RT_minutes)` for its method. Scoring uses
the stored training calibration without fitting to query RTs. The mapping
removes stereochemistry; correctness and candidate deduplication therefore use
the first 14 InChIKey characters. When several mapped structures share that key,
their best RT error represents the connectivity group. Exact error ties are
broken by alphabetical key order; optimistic and pessimistic tie bounds are
also reported. Molecular encoding is reused across methods, with equivalence
checked against the standard prediction path.

Reports distinguish all-query top-K accuracy (an absent truth cannot be a hit)
from accuracy conditional on truth being available. They include candidate
coverage, per-method results, equal weighting per 2D query key, a uniform random
ranking baseline, candidate-count statistics, every query rank, all candidate
predictions, and exclusion audits. Defaults report K=1,5,10,20,50,100. Snapshot
hashes identify the model and candidate database. These results measure
retrieval over that local PubChem snapshot, rather than current live PubChem.

Formula extraction can run once using `--extract_only`. Disjoint GPU workers
can reuse its `formula_subset.p` via `--formula_subset` and
`--shard_index i --shard_count n`. Keep separate output directories per shard
and verify their union contains every test query exactly once before reporting
pooled results. Completed formulas are reusable when resuming an identical run.

## Smoke test and focused checks

```bash
conda run -n 2-step-specialized python -m pytest -q
conda run -n 2-step-specialized python train_specialized.py \
    --smoke --data_dir data/specialized --out_dir results/specialized_smoke
```

The smoke run samples 32 training and 16 validation/test rows per method, fits
its own training-only scalers, uses a smaller network, and trains one epoch.
It writes the actual subset and marks the package `smoke_only`. Its metrics
demonstrate pipeline operation and are not full-dataset performance estimates.
