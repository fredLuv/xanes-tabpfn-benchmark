# Reproduce the benchmark

There are two routes. **Route A** recomputes metrics and rebuilds every figure from released individual predictions; it requires neither model weights nor the original archive. **Route B** generates predictions again by fitting RF or conditioning the pretrained TabPFN on the original training arrays, then evaluating the original held-out rows.

Use Python 3.11 and run commands from the repository root. No files from another repository or private workspace are needed. The reference runs used macOS on an M4 Mac with 16 GB memory. Analysis works on other platforms; exact model reruns retain the recorded CPU/Apple MPS settings.

## Route A — rebuild the analysis and notebook

```sh
git clone https://github.com/fredLuv/xanes-tabpfn-benchmark.git
cd xanes-tabpfn-benchmark
python3.11 -m venv .venv-report
.venv-report/bin/pip install -r environments/report.txt
.venv-report/bin/python -m xanes_benchmark verify
.venv-report/bin/python -m xanes_benchmark report
.venv-report/bin/python scripts/execute_notebook.py
```

The last command executes [the notebook](../notebooks/01_benchmark_results.ipynb) from top to bottom, rebuilding all four figures in `output/figures/`. It uses the chosen interpreter without requiring a globally registered Jupyter kernel. To rebuild figures alone:

```sh
.venv-report/bin/python -m xanes_benchmark figures
```

Expected verification: **143 test runs** (72 RF, 71 TabPFN), matched test rows, and **16 importance profiles** (eight elements × two models). V coordination seed 43 was memory-limited, so comparisons use its two available matched seeds. This is preserved rather than filled in. New renderings may differ in font metadata across systems; plotted numbers are checked against the released evidence.

## Route B — reproduce model predictions from the authors’ inputs

### 1. Download and verify the data

```sh
.venv-report/bin/python -m xanes_benchmark prepare
# If you already have the original archive, use this instead:
# .venv-report/bin/python -m xanes_benchmark prepare --archive /path/to/xanes_2019.zip
```

The [authors’ archive](https://data.matr.io/4/) is approximately 1.08 GB. The command checks its SHA-256 and extracts only the **144 pointwise arrays** used here into ignored `data/`. Each array is checked against the checksum and shape in [benchmark.json](../configs/benchmark.json). Original row order, labels, splits, normalization and training-only coordination oversampling are retained. There is no new split, oversampling or feature transformation in the public adapter.

### 2. Fit and test one RF model

```sh
python3.11 -m venv .venv-rf
.venv-rf/bin/pip install -r environments/rf.txt
.venv-rf/bin/python -m xanes_benchmark run --model rf --element Cr --task coord --seed 42
.venv-rf/bin/python -m xanes_benchmark compare runs/Cr_coord_rf_42
```

The runner fits on `train_x/train_y`, predicts `test_x`, and saves predictions, measured metrics, package versions, runtime and memory in `runs/Cr_coord_rf_42/`. `compare` checks row alignment and reports prediction and metric differences from the release; discrepancies are never relabeled as a match.

### 3. Condition and test one TabPFN model

```sh
python3.11 -m venv .venv-tabpfn
.venv-tabpfn/bin/pip install -r environments/tabpfn.txt
```

Obtain `tabpfn-v3.5-20260909.safetensors` from [Prior-Labs/tabpfn_3_5](https://huggingface.co/Prior-Labs/tabpfn_3_5), revision **`06bf2ba35c80a92a3b9abb436b99cf49e7a0365e`**, and save it under `weights/` as a real file, not a symlink to an extensionless cache file. The runner verifies its SHA-256. Access may require a provider account and license acceptance; weights and credentials are not redistributed. If you already have access through Hugging Face, this helper downloads the pinned file:

```sh
.venv-tabpfn/bin/python scripts/download_checkpoint.py
.venv-tabpfn/bin/python -m xanes_benchmark run --model tabpfn --element Cr --task coord --seed 42 \
  --checkpoint weights/tabpfn-v3.5-20260909.safetensors
.venv-tabpfn/bin/python -m xanes_benchmark compare runs/Cr_coord_tabpfn_42
```

TabPFN's pretrained weights remain fixed. Its `fit` method supplies the labeled training context; it does not perform gradient training on this dataset. Some cases, including Cr, use Apple MPS; others use CPU and memory-saving settings. The runner refuses silent backend substitution. A Linux/Windows reader can reproduce the analysis and RF, and CPU TabPFN cases; reproducing MPS cases exactly requires compatible Apple hardware. Provider access and hardware requirements mean we cannot promise every reader an identical inference environment.

### 4. Recompute actual regional permutation importance

```sh
.venv-rf/bin/python -m xanes_benchmark importance --model rf --element Cr
.venv-rf/bin/python -m xanes_benchmark compare runs/importance_Cr_rf_42

.venv-tabpfn/bin/python -m xanes_benchmark importance --model tabpfn --element Cr \
  --checkpoint weights/tabpfn-v3.5-20260909.safetensors
.venv-tabpfn/bin/python -m xanes_benchmark compare runs/importance_Cr_tabpfn_42
```

This **runs the model on newly constructed shuffled inputs**, rather than reading the answers from JSON:

1. Fit the original configuration on training rows only.
2. Predict validation rows and require exact agreement with the released baseline labels.
3. For each released donor permutation, jointly shuffle the ten absorption channels in one region across validation spectra.
4. Predict those modified spectra and calculate **baseline macro-F1 − shuffled macro-F1**.
5. Save each prediction checkpoint, its checksum, all importance values, and its agreement or disagreement with the original run.

The code is in [`importance_worker`](../xanes_benchmark/inference.py). The released arrays contain the exact donor orders and window boundaries. RF has three archived shuffles for every element; TabPFN has three for Cr and one for the others. The paired figure uses the **same first one or three donor orders for both models**. Additional RF repetitions do not inflate the paired evidence. The optional five-shuffle follow-up is not included in the main release.

### 5. Run the full released comparison

```sh
# 8 elements × 3 targets × 3 seeds, one model at a time:
.venv-rf/bin/python scripts/reproduce.py --model rf --phase test
.venv-tabpfn/bin/python scripts/reproduce.py --model tabpfn --phase test \
  --checkpoint weights/tabpfn-v3.5-20260909.safetensors

# The original eight coordination importance profiles per model:
.venv-rf/bin/python scripts/reproduce.py --model rf --phase importance
.venv-tabpfn/bin/python scripts/reproduce.py --model tabpfn --phase importance \
  --checkpoint weights/tabpfn-v3.5-20260909.safetensors
```

Use `--dry-run` to inspect the complete command list before execution. Each phase writes a summary under `runs/`, including failures and prediction comparisons. Completed cases are compared and skipped; existing failed test outputs are preserved. To make an independent rerun, add `--runs-dir runs/second_run` to the individual command or suite.

Full TabPFN inference takes substantially longer than analysis. One process runs at a time with four CPU threads, a conservative 10 GB RSS-plus-MPS ceiling and a 30-minute limit per invocation. Shared memory may be counted twice. A lock prevents competing benchmark jobs. Importance can resume atomic checkpoints with `--resume` after a limit; it refits the model and rechecks the baseline before continuing. No automatic resource relaxation, dataset reduction or new experimental configurations occur. Reruns never overwrite the released `results/`.

## Where each result comes from

| Stage | Runnable code | Inputs → outputs |
|---|---|---|
| Data preparation | [`prepare`](../xanes_benchmark/__main__.py) | Authors’ ZIP → 144 verified arrays |
| Model construction | [`build_model`](../xanes_benchmark/inference.py) | Frozen parameters + checkpoint → RF/TabPFN |
| Held-out prediction | [`worker`](../xanes_benchmark/__main__.py) | Train/test arrays → predictions and metrics |
| Regional importance | [`importance_worker`](../xanes_benchmark/inference.py) | Train/validation arrays + donor orders → shuffled predictions |
| Rerun comparison | [`compare`](../xanes_benchmark/inference.py) | Fresh predictions ↔ released predictions |
| Metric reconstruction | [`analysis.py`](../xanes_benchmark/analysis.py) | Individual predictions → paired tables |
| Chart construction | [`figures.py`](../xanes_benchmark/figures.py) | Predictions → all four figures and plotted values |
| Conditional resampling | [`stability.py`](../xanes_benchmark/stability.py) | Existing shuffled predictions → row-resampling sensitivity |
| Presentation | [Executed notebook](../notebooks/01_benchmark_results.ipynb) | Verified metrics + regenerated charts → readable results |

The notebook's default is Route A: it rebuilds from **released** predictions. Route B keeps new predictions separate and explicitly compares them against that release. It does not silently swap a new run into the published analysis.

## Provenance and limits

`results/test_predictions.csv.gz` contains 59,637 rows across 143 completed runs. Row numbers are original test-array positions, **not material identities**. `run_records.json` retains all 144 original statuses and provenance. `results/importance/` contains labels, predictions, donor orders and energy metadata, without raw absorption features. `SHA256SUMS.json` pins released evidence, configuration and figures. Package records document the original environments; minimal requirements install only the packages needed for each workflow.

Five new shuffles assess permutation randomness; they do not improve test accuracy or establish unseen-material generalization. Existing one-shuffle rankings remain preliminary. See [methods and limitations](METHODS.md). Code and derived results are public; the manuscript, correspondence, raw feature archive and weights are not included.
