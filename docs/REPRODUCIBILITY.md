# Reproduce the evidence

## 1. Verify and read the results (no model access needed)

Use Python 3.11 and `environments/report.txt`, then run:

```sh
python -m xanes_benchmark verify
python -m xanes_benchmark report
python -m xanes_benchmark.stability
```

Run `notebooks/01_benchmark_results.ipynb` top to bottom with that environment. It reads only released files, recalculates metrics and shows the verified saved figures. It does not call inference or download data. The notebook includes executed outputs for reading on GitHub.

`results/test_predictions.csv.gz` contains 59,637 rows across 143 completed model runs. The row index is the original test-array position within an element/task; it is **not a material identity**. `run_records.json` retains original metrics, timings, package versions, hashes and the one resource-limited case. `SHA256SUMS.json` pins the released evidence, configuration and figures. Exported compact files have their own hashes; original prediction hashes refer to the unmodified source archives.

`results/importance/` contains labels and model predictions, donor permutations and energy metadata, without absorption input arrays. Recomputing importance from those predictions does not create new shuffled model predictions.

## 2. Obtain the original input arrays

Raw spectra and weights are excluded. Check the providers' terms before use. Prepare verifies the authors' archive checksum and extracts only the 144 arrays needed by this pointwise comparison, retaining original row order.

```sh
python -m xanes_benchmark prepare
# Or use an existing original archive:
python -m xanes_benchmark prepare --archive /path/to/xanes_2019.zip
```

The download is approximately 1.08 GB. Inputs are placed in ignored `data/`. The configuration pins each array's checksum and shape. No further oversampling or learned feature preprocessing is applied by the adapter.

## 3. Optional frozen model reruns

Create **separate** RF and TabPFN environments using `environments/rf.txt` and `environments/tabpfn.txt`. The `*-recorded.txt` files document the broader original development environments; unrelated optional model packages in those records are not required. The lightweight public adapter was checked against saved RF predictions. The packaged TabPFN adapter follows the recorded constructor and checkpoint settings; a full new TabPFN rerun was not performed for this release. Numerical agreement may depend on the recorded platform/backend.

```sh
# In the RF environment:
python -m xanes_benchmark run --model rf --element Cr --task coord --seed 42

# In the TabPFN environment, after obtaining the pinned checkpoint:
python -m xanes_benchmark run --model tabpfn --element Cr --task coord --seed 42 \
  --checkpoint /path/to/tabpfn-v3.5-20260909.safetensors
```

Obtain the checkpoint from [Prior-Labs/tabpfn_3_5](https://huggingface.co/Prior-Labs/tabpfn_3_5), revision `06bf2ba35c80a92a3b9abb436b99cf49e7a0365e`. Its SHA-256 is verified before fitting. Account access and the provider's model license may be required. No credential or checkpoint is included here; package and weight licenses are distinct.

The adapter runs one case at a time, with four threads, a 10 GB conservative RSS-plus-MPS limit and a 30-minute limit. This memory accounting may double count shared memory. Saved records distinguish limits from success. Existing outputs are never silently overwritten; test results are not used for tuning. Optional runs go to ignored `runs/` and do not replace the released evidence. The adapter refuses to substitute CPU for a case recorded on Apple MPS.

## 4. Reconstruct the importance calculation

The essential inference operation used for each saved donor order is:

```python
shuffled = valid_x.copy()
shuffled[:, start:stop] = valid_x[donor_order, start:stop]
shuffled_prediction = fitted_model.predict(shuffled)
importance = macro_f1(valid_y, baseline_prediction) - macro_f1(valid_y, shuffled_prediction)
```

Fit only on `train_x, train_y`, verify the baseline predictions against the saved validation baseline, then use the released donor orders and windows. Refit settings for existing importance runs are retained in `results/importance/provenance.json`; some used memory-saving batch scheduling. This release provides metric verification and conditional resampling, not a new automatic importance queue. Follow-up donor seeds must be frozen before running additional shuffles.

## Provenance and release scope

The public companion has fresh Git history and includes only code, notebooks, results and documentation. It does not include a manuscript, full-paper PDF, correspondence, raw feature arrays or weights. Original record hashes allow provenance tracking; no local filesystem paths are needed to run the analysis. The main analysis is restricted to complete evidence, with the V exception and preliminary importance status made explicit.
