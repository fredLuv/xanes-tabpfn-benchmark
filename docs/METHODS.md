# Methods and limitations

## Scope

Eight elements (Ti, V, Cr, Mn, Fe, Co, Ni, Cu), three targets, 100 pointwise input features with FEFF/post-edge normalization. The original 157-feature polynomial representation and other models are outside this release. This is a comparison against the **fixed published RF configuration**, not a benchmark of fully tuned alternatives.

## Data and evaluation

The original prepared train/validation/test arrays are used without row reordering or repartitioning. Input checksums and shapes appear in `configs/benchmark.json`. The code's split is approximately 81/9/10 before coordination oversampling; the paper describes 80/10/10. Coordination training arrays are already oversampled. Validation and test arrays are not oversampled. No new spectral augmentation is used.

Configurations were carried forward from validation experiments and frozen before this test evaluation. The original protocol freeze time and hash are retained in the configuration. The study followed exploratory validation; it is not a preregistered blind model-selection exercise. Test arrays were not added to the fitting context. RF fits the training arrays; TabPFN conditions on the labeled training arrays with pretrained weights fixed.

RF has 300 trees, maximum depth 35, 8 candidate features per split, minimum split size 2 and minimum leaf size 1. The RF reproduction uses scikit-learn 1.3.2; the publication used 0.21.3. TabPFN uses package 9.1.0, standard model version 3.5, and a pinned checkpoint. Its default estimator scaling produced eight ensemble estimators in the recorded runs. CPU/MPS placement is recorded per case; backend changes are not silently substituted by the public rerun adapter.

Seeds 42, 43 and 44 describe model variability on a fixed data split. Means and population SD use matched seeds for both models. V coordination seed 43 exceeded memory, leaving two matched seeds for that case. Three model seeds do not establish statistical significance. Classification reports accuracy, macro-F1 (equal weight to CN = 4, 5, 6), and per-class F1. Regression reports MAE and R². Percentage changes use RF as the denominator. No pooling across elements is used to claim element-specific improvement.

## Regional permutation importance

On the archived validation split, model seed 42, divide the 100 channels into ten adjacent windows. For each window, transfer all ten values together from a randomly selected donor row, using a permutation without replacement. Leave the other 90 channels unchanged, predict again, and calculate:

`importance = baseline macro-F1 − shuffled macro-F1`.

Both models use identical donor orders. The data bundle includes the saved donor orders, labels, baseline predictions, shuffled predictions, window boundaries and energy grid, allowing every importance value to be recalculated. For Cr, the figure averages three shuffles; elsewhere it uses the first matched shuffle. Extra saved RF shuffles are retained in the evidence files but are not used to inflate the paired comparison. Negative importance is not clipped. The absorption strip is the distribution of spectrum maxima over validation rows, not an importance curve.

Permutation breaks associations with labels and other channels; correlated channels can share or mask predictive information, and shuffled spectra may be physically atypical. Effects depend on the chosen window width and metric. They do not measure internal attention/weights, establish causality, or explain why a transformer architecture performs better.

## Completed conditional stability check

A fixed protocol resampled validation rows 1,000 times, stratified to preserve CN class counts, using seed 20261014. Both models and all predictions share the same resampling weights. Fitted models and donor assignments remain fixed. The implementation verifies effects against scikit-learn; the public module reconstructs the saved top-retention rates and conditional effect envelopes.

Top-window retention is descriptive sensitivity to evaluated rows, **not a probability of a true ranking, a material-level confidence interval, or stability to new donor shuffles**. Close or unstable maxima should be described as influential regions rather than unique winners.

## Minimum importance follow-up

Five independently drawn donor permutations per window would assess sensitivity to the chosen donor order. This is an optional follow-up, not part of the released evidence. The released comparison retains the paired original three shuffles for Cr and one for the other seven elements. Exact rankings remain preliminary. Additional runs were stopped to prioritize reproducibility; partial follow-up results do not replace the paired comparison.

Repeating donor shuffles addresses permutation variability only. Robustness to model seed, window size, alternative perturbations and new materials are separate questions. Repeated shuffles would strengthen the current importance claims but would not cure split leakage or establish physical mechanisms.

## Additional work

Material overlap exists between the archived partitions. Grouped material splits, alias/duplicate audits, tuned RF and gradient-boosting baselines, pretraining-overlap investigation, and experimental spectra are needed for broader conclusions. These are extensions beyond the original-benchmark claim; incomplete work is not presented as completed evidence.
