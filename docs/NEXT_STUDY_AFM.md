# Next study: sample-efficient analysis of biological force curves

**Status:** proposed follow-up; no AFM experiments have been run.

**Recorded:** 15 October 2026.

**Research lead:** Feiyu Lu. No external collaboration or endorsement is implied.

## Research question

Can a pretrained tabular transformer classify molecular rupture events from atomic-force microscopy (AFM) force–distance curves with fewer expert labels than conventional machine learning and a specialized few-shot neural network?

The primary outcome is prediction quality at a fixed labeling budget. A second question is whether any advantage transfers to a different experimental session or biological sample. Regional sensitivity analysis will examine which portions of the curve support predictions, without equating importance with molecular causality.

## Why this follows the XANES study

Both problems involve numerical measurement curves, limited labeled data, established baselines, and signal regions that can be examined physically. AFM adds a practical outcome: reducing the number of curves an expert must label while preserving useful classification performance. The model comparison therefore addresses an experimental workflow rather than accuracy alone.

An AFM probe measures forces as it approaches or retracts from a sample. Molecular rupture events create characteristic changes in force–distance curves. Screening these measurements manually can be laborious. A small-data foundation model is a plausible candidate, but its advantage is a hypothesis: convolutional models may handle shifted events and local shape more effectively.

## Starting evidence and novelty check

The initial benchmark is [Waite et al., *Patterns* (2023), “Few-shot deep learning for AFM force curve characterization of single-molecule interactions”](https://doi.org/10.1016/j.patter.2022.100672) ([open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9868661/)). The authors report a balanced set of 216 carefully labeled curves, selected from approximately 1,600 measurements, and publicly released data and annotations. Their three-class task distinguishes single rupture, double rupture, and a combined no-/multiple-rupture class. The comparison includes a specialized few-shot neural network, random forest, SVM and nearest neighbors.

The released files, executable code, licensing, exact splits and sample identifiers have **not yet been audited locally**. The stringent curve selection also limits what a benchmark result would say about noisy, routine measurements.

A targeted literature search on 15 October 2026 found no direct TabPFN comparison for this force-curve task. This is not an exhaustive novelty claim. Before freezing the protocol, search the paper's citing literature, AFM/force-spectroscopy variants, TabPFN and other tabular foundation models; record search terms, dates and relevant exclusions.

Related work to inspect as a possible external benchmark is [“Automated atomic force microscopy analysis using convolutional and recurrent neural networks” (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12256878/). Its curve-quality and contact-point tasks differ from rupture classification; do not merge their labels or claim transfer between them without a valid common task.

## Stage 1 — establish whether the benchmark is usable

1. Obtain the authors' released data, annotations and code through the publication's data-availability links. Save source URLs, license terms, file hashes and acquisition dates. Keep raw inputs immutable.
2. Reconcile curve counts, labels, preprocessing, original partitions and reported metrics. Preserve the combined no-/multiple-rupture class for reproduction.
3. Check duplicate and near-duplicate curves, lengths, units, acquisition parameters, label provenance and missing data. Identify cell, specimen, cantilever, experimental-day and session identifiers where available.
4. Determine the independent experimental unit. Keep all repeated measurements from that unit in one partition. Do not invent independent groups for curves whose identities are missing.
5. Audit available checkpoint pretraining information for benchmark overlap. Uncertain overlap must be disclosed; it cannot be solved merely by withholding local test labels.

**Decision gate:** continue only if labels and original evaluation are reconstructable. If group identifiers are absent, retain the archived-split reproduction but label it as such. A new-experiment claim then requires an additional, independently acquired labeled dataset or collaboration with the data authors.

## Stage 2 — freeze the comparison before test evaluation

Commit the protocol, dataset manifest and configurations before examining challenger test scores. Pin software and checkpoint revisions after a validation-only compatibility check.

### Models and inputs

- Reproduce the authors' few-shot neural model and reported conventional baselines when code and dependencies permit.
- Compare standard TabPFN, a tuned RF, SVM and one strong boosting baseline, initially CatBoost. Treat the paper's fixed RF separately from the tuned RF.
- Evaluate the same information for every model. Start with the authors' preprocessing; audit whether curve trimming or alignment assumes knowledge of the label or rupture location.
- Allow only a small, declared set of representations: resampled curves and train-fitted dimensionality reduction or physical descriptors. Fit learned preprocessing exclusively on the training partition, within each validation fold.
- Keep test labels unavailable to feature selection, model selection, preprocessing choices and interpretation-driven revisions.

### Evaluation

- First reproduce the original partition and metrics. Then use grouped evaluation if metadata support it; retain an untouched group-level test set, with model selection within the remaining groups.
- Plot learning curves at predetermined labeling budgets. Candidate budgets are 10, 20 and 40 labels per class, subject to the audited training-set size and group constraints. Use identical training subsets across models and repeat subset selection with five fixed seeds where feasible.
- Primary metric: macro-F1. Also report per-class recall/F1, confusion matrices, accuracy, probability calibration where applicable, runtime, GPU memory and CPU memory.
- Report uncertainty at the independent experimental-unit level. With too few groups, present results descriptively instead of treating individual curves as independent biological replicates.
- Treat derived “labels saved at a target score” as descriptive unless a target and estimation rule were fixed in advance. Record actual labeling time only if measured; model accuracy alone does not establish hours saved.
- Do not discard difficult curves or change the principal metric after viewing results. Report adverse and neutral comparisons.

## Stage 3 — interpretation with a practical check

Use matched regional permutations on held-out validation curves for RF and TabPFN. Define curve regions before analysis and use five shared donor shuffles. Preserve each region's internal shape. Report macro-F1 decrease, variability across shuffles and stability across training subsets; these are distinct sources of uncertainty.

Check that perturbations do not create scientifically implausible conclusions. Rupture locations can shift between curves, so absolute-distance importance may partly reflect alignment. Compare with a small, predefined region-removal/retraining control selected on validation data. Any alignment inferred from a full curve must be label-independent and documented.

An importance peak is model sensitivity under a specified perturbation. It is neither an attention weight nor proof of the physical mechanism of binding.

## Cloud execution plan

Use local computation for inspection, code review and report generation. Plan model execution on a single cloud GPU to avoid the memory constraints of the earlier 16 GB Mac runs.

| Item | Proposed initial choice |
|---|---|
| Compute | One CUDA GPU with at least 24 GB VRAM; 32–64 GB host RAM |
| CPU workers | Four; one experiment at a time |
| Environment | Pinned Linux container; separate legacy reproduction environment if required |
| Initial smoke test | One training budget, one seed, validation only; compare deterministic metrics independently |
| Pilot limit | At most two GPU-hours initially; expand only after measuring memory, runtime and usefulness |
| Storage | Persistent data/results storage independent of the GPU instance |
| Checkpointing | Save each completed model/subset/permutation with hashes and predictions; resume without repeating completed work |
| Shutdown | Explicit timeout, idle shutdown and provider spending controls; export artifacts before termination |

The GPU class is a starting resource envelope, not a measured requirement. With 216 curves, CPU execution or a smaller instance may suffice; the smoke test determines the economical option. Record the actual accelerator, driver, package versions and precision. If comparing with local inference, use the same small input and assess agreement before interpreting backend differences.

**Provider, hourly price and dollar budget remain undecided.** This note does not launch a machine, purchase compute or upload data. Verify dataset/model terms before cloud execution; exclude credentials from repositories and logs.

## Deliverables and criteria for proceeding

Create a separate AFM experiment repository once the data audit passes, keeping this repository focused on the completed XANES benchmark. The planned interface is `prepare`, `run`, `report` and `status`, with a clean executed notebook, immutable split manifests, saved predictions, compact provenance records and reproducible figure builders.

The main figures should show:

1. Representative annotated force curves and the prediction task.
2. Macro-F1 versus expert-label budget, with paired models and uncertainty.
3. Within-benchmark versus held-out-experiment performance, only if supported by metadata.
4. Regional sensitivity and the complementary region-removal result.

A useful outcome could be higher accuracy at the same labeling budget, comparable accuracy with fewer labels, or a reproducible finding that the specialized model transfers better. Publication or outreach should follow the evidence. A marginal win on one small curated split alone is insufficient for broad biological or deployment claims.

## Immediate next action

Audit the original benchmark's files and experimental identifiers, reproduce one baseline, and estimate the cloud pilot's cost. Freeze the final protocol after that audit and before challenger test evaluation. Training and cloud provisioning have not started.
