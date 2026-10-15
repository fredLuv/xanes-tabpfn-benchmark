# Release checks

Release validation on 2026-10-14:

- Recalculated metrics for all 143 successful test runs from individual predictions; matched the source metrics within 1e-12.
- Verified identical evaluation rows and labels in paired RF/TabPFN runs.
- Recomputed all 16 coordination importance profiles and confirmed matching donor permutations.
- Reconstructed all 16 conditional resampling summaries from released predictions.
- Verified the original archive checksum and all 144 pointwise input-array checksums and shapes.
- Executed all nine notebook code cells without errors; inspected figure labels and caveats.
- Refit the public RF worker for Cr coordination seed 42 in the recorded RF environment; predictions exactly matched the archived result.
- Inspected the release allowlist and scanned text for local private paths, credential patterns and private-document references. No manuscript, full-paper PDF, correspondence, raw feature arrays or checkpoint files are included.

The optional TabPFN adapter has not been rerun end to end for this release. The released TabPFN metrics are independently checked from the original saved inference outputs. The RF worker check did not exercise the resource-monitor wrapper, whose dependency is listed in the fresh RF environment. There are no new donor-shuffle experiments in this release.
