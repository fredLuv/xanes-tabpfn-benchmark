# Release checks

## Runnable release — 15 October 2026

[Machine-readable checks](../results/reproduction_checks.json) record the following verification of the public adapter:

- **Fresh RF environment:** installed `environments/rf.txt`, including the memory monitor; ran the command-line wrapper for Cr coordination seed 42. All held-out predictions exactly matched the release.
- **Actual RF importance inference:** refit Cr, recreated its 10 windows × 3 donor permutations, and compared all validation predictions. Baseline labels, every shuffled prediction and every importance value matched exactly.
- **Actual TabPFN inference:** ran Cr coordination seed 42 through the public wrapper on Apple MPS with the pinned existing TabPFN environment and checkpoint. All held-out predictions matched exactly; the run stayed within resource limits.
- **Original archive:** checked the archive hash and extracted all 144 required input arrays, verifying every checksum and shape.
- **Saved evidence:** independently reconstructed metrics for 143 test runs, checked paired labels and rows, and recomputed all 16 importance profiles and conditional resampling summaries.
- **Notebook and figures:** executed all nine notebook code cells; regenerated all four figures from individual predictions; checked numeric agreement with original plotted values and inspected the renderings.
- **Full-suite commands:** checked that the test phase enumerates 72 cases per model and the importance phase enumerates eight per model. Full research inference was not repeated for this software release.
- **Privacy and links:** checked local documentation links and scanned released text for private filesystem paths. No manuscript, full-paper PDF, correspondence, raw feature arrays, credentials or weights are included.

The full TabPFN importance inference loop was not rerun end to end for this release. It uses the recorded constructor, scheduling settings, validation labels, donor orders and window definitions, and requires baseline agreement before saving new predictions. The RF integration check exercises the same shuffle and checkpoint loop. A future rerun must report discrepancies if they occur, rather than assume bitwise recovery across platforms.

GitHub Actions verifies saved evidence, reconstructs conditional resampling, executes the notebook and rebuilds the figures on a clean Linux checkout. It does not download the 1.08 GB archive or access gated model weights.

The additional five-shuffle experiment was stopped at the user's request. Its partial results are not substituted into the released figures. The main importance comparison remains three paired shuffles for Cr and one for each other element.
