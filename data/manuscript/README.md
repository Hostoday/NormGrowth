# Current manuscript observations

These files follow the revised manuscript updated on 2026-10-01. Numeric cells retain their source precision. Machine-specific source-path columns were removed; file hashes and selected columns are recorded in [data_sources.json](../provenance/data_sources.json).

| Files | Purpose |
|---|---|
| `rq1_per_edit.csv`, `rq1_vectors_per_edit.csv`, `rq1_reference_sensitivity_per_edit.csv` | Scalar summaries, finite-vector comparisons and normalization-reference sensitivity |
| `checkpoints.csv` | All 360 canonical checkpoints, both prompt contexts and case-wise second moments |
| `rewrite_endpoints.csv`, `locality_endpoints.csv` | All 40 endpoint geometry summaries |
| `endpoint_tf_1000.csv` | Complete-target teacher-forced EFF/GEN and Base-agreement LOC on 1,000 cases per family |
| `intervention_doses.csv`, `same_norm_conditions.csv` | Orthogonal-response and norm-matched state summaries |
| `rq3_selected_case_scores.csv` | Paired case scores, answer-content/terminator scores and common-dose membership |
| `same_norm_paired_contrasts.csv` | The 204 saved means and pointwise 95% intervals used by the forest plot |
| `expected/` | Independent reference tables for reaggregation and bootstrap checks |

Checkpoint rows also retain historical free-generation EFF/GEN columns for traceability. They are not used as the manuscript's endpoint TF scores or as the outcome of its RQ2 geometry–LOC analyses.

Case-level vector metrics are stored measurements. CPU reproduction does not reconstruct raw hidden vectors. TF endpoint rows are stored score aggregates; generating their token predictions requires the original external model and evaluation inputs.
