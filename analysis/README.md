# Analysis

These modules compute statistics and figures from measurements produced by your experiments. Supply a flat input directory using `--data-dir`; the [input specification](INPUTS.md) describes the required tables. CSV inputs, computed results, and figures are not distributed in this repository.

Use the `normgrowth` environment, or Python 3.9 with `python -m pip install -r analysis/requirements.txt` for analysis alone. Run from the repository root:

```bash
python -m analysis.reproduce --data-dir inputs/measurements --output-dir build/results
python -m analysis.plot_core_figures --data-dir inputs/measurements --contrasts-file build/results/rq1_rq3/same_norm_paired_contrasts.csv --output-dir build/figures
```

`reproduce` calculates the statistics from input observations without comparing them to stored paper results. The figure command reads your checkpoint/endpoint measurements and the paired contrasts produced by the analysis. Both commands require the study's experimental design; they are not model execution commands. Input measurements and output directories must be separate.

## Computations

| Module | Computation |
|---|---|
| `reproduce.py` | RQ1 scalar summaries, intervention dose summaries, same-norm contrasts and the two modules below |
| `manuscript_statistics.py` | RQ2 rank correlations, controls, sensitivity panels, trajectory bootstrap intervals |
| `manuscript_vectors_interventions.py` | RQ1 vector/reference comparisons and RQ3 paired-case bootstrap intervals |
| `plot_core_figures.py` | Geometry and locality figures, restricted sensitivity views, combined RQ3 forest plot |
| `plot_rq3_forest.py` | Standalone combined RQ3 forest plot from computed contrast intervals |

RQ2 uses all 40 trajectories and nine checkpoints per trajectory as its primary panel. The geometry sensitivity requires both prompt contexts' mean orthogonal displacement, norm ratio, and absolute norm deviation to be at most 3. A separate sensitivity removes the Llama–MEMIT Native, SPHERE, and SADR trajectories on both datasets. Retained counts are determined from the supplied measurements.

Partial-rank correlations are Pearson correlations between residuals after ranking continuous variables and regressing each variable on the specified controls. Categorical edit-count and trajectory indicators are not ranked. The 5,000-draw bootstrap resamples whole trajectories, keeping their nine checkpoints together. Its intervals apply to full-panel unadjusted correlations and their paired differences; adjusted and sensitivity-panel coefficients are descriptive.

RQ3 pairs common eligible request IDs across operations and input families within each endpoint state and reduction fraction. The 10,000-draw intervals are pointwise and unadjusted for multiple comparisons. They are conditional on the measured endpoint, not confidence intervals over independent editing runs or the overall mean across states. Eligibility and contrast counts are computed from the inputs.

If `endpoint_tf_1000.csv` is supplied, the main command checks its schema, sample counts, and score range and exports it alongside the computed statistics. That optional step does not regenerate token predictions. EFF/GEN use complete-target teacher-forced accuracy including EOS/EOT; locality compares edited-model and Base predictions without an added terminator.

## Individual commands

```bash
python -m analysis.manuscript_statistics --data-dir inputs/measurements --output-dir build/rq2
python -m analysis.manuscript_vectors_interventions --data-dir inputs/measurements --output-dir build/rq1_rq3
python -m analysis.plot_rq3_forest --input-file build/rq1_rq3/same_norm_paired_contrasts.csv --output-dir build/figures
```

Computed tables and run reports go to the selected output directory. Figures are written as PDF, PNG, and SVG with their plotting coordinates. The combined forest plot keeps panel labels (a)–(d) and shared dataset labels; the model mapping belongs in the manuscript caption.

Keep measurements under `inputs/` or `local_data/`, and outputs under `build/` or `outputs/`. All are ignored by Git. The source-only check is `python -m analysis.check_release`.
