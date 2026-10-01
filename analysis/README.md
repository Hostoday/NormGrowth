# Analysis

Run from the repository root in Python 3.9 after installing `requirements.txt`:

```bash
python -m analysis.reproduce --output-dir build/results
python -m analysis.plot_core_figures --output-dir build/figures
python -m analysis.check_release
```

The default snapshot follows the current manuscript. It reads portable observations and independent reference tables from [data/manuscript/](../data/manuscript/README.md). Input hashes are checked before numerical reproduction. Outputs remain in the selected output directory.

## Numerical reproduction

| Analysis | Computation |
|---|---|
| RQ1 norms | 40,000 edits → 40 summaries and a common-denominator comparison |
| RQ1 vectors | 39,473 finite records; 39,257 paired displacement cosines; configuration and reference-sensitivity summaries |
| RQ2 primary | 360 checkpoints across 40 trajectories; dataset-specific and model-specific rank associations |
| RQ2 sensitivity | 337 checkpoints under the shared geometry restriction; 306 after excluding six collapsed trajectories |
| RQ2 controls | Norm deviation, edit count and trajectory controls; component composition after total-displacement adjustment |
| RQ2 intervals | 5,000 whole-trajectory bootstrap draws preserving all nine checkpoints; paired correlation differences |
| RQ3 | Equal-state means, 204 direct paired contrasts, 10,000-draw case-bootstrap intervals, common-dose and answer-token sensitivity |
| Endpoint performance | Validate and export the 40 saved complete-target TF endpoint score rows |

`reproduce.py` calls [manuscript_statistics.py](manuscript_statistics.py) for the RQ2 analyses and [manuscript_vectors_interventions.py](manuscript_vectors_interventions.py) for RQ1 vector and RQ3 case-level analyses. `build/results/validation.json` records the comparisons; the `rq2/` and `rq1_rq3/` subdirectories contain the regenerated tables and detailed validation reports.

Partial-rank correlations are Pearson correlations between residuals after ranking continuous variables and regressing each variable on the stated controls. Categorical edit-count and trajectory indicators are not ranked. The 5,000-draw intervals apply to full-panel unadjusted correlations and their paired differences; adjusted coefficients and sensitivity-panel coefficients do not have bootstrap intervals.

The geometry restriction requires both prompt contexts' mean orthogonal displacement, norm ratio and absolute norm deviation to be at most 3. The collapse sensitivity removes Llama–MEMIT Native, SPHERE and SADR trajectories on both datasets. These are distinct sensitivity panels. Case-wise second moments preserve the squared-norm decomposition; squares of checkpoint means are not substituted.

RQ3 uses common eligible request IDs across both operations and all three input families within each state and reduction fraction. The 204 intervals are pointwise, unadjusted for multiple comparisons and conditional on the fixed endpoint. They do not describe independent editing runs or the uncertainty of the 34-state mean.

## Figures

The figure command writes PDF, PNG, SVG, exact-coordinate CSVs and `figure_validation.json`:

- `fig4_pq_plane`: all 40 rewrite endpoints, using the manuscript's symlog axes.
- `fig5_geometry_locality`: all 360 checkpoints, with 2,160 context/metric coordinates.
- The corresponding `_restricted` figures: 34 endpoints and 337 checkpoints on linear axes.
- `rq3_paired_forest_combined`: 204 saved contrasts in a horizontal layout with groups (a)–(d). The graphic omits model-name headings and N/A labels; model/dataset mapping and unavailable cases are explained in its caption in the manuscript.

## Model measurements and maintenance

This workflow reads stored numeric observations. Model execution requires the external inputs described in [experiments/README.md](../experiments/README.md).

After reviewing intentional source changes, refresh the release manifest with `python -m analysis.check_release --write-manifest`. A portable archive can be created with `python -m analysis.package_release --output build/BeyondNormGrowth.zip`.
