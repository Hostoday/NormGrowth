# Beyond Norm Growth

**A Geometric Decomposition of Hidden-State Drift in Sequential Knowledge Editing**

Code for analyzing target/realized-state norm changes, parallel and orthogonal hidden-state displacement, and component-wise interventions in sequential knowledge editing.

The study covers MEMIT and AlphaEdit with Native, NAS, ENCORE, SPHERE, and SADR on Llama-3-8B-Instruct and GPT-2 XL, using zsRE and CounterFact.

## Installation and analysis

Use Python 3.9 and run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m analysis.reproduce --output-dir build/results
python -m analysis.plot_core_figures --output-dir build/figures
```

The analysis recomputes RQ1–RQ3 summaries from the included numeric observations and checks eight reference tables. The plotting command generates the endpoint decomposition and prompt-wise correlation plots (Figures 4 and 5) with Matplotlib. Both commands run on CPU without model weights. The canonical snapshot excludes both additional edit-order trajectories; Figures 4 and 5 and the correlations use the same canonical input collection.

| Directory | Contents |
|---|---|
| [analysis/](analysis/README.md) | Statistical analysis, visualization, and validation |
| [data/](data/README.md) | Numeric observations, reference tables, and source records |
| [experiments/](experiments/README.md) | Editing, geometry capture, intervention code, and model settings |

## Model experiments

See [experiment instructions](experiments/README.md) for the model environment, entry points, and required inputs. Paths are derived from the repository location or configurable through `BNG_*` environment variables; relative environment values are resolved from the repository root.

New model experiments require model/tokenizer access, benchmark inputs, covariance assets, and the corresponding checkpoints and evaluation protocols. These assets are external to this repository. Saved-data reproduction has been validated separately from model-source import and component checks; full GPU experiments and a fresh installation of the model environment were not rerun for this release.

## Analysis scope

- **RQ1:** Correspondence between target and realized-state norm ratios across 40,000 edits and 40 conditions. Their input contexts and normalization references differ.
- **RQ2:** Geometry–locality associations using canonical edit orders and a shared exclusion mask. Snapshot counts and any missing trajectory are recorded in `data/analysis_snapshot.json`. Rewrite uses subject-last; locality uses prompt-last.
- **RQ3:** Separate orthogonal and parallel interventions at each input's prompt-last activation. Orthogonal dose summaries use 40 states; same-final-norm comparisons use the eligible states at each dose, with counts recorded in the snapshot and validation report.

Checkpoints are repeated observations along editing trajectories. RQ3 changes each evaluated input's own activation, and its teacher-forced metrics differ from free-generation EFF/GEN. See [analysis protocol](analysis/README.md) for details.

## Validation

```bash
python -m analysis.check_release
```

This checks source syntax, relative document links, hardcoded filesystem paths, and recorded file hashes. After intentional source edits, refresh the manifest with `python -m analysis.check_release --write-manifest`. The GitHub workflow runs the CPU analysis and release checks.

## License and citation

Project code is provided under the [MIT license](LICENSE). The vendored EasyEdit/AlphaEdit sources retain their licenses and notices; see [source attribution](experiments/NOTICE.md). Citation metadata is available in [CITATION.cff](CITATION.cff).
