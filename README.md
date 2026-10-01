# Beyond Norm Growth

**Base-Relative Geometry of Hidden-State Drift in Sequential Knowledge Editing**

Code for analyzing target/realized-state norm changes, parallel and orthogonal hidden-state displacement, and component-wise interventions in sequential knowledge editing.

The study covers MEMIT and AlphaEdit with Native, NAS, ENCORE, SPHERE, and SADR on Llama-3-8B-Instruct and GPT-2 XL, using zsRE and CounterFact.

## Installation and analysis

Use Python 3.9 and run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m analysis.reproduce --output-dir build/results
python -m analysis.plot_core_figures --output-dir build/figures
```

The analysis reproduces the current manuscript's saved-data results: RQ1 norm and vector comparisons, RQ2 associations on all 360 checkpoints with 337/306-checkpoint sensitivity analyses, and RQ3 paired contrasts and bootstrap intervals. The endpoint table uses complete-target teacher-forced EFF/GEN. The plotting command generates the full-sample Figures 4 and 5, their restricted supplements, and the combined RQ3 forest plot. Both commands run on CPU without model weights. Inputs and version metadata are recorded in `data/analysis_snapshot.json`.

| Directory | Contents |
|---|---|
| [analysis/](analysis/README.md) | Statistical analysis, visualization, and validation |
| [data/](data/README.md) | Numeric observations, reference tables, and source records |
| [experiments/](experiments/README.md) | Editing, geometry capture, intervention code, and model settings |

## Model experiments

See [experiment instructions](experiments/README.md) for the model environment, entry points, and required inputs. Paths are derived from the repository location or configurable through `BNG_*` environment variables; relative environment values are resolved from the repository root.

New model experiments require model/tokenizer access, benchmark inputs, covariance assets, and the corresponding checkpoints and evaluation protocols. These assets are external to this repository. Saved-data reproduction has been validated separately from model-source import and component checks; full GPU experiments and a fresh installation of the model environment were not rerun for this release.

## Analysis scope

- **RQ1:** Target/realized norm ratios across 40,000 edits; vector comparisons on 39,473 finite observations and 39,257 paired displacement cosines.
- **RQ2:** All 360 checkpoints are the primary panel. The 337-state geometry restriction and exclusion of six complete trajectories (306 checkpoints) are sensitivity analyses. Partial-rank controls distinguish norm deviation, edit count, trajectory and total displacement. Correlation intervals use 5,000 whole-trajectory bootstrap draws.
- **RQ3:** Prompt-last interventions at 40 endpoint states; same-final-norm contrasts use 34 eligible states at each reduction fraction. The 204 state-specific intervals use 10,000 paired-case bootstrap draws. Overall means weight states equally and have no aggregate confidence interval.
- **Endpoint performance:** EFF/GEN average token accuracy within each case, then average cases; answers include EOS/EOT. LOC measures agreement with Base predictions under the same gold prefix without an added terminator.

The packaged endpoint scores and vector measurements are saved observations. CPU reproduction recalculates summaries and intervals; it does not rerun model predictions or hidden-state extraction. See the [analysis protocol](analysis/README.md) and [numeric data description](data/README.md).

## Validation

```bash
python -m analysis.check_release
```

This checks source syntax, relative document links, hardcoded filesystem paths, and recorded file hashes. After intentional source edits, refresh the manifest with `python -m analysis.check_release --write-manifest`. The GitHub workflow runs the CPU analysis and release checks.

## License and citation

Project code is provided under the [MIT license](LICENSE). The vendored EasyEdit/AlphaEdit sources retain their licenses and notices; see [source attribution](experiments/NOTICE.md). Citation metadata is available in [CITATION.cff](CITATION.cff).
