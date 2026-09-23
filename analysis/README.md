# Analysis

Run commands from the repository root after installing `requirements.txt` in Python 3.9.

```bash
python -m analysis.reproduce --output-dir build/results
python -m analysis.plot_core_figures --output-dir build/figures
```

## Numerical reproduction

`reproduce.py` reads the observations described in [data/README.md](../data/README.md), recalculates the summaries, and compares six recorded reference tables.

| Analysis | Input and aggregation |
|---|---|
| RQ1 | 40,000 per-edit observations → 40 condition summaries; regression, median ratios, and non-expansion rates |
| RQ2 | 360 checkpoint observations → 337 retained checkpoints and 12 prompt/metric correlations |
| RQ3 orthogonal dose | 200 state-dose rows → means across 40 states |
| RQ3 same final norm | 480 condition rows → comparisons across 34 eligible states, using common eligible cases at each dose |

The shared RQ2 exclusion mask removes a checkpoint when either prompt's mean orthogonal displacement, norm ratio, or absolute norm deviation exceeds 3. Both prompts share the checkpoint's fixed-1,000 LOC outcome. Checkpoints are repeated states along an editing trajectory, not independent editing replicates.

RQ3 modifies each rewrite, rephrase, or locality input at its own prompt-last position. This differs from the rewrite subject-last measurement in RQ2. The parallel control matches the orthogonal intervention's final norm on geometrically eligible cases; it does not jointly reduce both components. Teacher-forced intervention metrics and free-generation EFF/GEN are distinct outcomes. The bootstrap seed controls case resampling, not repeated editing runs.

## Plotting

`plot_core_figures.py` reads the numeric plot data in `data/` and generates PNG, PDF, and SVG with Matplotlib/STIX typography:

- Figure 4: 34 rewrite endpoints, with a subplot for each model/dataset combination.
- Figure 5: 2,022 plotted coordinates from the 337 retained checkpoints; prompt context is encoded by color.

The renderer checks plotted coordinates against the included observations. Generated graphics and temporary outputs are written to the chosen output directory and excluded from Git.

## Model measurements

The saved-data workflow does not load a model or create new predictions. For editing, activation capture, and intervention entry points, see [experiments/README.md](../experiments/README.md). Model-source checks and numerical reaggregation are recorded separately from full GPU experiments.

## Repository checks

```bash
python -m analysis.check_release
python -m analysis.package_release --output build/BeyondNormGrowth.zip
```

The checker validates Python syntax, local document links, fixed filesystem paths, file sizes, and content hashes. Packaging excludes local model inputs, caches, and generated outputs. Run `python -m analysis.check_release --write-manifest` after reviewing intentional changes to tracked files.
