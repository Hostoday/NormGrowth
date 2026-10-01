# Analysis

Run commands from the repository root after installing `requirements.txt` in Python 3.9.

```bash
python -m analysis.reproduce --output-dir build/results
python -m analysis.plot_core_figures --output-dir build/figures
```

## Numerical reproduction

`reproduce.py` reads the observations described in [data/README.md](../data/README.md), recalculates the summaries, and compares eight recorded reference tables.

| Analysis | Input and aggregation |
|---|---|
| RQ1 | 40,000 per-edit observations → 40 condition summaries; regression, median ratios, and non-expansion rates |
| RQ2 | Canonical checkpoint observations → shared retained panel, 12 prompt/metric correlations, q–d correlations and norm decomposition |
| RQ3 orthogonal dose | 200 state-dose rows → means across 40 states |
| RQ3 same final norm | 480 condition rows → comparisons across the states eligible at each dose, using common cases across operations and families |

The shared RQ2 exclusion mask removes a checkpoint when either prompt's mean orthogonal displacement, norm ratio, or absolute norm deviation exceeds 3. Both prompts share the checkpoint's fixed-1,000 LOC outcome. `data/analysis_snapshot.json` records available and retained counts and missing trajectories. The q–d table uses exactly the same retained panel as Figure 5. Second moments are means of case-wise squares and retain the norm identity; they are not squared checkpoint means. Checkpoints are repeated states along an editing trajectory, not independent editing replicates.

RQ3 modifies each rewrite, rephrase, or locality input at its own prompt-last position. This differs from the rewrite subject-last measurement in RQ2. The parallel control matches the orthogonal intervention's final norm on geometrically eligible cases; it does not jointly reduce both components. Teacher-forced intervention metrics and free-generation EFF/GEN are distinct outcomes. The bootstrap seed controls case resampling, not repeated editing runs.

## Plotting

`plot_core_figures.py` reads the numeric plot data in `data/` and generates PNG, PDF, and SVG with Matplotlib/STIX typography:

- Figure 4: rewrite endpoints, with a subplot for each model/dataset combination.
- Figure 5: six plotted coordinates per retained checkpoint; prompt context is encoded by color. Additional edit orders and the extra-order legend entry are excluded.

The renderer checks plotted coordinates against the included observations. Generated graphics and temporary outputs are written to the chosen output directory and excluded from Git.

## Model measurements

The saved-data workflow does not load a model or create new predictions. For editing, activation capture, and intervention entry points, see [experiments/README.md](../experiments/README.md). Model-source checks and numerical reaggregation are recorded separately from full GPU experiments.

## Repository checks

```bash
python -m analysis.check_release
python -m analysis.package_release --output build/BeyondNormGrowth.zip
```

The checker validates Python syntax, local document links, fixed filesystem paths, file sizes, and content hashes. Packaging excludes local model inputs, caches, and generated outputs. Run `python -m analysis.check_release --write-manifest` after reviewing intentional changes to tracked files.

## Importing a revised numeric snapshot

Stage a completed canonical paper bundle without replacing the included data:

```bash
python -m analysis.import_snapshot --source-dir PAPER_BUNDLE --output-dir build/candidate
python -m analysis.reproduce --data-dir build/candidate --output-dir build/candidate_results
python -m analysis.plot_core_figures --data-dir build/candidate --output-dir build/candidate_figures
```

The importer preserves numeric CSV cells and records source hashes and selected columns. It normalizes the legacy locality `mean_d` alias to `locality_mean_d`. Install only a reviewed snapshot; the importer deliberately writes outside `data/`.

For a completed 40-trajectory bundle, the installer stages the data, validates the
statistics and figures, preserves existing files in a dated backup, and installs
only after the checks pass:

```bash
python -m analysis.finalize_canonical --prepare-destination-state --work-dir build/canonical_release
python -m analysis.finalize_canonical --source-dir PAPER_BUNDLE --work-dir build/canonical_release --install
```

Preparation seals the hashes of destinations that installation can overwrite.
If any of those files changes before installation, automatic installation is
refused. `--verify-only` runs the checks without installing. The installer
rejects incomplete canonical coverage before touching `data/`. The actual
shared threshold mask determines the retained count. Any installation or release
check failure restores the previous data and manifest. Success is recorded in
`build/canonical_release/install_validation.json`, including artifact hashes.
