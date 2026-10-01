"""Render the latest horizontal RQ3 forest plot from saved paired contrasts.

Run: python -m analysis.plot_rq3_forest --data-dir data --output-dir build/figures
The 204 estimates and pointwise intervals are used without recomputation.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data"

PANELS = [("Llama", "zsRE"), ("Llama", "CounterFact"),
          ("GPT-2 XL", "zsRE"), ("GPT-2 XL", "CounterFact")]
PANEL_LABELS = ["(a)", "(c)", "(b)", "(d)"]
METRICS = [("locality", "LOC", (-14, 7), [-10, -5, 0, 5]),
           ("rewrite", r"EFF$_{\mathrm{TF}}$", (-16, 11), [-10, 0, 10]),
           ("rephrase", r"GEN$_{\mathrm{TF}}$", (-11, 10), [-10, -5, 0, 5, 10])]
DOSES = [(0.25, "#0072B2", "o", -0.19), (0.5, "#D55E00", "s", 0.19)]
CONDITIONS = [(editor, method) for editor in ["MEMIT", "AlphaEdit"]
              for method in ["Native", "NAS", "ENCORE", "SPHERE", "SADR"]]
POSITIONS = [float(i) + (0.65 if i >= 5 else 0) for i in range(10)]


def _render(data_dir, output_dir):
    """Draw all saved contrasts, returning a machine-readable validation record."""
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir == data_dir or data_dir in output_dir.parents:
        raise ValueError("Output must not overwrite the source data directory")
    base = data_dir / "manuscript" if (data_dir / "manuscript").is_dir() else data_dir
    SOURCE = base / "same_norm_paired_contrasts.csv"
    with SOURCE.open(newline='') as handle:
        ROWS = list(csv.DictReader(handle))
    lookup = {}
    for row in ROWS:
        dose = float(row["dose_fraction"])
        assert row["scope"] == "B_common_all_families"
        assert row["order_id"] == "canonical" and int(row["edit_count"]) == 1000
        assert row["lhs"] == f"projection_match_f{int(dose * 100):03d}"
        assert row["rhs"] == f"perp_f{int(dose * 100):03d}"
        key = (row["model"], row["dataset"], row["editor"], row["method"], dose, row["family"])
        assert key not in lookup
        lookup[key] = row
    assert len(lookup) == 204

    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 11.5,
        "axes.titlesize": 13, "xtick.labelsize": 10.5, "ytick.labelsize": 11.5,
        "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(15.6, 6.75))
    axes = []
    for left, right in [(0.144, 0.487), (0.651, 0.994)]:
        grid = fig.add_gridspec(2, 3, left=left, right=right, bottom=0.18,
                               top=0.89, wspace=0.14, hspace=0.40)
        axes.extend([[fig.add_subplot(grid[ri, ci]) for ci in range(3)] for ri in range(2)])
    handles = [Line2D([], [], color=color, marker=marker, lw=1.1, ms=4.4,
                      label=rf"$\gamma={int(dose * 100)}\%$")
               for dose, color, marker, _ in DOSES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.562, 0.998),
               ncol=2, frameon=False, handlelength=2, columnspacing=2, fontsize=11.5)

    plotted = set()
    panel_info = []
    for ri, (model, dataset) in enumerate(PANELS):
        for ci, (family, metric, limits, ticks) in enumerate(METRICS):
            ax = axes[ri][ci]
            ax.set_xlim(*limits)
            ax.set_xticks(ticks)
            ax.set_ylim(10.35, -0.7)
            ax.set_yticks(POSITIONS)
            labels = [f"{editor} (Native)" if method == "Native" else method
                      for editor, method in CONDITIONS]
            ax.set_yticklabels(labels if ci == 0 else [])
            ax.tick_params(axis="y", length=0, pad=8)
            ax.tick_params(axis="x", length=3, pad=3)
            for label, (_, method) in zip(ax.get_yticklabels(), CONDITIONS):
                if method == "Native":
                    label.set_fontweight("bold")
            ax.axhspan(-0.55, 4.5, facecolor="#F5F6F7", edgecolor="none", zorder=0)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#E0E3E5", linewidth=0.5)
            ax.axvline(0, color="#444444", linewidth=0.9, zorder=1)
            ax.axhline(5.0, color="#C2C5C8", linewidth=0.55)
            if ri in (0, 2):
                ax.set_title(metric, pad=9)
            for (editor, method), y in zip(CONDITIONS, POSITIONS):
                available = 0
                for dose, color, marker, offset in DOSES:
                    key = (model, dataset, editor, method, dose, family)
                    if key not in lookup:
                        continue
                    row = lookup[key]
                    mean, lo, hi = [float(row[k]) for k in
                                    ["mean_difference_pp", "ci_low_pp", "ci_high_pp"]]
                    assert limits[0] < lo <= mean <= hi < limits[1]
                    artist = ax.errorbar(mean, y + offset, xerr=[[mean - lo], [hi - mean]],
                                fmt=marker, markersize=4.8, color=color, ecolor=color,
                                elinewidth=1.05, capsize=1.9, capthick=0.75,
                                markeredgewidth=0.5, zorder=3)
                    assert float(artist.lines[0].get_xdata()[0]) == mean
                    assert float(artist.lines[0].get_ydata()[0]) == y + offset
                    assert key not in plotted
                    plotted.add(key)
                    available += 1
                if not available:
                    assert model == "Llama" and editor == "MEMIT" and method in ["Native", "SPHERE", "SADR"]
                else:
                    assert available == 2
            panel_info.append({"panel": PANEL_LABELS[ri], "model": model, "dataset": dataset, "metric": family,
                               "xlim": list(limits), "xticks": ticks})
        top = axes[ri][0].get_position().y1
        heading = PANEL_LABELS[ri] + ("  " + dataset if ri < 2 else "")
        fig.text(0.052 + (0.507 if ri >= 2 else 0), top + 0.014,
                 heading, fontsize=11.5, fontweight="bold", va="bottom")

    for left, right in [(0.144, 0.487), (0.651, 0.994)]:
        fig.text((left + right) / 2, 0.113, "Parallel-only − Orthogonal (pp)", ha="center", fontsize=11.5)
        fig.text(left, 0.064, "← Favors orthogonal", ha="left", fontsize=10.5)
        fig.text(right, 0.064, "Favors parallel-only →", ha="right", fontsize=10.5)
    assert plotted == set(lookup)
    assert fig._suptitle is None
    assert not any("Llama" in t.get_text() or "GPT" in t.get_text() or "N/A" in t.get_text()
                   for t in fig.findobj(matplotlib.text.Text))
    for ci in range(3):
        assert len({tuple(axes[ri][ci].get_xlim()) for ri in range(4)}) == 1
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text_artist in fig.findobj(matplotlib.text.Text):
        if not text_artist.get_visible() or not text_artist.get_text():
            continue
        bbox = text_artist.get_window_extent(renderer)
        if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > fig.bbox.width + 1 or bbox.y1 > fig.bbox.height + 1:
            outside.append(text_artist.get_text())
    assert not outside, outside
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for ext in ["pdf", "png", "svg"]:
        path = output / f"rq3_paired_forest_combined.{ext}"
        fig.savefig(path, dpi=300)
        files.append({"path": path.name,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    plt.close(fig)
    report = {"source": str(SOURCE.relative_to(data_dir.resolve())),
              "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "n_saved_contrasts": len(lookup), "n_plotted_contrasts": len(plotted),
              "layout": "2 rows by 6 columns; Llama data on the left, GPT-2 XL data on the right; no model names anywhere in the graphic",
              "model_names_in_graphic": False,
              "missing_cases_display": "Blank; no N/A labels or footnote",
              "dataset_names": "Shared row labels on the left only",
              "all_intervals_within_axes": True, "all_text_within_canvas": True,
              "matched_x_ranges_per_metric": True, "statistics_recomputed": False,
              "panels": panel_info, "files": files}
    (output / "rq3_forest_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def render(data_dir, output_dir):
    """Keep the forest style identical in standalone and combined commands."""
    with plt.rc_context():
        plt.rcdefaults()
        return _render(data_dir, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("build/figures"))
    args = parser.parse_args()
    report = render(args.data_dir, args.output_dir)
    print(json.dumps({"n_plotted_contrasts": report["n_plotted_contrasts"],
                      "files": report["files"]}, indent=2))


if __name__ == "__main__":
    main()
