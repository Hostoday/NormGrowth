#!/usr/bin/env python3
"""Create the standard 2-row branch-RMS trajectory figure.

Layout
------
Top row:    edit prompts / subject-last
Bottom row: locality prompts / prompt-last
Columns:    residual input, attention write, MLP write, behavioral locality

The locality axis spans both rows.  This script is CPU-only and reads the
already aggregated trajectory CSVs; no model is loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONTEXT_ROWS: Tuple[Tuple[str, str, str], ...] = (
    ("edit", "subject_last", "Edit / subject-last"),
    ("locality", "prompt_last", "Locality / prompt-last"),
)

METRIC_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("input", "Residual-input RMS / Base"),
    ("attention", "Attention-write RMS / Base"),
    ("mlp", "MLP-write RMS / Base"),
)

OUTPUT_NAME = "trajectory_input_attention_mlp_locality.png"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(raw: Any) -> float | None:
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def infer_editor(run_dir: Path) -> str:
    config_path = run_dir / "run_config.json"
    if config_path.is_file():
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            method = config.get("editing_method") or config.get("method")
            if method:
                return str(method)
            experiment = str(config.get("experiment", "")).lower()
            if "alphaedit" in experiment:
                return "AlphaEdit"
            if "memit" in experiment:
                return "MEMIT"
        except (OSError, json.JSONDecodeError):
            pass
    lowered = run_dir.name.lower()
    if lowered.startswith("alphaedit"):
        return "AlphaEdit"
    if lowered.startswith("memit"):
        return "MEMIT"
    return "Editor"


def infer_condition(run_dir: Path, editor: str) -> str:
    name = run_dir.name
    pattern = re.compile(
        rf"^{re.escape(editor.lower())}_[a-z0-9]+_bs\d+_n\d+_(.+?)_seed\d+$",
        flags=re.IGNORECASE,
    )
    match = pattern.match(name)
    condition = match.group(1) if match else ""
    labels = {
        "baseline": "Baseline",
        "sadr": "SADR",
        "encore": "ENCORE",
        "nas": "NAS",
        "rgr": "RGR",
    }
    if condition in labels:
        return labels[condition]
    return condition.replace("_", " ")


def display_name(run_dir: Path, editing_method: str | None) -> str:
    editor = editing_method or infer_editor(run_dir)
    condition = infer_condition(run_dir, editor)
    if not condition or condition == "Baseline":
        return editor if not condition else f"{editor} Baseline"
    return f"{editor} + {condition}"


def required_data_available(rows: Sequence[Mapping[str, str]]) -> bool:
    available = {
        (str(row.get("context")), str(row.get("position")), str(row.get("write_kind")))
        for row in rows
    }
    return all(
        (context, position, kind) in available
        for context, position, _ in CONTEXT_ROWS
        for kind, _ in METRIC_COLUMNS
    )


def common_complete_steps(
    branch_rows: Sequence[Mapping[str, str]],
    outcome_rows: Sequence[Mapping[str, str]],
) -> List[int]:
    """Return checkpoints represented in every panel and in locality output."""

    step_sets: List[set[int]] = []
    for context, position, _ in CONTEXT_ROWS:
        for kind, _ in METRIC_COLUMNS:
            steps = {
                int(row["edit_count"])
                for row in branch_rows
                if row.get("context") == context
                and row.get("position") == position
                and row.get("write_kind") == kind
                and finite(row.get("rms_l2_ratio_to_base")) is not None
            }
            if not steps:
                raise ValueError(
                    "Missing branch series for "
                    f"context={context}, position={position}, kind={kind}"
                )
            step_sets.append(steps)
    locality_steps = {
        int(row["edit_count"])
        for row in outcome_rows
        if finite(row.get("locality_acc")) is not None
    }
    if not locality_steps:
        raise ValueError("No finite locality_acc values")
    steps = sorted(set.intersection(*step_sets, locality_steps))
    if 0 not in steps:
        raise ValueError("The common checkpoint set must include edit 0")
    if len(steps) < 2:
        raise ValueError("At least Base and one edited checkpoint are required")
    return steps


def validate_unique_rows(rows: Sequence[Mapping[str, str]]) -> None:
    seen: set[Tuple[str, str, int, int, str]] = set()
    for row in rows:
        key = (
            str(row.get("context")),
            str(row.get("position")),
            int(row["edit_count"]),
            int(row["layer"]),
            str(row.get("write_kind")),
        )
        if key in seen:
            raise ValueError(f"Duplicate trajectory branch row: {key}")
        seen.add(key)


def maybe_log_scale(axis: Any, values: Sequence[float]) -> bool:
    positive = [value for value in values if value > 0 and math.isfinite(value)]
    if not positive:
        return False
    lower = min(positive)
    upper = max(positive)
    if upper > 100.0 and upper / max(lower, 1e-12) > 100.0:
        axis.set_yscale("log")
        return True
    return False


def plot_run(
    run_dir: Path,
    *,
    editing_method: str | None = None,
    output_name: str = OUTPUT_NAME,
) -> Path | None:
    run_dir = run_dir.expanduser().resolve()
    branch_path = run_dir / "trajectory_branch_summary.csv"
    outcome_path = run_dir / "trajectory_checkpoint_outcomes.csv"
    if not branch_path.is_file() or not outcome_path.is_file():
        print(f"[unified-rms] skipped missing CSVs: {run_dir}")
        return None

    branch_rows = read_csv(branch_path)
    outcome_rows = read_csv(outcome_path)
    if not required_data_available(branch_rows):
        print(
            "[unified-rms] skipped; edit/subject-last and "
            f"locality/prompt-last branch data are required: {run_dir}"
        )
        return None

    validate_unique_rows(branch_rows)
    steps = common_complete_steps(branch_rows, outcome_rows)
    color_map = plt.get_cmap("viridis")
    colors = {
        step: color_map(index / max(len(steps) - 1, 1))
        for index, step in enumerate(steps)
    }

    fig = plt.figure(figsize=(24.0, 9.5))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=(1.0, 1.0, 1.0, 1.10),
        hspace=0.34,
        wspace=0.28,
    )
    component_axes: List[List[Any]] = [[], []]
    for row_index, (context, position, row_label) in enumerate(CONTEXT_ROWS):
        for column_index, (kind, title) in enumerate(METRIC_COLUMNS):
            axis = fig.add_subplot(grid[row_index, column_index])
            component_axes[row_index].append(axis)
            selected = [
                row
                for row in branch_rows
                if row.get("context") == context
                and row.get("position") == position
                and row.get("write_kind") == kind
            ]
            values: List[float] = []
            for step in steps:
                current = sorted(
                    [row for row in selected if int(row["edit_count"]) == step],
                    key=lambda row: int(row["layer"]),
                )
                if not current:
                    continue
                x_values = [int(row["layer"]) for row in current]
                y_values = [
                    value
                    for row in current
                    if (value := finite(row.get("rms_l2_ratio_to_base")))
                    is not None
                ]
                if len(y_values) != len(x_values):
                    raise ValueError(
                        "Non-finite branch RMS ratio at "
                        f"{context}/{position}/{kind}/edit={step}"
                    )
                values.extend(y_values)
                axis.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=1.7,
                    color=colors[step],
                    label=f"edit {step}",
                )
            axis.axhline(1.0, color="#666666", linestyle="--", linewidth=1)
            axis.axvline(8.5, color="#999999", linestyle=":", linewidth=1)
            is_log = maybe_log_scale(axis, values)
            axis.set_title(f"{title}{' [log y]' if is_log else ''}")
            axis.set_xlabel("decoder layer")
            axis.set_ylabel(f"{row_label}\nRMS / Base")
            axis.grid(alpha=0.24, which="both")

    locality_axis = fig.add_subplot(grid[:, 3])
    locality_values: List[Tuple[int, float]] = []
    for row in outcome_rows:
        if int(row["edit_count"]) not in steps:
            continue
        value = finite(row.get("locality_acc"))
        if value is not None:
            locality_values.append((int(row["edit_count"]), value))
    locality_values.sort()
    locality_axis.plot(
        [step for step, _ in locality_values],
        [value for _, value in locality_values],
        color="#D81B60",
        marker="o",
        linewidth=2.2,
    )
    for step, value in locality_values:
        locality_axis.annotate(
            str(step),
            (step, value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    locality_axis.set_title("Locality preservation")
    locality_axis.set_xlabel("cumulative edits")
    locality_axis.set_ylabel("Base-output token agreement")
    locality_axis.grid(alpha=0.24)

    handles, labels = component_axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=min(len(labels), 12),
        frameon=False,
    )
    fig.suptitle(
        f"{display_name(run_dir, editing_method)} — branch-RMS and locality trajectory",
        fontweight="bold",
        y=1.065,
    )
    output_path = run_dir / output_name
    temporary_path = output_path.with_name(
        f".{output_path.name}.tmp-{os.getpid()}"
    )
    try:
        fig.savefig(
            temporary_path,
            dpi=190,
            bbox_inches="tight",
            format="png",
        )
        os.replace(temporary_path, output_path)
    finally:
        plt.close(fig)
        if temporary_path.exists():
            temporary_path.unlink()
    print(f"[unified-rms] {output_path}")
    return output_path


def plot_run_safely(
    run_dir: Path,
    *,
    editing_method: str | None = None,
    output_name: str = OUTPUT_NAME,
) -> Path | None:
    """Generate the derivative plot without interrupting a long edit run."""

    try:
        return plot_run(
            run_dir,
            editing_method=editing_method,
            output_name=output_name,
        )
    except Exception as error:  # plotting must never terminate model editing
        print(f"[unified-rms] warning: {type(error).__name__}: {error}")
        traceback.print_exc()
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--scan-root", type=Path, default=None)
    parser.add_argument("--editing-method", default=None)
    parser.add_argument("--output-name", default=OUTPUT_NAME)
    args = parser.parse_args()

    run_dirs = list(args.run_dir)
    if args.scan_root is not None:
        run_dirs.extend(
            path.parent
            for path in args.scan_root.expanduser().resolve().rglob(
                "trajectory_branch_summary.csv"
            )
        )
    unique_dirs = sorted({path.expanduser().resolve() for path in run_dirs})
    if not unique_dirs:
        raise ValueError("Provide --run-dir and/or --scan-root")

    created = 0
    for run_dir in unique_dirs:
        if plot_run(
            run_dir,
            editing_method=args.editing_method,
            output_name=args.output_name,
        ) is not None:
            created += 1
    print(f"[done] unified branch-RMS figures: {created}/{len(unique_dirs)}")


if __name__ == "__main__":
    main()
