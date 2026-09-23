#!/usr/bin/env python3
"""Join fixed residual geometry to per-prompt locality preservation scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .analyze_residual_spectrum import _state_colors, parse_int_list, write_csv
except ImportError:
    from analyze_residual_spectrum import _state_colors, parse_int_list, write_csv


def parse_label_path(text: str) -> Tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = text.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected non-empty LABEL=PATH")
    return label.strip(), path.strip()


def parse_contrast(text: str) -> Tuple[str, str]:
    if ":" not in text:
        raise argparse.ArgumentTypeError("contrast must be LESS:MORE")
    less, more = text.split(":", 1)
    if not less.strip() or not more.strip():
        raise argparse.ArgumentTypeError("contrast must contain two labels")
    return less.strip(), more.strip()


def case_key(value: Any) -> str:
    return str(value).strip()


def rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if len(x_values) < 2 or np.std(x_values) <= 0 or np.std(y_values) <= 0:
        return float("nan")
    return float(np.corrcoef(x_values, y_values)[0, 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def bootstrap_correlation(
    x: Sequence[float],
    y: Sequence[float],
    samples: int,
    seed: int,
) -> Tuple[float, float]:
    if samples <= 0 or len(x) < 3:
        return float("nan"), float("nan")
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates: List[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(x_values), size=len(x_values))
        value = pearson(x_values[indices], y_values[indices])
        if math.isfinite(value):
            estimates.append(value)
    if not estimates:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.percentile(estimates, [2.5, 97.5]))


def load_geometry(
    path: Path,
    position: str,
    layers: Sequence[int],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["position"] != position or int(row["layer"]) not in layers:
                continue
            key = (str(row["state"]), case_key(row["case_id"]))
            values = grouped.setdefault(
                key,
                {
                    "paired_displacement_l2": [],
                    "paired_displacement_base_whitened_l2": [],
                    "log_radial_ratio_to_base": [],
                },
            )
            values["paired_displacement_l2"].append(float(row["paired_displacement_l2"]))
            values["paired_displacement_base_whitened_l2"].append(
                float(row["paired_displacement_base_whitened_l2"])
            )
            ratio = max(float(row["radial_ratio_to_base"]), 1e-12)
            values["log_radial_ratio_to_base"].append(math.log(ratio))
    return {
        key: {metric: float(np.mean(values)) for metric, values in metrics.items()}
        for key, metrics in grouped.items()
    }


def load_locality_scores(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("results", [])
    scores: Dict[str, float] = {}
    for row in results:
        if "case_id" in row and "locality_acc" in row:
            scores[case_key(row["case_id"])] = float(row["locality_acc"])
    if not scores:
        raise ValueError(f"No per-case locality_acc entries found in {path}")
    return scores


def correlation_rows(
    joined_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    contrasts: Sequence[Tuple[str, str]],
    bootstrap_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    metrics = [
        "paired_displacement_l2",
        "paired_displacement_base_whitened_l2",
        "log_radial_ratio_to_base",
    ]
    output: List[Dict[str, Any]] = []
    for label in labels:
        selected = [row for row in joined_rows if row["state"] == label]
        for metric in metrics:
            x = [float(row[metric]) for row in selected]
            y = [float(row["locality_loss"]) for row in selected]
            lo, hi = bootstrap_correlation(x, y, bootstrap_samples, seed)
            output.append(
                {
                    "analysis": "state",
                    "label": label,
                    "less_state": "",
                    "more_state": "",
                    "metric": metric,
                    "n_cases": len(selected),
                    "mean_geometry": float(np.mean(x)) if x else float("nan"),
                    "mean_locality_loss": float(np.mean(y)) if y else float("nan"),
                    "pearson": pearson(x, y),
                    "pearson_bootstrap_low": lo,
                    "pearson_bootstrap_high": hi,
                    "spearman": spearman(x, y),
                }
            )

    row_lookup = {(str(row["state"]), case_key(row["case_id"])): row for row in joined_rows}
    for less, more in contrasts:
        case_ids = sorted(
            {case for state, case in row_lookup if state == less}
            & {case for state, case in row_lookup if state == more}
        )
        for metric in metrics:
            delta_geometry = [
                float(row_lookup[(more, case)][metric])
                - float(row_lookup[(less, case)][metric])
                for case in case_ids
            ]
            delta_loss = [
                float(row_lookup[(more, case)]["locality_loss"])
                - float(row_lookup[(less, case)]["locality_loss"])
                for case in case_ids
            ]
            lo, hi = bootstrap_correlation(
                delta_geometry, delta_loss, bootstrap_samples, seed
            )
            output.append(
                {
                    "analysis": "contrast",
                    "label": f"{less}:{more}",
                    "less_state": less,
                    "more_state": more,
                    "metric": metric,
                    "n_cases": len(case_ids),
                    "mean_geometry": (
                        float(np.mean(delta_geometry)) if delta_geometry else float("nan")
                    ),
                    "mean_locality_loss": (
                        float(np.mean(delta_loss)) if delta_loss else float("nan")
                    ),
                    "pearson": pearson(delta_geometry, delta_loss),
                    "pearson_bootstrap_low": lo,
                    "pearson_bootstrap_high": hi,
                    "spearman": spearman(delta_geometry, delta_loss),
                }
            )
    return output


def plot_correlations(
    output_dir: Path,
    joined_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
) -> None:
    colors = _state_colors(labels)
    specs = [
        ("paired_displacement_base_whitened_l2", "Base-whitened paired displacement"),
        ("log_radial_ratio_to_base", "mean log radial ratio (<0: contracted)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for axis, (metric, title) in zip(axes, specs):
        for label in labels:
            selected = [row for row in joined_rows if row["state"] == label]
            axis.scatter(
                [float(row[metric]) for row in selected],
                [float(row["locality_loss"]) for row in selected],
                s=27,
                alpha=0.65,
                color=colors[label],
                label=label,
                edgecolors="none",
            )
        axis.set_xlabel(title)
        axis.set_ylabel("locality loss (1 - token agreement)")
        axis.grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Per-prompt fixed geometry vs locality loss", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "geometry_locality_scatter.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-dir", required=True)
    parser.add_argument("--eval", type=parse_label_path, action="append", default=[])
    parser.add_argument("--contrast", type=parse_contrast, action="append", default=[])
    parser.add_argument("--position", default="prompt_last")
    parser.add_argument("--layers", type=parse_int_list, default=parse_int_list("4,5,6,7,8,12,16,20"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if not args.eval:
        raise ValueError("At least one --eval LABEL=FINAL_CHECKPOINT_EVAL_JSON is required")

    geometry_dir = Path(args.geometry_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else geometry_dir / "locality_correlation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = load_geometry(
        geometry_dir / "fixed_geometry_sample_metrics.csv", args.position, args.layers
    )

    joined: List[Dict[str, Any]] = []
    labels: List[str] = []
    for label, path_text in args.eval:
        labels.append(label)
        scores = load_locality_scores(Path(path_text).expanduser().resolve())
        case_ids = sorted(
            {case for state, case in geometry if state == label} & set(scores)
        )
        if not case_ids:
            raise ValueError(f"No geometry/evaluation cases overlap for state {label}")
        for case in case_ids:
            joined.append(
                {
                    "state": label,
                    "case_id": case,
                    **geometry[(label, case)],
                    "locality_acc": scores[case],
                    "locality_loss": 1.0 - scores[case],
                }
            )

    summaries = correlation_rows(
        joined,
        labels,
        args.contrast,
        args.bootstrap_samples,
        args.seed,
    )
    write_csv(output_dir / "per_prompt_geometry_locality.csv", joined)
    write_csv(output_dir / "geometry_locality_correlation.csv", summaries)
    plot_correlations(output_dir, joined, labels)
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "geometry_dir": str(geometry_dir),
                "evals": [{"label": label, "path": path} for label, path in args.eval],
                "contrasts": [list(value) for value in args.contrast],
                "position": args.position,
                "layers": args.layers,
                "bootstrap_samples": args.bootstrap_samples,
                "seed": args.seed,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[done] geometry/locality correlation written to {output_dir}")


if __name__ == "__main__":
    main()
