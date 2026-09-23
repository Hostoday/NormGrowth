#!/usr/bin/env python3
"""Aggregate per-edit latent/inner/outer records and per-case outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean(values: Iterable[Any]) -> float:
    finite = [float(value) for value in values if _finite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def _sum(values: Iterable[Any]) -> float:
    """Return the sum of finite optional diagnostics, or NaN when absent.

    The post-projection SPHERE fields do not exist in older artifacts.  Using
    ``sum(..., 0)`` for those optional fields would make an unavailable
    measurement look like a true zero, so keep absence explicit instead.
    """

    finite = [float(value) for value in values if _finite(value)]
    return sum(finite) if finite else float("nan")


def _flatten_scalars(
    value: Mapping[str, Any], prefix: str = ""
) -> Dict[str, Any]:
    """Flatten nested scalar diagnostics into stable CSV column names."""

    output: Dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(item, Mapping):
            output.update(_flatten_scalars(item, name))
        elif isinstance(item, (str, int, float, bool)) or item is None:
            output[name] = item
    return output


def aggregate_mechanisms(artifact_dir: Path) -> List[Dict[str, Any]]:
    records: Dict[int, Dict[str, Any]] = {}
    for path in sorted((artifact_dir / "latents").glob("edit_*.json")):
        payload = _read_json(path)
        index = int(payload["edit_index"])
        records.setdefault(index, {}).update(payload)

    for path in sorted((artifact_dir / "inner_gain").glob("edit_*.jsonl")):
        trajectory = _read_jsonl(path)
        if not trajectory:
            continue
        first = trajectory[0]
        last = trajectory[-1]
        index = int(first["edit_index"])
        row = records.setdefault(index, {})
        row.update(
            {
                "edit_index": index,
                "case_id": first.get("case_id"),
                "inner_steps_recorded": len(trajectory),
                "inner_g0": first.get(
                    "residual_gain_current_gain_mean", float("nan")
                ),
                "inner_g_final": last.get(
                    "residual_gain_current_gain_mean", float("nan")
                ),
                "inner_gain_change": (
                    float(
                        last.get(
                            "residual_gain_current_gain_mean", float("nan")
                        )
                    )
                    - float(
                        first.get(
                            "residual_gain_current_gain_mean", float("nan")
                        )
                    )
                ),
                "inner_final_excess_gain": last.get(
                    "residual_gain_excess_gain_mean", float("nan")
                ),
                "inner_final_write_relative_energy": last.get(
                    "residual_gain_current_write_relative_energy_mean",
                    float("nan"),
                ),
                "inner_final_alignment_contribution": last.get(
                    "residual_gain_current_alignment_contribution_mean",
                    float("nan"),
                ),
                "inner_final_input_write_cosine": last.get(
                    "residual_gain_current_input_write_cosine_mean",
                    float("nan"),
                ),
                "inner_rho0": first.get(
                    "residual_gain_current_rho_mean", float("nan")
                ),
                "inner_rho_final": last.get(
                    "residual_gain_current_rho_mean", float("nan")
                ),
                "inner_rho_minus_one_final": last.get(
                    "residual_gain_current_rho_minus_one_mean", float("nan")
                ),
                "inner_excess_rho_minus_one_final": last.get(
                    "residual_gain_excess_rho_minus_one_mean", float("nan")
                ),
                "inner_rho_positive_l1_loss_final": last.get(
                    "residual_gain_rho_positive_l1_loss", float("nan")
                ),
                "inner_log_rho0": first.get(
                    "residual_gain_current_log_rho_mean", float("nan")
                ),
                "inner_log_rho_final": last.get(
                    "residual_gain_current_log_rho_mean", float("nan")
                ),
                "inner_excess_log_rho_final": last.get(
                    "residual_gain_excess_log_rho_mean", float("nan")
                ),
                "inner_log_rho_positive_l1_loss_final": last.get(
                    "residual_gain_log_rho_positive_l1_loss", float("nan")
                ),
                "inner_objective": last.get(
                    "residual_gain_objective", "gain"
                ),
                "inner_alignment_weight": last.get(
                    "residual_gain_alignment_weight", 1.0
                ),
                "inner_objective0": first.get(
                    "residual_gain_current_objective_mean", float("nan")
                ),
                "inner_objective_final": last.get(
                    "residual_gain_current_objective_mean", float("nan")
                ),
                "inner_objective_change": (
                    float(
                        last.get(
                            "residual_gain_current_objective_mean", float("nan")
                        )
                    )
                    - float(
                        first.get(
                            "residual_gain_current_objective_mean", float("nan")
                        )
                    )
                ),
                "inner_final_excess_objective": last.get(
                    "residual_gain_excess_objective_mean", float("nan")
                ),
                "inner_final_violation_fraction": last.get(
                    "residual_gain_violation_fraction", float("nan")
                ),
                "inner_final_violation_mean": last.get(
                    "residual_gain_violation_mean", float("nan")
                ),
                "inner_final_violation_max": last.get(
                    "residual_gain_violation_max", float("nan")
                ),
                "inner_final_reference_input_l2": last.get(
                    "residual_gain_reference_input_l2_mean", float("nan")
                ),
                "inner_final_current_input_l2": last.get(
                    "residual_gain_current_input_l2_mean", float("nan")
                ),
                "inner_final_reference_output_l2": last.get(
                    "residual_gain_reference_output_l2_mean", float("nan")
                ),
                "inner_final_current_output_l2": last.get(
                    "residual_gain_current_output_l2_mean", float("nan")
                ),
                "inner_final_input_state_delta_l2": last.get(
                    "residual_gain_input_state_delta_l2_mean", float("nan")
                ),
                "inner_final_output_state_delta_l2": last.get(
                    "residual_gain_output_state_delta_l2_mean", float("nan")
                ),
                "inner_final_output_to_reference_norm_ratio": last.get(
                    "residual_gain_output_to_reference_norm_ratio_mean",
                    float("nan"),
                ),
                "inner_final_output_norm_positive_relative_excess": last.get(
                    "residual_gain_output_norm_positive_relative_excess_mean",
                    float("nan"),
                ),
                "inner_final_output_norm_positive_l2_excess": last.get(
                    "residual_gain_output_norm_positive_l2_excess_mean",
                    float("nan"),
                ),
                "inner_final_rgr_loss": last.get(
                    "residual_gain_loss", float("nan")
                ),
                "inner_final_weighted_rgr_loss": last.get(
                    "weighted_residual_gain_loss", float("nan")
                ),
                "inner_cosine_aux_lambda": last.get(
                    "residual_gain_cosine_aux_lambda", 0.0
                ),
                "inner_cosine_aux_sharpness": last.get(
                    "residual_gain_cosine_aux_sharpness", 1.0
                ),
                "inner_final_weighted_rgr_base_loss": last.get(
                    "weighted_residual_gain_base_loss", float("nan")
                ),
                "inner_final_cosine_aux_loss": last.get(
                    "residual_gain_cosine_aux_loss", float("nan")
                ),
                "inner_final_weighted_cosine_aux_loss": last.get(
                    "weighted_residual_gain_cosine_aux_loss", float("nan")
                ),
                "inner_final_cosine_aux_slope": last.get(
                    "residual_gain_cosine_aux_slope_mean", float("nan")
                ),
                "inner_final_cosine_below_minus_0_9_fraction": last.get(
                    "residual_gain_cosine_aux_fraction_below_minus_0_9",
                    float("nan"),
                ),
                "inner_final_cosine_below_minus_0_99_fraction": last.get(
                    "residual_gain_cosine_aux_fraction_below_minus_0_99",
                    float("nan"),
                ),
                "inner_final_write_to_input_norm_ratio": last.get(
                    "residual_gain_current_write_to_input_norm_ratio_mean",
                    float("nan"),
                ),
                "inner_final_efficacy": last.get("efficacy_score", float("nan")),
                "inner_final_base_edit_loss": last.get(
                    "base_edit_loss", float("nan")
                ),
                "inner_trajectory_path": str(path),
            }
        )

    for path in sorted((artifact_dir / "outer").glob("edit_*.jsonl")):
        layers = _read_jsonl(path)
        if not layers:
            continue
        index = int(layers[0]["edit_index"])
        row = records.setdefault(index, {})
        last_layer = max(layers, key=lambda item: int(item["layer"]))
        row.update(
            {
                "edit_index": index,
                "case_id": layers[0].get("case_id"),
                "outer_layers_recorded": len(layers),
                "outer_mean_key_l2": _mean(item.get("key_l2") for item in layers),
                "outer_mean_target_error_l2": _mean(
                    item.get("target_error_l2") for item in layers
                ),
                "outer_mean_delta_w_k_l2": _mean(
                    item.get("delta_w_k_l2") for item in layers
                ),
                "outer_sum_delta_w_frobenius": sum(
                    float(item.get("delta_w_frobenius", 0.0))
                    for item in layers
                ),
                "outer_mean_target_realization_cosine": _mean(
                    item.get("target_error_delta_w_k_cosine")
                    for item in layers
                ),
                # ``delta_w_k`` above is the native closed-form realization.
                # For post-processed methods such as SPHERE, the committed
                # fields below describe the update that is actually written
                # to the model and the change introduced by post-processing.
                # They remain NaN for legacy/non-SPHERE artifacts.
                "outer_mean_committed_delta_w_k_l2": _mean(
                    item.get("committed_delta_w_k_l2") for item in layers
                ),
                "outer_mean_postprocess_delta_w_k_l2": _mean(
                    item.get("postprocess_delta_w_k_l2") for item in layers
                ),
                "outer_mean_committed_delta_w_frobenius": _mean(
                    item.get("committed_delta_w_frobenius") for item in layers
                ),
                "outer_sum_committed_delta_w_frobenius": _sum(
                    item.get("committed_delta_w_frobenius") for item in layers
                ),
                "outer_mean_committed_target_error_delta_w_k_cosine": _mean(
                    item.get("committed_target_error_delta_w_k_cosine")
                    for item in layers
                ),
                "outer_last_layer": int(last_layer["layer"]),
                "outer_last_delta_w_k_l2": last_layer.get(
                    "delta_w_k_l2", float("nan")
                ),
                "outer_last_target_realization_cosine": last_layer.get(
                    "target_error_delta_w_k_cosine", float("nan")
                ),
                "outer_last_committed_delta_w_k_l2": last_layer.get(
                    "committed_delta_w_k_l2", float("nan")
                ),
                "outer_last_postprocess_delta_w_k_l2": last_layer.get(
                    "postprocess_delta_w_k_l2", float("nan")
                ),
                "outer_last_committed_delta_w_frobenius": last_layer.get(
                    "committed_delta_w_frobenius", float("nan")
                ),
                "outer_last_committed_target_error_delta_w_k_cosine": (
                    last_layer.get(
                        "committed_target_error_delta_w_k_cosine",
                        float("nan"),
                    )
                ),
                "outer_trajectory_path": str(path),
            }
        )

    post_update_rows: List[Dict[str, Any]] = []
    for path in sorted((artifact_dir / "post_update").glob("edit_*.json")):
        payload = _read_json(path)
        index = int(payload["edit_index"])
        flattened = {
            "edit_index": index,
            "case_id": payload.get("case_id"),
            "post_update_measurement": payload.get("measurement"),
            "post_update_artifact_path": str(path),
            **{
                f"post_update.{key}": value
                for key, value in _flatten_scalars(
                    {
                        "vector_metrics": payload.get("vector_metrics", {}),
                        "bridge_metrics": payload.get("bridge_metrics", {}),
                        "behavior": payload.get("behavior", {}),
                    }
                ).items()
            },
        }
        post_update_rows.append(flattened)
        records.setdefault(index, {}).update(flattened)
    _write_csv(artifact_dir / "post_update_summary.csv", post_update_rows)

    rows = [records[index] for index in sorted(records)]
    _write_csv(artifact_dir / "edit_mechanism_summary.csv", rows)
    return rows


def aggregate_case_outcomes(output_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(output_dir.glob("step_*/evaluation.json")):
        payload = _read_json(path)
        step = int(payload["edit_count"])
        by_case: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"edit_count": step}
        )
        for metric, value_name in (
            ("rewrite", "rewrite_acc"),
            ("rephrase", "rephrase_acc"),
            ("locality", "locality_acc"),
        ):
            for item in payload.get(metric, []):
                case_id = str(item.get("case_id"))
                by_case[case_id]["case_id"] = item.get("case_id")
                # Preserve every scalar field, including continuous NLL,
                # probability, and target-margin additions.  Existing
                # accuracy column names remain unchanged.
                for key, value in item.items():
                    if key != "case_id" and (
                        isinstance(value, (str, int, float, bool)) or value is None
                    ):
                        by_case[case_id][key] = value
                by_case[case_id].setdefault(
                    value_name, item.get(value_name, float("nan"))
                )
                if metric == "locality":
                    by_case[case_id]["n_locality_pairs"] = item.get(
                        "n_locality_pairs", 0
                    )
        rows.extend(by_case.values())
    rows.sort(key=lambda item: (int(item["edit_count"]), str(item["case_id"])))
    _write_csv(output_dir / "trajectory_per_case_outcomes.csv", rows)
    return rows


def join_mechanisms_and_outcomes(
    output_dir: Path,
    mechanism_rows: Sequence[Mapping[str, Any]],
    outcome_rows: Sequence[Mapping[str, Any]],
) -> None:
    mechanism_by_case = {
        str(row.get("case_id")): row
        for row in mechanism_rows
        if row.get("case_id") is not None
    }
    joined: List[Dict[str, Any]] = []
    for outcome in outcome_rows:
        mechanism = mechanism_by_case.get(str(outcome.get("case_id")))
        if mechanism is None:
            continue
        edit_index = int(mechanism["edit_index"])
        checkpoint = int(outcome["edit_count"])
        if checkpoint < edit_index:
            continue
        joined.append(
            {
                **dict(mechanism),
                **dict(outcome),
                "edit_index": edit_index,
                "evaluation_checkpoint": checkpoint,
                "evaluation_lag": checkpoint - edit_index,
            }
        )
    _write_csv(output_dir / "mechanism_outcome_checkpoint_join.csv", joined)


def aggregate_run(output_dir: str | Path) -> None:
    root = Path(output_dir).expanduser().resolve()
    artifact_dir = root / "edit_artifacts"
    mechanism_rows = (
        aggregate_mechanisms(artifact_dir) if artifact_dir.exists() else []
    )
    outcome_rows = aggregate_case_outcomes(root)
    if mechanism_rows and outcome_rows:
        join_mechanisms_and_outcomes(root, mechanism_rows, outcome_rows)
    post_update_count = len(
        list((artifact_dir / "post_update").glob("edit_*.json"))
    )
    aggregate_files = [
        "edit_artifacts/edit_mechanism_summary.csv",
        "trajectory_per_case_outcomes.csv",
        "mechanism_outcome_checkpoint_join.csv",
    ]
    if post_update_count:
        aggregate_files.append("edit_artifacts/post_update_summary.csv")
    manifest = {
        "artifact_root": str(artifact_dir),
        "mechanism_records": len(mechanism_rows),
        "per_case_outcome_records": len(outcome_rows),
        "post_update_records": post_update_count,
        "inner_gain_definition": (
            "g_l=(mean(H_{l+1}^2)-mean(H_l^2))/"
            "(mean(H_l^2)+eps)"
        ),
        "latent_arrays": [
            "target_init",
            "optimizer_delta",
            "delta_star=v_star-target_init",
            "v_star",
        ],
        "outer_arrays_per_layer": [
            "k_star",
            "target_error",
            "distributed_residual",
            "delta_w_k",
            "committed_delta_w_k",
            "postprocess_delta_w_k",
        ],
        "outer_optional_array_definition": (
            "committed_delta_w_k is the realization of the actual "
            "post-processed update; postprocess_delta_w_k equals "
            "committed_delta_w_k-delta_w_k. Both are optional and are "
            "currently emitted for SPHERE."
        ),
        "checkpoint_delta_k_definition": (
            "delta_k=current down-projection input key-Base key on the same "
            "fixed prompt/token probe"
        ),
        "post_update_definition": (
            "paired actual model re-forward of the current request immediately "
            "before and after apply_algo; stored vectors are pre/post float16 "
            "and Delta is reconstructed as post-pre"
        ),
        "aggregate_files": aggregate_files,
    }
    with (root / "edit_artifact_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    aggregate_run(args.output_dir)


if __name__ == "__main__":
    main()
