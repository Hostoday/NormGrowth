#!/usr/bin/env python3
"""Compare Jacobian, hidden-state, and attention locality-risk proxies.

The comparison is deliberately paired.  Every proxy is measured for the same
model, prompt, decoder layer, and token position, then joined to the per-case
locality score from ``final_checkpoint_eval.json``.

The projected Jacobians are loaded from ``analyze_residual_spectrum.py``
outputs.  Hidden-state and attention changes are measured against the Base
model with one forward pass per prompt:

* ``hidden_delta_relative_l2`` = ||h_edit - h_base|| / ||h_base||;
* ``attention_symmetric_kl`` = 0.5 * (KL(A_base||A_edit) +
  KL(A_edit||A_base)), averaged across heads;
* ``jacobian_rhat_delta_frobenius`` = ||Rhat_edit - Rhat_base||_F, where
  Rhat = (J-I)/||J-I||_F in the shared Base-PCA coordinates;
* ``jacobian_eig_contraction`` = -log(spread(eig(Rhat_edit)) /
  spread(eig(Rhat_base))).  Positive values mean a contracted eigen cloud.

Checkpoint-level association, within-state prompt-level association, and
paired checkpoint contrasts are reported separately.  This distinction is
important: a proxy can rank damaged checkpoints without predicting which
individual prompt will lose locality.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

try:
    from .analyze_geometry_locality_correlation import (
        bootstrap_correlation,
        load_locality_scores,
        pearson,
        spearman,
    )
    from .analyze_residual_spectrum import (
        EPS,
        StateSpec,
        _state_colors,
        load_model,
        load_probes,
        model_input_device,
        parse_int_list,
        parse_state_spec,
        probe_inputs,
        safe_key,
        write_csv,
    )
except ImportError:
    from analyze_geometry_locality_correlation import (
        bootstrap_correlation,
        load_locality_scores,
        pearson,
        spearman,
    )
    from analyze_residual_spectrum import (
        EPS,
        StateSpec,
        _state_colors,
        load_model,
        load_probes,
        model_input_device,
        parse_int_list,
        parse_state_spec,
        probe_inputs,
        safe_key,
        write_csv,
    )


MAIN_PROXIES = (
    "jacobian_rhat_delta_frobenius",
    "jacobian_eig_contraction",
    "hidden_delta_relative_l2",
    "attention_symmetric_kl",
)

ALL_PROXIES = (
    "jacobian_delta_relative_frobenius",
    "jacobian_rhat_delta_frobenius",
    "jacobian_rhat_cosine_distance",
    "jacobian_eig_contraction",
    "jacobian_eig_centroid_shift",
    "hidden_delta_l2",
    "hidden_delta_relative_l2",
    "hidden_cosine_distance",
    "attention_kl_base_to_state",
    "attention_kl_state_to_base",
    "attention_symmetric_kl",
    "attention_js",
)


def parse_label_path(text: str) -> Tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = (part.strip() for part in text.split("=", 1))
    if not label or not path:
        raise argparse.ArgumentTypeError("expected non-empty LABEL=PATH")
    return label, path


def parse_contrast(text: str) -> Tuple[str, str]:
    if ":" not in text:
        raise argparse.ArgumentTypeError("contrast must be LESS:MORE")
    less, more = (part.strip() for part in text.split(":", 1))
    if not less or not more:
        raise argparse.ArgumentTypeError("contrast must be LESS:MORE")
    return less, more


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= EPS:
        return 0.0 if np.linalg.norm(left - right) <= EPS else 1.0
    value = float(np.dot(left, right) / denominator)
    return float(1.0 - np.clip(value, -1.0, 1.0))


def attention_divergences(base: np.ndarray, state: np.ndarray) -> Dict[str, float]:
    """Return head-averaged divergences for [heads, attended_tokens] arrays."""

    base = np.asarray(base, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    if base.shape != state.shape or base.ndim != 2:
        raise ValueError(f"Attention shapes must match [heads, keys], got {base.shape}, {state.shape}")
    floor = 1e-12
    base = np.maximum(base, floor)
    state = np.maximum(state, floor)
    base = base / base.sum(axis=-1, keepdims=True)
    state = state / state.sum(axis=-1, keepdims=True)
    midpoint = 0.5 * (base + state)
    kl_base_state = np.sum(base * (np.log(base) - np.log(state)), axis=-1)
    kl_state_base = np.sum(state * (np.log(state) - np.log(base)), axis=-1)
    js = 0.5 * (
        np.sum(base * (np.log(base) - np.log(midpoint)), axis=-1)
        + np.sum(state * (np.log(state) - np.log(midpoint)), axis=-1)
    )
    return {
        "attention_kl_base_to_state": float(kl_base_state.mean()),
        "attention_kl_state_to_base": float(kl_state_base.mean()),
        "attention_symmetric_kl": float(0.5 * (kl_base_state + kl_state_base).mean()),
        "attention_js": float(js.mean()),
    }


def eig_spread(matrix: np.ndarray) -> Tuple[float, complex]:
    values = np.linalg.eigvals(np.asarray(matrix, dtype=np.float64))
    spread = float(np.var(values.real) + np.var(values.imag))
    centroid = complex(values.mean())
    return spread, centroid


def jacobian_divergences(base_j: np.ndarray, state_j: np.ndarray) -> Dict[str, float]:
    base_j = np.asarray(base_j, dtype=np.float64)
    state_j = np.asarray(state_j, dtype=np.float64)
    if base_j.shape != state_j.shape or base_j.ndim != 2 or base_j.shape[0] != base_j.shape[1]:
        raise ValueError(f"Projected Jacobian shapes must be equal square matrices, got {base_j.shape}, {state_j.shape}")
    identity = np.eye(base_j.shape[0], dtype=np.float64)
    base_r = base_j - identity
    state_r = state_j - identity
    base_rhat = base_r / (np.linalg.norm(base_r, ord="fro") + EPS)
    state_rhat = state_r / (np.linalg.norm(state_r, ord="fro") + EPS)
    base_spread, base_centroid = eig_spread(base_rhat)
    state_spread, state_centroid = eig_spread(state_rhat)
    return {
        "jacobian_delta_relative_frobenius": float(
            np.linalg.norm(state_j - base_j, ord="fro")
            / (np.linalg.norm(base_j, ord="fro") + EPS)
        ),
        "jacobian_rhat_delta_frobenius": float(
            np.linalg.norm(state_rhat - base_rhat, ord="fro")
        ),
        "jacobian_rhat_cosine_distance": cosine_distance(base_rhat, state_rhat),
        "jacobian_eig_contraction": float(
            -math.log((state_spread + EPS) / (base_spread + EPS))
        ),
        "jacobian_eig_centroid_shift": float(abs(state_centroid - base_centroid)),
    }


@torch.inference_mode()
def collect_base_features(
    model: Any,
    probes: Sequence[Any],
    layers: Sequence[int],
    position: str,
) -> Dict[Tuple[int, int, str], np.ndarray]:
    features: Dict[Tuple[int, int, str], np.ndarray] = {}
    device = model_input_device(model)
    for prompt_index, probe in enumerate(probes):
        outputs = model(
            **probe_inputs(probe, device),
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        if outputs.hidden_states is None or outputs.attentions is None:
            raise RuntimeError("Model did not return hidden states and attentions; use --attn-implementation eager")
        token_pos = int(probe.positions[position])
        for layer in layers:
            if layer >= len(outputs.attentions) or layer >= len(outputs.hidden_states) - 1:
                raise ValueError(f"Layer {layer} is unavailable in model outputs")
            hidden = outputs.hidden_states[layer][0, token_pos].detach().float().cpu().numpy()
            attention = (
                outputs.attentions[layer][0, :, token_pos, : token_pos + 1]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            features[(prompt_index, layer, "hidden")] = hidden
            features[(prompt_index, layer, "attention")] = attention
        del outputs
        print(f"[forward] Base {prompt_index + 1}/{len(probes)}")
    return features


@torch.inference_mode()
def compare_forward_features(
    label: str,
    model: Any,
    probes: Sequence[Any],
    layers: Sequence[int],
    position: str,
    base_features: Mapping[Tuple[int, int, str], np.ndarray],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    device = model_input_device(model)
    for prompt_index, probe in enumerate(probes):
        outputs = model(
            **probe_inputs(probe, device),
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        if outputs.hidden_states is None or outputs.attentions is None:
            raise RuntimeError("Model did not return hidden states and attentions; use --attn-implementation eager")
        token_pos = int(probe.positions[position])
        for layer in layers:
            base_hidden = base_features[(prompt_index, layer, "hidden")]
            state_hidden = outputs.hidden_states[layer][0, token_pos].detach().float().cpu().numpy()
            delta = state_hidden.astype(np.float64) - base_hidden.astype(np.float64)
            base_hidden_rms = float(
                np.sqrt(np.mean(base_hidden.astype(np.float64) ** 2))
            )
            state_hidden_rms = float(
                np.sqrt(np.mean(state_hidden.astype(np.float64) ** 2))
            )
            base_attention = base_features[(prompt_index, layer, "attention")]
            state_attention = (
                outputs.attentions[layer][0, :, token_pos, : token_pos + 1]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            rows.append(
                {
                    "state": label,
                    "position": position,
                    "layer": layer,
                    "prompt_index": prompt_index,
                    "case_id": probe.case_id,
                    "token_position": token_pos,
                    "hidden_delta_l2": float(np.linalg.norm(delta)),
                    "hidden_delta_relative_l2": float(
                        np.linalg.norm(delta) / (np.linalg.norm(base_hidden) + EPS)
                    ),
                    "hidden_cosine_distance": cosine_distance(base_hidden, state_hidden),
                    "base_hidden_rms": base_hidden_rms,
                    "hidden_rms": state_hidden_rms,
                    "hidden_rms_ratio": state_hidden_rms / (base_hidden_rms + EPS),
                    "hidden_energy_ratio": (
                        state_hidden_rms * state_hidden_rms
                        / (base_hidden_rms * base_hidden_rms + EPS)
                    ),
                    **attention_divergences(base_attention, state_attention),
                }
            )
        del outputs
        print(f"[forward] {label} {prompt_index + 1}/{len(probes)}")
    return rows


def attach_jacobian_metrics(
    rows: Sequence[MutableMapping[str, Any]],
    jacobians: Mapping[str, np.ndarray],
    base_label: str,
) -> int:
    attached = 0
    for row in rows:
        state_key = safe_key(
            row["state"], row["position"], f"L{row['layer']}", f"P{row['prompt_index']}"
        )
        base_key = safe_key(
            base_label, row["position"], f"L{row['layer']}", f"P{row['prompt_index']}"
        )
        if state_key not in jacobians or base_key not in jacobians:
            for metric in ALL_PROXIES[:5]:
                row[metric] = float("nan")
            continue
        row.update(jacobian_divergences(jacobians[base_key], jacobians[state_key]))
        attached += 1
    return attached


def aggregate_per_prompt(
    layer_rows: Sequence[Mapping[str, Any]],
    locality_scores: Mapping[str, Mapping[str, float]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in layer_rows:
        grouped[(str(row["state"]), str(row["case_id"]))].append(row)
    output: List[Dict[str, Any]] = []
    for (state, case_id), rows in grouped.items():
        if state not in locality_scores or case_id not in locality_scores[state]:
            continue
        item: Dict[str, Any] = {
            "state": state,
            "case_id": case_id,
            "n_layers": len(rows),
            "locality_acc": locality_scores[state][case_id],
            "locality_loss": 1.0 - locality_scores[state][case_id],
        }
        for metric in ALL_PROXIES:
            item[metric] = finite_mean(float(row.get(metric, float("nan"))) for row in rows)
        output.append(item)
    return output


def correlation_record(
    analysis: str,
    label: str,
    proxy: str,
    x: Sequence[float],
    y: Sequence[float],
    bootstraps: int,
    seed: int,
    less_state: str = "",
    more_state: str = "",
) -> Dict[str, Any]:
    low, high = bootstrap_correlation(x, y, bootstraps, seed)
    return {
        "analysis": analysis,
        "label": label,
        "less_state": less_state,
        "more_state": more_state,
        "proxy": proxy,
        "n_cases": len(x),
        "mean_proxy": finite_mean(x),
        "mean_locality_loss": finite_mean(y),
        "pearson": pearson(x, y),
        "pearson_bootstrap_low": low,
        "pearson_bootstrap_high": high,
        "spearman": spearman(x, y),
    }


def build_correlations(
    prompt_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    contrasts: Sequence[Tuple[str, str]],
    bootstraps: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    correlations: List[Dict[str, Any]] = []
    lookup = {(str(row["state"]), str(row["case_id"])): row for row in prompt_rows}
    for label in labels:
        selected = [row for row in prompt_rows if row["state"] == label]
        for proxy in ALL_PROXIES:
            valid = [row for row in selected if math.isfinite(float(row[proxy]))]
            correlations.append(
                correlation_record(
                    "within_state",
                    label,
                    proxy,
                    [float(row[proxy]) for row in valid],
                    [float(row["locality_loss"]) for row in valid],
                    bootstraps,
                    seed,
                )
            )

    pooled = list(prompt_rows)
    for proxy in ALL_PROXIES:
        valid = [row for row in pooled if math.isfinite(float(row[proxy]))]
        correlations.append(
            correlation_record(
                "pooled",
                "all_states",
                proxy,
                [float(row[proxy]) for row in valid],
                [float(row["locality_loss"]) for row in valid],
                bootstraps,
                seed,
            )
        )

        # Remove checkpoint-level offsets before asking whether the proxy
        # explains which prompts are fragile within a fixed edited state.
        demeaned_x: List[float] = []
        demeaned_y: List[float] = []
        for state in labels:
            state_rows = [row for row in valid if row["state"] == state]
            if not state_rows:
                continue
            state_x = np.asarray([float(row[proxy]) for row in state_rows])
            state_y = np.asarray([float(row["locality_loss"]) for row in state_rows])
            demeaned_x.extend((state_x - state_x.mean()).tolist())
            demeaned_y.extend((state_y - state_y.mean()).tolist())
        correlations.append(
            correlation_record(
                "state_demeaned",
                "all_states",
                proxy,
                demeaned_x,
                demeaned_y,
                bootstraps,
                seed,
            )
        )

    contrast_rows: List[Dict[str, Any]] = []
    for less, more in contrasts:
        cases = sorted(
            {case for state, case in lookup if state == less}
            & {case for state, case in lookup if state == more}
        )
        for case in cases:
            less_row, more_row = lookup[(less, case)], lookup[(more, case)]
            item: Dict[str, Any] = {
                "label": f"{less}:{more}",
                "less_state": less,
                "more_state": more,
                "case_id": case,
                "delta_locality_loss": float(more_row["locality_loss"])
                - float(less_row["locality_loss"]),
            }
            for proxy in ALL_PROXIES:
                item[f"delta_{proxy}"] = float(more_row[proxy]) - float(less_row[proxy])
            contrast_rows.append(item)

        selected = [row for row in contrast_rows if row["less_state"] == less and row["more_state"] == more]
        for proxy in ALL_PROXIES:
            valid = [row for row in selected if math.isfinite(float(row[f"delta_{proxy}"]))]
            correlations.append(
                correlation_record(
                    "contrast",
                    f"{less}:{more}",
                    proxy,
                    [float(row[f"delta_{proxy}"]) for row in valid],
                    [float(row["delta_locality_loss"]) for row in valid],
                    bootstraps,
                    seed,
                    less_state=less,
                    more_state=more,
                )
            )

    checkpoint_rows: List[Dict[str, Any]] = []
    for label in labels:
        selected = [row for row in prompt_rows if row["state"] == label]
        item: Dict[str, Any] = {
            "state": label,
            "n_cases": len(selected),
            "mean_locality_acc": finite_mean(float(row["locality_acc"]) for row in selected),
            "mean_locality_loss": finite_mean(float(row["locality_loss"]) for row in selected),
        }
        for proxy in ALL_PROXIES:
            item[proxy] = finite_mean(float(row[proxy]) for row in selected)
        checkpoint_rows.append(item)

    for proxy in ALL_PROXIES:
        valid = [row for row in checkpoint_rows if math.isfinite(float(row[proxy]))]
        correlations.append(
            correlation_record(
                "checkpoint",
                "edited_states",
                proxy,
                [float(row[proxy]) for row in valid],
                [float(row["mean_locality_loss"]) for row in valid],
                0,
                seed,
            )
        )
    return correlations, contrast_rows, checkpoint_rows


def plot_proxy_scatter(
    output_dir: Path,
    prompt_rows: Sequence[Mapping[str, Any]],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
) -> None:
    colors = _state_colors(labels)
    specs = [
        ("jacobian_rhat_delta_frobenius", "Jacobian Rhat change"),
        ("hidden_delta_relative_l2", "relative delta hidden"),
        ("attention_symmetric_kl", "attention symmetric KL"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 9.0), squeeze=False)
    for column, (proxy, title) in enumerate(specs):
        for label in labels:
            selected = [
                row
                for row in prompt_rows
                if row["state"] == label and math.isfinite(float(row[proxy]))
            ]
            axes[0, column].scatter(
                [float(row[proxy]) for row in selected],
                [float(row["locality_loss"]) for row in selected],
                color=colors[label],
                alpha=0.72,
                s=32,
                edgecolors="none",
                label=label,
            )
        axes[0, column].set_title(f"Per prompt: {title}")
        axes[0, column].set_xlabel(proxy)
        axes[0, column].set_ylabel("locality loss")
        axes[0, column].grid(alpha=0.22)

        for row in checkpoint_rows:
            if not math.isfinite(float(row[proxy])):
                continue
            axes[1, column].scatter(
                float(row[proxy]),
                float(row["mean_locality_loss"]),
                color=colors[str(row["state"])],
                s=58,
            )
            axes[1, column].annotate(
                str(row["state"]),
                (float(row[proxy]), float(row["mean_locality_loss"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axes[1, column].set_title(f"Checkpoint mean: {title}")
        axes[1, column].set_xlabel(proxy)
        axes[1, column].set_ylabel("mean locality loss")
        axes[1, column].grid(alpha=0.22)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Locality-risk proxy comparison", y=0.995, fontweight="bold")
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(labels),
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(output_dir / "locality_proxy_scatter.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_layer_profiles(
    output_dir: Path,
    layer_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
) -> None:
    colors = _state_colors(labels)
    specs = [
        ("jacobian_rhat_delta_frobenius", "Jacobian Rhat change"),
        ("jacobian_eig_contraction", "Jacobian eig-cloud contraction"),
        ("hidden_delta_relative_l2", "relative delta hidden"),
        ("attention_symmetric_kl", "attention symmetric KL"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.3), squeeze=False)
    for axis, (proxy, title) in zip(axes.flat, specs):
        for label in labels:
            selected = [row for row in layer_rows if row["state"] == label]
            layers = sorted({int(row["layer"]) for row in selected})
            values = [
                finite_mean(
                    float(row[proxy])
                    for row in selected
                    if int(row["layer"]) == layer
                )
                for layer in layers
            ]
            axis.plot(layers, values, marker="o", color=colors[label], label=label)
        axis.set_title(title)
        axis.set_xlabel("decoder layer")
        axis.grid(alpha=0.22)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Base-relative proxy profiles", y=0.995, fontweight="bold")
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(labels),
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(output_dir / "locality_proxy_layer_profiles.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=parse_state_spec, action="append", default=[])
    parser.add_argument("--eval", type=parse_label_path, action="append", default=[])
    parser.add_argument("--contrast", type=parse_contrast, action="append", default=[])
    parser.add_argument("--requests-path", required=True)
    parser.add_argument("--jacobian-npz", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", type=parse_int_list, default=parse_int_list("4,5,6,7,8,12,16,20"))
    parser.add_argument("--position", choices=["prompt_last", "subject_last"], default="prompt_last")
    parser.add_argument("--probe-source", choices=["rewrite", "locality"], default="locality")
    parser.add_argument(
        "--probe-selection",
        choices=["prefix", "random"],
        default="prefix",
        help="How to select diagnostic requests before taking --max-prompts.",
    )
    parser.add_argument("--max-prompts", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", choices=["eager"], default="eager")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    if len(args.state) < 2:
        raise ValueError("Provide Base first and at least one edited --state")
    if args.state[0].label != "Base":
        raise ValueError("The first state must be labelled Base")
    labels = [state.label for state in args.state[1:]]
    if len(labels) != len(set(labels)):
        raise ValueError("Edited-state labels must be unique")
    eval_paths = dict(args.eval)
    missing_evals = sorted(set(labels) - set(eval_paths))
    if missing_evals:
        raise ValueError(f"Missing --eval entries for: {missing_evals}")
    if args.probe_source == "locality" and args.position != "prompt_last":
        raise ValueError("Locality probes only support prompt_last")
    if args.max_prompts <= 1:
        raise ValueError("--max-prompts must be greater than one")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.tokenizer_path or args.state[0].path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=args.trust_remote_code
    )
    probes = load_probes(
        args.requests_path,
        tokenizer,
        [args.position],
        args.max_prompts,
        args.max_length,
        probe_source=args.probe_source,
        selection=args.probe_selection,
        seed=args.seed,
    )

    print(f"[model] loading Base: {args.state[0].path}")
    base_model = load_model(
        args.state[0].path,
        args.torch_dtype,
        args.device_map,
        args.trust_remote_code,
        args.attn_implementation,
    )
    base_features = collect_base_features(
        base_model, probes, args.layers, args.position
    )
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    layer_rows: List[Dict[str, Any]] = []
    for state in args.state[1:]:
        print(f"[model] loading {state.label}: {state.path}")
        model = load_model(
            state.path,
            args.torch_dtype,
            args.device_map,
            args.trust_remote_code,
            args.attn_implementation,
        )
        layer_rows.extend(
            compare_forward_features(
                state.label,
                model,
                probes,
                args.layers,
                args.position,
                base_features,
            )
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    jacobian_path = Path(args.jacobian_npz).expanduser().resolve()
    with np.load(jacobian_path) as jacobians:
        attached = attach_jacobian_metrics(layer_rows, jacobians, args.state[0].label)
    if attached == 0:
        raise ValueError(
            "No projected Jacobians matched the requested labels, position, layers, and prompts"
        )
    expected = len(layer_rows)
    if attached < expected:
        print(f"[jacobian] attached {attached}/{expected} rows; missing rows remain NaN")
    else:
        print(f"[jacobian] attached all {attached} rows")

    locality_scores = {
        label: load_locality_scores(Path(eval_paths[label]).expanduser().resolve())
        for label in labels
    }
    prompt_rows = aggregate_per_prompt(layer_rows, locality_scores)
    correlations, contrast_rows, checkpoint_rows = build_correlations(
        prompt_rows,
        labels,
        args.contrast,
        args.bootstrap_samples,
        args.seed,
    )

    write_csv(output_dir / "per_layer_proxy_metrics.csv", layer_rows)
    write_csv(output_dir / "per_prompt_proxy_metrics.csv", prompt_rows)
    write_csv(output_dir / "proxy_locality_correlations.csv", correlations)
    write_csv(output_dir / "proxy_contrasts.csv", contrast_rows)
    write_csv(output_dir / "checkpoint_proxy_summary.csv", checkpoint_rows)
    plot_proxy_scatter(output_dir, prompt_rows, checkpoint_rows, labels)
    plot_layer_profiles(output_dir, layer_rows, labels)

    manifest = {
        "states": [{"label": state.label, "path": state.path} for state in args.state],
        "evals": [{"label": label, "path": path} for label, path in args.eval],
        "contrasts": [list(value) for value in args.contrast],
        "requests_path": str(Path(args.requests_path).expanduser().resolve()),
        "jacobian_npz": str(jacobian_path),
        "tokenizer_path": tokenizer_path,
        "position": args.position,
        "probe_source": args.probe_source,
        "probe_selection": args.probe_selection,
        "layers": args.layers,
        "n_prompts": len(probes),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "definitions": {
            "jacobian_rhat_delta_frobenius": "||Rhat_state - Rhat_base||_F in shared Base-PCA coordinates",
            "jacobian_eig_contraction": "-log(eigenvalue-cloud spread_state / spread_base) for Rhat; positive is contracted",
            "hidden_delta_relative_l2": "||h_state-h_base||_2 / ||h_base||_2 in the full residual stream",
            "attention_symmetric_kl": "head mean of 0.5*(KL(A_base||A_state)+KL(A_state||A_base)) at the selected query token",
        },
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"[done] proxy comparison written to {output_dir}")


if __name__ == "__main__":
    main()
