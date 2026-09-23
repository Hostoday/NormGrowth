#!/usr/bin/env python3
"""Disambiguate residual under-writing, constantization, and gain rotation.

For a decoder block ``Y = H + F(H)`` and selected output token ``t``, this
diagnostic records:

* the magnitude and across-prompt covariance of total, attention, and MLP
  writes;
* the same common/variable decomposition for paired edit-induced writes
  ``Delta F = F_state - F_Base``;
* a Hutchinson estimate of the full-sequence Frobenius gain
  ``||d F_t / d H_{<=t}||_F``, split into self-token and context terms;
* a Base-PCA estimate ``||U^T (d F_t / d h_t) U||_F`` to distinguish true
  attenuation from rotation outside the Base subspace; and
* raw, RMS-adjusted, and paired Base-norm-matched gains.

The first ``--state`` is the Base reference.  Models are loaded one at a time
and no checkpoint is written.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.analyze_residual_spectrum import (
    EPS,
    LayerContext,
    Probe,
    StateSpec,
    _detach_tree,
    _layer_hidden_output,
    _state_colors,
    capture_layer_contexts,
    fit_shared_bases,
    load_external_bases,
    load_model,
    load_probes,
    model_input_device,
    model_layers,
    parse_int_list,
    parse_state_spec,
    parse_str_list,
    probe_inputs,
    resolve_model_load_spec,
    safe_key,
    spectrum_metrics,
    write_csv,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_shared_reference(
    output_dir: Path,
    positions: Sequence[str],
    layers: Sequence[int],
    n_prompts: int,
) -> Tuple[Dict[str, Dict[int, torch.Tensor]], Dict[Tuple[int, int], torch.Tensor]]:
    basis_path = output_dir / "shared_pca_bases.npz"
    rms_path = output_dir / "base_sequence_rms.npz"
    bases: Dict[str, Dict[int, torch.Tensor]] = {position: {} for position in positions}
    with np.load(basis_path) as loaded:
        for position in positions:
            for layer in layers:
                key = safe_key(position, f"L{layer}")
                if key not in loaded:
                    raise KeyError(f"Missing shared basis {key!r} in {basis_path}")
                bases[position][layer] = torch.from_numpy(
                    np.asarray(loaded[key], dtype=np.float32).copy()
                )
    rms: Dict[Tuple[int, int], torch.Tensor] = {}
    with np.load(rms_path) as loaded:
        for prompt_index in range(n_prompts):
            for layer in layers:
                key = safe_key(f"P{prompt_index}", f"L{layer}")
                if key not in loaded:
                    raise KeyError(f"Missing Base RMS {key!r} in {rms_path}")
                rms[(prompt_index, layer)] = torch.from_numpy(
                    np.asarray(loaded[key], dtype=np.float32).copy()
                )
    return bases, rms


def state_is_complete(
    state_dir: Path,
    run_fingerprint: str,
    *,
    require_delta: bool,
    require_hutchinson: bool = True,
) -> bool:
    completion = state_dir / "complete.json"
    required = [
        "write_prompt_metrics.csv",
        "write_summary_metrics.csv",
    ]
    if require_hutchinson:
        required.extend(
            ["hutchinson_prompt_metrics.csv", "hutchinson_summary_metrics.csv"]
        )
    if require_delta:
        required.append("delta_write_summary_metrics.csv")
    if not completion.exists() or not all((state_dir / name).exists() for name in required):
        return False
    try:
        return json.loads(completion.read_text(encoding="utf-8")).get("fingerprint") == run_fingerprint
    except (OSError, json.JSONDecodeError):
        return False


def call_layer(
    layer: torch.nn.Module,
    context: LayerContext,
    hidden: torch.Tensor,
    *,
    capture_components: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Re-evaluate one captured decoder block with an optional hidden input."""

    args, kwargs = context
    patched_args = (hidden,) + tuple(_detach_tree(args[1:]))
    patched_kwargs = _detach_tree(kwargs)
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    def save(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            captured[name] = _layer_hidden_output(output)

        return hook

    if capture_components:
        if not hasattr(layer, "self_attn") or not hasattr(layer, "mlp"):
            raise TypeError(
                f"{type(layer).__name__} has no self_attn/mlp modules; "
                "component write capture currently targets Llama-like blocks"
            )
        handles = [
            layer.self_attn.register_forward_hook(save("attention")),
            layer.mlp.register_forward_hook(save("mlp")),
        ]
    try:
        output = layer(*patched_args, **patched_kwargs)
        y = _layer_hidden_output(output)
    finally:
        for handle in handles:
            handle.remove()
    return y, captured


def token_rms(hidden: torch.Tensor) -> torch.Tensor:
    return hidden.float().square().mean(dim=-1).sqrt()


def _cos(left: torch.Tensor, right: torch.Tensor) -> float:
    dl = float(left.norm().item())
    dr = float(right.norm().item())
    if dl <= EPS or dr <= EPS:
        return float("nan")
    return float(torch.dot(left, right).item() / (dl * dr))


def branch_geometry(selected: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    """Exact per-branch radial/tangential geometry at one (prompt, layer, token).

    A pre-norm block is  H -> Y = H + A(RMSNorm(H)) -> H' = Y + M(RMSNorm(Y)).
    Each branch must therefore be decomposed against *its own* input:
    attention against ``H``, the FFN against the post-attention state ``Y``.
    ``Y`` needs no extra forward pass -- it is ``H + A``, both already captured.

    RMSNorm only rescales, so ``cos(Y, M) == cos(RMSNorm(Y), M)``; ``cos_post_attn_mlp``
    is therefore the FFN radial coefficient in the sense of arXiv:2608.02071 Eq.(19).

    Inner products are computed exactly here, so these columns carry no Jensen
    error -- unlike the layer-mean-RMS reconstructions used elsewhere.
    """

    # device_map="auto" can place the block input and its branch outputs on
    # different GPUs; these are single hidden vectors, so reduce on CPU.
    h = selected["input"].detach().float().cpu()
    a = selected["attention"].detach().float().cpu()
    m = selected["mlp"].detach().float().cpu()
    f = selected["total"].detach().float().cpu()
    y = h + a

    nh, na, nm, ny = (float(v.norm().item()) for v in (h, a, m, y))
    c_ha, c_ym, c_hf = _cos(h, a), _cos(y, m), _cos(h, f)
    rho_a = na / nh if nh > EPS else float("nan")
    rho_m = nm / ny if ny > EPS else float("nan")

    def _split(rho: float, cos: float) -> Tuple[float, float]:
        if math.isnan(rho) or math.isnan(cos):
            return float("nan"), float("nan")
        return rho * cos, rho * math.sqrt(max(0.0, 1.0 - cos * cos))

    radial_a, tangential_a = _split(rho_a, c_ha)
    radial_m, tangential_m = _split(rho_m, c_ym)
    return {
        "post_attn_l2": ny,
        "cos_input_attention": c_ha,
        "cos_post_attn_mlp": c_ym,
        "cos_input_total": c_hf,
        "rho_attention": rho_a,
        "rho_mlp": rho_m,
        "radial_attention": radial_a,
        "tangential_attention": tangential_a,
        "radial_mlp": radial_m,
        "tangential_mlp": tangential_m,
    }


def vector_summary(vectors: Sequence[np.ndarray]) -> Dict[str, float]:
    values = np.stack([np.asarray(value, dtype=np.float64) for value in vectors])
    norms = np.linalg.norm(values, axis=1)
    mean = values.mean(axis=0)
    centered = values - mean
    population_trace = float(np.mean(np.sum(centered**2, axis=1)))
    spectrum, metrics, _ = spectrum_metrics(torch.from_numpy(values).float())
    mean_squared = float(np.mean(norms**2))
    nonzero = norms > EPS
    direction_concentration = (
        float(np.linalg.norm((values[nonzero] / norms[nonzero, None]).mean(axis=0)))
        if np.any(nonzero)
        else 0.0
    )
    return {
        "n_prompts": int(values.shape[0]),
        "hidden_size": int(values.shape[1]),
        "mean_l2": float(norms.mean()),
        "rms_l2": float(math.sqrt(mean_squared)),
        "mean_squared_l2": mean_squared,
        "centroid_l2": float(np.linalg.norm(mean)),
        "centered_trace_population": population_trace,
        "centered_trace_sample": float(metrics["trace"]),
        "covariance_effective_rank": float(metrics["effective_rank"]),
        "covariance_participation_ratio": float(metrics["participation_ratio"]),
        "covariance_top_eigenvalue_fraction": float(metrics["top_eigenvalue_fraction"]),
        "direction_concentration": direction_concentration,
        "write_variability_fraction": population_trace / (mean_squared + EPS),
        "constant_component_fraction": float(np.dot(mean, mean)) / (mean_squared + EPS),
        "spectrum_nonzero_sum": float(np.sum(spectrum)),
    }


def collect_forward_writes(
    model: Any,
    probes: Sequence[Probe],
    layers: Sequence[int],
    positions: Sequence[str],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Dict[int, torch.Tensor]],
    Dict[Tuple[int, int], torch.Tensor],
    Dict[Tuple[str, int, str], np.ndarray],
]:
    """Collect block inputs and total/attention/MLP writes for all prompts."""

    decoder_layers = model_layers(model)
    device = model_input_device(model)
    vectors: Dict[Tuple[str, int, str], List[np.ndarray]] = defaultdict(list)
    prompt_rows: List[Dict[str, Any]] = []
    input_states: Dict[str, Dict[int, List[torch.Tensor]]] = {
        position: {layer: [] for layer in layers} for position in positions
    }
    sequence_rms: Dict[Tuple[int, int], torch.Tensor] = {}

    for prompt_index, probe in enumerate(probes):
        contexts = capture_layer_contexts(model, probe_inputs(probe, device), layers)
        for layer_id in layers:
            context = contexts[layer_id]
            hidden = context[0][0].detach()
            with torch.no_grad():
                y, components = call_layer(
                    decoder_layers[layer_id], context, hidden, capture_components=True
                )
            hidden_on_output = hidden.to(y.device)
            total = y - hidden_on_output
            attention = components["attention"]
            mlp = components["mlp"]
            closure = total - attention - mlp
            sequence_rms[(prompt_index, layer_id)] = token_rms(hidden[0]).cpu()

            for position in positions:
                token_pos = int(probe.positions[position])
                selected = {
                    "input": hidden[0, token_pos],
                    "total": total[0, token_pos],
                    "attention": attention[0, token_pos],
                    "mlp": mlp[0, token_pos],
                }
                input_states[position][layer_id].append(selected["input"].float().cpu())
                component_norms = {
                    name: float(value.float().norm().item()) for name, value in selected.items()
                }
                attn_mlp_cosine = float(
                    torch.nn.functional.cosine_similarity(
                        selected["attention"].float(), selected["mlp"].float(), dim=0
                    ).item()
                )
                for kind, value in selected.items():
                    vector = value.detach().float().cpu().numpy()
                    vectors[(position, layer_id, kind)].append(vector)
                    row: Dict[str, Any] = {
                        "position": position,
                        "layer": layer_id,
                        "prompt_index": prompt_index,
                        "case_id": probe.case_id,
                        "token_position": token_pos,
                        "write_kind": kind,
                        "l2": component_norms[kind],
                        "rms": component_norms[kind] / math.sqrt(max(value.numel(), 1)),
                    }
                    if kind == "total":
                        row.update(
                            {
                                "closure_relative_l2": float(closure[0, token_pos].float().norm().item())
                                / (component_norms["total"] + EPS),
                                "attention_mlp_cosine": attn_mlp_cosine,
                                "cancellation_ratio": component_norms["total"]
                                / (component_norms["attention"] + component_norms["mlp"] + EPS),
                                **branch_geometry(selected),
                            }
                        )
                    prompt_rows.append(row)
        print(f"[write] completed {prompt_index + 1}/{len(probes)} prompts")
        del contexts

    summary_rows: List[Dict[str, Any]] = []
    for (position, layer, kind), selected in sorted(vectors.items()):
        summary_rows.append(
            {
                "position": position,
                "layer": layer,
                "write_kind": kind,
                **vector_summary(selected),
            }
        )
    stacked_inputs = {
        position: {
            layer: torch.stack(values) for layer, values in layer_values.items()
        }
        for position, layer_values in input_states.items()
    }
    stacked_writes = {
        key: np.stack(selected).astype(np.float32, copy=False)
        for key, selected in vectors.items()
    }
    return prompt_rows, summary_rows, stacked_inputs, sequence_rms, stacked_writes


def save_write_vectors(
    path: Path,
    vectors: Mapping[Tuple[str, int, str], np.ndarray],
) -> None:
    np.savez(
        path,
        **{
            safe_key(position, f"L{layer}", kind): values
            for (position, layer, kind), values in vectors.items()
            if kind in {"total", "attention", "mlp"}
        },
    )


def load_write_vectors(
    path: Path,
    positions: Sequence[str],
    layers: Sequence[int],
) -> Dict[Tuple[str, int, str], np.ndarray]:
    output: Dict[Tuple[str, int, str], np.ndarray] = {}
    with np.load(path) as loaded:
        for position in positions:
            for layer in layers:
                for kind in ("total", "attention", "mlp"):
                    key = safe_key(position, f"L{layer}", kind)
                    if key not in loaded:
                        raise KeyError(f"Missing Base write vectors {key!r} in {path}")
                    output[(position, layer, kind)] = np.asarray(
                        loaded[key], dtype=np.float32
                    ).copy()
    return output


def summarize_delta_writes(
    current: Mapping[Tuple[str, int, str], np.ndarray],
    base: Mapping[Tuple[str, int, str], np.ndarray],
) -> List[Dict[str, Any]]:
    """Summarize the paired edit-induced write Delta F = F_state - F_Base."""

    rows: List[Dict[str, Any]] = []
    for key in sorted(set(current) & set(base)):
        position, layer, kind = key
        if kind not in {"total", "attention", "mlp"}:
            continue
        current_values = np.asarray(current[key], dtype=np.float32)
        base_values = np.asarray(base[key], dtype=np.float32)
        if current_values.shape != base_values.shape:
            raise ValueError(
                f"Paired write shape mismatch for {key}: "
                f"{current_values.shape} vs {base_values.shape}"
            )
        rows.append(
            {
                "position": position,
                "layer": layer,
                "write_kind": kind,
                **vector_summary(list(current_values - base_values)),
            }
        )
    return rows


def rademacher(shape: Tuple[int, int], seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    bits = torch.randint(0, 2, shape, generator=generator, dtype=torch.int8)
    return bits.to(device=device, dtype=dtype).mul_(2).sub_(1)


def _gain_from_output_probes(
    delta: torch.Tensor,
    hidden: torch.Tensor,
    token_pos: int,
    probes: torch.Tensor,
    *,
    input_basis: torch.Tensor | None = None,
    release_graph: bool,
) -> Dict[str, float]:
    total_sq: List[float] = []
    self_sq: List[float] = []
    context_sq: List[float] = []
    future_sq: List[float] = []
    projected_sq: List[float] = []
    for index, probe in enumerate(probes):
        gradient = torch.autograd.grad(
            torch.dot(delta, probe),
            hidden,
            retain_graph=not (release_graph and index + 1 == probes.size(0)),
            create_graph=False,
            allow_unused=False,
        )[0][0].float()
        self_value = float(gradient[token_pos].square().sum().item())
        context_value = float(gradient[:token_pos].square().sum().item())
        future_value = float(gradient[token_pos + 1 :].square().sum().item())
        self_sq.append(self_value)
        context_sq.append(context_value)
        future_sq.append(future_value)
        total_sq.append(self_value + context_value)
        if input_basis is not None:
            basis = input_basis.to(device=gradient.device, dtype=gradient.dtype)
            projected_sq.append(float((gradient[token_pos] @ basis).square().sum().item()))

    def summarize(values: Sequence[float], prefix: str) -> Dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        mean = float(array.mean())
        stderr = float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
        return {
            f"{prefix}_frobenius_sq": mean,
            f"{prefix}_frobenius": float(math.sqrt(max(mean, 0.0))),
            f"{prefix}_frobenius_sq_stderr": stderr,
        }

    output: Dict[str, float] = {}
    if input_basis is None:
        output.update(summarize(total_sq, "full_sequence"))
        output.update(summarize(self_sq, "full_self"))
        output.update(summarize(context_sq, "full_context"))
        output.update(summarize(future_sq, "future_leakage"))
    else:
        output.update(summarize(projected_sq, "projected_self"))
    return output


def estimate_hutchinson_gain(
    layer: torch.nn.Module,
    context: LayerContext,
    token_pos: int,
    basis: torch.Tensor,
    base_token_rms: torch.Tensor,
    *,
    full_probes: int,
    projected_probes: int,
    seed: int,
) -> Dict[str, float]:
    """Estimate natural and Base-norm-matched residual-branch gains."""

    original_hidden = context[0][0].detach()
    hidden = original_hidden.clone().requires_grad_(True)
    with torch.enable_grad():
        y, _ = call_layer(layer, context, hidden, capture_components=False)
        hidden_at_output = hidden[0, token_pos].to(y.device)
        delta = y[0, token_pos] - hidden_at_output
        output_dim = int(delta.numel())
        full_z = rademacher(
            (full_probes, output_dim), seed, delta.device, delta.dtype
        )
        natural = _gain_from_output_probes(
            delta,
            hidden,
            token_pos,
            full_z,
            release_graph=projected_probes <= 0,
        )
        if projected_probes > 0:
            output_basis = basis.to(device=delta.device, dtype=delta.dtype)
            coordinate_z = rademacher(
                (projected_probes, output_basis.size(1)),
                seed + 1,
                delta.device,
                delta.dtype,
            )
            projected_z = coordinate_z @ output_basis.T
            natural.update(
                _gain_from_output_probes(
                    delta,
                    hidden,
                    token_pos,
                    projected_z,
                    input_basis=basis,
                    release_graph=True,
                )
            )

    current_rms = token_rms(original_hidden[0]).clamp_min(EPS)
    target_rms = base_token_rms.to(device=original_hidden.device, dtype=torch.float32)
    if target_rms.numel() != original_hidden.size(1):
        raise ValueError(
            f"Base/state sequence lengths differ: {target_rms.numel()} vs {original_hidden.size(1)}"
        )
    scale = (target_rms / current_rms.float()).to(original_hidden.dtype)
    matched_hidden = (original_hidden * scale.view(1, -1, 1)).detach().requires_grad_(True)
    with torch.enable_grad():
        matched_y, _ = call_layer(layer, context, matched_hidden, capture_components=False)
        matched_delta = matched_y[0, token_pos] - matched_hidden[0, token_pos].to(matched_y.device)
        matched_z = rademacher(
            (full_probes, matched_delta.numel()),
            seed,
            matched_delta.device,
            matched_delta.dtype,
        )
        matched = _gain_from_output_probes(
            matched_delta,
            matched_hidden,
            token_pos,
            matched_z,
            release_graph=True,
        )

    selected_rms = float(current_rms[token_pos].item())
    prefix_rms = float(current_rms[: token_pos + 1].square().mean().sqrt().item())
    output = {
        **natural,
        **{f"norm_matched_{key}": value for key, value in matched.items()},
        "selected_input_rms": selected_rms,
        "prefix_input_rms": prefix_rms,
        "full_self_rms_adjusted": natural["full_self_frobenius"] * selected_rms,
        "full_sequence_rms_adjusted": natural["full_sequence_frobenius"] * prefix_rms,
        "norm_matched_full_self_ratio_to_raw": matched["full_self_frobenius"]
        / (natural["full_self_frobenius"] + EPS),
        "norm_matched_full_sequence_ratio_to_raw": matched["full_sequence_frobenius"]
        / (natural["full_sequence_frobenius"] + EPS),
    }
    if projected_probes > 0:
        output["projected_to_full_self_energy_fraction"] = natural[
            "projected_self_frobenius_sq"
        ] / (natural["full_self_frobenius_sq"] + EPS)
    return output


def collect_hutchinson_rows(
    model: Any,
    probes: Sequence[Probe],
    layers: Sequence[int],
    positions: Sequence[str],
    bases: Mapping[str, Mapping[int, torch.Tensor]],
    base_sequence_rms: Mapping[Tuple[int, int], torch.Tensor],
    *,
    full_probes: int,
    projected_probes: int,
    seed: int,
) -> List[Dict[str, Any]]:
    decoder_layers = model_layers(model)
    device = model_input_device(model)
    rows: List[Dict[str, Any]] = []
    for prompt_index, probe in enumerate(probes):
        contexts = capture_layer_contexts(model, probe_inputs(probe, device), layers)
        for position_index, position in enumerate(positions):
            token_pos = int(probe.positions[position])
            for layer_id in layers:
                probe_seed = (
                    int(seed)
                    + prompt_index * 1_000_003
                    + layer_id * 10_007
                    + position_index * 101
                )
                metrics = estimate_hutchinson_gain(
                    decoder_layers[layer_id],
                    contexts[layer_id],
                    token_pos,
                    bases[position][layer_id],
                    base_sequence_rms[(prompt_index, layer_id)],
                    full_probes=full_probes,
                    projected_probes=projected_probes,
                    seed=probe_seed,
                )
                rows.append(
                    {
                        "position": position,
                        "layer": layer_id,
                        "prompt_index": prompt_index,
                        "case_id": probe.case_id,
                        "token_position": token_pos,
                        "full_probes": full_probes,
                        "projected_probes": projected_probes,
                        **metrics,
                    }
                )
        print(f"[hutchinson] completed {prompt_index + 1}/{len(probes)} prompts")
        del contexts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def mean_finite(values: Iterable[Any]) -> float:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def summarize_hutchinson(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["position"]), int(row["layer"]))].append(row)
    ignore = {
        "state",
        "position",
        "layer",
        "prompt_index",
        "case_id",
        "token_position",
        "full_probes",
        "projected_probes",
    }
    metric_names = sorted({key for row in rows for key in row if key not in ignore})
    output: List[Dict[str, Any]] = []
    for (position, layer), selected in sorted(groups.items()):
        item: Dict[str, Any] = {
            "position": position,
            "layer": layer,
            "n_prompts": len(selected),
        }
        for metric in metric_names:
            item[metric] = mean_finite(row.get(metric, float("nan")) for row in selected)
        output.append(item)
    return output


def add_base_ratios(
    rows: Sequence[MutableMapping[str, Any]],
    base_label: str,
    keys: Sequence[str],
    identity: Sequence[str],
) -> None:
    base = {
        tuple(str(row[name]) for name in identity): row
        for row in rows
        if str(row["state"]) == base_label
    }
    for row in rows:
        reference = base.get(tuple(str(row[name]) for name in identity))
        for key in keys:
            row[f"{key}_ratio_to_base"] = (
                float(row[key]) / (float(reference[key]) + EPS)
                if reference is not None and key in row and key in reference
                else float("nan")
            )


def plot_summaries(
    output_dir: Path,
    labels: Sequence[str],
    positions: Sequence[str],
    write_rows: Sequence[Mapping[str, Any]],
    hutch_rows: Sequence[Mapping[str, Any]],
) -> None:
    colors = _state_colors(labels)
    write_specs = [
        ("rms_l2_ratio_to_base", "total write RMS / Base"),
        ("centered_trace_population_ratio_to_base", "write diversity trace / Base"),
        ("covariance_effective_rank_ratio_to_base", "write covariance eRank / Base"),
    ]
    hutch_specs = [
        ("projected_self_frobenius_ratio_to_base", "Base-PCA self gain / Base"),
        ("full_self_frobenius_ratio_to_base", "full self-token gain / Base"),
        ("full_context_frobenius_ratio_to_base", "full context gain / Base"),
        (
            "norm_matched_full_sequence_frobenius_ratio_to_base",
            "norm-matched full-sequence gain / Base",
        ),
    ]
    for position in positions:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for axis, (metric, title) in zip(axes, write_specs):
            for label in labels:
                selected = sorted(
                    [
                        row
                        for row in write_rows
                        if row["state"] == label
                        and row["position"] == position
                        and row["write_kind"] == "total"
                    ],
                    key=lambda row: int(row["layer"]),
                )
                axis.plot(
                    [int(row["layer"]) for row in selected],
                    [float(row[metric]) for row in selected],
                    marker="o",
                    label=label,
                    color=colors[label],
                )
            axis.axhline(1.0, color="#666666", linestyle="--", linewidth=1)
            axis.set_title(title)
            axis.set_xlabel("layer")
            axis.grid(alpha=0.22)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"Residual write diagnostics — {position}", fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / f"branch_write_diagnostics_{position}.png", dpi=190)
        plt.close(fig)

        if not hutch_rows:
            continue
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
        for axis, (metric, title) in zip(axes, hutch_specs):
            for label in labels:
                selected = sorted(
                    [
                        row
                        for row in hutch_rows
                        if row["state"] == label and row["position"] == position
                    ],
                    key=lambda row: int(row["layer"]),
                )
                axis.plot(
                    [int(row["layer"]) for row in selected],
                    [float(row[metric]) for row in selected],
                    marker="o",
                    label=label,
                    color=colors[label],
                )
            axis.axhline(1.0, color="#666666", linestyle="--", linewidth=1)
            axis.set_title(title)
            axis.set_xlabel("layer")
            axis.grid(alpha=0.22)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"Hutchinson residual-branch gain — {position}", fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / f"branch_gain_diagnostics_{position}.png", dpi=190)
        plt.close(fig)


def build_mechanism_summary(
    write_rows: Sequence[Mapping[str, Any]],
    hutch_rows: Sequence[Mapping[str, Any]],
    delta_write_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    total_write = {
        (str(row["state"]), str(row["position"]), int(row["layer"])): row
        for row in write_rows
        if str(row["write_kind"]) == "total"
    }
    hutch = {
        (str(row["state"]), str(row["position"]), int(row["layer"])): row
        for row in hutch_rows
    }
    delta_total_write = {
        (str(row["state"]), str(row["position"]), int(row["layer"])): row
        for row in delta_write_rows
        if str(row["write_kind"]) == "total"
    }
    write_metrics = (
        "rms_l2_ratio_to_base",
        "centroid_l2_ratio_to_base",
        "centered_trace_population_ratio_to_base",
        "covariance_effective_rank_ratio_to_base",
        "direction_concentration_ratio_to_base",
        "write_variability_fraction",
        "constant_component_fraction",
    )
    hutch_metrics = (
        "projected_self_frobenius_ratio_to_base",
        "full_self_frobenius_ratio_to_base",
        "full_context_frobenius_ratio_to_base",
        "full_sequence_frobenius_ratio_to_base",
        "norm_matched_full_self_frobenius_ratio_to_base",
        "norm_matched_full_sequence_frobenius_ratio_to_base",
        "full_self_rms_adjusted_ratio_to_base",
        "full_sequence_rms_adjusted_ratio_to_base",
        "selected_input_rms_ratio_to_base",
        "prefix_input_rms_ratio_to_base",
        "projected_to_full_self_energy_fraction",
        "norm_matched_full_self_ratio_to_raw",
        "norm_matched_full_sequence_ratio_to_raw",
        "future_leakage_frobenius",
    )
    rows: List[Dict[str, Any]] = []
    for key in sorted(set(total_write) & set(hutch)):
        state, position, layer = key
        write = total_write[key]
        gain = hutch[key]
        delta_write = delta_total_write.get(key, {})
        rows.append(
            {
                "state": state,
                "position": position,
                "layer": layer,
                **{name: write.get(name, float("nan")) for name in write_metrics},
                **{name: gain.get(name, float("nan")) for name in hutch_metrics},
                **{
                    f"delta_total_{name}": delta_write.get(name, float("nan"))
                    for name in (
                        "rms_l2",
                        "centroid_l2",
                        "centered_trace_population",
                        "covariance_effective_rank",
                        "direction_concentration",
                        "write_variability_fraction",
                        "constant_component_fraction",
                    )
                },
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=parse_state_spec, action="append", default=[])
    parser.add_argument("--requests-path", required=True)
    parser.add_argument("--exclude-requests-path", action="append", default=[])
    parser.add_argument("--probe-source", choices=["rewrite", "locality"], default="rewrite")
    parser.add_argument("--probe-selection", choices=["prefix", "random"], default="random")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument(
        "--base-model-path",
        default=None,
        help=(
            "Base Hugging Face model for compact edited_parameter_deltas.pt "
            "states. Defaults to base_model in each checkpoint manifest."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", type=parse_int_list, default=parse_int_list("4,6,8,10,12,14,16,18,20"))
    parser.add_argument("--positions", type=parse_str_list, default=parse_str_list("subject_last,prompt_last"))
    parser.add_argument("--max-prompts", type=int, default=50)
    parser.add_argument("--hutchinson-prompts", type=int, default=8)
    parser.add_argument("--full-probes", type=int, default=8)
    parser.add_argument("--projected-probes", type=int, default=8)
    parser.add_argument(
        "--skip-hutchinson",
        action="store_true",
        help=(
            "Collect only forward residual/attention/MLP writes. This is the "
            "fast path for boundary-location controls that do not test gains."
        ),
    )
    parser.add_argument("--pca-rank", type=int, default=32)
    parser.add_argument("--pca-bases-path", default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="eager")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.state:
        raise ValueError("At least one --state is required; the first must be Base")
    labels = [state.label for state in args.state]
    if len(labels) != len(set(labels)):
        raise ValueError("State labels must be unique")
    if args.max_prompts <= 1:
        raise ValueError("--max-prompts must exceed one")
    if (
        not args.skip_hutchinson
        and not 0 < args.hutchinson_prompts <= args.max_prompts
    ):
        raise ValueError("--hutchinson-prompts must be in [1, max-prompts]")
    if (
        not args.skip_hutchinson
        and (args.full_probes <= 0 or args.projected_probes <= 0)
    ):
        raise ValueError("Hutchinson probe counts must be positive")
    if args.pca_rank <= 0:
        raise ValueError("--pca-rank must be positive")
    if args.probe_source == "locality" and "subject_last" in args.positions:
        raise ValueError("locality probes support prompt_last, not subject_last")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    first_model_source, _, _ = resolve_model_load_spec(
        args.state[0].path,
        args.base_model_path,
    )
    tokenizer_path = args.tokenizer_path or first_model_source
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=args.trust_remote_code
    )
    probes = load_probes(
        args.requests_path,
        tokenizer,
        args.positions,
        args.max_prompts,
        args.max_length,
        probe_source=args.probe_source,
        exclude_requests_paths=args.exclude_requests_path,
        selection=args.probe_selection,
        seed=args.seed,
    )
    config = {
        "states": [{"label": state.label, "path": state.path} for state in args.state],
        "base_model_path": args.base_model_path,
        "requests_path": str(Path(args.requests_path).expanduser().resolve()),
        "exclude_requests_paths": [
            str(Path(path).expanduser().resolve()) for path in args.exclude_requests_path
        ],
        "probe_source": args.probe_source,
        "probe_selection": args.probe_selection,
        "probe_case_ids": [str(probe.case_id) for probe in probes],
        "layers": args.layers,
        "positions": args.positions,
        "max_prompts": args.max_prompts,
        "hutchinson_prompts": args.hutchinson_prompts,
        "full_probes": args.full_probes,
        "projected_probes": args.projected_probes,
        "skip_hutchinson": args.skip_hutchinson,
        "pca_rank": args.pca_rank,
        "pca_bases_path": args.pca_bases_path,
        "max_length": args.max_length,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "seed": args.seed,
    }
    run_fingerprint = fingerprint(config)
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != run_fingerprint:
            raise ValueError(f"Existing run config differs: {config_path}; use another output dir")
    else:
        atomic_json(config_path, {"fingerprint": run_fingerprint, **config})

    labels = [state.label for state in args.state]
    all_write_prompt: List[MutableMapping[str, Any]] = []
    all_write_summary: List[MutableMapping[str, Any]] = []
    all_delta_write_summary: List[MutableMapping[str, Any]] = []
    all_hutch_prompt: List[MutableMapping[str, Any]] = []
    all_hutch_summary: List[MutableMapping[str, Any]] = []
    bases: Dict[str, Dict[int, torch.Tensor]] | None = None
    base_sequence_rms: Dict[Tuple[int, int], torch.Tensor] | None = None
    base_write_vectors: Dict[Tuple[str, int, str], np.ndarray] | None = None
    base_write_vectors_path = output_dir / "base_write_vectors.npz"

    for state_index, state in enumerate(args.state):
        state_dir = output_dir / "states" / safe_key(state.label)
        reference_ready = (
            base_write_vectors_path.exists()
            and (
                args.skip_hutchinson
                or (
                    (output_dir / "shared_pca_bases.npz").exists()
                    and (output_dir / "base_sequence_rms.npz").exists()
                )
            )
        )
        if state_is_complete(
            state_dir,
            run_fingerprint,
            require_delta=state_index > 0,
            require_hutchinson=not args.skip_hutchinson,
        ) and (
            state_index > 0 or reference_ready
        ):
            if state_index == 0:
                if not args.skip_hutchinson:
                    bases, base_sequence_rms = load_shared_reference(
                        output_dir, args.positions, args.layers, len(probes)
                    )
                base_write_vectors = load_write_vectors(
                    base_write_vectors_path, args.positions, args.layers
                )
            all_write_prompt.extend(read_csv(state_dir / "write_prompt_metrics.csv"))
            all_write_summary.extend(read_csv(state_dir / "write_summary_metrics.csv"))
            if not args.skip_hutchinson:
                all_hutch_prompt.extend(
                    read_csv(state_dir / "hutchinson_prompt_metrics.csv")
                )
                all_hutch_summary.extend(
                    read_csv(state_dir / "hutchinson_summary_metrics.csv")
                )
            if state_index > 0:
                all_delta_write_summary.extend(
                    read_csv(state_dir / "delta_write_summary_metrics.csv")
                )
            print(f"[resume] preserved completed state {state.label}; model not loaded")
            continue

        print(f"[model] loading {state.label}: {state.path}")
        model = load_model(
            state.path,
            args.torch_dtype,
            args.device_map,
            args.trust_remote_code,
            args.attn_implementation,
            args.base_model_path,
        )
        prompt_rows, write_summary, input_states, sequence_rms, write_vectors = collect_forward_writes(
            model, probes, args.layers, args.positions
        )
        if state_index == 0:
            if not args.skip_hutchinson:
                fitted_bases, _, basis_rows = fit_shared_bases(
                    input_states, args.positions, args.layers, args.pca_rank
                )
                if args.pca_bases_path:
                    hidden_size = next(
                        iter(next(iter(input_states.values())).values())
                    ).size(1)
                    bases = load_external_bases(
                        args.pca_bases_path,
                        args.positions,
                        args.layers,
                        args.pca_rank,
                        hidden_size,
                    )
                    basis_source = str(
                        Path(args.pca_bases_path).expanduser().resolve()
                    )
                else:
                    bases = fitted_bases
                    basis_source = "current_probe_base_state"
                for row in basis_rows:
                    row["basis_source"] = basis_source
                write_csv(
                    output_dir / "shared_pca_basis_metrics.csv", basis_rows
                )
                np.savez(
                    output_dir / "shared_pca_bases.npz",
                    **{
                        safe_key(position, f"L{layer}"): basis.numpy()
                        for position, layer_bases in bases.items()
                        for layer, basis in layer_bases.items()
                    },
                )
                base_sequence_rms = sequence_rms
                np.savez(
                    output_dir / "base_sequence_rms.npz",
                    **{
                        safe_key(f"P{prompt_index}", f"L{layer}"): values.numpy()
                        for (prompt_index, layer), values in base_sequence_rms.items()
                    },
                )
            base_write_vectors = write_vectors
            save_write_vectors(base_write_vectors_path, base_write_vectors)

        delta_write_summary: List[Dict[str, Any]] = []
        if state_index > 0:
            if base_write_vectors is None:
                raise RuntimeError("Base write vectors were not initialized")
            delta_write_summary = summarize_delta_writes(
                write_vectors, base_write_vectors
            )

        if args.skip_hutchinson:
            hutch_rows: List[Dict[str, Any]] = []
            hutch_summary: List[Dict[str, Any]] = []
        else:
            if bases is None or base_sequence_rms is None:
                raise RuntimeError("Base PCA/RMS reference was not initialized")
            hutch_rows = collect_hutchinson_rows(
                model,
                probes[: args.hutchinson_prompts],
                args.layers,
                args.positions,
                bases,
                base_sequence_rms,
                full_probes=args.full_probes,
                projected_probes=args.projected_probes,
                seed=args.seed,
            )
            hutch_summary = summarize_hutchinson(hutch_rows)
        for rows in (prompt_rows, write_summary, hutch_rows, hutch_summary):
            for row in rows:
                row["state"] = state.label
        all_write_prompt.extend(prompt_rows)
        all_write_summary.extend(write_summary)
        all_hutch_prompt.extend(hutch_rows)
        all_hutch_summary.extend(hutch_summary)
        for row in delta_write_summary:
            row["state"] = state.label
        all_delta_write_summary.extend(delta_write_summary)

        state_dir.mkdir(parents=True, exist_ok=True)
        write_csv(state_dir / "write_prompt_metrics.csv", prompt_rows)
        write_csv(state_dir / "write_summary_metrics.csv", write_summary)
        if not args.skip_hutchinson:
            write_csv(state_dir / "hutchinson_prompt_metrics.csv", hutch_rows)
            write_csv(state_dir / "hutchinson_summary_metrics.csv", hutch_summary)
        if state_index > 0:
            write_csv(
                state_dir / "delta_write_summary_metrics.csv",
                delta_write_summary,
            )
        atomic_json(
            state_dir / "complete.json",
            {"fingerprint": run_fingerprint, "state": state.label, "model_saved": False},
        )
        del model, input_states, sequence_rms, write_vectors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[model] completed {state.label}")

    write_ratio_keys = [
        "mean_l2",
        "rms_l2",
        "centroid_l2",
        "centered_trace_population",
        "covariance_effective_rank",
        "direction_concentration",
    ]
    add_base_ratios(
        all_write_summary,
        labels[0],
        write_ratio_keys,
        ("position", "layer", "write_kind"),
    )
    if all_hutch_summary:
        hutch_ratio_keys = [
            key
            for key in all_hutch_summary[0]
            if key.endswith("_frobenius")
            or key
            in {
                "selected_input_rms",
                "prefix_input_rms",
                "full_self_rms_adjusted",
                "full_sequence_rms_adjusted",
            }
        ]
        add_base_ratios(
            all_hutch_summary,
            labels[0],
            hutch_ratio_keys,
            ("position", "layer"),
        )
    write_csv(output_dir / "write_prompt_metrics.csv", all_write_prompt)
    write_csv(output_dir / "write_summary_metrics.csv", all_write_summary)
    if not args.skip_hutchinson:
        write_csv(output_dir / "hutchinson_prompt_metrics.csv", all_hutch_prompt)
        write_csv(output_dir / "hutchinson_summary_metrics.csv", all_hutch_summary)
    write_csv(output_dir / "delta_write_summary_metrics.csv", all_delta_write_summary)
    write_csv(
        output_dir / "mechanism_summary.csv",
        build_mechanism_summary(
            all_write_summary,
            all_hutch_summary,
            all_delta_write_summary,
        ),
    )
    plot_summaries(
        output_dir,
        labels,
        args.positions,
        all_write_summary,
        all_hutch_summary,
    )
    atomic_json(
        output_dir / "run_manifest.json",
        {
            "fingerprint": run_fingerprint,
            **config,
            "definitions": {
                "total_write": "decoder_block(H)[t] - H[t]",
                "write_variability": "across-prompt centered covariance of F_t(H)",
                "full_sequence_gain": "Hutchinson estimate of ||dF_t/dH_<=t||_F",
                "full_self_gain": "same-token slice ||dF_t/dh_t||_F",
                "full_context_gain": "causal-prefix slice ||dF_t/dH_<t||_F",
                "projected_self_gain": "Base-PCA estimate ||U^T(dF_t/dh_t)U||_F",
                "norm_matched": "edited block evaluated after per-token hidden RMS is matched to paired Base",
                "delta_write": "paired edit-induced write F_state(H_state) - F_Base(H_Base) on the same prompt and token",
            },
            "model_checkpoint_saved": False,
        },
    )
    print(f"[done] residual-branch collapse diagnostics written to {output_dir}")


if __name__ == "__main__":
    main()
