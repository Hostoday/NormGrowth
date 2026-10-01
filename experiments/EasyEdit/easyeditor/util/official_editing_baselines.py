"""Opt-in ENCORE, NSE, and SPHERE components for MEMIT and AlphaEdit.

All runtime state lives outside the hyperparameter objects so large sequential
Gram matrices and cached value targets are not serialized into run manifests.
When all three feature flags are disabled, these helpers are no-ops.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
from transformers.pytorch_utils import Conv1D


@dataclass
class MPESState:
    """State matching ENCORE's released cumulative ``correct_counter``."""

    qualifying_steps: int = 0
    first_qualifying_step: Optional[int] = None


def mpes_observe(
    *,
    log_probs: torch.Tensor,
    rewriting_targets: torch.Tensor,
    state: MPESState,
    required_top1_steps: int,
    step: int,
    exclude_first_context: bool = True,
) -> Dict[str, Any]:
    """Observe one latent step and reproduce ENCORE MPES stopping."""

    if log_probs.ndim != 3 or rewriting_targets.ndim != 2:
        raise ValueError(
            "MPES expects log_probs [batch, seq, vocab] and targets [batch, seq]"
        )
    if required_top1_steps < 1:
        raise ValueError("ENCORE required_top1_steps must be >= 1")

    start = 1 if exclude_first_context and log_probs.shape[0] > 1 else 0
    scoped_log_probs = log_probs[start:]
    scoped_targets = rewriting_targets[start:].to(scoped_log_probs.device)
    batch_indices, token_indices = torch.nonzero(
        scoped_targets.ne(-100), as_tuple=True
    )
    total = int(batch_indices.numel())
    if total == 0:
        return {
            "all_top1": False,
            "num_top1": 0,
            "num_targets": 0,
            "qualifying_steps": state.qualifying_steps,
            "first_qualifying_step": state.first_qualifying_step,
            "should_stop": False,
        }

    target_ids = scoped_targets[batch_indices, token_indices]
    top_ids = scoped_log_probs[batch_indices, token_indices].argmax(dim=-1)
    num_top1 = int(target_ids.eq(top_ids).sum().item())
    all_top1 = num_top1 == total
    if all_top1:
        state.qualifying_steps += 1
        if state.first_qualifying_step is None:
            state.first_qualifying_step = int(step)
    return {
        "all_top1": all_top1,
        "num_top1": num_top1,
        "num_targets": total,
        "qualifying_steps": state.qualifying_steps,
        "first_qualifying_step": state.first_qualifying_step,
        "should_stop": state.qualifying_steps >= required_top1_steps,
    }


_NSE_TARGETS: Dict[int, Dict[str, torch.Tensor]] = {}
_NSE_GRAMS: Dict[int, Dict[int, torch.Tensor]] = {}
_ENCORE_GRAMS: Dict[int, Dict[int, torch.Tensor]] = {}
_ENCORE_COUNTS: Dict[int, Dict[int, int]] = {}


def _run_key(hparams: Any) -> int:
    return id(hparams)


def get_encore_gram(
    hparams: Any,
    *,
    layer: int,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Return ENCORE's sum of prior edit-key Gram matrices."""

    gram = _ENCORE_GRAMS.setdefault(_run_key(hparams), {}).get(int(layer))
    if gram is None:
        return None
    if gram.shape != (dimension, dimension):
        raise ValueError(
            f"ENCORE Gram shape changed at layer {layer}: {gram.shape} vs "
            f"{(dimension, dimension)}"
        )
    return gram.to(device=device, dtype=dtype)


def update_encore_gram(
    hparams: Any,
    *,
    layer: int,
    layer_keys: torch.Tensor,
) -> int:
    """Store the current ``K K^T`` for subsequent ENCORE updates."""

    keys = layer_keys.detach().to(device="cpu", dtype=torch.float32)
    increment = keys @ keys.T
    run_key = _run_key(hparams)
    grams = _ENCORE_GRAMS.setdefault(run_key, {})
    counts = _ENCORE_COUNTS.setdefault(run_key, {})
    layer = int(layer)
    if layer not in grams:
        grams[layer] = increment
    else:
        grams[layer].add_(increment)
    counts[layer] = counts.get(layer, 0) + int(layer_keys.shape[1])
    return counts[layer]


def nse_case_key(case_id: Any) -> str:
    return str(case_id)


def reset_nse_runtime(
    hparams: Any,
    *,
    clear_targets: bool = True,
    clear_grams: bool = True,
) -> None:
    key = _run_key(hparams)
    if clear_targets:
        _NSE_TARGETS[key] = {}
    if clear_grams:
        _NSE_GRAMS[key] = {}


def store_nse_target(hparams: Any, case_id: Any, target: torch.Tensor) -> None:
    targets = _NSE_TARGETS.setdefault(_run_key(hparams), {})
    targets[nse_case_key(case_id)] = target.detach().float().cpu()


def get_nse_target(
    hparams: Any,
    case_id: Any,
    *,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    key = nse_case_key(case_id)
    targets = _NSE_TARGETS.get(_run_key(hparams), {})
    if key not in targets:
        raise RuntimeError(
            f"NSE original-model target is missing for case {case_id!r}; "
            "the local runner must precompute all targets before editing"
        )
    target = targets[key]
    return target.to(device=device, dtype=dtype or target.dtype)


def nse_target_cache_file(
    cache_dir: str,
    *,
    layer: int,
    clamp_norm_factor: float,
    case_id: Any,
) -> str:
    safe_case = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in nse_case_key(case_id)
    )
    return os.path.join(
        cache_dir,
        f"layer_{int(layer)}_clamp_{clamp_norm_factor}_case_{safe_case}.pt",
    )


def nse_select_neurons(
    layer_keys: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Select the minimal high-activation neuron set from NSE Eq. (10)."""

    if layer_keys.ndim != 2:
        raise ValueError(f"NSE expects a key matrix, got {layer_keys.shape}")
    threshold = float(threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("NSE neuron_threshold must lie in (0, 1]")

    scores = layer_keys.detach().abs().sum(dim=1)
    total = scores.sum()
    if not torch.isfinite(total) or float(total.item()) <= 0.0:
        selected = torch.arange(layer_keys.shape[0], device=layer_keys.device)
        return selected, {
            "selected_neurons": float(selected.numel()),
            "total_neurons": float(layer_keys.shape[0]),
            "selected_fraction": 1.0,
            "activation_fraction": 1.0,
        }
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    cumulative = torch.cumsum(sorted_scores, dim=0)
    count = int(
        torch.searchsorted(cumulative, threshold * total, right=False).item()
    ) + 1
    count = max(1, min(count, int(sorted_indices.numel())))
    selected = sorted_indices[:count]
    return selected, {
        "selected_neurons": float(count),
        "total_neurons": float(layer_keys.shape[0]),
        "selected_fraction": float(count / layer_keys.shape[0]),
        "activation_fraction": float((cumulative[count - 1] / total).item()),
    }


def nse_restricted_solve(
    system: torch.Tensor,
    rhs: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    """Solve on NSE-selected neurons and embed the result in full space."""

    selected = selected.to(device=system.device, dtype=torch.long)
    restricted_system = system.index_select(0, selected).index_select(1, selected)
    restricted_rhs = rhs.index_select(0, selected)
    restricted_solution = torch.linalg.solve(
        restricted_system, restricted_rhs
    )
    solution = torch.zeros(
        (system.shape[0], rhs.shape[1]),
        device=restricted_solution.device,
        dtype=restricted_solution.dtype,
    )
    solution.index_copy_(0, selected, restricted_solution)
    return solution


def get_nse_gram(
    hparams: Any,
    *,
    layer: int,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    grams = _NSE_GRAMS.setdefault(_run_key(hparams), {})
    if int(layer) not in grams:
        grams[int(layer)] = torch.zeros(
            (dimension, dimension), dtype=torch.float32, device="cpu"
        )
    gram = grams[int(layer)]
    if gram.shape != (dimension, dimension):
        raise ValueError(
            f"NSE Gram shape changed at layer {layer}: {gram.shape} vs "
            f"{(dimension, dimension)}"
        )
    return gram.to(device=device, dtype=dtype)


def update_nse_gram(
    hparams: Any,
    *,
    layer: int,
    layer_keys: torch.Tensor,
) -> None:
    keys = layer_keys.detach().float().cpu()
    increment = keys @ keys.T
    grams = _NSE_GRAMS.setdefault(_run_key(hparams), {})
    if int(layer) not in grams:
        grams[int(layer)] = increment
    else:
        grams[int(layer)].add_(increment)


def append_official_baseline_record(
    path: Optional[str],
    record: Dict[str, Any],
) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sphere_project_update(
    weight: torch.Tensor,
    update: torch.Tensor,
    *,
    beta: float,
    alpha: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Project a canonical [output, input] weight/update pair.

    Stored model parameters must go through sphere_project_update_for_module;
    this tensor-only core deliberately retains the original Linear arithmetic.
    """

    beta = float(beta)
    alpha = float(alpha)
    if not 0.0 < beta <= 1.0:
        raise ValueError("SPHERE beta must lie in (0, 1]")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("SPHERE alpha must lie in [0, 1]")
    if weight.shape != update.shape:
        raise ValueError(
            f"SPHERE needs matched shapes, got {weight.shape} and {update.shape}"
        )

    a = weight.detach().to(device=weight.device, dtype=torch.float32)
    b = update.to(device=weight.device, dtype=torch.float32)
    a_hat = a / a.norm(dim=1, keepdim=True).clamp_min(1e-8)
    n_rows = int(a_hat.shape[0])
    dual = (a_hat @ a_hat.T) / float(n_rows)
    eigvals, left_eigvecs = torch.linalg.eigh(dual)
    positive = eigvals > torch.finfo(eigvals.dtype).eps * max(
        1.0, float(eigvals.max().item())
    )
    eigvals = eigvals[positive]
    left_eigvecs = left_eigvecs[:, positive]
    before_norm = float(b.norm().item())
    if eigvals.numel() == 0:
        return b.to(dtype=update.dtype), {
            "rank": 0.0,
            "captured_energy": 0.0,
            "update_norm_before": before_norm,
            "update_norm_after": before_norm,
            "norm_ratio": 1.0,
        }

    eigvals = eigvals.flip(0)
    left_eigvecs = left_eigvecs.flip(1)
    cumulative = torch.cumsum(eigvals, dim=0)
    total = cumulative[-1]
    rank = int(
        torch.searchsorted(
            cumulative / total,
            torch.tensor(beta, device=total.device),
        ).item()
    ) + 1
    rank = max(1, min(rank, int(eigvals.numel())))
    principal = (a_hat.T @ left_eigvecs[:, :rank]) / torch.sqrt(
        float(n_rows) * eigvals[:rank]
    ).unsqueeze(0).clamp_min(1e-12)
    principal = torch.linalg.qr(principal, mode="reduced").Q
    projected = b - alpha * ((b @ principal) @ principal.T)
    after_norm = float(projected.norm().item())
    return projected.to(dtype=update.dtype), {
        "rank": float(rank),
        "captured_energy": float((cumulative[rank - 1] / total).item()),
        "update_norm_before": before_norm,
        "update_norm_after": after_norm,
        "norm_ratio": after_norm / max(before_norm, 1e-12),
    }


def sphere_project_update_for_module(
    module: torch.nn.Module,
    weight: torch.Tensor,
    update: torch.Tensor,
    *,
    beta: float,
    alpha: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Apply SPHERE to MLP input directions for either supported storage layout.

    GPT-2 Conv1D stores [input, output], so both the reference weight and its
    update are transposed before row normalization and principal-space fitting.
    Detect the module type explicitly: square parameters cannot reveal layout.
    """
    if isinstance(module, Conv1D):
        transposed = True
        semantic_weight, semantic_update = weight.T, update.T
        stored_layout = "input_output"
    elif isinstance(module, torch.nn.Linear):
        transposed = False
        semantic_weight, semantic_update = weight, update
        stored_layout = "output_input"
    else:
        raise TypeError(
            f"Unsupported SPHERE rewrite module: {type(module).__name__}; "
            "an explicit weight-layout adapter is required"
        )
    projected, stats = sphere_project_update(
        semantic_weight, semantic_update, beta=beta, alpha=alpha
    )
    stats.update({
        "sphere_orientation_version": "semantic_output_input_v1",
        "module_type": f"{type(module).__module__}.{type(module).__qualname__}",
        "stored_weight_layout": stored_layout,
        "stored_weight_shape": list(weight.shape),
        "projection_weight_layout": "output_input",
        "projection_weight_shape": list(semantic_weight.shape),
        "projection_axis": "mlp_input",
        "projection_dimension": int(semantic_weight.shape[1]),
        "orientation_transposed": transposed,
    })
    return projected.T if transposed else projected, stats


def project_updates_with_sphere(
    model: Any,
    hparams: Any,
    update_matrices: Dict[str, torch.Tensor],
    *,
    method: str,
    parameter_getter: Any,
) -> Dict[str, torch.Tensor]:
    if not bool(getattr(hparams, "sphere_enabled", False)):
        return update_matrices

    beta = float(getattr(hparams, "sphere_beta", 0.5))
    alpha = float(getattr(hparams, "sphere_alpha", 0.8))
    projected_updates: Dict[str, torch.Tensor] = {}
    for weight_name, update in update_matrices.items():
        weight = parameter_getter(model, weight_name)
        module_name, _, parameter_name = weight_name.rpartition(".")
        if parameter_name != "weight":
            raise ValueError(f"SPHERE requires a rewrite weight parameter: {weight_name}")
        module = model.get_submodule(module_name)
        projected, stats = sphere_project_update_for_module(
            module,
            weight,
            update.to(weight.device),
            beta=beta,
            alpha=alpha,
        )
        projected_updates[weight_name] = projected.detach()
        print(
            f"[SPHERE][{method}] {weight_name} rank={int(stats['rank'])} "
            f"energy={stats['captured_energy']:.4f} "
            f"||dW'||/||dW||={stats['norm_ratio']:.4f}"
        )
        append_official_baseline_record(
            getattr(hparams, "official_baseline_log_path", None),
            {
                "method": method,
                "baseline": "SPHERE",
                "weight_name": weight_name,
                "beta": beta,
                "alpha": alpha,
                **stats,
            },
        )
    return projected_updates
