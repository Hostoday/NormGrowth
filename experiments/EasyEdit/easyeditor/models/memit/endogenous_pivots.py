"""Endogenous-pivot refinement for MEMIT's low-rank outer updates.

Native MEMIT returns one pair ``(Q_l, R_l)`` per edited layer.  For a
``torch.nn.Linear`` rewrite module the resulting weight update is

    delta W_l = R_l Q_l^T,

and its contribution to a token activation is ``(X_l Q_l) R_l^T``.  This
module keeps every ``Q_l`` fixed and refines only the ``R_l`` factors through
the actual transformer forward graph.  Consequently, the intermediate
states (the "pivots") are induced by the model dynamics instead of being
specified in advance.

The feature is deliberately optional.  Importing this module or leaving its
hparam disabled does not alter native MEMIT.
"""

from __future__ import annotations

import json
import math
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import torch

from ...util import nethook
from .compute_z import find_fact_lookup_idx


FactorPair = Tuple[torch.Tensor, torch.Tensor]


def validate_endogenous_pivot_hparams(hparams: Any) -> None:
    """Fail closed on configurations that would make refinement ambiguous."""

    mode = str(getattr(hparams, "endogenous_pivot_mode", "coordinate")).lower()
    if mode not in {"joint", "coordinate"}:
        raise ValueError(
            "endogenous_pivot_mode must be 'joint' or 'coordinate'; "
            f"got {mode!r}"
        )

    layers = [int(layer) for layer in getattr(hparams, "layers", [])]
    if not layers:
        raise ValueError("Endogenous-pivot MEMIT requires at least one edit layer")
    if layers != sorted(set(layers)):
        raise ValueError(
            "Endogenous-pivot MEMIT requires strictly increasing, unique layers; "
            f"got {layers}"
        )

    configured_trainable = getattr(
        hparams,
        "endogenous_pivot_trainable_layers",
        None,
    )
    if configured_trainable is not None:
        trainable_layers = [int(layer) for layer in configured_trainable]
        if not trainable_layers:
            raise ValueError(
                "endogenous_pivot_trainable_layers must be non-empty when set"
            )
        if trainable_layers != sorted(set(trainable_layers)):
            raise ValueError(
                "endogenous_pivot_trainable_layers must be strictly increasing "
                f"and unique; got {trainable_layers}"
            )
        unknown = [layer for layer in trainable_layers if layer not in layers]
        if unknown:
            raise ValueError(
                "endogenous_pivot_trainable_layers must be a subset of layers; "
                f"unknown layers: {unknown}"
            )

    positive_ints = {
        "endogenous_pivot_num_steps": int(
            getattr(hparams, "endogenous_pivot_num_steps", 5)
        ),
        "endogenous_pivot_num_sweeps": int(
            getattr(hparams, "endogenous_pivot_num_sweeps", 2)
        ),
    }
    for name, value in positive_ints.items():
        if value < 1:
            raise ValueError(f"{name} must be >= 1; got {value}")

    lr = float(getattr(hparams, "endogenous_pivot_lr", 1e-2))
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError(f"endogenous_pivot_lr must be finite and > 0; got {lr}")

    cost_lambda = float(
        getattr(hparams, "endogenous_pivot_l2_lambda", 1e-2)
    )
    if not math.isfinite(cost_lambda) or cost_lambda < 0.0:
        raise ValueError(
            "endogenous_pivot_l2_lambda must be finite and >= 0; "
            f"got {cost_lambda}"
        )

    grad_clip = float(
        getattr(hparams, "endogenous_pivot_grad_clip_norm", 1.0)
    )
    if not math.isfinite(grad_clip) or grad_clip < 0.0:
        raise ValueError(
            "endogenous_pivot_grad_clip_norm must be finite and >= 0; "
            f"got {grad_clip}"
        )


def _resolve_trainable_layers(hparams: Any, layers: List[int]) -> List[int]:
    configured = getattr(hparams, "endogenous_pivot_trainable_layers", None)
    if configured is None:
        return list(layers)
    return [int(layer) for layer in configured]


def memit_covariance_factor_gram(
    layer_keys: torch.Tensor,
    adjusted_keys: torch.Tensor,
    mom2_update_weight: float,
) -> torch.Tensor:
    """Return ``Q^T C Q`` without another multiplication by the large ``C``.

    Native MEMIT solves ``(mu C + K K^T) Q = K``.  Left-multiplication by
    ``Q^T`` gives

    ``mu Q^T C Q = Q^T K - (Q^T K)(K^T Q)``.

    Only rank-sized matrices are formed here.  Tiny negative eigenvalues from
    the solve are projected away so the ensuing update-cost penalty cannot
    acquire spurious negative directions.
    """

    mu = float(mom2_update_weight)
    if not math.isfinite(mu) or mu <= 0.0:
        raise ValueError(
            "mom2_update_weight must be finite and > 0 for endogenous-pivot "
            f"covariance cost; got {mu}"
        )
    if layer_keys.ndim != 2 or adjusted_keys.ndim != 2:
        raise ValueError("K and Q must both be rank-2 tensors")
    if layer_keys.shape != adjusted_keys.shape:
        raise ValueError(
            "K and Q must have identical [input_dim, rank] shapes; got "
            f"{tuple(layer_keys.shape)} and {tuple(adjusted_keys.shape)}"
        )

    k = layer_keys.double()
    q = adjusted_keys.to(device=k.device, dtype=torch.double)
    ktq = k.T @ q
    qtk = ktq.T
    gram = (qtk - qtk @ ktq) / mu
    gram = (gram + gram.T) * 0.5

    # The rank is the edit batch size, so this eigendecomposition is small
    # compared with MEMIT's input-dimensional covariance solve.
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    gram = (eigenvectors * eigenvalues.clamp_min(0.0).unsqueeze(0)) @ eigenvectors.T
    return ((gram + gram.T) * 0.5).detach()


def _unwrap_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        if not isinstance(output[0], torch.Tensor):
            raise TypeError("The first module output is not a tensor")
        return output[0]
    raise TypeError(f"Unsupported module output type: {type(output)!r}")


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError(f"Unsupported module output type: {type(output)!r}")


def add_low_rank_output_update(
    output: Any,
    module_inputs: Tuple[Any, ...],
    fixed_key_factor: torch.Tensor,
    value_factor: torch.Tensor,
) -> Any:
    """Inject ``(X Q) R^T`` into a rewrite module's output."""

    if not module_inputs or not isinstance(module_inputs[0], torch.Tensor):
        raise TypeError("The rewrite module's first input must be a tensor")
    hidden = _unwrap_hidden(output)
    inputs = module_inputs[0]
    if inputs.shape[-1] != fixed_key_factor.shape[0]:
        raise ValueError(
            "Rewrite input/key-factor mismatch: "
            f"input dim {inputs.shape[-1]}, Q dim {fixed_key_factor.shape[0]}"
        )
    if hidden.shape[-1] != value_factor.shape[0]:
        raise ValueError(
            "Rewrite output/value-factor mismatch: "
            f"output dim {hidden.shape[-1]}, R dim {value_factor.shape[0]}"
        )
    if fixed_key_factor.shape[1] != value_factor.shape[1]:
        raise ValueError(
            "Q and R rank mismatch: "
            f"{fixed_key_factor.shape[1]} != {value_factor.shape[1]}"
        )

    # Optimizer state and the low-rank products stay in fp32 even when the
    # frozen transformer runs in bf16.  Only the injected output is cast back.
    x32 = inputs.float()
    q32 = fixed_key_factor.to(device=inputs.device, dtype=torch.float32)
    r32 = value_factor.to(device=inputs.device, dtype=torch.float32)
    delta = (x32 @ q32) @ r32.T
    return _replace_hidden(output, hidden + delta.to(dtype=hidden.dtype))


class _LowRankHookBank(AbstractContextManager):
    def __init__(
        self,
        model: torch.nn.Module,
        module_names: Iterable[str],
        key_factors: Mapping[str, torch.Tensor],
        value_factors: Mapping[str, torch.Tensor],
    ) -> None:
        self.model = model
        self.module_names = list(module_names)
        self.key_factors = key_factors
        self.value_factors = value_factors
        self.handles: List[Any] = []

    def __enter__(self) -> "_LowRankHookBank":
        for name in self.module_names:
            module = nethook.get_module(self.model, name)
            key = self.key_factors[name]
            value = self.value_factors[name]

            def hook(_module, inputs, output, *, q=key, r=value):
                return add_low_rank_output_update(output, inputs, q, r)

            self.handles.append(module.register_forward_hook(hook))
        return self

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
        return None


def _model_input_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            return embeddings.weight.device
    return next(model.parameters()).device


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, MutableMapping):
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
    raise TypeError(f"Tokenizer returned unsupported batch type {type(batch)!r}")


def _prepare_prompt_batch(
    model: torch.nn.Module,
    tok: Any,
    requests: List[Dict[str, Any]],
    fact_token_strategy: str,
) -> Tuple[Any, List[int]]:
    prompts = [request["prompt"].format(request["subject"]) for request in requests]
    lookup_indices = [
        find_fact_lookup_idx(
            request["prompt"],
            request["subject"],
            tok,
            fact_token_strategy,
            verbose=False,
        )
        for request in requests
    ]
    batch = tok(prompts, return_tensors="pt", padding=True)
    batch = _move_batch_to_device(batch, _model_input_device(model))

    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        if any(index < 0 for index in lookup_indices):
            raise ValueError("Negative lookup indices require an attention_mask")
    else:
        left_padding = getattr(tok, "padding_side", "right") == "left"
        padded_width = int(attention_mask.shape[1])
        for row, lookup in enumerate(lookup_indices):
            valid_length = int(attention_mask[row].sum().item())
            unpadded_lookup = lookup if lookup >= 0 else valid_length + lookup
            if not 0 <= unpadded_lookup < valid_length:
                raise IndexError(
                    f"Lookup index {lookup} is outside prompt length {valid_length}"
                )
            left_offset = padded_width - valid_length if left_padding else 0
            lookup_indices[row] = unpadded_lookup + left_offset
    return batch, lookup_indices


def _batch_first_hidden(output: Any, batch_size: int) -> torch.Tensor:
    hidden = _unwrap_hidden(output)
    if hidden.ndim != 3:
        raise ValueError(
            "Expected a rank-3 transformer hidden state; "
            f"got shape {tuple(hidden.shape)}"
        )
    if hidden.shape[0] == batch_size:
        return hidden
    if hidden.shape[1] == batch_size:
        return hidden.transpose(0, 1)
    raise ValueError(
        "Could not locate the batch axis in hidden state shape "
        f"{tuple(hidden.shape)} for batch size {batch_size}"
    )


def _select_lookup_states(
    output: Any,
    lookup_indices: List[int],
) -> torch.Tensor:
    hidden = _batch_first_hidden(output, len(lookup_indices))
    rows = torch.arange(len(lookup_indices), device=hidden.device)
    columns = torch.tensor(lookup_indices, device=hidden.device, dtype=torch.long)
    return hidden[rows, columns]


def _forward_terminal_states(
    model: torch.nn.Module,
    model_inputs: Mapping[str, torch.Tensor],
    terminal_module_name: str,
    lookup_indices: List[int],
) -> torch.Tensor:
    with nethook.Trace(
        model,
        layer=terminal_module_name,
        retain_output=True,
        detach=False,
        stop=True,
    ) as trace:
        model(**model_inputs, use_cache=False)
    return _select_lookup_states(trace.output, lookup_indices)


def _capture_subject_trajectory(
    model: torch.nn.Module,
    model_inputs: Mapping[str, torch.Tensor],
    layer_module_names: List[str],
    lookup_indices: List[int],
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        with nethook.TraceDict(
            model,
            layers=layer_module_names,
            retain_input=True,
            retain_output=True,
            detach=True,
            stop=True,
        ) as traces:
            model(**model_inputs, use_cache=False)

    states = {
        f"input_{layer_module_names[0]}": _select_lookup_states(
            traces[layer_module_names[0]].input,
            lookup_indices,
        ).float().cpu()
    }
    for name in layer_module_names:
        states[f"output_{name}"] = _select_lookup_states(
            traces[name].output,
            lookup_indices,
        ).float().cpu()
    return states


def _capture_committed_subject_trajectory(
    model: torch.nn.Module,
    model_inputs: Mapping[str, torch.Tensor],
    layer_module_names: List[str],
    lookup_indices: List[int],
    weight_names: List[str],
    key_factors: Mapping[str, torch.Tensor],
    value_factors: Mapping[str, torch.Tensor],
    module_names: List[str],
    base_weights: Optional[Mapping[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Measure the path after the exact dtype-aware MEMIT weight merge."""

    restore_weights: Dict[str, torch.Tensor] = {}
    try:
        with torch.no_grad():
            for module_name, weight_name in zip(module_names, weight_names):
                weight = nethook.get_parameter(model, weight_name)
                if base_weights is not None and weight_name in base_weights:
                    base = base_weights[weight_name]
                else:
                    base = weight.detach().clone()
                restore_weights[weight_name] = base
                weight.copy_(base.to(device=weight.device, dtype=weight.dtype))

                key = key_factors[module_name].detach().to(device=weight.device)
                value = value_factors[module_name].detach().to(device=weight.device)
                # Reproduce apply_memit_to_model's multiplication and shape
                # matching order exactly.  This matters in finite precision:
                # (Q R^T)^T and R Q^T need not round identically.
                factor_product = key @ value.T
                if factor_product.shape == weight.shape:
                    stored_update = factor_product
                elif factor_product.T.shape == weight.shape:
                    stored_update = factor_product.T
                else:
                    raise ValueError(
                        "Refined update does not match committed weight shape: "
                        f"update={tuple(factor_product.shape)}, "
                        f"weight={tuple(weight.shape)}"
                    )
                weight.add_(stored_update.float())
                del factor_product, stored_update

        return _capture_subject_trajectory(
            model,
            model_inputs,
            layer_module_names,
            lookup_indices,
        )
    finally:
        with torch.no_grad():
            for weight_name, base in restore_weights.items():
                weight = nethook.get_parameter(model, weight_name)
                weight.copy_(base.to(device=weight.device, dtype=weight.dtype))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _trajectory_summary(
    states: Mapping[str, torch.Tensor],
    target: torch.Tensor,
) -> List[Dict[str, float]]:
    target_cpu = target.detach().float().cpu()
    summary: List[Dict[str, float]] = []
    previous = None
    for name, state in states.items():
        row: Dict[str, float] = {
            "state": name,
            "norm_mean": float(state.norm(dim=-1).mean().item()),
            "target_l2_mean": float((state - target_cpu).norm(dim=-1).mean().item()),
        }
        if previous is not None:
            row["step_l2_mean"] = float(
                (state - previous).norm(dim=-1).mean().item()
            )
        summary.append(row)
        previous = state
    return summary


def _covariance_update_energy(
    value_factor: torch.Tensor,
    covariance_gram: torch.Tensor,
) -> torch.Tensor:
    gram = covariance_gram.to(
        device=value_factor.device,
        dtype=torch.float32,
    )
    value = value_factor.float()
    # The double-precision Gram is PSD-projected at construction time.  Clamp
    # only protects against a tiny negative scalar after its fp32 device cast.
    return torch.einsum("or,rs,os->", value, gram, value).clamp_min(0.0)


def _factor_statistics(
    key_factor: torch.Tensor,
    value_factor: torch.Tensor,
    covariance_gram: torch.Tensor,
    native_value_factor: torch.Tensor,
) -> Dict[str, float]:
    """Return rank-sized diagnostics for one intended low-rank update."""

    value = value_factor.detach().float()
    key = key_factor.detach().to(device=value.device, dtype=torch.float32)
    native_value = native_value_factor.detach().to(
        device=value.device,
        dtype=torch.float32,
    )
    key_gram = key.T @ key
    update_squared_frobenius = torch.einsum(
        "or,rs,os->",
        value,
        key_gram,
        value,
    ).clamp_min(0.0)
    covariance_energy = _covariance_update_energy(
        value,
        covariance_gram,
    )
    return {
        "value_factor_l2": float(value.norm().item()),
        "value_factor_change_from_native_l2": float(
            (value - native_value).norm().item()
        ),
        "intended_update_frobenius": float(
            update_squared_frobenius.sqrt().item()
        ),
        "covariance_update_energy": float(covariance_energy.item()),
    }


def _committed_objective_metrics(
    states: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    target_scales: torch.Tensor,
    value_factors: Mapping[str, torch.Tensor],
    module_names: List[str],
    weight_names: List[str],
    covariance_grams: Mapping[str, torch.Tensor],
    energy_scale: float,
    cost_lambda: float,
) -> Dict[str, float]:
    """Evaluate the objective on a fresh forward of exactly committed weights."""

    terminal = list(states.values())[-1].detach().float().cpu()
    target_cpu = target.detach().float().cpu()
    per_case = (terminal - target_cpu).square().sum(dim=-1)
    update_energy = sum(
        max(
            float(
                _covariance_update_energy(
                    value_factors[module_name],
                    covariance_grams[weight_name],
                ).item()
            ),
            0.0,
        )
        for module_name, weight_name in zip(module_names, weight_names)
    )
    normalized_target = float(
        (per_case / target_scales.detach().float().cpu()).mean().item()
    )
    normalized_energy = update_energy / energy_scale
    return {
        "total_loss": normalized_target + cost_lambda * normalized_energy,
        "terminal_squared_l2": float(per_case.mean().item()),
        "normalized_terminal_loss": normalized_target,
        "covariance_update_energy": update_energy,
        "normalized_update_energy": normalized_energy,
    }


def _append_jsonl(path_value: Optional[str], record: Dict[str, Any]) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _copy_factor_values(
    value_factors: Mapping[str, torch.Tensor],
    names: Iterable[str],
) -> Dict[str, torch.Tensor]:
    return {
        name: value_factors[name].detach().clone()
        for name in names
    }


def _restore_factor_values(
    value_factors: Mapping[str, torch.Tensor],
    values: Mapping[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name, value in values.items():
            value_factors[name].copy_(value)


def refine_memit_value_factors(
    model: torch.nn.Module,
    tok: Any,
    requests: List[Dict[str, Any]],
    hparams: Any,
    target_zs: torch.Tensor,
    deltas: Mapping[str, FactorPair],
    covariance_grams: Mapping[str, torch.Tensor],
    base_weights: Optional[Mapping[str, torch.Tensor]] = None,
) -> Dict[str, FactorPair]:
    """Refine MEMIT ``R_l`` factors jointly or by block-coordinate descent.

    ``model`` must contain the base weights, not native MEMIT's temporarily
    inserted updates.  The returned dictionary preserves MEMIT's public
    ``(Q_l, R_l)`` factor contract and stores both factors on CPU.
    """

    validate_endogenous_pivot_hparams(hparams)
    if not requests:
        raise ValueError("Endogenous-pivot refinement received no requests")
    if (
        bool(getattr(hparams, "endogenous_pivot_capture_trajectory", False))
        and not getattr(hparams, "endogenous_pivot_artifact_dir", None)
    ):
        raise ValueError(
            "endogenous_pivot_capture_trajectory requires "
            "endogenous_pivot_artifact_dir before refinement starts"
        )

    layers = [int(layer) for layer in hparams.layers]
    trainable_layers = _resolve_trainable_layers(hparams, layers)
    weight_names = [
        f"{hparams.rewrite_module_tmp.format(layer)}.weight" for layer in layers
    ]
    module_names = [hparams.rewrite_module_tmp.format(layer) for layer in layers]
    trainable_module_names = [
        hparams.rewrite_module_tmp.format(layer) for layer in trainable_layers
    ]
    layer_module_names = [hparams.layer_module_tmp.format(layer) for layer in layers]
    missing = [name for name in weight_names if name not in deltas]
    if missing:
        raise KeyError(f"Missing native MEMIT factors for {missing}")
    missing_grams = [name for name in weight_names if name not in covariance_grams]
    if missing_grams:
        raise KeyError(f"Missing covariance-cost grams for {missing_grams}")

    model_inputs, lookup_indices = _prepare_prompt_batch(
        model,
        tok,
        requests,
        hparams.fact_token,
    )
    terminal_module_name = layer_module_names[-1]
    batch_size = len(requests)
    if target_zs.ndim != 2:
        raise ValueError(f"target_zs must be rank 2; got {tuple(target_zs.shape)}")
    if target_zs.shape[1] == batch_size:
        target = target_zs.T.detach().float()
    elif target_zs.shape[0] == batch_size:
        target = target_zs.detach().float()
    else:
        raise ValueError(
            "target_zs has no batch axis matching the requests: "
            f"shape={tuple(target_zs.shape)}, batch={batch_size}"
        )

    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    was_training = model.training
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    log_path = getattr(hparams, "endogenous_pivot_log_path", None)
    case_ids = [request.get("case_id") for request in requests]
    call_index = int(getattr(hparams, "_endogenous_pivot_call_index", 0))
    setattr(hparams, "_endogenous_pivot_call_index", call_index + 1)

    try:
        base_states = _capture_subject_trajectory(
            model,
            model_inputs,
            layer_module_names,
            lookup_indices,
        )
        base_terminal = list(base_states.values())[-1]
        target_cpu = target.detach().float().cpu()
        base_target_loss_per_case = (
            (base_terminal - target_cpu).square().sum(dim=-1)
        )
        base_target_loss = base_target_loss_per_case.mean()
        target_scales = base_target_loss_per_case.clamp_min(1e-12)

        fixed_keys: Dict[str, torch.Tensor] = {}
        value_factors: Dict[str, torch.nn.Parameter] = {}
        commit_keys: Dict[str, torch.Tensor] = {}
        native_commit_values: Dict[str, torch.Tensor] = {}
        for module_name, weight_name in zip(module_names, weight_names):
            module = nethook.get_module(model, module_name)
            device = next(module.parameters()).device
            key, value = deltas[weight_name]
            commit_keys[module_name] = key.detach().cpu()
            native_commit_values[module_name] = value.detach().cpu()
            fixed_keys[module_name] = key.detach().to(
                device=device,
                dtype=torch.float32,
            )
            value_factors[module_name] = torch.nn.Parameter(
                value.detach().to(device=device, dtype=torch.float32),
                requires_grad=False,
            )

        native_values = _copy_factor_values(value_factors, module_names)

        native_energy = 0.0
        for module_name, weight_name in zip(module_names, weight_names):
            energy = _covariance_update_energy(
                value_factors[module_name].detach(),
                covariance_grams[weight_name],
            )
            native_energy += max(float(energy.detach().item()), 0.0)
        energy_scale = max(native_energy, 1e-12)

        cost_lambda = float(
            getattr(hparams, "endogenous_pivot_l2_lambda", 1e-2)
        )

        def objective() -> Tuple[torch.Tensor, Dict[str, float]]:
            terminal = _forward_terminal_states(
                model,
                model_inputs,
                terminal_module_name,
                lookup_indices,
            ).float()
            target_on_device = target.to(device=terminal.device, dtype=torch.float32)
            target_loss_per_case = (
                (terminal - target_on_device).square().sum(dim=-1)
            )
            target_loss = target_loss_per_case.mean()
            energy_terms = []
            for module_name, weight_name in zip(module_names, weight_names):
                energy_terms.append(
                    _covariance_update_energy(
                        value_factors[module_name],
                        covariance_grams[weight_name],
                    ).to(target_loss.device)
                )
            update_energy = torch.stack(energy_terms).sum()
            normalized_target = (
                target_loss_per_case
                / target_scales.to(target_loss_per_case.device)
            ).mean()
            normalized_energy = update_energy / energy_scale
            total = normalized_target + cost_lambda * normalized_energy
            return total, {
                "total_loss": float(total.detach().item()),
                "terminal_squared_l2": float(target_loss.detach().item()),
                "normalized_terminal_loss": float(normalized_target.detach().item()),
                "covariance_update_energy": float(update_energy.detach().item()),
                "normalized_update_energy": float(normalized_energy.detach().item()),
            }

        optimization_records: List[Dict[str, Any]] = []
        native_hook_states: Dict[str, torch.Tensor]
        status = "completed"
        with _LowRankHookBank(
            model,
            module_names,
            fixed_keys,
            value_factors,
        ):
            native_hook_states = _capture_subject_trajectory(
                model,
                model_inputs,
                layer_module_names,
                lookup_indices,
            )

            initial_total, initial_metrics = objective()
            if not torch.isfinite(initial_total):
                raise FloatingPointError(
                    "Native MEMIT factors produced a non-finite endogenous-pivot objective"
                )
            initial_metrics.update({"phase": "native", "step": 0})
            optimization_records.append(initial_metrics)

            mode = str(
                getattr(hparams, "endogenous_pivot_mode", "coordinate")
            ).lower()
            num_steps = int(getattr(hparams, "endogenous_pivot_num_steps", 5))
            num_sweeps = int(getattr(hparams, "endogenous_pivot_num_sweeps", 2))
            lr = float(getattr(hparams, "endogenous_pivot_lr", 1e-2))
            grad_clip = float(
                getattr(hparams, "endogenous_pivot_grad_clip_norm", 1.0)
            )

            def optimize_block(
                active_names: List[str],
                *,
                sweep: int,
                layer: Optional[int],
                steps: int,
            ) -> bool:
                for name, factor in value_factors.items():
                    factor.requires_grad_(name in active_names)
                    factor.grad = None
                active = [value_factors[name] for name in active_names]
                optimizer = torch.optim.Adam(
                    active,
                    lr=lr,
                    foreach=False,
                )
                best_values = _copy_factor_values(value_factors, active_names)
                best_loss = float("inf")

                for step in range(steps):
                    optimizer.zero_grad(set_to_none=True)
                    total, metrics = objective()
                    total_value = float(total.detach().item())
                    metrics.update(
                        {
                            "phase": mode,
                            "sweep": sweep,
                            "layer": layer,
                            "step": step,
                        }
                    )
                    optimization_records.append(metrics)
                    if not math.isfinite(total_value):
                        _restore_factor_values(value_factors, best_values)
                        return False
                    if total_value < best_loss:
                        best_loss = total_value
                        best_values = _copy_factor_values(value_factors, active_names)

                    total.backward()
                    if any(
                        factor.grad is None
                        or not bool(torch.isfinite(factor.grad).all().item())
                        for factor in active
                    ):
                        _restore_factor_values(value_factors, best_values)
                        return False
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(active, grad_clip)
                    optimizer.step()

                final_total, final_metrics = objective()
                final_value = float(final_total.detach().item())
                final_metrics.update(
                    {
                        "phase": mode,
                        "sweep": sweep,
                        "layer": layer,
                        "step": steps,
                        "candidate": "post_step",
                    }
                )
                optimization_records.append(final_metrics)
                if math.isfinite(final_value) and final_value < best_loss:
                    best_values = _copy_factor_values(value_factors, active_names)
                elif not math.isfinite(final_value):
                    _restore_factor_values(value_factors, best_values)
                    return False
                _restore_factor_values(value_factors, best_values)
                return True

            if bool((base_target_loss_per_case <= 1e-12).all().item()):
                status = "skipped_base_already_at_target"
            elif mode == "joint":
                if not optimize_block(
                    trainable_module_names,
                    sweep=0,
                    layer=None,
                    steps=num_steps,
                ):
                    status = "nonfinite_rollback"
            else:
                keep_going = True
                for sweep in range(num_sweeps):
                    for layer, module_name in zip(
                        trainable_layers,
                        trainable_module_names,
                    ):
                        if not optimize_block(
                            [module_name],
                            sweep=sweep,
                            layer=layer,
                            steps=num_steps,
                        ):
                            status = "nonfinite_rollback"
                            keep_going = False
                            break
                    if not keep_going:
                        break

            for factor in value_factors.values():
                factor.requires_grad_(False)
                factor.grad = None
            final_total, final_metrics = objective()
            final_metrics.update({"phase": "final"})
            optimization_records.append(final_metrics)
            refined_hook_states = _capture_subject_trajectory(
                model,
                model_inputs,
                layer_module_names,
                lookup_indices,
            )
            candidate_values = _copy_factor_values(value_factors, module_names)

        # The efficient hook objective is algebraically exact in real
        # arithmetic, but a bf16/fp16 commit rounds the *weight elements*, not
        # just the activation delta.  Evaluate both native and refined factors
        # through the exact production merge and never return a factor set that
        # is worse than native MEMIT under that committed objective.
        candidate_commit_values: Dict[str, torch.Tensor] = {}
        for module_name in module_names:
            if (
                module_name not in trainable_module_names
                or status == "skipped_base_already_at_target"
            ):
                candidate_commit_values[module_name] = native_commit_values[
                    module_name
                ]
            else:
                candidate_commit_values[module_name] = candidate_values[
                    module_name
                ].to(
                    dtype=native_commit_values[module_name].dtype,
                    device="cpu",
                )

        native_committed_states = _capture_committed_subject_trajectory(
            model,
            model_inputs,
            layer_module_names,
            lookup_indices,
            weight_names,
            commit_keys,
            native_commit_values,
            module_names,
            base_weights,
        )
        native_committed_metrics = _committed_objective_metrics(
            native_committed_states,
            target_cpu,
            target_scales,
            native_commit_values,
            module_names,
            weight_names,
            covariance_grams,
            energy_scale,
            cost_lambda,
        )
        if not math.isfinite(native_committed_metrics["total_loss"]):
            raise FloatingPointError(
                "Native MEMIT factors produced a non-finite committed objective"
            )

        candidate_committed_states = _capture_committed_subject_trajectory(
            model,
            model_inputs,
            layer_module_names,
            lookup_indices,
            weight_names,
            commit_keys,
            candidate_commit_values,
            module_names,
            base_weights,
        )
        candidate_committed_metrics = _committed_objective_metrics(
            candidate_committed_states,
            target_cpu,
            target_scales,
            candidate_commit_values,
            module_names,
            weight_names,
            covariance_grams,
            energy_scale,
            cost_lambda,
        )
        comparison_tolerance = 1e-7 + 1e-6 * abs(
            native_committed_metrics["total_loss"]
        )
        target_tolerance = 1e-7 + 1e-6 * abs(
            native_committed_metrics["normalized_terminal_loss"]
        )
        candidate_is_safe = (
            math.isfinite(candidate_committed_metrics["total_loss"])
            and math.isfinite(
                candidate_committed_metrics["normalized_terminal_loss"]
            )
            and candidate_committed_metrics["total_loss"]
            < native_committed_metrics["total_loss"] - comparison_tolerance
            and candidate_committed_metrics["normalized_terminal_loss"]
            <= native_committed_metrics["normalized_terminal_loss"]
            + target_tolerance
        )
        if candidate_is_safe:
            selected_factor_set = "refined"
            final_states = candidate_committed_states
            selected_hook_states = refined_hook_states
            selected_commit_values = candidate_commit_values
        else:
            _restore_factor_values(value_factors, native_values)
            selected_factor_set = (
                "native_memit_noop"
                if status == "skipped_base_already_at_target"
                else "native_memit_rollback"
            )
            final_states = native_committed_states
            selected_hook_states = native_hook_states
            selected_commit_values = native_commit_values
            if status != "skipped_base_already_at_target":
                status = "committed_objective_rollback"

        final_committed_metrics = (
            candidate_committed_metrics
            if candidate_is_safe
            else native_committed_metrics
        )
        layer_allocation: List[Dict[str, Any]] = []
        for layer, module_name, weight_name in zip(
            layers,
            module_names,
            weight_names,
        ):
            layer_allocation.append(
                {
                    "layer": layer,
                    "trainable": layer in trainable_layers,
                    "native": _factor_statistics(
                        commit_keys[module_name],
                        native_commit_values[module_name],
                        covariance_grams[weight_name],
                        native_commit_values[module_name],
                    ),
                    "candidate": _factor_statistics(
                        commit_keys[module_name],
                        candidate_commit_values[module_name],
                        covariance_grams[weight_name],
                        native_commit_values[module_name],
                    ),
                    "final": _factor_statistics(
                        commit_keys[module_name],
                        selected_commit_values[module_name],
                        covariance_grams[weight_name],
                        native_commit_values[module_name],
                    ),
                }
            )
        for stage in ("native", "candidate", "final"):
            total_energy = sum(
                row[stage]["covariance_update_energy"]
                for row in layer_allocation
            )
            for row in layer_allocation:
                row[stage]["covariance_energy_fraction"] = (
                    row[stage]["covariance_update_energy"] / total_energy
                    if total_energy > 0.0
                    else 0.0
                )

        record = {
            "method": "MEMIT",
            "feature": "endogenous_pivots",
            "mode": str(getattr(hparams, "endogenous_pivot_mode", "coordinate")),
            "status": status,
            "selected_factor_set": selected_factor_set,
            "call_index": call_index,
            "case_ids": case_ids,
            "layers": layers,
            "trainable_layers": trainable_layers,
            "target_state": f"output_{terminal_module_name}",
            "prompt_scope": "full_canonical_edit_prompt",
            "fixed_factors": "native_memit_adjusted_keys_Q",
            "optimized_factors": "value_factors_R",
            "optimization_forward": "factorized_fp32_activation_hook",
            "selection_forward": "exact_dtype_aware_temporary_weight_commit",
            "target_loss_scale_mean": float(base_target_loss.item()),
            "target_loss_scales_per_case": [
                float(value) for value in base_target_loss_per_case.tolist()
            ],
            "native_update_energy_scale": energy_scale,
            "cost_lambda": cost_lambda,
            "learning_rate": lr,
            "num_steps": num_steps,
            "num_sweeps": num_sweeps,
            "grad_clip_norm": grad_clip,
            "optimization": optimization_records,
            "committed_native_objective": native_committed_metrics,
            "committed_candidate_objective": candidate_committed_metrics,
            "committed_final_objective": final_committed_metrics,
            "layer_allocation": layer_allocation,
            "base_trajectory": _trajectory_summary(base_states, target_cpu),
            "native_hook_trajectory": _trajectory_summary(
                native_hook_states,
                target_cpu,
            ),
            "native_trajectory": _trajectory_summary(
                native_committed_states,
                target_cpu,
            ),
            "refined_hook_trajectory": _trajectory_summary(
                refined_hook_states, target_cpu
            ),
            "candidate_committed_trajectory": _trajectory_summary(
                candidate_committed_states,
                target_cpu,
            ),
            "final_trajectory": _trajectory_summary(final_states, target_cpu),
            "final_trajectory_execution": "temporary_weight_commit",
        }
        native_hook_terminal = list(native_hook_states.values())[-1]
        native_committed_terminal = list(native_committed_states.values())[-1]
        candidate_hook_terminal = list(refined_hook_states.values())[-1]
        candidate_committed_terminal = list(candidate_committed_states.values())[-1]
        selected_hook_terminal = list(selected_hook_states.values())[-1]
        final_terminal = list(final_states.values())[-1]
        record["native_hook_commit_terminal_l2"] = float(
            (native_hook_terminal - native_committed_terminal)
            .norm(dim=-1)
            .mean()
            .item()
        )
        record["candidate_hook_commit_terminal_l2"] = float(
            (candidate_hook_terminal - candidate_committed_terminal)
            .norm(dim=-1)
            .mean()
            .item()
        )
        record["hook_commit_terminal_l2"] = float(
            (selected_hook_terminal - final_terminal).norm(dim=-1).mean().item()
        )
        record["committed_terminal_squared_l2"] = final_committed_metrics[
            "terminal_squared_l2"
        ]

        if bool(getattr(hparams, "endogenous_pivot_capture_trajectory", False)):
            artifact_dir = Path(hparams.endogenous_pivot_artifact_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / f"batch_{call_index:06d}.pt"
            torch.save(
                {
                    "method": "MEMIT",
                    "feature": "endogenous_pivots",
                    "case_ids": case_ids,
                    "layers": layers,
                    "trainable_layers": trainable_layers,
                    "target_zs": target_cpu,
                    "base": base_states,
                    "native_hook": native_hook_states,
                    "native": native_committed_states,
                    "refined_hook": refined_hook_states,
                    "candidate_committed": candidate_committed_states,
                    "committed": final_states,
                    "final": final_states,
                    "selected_factor_set": selected_factor_set,
                },
                artifact_path,
            )
            record["trajectory_artifact"] = str(artifact_path)

        _append_jsonl(log_path, record)
        print(
            "[EndogenousPivot][MEMIT] "
            f"mode={record['mode']} status={status} "
            f"selected={selected_factor_set} "
            f"committed_terminal="
            f"{native_committed_metrics['terminal_squared_l2']:.6e}"
            f"->{final_committed_metrics['terminal_squared_l2']:.6e} "
            f"committed_objective={native_committed_metrics['total_loss']:.6e}"
            f"->{final_committed_metrics['total_loss']:.6e}"
        )

        refined: Dict[str, FactorPair] = {}
        for module_name, weight_name in zip(module_names, weight_names):
            refined[weight_name] = (
                commit_keys[module_name].detach().cpu(),
                selected_commit_values[module_name].detach().cpu(),
            )
        return refined
    finally:
        for parameter, requires_grad in zip(model.parameters(), original_requires_grad):
            parameter.requires_grad_(requires_grad)
        model.train(was_training)
