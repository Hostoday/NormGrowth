"""Strict joint layer-allocation plumbing for MEMIT-style outer writes.

The mathematical allocator lives in :mod:`tangent_layer_allocation`.  This
module adds the tensor-layout and activation-capture conventions needed by an
editor such as AlphaEdit:

* one unexpanded target column per edited fact, shaped ``[H, B]``;
* one clean residual-stream reference at every edited layer, shaped
  ``[B, L, H]``; and
* one precomputed, locally tangent output component per layer, returned in
  the editor-friendly layout ``[L, H, B]``.

The plan is deliberately computed *jointly and once*, before any temporary
layer update is installed.  Replacing it with ``remaining_error / remaining``
inside the layer loop would reintroduce the terminal-layer clean-up behaviour
that this ablation is intended to test.

The first AlphaEdit experiment is restricted to batch size one.  With one
key, AlphaEdit's projected/ridge outer solve can rescale the requested output
component but cannot rotate it.  With multiple facts in one solve, cross-key
mixing can rotate an individual fact's realised write out of its local tangent
space, so the current construction would no longer be a strict intervention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from .tangent_layer_allocation import (
    LocalTangentAllocation,
    allocate_local_tangent_components,
)


@dataclass(frozen=True)
class StrictTangentLayerPlan:
    """A fixed joint decomposition consumed by an editor's layer loop.

    Attributes:
        layer_residuals: Requested output components in ``[L, H, B]`` layout.
        local_references: Clean block-output references in ``[B, L, H]``.
        allocation: Complete output from the mathematical tangent allocator.
        target_norm: Per-fact norm of the requested final-boundary target.
        slack_norm: Per-fact norm that cannot be represented by the active
            local tangent spaces at the configured numerical cutoff.
        relative_slack: ``slack_norm / target_norm`` (with a safe zero-target
            convention).
        reconstruction_error: Norm of
            ``sum_l layer_residuals[l] + slack - target`` for each fact.
        max_abs_tangency_dot: Maximum absolute ``<unit_reference, component>``
            over all facts and layers.
    """

    layer_residuals: torch.Tensor
    local_references: torch.Tensor
    allocation: LocalTangentAllocation
    target_norm: torch.Tensor
    slack_norm: torch.Tensor
    relative_slack: torch.Tensor
    reconstruction_error: torch.Tensor
    max_abs_tangency_dot: torch.Tensor


def collect_clean_local_output_references(
    model: Any,
    tok: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    representation_getter: Callable[..., Any],
    *,
    layers: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Capture subject-token block outputs before an edit is provisionally applied.

    ``representation_getter`` follows the existing AlphaEdit
    ``get_module_input_output_at_words`` interface and must return
    ``(module_input, module_output)``.  The requested module is
    ``hparams.layer_module_tmp`` (the decoder block), *not*
    ``hparams.rewrite_module_tmp`` (the MLP down projection).  The latter has
    a 14336-dimensional key input for Llama-3 and is not in the 4096-dimensional
    residual-output space in which the target and tangent constraint live.

    Args:
        model, tok, requests, hparams: Active editor objects.  Requests must
            already use the editor's single ``{}`` subject placeholder.
        representation_getter: Callable used by the editor to capture module
            input/output representations at the configured fact token.
        layers: Optional explicit edited-layer order.  Defaults to
            ``hparams.layers``.

    Returns:
        Detached tensor shaped ``[B, L, H]``.
    """

    selected_layers = [
        int(layer) for layer in (layers if layers is not None else hparams.layers)
    ]
    if not selected_layers:
        raise ValueError("At least one edited layer is required")
    if not requests:
        raise ValueError("At least one edit request is required")

    prompts = [str(request["prompt"]) for request in requests]
    subjects = [str(request["subject"]) for request in requests]
    if any(prompt.count("{}") != 1 for prompt in prompts):
        raise ValueError(
            "Every request prompt must contain exactly one '{}' placeholder"
        )

    outputs = []
    for layer in selected_layers:
        representations = representation_getter(
            model,
            tok,
            layer,
            context_templates=prompts,
            words=subjects,
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )
        if not isinstance(representations, (tuple, list)) or len(representations) != 2:
            raise TypeError(
                "representation_getter must return (module_input, module_output)"
            )
        layer_output = representations[1]
        if not isinstance(layer_output, torch.Tensor) or layer_output.ndim != 2:
            raise ValueError(
                "Each captured module output must be a [B, H] tensor"
            )
        if layer_output.shape[0] != len(requests):
            raise ValueError(
                "Captured module-output batch does not match requests: "
                f"got {tuple(layer_output.shape)}, requests={len(requests)}"
            )
        outputs.append(layer_output.detach())

    hidden_sizes = {int(output.shape[1]) for output in outputs}
    if len(hidden_sizes) != 1:
        raise ValueError(
            f"All local outputs must share one hidden size, got {hidden_sizes}"
        )
    # Decoder blocks may straddle devices under model parallelism.  Move the
    # small [B, H] references to the terminal edited layer's device before the
    # joint allocation instead of requiring one-device placement.
    common_device = outputs[-1].device
    common_dtype = outputs[-1].dtype
    return torch.stack(
        [output.to(device=common_device, dtype=common_dtype) for output in outputs],
        dim=1,
    )


def build_strict_joint_tangent_plan(
    target_columns: torch.Tensor,
    local_references: torch.Tensor,
    *,
    layer_costs: Optional[torch.Tensor] = None,
    layer_strengths: Optional[torch.Tensor] = None,
    rcond: Optional[float] = None,
    eps: Optional[float] = None,
    require_batch_size_one: bool = True,
    max_relative_slack: Optional[float] = None,
) -> StrictTangentLayerPlan:
    """Build one immutable joint tangent plan before the editor's layer loop.

    Args:
        target_columns: Initial final-boundary error ``z* - current_z`` in the
            native editor layout ``[H, B]``.  These are fact columns, not raw
            prefix/context columns.
        local_references: Clean local block outputs shaped ``[B, L, H]``.
        layer_costs, layer_strengths, rcond, eps: Forwarded to
            :func:`allocate_local_tangent_components`.
        require_batch_size_one: Keep true for the strict AlphaEdit experiment.
        max_relative_slack: Optional fail-fast upper bound per fact.  Slack is
            never assigned radially to the terminal layer; doing that would
            violate strict local tangency.  ``None`` records but accepts it.

    Returns:
        :class:`StrictTangentLayerPlan`.  The caller should use
        ``plan.layer_residuals[layer_index]`` instead of recomputing a
        remaining-layer fraction.
    """

    if target_columns.ndim != 2:
        raise ValueError("target_columns must have shape [H, B]")
    if local_references.ndim != 3:
        raise ValueError("local_references must have shape [B, L, H]")
    hidden_size, batch_size = target_columns.shape
    ref_batch, num_layers, ref_hidden = local_references.shape
    if hidden_size == 0 or batch_size == 0 or num_layers == 0:
        raise ValueError("H, B, and L dimensions must all be non-zero")
    if (ref_batch, ref_hidden) != (batch_size, hidden_size):
        raise ValueError(
            "local_references must match target_columns' B and H: "
            f"targets={tuple(target_columns.shape)}, "
            f"references={tuple(local_references.shape)}"
        )
    if require_batch_size_one and batch_size != 1:
        raise ValueError(
            "Strict AlphaEdit tangent allocation currently requires batch "
            f"size 1; received B={batch_size}. Cross-key mixing in a batched "
            "outer solve can rotate per-fact realised writes."
        )
    if max_relative_slack is not None:
        max_relative_slack = float(max_relative_slack)
        if not 0.0 <= max_relative_slack < float("inf"):
            raise ValueError("max_relative_slack must be finite and non-negative")

    # The outer solve does not backpropagate into z*.  Detaching here avoids
    # retaining all inner-loop activations through five dense linear solves.
    target = target_columns.detach().T
    references = local_references.detach().to(device=target.device)
    allocation = allocate_local_tangent_components(
        target,
        references,
        layer_costs=layer_costs,
        layer_strengths=layer_strengths,
        rcond=rcond,
        eps=eps,
    )
    layer_residuals = allocation.components.permute(1, 2, 0).contiguous()

    target_work = target.to(dtype=allocation.components.dtype)
    target_norm = torch.linalg.vector_norm(target_work, dim=-1)
    slack_norm = torch.linalg.vector_norm(allocation.slack, dim=-1)
    safe_denominator = target_norm.clamp_min(
        torch.finfo(target_norm.dtype).eps
    )
    relative_slack = torch.where(
        target_norm > 0,
        slack_norm / safe_denominator,
        torch.zeros_like(slack_norm),
    )
    reconstruction_error = torch.linalg.vector_norm(
        allocation.projectable_target + allocation.slack - target_work,
        dim=-1,
    )
    max_abs_tangency_dot = allocation.tangency_dot.abs().max()

    if max_relative_slack is not None and bool(
        (relative_slack > max_relative_slack).any()
    ):
        raise RuntimeError(
            "Joint tangent spaces cannot represent enough of the target: "
            f"max relative slack={float(relative_slack.max().item()):.6g}, "
            f"allowed={max_relative_slack:.6g}. Slack was not assigned to "
            "the terminal layer because that would break strict tangency."
        )

    return StrictTangentLayerPlan(
        layer_residuals=layer_residuals,
        local_references=references,
        allocation=allocation,
        target_norm=target_norm,
        slack_norm=slack_norm,
        relative_slack=relative_slack,
        reconstruction_error=reconstruction_error,
        max_abs_tangency_dot=max_abs_tangency_dot,
    )


def tangent_plan_diagnostics(plan: StrictTangentLayerPlan) -> dict[str, Any]:
    """Return JSON-serialisable scalar/per-layer diagnostics for one plan."""

    components = plan.allocation.components.detach().float()
    component_norms = torch.linalg.vector_norm(components, dim=-1)
    reference_norms = torch.linalg.vector_norm(
        plan.local_references.detach().float(), dim=-1
    )
    target_norms = plan.target_norm.detach().float()
    safe_target_norms = target_norms.clamp_min(
        torch.finfo(target_norms.dtype).eps
    )
    component_norm_sum = component_norms.sum(dim=1)
    component_norm_max = component_norms.max(dim=1).values
    component_sum_ratio = torch.where(
        target_norms > 0,
        component_norm_sum / safe_target_norms,
        torch.zeros_like(component_norm_sum),
    )
    component_max_ratio = torch.where(
        target_norms > 0,
        component_norm_max / safe_target_norms,
        torch.zeros_like(component_norm_max),
    )
    eigenvalues = plan.allocation.small_system_eigenvalues.detach().float()
    eigenvalue_scale = plan.allocation.mobility.detach().float().sum(dim=1)
    positive_cutoff = (
        torch.finfo(eigenvalues.dtype).eps
        * eigenvalue_scale.clamp_min(torch.finfo(eigenvalues.dtype).eps)
    ).unsqueeze(1)
    positive = eigenvalues > positive_cutoff
    positive_min = torch.where(
        positive,
        eigenvalues,
        torch.full_like(eigenvalues, float("inf")),
    ).min(dim=1).values
    positive_max = torch.where(
        positive,
        eigenvalues,
        torch.zeros_like(eigenvalues),
    ).max(dim=1).values
    finite_condition = torch.where(
        torch.isfinite(positive_min) & (positive_min > 0),
        positive_max / positive_min,
        torch.full_like(positive_max, float("inf")),
    )
    return {
        "allocation_mode": "strict_joint_local_tangent_jacobian_identity",
        "batch_size": int(components.shape[0]),
        "num_layers": int(components.shape[1]),
        "hidden_size": int(components.shape[2]),
        "target_norm_mean": float(plan.target_norm.float().mean().item()),
        "slack_norm_mean": float(plan.slack_norm.float().mean().item()),
        "relative_slack_max": float(plan.relative_slack.float().max().item()),
        "reconstruction_error_max": float(
            plan.reconstruction_error.float().max().item()
        ),
        "max_abs_tangency_dot": float(
            plan.max_abs_tangency_dot.float().item()
        ),
        # These ratios expose large cancelling layer components when local
        # reference directions are nearly parallel.  We log rather than impose
        # an arbitrary automatic threshold in the first experiment.
        "component_norm_sum_over_target_max": float(
            component_sum_ratio.max().item()
        ),
        "component_norm_max_over_target_max": float(
            component_max_ratio.max().item()
        ),
        "component_norm_mean_by_layer": [
            float(value) for value in component_norms.mean(dim=0).cpu().tolist()
        ],
        "reference_norm_mean_by_layer": [
            float(value) for value in reference_norms.mean(dim=0).cpu().tolist()
        ],
        "common_radial_nullity_max": int(
            plan.allocation.common_radial_nullity.max().item()
        ),
        "small_system_eigenvalue_min_positive": float(
            positive_min.min().item()
        ),
        "small_system_eigenvalue_max": float(positive_max.max().item()),
        "small_system_positive_condition_max": float(
            finite_condition.max().item()
        ),
        "effective_rcond": float(plan.allocation.rcond),
        "effective_eps": float(plan.allocation.eps),
        "transport_assumption": "J_l ~= I",
        "terminal_slack_cleanup": False,
    }
