"""Residual Gain Regularization (RGR) for MEMIT and AlphaEdit.

RGR is deliberately self-contained.  It uses only the current edit's prompts
and the zero-delta forward pass of the current model.  It has no replay buffer,
history direction, held-out probe set, or outer-solve correction.  An opt-in
absolute cosine auxiliary can additionally steer ``cos(H, F)`` toward ``-1``
without replacing or algebraically coupling to the original RGR objective.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .edit_analysis_artifacts import captures_inner_gain


def append_rgr_record(path: Optional[str], record: Mapping[str, Any]) -> None:
    """Append one JSON-serializable optimization record."""

    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, default=str) + "\n")


def exponential_cosine_target_loss(
    current_input: torch.Tensor,
    current_output: torch.Tensor,
    *,
    sharpness: float = 1.0,
    reduction_device: Optional[torch.device] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return an absolute, scale-invariant target loss for ``cos(H, F) -> -1``.

    ``F = H_out - H_in`` and

    ``phi_gamma(c) = expm1(gamma * (c + 1)) / expm1(2 * gamma)``.

    The normalization keeps ``phi_gamma`` in ``[0, 1]`` for
    ``c in [-1, 1]``.  Unlike the reference-relative RGR hinge, this loss is
    absolute: it remains active below the current edit's zero-delta cosine
    and has its unique scalar minimum at ``c = -1``.  Because it depends only
    on cosine, changing ``||F||`` alone cannot directly lower the objective.
    """

    if current_input.ndim != 2 or current_output.ndim != 2:
        raise ValueError("Cosine-target states must have shape [samples, hidden]")
    if current_input.shape != current_output.shape:
        raise ValueError(
            "Cosine-target input/output shapes must match; got "
            f"{tuple(current_input.shape)} and {tuple(current_output.shape)}"
        )
    sharpness = float(sharpness)
    if not math.isfinite(sharpness) or not 0.0 < sharpness <= 20.0:
        raise ValueError(
            "RGR cosine auxiliary sharpness must be finite and in (0, 20], "
            f"got {sharpness}"
        )

    current_input = current_input.float()
    current_write = current_output.float() - current_input
    cosine = torch.nn.functional.cosine_similarity(
        current_input,
        current_write,
        dim=-1,
        eps=eps,
    ).clamp(min=-1.0, max=1.0)
    denominator = math.expm1(2.0 * sharpness)
    penalty = torch.expm1(sharpness * (cosine + 1.0)) / denominator
    loss = penalty.mean()
    if reduction_device is not None:
        loss = loss.to(reduction_device)

    detached_cosine = cosine.detach()
    detached_penalty = penalty.detach()
    slope = (
        sharpness
        * torch.exp(sharpness * (detached_cosine + 1.0))
        / denominator
    )
    return loss, {
        "cosine_aux_sharpness": sharpness,
        "cosine_aux_cosine_mean": float(detached_cosine.mean().item()),
        "cosine_aux_cosine_min": float(detached_cosine.min().item()),
        "cosine_aux_cosine_max": float(detached_cosine.max().item()),
        "cosine_aux_penalty_mean": float(detached_penalty.mean().item()),
        "cosine_aux_penalty_max": float(detached_penalty.max().item()),
        "cosine_aux_slope_mean": float(slope.mean().item()),
        "cosine_aux_fraction_below_minus_0_9": float(
            (detached_cosine <= -0.9).float().mean().item()
        ),
        "cosine_aux_fraction_below_minus_0_99": float(
            (detached_cosine <= -0.99).float().mean().item()
        ),
    }


def residual_gain_loss(
    current_input: torch.Tensor,
    reference_input: torch.Tensor,
    current_output: torch.Tensor,
    reference_output: torch.Tensor,
    *,
    margin: float = 0.0,
    loss_type: str = "positive_l1",
    objective: str = "gain",
    alignment_weight: float = 1.0,
    target_ratio: float = 1.0,
    reduction_device: Optional[torch.device] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float | str]]:
    """Penalize excess block-level gain or state--write alignment.

    For one decoder block ``l``,

    ``g_l = (mean(H_(l+1)^2) - mean(H_l^2)) / (mean(H_l^2) + eps)``.

    ``reference_*`` is the zero-delta forward at the start of the current
    ``compute_z`` call.  The default gain objective is

    ``mean(relu(g_l(delta) - g_l(0) - margin))``.

    Opt-in objectives reuse the same current-edit reference:

    - ``write_relative_energy`` controls
      ``||F||^2 / ||H||^2 = r^2``;
    - ``cosine_drift`` controls ``cos(H, F)``;
    - ``alignment_drift`` controls ``2 <H,F> / ||H||^2``.
    - ``cosine_amplified_gain`` controls
      ``||F||^2/||H||^2 + alignment_weight * 2<H,F>/||H||^2``.
    - ``shifted_cosine_energy`` controls
      ``r^2 + 2 r (cos(H,F) + 1)``, where ``r = ||F||/||H||``.
    - ``rho_minus_one`` controls ``||H_out|| / ||H_in|| - 1``;
    - ``log_rho`` controls ``log(||H_out|| / ||H_in||)``.
    - ``tangential_deficit`` is the only *floor*: it penalises the amount by
      which the relative tangential write energy ``||P_u F||^2 / ||H||^2``
      falls **below** the zero-delta reference.  Every other objective is a
      ceiling on how much some quantity may grow.  The tangential component is
      the only part of the write that rotates the residual direction, i.e. the
      only part the next block can see through RMSNorm, so a collapsing block
      (``cos(H,F) -> 1``, tangential energy -> 0) is exactly what this floor
      forbids.  Internally the sign is flipped so the shared
      ``relu(current - reference - margin)`` hinge applies unchanged; the
      reported ``*_tangential_energy_mean`` diagnostics stay positive.

    The cosine-amplified expression is a gain surrogate.  Since
    ``2<H,F>/||H||^2 = 2 ||F||/||H|| cos(H,F)``, a weight greater than one
    amplifies the input--write cosine contribution without adding a separate
    direction-preservation loss.  ``alignment_weight=1`` exactly recovers the
    ordinary finite residual gain.

    The shifted-cosine energy is nonnegative, is lower-bounded by ``r^2``, and
    for fixed nonzero ``r`` is minimized at ``cos(H,F)=-1``.  It therefore
    cannot make an arbitrarily large write look safe solely through a negative
    alignment term.

    Here ``F = H_(l+1) - H_l`` is the complete decoder-block residual write.
    For ``rho_minus_one`` only, ``target_ratio=R>1`` changes the one-sided
    ceiling to ``current_rho <= R * reference_rho``.  The exact ``R == 1``
    branch deliberately retains the historical subtraction order instead of
    using an algebraically equivalent rewrite.

    Defaults remain exactly backward compatible with the original RGR loss.
    """

    tensors = (current_input, reference_input, current_output, reference_output)
    if any(value.ndim != 2 for value in tensors):
        raise ValueError("RGR states must all have shape [samples, hidden]")
    if not (
        current_input.shape
        == reference_input.shape
        == current_output.shape
        == reference_output.shape
    ):
        raise ValueError(
            "RGR states must have matching shapes; got "
            f"{tuple(current_input.shape)}, {tuple(reference_input.shape)}, "
            f"{tuple(current_output.shape)}, {tuple(reference_output.shape)}"
        )
    if margin < 0:
        raise ValueError(f"RGR margin must be nonnegative, got {margin}")
    normalized_type = str(loss_type).lower()
    if normalized_type not in {"positive_l1", "positive_squared", "absolute_l1"}:
        raise ValueError(
            "RGR loss_type must be positive_l1, positive_squared, or "
            f"absolute_l1; got {loss_type!r}"
        )
    normalized_objective = str(objective).lower()
    if normalized_objective not in {
        "gain",
        "delta_relative_energy",
        "write_relative_energy",
        "cosine_drift",
        "alignment_drift",
        "cosine_amplified_gain",
        "shifted_cosine_energy",
        "rho_minus_one",
        "log_rho",
        "norm_difference",
        "write_energy",
        "tangential_deficit",
    }:
        raise ValueError(
            "RGR objective must be gain, write_relative_energy, "
            "cosine_drift, alignment_drift, cosine_amplified_gain, "
            "shifted_cosine_energy, rho_minus_one, log_rho, "
            "norm_difference, write_energy, or tangential_deficit; "
            f"got {objective!r}"
        )
    target_ratio = float(target_ratio)
    if not math.isfinite(target_ratio) or target_ratio < 1.0:
        raise ValueError(
            "RGR target_ratio must be finite and >= 1, got "
            f"{target_ratio}"
        )
    if target_ratio != 1.0 and normalized_objective != "rho_minus_one":
        raise ValueError(
            "RGR target_ratio != 1 is defined only for the "
            f"rho_minus_one objective, got {normalized_objective!r}"
        )
    if target_ratio != 1.0 and normalized_type == "absolute_l1":
        raise ValueError(
            "RGR rho target_ratio is a one-sided ceiling and cannot be "
            "combined with absolute_l1"
        )
    alignment_weight = float(alignment_weight)
    if not math.isfinite(alignment_weight) or alignment_weight < 0.0:
        raise ValueError(
            "RGR alignment_weight must be finite and nonnegative, got "
            f"{alignment_weight}"
        )
    if normalized_objective == "cosine_amplified_gain" and alignment_weight < 1.0:
        raise ValueError(
            "cosine_amplified_gain requires alignment_weight >= 1, got "
            f"{alignment_weight}"
        )

    current_input = current_input.float()
    current_output = current_output.float()
    reference_input = reference_input.to(current_input).float().detach()
    reference_output = reference_output.to(current_output).float().detach()

    current_input_energy = current_input.square().mean(dim=-1)
    current_output_energy = current_output.square().mean(dim=-1)
    reference_input_energy = reference_input.square().mean(dim=-1)
    reference_output_energy = reference_output.square().mean(dim=-1)

    current_gain = (
        current_output_energy - current_input_energy
    ) / current_input_energy.clamp_min(eps)
    reference_gain = (
        reference_output_energy - reference_input_energy
    ) / reference_input_energy.clamp_min(eps)
    excess_gain = current_gain - reference_gain
    current_rho = torch.sqrt(
        current_output_energy.clamp_min(eps)
        / current_input_energy.clamp_min(eps)
    )
    reference_rho = torch.sqrt(
        reference_output_energy.clamp_min(eps)
        / reference_input_energy.clamp_min(eps)
    )
    current_rho_minus_one = current_rho - 1.0
    reference_rho_minus_one = reference_rho - 1.0
    current_log_rho = 0.5 * (
        torch.log(current_output_energy.clamp_min(eps))
        - torch.log(current_input_energy.clamp_min(eps))
    )
    reference_log_rho = 0.5 * (
        torch.log(reference_output_energy.clamp_min(eps))
        - torch.log(reference_input_energy.clamp_min(eps))
    )
    excess_rho_minus_one = current_rho_minus_one - reference_rho_minus_one
    excess_log_rho = current_log_rho - reference_log_rho
    if target_ratio == 1.0:
        # Exact legacy arithmetic branch.  Do not replace this with
        # current_rho - reference_rho: bit-level regression depends on the
        # historical (rho - 1) subtraction order.
        rho_ceiling_excess = excess_rho_minus_one
    else:
        rho_ceiling_excess = current_rho - target_ratio * reference_rho
    rho_positive_l1 = torch.relu(rho_ceiling_excess - float(margin))
    log_rho_positive_l1 = torch.relu(excess_log_rho - float(margin))

    current_write = current_output - current_input
    reference_write = reference_output - reference_input
    current_write_relative_energy = (
        current_write.square().mean(dim=-1)
        / current_input_energy.clamp_min(eps)
    )
    reference_write_relative_energy = (
        reference_write.square().mean(dim=-1)
        / reference_input_energy.clamp_min(eps)
    )
    # --- 정규화하지 않은 두 변형 -------------------------------------------
    # gain / write_relative_energy 는 E(H_in) 으로 나누므로 hidden 차원 d 가
    # 소거되어 mean 이든 sum 이든 값이 같다.  아래 두 objective 는 나누지
    # 않으므로 규약을 명시해야 한다.  여기서는 진단 CSV 의 ``rms_l2`` 와 같은
    # L2 규약, 즉 d 로 나누지 않은 제곱 L2 노름을 쓴다.
    #
    #   norm_difference : ||H_out||^2 - ||H_in||^2     제곱한 뒤 뺀다
    #   write_energy    : ||H_out - H_in||^2 = ||F||^2 빼고 나서 제곱한다
    #
    # 기존 gain 계열의 계산 경로는 건드리지 않는다.
    current_output_sq_l2 = current_output.square().sum(dim=-1)
    current_input_sq_l2 = current_input.square().sum(dim=-1)
    reference_output_sq_l2 = reference_output.square().sum(dim=-1)
    reference_input_sq_l2 = reference_input.square().sum(dim=-1)
    # Lightweight, objective-independent state diagnostics.  These scalars
    # make different regularizers comparable by their *achieved* block-output
    # non-expansion, rather than by nominal lambda or by objective-specific
    # loss units.  In the subject_last transition at the write layer,
    # ``output_state_delta`` is also the optimized latent delta.  At downstream
    # transitions it should be read only as the transported state change.
    current_input_l2 = torch.sqrt(current_input_sq_l2.clamp_min(eps))
    current_output_l2 = torch.sqrt(current_output_sq_l2.clamp_min(eps))
    reference_input_l2 = torch.sqrt(reference_input_sq_l2.clamp_min(eps))
    reference_output_l2 = torch.sqrt(reference_output_sq_l2.clamp_min(eps))
    input_state_delta_l2 = torch.linalg.vector_norm(
        current_input - reference_input,
        dim=-1,
    )
    output_state_delta_l2 = torch.linalg.vector_norm(
        current_output - reference_output,
        dim=-1,
    )
    output_to_reference_norm_ratio = (
        current_output_l2 / reference_output_l2.clamp_min(math.sqrt(eps))
    )
    output_norm_positive_relative_excess = torch.relu(
        output_to_reference_norm_ratio - 1.0
    )
    output_norm_positive_l2_excess = torch.relu(
        current_output_l2 - reference_output_l2
    )
    current_norm_difference = current_output_sq_l2 - current_input_sq_l2
    reference_norm_difference = reference_output_sq_l2 - reference_input_sq_l2
    current_write_energy = current_write.square().sum(dim=-1)
    reference_write_energy = reference_write.square().sum(dim=-1)

    current_alignment = (
        2.0
        * (current_input * current_write).mean(dim=-1)
        / current_input_energy.clamp_min(eps)
    )
    reference_alignment = (
        2.0
        * (reference_input * reference_write).mean(dim=-1)
        / reference_input_energy.clamp_min(eps)
    )
    current_cosine = torch.nn.functional.cosine_similarity(
        current_input, current_write, dim=-1, eps=eps
    )
    reference_cosine = torch.nn.functional.cosine_similarity(
        reference_input, reference_write, dim=-1, eps=eps
    )
    current_cosine_amplified_gain = (
        current_write_relative_energy + alignment_weight * current_alignment
    )
    reference_cosine_amplified_gain = (
        reference_write_relative_energy + alignment_weight * reference_alignment
    )
    # Compute r from vector norms rather than sqrt(r^2).  The direct norm has
    # a finite zero subgradient when F=0, whereas differentiating sqrt at zero
    # can produce an unstable 0 * inf path in the inner optimization.
    norm_floor = math.sqrt(float(current_input.shape[-1]) * eps)
    current_write_to_input_norm_ratio = (
        torch.linalg.vector_norm(current_write, dim=-1)
        / torch.linalg.vector_norm(current_input, dim=-1).clamp_min(norm_floor)
    )
    reference_write_to_input_norm_ratio = (
        torch.linalg.vector_norm(reference_write, dim=-1)
        / torch.linalg.vector_norm(reference_input, dim=-1).clamp_min(norm_floor)
    )
    current_shifted_cosine_energy = (
        current_write_relative_energy
        + 2.0
        * current_write_to_input_norm_ratio
        * (current_cosine + 1.0)
    )
    reference_shifted_cosine_energy = (
        reference_write_relative_energy
        + 2.0
        * reference_write_to_input_norm_ratio
        * (reference_cosine + 1.0)
    )
    # --- 접선 성분 ---------------------------------------------------------
    # F 를 H 방향과 그에 수직인 부분으로 나누면 (Pythagoras)
    #
    #   rho^2 = alpha^2 + tang ,   alpha = <H,F>/||H||^2 ,
    #   tang  = ||P_u F||^2 / ||H||^2 .
    #
    # alignment = 2*alpha 이므로 tang 은 이미 계산된 두 양에서 바로 나온다.
    # 새 텐서 연산 없이 gain 계열과 정확히 같은 정규화 규약을 쓴다.
    current_tangential_energy = torch.relu(
        current_write_relative_energy - (current_alignment / 2.0).square()
    )
    reference_tangential_energy = torch.relu(
        reference_write_relative_energy - (reference_alignment / 2.0).square()
    )
    # 접선은 방향을 바꾸는 유일한 성분이므로 상한이 아니라 *하한* 을 원한다.
    # 부호를 뒤집어 두면 공통 `[current - reference - margin]_+` 힌지가
    # 그대로 "참조보다 접선이 줄어든 만큼" 을 벌점화한다.
    current_tangential_deficit = -current_tangential_energy
    reference_tangential_deficit = -reference_tangential_energy

    # --- 순수 이차항 통제 (reviewer 요구 ablation) ---------------------------
    # RGR main( subject_last@L8 )의 excess 는 정확히
    #     e = ( ||delta||^2 + 2<H9_ref, delta> ) / ||H8_ref||^2
    # 이다.  이 objective 는 그 중 **이차항만** 남긴다:
    #     [ ||delta||^2 / ||H8_ref||^2 ]_+
    # delta 는 인자로 들어오지 않지만, delta 가 block-l 출력에 더해지므로
    # current_output - reference_output == delta 가 정확히 성립한다
    # (H_in 은 delta 의 상류라 불변).  reference 시점에는 delta = 0 이므로
    # reference 값은 정확히 0 이고, 공통 힌지 [current - reference]_+ 가
    # 그대로 [ ||delta||^2 / ||H8_ref||^2 ]_+ 가 된다.
    #
    # write_relative_energy 와의 차이: 그쪽 drift 는
    #     ( ||delta||^2 + 2<F8_ref, delta> ) / ||H8_ref||^2
    # 로 교차항 2<F8_ref, delta> 를 포함한다.  r0 ~ 0.53-0.72 이므로 이 항은
    # 무시할 수 없다.  따라서 write_relative_energy 는 순수 이차항 통제가
    # 아니며, 이 objective 만이 "delta 를 그냥 작게 만든 것"과 RGR 을 분리한다.
    current_delta_relative_energy = (
        (current_output - reference_output).square().mean(dim=-1)
        / reference_input_energy.clamp_min(eps)
    )
    reference_delta_relative_energy = torch.zeros_like(
        current_delta_relative_energy
    )

    objective_pairs = {
        "delta_relative_energy": (
            current_delta_relative_energy,
            reference_delta_relative_energy,
        ),
        "tangential_deficit": (
            current_tangential_deficit,
            reference_tangential_deficit,
        ),
        "gain": (current_gain, reference_gain),
        "norm_difference": (current_norm_difference, reference_norm_difference),
        "write_energy": (current_write_energy, reference_write_energy),
        "write_relative_energy": (
            current_write_relative_energy,
            reference_write_relative_energy,
        ),
        "cosine_drift": (current_cosine, reference_cosine),
        "alignment_drift": (current_alignment, reference_alignment),
        "cosine_amplified_gain": (
            current_cosine_amplified_gain,
            reference_cosine_amplified_gain,
        ),
        "shifted_cosine_energy": (
            current_shifted_cosine_energy,
            reference_shifted_cosine_energy,
        ),
        "rho_minus_one": (current_rho_minus_one, reference_rho_minus_one),
        "log_rho": (current_log_rho, reference_log_rho),
    }
    current_objective, reference_objective = objective_pairs[
        normalized_objective
    ]
    excess_objective = current_objective - reference_objective
    if normalized_objective == "rho_minus_one" and target_ratio != 1.0:
        ceiling_excess_objective = rho_ceiling_excess
    else:
        # This is the exact historical path, including for rho at R=1.
        ceiling_excess_objective = excess_objective
    if normalized_type == "absolute_l1":
        violation = torch.relu(
            ceiling_excess_objective.abs() - float(margin)
        )
        loss = violation.mean()
    else:
        violation = torch.relu(ceiling_excess_objective - float(margin))
        loss = (
            violation.square().mean()
            if normalized_type == "positive_squared"
            else violation.mean()
        )

    if reduction_device is not None:
        loss = loss.to(reduction_device)

    return loss, {
        "loss_type": normalized_type,
        "objective": normalized_objective,
        "alignment_weight": alignment_weight,
        "target_ratio": target_ratio,
        "margin": float(margin),
        "reference_objective_mean": float(reference_objective.mean().item()),
        "current_objective_mean": float(
            current_objective.detach().mean().item()
        ),
        "excess_objective_mean": float(
            excess_objective.detach().mean().item()
        ),
        "reference_gain_mean": float(reference_gain.mean().item()),
        "current_gain_mean": float(current_gain.detach().mean().item()),
        "excess_gain_mean": float(excess_gain.detach().mean().item()),
        "excess_gain_max": float(excess_gain.detach().max().item()),
        "reference_rho_mean": float(reference_rho.mean().item()),
        "current_rho_mean": float(current_rho.detach().mean().item()),
        "reference_rho_minus_one_mean": float(
            reference_rho_minus_one.mean().item()
        ),
        "current_rho_minus_one_mean": float(
            current_rho_minus_one.detach().mean().item()
        ),
        "excess_rho_minus_one_mean": float(
            excess_rho_minus_one.detach().mean().item()
        ),
        "rho_positive_l1_loss": float(rho_positive_l1.detach().mean().item()),
        "reference_log_rho_mean": float(reference_log_rho.mean().item()),
        "current_log_rho_mean": float(current_log_rho.detach().mean().item()),
        "excess_log_rho_mean": float(excess_log_rho.detach().mean().item()),
        "log_rho_positive_l1_loss": float(
            log_rho_positive_l1.detach().mean().item()
        ),
        "violation_mean": float(violation.detach().mean().item()),
        "violation_max": float(violation.detach().max().item()),
        "violation_fraction": float(
            (violation.detach() > 0).float().mean().item()
        ),
        "reference_input_l2_mean": float(reference_input_l2.mean().item()),
        "current_input_l2_mean": float(
            current_input_l2.detach().mean().item()
        ),
        "reference_output_l2_mean": float(reference_output_l2.mean().item()),
        "current_output_l2_mean": float(
            current_output_l2.detach().mean().item()
        ),
        "input_state_delta_l2_mean": float(
            input_state_delta_l2.detach().mean().item()
        ),
        "output_state_delta_l2_mean": float(
            output_state_delta_l2.detach().mean().item()
        ),
        "output_to_reference_norm_ratio_mean": float(
            output_to_reference_norm_ratio.detach().mean().item()
        ),
        "output_norm_positive_relative_excess_mean": float(
            output_norm_positive_relative_excess.detach().mean().item()
        ),
        "output_norm_positive_l2_excess_mean": float(
            output_norm_positive_l2_excess.detach().mean().item()
        ),
        "reference_tangential_energy_mean": float(
            reference_tangential_energy.mean().item()
        ),
        "current_tangential_energy_mean": float(
            current_tangential_energy.detach().mean().item()
        ),
        "tangential_shortfall_mean": float(
            torch.relu(
                reference_tangential_energy - current_tangential_energy.detach()
            ).mean().item()
        ),
        "reference_write_relative_energy_mean": float(
            reference_write_relative_energy.mean().item()
        ),
        "current_write_relative_energy_mean": float(
            current_write_relative_energy.detach().mean().item()
        ),
        "reference_alignment_contribution_mean": float(
            reference_alignment.mean().item()
        ),
        "current_alignment_contribution_mean": float(
            current_alignment.detach().mean().item()
        ),
        "reference_input_write_cosine_mean": float(
            reference_cosine.mean().item()
        ),
        "current_input_write_cosine_mean": float(
            current_cosine.detach().mean().item()
        ),
        "reference_cosine_amplified_gain_mean": float(
            reference_cosine_amplified_gain.mean().item()
        ),
        "current_cosine_amplified_gain_mean": float(
            current_cosine_amplified_gain.detach().mean().item()
        ),
        "reference_write_to_input_norm_ratio_mean": float(
            reference_write_to_input_norm_ratio.mean().item()
        ),
        "current_write_to_input_norm_ratio_mean": float(
            current_write_to_input_norm_ratio.detach().mean().item()
        ),
        "reference_shifted_cosine_energy_mean": float(
            reference_shifted_cosine_energy.mean().item()
        ),
        "current_shifted_cosine_energy_mean": float(
            current_shifted_cosine_energy.detach().mean().item()
        ),
    }


def _active_token_position(
    attention_mask: torch.Tensor,
    batch_index: int,
    unpadded_index: int,
    state_device: torch.device,
) -> torch.Tensor:
    active = torch.nonzero(attention_mask[batch_index], as_tuple=False).flatten()
    if active.numel() == 0:
        raise ValueError("attention mask row contains no active token")
    if unpadded_index < 0:
        unpadded_index += int(active.numel())
    if not 0 <= unpadded_index < int(active.numel()):
        raise IndexError(
            f"token index {unpadded_index} is outside unpadded length "
            f"{int(active.numel())}"
        )
    return active.to(state_device)[unpadded_index]


class ResidualGainRegularizer:
    """Current-edit, zero-delta-referenced RGR objective."""

    def __init__(
        self,
        *,
        hparams: Any,
        write_layer: int,
        num_hidden_layers: int,
        attention_mask: torch.Tensor,
        rewrite_batch_indices: Sequence[int],
        subject_last_indices: Sequence[int],
        prompt_last_indices: Sequence[int],
        reduction_device: torch.device,
    ) -> None:
        self.optimization_enabled = bool(
            getattr(
                hparams,
                "residual_gain_regularization",
                getattr(hparams, "residual_energy_gain_regularization", False),
            )
        )
        self.record_only = bool(captures_inner_gain(hparams)) and not bool(
            self.optimization_enabled
        )
        self.enabled = bool(self.optimization_enabled or self.record_only)
        self.lambda_ = float(
            getattr(
                hparams,
                "residual_gain_lambda",
                getattr(hparams, "residual_energy_gain_lambda", 0.1),
            )
        )
        self.margin = float(
            getattr(
                hparams,
                "residual_gain_margin",
                getattr(hparams, "residual_energy_gain_margin", 0.0),
            )
        )
        self.loss_type = str(
            getattr(
                hparams,
                "residual_gain_loss_type",
                getattr(hparams, "residual_energy_gain_loss_type", "positive_l1"),
            )
        ).lower()
        self.objective = str(
            getattr(hparams, "residual_gain_objective", "gain")
        ).lower()
        self.alignment_weight = float(
            getattr(hparams, "residual_gain_alignment_weight", 1.0)
        )
        self.target_ratio = float(
            getattr(hparams, "residual_gain_target_ratio", 1.0)
        )
        self.cosine_aux_lambda = float(
            getattr(hparams, "residual_gain_cosine_aux_lambda", 0.0)
        )
        self.cosine_aux_sharpness = float(
            getattr(hparams, "residual_gain_cosine_aux_sharpness", 1.0)
        )
        self.efficacy_threshold = float(
            getattr(hparams, "residual_gain_efficacy_threshold", 0.0)
        )
        self.select_best = bool(
            getattr(
                hparams,
                "residual_gain_select_best",
                getattr(hparams, "residual_energy_gain_select_best", False),
            )
        )
        if not self.optimization_enabled:
            self.select_best = False
        self.selection_threshold = float(
            getattr(
                hparams,
                "residual_gain_selection_threshold",
                getattr(
                    hparams,
                    "residual_energy_gain_selection_threshold",
                    0.5,
                ),
            )
        )
        self.log_path = getattr(hparams, "residual_gain_log_path", None)
        self.reduction_device = reduction_device
        self.attention_mask = attention_mask
        self.rewrite_batch_indices = [int(value) for value in rewrite_batch_indices]
        self.subject_last_indices = [int(value) for value in subject_last_indices]
        self.prompt_last_indices = [int(value) for value in prompt_last_indices]

        subject_layers = getattr(hparams, "residual_gain_subject_layers", None)
        prompt_layers = getattr(hparams, "residual_gain_prompt_layers", None)
        if self.record_only:
            subject_layers = getattr(
                hparams,
                "analysis_gain_subject_layers",
                [int(write_layer)],
            )
            prompt_layers = getattr(
                hparams,
                "analysis_gain_prompt_layers",
                [],
            )
        if subject_layers is None:
            subject_layers = getattr(
                hparams, "residual_energy_gain_subject_layers", None
            )
        if prompt_layers is None:
            prompt_layers = getattr(
                hparams, "residual_energy_gain_prompt_layers", None
            )
        configured_layers = getattr(hparams, "residual_gain_layers", None)
        if configured_layers is None:
            configured_layers = getattr(
                hparams, "residual_energy_gain_layers", None
            )
        configured_scope = str(
            getattr(
                hparams,
                "residual_gain_token_scope",
                getattr(
                    hparams,
                    "residual_energy_gain_token_scope",
                    "subject_last",
                ),
            )
        ).lower()

        self.paired_mode = subject_layers is not None or prompt_layers is not None
        self.position_layers: Dict[str, List[int]] = {}
        if self.paired_mode:
            if subject_layers:
                self.position_layers["subject_last"] = [
                    int(value) for value in subject_layers
                ]
            if prompt_layers:
                self.position_layers["prompt_last"] = [
                    int(value) for value in prompt_layers
                ]
        else:
            layers = (
                [int(write_layer)]
                if configured_layers is None
                else [int(value) for value in configured_layers]
            )
            if configured_scope in {"subject_last", "both"}:
                self.position_layers["subject_last"] = list(layers)
            if configured_scope in {"prompt_last", "both"}:
                self.position_layers["prompt_last"] = list(layers)

        subject_lambda = getattr(hparams, "residual_gain_subject_lambda", None)
        prompt_lambda = getattr(hparams, "residual_gain_prompt_lambda", None)
        if subject_lambda is None:
            subject_lambda = getattr(
                hparams, "residual_energy_gain_subject_lambda", None
            )
        if prompt_lambda is None:
            prompt_lambda = getattr(
                hparams, "residual_energy_gain_prompt_lambda", None
            )
        self.position_lambdas = {
            "subject_last": (
                self.lambda_ if subject_lambda is None else float(subject_lambda)
            ),
            "prompt_last": (
                self.lambda_ if prompt_lambda is None else float(prompt_lambda)
            ),
        }
        self.layers = sorted(
            {layer for layers in self.position_layers.values() for layer in layers}
        )
        self.token_scope = (
            "causal_pairs"
            if self.paired_mode and len(self.position_layers) > 1
            else next(iter(self.position_layers), configured_scope)
        )
        self._reference: Dict[
            Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]
        ] = {}

        if not self.enabled:
            return
        if self.lambda_ < 0 or any(value < 0 for value in self.position_lambdas.values()):
            raise ValueError("RGR lambdas must be nonnegative")
        if self.margin < 0:
            raise ValueError("RGR margin must be nonnegative")
        if self.objective not in {
            "gain",
            "delta_relative_energy",
            "write_relative_energy",
            "cosine_drift",
            "alignment_drift",
            "cosine_amplified_gain",
            "shifted_cosine_energy",
            "rho_minus_one",
            "log_rho",
            "norm_difference",
            "write_energy",
            "tangential_deficit",
        }:
            raise ValueError(
                "RGR objective must be gain, delta_relative_energy, "
                "write_relative_energy, cosine_drift, alignment_drift, "
                "cosine_amplified_gain, shifted_cosine_energy, rho_minus_one, "
                "log_rho, norm_difference, write_energy, or "
                "tangential_deficit; got "
                f"{self.objective!r}"
            )
        if not math.isfinite(self.alignment_weight) or self.alignment_weight < 0.0:
            raise ValueError(
                "RGR alignment weight must be finite and nonnegative"
            )
        if not math.isfinite(self.target_ratio) or self.target_ratio < 1.0:
            raise ValueError(
                "RGR target ratio must be finite and >= 1, got "
                f"{self.target_ratio}"
            )
        if self.target_ratio != 1.0 and self.objective != "rho_minus_one":
            raise ValueError(
                "residual_gain_target_ratio != 1 requires "
                "residual_gain_objective='rho_minus_one'"
            )
        if self.target_ratio != 1.0 and self.loss_type == "absolute_l1":
            raise ValueError(
                "residual_gain_target_ratio is a one-sided ceiling and "
                "cannot be combined with absolute_l1"
            )
        if not math.isfinite(self.cosine_aux_lambda) or self.cosine_aux_lambda < 0.0:
            raise ValueError(
                "RGR cosine auxiliary lambda must be finite and nonnegative"
            )
        if (
            not math.isfinite(self.cosine_aux_sharpness)
            or not 0.0 < self.cosine_aux_sharpness <= 20.0
        ):
            raise ValueError(
                "RGR cosine auxiliary sharpness must be finite and in (0, 20]"
            )
        if (
            self.objective == "cosine_amplified_gain"
            and self.alignment_weight < 1.0
        ):
            raise ValueError(
                "cosine_amplified_gain requires residual_gain_alignment_weight >= 1"
            )
        if not 0.0 <= self.efficacy_threshold <= 1.0:
            raise ValueError("RGR efficacy threshold must be in [0, 1]")
        if not 0.0 <= self.selection_threshold <= 1.0:
            raise ValueError("RGR selection threshold must be in [0, 1]")
        if not self.position_layers:
            raise ValueError("RGR requires at least one layer/token position")
        if len(self.rewrite_batch_indices) != len(self.subject_last_indices):
            raise ValueError("RGR needs one subject index per rewrite prompt")
        if len(self.rewrite_batch_indices) != len(self.prompt_last_indices):
            raise ValueError("RGR needs one prompt-last index per rewrite prompt")
        for position, layers in self.position_layers.items():
            for layer in layers:
                if not write_layer <= layer < num_hidden_layers:
                    raise ValueError(
                        f"RGR {position} layer L{layer} must be in "
                        f"[{write_layer}, {num_hidden_layers})"
                    )
                if position == "prompt_last" and layer <= write_layer:
                    raise ValueError(
                        "prompt_last RGR must be strictly downstream of the "
                        "subject-position latent injection layer"
                    )

    def _extract(
        self, hidden_states: Sequence[torch.Tensor]
    ) -> Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]]:
        transitions: Dict[
            Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        for position, layers in self.position_layers.items():
            indices = (
                self.subject_last_indices
                if position == "subject_last"
                else self.prompt_last_indices
            )
            for layer in layers:
                input_states = hidden_states[layer]
                output_states = hidden_states[layer + 1]
                selected_input: List[torch.Tensor] = []
                selected_output: List[torch.Tensor] = []
                for batch_index, token_index in zip(
                    self.rewrite_batch_indices, indices
                ):
                    absolute_input = _active_token_position(
                        self.attention_mask,
                        batch_index,
                        token_index,
                        input_states.device,
                    )
                    absolute_output = absolute_input.to(output_states.device)
                    selected_input.append(
                        input_states[batch_index, absolute_input, :]
                    )
                    selected_output.append(
                        output_states[batch_index, absolute_output, :]
                    )
                transitions[(position, layer)] = (
                    torch.stack(selected_input),
                    torch.stack(selected_output),
                )
        return transitions

    def observe(
        self, hidden_states: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Observe one inner-loop forward and return the weighted RGR loss."""

        if not self.enabled:
            return None, {}
        if hidden_states is None:
            raise RuntimeError("RGR was enabled but the model returned no hidden states")
        current = self._extract(hidden_states)
        if not self._reference:
            self._reference = {
                key: (input_state.detach(), output_state.detach())
                for key, (input_state, output_state) in current.items()
            }
            diagnostics: Dict[str, Any] = {
                "reference_initialized": True,
                "residual_gain_layers": list(self.layers),
                "residual_gain_token_scope": self.token_scope,
                "residual_gain_objective": self.objective,
                "residual_gain_alignment_weight": self.alignment_weight,
                "residual_gain_target_ratio": self.target_ratio,
                "residual_gain_cosine_aux_lambda": self.cosine_aux_lambda,
                "residual_gain_cosine_aux_sharpness": self.cosine_aux_sharpness,
            }
            reference_gains: List[float] = []
            reference_writes: List[float] = []
            reference_alignments: List[float] = []
            reference_cosines: List[float] = []
            reference_norm_ratios: List[float] = []
            reference_rhos: List[float] = []
            reference_rho_minus_one: List[float] = []
            reference_log_rhos: List[float] = []
            reference_objectives: List[float] = []
            reference_cosine_aux_penalties: List[float] = []
            reference_input_l2s: List[float] = []
            reference_output_l2s: List[float] = []
            for (position, layer), (
                input_state,
                output_state,
            ) in current.items():
                _, layer_diagnostics = residual_gain_loss(
                    input_state,
                    input_state,
                    output_state,
                    output_state,
                    margin=self.margin,
                    loss_type=self.loss_type,
                    objective=self.objective,
                    alignment_weight=self.alignment_weight,
                    target_ratio=self.target_ratio,
                    reduction_device=self.reduction_device,
                )
                diagnostics.update(
                    {
                        f"residual_gain_{position}_L{layer}_{key}": value
                        for key, value in layer_diagnostics.items()
                    }
                )
                cosine_aux_diagnostics: Dict[str, float] = {}
                if self.cosine_aux_lambda > 0.0:
                    _, cosine_aux_diagnostics = exponential_cosine_target_loss(
                        input_state,
                        output_state,
                        sharpness=self.cosine_aux_sharpness,
                        reduction_device=self.reduction_device,
                    )
                    diagnostics.update(
                        {
                            f"residual_gain_{position}_L{layer}_{key}": value
                            for key, value in cosine_aux_diagnostics.items()
                        }
                    )
                reference_gains.append(
                    float(layer_diagnostics["reference_gain_mean"])
                )
                reference_writes.append(
                    float(
                        layer_diagnostics[
                            "reference_write_relative_energy_mean"
                        ]
                    )
                )
                reference_alignments.append(
                    float(
                        layer_diagnostics[
                            "reference_alignment_contribution_mean"
                        ]
                    )
                )
                reference_cosines.append(
                    float(
                        layer_diagnostics[
                            "reference_input_write_cosine_mean"
                        ]
                    )
                )
                reference_norm_ratios.append(
                    float(
                        layer_diagnostics[
                            "reference_write_to_input_norm_ratio_mean"
                        ]
                    )
                )
                reference_rhos.append(
                    float(layer_diagnostics["reference_rho_mean"])
                )
                reference_rho_minus_one.append(
                    float(layer_diagnostics["reference_rho_minus_one_mean"])
                )
                reference_log_rhos.append(
                    float(layer_diagnostics["reference_log_rho_mean"])
                )
                reference_objectives.append(
                    float(layer_diagnostics["reference_objective_mean"])
                )
                reference_input_l2s.append(
                    float(layer_diagnostics["reference_input_l2_mean"])
                )
                reference_output_l2s.append(
                    float(layer_diagnostics["reference_output_l2_mean"])
                )
                reference_cosine_aux_penalties.append(
                    float(cosine_aux_diagnostics.get("cosine_aux_penalty_mean", 0.0))
                )
            diagnostics.update(
                {
                    "residual_gain_current_gain_mean": sum(reference_gains)
                    / len(reference_gains),
                    "residual_gain_reference_gain_mean": sum(reference_gains)
                    / len(reference_gains),
                    "residual_gain_excess_gain_mean": 0.0,
                    "residual_gain_violation_mean": 0.0,
                    "residual_gain_violation_max": 0.0,
                    "residual_gain_violation_fraction": 0.0,
                    "residual_gain_current_input_l2_mean": sum(
                        reference_input_l2s
                    )
                    / len(reference_input_l2s),
                    "residual_gain_reference_input_l2_mean": sum(
                        reference_input_l2s
                    )
                    / len(reference_input_l2s),
                    "residual_gain_current_output_l2_mean": sum(
                        reference_output_l2s
                    )
                    / len(reference_output_l2s),
                    "residual_gain_reference_output_l2_mean": sum(
                        reference_output_l2s
                    )
                    / len(reference_output_l2s),
                    "residual_gain_input_state_delta_l2_mean": 0.0,
                    "residual_gain_output_state_delta_l2_mean": 0.0,
                    "residual_gain_output_to_reference_norm_ratio_mean": 1.0,
                    "residual_gain_output_norm_positive_relative_excess_mean": 0.0,
                    "residual_gain_output_norm_positive_l2_excess_mean": 0.0,
                    "residual_gain_current_write_relative_energy_mean": sum(
                        reference_writes
                    )
                    / len(reference_writes),
                    "residual_gain_current_alignment_contribution_mean": sum(
                        reference_alignments
                    )
                    / len(reference_alignments),
                    "residual_gain_current_input_write_cosine_mean": sum(
                        reference_cosines
                    )
                    / len(reference_cosines),
                    "residual_gain_reference_input_write_cosine_mean": sum(
                        reference_cosines
                    )
                    / len(reference_cosines),
                    "residual_gain_current_write_to_input_norm_ratio_mean": sum(
                        reference_norm_ratios
                    )
                    / len(reference_norm_ratios),
                    "residual_gain_reference_write_to_input_norm_ratio_mean": sum(
                        reference_norm_ratios
                    )
                    / len(reference_norm_ratios),
                    "residual_gain_current_rho_mean": sum(reference_rhos)
                    / len(reference_rhos),
                    "residual_gain_reference_rho_mean": sum(reference_rhos)
                    / len(reference_rhos),
                    "residual_gain_current_rho_minus_one_mean": sum(
                        reference_rho_minus_one
                    )
                    / len(reference_rho_minus_one),
                    "residual_gain_reference_rho_minus_one_mean": sum(
                        reference_rho_minus_one
                    )
                    / len(reference_rho_minus_one),
                    "residual_gain_excess_rho_minus_one_mean": 0.0,
                    "residual_gain_rho_positive_l1_loss": 0.0,
                    "residual_gain_current_log_rho_mean": sum(reference_log_rhos)
                    / len(reference_log_rhos),
                    "residual_gain_reference_log_rho_mean": sum(reference_log_rhos)
                    / len(reference_log_rhos),
                    "residual_gain_excess_log_rho_mean": 0.0,
                    "residual_gain_log_rho_positive_l1_loss": 0.0,
                    "residual_gain_current_objective_mean": sum(
                        reference_objectives
                    )
                    / len(reference_objectives),
                    "residual_gain_reference_objective_mean": sum(
                        reference_objectives
                    )
                    / len(reference_objectives),
                    "residual_gain_excess_objective_mean": 0.0,
                    "residual_gain_cosine_aux_penalty_mean": sum(
                        reference_cosine_aux_penalties
                    )
                    / len(reference_cosine_aux_penalties),
                    "residual_gain_cosine_aux_loss": 0.0,
                    "weighted_residual_gain_base_loss": 0.0,
                    "weighted_residual_gain_cosine_aux_loss": 0.0,
                    "residual_gain_loss": 0.0,
                    "weighted_residual_gain_loss": 0.0,
                }
            )
            return None, diagnostics

        losses: List[torch.Tensor] = []
        by_position: Dict[str, List[torch.Tensor]] = {}
        diagnostics: Dict[str, Any] = {
            "reference_initialized": False,
            "residual_gain_layers": list(self.layers),
            "residual_gain_token_scope": self.token_scope,
            "residual_gain_loss_type": self.loss_type,
            "residual_gain_objective": self.objective,
            "residual_gain_alignment_weight": self.alignment_weight,
            "residual_gain_target_ratio": self.target_ratio,
            "residual_gain_cosine_aux_lambda": self.cosine_aux_lambda,
            "residual_gain_cosine_aux_sharpness": self.cosine_aux_sharpness,
            "residual_gain_margin": self.margin,
            "residual_gain_lambda": self.lambda_,
        }
        cosine_aux_losses: List[torch.Tensor] = []
        cosine_aux_metrics: Dict[str, List[float]] = {
            "cosine_aux_cosine_mean": [],
            "cosine_aux_penalty_mean": [],
            "cosine_aux_slope_mean": [],
            "cosine_aux_fraction_below_minus_0_9": [],
            "cosine_aux_fraction_below_minus_0_99": [],
        }
        metric_values: Dict[str, List[float]] = {
            "current_gain_mean": [],
            "reference_gain_mean": [],
            "excess_gain_mean": [],
            "violation_mean": [],
            "violation_fraction": [],
            "current_input_l2_mean": [],
            "reference_input_l2_mean": [],
            "current_output_l2_mean": [],
            "reference_output_l2_mean": [],
            "input_state_delta_l2_mean": [],
            "output_state_delta_l2_mean": [],
            "output_to_reference_norm_ratio_mean": [],
            "output_norm_positive_relative_excess_mean": [],
            "output_norm_positive_l2_excess_mean": [],
            "current_write_relative_energy_mean": [],
            "current_alignment_contribution_mean": [],
            "current_input_write_cosine_mean": [],
            "reference_input_write_cosine_mean": [],
            "current_write_to_input_norm_ratio_mean": [],
            "reference_write_to_input_norm_ratio_mean": [],
            "current_rho_mean": [],
            "reference_rho_mean": [],
            "current_rho_minus_one_mean": [],
            "reference_rho_minus_one_mean": [],
            "excess_rho_minus_one_mean": [],
            "rho_positive_l1_loss": [],
            "current_log_rho_mean": [],
            "reference_log_rho_mean": [],
            "excess_log_rho_mean": [],
            "log_rho_positive_l1_loss": [],
            "current_objective_mean": [],
            "reference_objective_mean": [],
            "excess_objective_mean": [],
        }
        violation_max_values: List[float] = []
        for (position, layer), (current_input, current_output) in current.items():
            reference_input, reference_output = self._reference[(position, layer)]
            layer_loss, layer_diagnostics = residual_gain_loss(
                current_input,
                reference_input,
                current_output,
                reference_output,
                margin=self.margin,
                loss_type=self.loss_type,
                objective=self.objective,
                alignment_weight=self.alignment_weight,
                target_ratio=self.target_ratio,
                reduction_device=self.reduction_device,
            )
            losses.append(layer_loss)
            by_position.setdefault(position, []).append(layer_loss)
            for key in metric_values:
                metric_values[key].append(float(layer_diagnostics[key]))
            violation_max_values.append(float(layer_diagnostics["violation_max"]))
            diagnostics.update(
                {
                    f"residual_gain_{position}_L{layer}_{key}": value
                    for key, value in layer_diagnostics.items()
                }
            )
            if self.cosine_aux_lambda > 0.0:
                cosine_aux_loss, cosine_aux_diagnostics = (
                    exponential_cosine_target_loss(
                        current_input,
                        current_output,
                        sharpness=self.cosine_aux_sharpness,
                        reduction_device=self.reduction_device,
                    )
                )
                cosine_aux_losses.append(cosine_aux_loss)
                for key in cosine_aux_metrics:
                    cosine_aux_metrics[key].append(
                        float(cosine_aux_diagnostics[key])
                    )
                diagnostics.update(
                    {
                        f"residual_gain_{position}_L{layer}_{key}": value
                        for key, value in cosine_aux_diagnostics.items()
                    }
                )

        raw_loss = torch.stack(losses).mean()
        if self.paired_mode:
            weighted_parts = []
            for position, position_losses in by_position.items():
                position_loss = torch.stack(position_losses).mean()
                position_lambda = self.position_lambdas[position]
                weighted_parts.append(position_lambda * position_loss)
                diagnostics[f"residual_gain_{position}_loss"] = float(
                    position_loss.detach().item()
                )
                diagnostics[f"residual_gain_{position}_lambda"] = position_lambda
            weighted_loss = torch.stack(weighted_parts).sum()
        else:
            weighted_loss = self.lambda_ * raw_loss
        weighted_base_loss = weighted_loss
        if cosine_aux_losses:
            raw_cosine_aux_loss = torch.stack(cosine_aux_losses).mean()
            weighted_cosine_aux_loss = (
                self.cosine_aux_lambda * raw_cosine_aux_loss
            )
            weighted_loss = weighted_base_loss + weighted_cosine_aux_loss
        else:
            raw_cosine_aux_loss = torch.zeros(
                (), device=weighted_base_loss.device, dtype=weighted_base_loss.dtype
            )
            weighted_cosine_aux_loss = raw_cosine_aux_loss
            # Keep the legacy RGR graph and arithmetic path exact when the
            # new auxiliary is disabled (the dataclass default).
            weighted_loss = weighted_base_loss
        if not self.optimization_enabled:
            weighted_loss = weighted_loss * 0.0
            weighted_base_loss = weighted_base_loss * 0.0
            weighted_cosine_aux_loss = weighted_cosine_aux_loss * 0.0

        diagnostics["residual_gain_loss"] = float(raw_loss.detach().item())
        diagnostics["weighted_residual_gain_loss"] = float(
            weighted_loss.detach().item()
        )
        diagnostics["weighted_residual_gain_base_loss"] = float(
            weighted_base_loss.detach().item()
        )
        diagnostics["residual_gain_cosine_aux_loss"] = float(
            raw_cosine_aux_loss.detach().item()
        )
        diagnostics["weighted_residual_gain_cosine_aux_loss"] = float(
            weighted_cosine_aux_loss.detach().item()
        )
        diagnostics["residual_gain_cosine_aux_penalty_mean"] = float(
            raw_cosine_aux_loss.detach().item()
        )
        for key, values in cosine_aux_metrics.items():
            diagnostics[f"residual_gain_{key}"] = (
                sum(values) / len(values) if values else 0.0
            )
        for key, values in metric_values.items():
            diagnostics[f"residual_gain_{key}"] = sum(values) / len(values)
        diagnostics["residual_gain_violation_max"] = max(violation_max_values)
        return weighted_loss, diagnostics
