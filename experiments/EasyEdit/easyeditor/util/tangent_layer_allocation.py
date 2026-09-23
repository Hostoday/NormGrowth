"""Allocate a target displacement across local tangent spaces.

This module is intentionally independent of MEMIT/AlphaEdit.  It implements
the ``J_l ~= I`` diagnostic model in which a write made at any edited layer is
assumed to arrive unchanged at a common residual-stream boundary.

For target ``d[b]`` and local reference directions ``u[b, l]``, the allocator
solves

    min_{a_1,...,a_L}  1/2 sum_l ||a_l||^2 / m_l
    subject to          u_l^T a_l = 0
                        sum_l a_l = d_projectable,

where ``m_l = strength_l / cost_l`` is a non-negative layer mobility and
``d_projectable`` is the Euclidean projection of ``d`` onto the sum of the
active tangent spaces.  The unrepresentable component is returned explicitly
as ``slack = d - d_projectable``.  In particular, if all active references are
parallel, their common radial direction cannot be represented by tangent
writes and appears in ``slack`` instead of being silently assigned to the last
layer.

No hidden_size x hidden_size matrix is formed or inverted.  The pseudoinverse
is evaluated from the eigendecomposition of an ``L x L`` Gram matrix, which is
small for the usual five edited layers.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch


class LocalTangentAllocation(NamedTuple):
    """Result of :func:`allocate_local_tangent_components`.

    Attributes:
        components: Per-layer tangent writes, shaped ``[B, L, H]``.
        projectable_target: Sum of ``components``, shaped ``[B, H]``.
        slack: ``target - projectable_target``.  This contains a common
            radial component when the tangent-space sum is rank deficient.
        unit_references: Normalized reference directions used by the solve.
        mobility: ``layer_strengths / layer_costs`` after broadcasting.
        tangency_dot: ``<unit_references, components>`` for diagnostics.
        common_radial_nullity: Number of numerically unresolved common radial
            directions.  It is zero or one when at least one layer is active,
            and ``H`` when all layer strengths are zero.
        small_system_eigenvalues: Eigenvalues of
            ``sum_l m_l (I - u_l u_l^T)`` restricted to the span of the local
            references.  Values below the relative cutoff are pseudoinverted
            to zero and their target component is returned in ``slack``.
        rcond: Effective relative eigenvalue cutoff used by the solve.
        eps: Effective absolute numerical tolerance used by the solve.
    """

    components: torch.Tensor
    projectable_target: torch.Tensor
    slack: torch.Tensor
    unit_references: torch.Tensor
    mobility: torch.Tensor
    tangency_dot: torch.Tensor
    common_radial_nullity: torch.Tensor
    small_system_eigenvalues: torch.Tensor
    rcond: float
    eps: float


def _broadcast_layer_parameter(
    value: Optional[torch.Tensor],
    *,
    default: float,
    batch_size: int,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if value is None:
        return torch.full(
            (batch_size, num_layers), default, device=device, dtype=dtype
        )

    result = torch.as_tensor(value, device=device, dtype=dtype)
    if result.ndim == 0:
        return result.expand(batch_size, num_layers)
    if result.shape == (num_layers,):
        return result.unsqueeze(0).expand(batch_size, -1)
    if result.shape == (batch_size, num_layers):
        return result
    raise ValueError(
        f"{name} must be scalar, [L], or [B, L]; got {tuple(result.shape)}"
    )


def allocate_local_tangent_components(
    target: torch.Tensor,
    local_references: torch.Tensor,
    *,
    layer_costs: Optional[torch.Tensor] = None,
    layer_strengths: Optional[torch.Tensor] = None,
    rcond: Optional[float] = None,
    eps: Optional[float] = None,
) -> LocalTangentAllocation:
    """Return the minimum-energy local-tangent decomposition of ``target``.

    Args:
        target: Desired common-boundary displacement ``d``, shaped ``[B, H]``.
        local_references: Local residual-stream states or other same-space
            reference vectors, shaped ``[B, L, H]``.  They need not already be
            unit length; normalization is performed internally.  A reference
            for a positive-strength layer must have non-zero norm.
        layer_costs: Optional positive scalar, ``[L]``, or ``[B, L]`` tensor.
            Larger values make a layer carry less of the target.
        layer_strengths: Optional non-negative scalar, ``[L]``, or ``[B, L]``
            tensor.  Zero disables a layer; larger values make a layer carry
            more.  Strength is an allocation prior, not a downstream Jacobian.
        rcond: Relative eigenvalue cutoff.  Nearly common radial directions
            below ``rcond * sum(mobility)`` are treated as unrepresentable and
            returned as slack rather than producing cancelling huge writes.
        eps: Positive absolute tolerance used for reference norms and inactive
            mobility sums.

    Returns:
        :class:`LocalTangentAllocation`.  Computation and outputs use at least
        float32 precision, even when half-precision inputs are supplied.

    Notes:
        The closed form is ``a_l = m_l P_l A^+ d``, where
        ``P_l = I - u_l u_l^T`` and ``A = sum_l m_l P_l``.  We obtain ``A^+``
        from the ``L x L`` Gram matrix of columns ``sqrt(m_l) u_l``; no
        ``H x H`` inverse is constructed.

        Tangency removes the first-order radial term, but does *not* exactly
        preserve norm: for ``a_l perpendicular to h_l``,
        ``||h_l + a_l||^2 = ||h_l||^2 + ||a_l||^2``.  Exact finite-step norm
        preservation needs a separate inward radial correction or endpoint
        norm constraint.
    """
    if target.ndim != 2:
        raise ValueError("target must have shape [B, H]")
    if local_references.ndim != 3:
        raise ValueError("local_references must have shape [B, L, H]")
    batch_size, hidden_size = target.shape
    ref_batch, num_layers, ref_hidden = local_references.shape
    if batch_size == 0 or hidden_size == 0 or num_layers == 0:
        raise ValueError("B, L, and H dimensions must all be non-zero")
    if ref_batch != batch_size or ref_hidden != hidden_size:
        raise ValueError(
            "local_references must share target's B and H dimensions; got "
            f"target={tuple(target.shape)}, refs={tuple(local_references.shape)}"
        )
    if not target.is_floating_point() or not local_references.is_floating_point():
        raise TypeError("target and local_references must be floating tensors")
    if not bool(torch.isfinite(target).all()):
        raise ValueError("target must be finite")
    if not bool(torch.isfinite(local_references).all()):
        raise ValueError("local_references must be finite")

    work_dtype = torch.promote_types(target.dtype, local_references.dtype)
    if work_dtype in (torch.float16, torch.bfloat16):
        work_dtype = torch.float32
    device = target.device
    d = target.to(device=device, dtype=work_dtype)
    references = local_references.to(device=device, dtype=work_dtype)

    finfo = torch.finfo(work_dtype)
    if eps is None:
        eps = max(10.0 * finfo.eps, 1e-12)
    eps = float(eps)
    if not 0.0 < eps < float("inf"):
        raise ValueError(f"eps must be finite and positive, got {eps}")
    if rcond is None:
        rcond = max(10.0 * finfo.eps, 1e-12)
    rcond = float(rcond)
    if not 0.0 <= rcond < 1.0:
        raise ValueError(f"rcond must be in [0, 1), got {rcond}")

    costs = _broadcast_layer_parameter(
        layer_costs,
        default=1.0,
        batch_size=batch_size,
        num_layers=num_layers,
        device=device,
        dtype=work_dtype,
        name="layer_costs",
    )
    strengths = _broadcast_layer_parameter(
        layer_strengths,
        default=1.0,
        batch_size=batch_size,
        num_layers=num_layers,
        device=device,
        dtype=work_dtype,
        name="layer_strengths",
    )
    if not bool(torch.isfinite(costs).all()) or bool((costs <= 0).any()):
        raise ValueError("layer_costs must be finite and strictly positive")
    if not bool(torch.isfinite(strengths).all()) or bool((strengths < 0).any()):
        raise ValueError("layer_strengths must be finite and non-negative")
    mobility = strengths / costs
    active = mobility > 0

    reference_norms = torch.linalg.vector_norm(references, dim=-1)
    if bool((active & (reference_norms <= eps)).any()):
        raise ValueError(
            "Every positive-strength layer needs a non-zero local reference"
        )
    unit_references = references / reference_norms.clamp_min(eps).unsqueeze(-1)
    unit_references = torch.where(
        active.unsqueeze(-1), unit_references, torch.zeros_like(unit_references)
    )

    mobility_sum = mobility.sum(dim=1)  # [B]
    safe_mobility_sum = mobility_sum.clamp_min(eps)

    # V has rows sqrt(m_l) u_l.  A = M I - V^T V, while the only non-trivial
    # spectrum of V^T V is available from the small Gram G = V V^T.
    weighted_references = mobility.sqrt().unsqueeze(-1) * unit_references
    gram = torch.bmm(weighted_references, weighted_references.transpose(1, 2))
    gram_eigenvalues, gram_eigenvectors = torch.linalg.eigh(gram)
    gram_eigenvalues = gram_eigenvalues.clamp_min(0)
    small_system_eigenvalues = (
        mobility_sum.unsqueeze(1) - gram_eigenvalues
    ).clamp_min(0)

    # Columns are normalized eigenvectors of V^T V in hidden space:
    # w_j = V^T q_j / sqrt(lambda_j).  Zero-lambda columns carry no hidden
    # direction and need no Woodbury correction to the isotropic 1/M term.
    hidden_eigenvectors = torch.bmm(
        weighted_references.transpose(1, 2), gram_eigenvectors
    )
    gram_cutoff = rcond * safe_mobility_sum.unsqueeze(1)
    nonzero_gram = gram_eigenvalues > gram_cutoff
    hidden_eigenvectors = hidden_eigenvectors / (
        gram_eigenvalues.clamp_min(eps).sqrt().unsqueeze(1)
    )
    hidden_eigenvectors = torch.where(
        nonzero_gram.unsqueeze(1),
        hidden_eigenvectors,
        torch.zeros_like(hidden_eigenvectors),
    )

    active_batch = mobility_sum > eps
    invertible = (
        small_system_eigenvalues > gram_cutoff
    ) & nonzero_gram & active_batch.unsqueeze(1)
    inverse_small_eigenvalues = torch.where(
        invertible,
        small_system_eigenvalues.clamp_min(eps).reciprocal(),
        torch.zeros_like(small_system_eigenvalues),
    )
    inverse_isotropic = torch.where(
        active_batch,
        safe_mobility_sum.reciprocal(),
        torch.zeros_like(safe_mobility_sum),
    )

    dual = inverse_isotropic.unsqueeze(1) * d
    hidden_coefficients = torch.einsum("bhl,bh->bl", hidden_eigenvectors, d)
    corrections = hidden_coefficients * (
        inverse_small_eigenvalues - inverse_isotropic.unsqueeze(1)
    )
    dual = dual + torch.einsum(
        "bhl,bl->bh", hidden_eigenvectors, corrections
    )

    radial_coefficients = torch.einsum("blh,bh->bl", unit_references, dual)
    projected_dual = dual.unsqueeze(1) - (
        radial_coefficients.unsqueeze(-1) * unit_references
    )
    components = mobility.unsqueeze(-1) * projected_dual
    projectable_target = components.sum(dim=1)
    slack = d - projectable_target
    tangency_dot = torch.einsum(
        "blh,blh->bl", unit_references, components
    )

    unresolved_common = (
        nonzero_gram
        & (small_system_eigenvalues <= gram_cutoff)
        & active_batch.unsqueeze(1)
    ).sum(dim=1)
    common_radial_nullity = torch.where(
        active_batch,
        unresolved_common,
        torch.full_like(unresolved_common, hidden_size),
    )

    return LocalTangentAllocation(
        components=components,
        projectable_target=projectable_target,
        slack=slack,
        unit_references=unit_references,
        mobility=mobility,
        tangency_dot=tangency_dot,
        common_radial_nullity=common_radial_nullity,
        small_system_eigenvalues=small_system_eigenvalues,
        rcond=rcond,
        eps=eps,
    )
