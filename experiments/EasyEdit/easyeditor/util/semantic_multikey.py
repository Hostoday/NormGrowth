"""Exact normal-equation compression for same-target semantic multi-keys.

For fact ``i``, let ``k[i, s]`` be semantic-wording keys with non-negative
weights ``w[i, s]`` that sum to one, and let every wording share target
response ``r[i]``.  The explicit weighted objective is

    sum_i sum_s w[i, s] ||DeltaW k[i, s] - r[i]||^2.

It has the same normal equations as

    sum_i ||DeltaW mean[i] - r[i]||^2
      + sum_i sum_s w[i, s] ||DeltaW deviation[i, s]||^2,

where ``mean[i] = sum_s w[i, s] k[i, s]`` and
``deviation[i, s] = k[i, s] - mean[i]``.  Consequently, semantic
deviations contribute to the key Gram matrix but have zero desired
response and do not contribute to the target-key cross term.

The helpers below only form the data-dependent normal-equation terms.  A
caller can add MEMIT's covariance regularizer or AlphaEdit's projection and
cache terms afterwards.  Calling :func:`same_target_multikey_terms` on
successive chunks and summing its outputs is an exact streaming
implementation; all semantic keys need not be retained at once.
"""

from typing import NamedTuple, Optional

import torch


class SameTargetMultiKeyTerms(NamedTuple):
    """Compressed terms for a linear same-target multi-key solve.

    Attributes:
        mean_keys: Weighted key means with shape ``[key_dim, num_facts]``.
        key_gram: Mean Gram plus weighted within-fact deviation Gram; exactly
            equal to the Gram
            of explicitly expanded, square-root-weighted keys.
        target_key_cross: Optionally, ``sum_i r[i] mean[i].T`` with shape
            ``[value_dim, key_dim]``; exactly equal to the explicit weighted
            target-key cross term.  Both MEMIT and AlphaEdit can omit this
            allocation when they multiply the shared targets by
            ``mean_keys`` directly after solving/applying the projector.
    """

    mean_keys: torch.Tensor
    key_gram: torch.Tensor
    target_key_cross: Optional[torch.Tensor]


def protect_accumulator_owner_from_inplace_add(
    candidate: torch.Tensor,
    owner: torch.Tensor,
) -> torch.Tensor:
    """Clone ``candidate`` only when it aliases a persistent accumulator.

    ``Tensor.to(device, dtype)`` returns the original tensor when no transfer
    or cast is needed.  Since :func:`same_target_multikey_terms` intentionally
    adds into ``key_gram`` in place, a provisional CPU solve could otherwise
    mutate AlphaEdit's cumulative cache before the edit is committed.  A GPU
    transfer is already private and must not be cloned again: for Llama-3 a
    14336 x 14336 float32 clone costs roughly 0.77 GiB.
    """

    aliases_owner = (
        candidate.device == owner.device
        and candidate.dtype == owner.dtype
        and candidate.numel() > 0
        and candidate.data_ptr() == owner.data_ptr()
    )
    return candidate.clone() if aliases_owner else candidate


def _normalized_weights(
    keys: torch.Tensor,
    weights: Optional[torch.Tensor],
) -> torch.Tensor:
    num_facts, num_variants, _ = keys.shape
    if weights is None:
        return keys.new_full(
            (num_facts, num_variants),
            1.0 / float(num_variants),
        )

    weights = torch.as_tensor(weights, device=keys.device, dtype=keys.dtype)
    if weights.ndim == 1:
        if weights.shape[0] != num_variants:
            raise ValueError(
                "One-dimensional weights must have length num_variants"
            )
        weights = weights.unsqueeze(0).expand(num_facts, -1)
    elif weights.shape != (num_facts, num_variants):
        raise ValueError(
            "weights must have shape [num_variants] or "
            "[num_facts, num_variants]"
        )
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("weights must be finite")
    if bool((weights < 0).any()):
        raise ValueError("weights must be non-negative")

    totals = weights.sum(dim=1, keepdim=True)
    if bool((totals <= 0).any()):
        raise ValueError("Each fact must have positive total weight")

    # Per-fact normalization keeps the total data strength equal to one
    # original key/value constraint per fact rather than multiplying it by V,
    # the number of semantic wordings.
    return weights / totals


def same_target_multikey_terms(
    keys: torch.Tensor,
    targets: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    key_gram: Optional[torch.Tensor] = None,
    target_key_cross: Optional[torch.Tensor] = None,
    compute_target_key_cross: bool = True,
) -> SameTargetMultiKeyTerms:
    """Return exact compressed normal-equation terms.

    Args:
        keys: Semantic keys shaped ``[num_facts, num_variants, key_dim]``.
        targets: One shared target response per fact, shaped
            ``[num_facts, value_dim]``.
        weights: Optional non-negative weights shaped ``[num_variants]`` or
            ``[num_facts, num_variants]``.  They are normalized separately
            for every fact so expanding from one to ``J`` variants does not
            silently increase the fact's total objective weight.  The second
            axis enumerates semantic wordings ``s=0,...,V-1``.
        key_gram: Optional existing ``[key_dim, key_dim]`` accumulator.  The
            current chunk's exact key Gram is added to it in place.
        target_key_cross: Optional existing ``[value_dim, key_dim]``
            accumulator.  The current chunk's target-key cross term is added
            to it in place.
        compute_target_key_cross: If false, do not allocate the potentially
            large ``[value_dim, key_dim]`` cross term.  This is preferred when
            the caller can use ``mean_keys @ targets`` directly (including
            AlphaEdit's projected RHS and MEMIT's same-target solve).

    Returns:
        :class:`SameTargetMultiKeyTerms`.  For MEMIT, add ``key_gram`` to the
        covariance-regularized system.  The RHS can either use
        ``target_key_cross`` or form the equivalent product from
        ``mean_keys`` and the unexpanded targets in the caller.

    Notes:
        This function is chunk-additive.  Passing the returned ``key_gram``
        and ``target_key_cross`` back as accumulators for subsequent disjoint
        fact chunks gives the full-batch terms exactly (up to normal
        floating-point reduction order).  ``mean_keys`` contains only the
        current chunk and can be consumed immediately by MEMIT's solve or
        retained separately if a single final solve is required.
    """
    if keys.ndim != 3:
        raise ValueError("keys must have shape [num_facts, num_variants, key_dim]")
    if keys.shape[0] == 0 or keys.shape[1] == 0 or keys.shape[2] == 0:
        raise ValueError("keys dimensions must all be non-zero")
    if targets.ndim != 2 or targets.shape[0] != keys.shape[0]:
        raise ValueError("targets must have shape [num_facts, value_dim]")
    if targets.device != keys.device or targets.dtype != keys.dtype:
        targets = targets.to(device=keys.device, dtype=keys.dtype)

    normalized_weights = _normalized_weights(keys, weights)
    means = torch.einsum("bj,bjd->bd", normalized_weights, keys)
    mean_keys = means.T.contiguous()

    key_dim = keys.shape[-1]
    value_dim = targets.shape[-1]
    if key_gram is None:
        key_gram = keys.new_zeros((key_dim, key_dim))
    elif key_gram.shape != (key_dim, key_dim):
        raise ValueError("key_gram accumulator has the wrong shape")
    elif key_gram.device != keys.device or key_gram.dtype != keys.dtype:
        raise ValueError("key_gram accumulator must match keys device and dtype")

    if not compute_target_key_cross and target_key_cross is not None:
        raise ValueError(
            "target_key_cross cannot be supplied when its computation is disabled"
        )
    if compute_target_key_cross:
        if target_key_cross is None:
            target_key_cross = keys.new_zeros((value_dim, key_dim))
        elif target_key_cross.shape != (value_dim, key_dim):
            raise ValueError("target_key_cross accumulator has the wrong shape")
        elif (
            target_key_cross.device != keys.device
            or target_key_cross.dtype != keys.dtype
        ):
            raise ValueError(
                "target_key_cross accumulator must match keys device and dtype"
            )

    # Add the between/mean component once per fact.
    key_gram.addmm_(mean_keys, mean_keys.T)
    # Add each within-fact semantic-deviation component in turn.  Iterating
    # over V avoids materializing another [B, V, D] tensor, which matters for
    # large edit batches.
    for variant in range(keys.shape[1]):
        deviations = keys[:, variant, :] - means
        deviations.mul_(normalized_weights[:, variant].sqrt().unsqueeze(1))
        key_gram.addmm_(deviations.T, deviations)

    if target_key_cross is not None:
        target_key_cross.addmm_(targets.T, means)

    return SameTargetMultiKeyTerms(
        mean_keys=mean_keys,
        key_gram=key_gram,
        target_key_cross=target_key_cross,
    )
