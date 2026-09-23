"""Released SADR attention-distribution objective, isolated and opt-in."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch


def detach_attention_tensors(
    attentions: Sequence[Optional[torch.Tensor]],
    layers: Sequence[int],
) -> Dict[int, torch.Tensor]:
    detached: Dict[int, torch.Tensor] = {}
    for layer in layers:
        if layer < 0 or layer >= len(attentions):
            raise IndexError(
                f"attention layer {layer} is outside [0, {len(attentions)})"
            )
        value = attentions[layer]
        if value is None:
            raise RuntimeError(
                f"model returned no attention tensor for layer {layer}; "
                "SADR requires output_attentions-compatible eager attention"
            )
        detached[int(layer)] = value.detach()
    return detached


def official_sadr_attention_kl_loss(
    attentions: Sequence[Optional[torch.Tensor]],
    reference: Mapping[int, torch.Tensor],
    attention_mask: torch.Tensor,
    batch_indices: Sequence[int],
    query_indices: Sequence[int],
    subject_indices: Sequence[int],
    *,
    layers: Sequence[int],
    reduction_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Reproduce SADR's selected-head ``KL(reference || current)``."""

    if not (
        len(batch_indices) == len(query_indices) == len(subject_indices)
    ):
        raise ValueError(
            "SADR batch, query, and subject index lists must have equal length"
        )
    if set(layers) != set(reference) or not layers:
        raise ValueError("SADR current/reference attention layers do not match")

    layer_losses = []
    selected_count = 0
    considered_count = 0
    selected_kl_sum = 0.0
    current_subject_sum = 0.0
    reference_subject_sum = 0.0
    subject_value_count = 0

    for layer in layers:
        current_layer = attentions[layer]
        if current_layer is None:
            raise RuntimeError(f"model returned no attention for layer {layer}")
        reference_layer = reference[layer].to(
            device=current_layer.device,
            dtype=current_layer.dtype,
        )
        eps = 1e-7 if current_layer.dtype == torch.float16 else 1e-8
        example_losses = []
        for batch_index, query_index, subject_index in zip(
            batch_indices,
            query_indices,
            subject_indices,
        ):
            active = torch.nonzero(
                attention_mask[batch_index], as_tuple=False
            ).flatten()
            active_length = int(active.numel())
            if active_length == 0:
                raise ValueError("SADR attention mask row has no active token")
            if not 0 <= int(query_index) < active_length:
                raise IndexError(
                    f"SADR query {query_index} outside length {active_length}"
                )
            if not 0 <= int(subject_index) < active_length:
                raise IndexError(
                    f"SADR subject {subject_index} outside length {active_length}"
                )

            active = active.to(current_layer.device)
            query_position = active[int(query_index)]
            subject_position = active[int(subject_index)]
            current_distribution = (
                current_layer[batch_index, :, query_position, :] + eps
            )
            reference_distribution = (
                reference_layer[batch_index, :, query_position, :] + eps
            )
            reference_subject = reference_distribution[:, subject_position]
            current_subject = current_distribution[:, subject_position]
            selected = (
                current_subject > reference_subject.max()
            ).detach()
            kl_per_head = (
                reference_distribution
                * (reference_distribution / current_distribution).log()
            ).sum(dim=-1)
            example_losses.append(
                (kl_per_head * selected.to(kl_per_head.dtype)).sum()
            )

            selected_count += int(selected.sum().item())
            considered_count += int(selected.numel())
            selected_kl_sum += float(
                (kl_per_head.detach() * selected).sum().float().item()
            )
            current_subject_sum += float(
                current_subject.detach().sum().float().item()
            )
            reference_subject_sum += float(
                reference_subject.detach().sum().float().item()
            )
            subject_value_count += int(current_subject.numel())

        layer_losses.append(torch.stack(example_losses).mean())

    loss = torch.stack(
        [
            value.to(reduction_device)
            if reduction_device is not None
            else value
            for value in layer_losses
        ]
    ).sum()
    return loss, {
        "sadr_selected_heads": float(selected_count),
        "sadr_considered_heads": float(considered_count),
        "sadr_selected_fraction": selected_count / max(considered_count, 1),
        "sadr_selected_kl_mean": selected_kl_sum / max(selected_count, 1),
        "sadr_current_subject_mass_mean": (
            current_subject_sum / max(subject_value_count, 1)
        ),
        "sadr_reference_subject_mass_mean": (
            reference_subject_sum / max(subject_value_count, 1)
        ),
        "sadr_layer_reduction_sum": 1.0,
    }
