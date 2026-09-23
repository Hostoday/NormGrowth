"""Preserve signed early-attention projections during latent-target optimization.

The reference is the current model's pre-edit forward, not a clean base model.
Only the selected rewrite-context subject tokens are retained.  The caller must
initialize the reference before injecting delta and pass the post-injection
block output to ``loss``.  The loss is unweighted; its lambda belongs to the
editing objective.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import nn


def early_attention_projection_loss(
    current: torch.Tensor,
    reference: torch.Tensor,
    basis: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Mean-context squared signed projection drift of normalized vectors.

    ``current`` and ``reference`` have shape C x D; ``basis`` has shape
    C x D x R.  Rank-truncated columns are zero.  There is no division by
    rank: each context contributes its squared projected drift.  Reference
    and basis are always detached, while casts/device transfers of current
    retain its gradient.
    """
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if current.ndim != 2 or reference.shape != current.shape:
        raise ValueError("current and reference must have matching C x D shapes")
    if current.shape[0] == 0:
        raise ValueError("at least one rewrite context is required")
    if basis.ndim != 3 or basis.shape[:2] != current.shape:
        raise ValueError("basis must have shape C x D x R")
    z = current.float()
    h = reference.detach().to(device=z.device, dtype=torch.float32)
    u = basis.detach().to(device=z.device, dtype=torch.float32)
    z_norm = torch.linalg.vector_norm(z, dim=-1, keepdim=True)
    h_norm = torch.linalg.vector_norm(h, dim=-1, keepdim=True)
    raw_z = torch.einsum("cdr,cd->cr", u, z)
    raw_h = torch.einsum("cdr,cd->cr", u, h)
    normalized_z = raw_z / z_norm.clamp_min(eps)
    normalized_h = raw_h / h_norm.clamp_min(eps)
    per_context = (normalized_z - normalized_h).square().sum(dim=-1)
    loss = per_context.mean()
    with torch.no_grad():
        diagnostics = {
            "early_attention_preservation_loss": float(loss.detach().cpu()),
            "early_attention_preservation_loss_per_context": per_context.detach().cpu().tolist(),
            "early_attention_preservation_normalized_projection_drift_l2_mean": float(
                per_context.sqrt().mean().cpu()
            ),
            "early_attention_preservation_raw_projection_drift_l2_mean": float(
                torch.linalg.vector_norm(raw_z - raw_h, dim=-1).mean().cpu()
            ),
            "early_attention_preservation_current_norm_per_context": z_norm[:, 0].cpu().tolist(),
            "early_attention_preservation_reference_norm_per_context": h_norm[:, 0].cpu().tolist(),
            "early_attention_preservation_current_raw_projections": raw_z.cpu().tolist(),
            "early_attention_preservation_reference_raw_projections": raw_h.cpu().tolist(),
            "early_attention_preservation_current_normalized_projections": normalized_z.cpu().tolist(),
            "early_attention_preservation_reference_normalized_projections": normalized_h.cpu().tolist(),
        }
    return loss, diagnostics


class EarlyAttentionPreservationRegularizer:
    """Capture frozen per-context early writes and a pre-injection reference.

    Re-entering the context manager is supported.  Hooks are only necessary
    until initialization; subsequent contexts install none.  No target hook
    is installed, so this helper cannot race with the editor's output hook.
    ``batch_first=False`` explicitly supports sequence-first modules.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        enabled: bool,
        rewrite_batch_indices: Sequence[int],
        subject_last_indices: Sequence[int],
        layers: Sequence[int] = (0, 1, 2, 3, 4),
        attention_module_template: str = "model.layers.{}.self_attn",
        svd_rtol: float = 1e-5,
        svd_atol: float = 0.0,
        eps: float = 1e-8,
        attention_mask: Optional[torch.Tensor] = None,
        batch_first: bool = True,
    ) -> None:
        self.model = model
        self.enabled = bool(enabled)
        self.layers = tuple(int(layer) for layer in layers)
        self.rewrite_batch_indices = tuple(int(index) for index in rewrite_batch_indices)
        self.subject_last_indices = tuple(int(index) for index in subject_last_indices)
        self.attention_module_template = attention_module_template
        self.svd_rtol = float(svd_rtol)
        self.svd_atol = float(svd_atol)
        self.eps = float(eps)
        self.attention_mask = attention_mask
        self.batch_first = bool(batch_first)
        self._handles: list[Any] = []
        self._entered = False
        self._attention: Dict[int, torch.Tensor] = {}
        self.reference: Optional[torch.Tensor] = None
        self.basis: Optional[torch.Tensor] = None
        self.ranks: Optional[torch.Tensor] = None
        self.singular_values: Optional[torch.Tensor] = None
        self.attention_unit_vectors: Optional[torch.Tensor] = None
        if self.enabled:
            if not self.layers or len(set(self.layers)) != len(self.layers) or min(self.layers) < 0:
                raise ValueError("layers must contain distinct nonnegative layer indices")
            if not self.rewrite_batch_indices or len(self.rewrite_batch_indices) != len(self.subject_last_indices):
                raise ValueError("rewrite and subject indices must have the same nonzero length")
            if len(set(self.rewrite_batch_indices)) != len(self.rewrite_batch_indices):
                raise ValueError("rewrite batch indices must be distinct")
            if min(self.rewrite_batch_indices) < 0 or min(self.subject_last_indices) < 0:
                raise ValueError("rewrite and subject indices must be nonnegative")
            if any(not math.isfinite(value) or value < 0 for value in (self.svd_rtol, self.svd_atol)):
                raise ValueError("SVD tolerances must be finite and nonnegative")
            if not math.isfinite(self.eps) or self.eps <= 0:
                raise ValueError("eps must be finite and positive")

    @property
    def initialized(self) -> bool:
        return self.reference is not None

    def _select(self, output: Any) -> torch.Tensor:
        values = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(values, torch.Tensor) or values.ndim != 3:
            raise ValueError("module output must be a B x T x D tensor or tensor-first tuple")
        if not self.batch_first:
            values = values.transpose(0, 1)
        if max(self.rewrite_batch_indices) >= values.shape[0] or max(self.subject_last_indices) >= values.shape[1]:
            raise IndexError("rewrite-context subject index is outside the module output")
        batch = torch.tensor(self.rewrite_batch_indices, device=values.device, dtype=torch.long)
        token = torch.tensor(self.subject_last_indices, device=values.device, dtype=torch.long)
        if self.attention_mask is not None:
            mask = self.attention_mask
            if mask.ndim != 2 or tuple(mask.shape) != tuple(values.shape[:2]):
                raise ValueError("attention mask shape must match batch and sequence dimensions")
            valid = mask[batch.to(mask.device), token.to(mask.device)]
            if not bool(valid.bool().all()):
                raise ValueError("a selected subject token is padding")
        return values[batch, token, :]

    def __enter__(self) -> "EarlyAttentionPreservationRegularizer":
        if self._entered:
            raise RuntimeError("early-attention capture contexts must not be nested")
        self._entered = True
        if not self.enabled or self.initialized:
            return self
        self._attention.clear()
        try:
            for layer in self.layers:
                module = self.model.get_submodule(self.attention_module_template.format(layer))

                def capture(_module: nn.Module, _inputs: Any, output: Any, index: int = layer) -> None:
                    if not self.initialized:
                        self._attention[index] = self._select(output).detach().clone()

                self._handles.append(module.register_forward_hook(capture))
        except Exception:
            self.close()
            raise
        return self

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._attention.clear()
        self._entered = False

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.close()

    def initialize_reference(self, target_output: Any) -> None:
        """Freeze current pre-edit context states; call BEFORE delta injection."""
        if not self.enabled or self.initialized:
            return
        missing = [layer for layer in self.layers if layer not in self._attention]
        if missing:
            raise RuntimeError(f"early attention was not captured for layers {missing}")
        reference = self._select(target_output).detach().float().clone()
        # Only C x D token slices cross model-shard devices, never full sequences.
        writes = torch.stack(
            [self._attention[layer].to(device=reference.device, dtype=torch.float32) for layer in self.layers],
            dim=-1,
        )
        if writes.shape[:2] != reference.shape:
            raise ValueError("attention and target hidden dimensions must match")
        if not bool(torch.isfinite(writes).all()) or not bool(torch.isfinite(reference).all()):
            raise ValueError("early attention and reference must be finite")
        with torch.no_grad():
            u, singular_values, _ = torch.linalg.svd(writes, full_matrices=False)
            threshold = torch.maximum(
                singular_values[:, :1] * self.svd_rtol,
                torch.full_like(singular_values[:, :1], self.svd_atol),
            )
            keep = singular_values > threshold
            self.basis = (u * keep[:, None, :]).detach()
            self.ranks = keep.sum(dim=-1).detach()
            self.singular_values = singular_values.detach()
            self.attention_unit_vectors = (
                writes / torch.linalg.vector_norm(writes, dim=1, keepdim=True).clamp_min(self.eps)
            ).detach()
            self.reference = reference
        self._attention.clear()

    def loss(self, target_output: Any) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        if not self.enabled:
            return None, {}
        if not self.initialized or self.basis is None:
            raise RuntimeError("initialize_reference must run before computing preservation loss")
        current = self._select(target_output)
        loss, diagnostics = early_attention_projection_loss(current, self.reference, self.basis, eps=self.eps)
        diagnostics.update({
            "early_attention_preservation_reference": "current_pre_edit",
            "early_attention_preservation_layers": list(self.layers),
            "early_attention_preservation_rank_per_context": self.ranks.cpu().tolist(),
            "early_attention_preservation_context_count": len(self.rewrite_batch_indices),
        })
        # Report original A_j directions as well as the orthonormal SVD basis.
        with torch.no_grad():
            z = current.detach().float()
            h = self.reference.to(z.device)
            axes = self.attention_unit_vectors.to(z.device)
            raw_z = torch.einsum("cdk,cd->ck", axes, z)
            raw_h = torch.einsum("cdk,cd->ck", axes, h)
            diagnostics.update({
                "early_attention_preservation_current_attention_projection": raw_z.cpu().tolist(),
                "early_attention_preservation_reference_attention_projection": raw_h.cpu().tolist(),
                "early_attention_preservation_current_attention_cosine": (raw_z / z.norm(dim=-1, keepdim=True).clamp_min(self.eps)).cpu().tolist(),
                "early_attention_preservation_reference_attention_cosine": (raw_h / h.norm(dim=-1, keepdim=True).clamp_min(self.eps)).cpu().tolist(),
            })
        return loss, diagnostics

    def export_state(self) -> Dict[str, Any]:
        """Compact per-edit detached payload; serialization is the caller's job."""
        payload: Dict[str, Any] = {
            "schema_version": 1,
            "enabled": self.enabled,
            "initialized": self.initialized,
            "reference_semantics": "current_pre_edit",
            "objective": "mean_context_squared_signed_projection_drift_of_normalized_target",
            "layers": list(self.layers),
            "rewrite_batch_indices": list(self.rewrite_batch_indices),
            "subject_last_indices": list(self.subject_last_indices),
            "basis_layout": "context, hidden_dimension, padded_rank",
            "svd_rtol": self.svd_rtol,
            "svd_atol": self.svd_atol,
            "eps": self.eps,
        }
        if self.initialized:
            for name in ("reference", "basis", "ranks", "singular_values", "attention_unit_vectors"):
                payload[name] = getattr(self, name).detach().cpu().clone()
        return payload
