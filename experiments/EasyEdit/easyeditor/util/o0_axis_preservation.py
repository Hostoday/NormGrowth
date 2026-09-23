"""Hard preservation of O8 coefficients on selected early-output directions.

Capture ONLY selected early block-output tokens (O0 by default, or O0..O3).
At the first target-block
callback, call ``initialize_reference`` BEFORE adding delta. This freezes all
context references and an orthonormal basis of all selected output directions. Every
forward must inject ``effective_delta(raw_delta)`` and the finally returned
target must use the same effective delta. This is a parameterization, not a
penalty or a one-time repair of an unconstrained optimum.

The preserved quantity is the signed ABSOLUTE coefficient, not cosine, energy,
or overall norm. The reference is the CURRENT pre-edit model, not original
Base. Every supplied context is constrained; callers must supply ALL rows that
receive shared delta, including a KL row if delta is injected there. Actual
BF16 addition and the later shared-weight solver can violate the FP32 virtual
constraint, so diagnostics deliberately measure actual target vectors too.

This standalone module imports only PyTorch and the standard library.
"""
from __future__ import annotations

import math
import operator
from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn


def project_shared_delta(raw_delta: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """FP32 differentiable projection onto the complement of fixed columns U.

    Input delta is [D], basis is [D,R] and must already be orthonormal. The
    delta retains gradients through casting/device operations; basis NEVER
    receives gradients. The owning preserver verifies the basis once at init.
    """
    if not isinstance(raw_delta, torch.Tensor) or raw_delta.ndim != 1:
        raise ValueError("raw_delta must be a one-dimensional tensor")
    if not torch.is_floating_point(raw_delta):
        raise ValueError("raw_delta must be floating point")
    if not isinstance(basis, torch.Tensor) or basis.ndim != 2 or basis.shape[0] != raw_delta.numel():
        raise ValueError("basis must have shape [hidden_dimension, rank]")
    delta = raw_delta.float()
    u = basis.detach().to(device=delta.device, dtype=torch.float32)
    return delta - u @ (u.T @ delta)


def _integer_indices(values: Sequence[int], name: str):
    try:
        return tuple(operator.index(value) for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain integer indices") from exc


class O0AxisPreserver:
    """Freeze early-output axes and project shared delta while retaining gradients.

    Lifecycle: ``with preserver:`` installs selected output hooks until the first
    initialization. ``initialize_reference(target_output)`` is invoked inside
    the target edit callback BEFORE injection. No target hook is installed, so
    ordering with an editor's callback is explicit. Reentry reuses frozen axes
    without installing hooks; ``close()`` removes capture hooks but keeps the
    reference available for post-update diagnostics and artifact export.

    Absolute or Python-style negative lookup indices are accepted. They are
    resolved against the actual tensor sequence length, then checked against
    the supplied [B,T] attention mask. Negative indices selecting right-hand
    padding therefore fail rather than silently measuring the wrong token.
    ``batch_first=False`` supports [T,B,D] module outputs explicitly.
    ``axis_layers=None`` preserves the original O0-only behavior; supplying
    [0,1,2,3] uses every context/layer direction in one common span. The legacy
    o0_module_name overrides the layer0 path, while later paths use the template.
    """

    def __init__(self, model: nn.Module, *, enabled: bool,
                 context_batch_indices: Sequence[int], lookup_indices: Sequence[int],
                 o0_module_name: Optional[str] = None,
                 axis_layers: Optional[Sequence[int]] = None,
                 module_template: str = "model.layers.{}",
                 attention_mask: Optional[torch.Tensor] = None,
                 batch_first: bool = True, svd_rtol: float = 1e-6,
                 svd_atol: float = 0.0, eps: float = 1e-8) -> None:
        self.model, self.enabled = model, bool(enabled)
        self.context_batch_indices = _integer_indices(context_batch_indices, "context_batch_indices")
        self.lookup_indices = _integer_indices(lookup_indices, "lookup_indices")
        self.axis_layers = _integer_indices((0,) if axis_layers is None else axis_layers, "axis_layers")
        self.module_template = str(module_template)
        self.o0_module_name = str(o0_module_name) if o0_module_name is not None else self.module_template.format(0)
        self.batch_first = bool(batch_first)
        self.svd_rtol, self.svd_atol, self.eps = float(svd_rtol), float(svd_atol), float(eps)
        self.attention_mask = None if attention_mask is None else attention_mask.detach().clone()
        self._handles = []
        self._entered = False
        self._captured_o0 = None
        self._captured_outputs = {}
        self._captured_shape = None
        self.resolved_lookup_indices = None
        self.o0_context_vectors = self.context_unit_axes = None
        self.context_layer_output_vectors = self.context_layer_unit_axes = None
        self.context_layer_reference_coefficients = self.context_layer_norms = None
        self.context_layer_basis_span_residual_norm = None
        self.reference = self.reference_coefficients = self.basis = None
        self.singular_values = self.o0_norms = None
        self.basis_span_residual_norm_per_context = None
        self.basis_orthogonality_max_abs = None
        self.rank = 0
        if self.enabled:
            if not self.axis_layers or self.axis_layers[0] != 0 or tuple(sorted(set(self.axis_layers))) != self.axis_layers:
                raise ValueError("axis_layers must be increasing distinct nonnegative indices including layer0")
            if not self.context_batch_indices or len(self.context_batch_indices) != len(self.lookup_indices):
                raise ValueError("context and lookup indices must have matching nonzero lengths")
            if min(self.context_batch_indices) < 0 or len(set(self.context_batch_indices)) != len(self.context_batch_indices):
                raise ValueError("context batch indices must be distinct and nonnegative")
            if not math.isfinite(self.eps) or self.eps <= 0:
                raise ValueError("eps must be positive and finite")
            if any(not math.isfinite(x) or x < 0 for x in (self.svd_rtol, self.svd_atol)):
                raise ValueError("SVD tolerances must be nonnegative and finite")

    @property
    def initialized(self) -> bool:
        return self.reference is not None

    def _select(self, output: Any, *, allow_selected: bool = False) -> torch.Tensor:
        values = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(values, torch.Tensor):
            raise ValueError("output must be a tensor or a tensor-first tuple/list")
        if allow_selected and values.ndim == 2:
            if values.shape != self.reference.shape:
                raise ValueError("selected current targets must have reference shape [contexts, hidden]")
            return values
        if values.ndim != 3:
            raise ValueError("module output must have shape [batch, sequence, hidden]")
        if not self.batch_first:
            values = values.transpose(0, 1)
        batch_size, sequence_length = values.shape[:2]
        if max(self.context_batch_indices) >= batch_size:
            raise IndexError("context batch index outside module output")
        if any(index < -sequence_length or index >= sequence_length for index in self.lookup_indices):
            raise IndexError("lookup index outside module output")
        resolved = tuple(index % sequence_length for index in self.lookup_indices)
        if self._captured_shape is not None and tuple(values.shape[:2]) != self._captured_shape:
            raise ValueError("target batch/sequence shape differs from the captured O0 forward")
        batch = torch.tensor(self.context_batch_indices, device=values.device, dtype=torch.long)
        token = torch.tensor(resolved, device=values.device, dtype=torch.long)
        if self.attention_mask is not None:
            mask = self.attention_mask
            if mask.ndim != 2 or tuple(mask.shape) != tuple(values.shape[:2]):
                raise ValueError("attention mask must match module batch and sequence dimensions")
            selected = mask[batch.to(mask.device), token.to(mask.device)]
            if not bool(selected.bool().all()):
                raise ValueError("an injected-context lookup selects padding")
        return values[batch, token, :]

    def __enter__(self):
        if self._entered:
            raise RuntimeError("O0 capture contexts must not be nested")
        self._entered = True
        if not self.enabled or self.initialized:
            return self
        self._captured_o0 = self._captured_shape = None
        self._captured_outputs.clear()
        try:
            for layer in self.axis_layers:
                name = self.o0_module_name if layer == 0 else self.module_template.format(layer)
                module = self.model.get_submodule(name)

                def capture(_module, _inputs, output, index=layer):
                    if self.initialized:
                        return
                    values = output[0] if isinstance(output, (tuple, list)) else output
                    selected = self._select(output).detach().clone()
                    self._captured_outputs[index] = selected
                    if index == 0:
                        self._captured_o0 = selected
                    shape = values.shape[:2] if self.batch_first else (values.shape[1], values.shape[0])
                    self._captured_shape = tuple(shape)
                    self.resolved_lookup_indices = tuple(i % shape[1] for i in self.lookup_indices)

                self._handles.append(module.register_forward_hook(capture))
        except Exception:
            self.close()
            raise
        return self

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._captured_o0 = None
        self._captured_outputs.clear()
        self._entered = False

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()

    def initialize_reference(self, target_output: Any) -> None:
        """Freeze selected early outputs and O8 BEFORE any delta mutation.

        ``context_layer_unit_axes[C,K,D]`` uses raw O_j unit vectors, not
        chronological innovations. All C*K rows share ONE delta projector.
        Legacy O0 tensor fields remain layer0 slices even for K>1.
        """
        if not self.enabled or self.initialized:
            return
        missing = [layer for layer in self.axis_layers if layer not in self._captured_outputs]
        if missing:
            raise RuntimeError(f"early outputs must be captured before initialize_reference; missing layers {missing}")
        reference = self._select(target_output).detach().float().clone()
        if any(value.shape != reference.shape for value in self._captured_outputs.values()):
            raise ValueError("early outputs and target context/hidden dimensions differ")
        vectors = torch.stack([self._captured_outputs[layer].to(device=reference.device, dtype=torch.float32)
                               for layer in self.axis_layers], dim=1).clone()
        if not bool(torch.isfinite(vectors).all()) or not bool(torch.isfinite(reference).all()):
            raise ValueError("early outputs and reference targets must be finite")
        with torch.no_grad():
            norms = torch.linalg.vector_norm(vectors, dim=-1)
            if bool((norms <= self.eps).any()):
                raise ValueError("a selected early-output vector has zero or too-small norm")
            unit = vectors / norms[..., None]
            flat_unit = unit.reshape(-1, unit.shape[-1])
            # Columns span all supplied context axes, not a separate projector
            # per context: every context receives the SAME effective delta.
            left, singular, _ = torch.linalg.svd(flat_unit.T, full_matrices=False)
            cutoff = max(self.svd_atol, self.svd_rtol * float(singular[0]))
            keep = singular > cutoff
            rank = int(keep.sum())
            if rank == 0:
                raise ValueError("SVD tolerance removed every O0 constraint")
            basis = torch.linalg.qr(left[:, keep], mode="reduced")[0]
            gram_error = basis.T @ basis - torch.eye(rank, device=basis.device)
            span_residual = flat_unit - (flat_unit @ basis) @ basis.T
            self.context_layer_output_vectors = vectors.detach()
            self.context_layer_unit_axes = unit.detach()
            self.context_layer_norms = norms.detach()
            self.context_layer_reference_coefficients = (reference[:, None, :] * unit).sum(dim=-1).detach()
            self.context_layer_basis_span_residual_norm = span_residual.norm(dim=-1).reshape(unit.shape[:2]).detach()
            self.o0_context_vectors = vectors[:, 0, :].detach()
            self.context_unit_axes = unit[:, 0, :].detach()
            self.reference = reference
            self.reference_coefficients = self.context_layer_reference_coefficients[:, 0]
            self.basis = basis.detach()
            self.singular_values, self.o0_norms = singular.detach(), norms[:, 0].detach()
            self.rank = rank
            self.basis_span_residual_norm_per_context = self.context_layer_basis_span_residual_norm[:, 0]
            self.basis_orthogonality_max_abs = float(gram_error.abs().max())
        self._captured_o0 = None
        self._captured_outputs.clear()

    def effective_delta(self, raw_delta: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return raw_delta
        if not self.initialized:
            raise RuntimeError("initialize_reference must run before effective_delta")
        return project_shared_delta(raw_delta, self.basis)

    def coefficient_diagnostics(self, current_targets_or_output: Any) -> Dict[str, Any]:
        """Measure signed drift on provided FP32 virtual or actual BF16 targets.

        Accepts selected [C,D] targets or a complete module output [B,T,D]. The
        latter must retain the original optimization batch/token coordinates.
        For differently shaped raw-prompt observations, select targets outside
        this utility and compare to explicitly matching reference axes instead.
        """
        if not self.enabled:
            return {}
        if not self.initialized:
            raise RuntimeError("initialize_reference must run before diagnostics")
        with torch.no_grad():
            current = self._select(current_targets_or_output, allow_selected=True).detach().float()
            if not bool(torch.isfinite(current).all()):
                raise ValueError("current target vectors must be finite")
            axes = self.context_unit_axes.to(current.device)
            reference_coeff = self.reference_coefficients.to(current.device)
            coefficients = (current * axes).sum(dim=-1)
            drift = coefficients - reference_coeff
            reference_norm = self.reference.to(current.device).norm(dim=-1)
            layer_axes = self.context_layer_unit_axes.to(current.device)
            layer_reference = self.context_layer_reference_coefficients.to(current.device)
            layer_coefficients = (current[:, None, :] * layer_axes).sum(dim=-1)
            layer_drift = layer_coefficients - layer_reference
            prefix = "o0_axis_preservation_"
            return {
                prefix + "reference": "current_pre_edit",
                prefix + "context_scope": "all_supplied_injected_contexts",
                prefix + "context_count": len(self.context_batch_indices),
                prefix + "rank": self.rank,
                prefix + "reference_coefficients": reference_coeff.cpu().tolist(),
                prefix + "current_coefficients": coefficients.cpu().tolist(),
                prefix + "coefficient_drift_per_context": drift.cpu().tolist(),
                prefix + "coefficient_drift_abs_max": float(drift.abs().max().cpu()),
                prefix + "coefficient_drift_rms": float(drift.square().mean().sqrt().cpu()),
                prefix + "coefficient_drift_over_reference_norm_abs_max": float((drift.abs()/reference_norm.clamp_min(self.eps)).max().cpu()),
                prefix + "reference_target_norms": reference_norm.cpu().tolist(),
                prefix + "current_target_norms": current.norm(dim=-1).cpu().tolist(),
                prefix + "axis_layers": list(self.axis_layers),
                prefix + "context_layer_reference_coefficients": layer_reference.cpu().tolist(),
                prefix + "context_layer_current_coefficients": layer_coefficients.cpu().tolist(),
                prefix + "context_layer_coefficient_drift_per_context_layer": layer_drift.cpu().tolist(),
                prefix + "context_layer_coefficient_drift_abs_max": float(layer_drift.abs().max().cpu()),
                prefix + "context_layer_coefficient_drift_rms": float(layer_drift.square().mean().sqrt().cpu()),
                prefix + "context_layer_coefficient_drift_over_reference_norm_abs_max": float((layer_drift.abs()/reference_norm[:, None].clamp_min(self.eps)).max().cpu()),
            }

    def delta_diagnostics(self, raw_delta: torch.Tensor) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        with torch.no_grad():
            effective = self.effective_delta(raw_delta).detach()
            raw = raw_delta.detach().float()
            axes = self.context_unit_axes.to(effective.device)
            constraints = axes @ effective
            basis_constraints = self.basis.to(effective.device).T @ effective
            layer_axes = self.context_layer_unit_axes.to(effective.device)
            layer_constraints = layer_axes @ effective
            prefix = "o0_axis_preservation_"
            return {
                prefix + "raw_delta_norm": float(raw.norm().cpu()),
                prefix + "effective_delta_norm": float(effective.norm().cpu()),
                prefix + "removed_delta_norm": float((raw-effective).norm().cpu()),
                prefix + "raw_axis_projection_per_context": (axes@raw).cpu().tolist(),
                prefix + "effective_axis_projection_per_context": constraints.cpu().tolist(),
                prefix + "effective_axis_projection_abs_max": float(constraints.abs().max().cpu()),
                prefix + "effective_basis_projection_abs_max": float(basis_constraints.abs().max().cpu()),
                prefix + "axis_layers": list(self.axis_layers),
                prefix + "raw_context_layer_projection_per_context_layer": (layer_axes@raw).cpu().tolist(),
                prefix + "effective_context_layer_projection_per_context_layer": layer_constraints.cpu().tolist(),
                prefix + "effective_context_layer_projection_abs_max": float(layer_constraints.abs().max().cpu()),
                prefix + "effective_context_layer_projection_rms": float(layer_constraints.square().mean().sqrt().cpu()),
            }

    def export_state(self) -> Dict[str, Any]:
        """Return top-level detached CPU tensors and JSON-compatible metadata."""
        payload = dict(schema_version=1, enabled=self.enabled, initialized=self.initialized,
                       reference_semantics="current_pre_edit",
                       context_scope="all_supplied_injected_contexts",
                       objective="hard_shared_delta_nullspace_parameterization",
                       preserved_quantity="signed_absolute_O0_coefficient_of_target",
                       context_batch_indices=list(self.context_batch_indices),
                       lookup_indices=list(self.lookup_indices),
                       resolved_lookup_indices=None if self.resolved_lookup_indices is None else list(self.resolved_lookup_indices),
                       o0_module_name=self.o0_module_name, batch_first=self.batch_first,
                       axis_layers=list(self.axis_layers), module_template=self.module_template,
                       projection_dtype="float32", svd_rtol=self.svd_rtol,
                       svd_atol=self.svd_atol, eps=self.eps, rank=self.rank,
                       basis_layout="hidden_dimension, retained_shared_rank",
                       context_vector_layout="injected_context, hidden_dimension",
                       context_layer_vector_layout="injected_context, selected_axis_layer, hidden_dimension",
                       legacy_context_tensor_semantics="layer0_slice",
                       axis_definition="raw_selected_block_output_unit_vectors_not_chronological_innovations")
        if len(self.axis_layers) > 1:
            payload["preserved_quantity"] = "signed_absolute_coefficients_of_selected_early_output_axes"
        if self.initialized:
            for name in ("o0_context_vectors", "context_unit_axes", "reference", "reference_coefficients",
                         "basis", "singular_values", "o0_norms", "basis_span_residual_norm_per_context",
                         "context_layer_output_vectors", "context_layer_unit_axes", "context_layer_norms",
                         "context_layer_reference_coefficients", "context_layer_basis_span_residual_norm"):
                payload[name] = getattr(self, name).detach().cpu().clone()
            payload["nullspace_dimension"] = self.reference.shape[1] - self.rank
            payload["basis_orthogonality_max_abs"] = self.basis_orthogonality_max_abs
        return payload
