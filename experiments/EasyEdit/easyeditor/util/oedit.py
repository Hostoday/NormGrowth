"""Paper-based O-Edit soft regularization for sequential, final-layer editing.

This is an explicit interpretation of the original O-Edit regularizer, not
O-Edit+ and not a claim of bitwise official reproduction.  In particular the
paper's ``sim`` is underspecified: mean absolute cosine is our default, with
squared and signed alternatives exposed and recorded.  The historical basis
is the LEFT SVD basis of the SUM of actual committed weight updates.  It is
not a buffer of latent deltas or a union of individual update directions.

Only batch_size=1 and the final edited weight are supported by this adapter.
Call ``loss`` during compute_z and ``commit_factors`` only AFTER applying the
outer update.  Factors have functional shapes [output, rank], [input, rank].
For Linear, stored weight shape is [output, input]; Conv1D uses output_axis=1.
CPU float64 QR/core-SVD updates retain the cumulative numerical rank without
forming a dense cumulative weight matrix.  The gradient cache must have been
computed separately on the ORIGINAL model, and is never estimated here.
The gradient term retains the selected-q mean denominator: numerically null
or fully projected-out columns contribute zero instead of increasing the
weight of surviving directions by removing columns from the mean.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch


LOSS_TYPES = {"mean_abs_cosine", "mean_squared_cosine", "signed_cosine"}
_SVD_RTOL = 1e-7
_SVD_ATOL = 1e-12
_COS_EPS = 1e-8
_DENSE_FACTOR_RTOL = 2e-6


def _nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _cpu_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional tensor")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain finite real floating point values")
    return value.detach().to(device="cpu", dtype=torch.float64)


def _read_gradient_cache(
    path: str, *, model_name: str, weight_name: str,
    weight_shape: Tuple[int, int], output_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if not path or not Path(path).is_file():
        raise ValueError("oedit_lambda_gradient > 0 requires an existing gradient cache")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("O-Edit gradient cache requires schema_version=1")
    for field, expected in (("model_name", model_name), ("weight_name", weight_name)):
        if payload.get(field) != expected:
            raise ValueError(f"O-Edit gradient cache {field} provenance mismatch")
    if tuple(payload.get("weight_shape", ())) != weight_shape:
        raise ValueError("O-Edit gradient cache weight_shape provenance mismatch")
    vectors = _cpu_matrix(payload.get("left_vectors"), "gradient left_vectors")
    singular = payload.get("singular_values")
    if not isinstance(singular, torch.Tensor) or singular.ndim != 1:
        raise ValueError("gradient singular_values must be a one-dimensional tensor")
    singular = singular.detach().to(device="cpu", dtype=torch.float64)
    rank = vectors.shape[1]
    if vectors.shape[0] != output_dim or rank > min(weight_shape):
        raise ValueError("gradient left_vectors has incompatible OUTPUT dimension/rank")
    if singular.numel() != rank or not bool(torch.isfinite(singular).all()):
        raise ValueError("gradient singular_values has incompatible rank/nonfinite values")
    if bool((singular < 0).any()) or bool((singular[1:] > singular[:-1]).any()):
        raise ValueError("gradient singular_values must be nonnegative and descending")
    if not torch.allclose(vectors.T @ vectors, torch.eye(rank, dtype=torch.float64),
                          atol=2e-4, rtol=2e-4):
        raise ValueError("gradient left_vectors must have orthonormal columns")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("gradient cache metadata must be a mapping")
    return vectors.clone(), singular.clone(), dict(metadata)


class OEditRegularizer:
    """One editor run's accumulated actual-update geometry and loss state."""

    def __init__(
        self, *, model_name: str, weight_name: str, weight_shape: Sequence[int],
        output_axis: int = 0, lambda_history: float = 1.0,
        lambda_gradient: float = 1.0, gradient_rank_per_edit: float = 1.0,
        gradient_cache_path: Optional[str] = None,
        loss_type: str = "mean_abs_cosine", batch_size: int = 1,
    ) -> None:
        if int(batch_size) != 1:
            raise ValueError("O-Edit currently supports sequential batch_size=1 only")
        if output_axis not in (0, 1):
            raise ValueError("output_axis must be 0 (Linear) or 1 (Conv1D)")
        shape = tuple(int(x) for x in weight_shape)
        if len(shape) != 2 or any(x <= 0 for x in shape):
            raise ValueError("weight_shape must contain two positive dimensions")
        if not model_name or not weight_name:
            raise ValueError("model_name and weight_name provenance are required")
        if loss_type not in LOSS_TYPES:
            raise ValueError(f"Unsupported O-Edit loss_type: {loss_type}")
        self.model_name, self.weight_name = str(model_name), str(weight_name)
        self.weight_shape = shape
        self.output_axis = output_axis
        self.output_dim, self.input_dim = shape[output_axis], shape[1 - output_axis]
        self.lambda_history = _nonnegative_finite(lambda_history, "lambda_history")
        self.lambda_gradient = _nonnegative_finite(lambda_gradient, "lambda_gradient")
        self.gradient_rank_per_edit = _nonnegative_finite(
            gradient_rank_per_edit, "gradient_rank_per_edit")
        self.loss_type = loss_type
        self.gradient_cache_path = str(gradient_cache_path or "")
        self.gradient_vectors = torch.empty(self.output_dim, 0, dtype=torch.float64)
        self.gradient_singular_values = torch.empty(0, dtype=torch.float64)
        self.gradient_metadata: Dict[str, Any] = {}
        if self.lambda_gradient > 0:
            (self.gradient_vectors, self.gradient_singular_values,
             self.gradient_metadata) = _read_gradient_cache(
                self.gradient_cache_path, model_name=self.model_name,
                weight_name=self.weight_name, weight_shape=self.weight_shape,
                output_dim=self.output_dim)
        self.reset()

    def reset(self) -> None:
        """Clear committed edits, retaining only the immutable original-model cache."""
        self.edit_count = 0
        self.left_vectors = torch.empty(self.output_dim, 0, dtype=torch.float64)
        self.singular_values = torch.empty(0, dtype=torch.float64)
        self.right_vectors = torch.empty(self.input_dim, 0, dtype=torch.float64)
        self._basis_cache: Dict[Any, Any] = {}
        self.last_commit_diagnostics: Dict[str, Any] = {}

    @property
    def numerical_rank(self) -> int:
        return int(self.singular_values.numel())

    @property
    def history_basis(self) -> torch.Tensor:
        """Top min(number of committed edits, numerical rank) LEFT directions."""
        return self.left_vectors[:, :min(self.edit_count, self.numerical_rank)]

    def _loss_bases(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if "cpu" in self._basis_cache:
            return self._basis_cache["cpu"]
        history = self.history_basis
        # Current edit is 1-indexed; the count changes only after a committed update.
        requested_q = math.floor(self.gradient_rank_per_edit * (self.edit_count + 1))
        q = min(requested_q, min(self.weight_shape))
        gradient = self.gradient_vectors[:, :0]
        if self.lambda_gradient > 0 and q:
            available = self.gradient_vectors.shape[1]
            if q > available and not self.gradient_metadata.get("full_rank_basis", False):
                raise ValueError(
                    f"O-Edit gradient cache has {available} vectors but edit "
                    f"{self.edit_count + 1} requires {q}; prepare a larger cache")
            selected = min(q, available)
            gradient = self.gradient_vectors[:, :selected]
            values = self.gradient_singular_values[:selected]
            cutoff = max(_SVD_ATOL, _SVD_RTOL * float(values[0])) if values.numel() else _SVD_ATOL
            gradient = gradient * (values > cutoff).unsqueeze(0)
            # Eq. 12: remove already represented historical output directions.
            # Preserve projected original columns, rather than replace them by
            # a QR basis (which would change a mean-cosine objective).
            gradient = gradient - history @ (history.T @ gradient)
            norms = torch.linalg.vector_norm(gradient, dim=0)
            # Eq. 27 averages over the selected q directions. Define cosine
            # with a numerically collapsed projected vector as zero; dropping
            # it would incorrectly strengthen every surviving direction.
            gradient = (gradient / norms.clamp_min(_COS_EPS).unsqueeze(0)
                        * (norms > _COS_EPS).unsqueeze(0))
            if selected < q:
                # A declared complete numerical basis may omit null singular
                # directions. They still occupy zero-valued slots in the mean.
                gradient = torch.cat((gradient, gradient.new_zeros(self.output_dim, q - selected)), dim=1)
        result = (history, gradient, requested_q)
        self._basis_cache["cpu"] = result
        return result

    def _penalty(self, delta: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        if not basis.shape[1]:
            return delta.sum() * 0.0
        cosine = (basis.T @ delta) / torch.linalg.vector_norm(delta).clamp_min(_COS_EPS)
        if self.loss_type == "mean_abs_cosine":
            return cosine.abs().mean()
        if self.loss_type == "mean_squared_cosine":
            return cosine.square().mean()
        return cosine.mean()

    def loss(self, delta: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if delta.ndim != 1 or delta.numel() != self.output_dim:
            raise ValueError("O-Edit delta must be one vector in the weight OUTPUT space")
        if not delta.is_floating_point() or not bool(torch.isfinite(delta.detach()).all()):
            raise ValueError("O-Edit delta must be finite and floating point")
        dtype = torch.float64 if delta.dtype == torch.float64 else torch.float32
        candidate = delta.to(dtype=dtype)
        history, gradient, requested_q = self._loss_bases()
        device_key = (str(candidate.device), dtype)
        if device_key not in self._basis_cache:
            self._basis_cache[device_key] = (
                history.to(device=candidate.device, dtype=dtype),
                gradient.to(device=candidate.device, dtype=dtype))
        history_device, gradient_device = self._basis_cache[device_key]
        history_loss = (self._penalty(candidate, history_device) if self.lambda_history
                        else candidate.sum() * 0.0)
        gradient_loss = (self._penalty(candidate, gradient_device) if self.lambda_gradient
                         else candidate.sum() * 0.0)
        total = self.lambda_history * history_loss + self.lambda_gradient * gradient_loss
        return total, {
            "oedit_enabled": True, "oedit_loss_type": self.loss_type,
            "oedit_edit_index": self.edit_count + 1,
            "oedit_committed_edits": self.edit_count,
            "oedit_cumulative_numerical_rank": self.numerical_rank,
            "oedit_history_rank": int(history.shape[1]),
            "oedit_gradient_rank_requested": requested_q,
            "oedit_gradient_rank_available": int(self.gradient_vectors.shape[1]),
            "oedit_gradient_rank_after_overlap": int((torch.linalg.vector_norm(gradient, dim=0) > _COS_EPS).sum()),
            "oedit_gradient_mean_denominator": int(gradient.shape[1]),
            "oedit_projected_zero_convention": "zero_contribution_preserve_selected_q",
            "oedit_lambda_history": self.lambda_history,
            "oedit_lambda_gradient": self.lambda_gradient,
            "oedit_history_loss": float(history_loss.detach()),
            "oedit_gradient_loss": float(gradient_loss.detach()),
            "oedit_weighted_loss": float(total.detach()),
            "oedit_scope": "final_edited_weight_output_space",
            "oedit_gradient_cache_path": self.gradient_cache_path,
            "oedit_reproduction": "paper_based_explicit_sim_interpretation",
        }

    def commit_factors(self, left: torch.Tensor, right: torch.Tensor) -> Dict[str, Any]:
        """Add actual applied Delta W = left @ right.T, then advance edit count.

        All numerical-rank components of the cumulative matrix are retained;
        no truncation to the number of edits or FIFO window is performed here.
        """
        left = _cpu_matrix(left, "update left factor")
        right = _cpu_matrix(right, "update right factor")
        if (left.shape[0] != self.output_dim or right.shape[0] != self.input_dim
                or left.shape[1] != right.shape[1]):
            raise ValueError("O-Edit factors require shapes [output, rank], [input, rank]")
        a = torch.cat((self.left_vectors * self.singular_values.unsqueeze(0), left), dim=1)
        b = torch.cat((self.right_vectors, right), dim=1)
        if a.shape[1]:
            qa, ra = torch.linalg.qr(a, mode="reduced")
            qb, rb = torch.linalg.qr(b, mode="reduced")
            core = ra @ rb.T
            u, singular, vh = torch.linalg.svd(core, full_matrices=False)
            # Reference scale makes exact cancellation rank zero even when QR
            # leaves roundoff, rather than renormalizing roundoff to rank one.
            reference_scale = (float(self.singular_values[0]) if self.numerical_rank else 0.0)
            if left.shape[1]:
                update_gram = (left.T @ left) * (right.T @ right)
                reference_scale += math.sqrt(max(float(update_gram.sum()), 0.0))
            cutoff = max(_SVD_ATOL, _SVD_RTOL * max(reference_scale, float(singular[0])))
            keep = singular > cutoff
            self.left_vectors = (qa @ u[:, keep]).contiguous()
            self.singular_values = singular[keep].contiguous()
            self.right_vectors = (qb @ vh[keep].T).contiguous()
        self.edit_count += 1
        self._basis_cache.clear()
        self.last_commit_diagnostics = {
            "oedit_committed_edits": self.edit_count,
            "oedit_cumulative_numerical_rank": self.numerical_rank,
            "oedit_svd_rtol": _SVD_RTOL,
            "oedit_svd_atol": _SVD_ATOL,
            "oedit_update_input_rank": int(left.shape[1]),
        }
        return dict(self.last_commit_diagnostics)

    def commit_update(self, actual_update: torch.Tensor) -> Dict[str, Any]:
        """Commit a dense native rank-one update, validating its factor recovery.

        Prefer native factors.  Sequential AlphaEdit may return only a dense
        rank-one solve: pivot recovery avoids a huge dense SVD, and records its
        relative Frobenius error.  Non-rank-one updates use a dense SVD
        fallback, which is expensive for large weights.  No additional
        directions are silently discarded by rank-one recovery.
        """
        matrix = _cpu_matrix(actual_update, "actual weight update")
        if tuple(matrix.shape) != self.weight_shape:
            raise ValueError("actual update must have the original stored weight_shape")
        if self.output_axis == 1:
            matrix = matrix.T
        column_energy = matrix.square().sum(dim=0)
        total_energy = float(column_energy.sum())
        if total_energy == 0:
            result = self.commit_factors(matrix[:, :0], matrix.T[:, :0])
            result["oedit_dense_factor_relative_error"] = 0.0
            self.last_commit_diagnostics = result
            return result
        pivot = int(column_energy.argmax())
        left = matrix[:, pivot:pivot + 1]
        right = (left.T @ matrix).T / column_energy[pivot]
        error_energy = 0.0
        for start in range(0, self.input_dim, 128):
            error = matrix[:, start:start + 128] - left @ right[start:start + 128].T
            error_energy += float(error.square().sum())
        relative_error = math.sqrt(error_energy / total_energy)
        dense_svd_fallback = False
        if relative_error > _DENSE_FACTOR_RTOL:
            u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
            cutoff = max(_SVD_ATOL, _SVD_RTOL * float(singular[0]))
            keep = singular > cutoff
            left = u[:, keep] * singular[keep].unsqueeze(0)
            right = vh[keep].T
            dense_svd_fallback = True
        result = self.commit_factors(left, right)
        result["oedit_dense_factor_relative_error"] = relative_error
        result["oedit_dense_svd_fallback"] = dense_svd_fallback
        self.last_commit_diagnostics = result
        return dict(result)


def reset_oedit_state(hparams: Any) -> None:
    """Drop runtime state before each new editor run (not before each edit)."""
    if hasattr(hparams, "_oedit_runtime"):
        delattr(hparams, "_oedit_runtime")


def get_oedit_regularizer(
    hparams: Any, *, model_name: str, weight_name: str,
    weight_shape: Sequence[int], output_axis: int = 0,
) -> Optional[OEditRegularizer]:
    """Disabled is an exact no-op: no cache read or state creation occurs."""
    if not bool(getattr(hparams, "oedit_enabled", False)):
        return None
    kwargs = dict(
        model_name=model_name, weight_name=weight_name, weight_shape=tuple(weight_shape),
        output_axis=output_axis, batch_size=int(getattr(hparams, "batch_size", 1)),
        lambda_history=float(getattr(hparams, "oedit_lambda_history", 1.0)),
        lambda_gradient=float(getattr(hparams, "oedit_lambda_gradient", 1.0)),
        gradient_rank_per_edit=float(getattr(hparams, "oedit_gradient_rank_per_edit", 1.0)),
        gradient_cache_path=str(getattr(hparams, "oedit_gradient_cache_path", "") or ""),
        loss_type=str(getattr(hparams, "oedit_loss_type", "mean_abs_cosine")),
    )
    state = getattr(hparams, "_oedit_runtime", None)
    signature = tuple((key, value) for key, value in kwargs.items())
    if state is None:
        state = OEditRegularizer(**kwargs)
        state._runtime_signature = signature
        setattr(hparams, "_oedit_runtime", state)
    elif not isinstance(state, OEditRegularizer) or state._runtime_signature != signature:
        raise ValueError("O-Edit configuration/model changed: reset_oedit_state before a new run")
    return state
