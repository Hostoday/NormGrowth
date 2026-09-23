"""Norm Anchor Scaling (NAS) for latent-value model editors.

NAS is applied after MEMIT has optimized its latent displacement ``delta`` and
before the regular closed-form outer update is computed.  Following the
released NAS decomposition,

    v_star = v0 + delta
    target = (target_init - v0) + NAS(v_star),

only the MLP-output component is norm-anchored.  The residual component that
enters the edited MLP is left unchanged.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass(frozen=True)
class NormAnchorResult:
    target: torch.Tensor
    method: str
    case_id: Any
    layer: int
    enabled: bool
    collect_stats: bool
    anchor_norm: Optional[float]
    outlier_factor: float
    outlier_mode: str
    safeguard_triggered: bool
    v0_norm: float
    delta_norm: float
    target_init_norm: float
    residual_pre_mlp_norm: float
    vstar_norm_before: float
    vstar_norm_after: float
    target_norm_before: float
    target_norm_after: float
    scale_factor: float

    def to_record(self) -> Dict[str, Any]:
        # Avoid dataclasses.asdict(), which attempts to deepcopy the tensor.
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "target"
        }


def scale_vector_to_norm(
    vector: torch.Tensor,
    target_norm: float,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Preserve ``vector`` direction while setting its L2 norm."""

    target_norm = float(target_norm)
    if not math.isfinite(target_norm) or target_norm <= 0.0:
        raise ValueError(
            f"NAS anchor norm must be finite and positive, got {target_norm}"
        )
    norm = vector.norm()
    if not torch.isfinite(norm):
        raise ValueError("NAS received a non-finite v* norm")
    if float(norm.detach().cpu()) <= eps:
        raise ValueError("NAS cannot rescale a zero-norm v* vector")
    return vector * (target_norm / norm)


@lru_cache(maxsize=32)
def _read_anchor_json(path_string: str) -> Dict[str, Any]:
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(f"NAS anchor file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"NAS anchor file must contain a JSON object: {path}")
    return payload


def _first_anchor_value(mapping: Any) -> Optional[float]:
    if not isinstance(mapping, dict):
        return None
    for key in ("anchor_norm", "mean_vstar_norm", "mean_norm"):
        value = mapping.get(key)
        if value is not None:
            return float(value)
    return None


def load_anchor_norm(
    path: str,
    *,
    layer: int,
    method: Optional[str] = None,
) -> float:
    """Load a NAS anchor from the local or a compact compatible schema."""

    payload = _read_anchor_json(str(Path(path).resolve()))
    declared_method = payload.get("method")
    # The released NAS code estimates the reference with MEMIT hparams and
    # reuses it for other L&E outer solvers, including AlphaEdit.
    method_pair = (
        str(declared_method).lower() if declared_method is not None else None,
        str(method).lower() if method is not None else None,
    )
    official_cross_editor_anchor = method_pair == ("memit", "alphaedit")
    if (
        method
        and declared_method is not None
        and str(declared_method).lower() != str(method).lower()
        and not official_cross_editor_anchor
    ):
        raise ValueError(
            f"NAS anchor method mismatch: file declares {declared_method!r}, "
            f"run requested {method!r}"
        )
    declared_layer = payload.get("layer")
    if declared_layer is not None and int(declared_layer) != int(layer):
        raise ValueError(
            f"NAS anchor layer mismatch: file declares L{declared_layer}, "
            f"run requested L{layer}"
        )

    candidates = []
    if method:
        methods = payload.get("methods")
        if isinstance(methods, dict):
            method_payload = methods.get(method) or methods.get(method.lower())
            candidates.append(method_payload)
            if isinstance(method_payload, dict):
                layers = method_payload.get("layers")
                if isinstance(layers, dict):
                    candidates.extend(
                        (layers.get(str(layer)), layers.get(layer))
                    )
    layers = payload.get("layers")
    if isinstance(layers, dict):
        candidates.extend((layers.get(str(layer)), layers.get(layer)))
    candidates.append(payload)

    for candidate in candidates:
        value = _first_anchor_value(candidate)
        if value is not None:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"Invalid NAS anchor norm {value} in {Path(path).resolve()}"
                )
            return value
    raise KeyError(
        f"No anchor norm for method={method!r}, layer={layer} "
        f"in {Path(path).resolve()}"
    )


def _resolve_anchor(hparams: Any, *, layer: int, method: str) -> Optional[float]:
    explicit = getattr(hparams, "nas_anchor_norm", None)
    if explicit is not None:
        value = float(explicit)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"nas_anchor_norm must be finite and positive, got {value}"
            )
        return value
    path = getattr(hparams, "nas_anchor_path", None)
    if path:
        return load_anchor_norm(str(path), layer=layer, method=method)
    return None


def _append_jsonl(path_string: Optional[str], record: Dict[str, Any]) -> None:
    if not path_string:
        return
    path = Path(path_string)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def apply_norm_anchor_scaling(
    *,
    target_init: torch.Tensor,
    delta: torch.Tensor,
    v0: torch.Tensor,
    hparams: Any,
    method: str,
    request: Dict[str, Any],
    layer: int,
) -> NormAnchorResult:
    """Apply NAS, or collect the same decomposition without changing the edit."""

    enabled = bool(getattr(hparams, "nas_enabled", False))
    collect_stats = bool(getattr(hparams, "nas_collect_stats", False))
    if not enabled and not collect_stats:
        target = target_init + delta.to(target_init.device)
        norm = float(target.detach().norm().cpu())
        return NormAnchorResult(
            target=target,
            method=method,
            case_id=request.get("case_id"),
            layer=int(layer),
            enabled=False,
            collect_stats=False,
            anchor_norm=None,
            outlier_factor=float(getattr(hparams, "nas_outlier_factor", 2.0)),
            outlier_mode=str(getattr(hparams, "nas_outlier_mode", "skip_delta")),
            safeguard_triggered=False,
            v0_norm=float("nan"),
            delta_norm=float(delta.detach().norm().cpu()),
            target_init_norm=float(target_init.detach().norm().cpu()),
            residual_pre_mlp_norm=float("nan"),
            vstar_norm_before=float("nan"),
            vstar_norm_after=float("nan"),
            target_norm_before=norm,
            target_norm_after=norm,
            scale_factor=1.0,
        )

    # Match PyTorch's native ``target_init + delta`` promotion.  In the Llama
    # setup target_init/v0 are often bf16 while the optimized delta is fp32.
    compute_dtype = torch.promote_types(target_init.dtype, delta.dtype)
    target_init_value = target_init.detach().to(dtype=compute_dtype)
    delta = delta.detach().to(
        device=target_init.device,
        dtype=compute_dtype,
    )
    v0 = v0.detach().to(
        device=target_init.device,
        dtype=compute_dtype,
    )
    if target_init.shape != delta.shape or target_init.shape != v0.shape:
        raise ValueError(
            "NAS expects target_init, delta, and v0 to have identical shapes, "
            f"got {tuple(target_init.shape)}, {tuple(delta.shape)}, "
            f"{tuple(v0.shape)}"
        )

    residual_pre_mlp = target_init_value - v0
    vstar_raw = v0 + delta
    native_target = target_init_value + delta
    anchor = _resolve_anchor(hparams, layer=int(layer), method=method)
    if enabled and anchor is None:
        raise ValueError(
            "NAS is enabled but neither nas_anchor_norm nor nas_anchor_path is set"
        )

    outlier_factor = float(getattr(hparams, "nas_outlier_factor", 2.0))
    if not math.isfinite(outlier_factor) or outlier_factor <= 0.0:
        raise ValueError(
            "nas_outlier_factor must be finite and positive, "
            f"got {outlier_factor}"
        )
    outlier_mode = str(
        getattr(hparams, "nas_outlier_mode", "skip_delta")
    ).lower()
    if outlier_mode not in {"skip_delta", "scale"}:
        raise ValueError(
            "nas_outlier_mode must be 'skip_delta' or 'scale', "
            f"got {outlier_mode!r}"
        )

    raw_norm = float(vstar_raw.norm().cpu())
    safeguard = bool(
        enabled
        and outlier_mode == "skip_delta"
        and anchor is not None
        and raw_norm > outlier_factor * anchor
    )
    if not enabled:
        vstar_effective = vstar_raw
    elif safeguard:
        # Released NAS safeguard: discard delta for this outlier request.
        vstar_effective = v0
    else:
        vstar_effective = scale_vector_to_norm(vstar_raw, float(anchor))

    target = residual_pre_mlp + vstar_effective
    after_norm = float(vstar_effective.norm().cpu())
    result = NormAnchorResult(
        target=target,
        method=method,
        case_id=request.get("case_id"),
        layer=int(layer),
        enabled=enabled,
        collect_stats=collect_stats,
        anchor_norm=anchor,
        outlier_factor=outlier_factor,
        outlier_mode=outlier_mode,
        safeguard_triggered=safeguard,
        v0_norm=float(v0.norm().cpu()),
        delta_norm=float(delta.norm().cpu()),
        target_init_norm=float(target_init_value.norm().cpu()),
        residual_pre_mlp_norm=float(residual_pre_mlp.norm().cpu()),
        vstar_norm_before=raw_norm,
        vstar_norm_after=after_norm,
        target_norm_before=float(native_target.norm().cpu()),
        target_norm_after=float(target.norm().cpu()),
        scale_factor=(
            after_norm / raw_norm if raw_norm > 0.0 else float("nan")
        ),
    )
    _append_jsonl(
        getattr(hparams, "nas_log_path", None),
        result.to_record(),
    )
    return result
