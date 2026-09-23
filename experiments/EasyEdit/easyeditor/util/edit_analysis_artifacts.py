"""Crash-safe, opt-in artifacts for sequential EasyEdit mechanism analysis.

The training/evaluation runners enable this module by attaching a few dynamic
attributes to the EasyEdit hparams object:

``analysis_artifact_dir``
    Root directory for the artifacts.
``analysis_current_edit_index``
    One-based cumulative edit index.
``analysis_capture_inner_gain``
    Record the zero-delta reference and every inner-loop residual-gain state.
``analysis_capture_latent_norms``
    Append only the final target/init scalar norms to one small JSONL file.
``analysis_capture_latents`` / ``analysis_capture_outer``
    Save latent vectors and outer-solve sufficient statistics.

Nothing is recorded, and no optimization behavior is changed, unless the
corresponding flag is explicitly enabled.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch


def _enabled(hparams: Any, name: str) -> bool:
    return bool(getattr(hparams, name, False)) and bool(
        getattr(hparams, "analysis_artifact_dir", None)
    )


def captures_inner_gain(hparams: Any) -> bool:
    return _enabled(hparams, "analysis_capture_inner_gain")


def captures_latents(hparams: Any) -> bool:
    return _enabled(hparams, "analysis_capture_latents")


def captures_latent_norms(hparams: Any) -> bool:
    return _enabled(hparams, "analysis_capture_latent_norms")


def captures_outer(hparams: Any) -> bool:
    return _enabled(hparams, "analysis_capture_outer")


def _root(hparams: Any) -> Path:
    root = Path(str(getattr(hparams, "analysis_artifact_dir"))).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_case_id(request: Mapping[str, Any]) -> str:
    raw = str(request.get("case_id", "unknown"))
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return safe or "unknown"


def edit_identity(
    hparams: Any,
    request: Mapping[str, Any],
) -> tuple[int, str, str]:
    index = int(
        getattr(
            hparams,
            "analysis_current_edit_index",
            request.get("case_id", -1),
        )
    )
    case_id = _safe_case_id(request)
    stem = f"edit_{index:06d}_case_{case_id}"
    return index, case_id, stem


def _atomic_jsonl_append(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(dict(record), ensure_ascii=False, default=str) + "\n"
        )
        handle.flush()


def _atomic_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(record), handle, ensure_ascii=False, indent=2, default=str)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez(temporary, **dict(arrays))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _array(
    value: torch.Tensor,
    *,
    compact: bool,
) -> np.ndarray:
    tensor = value.detach().to(device="cpu")
    if tensor.is_floating_point():
        tensor = tensor.float()
        if compact:
            tensor = tensor.half()
    return tensor.numpy()


def record_inner_gain_step(
    hparams: Any,
    *,
    method: str,
    request: Mapping[str, Any],
    write_layer: int,
    inner_step: int,
    efficacy_score: float,
    base_edit_loss: float,
    optimization_loss: float,
    diagnostics: Mapping[str, Any],
) -> None:
    """Append one scalar inner-loop gain observation."""

    if not captures_inner_gain(hparams):
        return
    edit_index, case_id, stem = edit_identity(hparams, request)
    path = _root(hparams) / "inner_gain" / f"{stem}.jsonl"
    record = {
            "method": method,
            "edit_index": edit_index,
            "case_id": request.get("case_id", case_id),
            "write_layer": int(write_layer),
            "inner_step": int(inner_step),
            "is_zero_delta_reference": bool(
                diagnostics.get("reference_initialized", False)
            ),
            "efficacy_score": float(efficacy_score),
            "base_edit_loss": float(base_edit_loss),
            "optimization_loss": float(optimization_loss),
            **dict(diagnostics),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if int(inner_step) == 0 else "a"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def save_latent_artifact(
    hparams: Any,
    *,
    method: str,
    request: Mapping[str, Any],
    write_layer: int,
    target_init: torch.Tensor,
    optimizer_delta: torch.Tensor,
    target: torch.Tensor,
) -> Optional[Path]:
    """Save ``v0``, optimizer ``delta``, effective ``delta*``, and ``v*``."""

    capture_vectors = captures_latents(hparams)
    capture_norms = captures_latent_norms(hparams)
    if not capture_vectors and not capture_norms:
        return None
    edit_index, case_id, stem = edit_identity(hparams, request)
    effective_delta = target.detach().to(target_init.device) - target_init.detach()
    target_init_l2 = float(target_init.detach().float().norm().item())
    optimizer_delta_l2 = float(optimizer_delta.detach().float().norm().item())
    delta_star_l2 = float(effective_delta.detach().float().norm().item())
    target_l2 = float(target.detach().float().norm().item())
    target_to_init_ratio = (
        target_l2 / target_init_l2 if target_init_l2 > 0.0 else float("nan")
    )
    scalar_record = {
        "method": method,
        "edit_index": edit_index,
        "case_id": request.get("case_id", case_id),
        "write_layer": int(write_layer),
        "target_init_l2": target_init_l2,
        "optimizer_delta_l2": optimizer_delta_l2,
        "delta_star_l2": delta_star_l2,
        # ``v_star_l2`` is retained for compatibility with the historical
        # artifact schema; the optimized object here is the full block target.
        "v_star_l2": target_l2,
        "target_l2": target_l2,
        "target_to_init_ratio": target_to_init_ratio,
    }
    if capture_norms:
        _atomic_jsonl_append(_root(hparams) / "latent_norms.jsonl", scalar_record)
    if not capture_vectors:
        return _root(hparams) / "latent_norms.jsonl"

    path = _root(hparams) / "latents" / f"{stem}.npz"
    _atomic_npz(
        path,
        {
            "target_init": _array(target_init, compact=False),
            "optimizer_delta": _array(optimizer_delta, compact=False),
            "delta_star": _array(effective_delta, compact=False),
            "v_star": _array(target, compact=False),
            "edit_index": np.asarray(edit_index, dtype=np.int64),
            "write_layer": np.asarray(int(write_layer), dtype=np.int64),
        },
    )
    _atomic_json(
        path.with_suffix(".json"),
        {
            **scalar_record,
            "path": str(path.relative_to(_root(hparams))),
        },
    )
    return path


def save_outer_layer_artifact(
    hparams: Any,
    *,
    method: str,
    requests: list[Mapping[str, Any]],
    layer: int,
    keys: torch.Tensor,
    target_error: torch.Tensor,
    distributed_residual: torch.Tensor,
    realized_update: torch.Tensor,
    update_frobenius: float,
) -> Optional[Path]:
    """Save per-edit/per-layer outer sufficient statistics.

    Batch-size one is the intended analysis protocol.  The array layout is
    nevertheless kept as ``[dimension, records]`` so a future batch run remains
    unambiguous.
    """

    if not captures_outer(hparams):
        return None
    if len(requests) != 1:
        raise ValueError(
            "Per-edit outer artifact capture currently requires batch size 1; "
            f"received {len(requests)} requests"
        )
    request = requests[0]
    edit_index, case_id, stem = edit_identity(hparams, request)
    path = _root(hparams) / "outer" / f"{stem}.npz"
    arrays: Dict[str, np.ndarray] = {}
    if path.exists():
        with np.load(path, allow_pickle=False) as current:
            arrays.update({key: current[key] for key in current.files})
    suffix = f"L{int(layer)}"
    arrays.update(
        {
            f"k_star__{suffix}": _array(keys, compact=True),
            f"target_error__{suffix}": _array(target_error, compact=True),
            f"distributed_residual__{suffix}": _array(
                distributed_residual, compact=True
            ),
            f"delta_w_k__{suffix}": _array(realized_update, compact=True),
            "edit_index": np.asarray(edit_index, dtype=np.int64),
        }
    )
    _atomic_npz(path, arrays)

    target_flat = target_error.detach().float().reshape(-1)
    realized_flat = realized_update.detach().float().reshape(-1)
    cosine = float("nan")
    if target_flat.numel() == realized_flat.numel():
        cosine = float(
            torch.nn.functional.cosine_similarity(
                target_flat.unsqueeze(0),
                realized_flat.unsqueeze(0),
                dim=-1,
            ).item()
        )
    summary_path = path.with_suffix(".jsonl")
    summary_record = {
            "method": method,
            "edit_index": edit_index,
            "case_id": request.get("case_id", case_id),
            "layer": int(layer),
            "path": str(path.relative_to(_root(hparams))),
            "key_l2": float(keys.detach().float().norm().item()),
            "target_error_l2": float(target_error.detach().float().norm().item()),
            "distributed_residual_l2": float(
                distributed_residual.detach().float().norm().item()
            ),
            "delta_w_k_l2": float(realized_update.detach().float().norm().item()),
            "delta_w_frobenius": float(update_frobenius),
            "target_error_delta_w_k_cosine": cosine,
        }
    summary_mode = (
        "w"
        if int(layer) == min(int(value) for value in getattr(hparams, "layers"))
        else "a"
    )
    with summary_path.open(summary_mode, encoding="utf-8") as handle:
        handle.write(
            json.dumps(summary_record, ensure_ascii=False, default=str) + "\n"
        )
    return path


def save_committed_outer_layer_artifact(
    hparams: Any,
    *,
    method: str,
    requests: list[Mapping[str, Any]],
    layer: int,
    committed_update: torch.Tensor,
    stage: str = "post_projection",
) -> Optional[Path]:
    """Augment an outer artifact with the update that is actually committed.

    ``save_outer_layer_artifact`` runs inside the native MEMIT/AlphaEdit solve.
    A post-processing editor such as SPHERE changes that update afterwards, so
    its original ``delta_w_k`` is a pre-projection diagnostic.  This function
    stores both the post-projection realization and the projection-induced
    change without discarding the native-solve record.
    """

    if not captures_outer(hparams):
        return None
    if len(requests) != 1:
        raise ValueError(
            "Committed outer artifact capture requires batch size 1; "
            f"received {len(requests)} requests"
        )
    request = requests[0]
    edit_index, case_id, stem = edit_identity(hparams, request)
    path = _root(hparams) / "outer" / f"{stem}.npz"
    if not path.exists():
        raise FileNotFoundError(
            "Native outer artifact is missing before committed-update "
            f"capture: {path}"
        )

    with np.load(path, allow_pickle=False) as current:
        arrays: Dict[str, np.ndarray] = {
            key: current[key] for key in current.files
        }
    suffix = f"L{int(layer)}"
    key_name = f"k_star__{suffix}"
    target_name = f"target_error__{suffix}"
    native_name = f"delta_w_k__{suffix}"
    missing = [
        name for name in (key_name, target_name, native_name) if name not in arrays
    ]
    if missing:
        raise KeyError(f"Outer artifact {path} is missing arrays: {missing}")

    update = committed_update.detach().float()
    keys = torch.from_numpy(arrays[key_name]).to(
        device=update.device,
        dtype=update.dtype,
    )
    committed_realization = update @ keys
    native_realization = torch.from_numpy(arrays[native_name]).to(
        device=update.device,
        dtype=update.dtype,
    )
    postprocess_realization = committed_realization - native_realization
    arrays.update(
        {
            f"committed_delta_w_k__{suffix}": _array(
                committed_realization, compact=True
            ),
            f"postprocess_delta_w_k__{suffix}": _array(
                postprocess_realization, compact=True
            ),
        }
    )
    _atomic_npz(path, arrays)

    target = torch.from_numpy(arrays[target_name]).to(
        device=update.device,
        dtype=update.dtype,
    )
    target_flat = target.reshape(-1)
    committed_flat = committed_realization.reshape(-1)
    cosine = float("nan")
    if target_flat.numel() == committed_flat.numel():
        cosine = float(
            torch.nn.functional.cosine_similarity(
                target_flat.unsqueeze(0),
                committed_flat.unsqueeze(0),
                dim=-1,
            ).item()
        )

    summary_path = path.with_suffix(".jsonl")
    records = []
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    matched = False
    for record in records:
        if int(record.get("layer", -1)) != int(layer):
            continue
        record.update(
            {
                "committed_update_stage": str(stage),
                "committed_delta_w_k_l2": float(
                    committed_realization.norm().item()
                ),
                "postprocess_delta_w_k_l2": float(
                    postprocess_realization.norm().item()
                ),
                "committed_delta_w_frobenius": float(update.norm().item()),
                "committed_target_error_delta_w_k_cosine": cosine,
            }
        )
        matched = True
        break
    if not matched:
        raise KeyError(
            f"Outer summary {summary_path} has no record for layer {layer}"
        )
    temporary = summary_path.with_name(
        f".{summary_path.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, default=str) + "\n"
                )
        os.replace(temporary, summary_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path
