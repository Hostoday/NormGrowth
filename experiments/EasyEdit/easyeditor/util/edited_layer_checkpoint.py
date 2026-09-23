"""Compact checkpoints for EasyEdit methods that modify selected modules only.

MEMIT and AlphaEdit in this repository write to ``mlp.down_proj.weight`` in
the configured rewrite layers.  Saving those tensors is sufficient to
reconstruct the edited model from the original Hugging Face checkpoint; the
remaining model parameters are unchanged and need not be duplicated.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch


FORMAT_NAME = "easyedit-edited-parameters"
FORMAT_VERSION = 1
STORAGE_ABSOLUTE = "absolute_parameters"
STORAGE_DELTA = "parameter_deltas"


def rewrite_parameter_names(hparams: Any, layers: Sequence[int]) -> list[str]:
    """Resolve the exact parameter names modified by MEMIT/AlphaEdit."""

    template = getattr(hparams, "rewrite_module_tmp", None)
    if not template:
        raise ValueError("hparams has no rewrite_module_tmp")
    return [f"{template.format(int(layer))}.weight" for layer in layers]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def save_edited_parameter_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    parameter_names: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Save only the named tensors and an adjacent JSON manifest."""

    path = Path(checkpoint_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    named_parameters = dict(model.named_parameters())
    missing = [name for name in parameter_names if name not in named_parameters]
    if missing:
        raise KeyError(f"Model has no requested edited parameters: {missing}")

    state_dict = {
        name: named_parameters[name].detach().to(device="cpu").clone()
        for name in parameter_names
    }
    tensor_metadata = {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()),
        }
        for name, tensor in state_dict.items()
    }
    manifest: Dict[str, Any] = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "checkpoint_file": path.name,
        "parameter_names": list(parameter_names),
        "tensor_metadata": tensor_metadata,
        "total_numel": int(sum(tensor.numel() for tensor in state_dict.values())),
        "storage_mode": STORAGE_ABSOLUTE,
        **dict(metadata or {}),
    }
    payload = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "metadata": manifest,
        "state_dict": state_dict,
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    _atomic_json(path.with_suffix(".json"), manifest)
    return manifest


def snapshot_parameters(
    model: torch.nn.Module,
    parameter_names: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """Clone selected Base parameters to CPU for later delta serialization."""

    named_parameters = dict(model.named_parameters())
    missing = [name for name in parameter_names if name not in named_parameters]
    if missing:
        raise KeyError(f"Model has no requested parameters: {missing}")
    return {
        name: named_parameters[name].detach().to(device="cpu").clone()
        for name in parameter_names
    }


def save_parameter_delta_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    base_parameters: Mapping[str, torch.Tensor],
    parameter_names: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Save cumulative ``current - Base`` tensors for selected parameters."""

    path = Path(checkpoint_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    named_parameters = dict(model.named_parameters())
    missing_model = [name for name in parameter_names if name not in named_parameters]
    missing_base = [name for name in parameter_names if name not in base_parameters]
    if missing_model or missing_base:
        raise KeyError(
            "Cannot form parameter deltas; "
            f"missing_model={missing_model}, missing_base={missing_base}"
        )
    state_dict = {
        name: (
            named_parameters[name].detach().to(device="cpu")
            - base_parameters[name].to(
                device="cpu",
                dtype=named_parameters[name].dtype,
            )
        )
        for name in parameter_names
    }
    tensor_metadata = {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()),
        }
        for name, tensor in state_dict.items()
    }
    manifest: Dict[str, Any] = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "checkpoint_file": path.name,
        "parameter_names": list(parameter_names),
        "tensor_metadata": tensor_metadata,
        "total_numel": int(sum(tensor.numel() for tensor in state_dict.values())),
        "storage_mode": STORAGE_DELTA,
        **dict(metadata or {}),
    }
    payload = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "metadata": manifest,
        "state_dict": state_dict,
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    _atomic_json(path.with_suffix(".json"), manifest)
    return manifest


def load_edited_parameter_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
) -> Dict[str, Any]:
    """Apply a compact checkpoint to a freshly loaded Base model."""

    path = Path(checkpoint_path).expanduser().resolve()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("format") != FORMAT_NAME:
        raise ValueError(f"{path} is not a {FORMAT_NAME!r} checkpoint")
    if int(payload.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported edited-parameter checkpoint version: "
            f"{payload.get('format_version')}"
        )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"{path} contains no edited state_dict")

    model_parameters = dict(model.named_parameters())
    unexpected = sorted(set(state_dict) - set(model_parameters))
    if unexpected:
        raise KeyError(f"Checkpoint parameters are absent from the model: {unexpected}")
    storage_mode = str(
        (payload.get("metadata") or {}).get(
            "storage_mode",
            STORAGE_ABSOLUTE,
        )
    )
    if storage_mode not in {STORAGE_ABSOLUTE, STORAGE_DELTA}:
        raise ValueError(f"Unsupported checkpoint storage_mode={storage_mode!r}")
    with torch.no_grad():
        for name, tensor in state_dict.items():
            destination = model_parameters[name]
            value = tensor.to(
                device=destination.device,
                dtype=destination.dtype,
            )
            if storage_mode == STORAGE_DELTA:
                destination.add_(value)
            else:
                destination.copy_(value)
    return dict(payload.get("metadata") or {})
