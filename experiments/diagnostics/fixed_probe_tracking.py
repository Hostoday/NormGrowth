#!/usr/bin/env python3
"""Lightweight per-edit tracking on a fixed, disjoint prompt panel.

The tracker is deliberately observer-only: it attaches forward hooks to the
already-loaded editor model and performs the model's ordinary full forward.
It never calls a decoder block manually, changes attention implementations,
or participates in the editing loss.

For the usual L4--L8 editing boundary it records, on the same probes after
every edit, ``H8``, ``F8 = H9 - H8``, attention/MLP writes, and the exact
finite state-energy decomposition

    G = (||H9||^2 - ||H8||^2) / ||H8||^2 = r^2 + 2 r c.

It also records ``rho = ||H9|| / ||H8||``, ``rho - 1``, ``log(rho)``, and the
temporal increment of the *same* probe's H9 relative to the preceding edit.
Repeated-measures inference must cluster/bootstrap over probes; edit x probe
rows are not independent samples.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch

from diagnostics.analyze_residual_spectrum import (
    Probe,
    _layer_hidden_output,
    load_probes,
    model_input_device,
    model_layers,
)


EPS = 1e-12
SCHEMA_VERSION = 1
DEFAULT_CONTEXTS = (
    "edit_subject_last",
    "edit_prompt_last",
    "locality_prompt_last",
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray], *, compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        writer = np.savez_compressed if compressed else np.savez
        writer(temporary, **dict(arrays))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_contexts(text: str | Sequence[str]) -> Tuple[str, ...]:
    raw = text.split(",") if isinstance(text, str) else text
    values = tuple(str(value).strip() for value in raw if str(value).strip())
    allowed = set(DEFAULT_CONTEXTS)
    unknown = sorted(set(values) - allowed)
    if not values or unknown:
        raise ValueError(
            f"fixed-probe contexts must be drawn from {sorted(allowed)}; "
            f"got {values}, unknown={unknown}"
        )
    if len(values) != len(set(values)):
        raise ValueError("fixed-probe contexts contain duplicates")
    return values


def load_fixed_probe_panel(
    path: Path,
    tokenizer: Any,
    *,
    count: int,
    max_length: int,
    contexts: Sequence[str] = DEFAULT_CONTEXTS,
) -> Dict[str, Tuple[List[Probe], Tuple[str, ...]]]:
    """Materialize one shared JSON panel as edit and locality prompt families."""

    contexts = parse_contexts(contexts)
    output: Dict[str, Tuple[List[Probe], Tuple[str, ...]]] = {}
    edit_positions: List[str] = []
    if "edit_subject_last" in contexts:
        edit_positions.append("subject_last")
    if "edit_prompt_last" in contexts:
        edit_positions.append("prompt_last")
    if edit_positions:
        probes = load_probes(
            str(path),
            tokenizer,
            edit_positions,
            count,
            max_length,
            probe_source="rewrite",
            selection="prefix",
        )
        if len(probes) != count:
            raise RuntimeError(
                f"fixed edit panel materialized {len(probes)} probes, expected {count}"
            )
        output["edit"] = (probes, tuple(edit_positions))
    if "locality_prompt_last" in contexts:
        probes = load_probes(
            str(path),
            tokenizer,
            ("prompt_last",),
            count,
            max_length,
            probe_source="locality",
            selection="prefix",
        )
        if len(probes) != count:
            raise RuntimeError(
                f"fixed locality panel materialized {len(probes)} probes, expected {count}"
            )
        output["locality"] = (probes, ("prompt_last",))
    return output


def _padded_batch(
    probes: Sequence[Probe],
    *,
    pad_token_id: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    max_length = max(len(probe.input_ids) for probe in probes)
    input_ids = torch.full(
        (len(probes), max_length),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, probe in enumerate(probes):
        length = len(probe.input_ids)
        input_ids[row, :length] = torch.as_tensor(
            probe.input_ids, dtype=torch.long, device=device
        )
        attention_mask[row, :length] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _capture_family(
    model: Any,
    tokenizer: Any,
    probes: Sequence[Probe],
    positions: Sequence[str],
    *,
    batch_size: int,
) -> Dict[Tuple[str, str], np.ndarray]:
    """Capture H8/H9/H10 and A/M writes from ordinary batched forwards."""

    decoder = model_layers(model)
    if len(decoder) <= 10:
        raise ValueError("fixed boundary tracking requires decoder layers 8, 9, and 10")
    requested_layers = (8, 9, 10)
    output_lists: MutableMapping[Tuple[str, str], List[np.ndarray]] = defaultdict(list)
    input_device = model_input_device(model)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")

    for start in range(0, len(probes), batch_size):
        batch_probes = probes[start : start + batch_size]
        captured: Dict[str, torch.Tensor] = {}
        handles = []

        for layer_id in requested_layers:
            layer = decoder[layer_id]

            def capture_input(
                _module: Any,
                args: Tuple[Any, ...],
                kwargs: Mapping[str, Any],
                *,
                key: int = layer_id,
            ) -> None:
                hidden = args[0] if args else kwargs.get("hidden_states")
                if not torch.is_tensor(hidden):
                    raise TypeError(f"could not capture H{key}")
                captured[f"H{key}"] = hidden.detach()

            handles.append(layer.register_forward_pre_hook(capture_input, with_kwargs=True))
            if layer_id in (8, 9):

                def capture_attention(
                    _module: Any,
                    _args: Tuple[Any, ...],
                    value: Any,
                    *,
                    key: int = layer_id,
                ) -> None:
                    captured[f"A{key}"] = _layer_hidden_output(value).detach()

                def capture_mlp(
                    _module: Any,
                    _args: Tuple[Any, ...],
                    value: Any,
                    *,
                    key: int = layer_id,
                ) -> None:
                    captured[f"M{key}"] = _layer_hidden_output(value).detach()

                handles.append(layer.self_attn.register_forward_hook(capture_attention))
                handles.append(layer.mlp.register_forward_hook(capture_mlp))

        try:
            inputs = _padded_batch(
                batch_probes,
                pad_token_id=int(pad_token_id),
                device=input_device,
            )
            with torch.no_grad():
                model(
                    **inputs,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()

        expected = {"H8", "H9", "H10", "A8", "M8", "A9", "M9"}
        missing = sorted(expected - set(captured))
        if missing:
            raise RuntimeError(f"fixed-probe hooks missed nodes: {missing}")
        for position in positions:
            for node in sorted(expected):
                tensor = captured[node]
                selected = []
                for row, probe in enumerate(batch_probes):
                    token_index = int(probe.positions[position])
                    selected.append(tensor[row, token_index].detach().float().cpu())
                output_lists[(position, node)].append(
                    torch.stack(selected).numpy().astype(np.float32, copy=False)
                )
        del captured

    return {
        key: np.concatenate(chunks, axis=0)
        for key, chunks in output_lists.items()
    }


def _rms(vector: np.ndarray) -> float:
    value = np.asarray(vector, dtype=np.float64)
    return float(np.sqrt(np.mean(value * value)))


def _finite_gain_row(
    hidden_in: np.ndarray,
    hidden_out: np.ndarray,
    attention: np.ndarray,
    mlp: np.ndarray,
) -> Dict[str, float]:
    h = np.asarray(hidden_in, dtype=np.float64)
    y = np.asarray(hidden_out, dtype=np.float64)
    f = y - h
    input_energy = float(np.mean(h * h))
    output_energy = float(np.mean(y * y))
    write_energy = float(np.mean(f * f))
    gain = (output_energy - input_energy) / (input_energy + EPS)
    r2 = write_energy / (input_energy + EPS)
    alignment = 2.0 * float(np.mean(h * f)) / (input_energy + EPS)
    rho = math.sqrt(max(output_energy, 0.0) / (input_energy + EPS))
    cosine = float(
        np.dot(h, f) / (np.linalg.norm(h) * np.linalg.norm(f) + EPS)
    )
    return {
        "input_rms": math.sqrt(max(input_energy, 0.0)),
        "output_rms": math.sqrt(max(output_energy, 0.0)),
        "total_write_rms": math.sqrt(max(write_energy, 0.0)),
        "attention_write_rms": _rms(attention),
        "mlp_write_rms": _rms(mlp),
        "finite_gain": gain,
        "write_relative_energy": r2,
        "alignment_contribution": alignment,
        "input_write_cosine": cosine,
        "decomposition_error": gain - r2 - alignment,
        "rho": rho,
        "rho_minus_one": rho - 1.0,
        "log_rho": math.log(max(rho, EPS)),
    }


def _temporal_row(previous: np.ndarray, current: np.ndarray) -> Dict[str, float]:
    left = np.asarray(previous, dtype=np.float64)
    right = np.asarray(current, dtype=np.float64)
    delta = right - left
    previous_energy = float(np.mean(left * left))
    current_energy = float(np.mean(right * right))
    delta_relative_energy = float(np.mean(delta * delta)) / (previous_energy + EPS)
    alignment = 2.0 * float(np.mean(left * delta)) / (previous_energy + EPS)
    gain = (current_energy - previous_energy) / (previous_energy + EPS)
    rho = math.sqrt(max(current_energy, 0.0) / (previous_energy + EPS))
    return {
        "temporal_delta_rms": _rms(delta),
        "temporal_delta_relative_energy": delta_relative_energy,
        "temporal_alignment_contribution": alignment,
        "temporal_energy_gain": gain,
        "temporal_decomposition_error": gain - delta_relative_energy - alignment,
        "temporal_rho": rho,
        "temporal_rho_minus_one": rho - 1.0,
        "temporal_log_rho": math.log(max(rho, EPS)),
        "temporal_positive_energy_gain": max(gain, 0.0),
        "temporal_positive_alignment": max(alignment, 0.0),
    }


def _normal_ci(values: Iterable[float]) -> Dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "se": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    se = std / math.sqrt(array.size)
    return {
        "n": int(array.size),
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_low": mean - 1.959963984540054 * se,
        "ci95_high": mean + 1.959963984540054 * se,
    }


class FixedProbeTracker:
    """Stateful per-edit fixed-panel observer."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        probe_path: Path,
        output_root: Path,
        count: int,
        max_length: int,
        batch_size: int,
        contexts: Sequence[str] = DEFAULT_CONTEXTS,
        vector_interval: int = 10,
    ) -> None:
        if count <= 1 or batch_size <= 0 or vector_interval < 0:
            raise ValueError("invalid fixed-probe count, batch size, or vector interval")
        self.model = model
        self.tokenizer = tokenizer
        self.probe_path = Path(probe_path).expanduser().resolve()
        self.output_root = Path(output_root)
        self.count = int(count)
        self.batch_size = int(batch_size)
        self.contexts = parse_contexts(contexts)
        self.vector_interval = int(vector_interval)
        self.panel = load_fixed_probe_panel(
            self.probe_path,
            tokenizer,
            count=self.count,
            max_length=max_length,
            contexts=self.contexts,
        )
        self.previous_h9: Dict[Tuple[str, str], np.ndarray] = {}
        self.previous_edit_count: int | None = None
        self.base_lookup: Dict[Tuple[str, str, str, int], Mapping[str, str]] = {}
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _step_dir(self, edit_count: int) -> Path:
        return self.output_root / f"step_{int(edit_count):06d}"

    def _load_base_lookup(self) -> None:
        if self.base_lookup:
            return
        path = self._step_dir(0) / "per_probe_metrics.csv"
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (
                    str(row["context"]),
                    str(row["position"]),
                    str(row["case_id"]),
                    int(row["layer"]),
                )
                self.base_lookup[key] = row

    def observe(self, edit_count: int, *, force_vector_save: bool = False) -> Path:
        """Measure the fixed panel at the current in-memory model state."""

        edit_count = int(edit_count)
        current_dir = self._step_dir(edit_count)
        complete_path = current_dir / "complete.json"
        state_path = current_dir / "temporal_h9_state.npz"
        if complete_path.is_file() and state_path.is_file():
            # A normal trajectory resume replays completed edits.  Reuse the
            # exact saved fixed-panel state so the next missing edit still has
            # a valid same-probe predecessor without repeating 400 forwards.
            restored: Dict[Tuple[str, str], np.ndarray] = {}
            with np.load(state_path) as loaded:
                for key in loaded.files:
                    if not key.endswith("__H9"):
                        continue
                    context, position, _ = key.split("__", 2)
                    restored[(context, position)] = np.asarray(
                        loaded[key], dtype=np.float32
                    )
            if restored:
                self.previous_h9 = restored
                self.previous_edit_count = edit_count
                return current_dir
        captured_by_context: Dict[str, Dict[Tuple[str, str], np.ndarray]] = {}
        probes_by_context: Dict[str, Sequence[Probe]] = {}
        for context, (probes, positions) in self.panel.items():
            captured_by_context[context] = _capture_family(
                self.model,
                self.tokenizer,
                probes,
                positions,
                batch_size=self.batch_size,
            )
            probes_by_context[context] = probes

        self._load_base_lookup()
        rows: List[Dict[str, Any]] = []
        h9_state: Dict[str, np.ndarray] = {}
        vector_arrays: Dict[str, np.ndarray] = {}
        for context, captured in captured_by_context.items():
            probes = probes_by_context[context]
            positions = self.panel[context][1]
            for position in positions:
                h8 = captured[(position, "H8")]
                h9 = captured[(position, "H9")]
                h10 = captured[(position, "H10")]
                state_key = (context, position)
                safe_prefix = f"{context}__{position}"
                h9_state[f"{safe_prefix}__H9"] = h9.astype(np.float16)
                save_vectors = force_vector_save or edit_count == 0 or (
                    self.vector_interval > 0 and edit_count % self.vector_interval == 0
                )
                if save_vectors:
                    vector_arrays[f"{safe_prefix}__H8"] = h8.astype(np.float16)
                    vector_arrays[f"{safe_prefix}__F8"] = (h9 - h8).astype(np.float16)
                    vector_arrays[f"{safe_prefix}__H9"] = h9.astype(np.float16)
                    vector_arrays[f"{safe_prefix}__F9"] = (h10 - h9).astype(np.float16)
                    vector_arrays[f"{safe_prefix}__H10"] = h10.astype(np.float16)

                for index, probe in enumerate(probes):
                    for layer, hidden_in, hidden_out in (
                        (8, h8[index], h9[index]),
                        (9, h9[index], h10[index]),
                    ):
                        row: Dict[str, Any] = {
                            "edit_count": edit_count,
                            "context": context,
                            "position": position,
                            "probe_index": index,
                            "case_id": probe.case_id,
                            "layer": layer,
                            **_finite_gain_row(
                                hidden_in,
                                hidden_out,
                                captured[(position, f"A{layer}")][index],
                                captured[(position, f"M{layer}")][index],
                            ),
                        }
                        base_key = (context, position, str(probe.case_id), layer)
                        base = self.base_lookup.get(base_key)
                        if base is not None:
                            rho_excess = float(row["rho"]) - float(base["rho"])
                            log_excess = float(row["log_rho"]) - float(base["log_rho"])
                            row.update(
                                {
                                    "rho_excess_vs_base": rho_excess,
                                    "log_rho_excess_vs_base": log_excess,
                                    "rho_positive_loss_vs_base": max(rho_excess, 0.0),
                                    "log_rho_positive_loss_vs_base": max(log_excess, 0.0),
                                }
                            )
                        if layer == 8 and state_key in self.previous_h9:
                            row.update(
                                _temporal_row(
                                    self.previous_h9[state_key][index],
                                    h9[index],
                                )
                            )
                            row["previous_observed_edit_count"] = (
                                self.previous_edit_count
                            )
                            row["temporal_edit_gap"] = (
                                edit_count - int(self.previous_edit_count)
                                if self.previous_edit_count is not None
                                else None
                            )
                        rows.append(row)
                self.previous_h9[state_key] = h9.copy()
        self.previous_edit_count = edit_count

        summary_rows: List[Dict[str, Any]] = []
        metrics = (
            "finite_gain",
            "write_relative_energy",
            "alignment_contribution",
            "rho_minus_one",
            "log_rho",
            "temporal_energy_gain",
            "temporal_alignment_contribution",
            "temporal_positive_energy_gain",
        )
        groups: MutableMapping[Tuple[str, str, int], List[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[(str(row["context"]), str(row["position"]), int(row["layer"]))].append(row)
        for (context, position, layer), selected in sorted(groups.items()):
            for metric in metrics:
                values = [float(row[metric]) for row in selected if metric in row]
                if not values:
                    continue
                summary_rows.append(
                    {
                        "edit_count": edit_count,
                        "context": context,
                        "position": position,
                        "layer": layer,
                        "metric": metric,
                        **_normal_ci(values),
                    }
                )

        _atomic_csv(current_dir / "per_probe_metrics.csv", rows)
        _atomic_csv(current_dir / "summary_metrics.csv", summary_rows)
        # H9 is retained every edit for exact resume and later direction audits.
        _atomic_npz(current_dir / "temporal_h9_state.npz", h9_state, compressed=False)
        if vector_arrays:
            _atomic_npz(current_dir / "boundary_vectors.npz", vector_arrays, compressed=False)
        _atomic_json(
            complete_path,
            {
                "schema_version": SCHEMA_VERSION,
                "measurement": "fixed_disjoint_panel_observer_forward",
                "edit_count": edit_count,
                "probe_path": str(self.probe_path),
                "probe_sha256": file_sha256(self.probe_path),
                "probe_count_per_prompt_family": self.count,
                "contexts": list(self.contexts),
                "batch_size": self.batch_size,
                "vector_interval": self.vector_interval,
                "required_files": [
                    "per_probe_metrics.csv",
                    "summary_metrics.csv",
                    "temporal_h9_state.npz",
                    *( ["boundary_vectors.npz"] if vector_arrays else [] ),
                ],
            },
        )
        return current_dir
