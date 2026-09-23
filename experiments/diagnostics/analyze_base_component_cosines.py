#!/usr/bin/env python3
"""Measure layerwise residual-component cosine similarities on a Base model.

For decoder block ``l`` the captured vectors are

``H_l``
    residual stream entering the block (``input``);
``A_l``
    self-attention branch output after the attention output projection;
``M_l``
    MLP branch output after the MLP down projection;
``H_(l+1)``
    complete decoder-block output (``output``); and
``M_(l-1)``
    MLP branch output from the preceding block (``previous_mlp``).

The primary requested comparisons use attention as the anchor:

    cos(A_l, H_l), cos(A_l, M_l), cos(A_l, H_(l+1)), cos(A_l, M_(l-1)).

All ten pairwise cosines among the five available nodes are retained as well.
Only the selected token vectors are captured; full-sequence activations are
never materialized on CPU or written to disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.analyze_residual_spectrum import (  # noqa: E402
    Probe,
    _layer_hidden_output,
    load_model,
    load_probes,
    layer_attention,
    model_input_device,
    model_layers,
    parse_str_list,
)


NODE_NAMES = ("input", "attention", "mlp", "output", "previous_mlp")
PAIR_SPECS = tuple(itertools.combinations(NODE_NAMES, 2))
PAIR_NAMES = tuple(f"cos_{left}__{right}" for left, right in PAIR_SPECS)
ATTENTION_REFERENCE_METRICS = (
    "cos_input__attention",
    "cos_attention__mlp",
    "cos_attention__output",
    "cos_attention__previous_mlp",
)
NORM_NAMES = tuple(f"l2_{node}" for node in NODE_NAMES)
DIAGNOSTIC_NAMES = ("closure_relative_l2",)
ALL_ARRAY_METRICS = PAIR_NAMES + NORM_NAMES + DIAGNOSTIC_NAMES
EPS = 1.0e-12


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def parse_layers(text: str, layer_count: int) -> Tuple[int, ...]:
    normalized = str(text).strip().lower()
    if normalized == "all":
        return tuple(range(layer_count))
    values = tuple(int(piece.strip()) for piece in str(text).split(",") if piece.strip())
    if not values:
        raise ValueError("--layers must be 'all' or a comma-separated integer list")
    if len(values) != len(set(values)):
        raise ValueError("--layers contains duplicates")
    invalid = [layer for layer in values if layer < 0 or layer >= layer_count]
    if invalid:
        raise ValueError(f"layers out of range for {layer_count} decoder blocks: {invalid}")
    return tuple(sorted(values))


def safe_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Row-wise float32 cosine, returning NaN when either vector has zero norm."""

    left32 = left.float()
    right32 = right.float().to(left32.device)
    numerator = torch.sum(left32 * right32, dim=-1)
    left_norm = torch.linalg.vector_norm(left32, dim=-1)
    right_norm = torch.linalg.vector_norm(right32, dim=-1)
    denominator = left_norm * right_norm
    cosine = numerator / torch.clamp(denominator, min=EPS)
    return torch.where(denominator > EPS, cosine, torch.full_like(cosine, torch.nan))


def _select_token_vectors(
    value: torch.Tensor,
    token_positions: torch.Tensor,
) -> torch.Tensor:
    """Select B x P token positions from a B x T x D activation."""

    if value.ndim != 3:
        raise ValueError(f"expected B x T x D activation, got {tuple(value.shape)}")
    positions = token_positions.to(device=value.device, dtype=torch.long)
    if positions.ndim != 2 or positions.size(0) != value.size(0):
        raise ValueError(
            "token position shape must be B x P and match activation batch; "
            f"got {tuple(positions.shape)} vs {tuple(value.shape)}"
        )
    rows = torch.arange(value.size(0), device=value.device)[:, None]
    return value[rows, positions].detach()


class LayerComponentCapture:
    """Forward-hook collector for selected token vectors at decoder blocks."""

    def __init__(self, model: Any, layers: Sequence[int]) -> None:
        self.decoder_layers = model_layers(model)
        self.layers = tuple(int(layer) for layer in layers)
        self.handles: List[Any] = []
        self.token_positions: torch.Tensor | None = None
        self.values: Dict[Tuple[int, str], torch.Tensor] = {}

    def _capture(self, layer: int, node: str, value: torch.Tensor) -> None:
        if self.token_positions is None:
            raise RuntimeError("capture token positions were not initialized")
        self.values[(int(layer), str(node))] = _select_token_vectors(
            value, self.token_positions
        )

    def install(self) -> None:
        if self.handles:
            raise RuntimeError("component hooks are already installed")

        requested = set(self.layers)
        mlp_layers = requested | {layer - 1 for layer in requested if layer > 0}
        for layer_id in sorted(requested):
            layer = self.decoder_layers[layer_id]

            def capture_input(
                _module: Any,
                args: Tuple[Any, ...],
                kwargs: Mapping[str, Any],
                *,
                key: int = layer_id,
            ) -> None:
                hidden = args[0] if args else kwargs.get("hidden_states")
                if not torch.is_tensor(hidden):
                    raise TypeError(f"could not capture input H{key}")
                self._capture(key, "input", hidden)

            def capture_attention(
                _module: Any,
                _args: Tuple[Any, ...],
                output: Any,
                *,
                key: int = layer_id,
            ) -> None:
                self._capture(key, "attention", _layer_hidden_output(output))

            def capture_output(
                _module: Any,
                _args: Tuple[Any, ...],
                output: Any,
                *,
                key: int = layer_id,
            ) -> None:
                self._capture(key, "output", _layer_hidden_output(output))

            self.handles.append(
                layer.register_forward_pre_hook(capture_input, with_kwargs=True)
            )
            self.handles.append(
                layer_attention(layer).register_forward_hook(capture_attention)
            )
            self.handles.append(layer.register_forward_hook(capture_output))

        for layer_id in sorted(mlp_layers):
            layer = self.decoder_layers[layer_id]

            def capture_mlp(
                _module: Any,
                _args: Tuple[Any, ...],
                output: Any,
                *,
                key: int = layer_id,
            ) -> None:
                self._capture(key, "mlp", _layer_hidden_output(output))

            self.handles.append(layer.mlp.register_forward_hook(capture_mlp))

    def begin_batch(self, token_positions: torch.Tensor) -> None:
        self.values.clear()
        self.token_positions = token_positions.detach().to(device="cpu", dtype=torch.long)

    def validate(self) -> None:
        expected = {
            (layer, node)
            for layer in self.layers
            for node in ("input", "attention", "mlp", "output")
        }
        expected.update((layer - 1, "mlp") for layer in self.layers if layer > 0)
        missing = sorted(expected - set(self.values))
        if missing:
            raise RuntimeError(f"component hooks did not capture: {missing}")

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.values.clear()
        self.token_positions = None


def _padded_batch(
    probes: Sequence[Probe],
    positions: Sequence[str],
    *,
    pad_token_id: int,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    max_tokens = max(len(probe.input_ids) for probe in probes)
    input_ids = torch.full(
        (len(probes), max_tokens),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    selected_positions = torch.empty(
        (len(probes), len(positions)), dtype=torch.long
    )
    for row, probe in enumerate(probes):
        length = len(probe.input_ids)
        input_ids[row, :length] = torch.as_tensor(
            probe.input_ids, dtype=torch.long, device=device
        )
        attention_mask[row, :length] = 1
        for position_index, position in enumerate(positions):
            selected_positions[row, position_index] = int(probe.positions[position])
    return {"input_ids": input_ids, "attention_mask": attention_mask}, selected_positions


def _model_backbone(model: Any) -> Any:
    # Avoid constructing vocabulary-sized logits from a CausalLM head.
    candidate = getattr(model, "model", None)
    return candidate if candidate is not None else model


@torch.inference_mode()
def collect_component_cosines(
    model: Any,
    tokenizer: Any,
    probes: Sequence[Probe],
    layers: Sequence[int],
    positions: Sequence[str],
    *,
    batch_size: int,
    progress_every: int = 100,
) -> Dict[str, np.ndarray]:
    """Run ordinary batched forwards and return N x P x L metric arrays."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not probes or not layers or not positions:
        raise ValueError("probes, layers, and positions must be non-empty")
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")

    shape = (len(probes), len(positions), len(layers))
    arrays = {
        name: np.full(shape, np.nan, dtype=np.float32)
        for name in ALL_ARRAY_METRICS
    }
    token_positions = np.empty((len(probes), len(positions)), dtype=np.int32)
    input_device = model_input_device(model)
    backbone = _model_backbone(model)
    capture = LayerComponentCapture(model, layers)
    capture.install()
    started = time.time()
    last_reported = 0
    try:
        for start in range(0, len(probes), batch_size):
            end = min(start + batch_size, len(probes))
            batch_probes = probes[start:end]
            inputs, batch_positions = _padded_batch(
                batch_probes,
                positions,
                pad_token_id=int(pad_token_id),
                device=input_device,
            )
            token_positions[start:end] = batch_positions.numpy().astype(
                np.int32, copy=False
            )
            capture.begin_batch(batch_positions)
            backbone(
                **inputs,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            capture.validate()

            for layer_index, layer_id in enumerate(layers):
                nodes: Dict[str, torch.Tensor] = {
                    node: capture.values[(layer_id, node)]
                    for node in ("input", "attention", "mlp", "output")
                }
                if layer_id > 0:
                    nodes["previous_mlp"] = capture.values[(layer_id - 1, "mlp")]

                for (left_name, right_name), metric_name in zip(
                    PAIR_SPECS, PAIR_NAMES
                ):
                    if left_name not in nodes or right_name not in nodes:
                        continue
                    values = safe_cosine(nodes[left_name], nodes[right_name])
                    arrays[metric_name][start:end, :, layer_index] = (
                        values.detach().cpu().numpy().astype(np.float32, copy=False)
                    )

                target_device = nodes["input"].device
                same_device = {
                    name: value.float().to(target_device)
                    for name, value in nodes.items()
                }
                for node in NODE_NAMES:
                    if node not in same_device:
                        continue
                    norms = torch.linalg.vector_norm(same_device[node], dim=-1)
                    arrays[f"l2_{node}"][start:end, :, layer_index] = (
                        norms.detach().cpu().numpy().astype(np.float32, copy=False)
                    )
                total = same_device["output"] - same_device["input"]
                closure = (
                    same_device["output"]
                    - same_device["input"]
                    - same_device["attention"]
                    - same_device["mlp"]
                )
                closure_relative = torch.linalg.vector_norm(closure, dim=-1) / torch.clamp(
                    torch.linalg.vector_norm(total, dim=-1), min=EPS
                )
                arrays["closure_relative_l2"][start:end, :, layer_index] = (
                    closure_relative.detach().cpu().numpy().astype(np.float32, copy=False)
                )

            completed = end
            if (
                completed == len(probes)
                or completed - last_reported >= max(progress_every, 1)
                or start == 0
            ):
                elapsed = time.time() - started
                rate = completed / max(elapsed, EPS)
                print(
                    f"[cosine] {completed}/{len(probes)} prompts "
                    f"({rate:.2f} prompts/s)",
                    flush=True,
                )
                last_reported = completed
    finally:
        capture.close()

    arrays["token_positions"] = token_positions
    return arrays


def finite_summary(values: np.ndarray) -> Dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "se": float("nan"),
            "median": float("nan"),
            "q05": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "q95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": std,
        "se": std / math.sqrt(finite.size),
        "median": float(np.median(finite)),
        "q05": float(np.quantile(finite, 0.05)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
        "q95": float(np.quantile(finite, 0.95)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def build_layer_summary_rows(
    arrays: Mapping[str, np.ndarray],
    layers: Sequence[int],
    positions: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for position_index, position in enumerate(positions):
        for layer_index, layer in enumerate(layers):
            for metric in PAIR_NAMES:
                rows.append(
                    {
                        "position": position,
                        "layer": int(layer),
                        "metric": metric,
                        **finite_summary(arrays[metric][:, position_index, layer_index]),
                    }
                )
    return rows


def build_global_summary_rows(
    arrays: Mapping[str, np.ndarray],
    layers: Sequence[int],
    positions: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    edit_layer_indices = [index for index, layer in enumerate(layers) if 4 <= layer <= 8]
    for position_index, position in enumerate(positions):
        for scope, indices in (
            ("all_layers", list(range(len(layers)))),
            ("edit_layers_L4_L8", edit_layer_indices),
        ):
            if not indices:
                continue
            for metric in PAIR_NAMES:
                rows.append(
                    {
                        "scope": scope,
                        "position": position,
                        "metric": metric,
                        **finite_summary(arrays[metric][:, position_index, indices]),
                    }
                )
    return rows


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **dict(arrays))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_outputs(
    output_dir: Path,
    *,
    probes: Sequence[Probe],
    layers: Sequence[int],
    positions: Sequence[str],
    arrays: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_rows = build_layer_summary_rows(arrays, layers, positions)
    global_rows = build_global_summary_rows(arrays, layers, positions)

    per_sample_rows: List[Dict[str, Any]] = []
    token_positions = np.asarray(arrays["token_positions"])
    for prompt_index, probe in enumerate(probes):
        for position_index, position in enumerate(positions):
            for layer_index, layer in enumerate(layers):
                row: Dict[str, Any] = {
                    "prompt_index": prompt_index,
                    "case_id": probe.case_id,
                    "position": position,
                    "token_position": int(token_positions[prompt_index, position_index]),
                    "layer": int(layer),
                }
                row.update(
                    {
                        metric: float(arrays[metric][prompt_index, position_index, layer_index])
                        for metric in ALL_ARRAY_METRICS
                    }
                )
                per_sample_rows.append(row)

    npz_arrays: Dict[str, np.ndarray] = {
        metric: np.asarray(arrays[metric], dtype=np.float32)
        for metric in ALL_ARRAY_METRICS
    }
    npz_arrays.update(
        {
            "token_positions": token_positions.astype(np.int32, copy=False),
            "layers": np.asarray(layers, dtype=np.int32),
            "positions": np.asarray(positions, dtype=np.str_),
            "prompt_indices": np.arange(len(probes), dtype=np.int32),
            "case_ids": np.asarray([str(probe.case_id) for probe in probes], dtype=np.str_),
        }
    )

    paths = {
        "arrays": output_dir / "component_cosines.npz",
        "per_sample": output_dir / "per_sample_layer_cosines.csv",
        "layer_summary": output_dir / "layer_summary.csv",
        "global_summary": output_dir / "global_summary.csv",
        "config": output_dir / "run_config.json",
    }
    _atomic_npz(paths["arrays"], npz_arrays)
    _atomic_csv(paths["per_sample"], per_sample_rows)
    _atomic_csv(paths["layer_summary"], layer_rows)
    _atomic_csv(paths["global_summary"], global_rows)
    _atomic_json(paths["config"], config)

    closure = np.asarray(arrays["closure_relative_l2"], dtype=np.float64)
    finite_closure = closure[np.isfinite(closure)]
    manifest = {
        "schema_version": 1,
        "measurement": "base_model_layer_component_cosines",
        "complete": True,
        "n_prompts": len(probes),
        "positions": list(positions),
        "layers": [int(layer) for layer in layers],
        "pair_metrics": list(PAIR_NAMES),
        "attention_reference_metrics": list(ATTENTION_REFERENCE_METRICS),
        "array_layout": "[prompt, position, layer]",
        "previous_mlp_at_layer_0": "undefined; stored as NaN",
        "max_closure_relative_l2": (
            float(np.max(finite_closure)) if finite_closure.size else float("nan")
        ),
        "files": {
            key: {
                "path": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for key, path in paths.items()
        },
    }
    _atomic_json(output_dir / "complete.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument("--requests-path", required=True)
    parser.add_argument(
        "--expected-requests-sha256",
        default=None,
        help="Optional fail-closed SHA256 check for the exact request cohort.",
    )
    parser.add_argument("--dataset-label", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positions", type=parse_str_list, default=["subject_last", "prompt_last"])
    parser.add_argument("--layers", default="all")
    parser.add_argument("--max-prompts", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--probe-selection", choices=["prefix", "random"], default="prefix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    requests_path = Path(args.requests_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.max_prompts <= 0:
        raise ValueError("--max-prompts must be positive")
    if not requests_path.is_file():
        raise FileNotFoundError(requests_path)
    requests_sha256 = sha256_file(requests_path)
    if (
        args.expected_requests_sha256
        and requests_sha256.lower() != args.expected_requests_sha256.strip().lower()
    ):
        raise ValueError(
            "request-file SHA256 mismatch: "
            f"observed={requests_sha256}, expected={args.expected_requests_sha256}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    positions = tuple(args.positions)
    unknown_positions = sorted(set(positions) - {"subject_last", "prompt_last"})
    if unknown_positions:
        raise ValueError(f"unsupported positions: {unknown_positions}")
    probes = load_probes(
        str(requests_path),
        tokenizer,
        positions,
        args.max_prompts,
        args.max_length,
        probe_source="rewrite",
        selection=args.probe_selection,
        seed=args.seed,
    )
    if len(probes) != args.max_prompts:
        raise RuntimeError(
            f"loaded {len(probes)} probes, expected exactly {args.max_prompts}"
        )

    model = load_model(
        args.model_name,
        args.torch_dtype,
        args.device_map,
        args.trust_remote_code,
        args.attn_implementation,
    )
    layers = parse_layers(args.layers, len(model_layers(model)))
    config = {
        "schema_version": 1,
        "model_name": args.model_name,
        "model_config_name_or_path": str(
            getattr(model.config, "_name_or_path", args.model_name)
        ),
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "model_type": str(getattr(model.config, "model_type", "")),
        "hidden_size": int(getattr(model.config, "hidden_size")),
        "decoder_layer_count": len(model_layers(model)),
        "requests_path": str(requests_path),
        "requests_sha256": requests_sha256,
        "expected_requests_sha256": args.expected_requests_sha256,
        "dataset_label": args.dataset_label,
        "probe_source": "rewrite",
        "probe_selection": args.probe_selection,
        "n_prompts": len(probes),
        "case_ids": [str(probe.case_id) for probe in probes],
        "positions": list(positions),
        "layers": list(layers),
        "max_length": args.max_length,
        "observed_token_length_min": min(len(probe.input_ids) for probe in probes),
        "observed_token_length_max": max(len(probe.input_ids) for probe in probes),
        "batch_size": args.batch_size,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "seed": args.seed,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "node_definitions": {
            "input": "H_l, decoder block-l residual input",
            "attention": "A_l, block-l self-attention branch output after output projection",
            "mlp": "M_l, block-l MLP branch output after down projection",
            "output": "H_(l+1), complete block-l output",
            "previous_mlp": "M_(l-1), preceding block MLP branch output",
        },
        "pair_metrics": list(PAIR_NAMES),
        "attention_reference_metrics": list(ATTENTION_REFERENCE_METRICS),
    }
    config["fingerprint"] = fingerprint(config)
    existing_config = output_dir / "run_config.json"
    if existing_config.exists():
        current = json.loads(existing_config.read_text(encoding="utf-8"))
        if current.get("fingerprint") != config["fingerprint"]:
            raise ValueError(
                f"existing output has a different fingerprint: {existing_config}"
            )
    else:
        _atomic_json(existing_config, config)
    complete_path = output_dir / "complete.json"
    if complete_path.exists():
        print(f"[done] existing complete artifact: {output_dir}")
        return

    arrays = collect_component_cosines(
        model,
        tokenizer,
        probes,
        layers,
        positions,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
    )
    manifest = write_outputs(
        output_dir,
        probes=probes,
        layers=layers,
        positions=positions,
        arrays=arrays,
        config=config,
    )
    print(
        f"[done] wrote {len(probes)} prompts x {len(positions)} positions x "
        f"{len(layers)} layers to {output_dir}; "
        f"max closure={manifest['max_closure_relative_l2']:.6g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
