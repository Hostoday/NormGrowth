#!/usr/bin/env python3
"""Measure H-output- and MLP-anchored cross-layer cosine similarities.

For decoder block ``l`` this script uses the unambiguous notation

``I_l``
    residual stream entering block ``l``;
``A_l``
    self-attention residual write after the output projection;
``M_l``
    MLP residual write after the down projection; and
``O_l``
    block output, so ``O_l = H_(l+1) = I_l + A_l + M_l``.

The H-anchored analysis treats ``O_l`` as the current hidden state and stores
full layer-by-layer cosine matrices against every ``O_j``, ``M_j``, and
``A_j``.  The MLP-anchored analysis stores the full ``M_l`` versus ``A_j``
matrix and the same-block comparisons against ``I_l`` and ``O_l``.

Matrix arrays have layout ``[prompt, position, anchor_layer, reference_layer]``.
Only the requested token vectors are captured; full-sequence activations are
never materialized on CPU or written to disk.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.analyze_base_component_cosines import (  # noqa: E402
    EPS,
    LayerComponentCapture,
    _atomic_csv,
    _atomic_json,
    _atomic_npz,
    _model_backbone,
    _padded_batch,
    finite_summary,
    fingerprint,
    parse_layers,
    safe_cosine,
    sha256_file,
)
from diagnostics.analyze_residual_spectrum import (  # noqa: E402
    Probe,
    load_model,
    load_probes,
    model_input_device,
    model_layers,
    parse_str_list,
)


MATRIX_METRICS = (
    "cos_h_output__h_output",
    "cos_h_output__mlp",
    "cos_h_output__attention",
    "cos_mlp__attention",
)
LAYER_METRICS = (
    "cos_mlp__layer_input",
    "cos_mlp__layer_output",
)
NORM_METRICS = (
    "l2_layer_input",
    "l2_attention",
    "l2_mlp",
    "l2_h_output",
)
DIAGNOSTIC_METRICS = ("closure_relative_l2",)

MATRIX_DEFINITIONS = {
    "cos_h_output__h_output": "cos(O_l, O_j)",
    "cos_h_output__mlp": "cos(O_l, M_j)",
    "cos_h_output__attention": "cos(O_l, A_j)",
    "cos_mlp__attention": "cos(M_l, A_j)",
}
LAYER_DEFINITIONS = {
    "cos_mlp__layer_input": "cos(M_l, I_l)",
    "cos_mlp__layer_output": "cos(M_l, O_l)",
}


def pairwise_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Pairwise FP32 cosine for B x P x L x D and B x P x R x D tensors."""

    if left.ndim != 4 or right.ndim != 4:
        raise ValueError(
            "pairwise cosine expects B x P x layer x D tensors; "
            f"got {tuple(left.shape)} and {tuple(right.shape)}"
        )
    if left.shape[:2] != right.shape[:2] or left.size(-1) != right.size(-1):
        raise ValueError(
            "pairwise cosine batch, position, and hidden dimensions must match; "
            f"got {tuple(left.shape)} and {tuple(right.shape)}"
        )
    left32 = left.float()
    right32 = right.float().to(left32.device)
    numerator = torch.einsum("bpld,bprd->bplr", left32, right32)
    left_norm = torch.linalg.vector_norm(left32, dim=-1)
    right_norm = torch.linalg.vector_norm(right32, dim=-1)
    denominator = left_norm[..., :, None] * right_norm[..., None, :]
    cosine = numerator / torch.clamp(denominator, min=EPS)
    return torch.where(
        denominator > EPS,
        cosine,
        torch.full_like(cosine, torch.nan),
    )


def reference_cosines_from_nodes(
    nodes: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute all requested metrics from stacked B x P x L x D nodes."""

    required = {"input", "attention", "mlp", "output"}
    missing = sorted(required - set(nodes))
    if missing:
        raise ValueError(f"missing stacked component nodes: {missing}")
    values = {name: nodes[name].float() for name in required}
    result = {
        "cos_h_output__h_output": pairwise_cosine(
            values["output"], values["output"]
        ),
        "cos_h_output__mlp": pairwise_cosine(
            values["output"], values["mlp"]
        ),
        "cos_h_output__attention": pairwise_cosine(
            values["output"], values["attention"]
        ),
        "cos_mlp__attention": pairwise_cosine(
            values["mlp"], values["attention"]
        ),
        "cos_mlp__layer_input": safe_cosine(values["mlp"], values["input"]),
        "cos_mlp__layer_output": safe_cosine(values["mlp"], values["output"]),
    }
    for node, metric in (
        ("input", "l2_layer_input"),
        ("attention", "l2_attention"),
        ("mlp", "l2_mlp"),
        ("output", "l2_h_output"),
    ):
        result[metric] = torch.linalg.vector_norm(values[node], dim=-1)
    total = values["output"] - values["input"]
    closure = total - values["attention"] - values["mlp"]
    result["closure_relative_l2"] = torch.linalg.vector_norm(
        closure, dim=-1
    ) / torch.clamp(torch.linalg.vector_norm(total, dim=-1), min=EPS)
    return result


def _stack_capture(
    capture: LayerComponentCapture,
    layers: Sequence[int],
    node: str,
    device: torch.device,
) -> torch.Tensor:
    return torch.stack(
        [capture.values[(int(layer), node)].to(device) for layer in layers], dim=2
    )


@torch.inference_mode()
def collect_h_m_reference_cosines(
    model: Any,
    tokenizer: Any,
    probes: Sequence[Probe],
    layers: Sequence[int],
    positions: Sequence[str],
    *,
    batch_size: int,
    progress_every: int = 100,
) -> Dict[str, np.ndarray]:
    """Run batched forwards and return cross-layer and same-layer metrics."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not probes or not layers or not positions:
        raise ValueError("probes, layers, and positions must be non-empty")
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")

    matrix_shape = (len(probes), len(positions), len(layers), len(layers))
    layer_shape = (len(probes), len(positions), len(layers))
    arrays: Dict[str, np.ndarray] = {
        metric: np.full(matrix_shape, np.nan, dtype=np.float32)
        for metric in MATRIX_METRICS
    }
    arrays.update(
        {
            metric: np.full(layer_shape, np.nan, dtype=np.float32)
            for metric in LAYER_METRICS + NORM_METRICS + DIAGNOSTIC_METRICS
        }
    )
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

            target_device = capture.values[(int(layers[0]), "output")].device
            stacked = {
                node: _stack_capture(capture, layers, node, target_device)
                for node in ("input", "attention", "mlp", "output")
            }
            batch_metrics = reference_cosines_from_nodes(stacked)
            for metric, value in batch_metrics.items():
                arrays[metric][start:end] = (
                    value.detach().cpu().numpy().astype(np.float32, copy=False)
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
                    f"[h-m cosine] {completed}/{len(probes)} prompts "
                    f"({rate:.2f} prompts/s)",
                    flush=True,
                )
                last_reported = completed
    finally:
        capture.close()

    arrays["token_positions"] = token_positions
    return arrays


def build_matrix_summary_rows(
    arrays: Mapping[str, np.ndarray],
    layers: Sequence[int],
    positions: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for position_index, position in enumerate(positions):
        for metric in MATRIX_METRICS:
            for anchor_index, anchor_layer in enumerate(layers):
                for reference_index, reference_layer in enumerate(layers):
                    delta = int(reference_layer) - int(anchor_layer)
                    relation = "same" if delta == 0 else ("previous" if delta < 0 else "future")
                    rows.append(
                        {
                            "position": position,
                            "metric": metric,
                            "anchor_layer": int(anchor_layer),
                            "reference_layer": int(reference_layer),
                            "layer_delta": delta,
                            "relation": relation,
                            **finite_summary(
                                arrays[metric][
                                    :, position_index, anchor_index, reference_index
                                ]
                            ),
                        }
                    )
    return rows


def build_layer_summary_rows(
    arrays: Mapping[str, np.ndarray],
    layers: Sequence[int],
    positions: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for position_index, position in enumerate(positions):
        for layer_index, layer in enumerate(layers):
            for metric in LAYER_METRICS + NORM_METRICS + DIAGNOSTIC_METRICS:
                rows.append(
                    {
                        "position": position,
                        "layer": int(layer),
                        "metric": metric,
                        **finite_summary(arrays[metric][:, position_index, layer_index]),
                    }
                )
    return rows


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
    matrix_rows = build_matrix_summary_rows(arrays, layers, positions)
    layer_rows = build_layer_summary_rows(arrays, layers, positions)

    per_sample_rows: List[Dict[str, Any]] = []
    token_positions = np.asarray(arrays["token_positions"])
    sample_metrics = LAYER_METRICS + NORM_METRICS + DIAGNOSTIC_METRICS
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
                        metric: float(
                            arrays[metric][prompt_index, position_index, layer_index]
                        )
                        for metric in sample_metrics
                    }
                )
                per_sample_rows.append(row)

    npz_arrays: Dict[str, np.ndarray] = {
        metric: np.asarray(arrays[metric], dtype=np.float32)
        for metric in MATRIX_METRICS
        + LAYER_METRICS
        + NORM_METRICS
        + DIAGNOSTIC_METRICS
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
        "arrays": output_dir / "h_m_reference_cosines.npz",
        "matrix_summary": output_dir / "matrix_summary.csv",
        "layer_summary": output_dir / "layer_summary.csv",
        "per_sample_layer": output_dir / "per_sample_layer_metrics.csv",
        "config": output_dir / "run_config.json",
    }
    _atomic_npz(paths["arrays"], npz_arrays)
    _atomic_csv(paths["matrix_summary"], matrix_rows)
    _atomic_csv(paths["layer_summary"], layer_rows)
    _atomic_csv(paths["per_sample_layer"], per_sample_rows)
    _atomic_json(paths["config"], config)

    closure = np.asarray(arrays["closure_relative_l2"], dtype=np.float64)
    finite_closure = closure[np.isfinite(closure)]
    manifest = {
        "schema_version": 1,
        "measurement": "base_model_h_m_reference_cosines",
        "complete": True,
        "n_prompts": len(probes),
        "positions": list(positions),
        "layers": [int(layer) for layer in layers],
        "matrix_metrics": list(MATRIX_METRICS),
        "layer_metrics": list(LAYER_METRICS),
        "matrix_layout": "[prompt, position, anchor_layer, reference_layer]",
        "h_anchor": "O_l = H_(l+1), decoder block-l output",
        "mean_closure_relative_l2": (
            float(np.mean(finite_closure)) if finite_closure.size else float("nan")
        ),
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
    parser.add_argument("--expected-requests-sha256", default=None)
    parser.add_argument("--dataset-label", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--positions", type=parse_str_list, default=["subject_last", "prompt_last"]
    )
    parser.add_argument("--layers", default="all")
    parser.add_argument("--max-prompts", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--probe-selection", choices=["prefix", "random"], default="prefix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="eager")
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
            "layer_input": "I_l = H_l, decoder block-l residual input",
            "attention": "A_l, block-l self-attention output after output projection",
            "mlp": "M_l, block-l MLP output after down projection",
            "h_output": "O_l = H_(l+1), complete decoder block-l output",
        },
        "matrix_metrics": MATRIX_DEFINITIONS,
        "layer_metrics": LAYER_DEFINITIONS,
        "matrix_layout": "[prompt, position, anchor_layer, reference_layer]",
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

    arrays = collect_h_m_reference_cosines(
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
        f"{len(layers)}x{len(layers)} layer matrices to {output_dir}; "
        f"mean closure={manifest['mean_closure_relative_l2']:.6g}, "
        f"max closure={manifest['max_closure_relative_l2']:.6g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
