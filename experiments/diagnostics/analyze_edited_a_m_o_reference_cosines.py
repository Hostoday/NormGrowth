#!/usr/bin/env python3
"""Repeat the Base A/M/O cosine analyses on one fixed edited endpoint.

The input state is a compact cumulative ``edited_parameter_deltas.pt``
checkpoint.  It is applied exactly once to a fresh Base model, after which the
same 1,000 rewrite prompts used by the Base analysis are forwarded through the
fixed endpoint model.  This is deliberately *not* an edit replay.

Two compatible artifacts are written below ``--output-dir``:

``attention_reference``
    The original A-anchored component analysis:
    cos(A_l,I_l), cos(A_l,M_l), cos(A_l,O_l), cos(A_l,M_(l-1)).

``h_m_reference``
    The O-anchored all-layer matrices and the M-anchored comparisons:
    cos(O_l,O_j), cos(O_l,M_j), cos(O_l,A_j), cos(M_l,A_j),
    cos(M_l,I_l), and cos(M_l,O_l).

Here ``I_l`` is the input residual, ``A_l`` the attention residual write,
``M_l`` the MLP residual write, and ``O_l = I_l + A_l + M_l`` the decoder
block output.  Matrix arrays use
``[prompt, position, anchor_layer, reference_layer]`` layout.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics import analyze_base_component_cosines as attention_analysis  # noqa: E402
from diagnostics import analyze_base_h_m_reference_cosines as h_m_analysis  # noqa: E402
from diagnostics.analyze_base_component_cosines import (  # noqa: E402
    _atomic_json,
    fingerprint,
    parse_layers,
    sha256_file,
)
from diagnostics.analyze_residual_spectrum import (  # noqa: E402
    compact_checkpoint_metadata,
    load_model,
    load_probes,
    model_layers,
    parse_str_list,
    resolve_model_load_spec,
)


EXPECTED_FORMAT = "easyedit-edited-parameters"
EXPECTED_STORAGE_MODE = "parameter_deltas"
DEFAULT_REWRITE_LAYERS = (4, 5, 6, 7, 8)


def validate_hiddennorm_condition(
    source_run_config: Mapping[str, Any], *, preserve_early_attention: bool = False
) -> None:
    """Keep HN and the opt-in HN+attn endpoint identities distinct."""

    hiddennorm_signature = (
        source_run_config.get("residual_gain_regularization") is True
        and float(source_run_config.get("residual_gain_lambda", float("nan"))) == 1.0
        and source_run_config.get("residual_gain_objective") == "rho_minus_one"
        and source_run_config.get("residual_gain_loss_type") == "positive_squared"
        and source_run_config.get("residual_gain_subject_layers") == [8]
    )
    if not hiddennorm_signature:
        raise ValueError("source run config is not the intended HN/HiddenNorm condition")
    enabled = source_run_config.get("early_attention_preservation_enabled", False)
    if not preserve_early_attention:
        if enabled is not False:
            raise ValueError("HN with early attention preservation requires hiddennorm_attn")
        return
    early_lambda = float(
        source_run_config.get("early_attention_preservation_lambda", float("nan"))
    )
    if not (
        enabled is True
        and math.isfinite(early_lambda)
        and early_lambda > 0.0
        and source_run_config.get("early_attention_preservation_layers") == [0, 1, 2, 3, 4]
        and source_run_config.get("early_attention_preservation_reference")
        == "current_pre_edit"
    ):
        raise ValueError("source run config is not the intended HN+attn condition")


def validate_source_edit_count(
    source_run_config: Mapping[str, Any], expected_edit_count: int
) -> None:
    """A saved checkpoint may precede the planned end of its source run."""

    sample_size = int(source_run_config.get("sample_size", -1))
    if not 0 < expected_edit_count <= sample_size:
        raise ValueError("checkpoint edit_count must be positive and within source sample_size")


def validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    expected_edit_count: int,
    expected_rewrite_layers: Sequence[int],
) -> None:
    """Fail closed if the compact endpoint is not the intended cumulative state."""

    if metadata.get("format") != EXPECTED_FORMAT:
        raise ValueError(
            f"unexpected checkpoint format for {checkpoint_path}: "
            f"{metadata.get('format')!r}"
        )
    if str(metadata.get("storage_mode")) != EXPECTED_STORAGE_MODE:
        raise ValueError(
            "A/M/O endpoint analysis requires cumulative parameter deltas; "
            f"got storage_mode={metadata.get('storage_mode')!r}"
        )
    if int(metadata.get("edit_count", -1)) != int(expected_edit_count):
        raise ValueError(
            f"checkpoint edit_count={metadata.get('edit_count')!r}, "
            f"expected {expected_edit_count}"
        )
    observed_layers = tuple(int(value) for value in metadata.get("rewrite_layers", ()))
    expected_layers = tuple(int(value) for value in expected_rewrite_layers)
    if observed_layers != expected_layers:
        raise ValueError(
            f"checkpoint rewrite_layers={observed_layers}, expected {expected_layers}"
        )
    module_template = str(metadata.get("rewrite_module_tmp", ""))
    expected_names = [
        f"model.layers.{layer}.mlp.down_proj.weight" for layer in expected_layers
    ]
    observed_names = list(metadata.get("parameter_names", ()))
    if module_template != "model.layers.{}.mlp.down_proj":
        raise ValueError(f"unexpected rewrite_module_tmp={module_template!r}")
    if observed_names != expected_names:
        raise ValueError(
            "checkpoint parameter set does not match the intended L4-L8 MLP "
            f"down projections: {observed_names}"
        )
    semantics = str(metadata.get("checkpoint_semantics", ""))
    if "Add these cumulative deltas" not in semantics:
        raise ValueError(f"unexpected checkpoint semantics: {semantics!r}")


def _max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    if left64.shape != right64.shape:
        return float("inf")
    if not np.array_equal(np.isfinite(left64), np.isfinite(right64)):
        return float("inf")
    valid = np.isfinite(left64) & np.isfinite(right64)
    if not np.any(valid):
        return float("nan")
    return float(np.max(np.abs(left64[valid] - right64[valid])))


def cross_artifact_validation(
    attention_arrays: Mapping[str, np.ndarray],
    h_m_arrays: Mapping[str, np.ndarray],
    *,
    tolerance: float,
) -> Dict[str, Any]:
    """Check identities shared by the independently captured A and M/O runs."""

    token_positions_match = np.array_equal(
        attention_arrays["token_positions"], h_m_arrays["token_positions"]
    )
    checks = {
        "O_vs_A_same_layer": _max_abs_difference(
            np.diagonal(
                h_m_arrays["cos_h_output__attention"], axis1=-2, axis2=-1
            ),
            attention_arrays["cos_attention__output"],
        ),
        "M_vs_A_same_layer": _max_abs_difference(
            np.diagonal(
                h_m_arrays["cos_mlp__attention"], axis1=-2, axis2=-1
            ),
            attention_arrays["cos_attention__mlp"],
        ),
        "M_vs_input_same_layer": _max_abs_difference(
            h_m_arrays["cos_mlp__layer_input"],
            attention_arrays["cos_input__mlp"],
        ),
        "M_vs_O_same_layer": _max_abs_difference(
            h_m_arrays["cos_mlp__layer_output"],
            attention_arrays["cos_mlp__output"],
        ),
        "closure_relative_l2": _max_abs_difference(
            h_m_arrays["closure_relative_l2"],
            attention_arrays["closure_relative_l2"],
        ),
    }
    h_h = np.asarray(h_m_arrays["cos_h_output__h_output"], dtype=np.float64)
    checks["O_vs_O_symmetry"] = _max_abs_difference(
        h_h, np.swapaxes(h_h, -1, -2)
    )
    checks["O_vs_O_diagonal_one"] = _max_abs_difference(
        np.diagonal(h_h, axis1=-2, axis2=-1),
        np.ones(h_h.shape[:-1], dtype=np.float64),
    )
    failures = {
        name: value
        for name, value in checks.items()
        if not np.isfinite(value) or value > tolerance
    }
    if not token_positions_match or failures:
        raise RuntimeError(
            "A and M/O captures failed cross-artifact validation: "
            f"token_positions_match={token_positions_match}, failures={failures}"
        )
    return {
        "passed": True,
        "tolerance": float(tolerance),
        "token_positions_match": token_positions_match,
        "max_absolute_differences": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-label", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--source-run-config", required=True)
    parser.add_argument(
        "--expected-condition",
        choices=["sphere", "hiddennorm", "hiddennorm_attn", "sadr", "encore", "nas"],
        required=True,
    )
    parser.add_argument("--expected-editor", choices=["MEMIT", "AlphaEdit"], required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--requests-path", required=True)
    parser.add_argument("--expected-requests-sha256", required=True)
    parser.add_argument("--dataset-label", default="canonical_zsre_prefix_1000")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--positions", type=parse_str_list, default=["subject_last", "prompt_last"]
    )
    parser.add_argument("--layers", default="all")
    parser.add_argument("--max-prompts", type=int, default=1000)
    parser.add_argument("--expected-edit-count", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--probe-selection", choices=["prefix", "random"], default="prefix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--validation-tolerance", type=float, default=5.0e-5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    base_model_path = Path(args.base_model_path).expanduser().resolve()
    tokenizer_path = Path(args.tokenizer_path or base_model_path).expanduser().resolve()
    requests_path = Path(args.requests_path).expanduser().resolve()
    source_run_config_path = Path(args.source_run_config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    attention_dir = output_dir / "attention_reference"
    h_m_dir = output_dir / "h_m_reference"

    if args.max_prompts <= 0:
        raise ValueError("--max-prompts must be positive")
    for path, description in (
        (checkpoint_path, "checkpoint"),
        (requests_path, "request file"),
        (source_run_config_path, "source run config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {description}: {path}")
    if not base_model_path.is_dir():
        raise FileNotFoundError(f"missing Base model directory: {base_model_path}")
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"missing tokenizer directory: {tokenizer_path}")

    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256.lower() != args.expected_checkpoint_sha256.strip().lower():
        raise ValueError(
            "checkpoint SHA256 mismatch: "
            f"observed={checkpoint_sha256}, "
            f"expected={args.expected_checkpoint_sha256}"
        )
    sidecar_path = checkpoint_path.with_suffix(".json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"missing compact-checkpoint sidecar: {sidecar_path}")
    metadata = compact_checkpoint_metadata(str(checkpoint_path))
    validate_checkpoint_metadata(
        metadata,
        checkpoint_path=checkpoint_path,
        expected_edit_count=args.expected_edit_count,
        expected_rewrite_layers=DEFAULT_REWRITE_LAYERS,
    )
    source_run_config = json.loads(
        source_run_config_path.read_text(encoding="utf-8")
    )
    if str(source_run_config.get("fingerprint")) != str(
        metadata.get("run_fingerprint")
    ):
        raise ValueError(
            "checkpoint/source-run fingerprint mismatch: "
            f"checkpoint={metadata.get('run_fingerprint')!r}, "
            f"run={source_run_config.get('fingerprint')!r}"
        )
    if str(metadata.get("editing_method")) != args.expected_editor:
        raise ValueError(
            f"checkpoint editor={metadata.get('editing_method')!r}, "
            f"expected {args.expected_editor!r}"
        )
    validate_source_edit_count(source_run_config, args.expected_edit_count)
    if int(source_run_config.get("batch_size", -1)) != 1:
        raise ValueError("source endpoint was not produced by sequential batch-1 editing")
    if int(source_run_config.get("seed", -1)) != 42 or int(
        source_run_config.get("analysis_seed", -1)
    ) != 42:
        raise ValueError("source run does not use the canonical seed/analysis_seed=42")
    if str(source_run_config.get("request_fingerprint")) != str(
        metadata.get("request_fingerprint")
    ):
        raise ValueError("checkpoint/source-run request fingerprints disagree")
    if args.expected_condition == "sphere":
        if (
            source_run_config.get("analysis_condition") != "sphere"
            or source_run_config.get("sphere_enabled") is not True
            or source_run_config.get("residual_gain_regularization") is not False
        ):
            raise ValueError("source run config is not the intended SPHERE condition")
    elif args.expected_condition in {"hiddennorm", "hiddennorm_attn"}:
        validate_hiddennorm_condition(
            source_run_config,
            preserve_early_attention=args.expected_condition == "hiddennorm_attn",
        )
    elif args.expected_condition == "sadr":
        sadr_signature = (
            source_run_config.get("analysis_condition") == "sadr"
            and source_run_config.get("sadr_regularization") is True
            and float(source_run_config.get("sadr_lambda", float("nan"))) == 0.01
            and source_run_config.get("sadr_attn_layers") == list(range(32))
            and float(
                source_run_config.get("sadr_efficacy_threshold", float("nan"))
            )
            == 0.5
            and source_run_config.get("residual_gain_regularization") is False
        )
        if not sadr_signature:
            raise ValueError("source run config is not the intended SADR condition")
    elif args.expected_condition == "encore":
        expected_norm_lambda = 20.0 if args.expected_editor == "MEMIT" else 0.0
        encore_signature = (
            source_run_config.get("analysis_condition") == "encore"
            and source_run_config.get("encore_enabled") is True
            and int(source_run_config.get("encore_mpes_top1_steps", -1)) == 2
            and source_run_config.get("encore_mpes_exclude_first_context") is True
            and float(
                source_run_config.get("encore_norm_lambda", float("nan"))
            )
            == expected_norm_lambda
            and source_run_config.get("residual_gain_regularization") is False
        )
        if not encore_signature:
            raise ValueError("source run config is not the intended ENCORE condition")
    else:
        anchor_path = Path(str(source_run_config.get("nas_anchor_path", "")))
        nas_signature = (
            source_run_config.get("analysis_condition") == "nas"
            and source_run_config.get("nas_enabled") is True
            and float(source_run_config.get("nas_outlier_factor", float("nan")))
            == 2.0
            and source_run_config.get("nas_outlier_mode") == "skip_delta"
            and anchor_path.name
            == "memit_llama3_L8_counterfact_n1000_seed0_q0.95_eos1.json"
            and source_run_config.get("residual_gain_regularization") is False
        )
        if not nas_signature:
            raise ValueError("source run config is not the intended NAS condition")
        if not anchor_path.is_file():
            raise FileNotFoundError(f"missing NAS anchor manifest: {anchor_path}")
        expected_anchor_sha256 = (
            "d07237a2f3846ad3e916f1dcca496531ac60819da22aba145cec84a569a3d4d1"
        )
        if sha256_file(anchor_path) != expected_anchor_sha256:
            raise ValueError(f"NAS anchor manifest SHA256 mismatch: {anchor_path}")
    resolved_source, resolved_checkpoint, resolved_metadata = resolve_model_load_spec(
        str(checkpoint_path), str(base_model_path)
    )
    if Path(resolved_source).expanduser().resolve() != base_model_path:
        raise RuntimeError(f"resolved Base source changed unexpectedly: {resolved_source}")
    if Path(str(resolved_checkpoint)).resolve() != checkpoint_path:
        raise RuntimeError(
            f"resolved compact checkpoint changed unexpectedly: {resolved_checkpoint}"
        )
    if resolved_metadata != metadata:
        raise RuntimeError("resolved compact-checkpoint metadata is inconsistent")

    requests_sha256 = sha256_file(requests_path)
    if requests_sha256.lower() != args.expected_requests_sha256.strip().lower():
        raise ValueError(
            "request-file SHA256 mismatch: "
            f"observed={requests_sha256}, expected={args.expected_requests_sha256}"
        )
    positions = tuple(args.positions)
    unknown_positions = sorted(set(positions) - {"subject_last", "prompt_last"})
    if unknown_positions:
        raise ValueError(f"unsupported positions: {unknown_positions}")
    if positions != ("subject_last", "prompt_last"):
        raise ValueError(
            "the Base-compatible A/M/O plot contract requires positions="
            "subject_last,prompt_last"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), trust_remote_code=args.trust_remote_code
    )
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
        str(checkpoint_path),
        args.torch_dtype,
        args.device_map,
        args.trust_remote_code,
        args.attn_implementation,
        base_model_path=str(base_model_path),
    )
    layers = parse_layers(args.layers, len(model_layers(model)))
    if layers != tuple(range(len(model_layers(model)))):
        raise ValueError(
            "the requested repeat must use every decoder layer; "
            f"resolved layers={layers}"
        )

    config: Dict[str, Any] = {
        "schema_version": 1,
        "measurement": "fixed_edited_endpoint_a_m_o_reference_cosines",
        "state_label": args.state_label,
        "state_semantics": (
            f"fixed cumulative endpoint after {args.expected_edit_count} sequential batch-1 edits; "
            "the delta checkpoint is applied once and no edits occur during probes"
        ),
        "model_name": str(base_model_path),
        "base_model_path": str(base_model_path),
        "model_config_name_or_path": str(
            getattr(model.config, "_name_or_path", base_model_path)
        ),
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "model_type": str(getattr(model.config, "model_type", "")),
        "hidden_size": int(getattr(model.config, "hidden_size")),
        "decoder_layer_count": len(model_layers(model)),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "expected_checkpoint_sha256": args.expected_checkpoint_sha256,
        "checkpoint_sidecar_path": str(sidecar_path),
        "checkpoint_sidecar_sha256": sha256_file(sidecar_path),
        "checkpoint_metadata": metadata,
        "source_run_config_path": str(source_run_config_path),
        "source_run_config_sha256": sha256_file(source_run_config_path),
        "source_run_fingerprint": source_run_config.get("fingerprint"),
        "condition": args.expected_condition,
        "editor": args.expected_editor,
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
            "output": "O_l = H_(l+1), complete decoder block-l output",
            "previous_mlp": "M_(l-1), preceding block MLP output",
        },
        "attention_reference_metrics": list(
            attention_analysis.ATTENTION_REFERENCE_METRICS
        ),
        "matrix_metrics": h_m_analysis.MATRIX_DEFINITIONS,
        "layer_metrics": h_m_analysis.LAYER_DEFINITIONS,
        "matrix_layout": "[prompt, position, anchor_layer, reference_layer]",
    }
    config["fingerprint"] = fingerprint(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != config["fingerprint"]:
            raise ValueError(f"existing output has a different fingerprint: {config_path}")
    else:
        _atomic_json(config_path, config)
    top_complete_path = output_dir / "complete.json"
    if top_complete_path.exists():
        print(f"[done] existing complete A/M/O endpoint artifact: {output_dir}")
        return

    parameter_versions_before = {
        name: int(parameter._version) for name, parameter in model.named_parameters()
    }
    attention_arrays = attention_analysis.collect_component_cosines(
        model,
        tokenizer,
        probes,
        layers,
        positions,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
    )
    h_m_arrays = h_m_analysis.collect_h_m_reference_cosines(
        model,
        tokenizer,
        probes,
        layers,
        positions,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
    )
    parameter_versions_after = {
        name: int(parameter._version) for name, parameter in model.named_parameters()
    }
    changed_parameters = sorted(
        name
        for name, version in parameter_versions_before.items()
        if parameter_versions_after[name] != version
    )
    if changed_parameters:
        raise RuntimeError(
            "model parameters mutated while probing the fixed endpoint: "
            f"{changed_parameters}"
        )
    validation = cross_artifact_validation(
        attention_arrays,
        h_m_arrays,
        tolerance=args.validation_tolerance,
    )

    attention_manifest = attention_analysis.write_outputs(
        attention_dir,
        probes=probes,
        layers=layers,
        positions=positions,
        arrays=attention_arrays,
        config=config,
    )
    attention_manifest.update(
        {
            "measurement": "fixed_edited_endpoint_attention_reference_cosines",
            "state_label": args.state_label,
            "checkpoint_sha256": checkpoint_sha256,
        }
    )
    _atomic_json(attention_dir / "complete.json", attention_manifest)

    h_m_manifest = h_m_analysis.write_outputs(
        h_m_dir,
        probes=probes,
        layers=layers,
        positions=positions,
        arrays=h_m_arrays,
        config=config,
    )
    h_m_manifest.update(
        {
            "measurement": "fixed_edited_endpoint_m_o_reference_cosines",
            "state_label": args.state_label,
            "checkpoint_sha256": checkpoint_sha256,
            "h_anchor": "O_l = H_(l+1), decoder block-l output",
        }
    )
    _atomic_json(h_m_dir / "complete.json", h_m_manifest)

    top_manifest = {
        "schema_version": 1,
        "measurement": "fixed_edited_endpoint_a_m_o_reference_cosines",
        "complete": True,
        "state_label": args.state_label,
        "state_semantics": config["state_semantics"],
        "n_prompts": len(probes),
        "positions": list(positions),
        "layers": list(layers),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_sidecar_sha256": sha256_file(sidecar_path),
        "source_run_config_path": str(source_run_config_path),
        "source_run_config_sha256": sha256_file(source_run_config_path),
        "source_run_fingerprint": source_run_config.get("fingerprint"),
        "condition": args.expected_condition,
        "editor": args.expected_editor,
        "requests_path": str(requests_path),
        "requests_sha256": requests_sha256,
        "validation": validation,
        "artifacts": {
            "attention_reference": {
                "path": str(attention_dir.relative_to(output_dir)),
                "complete_sha256": sha256_file(attention_dir / "complete.json"),
            },
            "h_m_reference": {
                "path": str(h_m_dir.relative_to(output_dir)),
                "complete_sha256": sha256_file(h_m_dir / "complete.json"),
            },
        },
        "files": {
            "config": {
                "path": config_path.name,
                "sha256": sha256_file(config_path),
                "bytes": config_path.stat().st_size,
            }
        },
    }
    _atomic_json(top_complete_path, top_manifest)
    print(
        f"[done] wrote fixed-endpoint A/M/O analysis for {args.state_label} "
        f"({len(probes)} prompts, {len(layers)} layers) to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
