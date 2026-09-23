#!/usr/bin/env python3
"""Measure projected residual gain during cumulative MEMIT/AlphaEdit editing.

The edited model stays in memory.  Full model checkpoints are never written;
optionally, the small set of rewrite parameters can be saved at selected edit
counts.  At a configured set of edit counts, this runner measures, in fixed
Base-PCA coordinates:

* projected residual gain ``||U^T (d h_{l+1}/d h_l - I) U||_F``;
* Base-relative gain attenuation and normalized-Jacobian change;
* residual-input RMS, full-space hidden drift, and attention divergence;
* actual total/attention/MLP write RMS, diversity, and common-mode fraction;
* optional full/fixed-Base-PCA residual covariance spectra; and
* locality preservation against cached Base-model teacher-forced outputs;
* checkpoint rewrite/rephrase/locality target likelihoods and new-vs-old
  target margins, in addition to the existing token-accuracy metrics.

Every completed edit-count directory has a fingerprinted completion marker.
By default, a resumed run replays edits in memory while preserving completed
analyses.  When ``--resume-layer-checkpoint`` is supplied, the saved cumulative
rewrite-parameter delta is applied to Base and editing resumes directly after
that checkpoint instead.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EVALUATE_DIR = PROJECT_ROOT / "evaluate"
EASYEDIT_DIR = PROJECT_ROOT / "EasyEdit"
for path in (PROJECT_ROOT, SCRIPT_DIR, EVALUATE_DIR, EASYEDIT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diagnostics.analyze_geometry_locality_correlation import pearson, spearman  # noqa: E402
from diagnostics.early_attention_condition import (  # noqa: E402
    add_early_attention_arguments,
    configure_early_attention_hparams,
    early_attention_config,
)
from diagnostics.o0_axis_condition import (  # noqa: E402
    CONDITION_COMPONENTS as O0_AXIS_CONDITION_COMPONENTS,
    add_o0_axis_arguments,
    configure_o0_axis_hparams,
    o0_axis_config,
    validate_o0_axis_args,
)
from diagnostics.analyze_residual_spectrum import (  # noqa: E402
    EPS,
    aggregate_jacobians,
    analyze_projected_jacobians,
    collect_residual_states,
    fit_shared_bases,
    load_external_bases,
    load_probes,
    model_input_device,
    model_layers,
    parse_int_list,
    probe_inputs,
    safe_key,
    spectrum_metrics,
    write_csv,
)
from diagnostics.analyze_residual_branch_collapse import (  # noqa: E402
    collect_forward_writes,
    load_write_vectors,
    save_write_vectors,
    summarize_delta_writes,
    vector_summary as branch_vector_summary,
)
from diagnostics.compare_locality_proxies import (  # noqa: E402
    ALL_PROXIES,
    collect_base_features,
    compare_forward_features,
    jacobian_divergences,
)
from diagnostics.plot_unified_branch_rms_trajectory import (  # noqa: E402
    plot_run_safely as plot_unified_branch_rms_trajectory,
)
from eval_hf_easyedit import (  # noqa: E402
    collect_group_items,
    compute_locality_outputs,
    compute_rewrite_scores,
)
from easyeditor.util.edited_layer_checkpoint import (  # noqa: E402
    load_edited_parameter_checkpoint,
    rewrite_parameter_names,
    save_parameter_delta_checkpoint,
    snapshot_parameters,
)
from diagnostics.aggregate_edit_analysis_artifacts import aggregate_run as aggregate_edit_artifacts  # noqa: E402
from diagnostics.post_update_tracking import (  # noqa: E402
    capture_boundary_nodes,
    parse_nodes as parse_post_update_nodes,
    save_paired_post_update_artifact,
    score_request_behavior,
)
from diagnostics.virtual_actual_tracking import VirtualActualEditRecorder  # noqa: E402
from diagnostics.fixed_probe_tracking import (  # noqa: E402
    DEFAULT_CONTEXTS as FIXED_PROBE_DEFAULT_CONTEXTS,
    FixedProbeTracker,
    file_sha256 as fixed_probe_file_sha256,
    parse_contexts as parse_fixed_probe_contexts,
)
from diagnostics.edit_order_manifest import (  # noqa: E402
    apply_manifest_order,
    file_sha256 as edit_order_file_sha256,
    validate_manifest as validate_edit_order_manifest,
)
from scripts import run_rgr_batch as edit_runner  # noqa: E402


CONTEXT_CHOICES = {
    "edit_subject_last",
    "edit_prompt_last",
    "heldout_subject_last",
    "heldout_prompt_last",
    "locality_prompt_last",
}
COMPLETION_FILE = "complete.json"


def parse_csv_strings(text: str) -> List[str]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("list contains duplicate values")
    return values


def parse_steps(text: str) -> List[int]:
    values = sorted(parse_int_list(text))
    if values[0] < 0:
        raise argparse.ArgumentTypeError("edit-count steps must be non-negative")
    if 0 not in values:
        values.insert(0, 0)
    return values


def add_oedit_arguments(parser: argparse.ArgumentParser) -> None:
    """OEdit is an explicit standalone condition; legacy defaults stay inactive."""
    parser.add_argument("--oedit-lambda-history", type=float, default=50.0)
    parser.add_argument("--oedit-lambda-gradient", type=float, default=50.0)
    parser.add_argument("--oedit-gradient-rank-per-edit", type=float, default=1.0)
    parser.add_argument("--oedit-gradient-cache-path", default=None)
    parser.add_argument("--oedit-reference-run", default=None,
                        help="Canonical HN run whose native hparams and ordered requests must match")
    parser.add_argument("--oedit-loss-type", choices=["mean_abs_cosine"], default="mean_abs_cosine")


def validate_oedit_args(args: argparse.Namespace, condition: str) -> None:
    if "oedit" not in str(condition).split("+"):
        return
    if condition != "oedit" or bool(getattr(args, "residual_gain_regularization", False)):
        raise ValueError("OEdit must be the standalone 'oedit' condition")
    if args.editing_method not in {"MEMIT", "AlphaEdit"} or args.batch_size != 1:
        raise ValueError("The HN-matched OEdit trajectory requires MEMIT/AlphaEdit and batch size 1")
    for name in ("context_multikey_enabled", "key_gaussian_noise_enabled",
                 "tangent_layer_allocation_enabled", "early_attention_preservation_enabled",
                 "o0_axis_preservation_enabled"):
        if bool(getattr(args, name, False)):
            raise ValueError(f"OEdit cannot be combined with {name}")
    if getattr(args, "inner_margin_schedule", "legacy") != "legacy":
        raise ValueError("OEdit requires the native legacy inner schedule")
    if getattr(args, "resume_layer_checkpoint", None):
        raise ValueError("OEdit parameter-only resume loses cumulative OEdit/editor state; replay from Base")
    if hasattr(args, "full_primary_diagnostics") and not bool(args.full_primary_diagnostics):
        raise ValueError("OEdit requires --full-primary-diagnostics=1")
    for name in ("oedit_lambda_history", "oedit_lambda_gradient", "oedit_gradient_rank_per_edit"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    cache = getattr(args, "oedit_gradient_cache_path", None)
    if not cache or not Path(cache).expanduser().is_file():
        raise FileNotFoundError("OEdit requires an existing --oedit-gradient-cache-path")
    reference = getattr(args, "oedit_reference_run", None)
    if reference:
        for name in ("run_config.json", "requests.json", "analysis_requests.json"):
            if not (Path(reference).expanduser() / name).is_file():
                raise FileNotFoundError(f"OEdit reference run is missing {name}")


def oedit_cli(args: argparse.Namespace, condition: str) -> List[str]:
    if condition != "oedit":
        return []
    result = [
        "--oedit-lambda-history", str(args.oedit_lambda_history),
        "--oedit-lambda-gradient", str(args.oedit_lambda_gradient),
        "--oedit-gradient-rank-per-edit", str(args.oedit_gradient_rank_per_edit),
        "--oedit-gradient-cache-path", str(Path(args.oedit_gradient_cache_path).expanduser().resolve()),
        "--oedit-loss-type", args.oedit_loss_type,
    ]
    if getattr(args, "oedit_reference_run", None):
        result.extend(["--oedit-reference-run", str(Path(args.oedit_reference_run).expanduser().resolve())])
    return result


def configure_oedit_hparams(args: argparse.Namespace, hparams: Any, *, condition: str, output_dir: Path) -> None:
    if condition != "oedit":
        if hasattr(hparams, "oedit_enabled"):
            hparams.oedit_enabled = False
        return
    validate_oedit_args(args, condition)
    if list(hparams.layers) != [4, 5, 6, 7, 8] or hparams.fact_token != "subject_last":
        raise ValueError("HN-matched OEdit requires native L4-L8 subject_last editing")
    if bool(getattr(hparams, "hn_recovery_enabled", False)):
        raise ValueError("HN-matched OEdit cannot use HN recovery")
    hparams.oedit_enabled = True
    for name in ("oedit_lambda_history", "oedit_lambda_gradient", "oedit_gradient_rank_per_edit", "oedit_loss_type"):
        setattr(hparams, name, getattr(args, name))
    hparams.oedit_gradient_cache_path = str(Path(args.oedit_gradient_cache_path).expanduser().resolve())
    hparams.oedit_log_path = str(output_dir / "oedit.jsonl")
    hparams.oedit_reference_run = (
        str(Path(args.oedit_reference_run).expanduser().resolve())
        if getattr(args, "oedit_reference_run", None) else None
    )
    if hparams.oedit_reference_run:
        reference = json.loads((Path(hparams.oedit_reference_run) / "run_config.json").read_text())
        for name, expected in reference["effective_editing_hparams"].items():
            if name == "residual_gain_objective":
                continue
            actual = getattr(hparams, name, None)
            if actual != expected:
                raise ValueError(f"Canonical HN native hparam mismatch: {name}={actual!r}, expected {expected!r}")
        for name in ("model_name", "sample_size", "batch_size", "seed", "append_eos_to_target"):
            if getattr(args, name) != reference[name]:
                raise ValueError(f"Canonical HN run setting mismatch: {name}")
        if getattr(args, "edit_order", "prefix") != "prefix" or getattr(args, "edit_order_file", None):
            raise ValueError("Canonical HN matching requires its original prefix order")
    hparams.mechanism_early_stop_mode = "base"
    hparams.residual_gain_early_stop_mode = "base"
    # This runner starts from Base and replays its complete prefix on restart.
    # Runtime tensors never enter run-config fingerprints or JSON payloads.
    from easyeditor.util.oedit import reset_oedit_state
    reset_oedit_state(hparams)


def oedit_config(hparams: Any) -> Dict[str, Any]:
    if not bool(getattr(hparams, "oedit_enabled", False)):
        return {}
    cache = Path(hparams.oedit_gradient_cache_path)
    digest = hashlib.sha256()
    with cache.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    payload = {
        "oedit_enabled": True,
        "oedit_lambda_history": float(hparams.oedit_lambda_history),
        "oedit_lambda_gradient": float(hparams.oedit_lambda_gradient),
        "oedit_gradient_rank_per_edit": float(hparams.oedit_gradient_rank_per_edit),
        "oedit_gradient_cache_path": str(cache),
        "oedit_gradient_cache_sha256": digest.hexdigest(),
        "oedit_loss_type": str(hparams.oedit_loss_type),
        "oedit_log_path": str(hparams.oedit_log_path),
        "mechanism_early_stop_mode": "base",
        "oedit_scope": "paper_based_last_target_layer_adaptation",
    }
    if getattr(hparams, "oedit_reference_run", None):
        reference = Path(hparams.oedit_reference_run)
        payload["oedit_reference_run"] = str(reference)
        payload["oedit_reference_sha256"] = {
            name: hashlib.sha256((reference / name).read_bytes()).hexdigest()
            for name in ("run_config.json", "requests.json", "analysis_requests.json")
        }
    return payload


def verify_oedit_reference_requests(hparams: Any, requests: Sequence[Mapping[str, Any]],
                                    analysis_requests: Sequence[Mapping[str, Any]]) -> None:
    if not bool(getattr(hparams, "oedit_enabled", False)) or not getattr(hparams, "oedit_reference_run", None):
        return
    reference = Path(hparams.oedit_reference_run)
    for name, actual in (("requests.json", requests), ("analysis_requests.json", analysis_requests)):
        expected = json.loads((reference / name).read_text())
        if list(actual) != expected:
            raise ValueError(f"Canonical HN {name} differs after request normalization/EOS append")


def record_oedit_state(hparams: Any, output_dir: Path, edit_count: int) -> None:
    if not bool(getattr(hparams, "oedit_enabled", False)):
        return
    state = getattr(hparams, "_oedit_runtime", None)
    committed = int(getattr(state, "edit_count", -1))
    if committed != edit_count:
        raise RuntimeError(f"OEdit committed {committed} edits after trajectory edit {edit_count}")
    atomic_json(output_dir / "oedit_state_summary.json", {
        "editing_method": hparams.alg_name, "edit_count": int(edit_count),
        "committed_edits": committed,
        "cumulative_numerical_rank": int(state.numerical_rank),
        "state_reconstruction": "replay_from_base_required",
    })


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=edit_runner.json_default)
    os.replace(temporary, path)


def prepare_append_log_for_checkpoint_resume(
    path: Path,
    requests: Sequence[Mapping[str, Any]],
    resume_edit_count: int,
) -> None:
    """Keep one clean log record per completed-prefix optimization step.

    Method optimization logs are append-only. A checkpoint resume would
    otherwise duplicate records if an earlier process ran beyond the saved
    checkpoint before stopping. Preserve the original file as a timestamped
    backup and atomically rewrite the active log to the checkpoint prefix.
    """

    if not path.is_file() or path.stat().st_size == 0:
        return
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    backup = path.with_name(
        f"{path.stem}.pre_resume_edit{resume_edit_count}_{timestamp}{path.suffix}"
    )
    shutil.copy2(path, backup)

    retained: List[Mapping[str, Any]] = []
    if resume_edit_count > 0:
        allowed_case_ids = {
            str(request.get("case_id"))
            for request in requests[:resume_edit_count]
        }
        seen: set[Tuple[str, str, str, str, str]] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # Abrupt process termination can leave one partial final
                    # JSONL record. The byte-identical backup above preserves
                    # it for audit; omit it from the clean resume trace.
                    print(
                        "[resume] dropping malformed optimizer-log record: "
                        f"{path}:{line_number}"
                    )
                    continue
                case_id = str(record.get("case_id"))
                if case_id not in allowed_case_ids:
                    continue
                key = (
                    case_id,
                    str(record.get("write_layer")),
                    str(record.get("inner_step")),
                    str(record.get("method")),
                    str(record.get("baseline")),
                )
                if key in seen:
                    continue
                seen.add(key)
                retained.append(record)

    temporary = path.with_name(f".{path.name}.{os.getpid()}.resume.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in retained:
                handle.write(
                    json.dumps(record, ensure_ascii=False, default=str) + "\n"
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        "[resume] cleaned append log to completed prefix: "
        f"records={len(retained)} backup={backup}"
    )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_fingerprint(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def request_fingerprint(requests: Sequence[Mapping[str, Any]]) -> str:
    signatures = [
        {
            "case_id": request.get("case_id"),
            "prompt": request.get("prompt"),
            "subject": request.get("subject"),
            "target_new": request.get("target_new"),
            "locality": request.get("locality", {}),
        }
        for request in requests
    ]
    return hashlib.sha256(canonical_json(signatures).encode("utf-8")).hexdigest()


def analysis_request_fingerprint(requests: Sequence[Mapping[str, Any]]) -> str:
    """Fingerprint diagnostic prompts while ignoring edit-target EOS normalization."""

    signatures = [
        {
            "case_id": request.get("case_id"),
            "prompt": request.get("prompt"),
            "subject": request.get("subject"),
            "locality": request.get("locality", {}),
        }
        for request in requests
    ]
    return hashlib.sha256(canonical_json(signatures).encode("utf-8")).hexdigest()


def validate_or_write_config(path: Path, config: Mapping[str, Any]) -> str:
    fingerprint = config_fingerprint(config)
    if path.exists():
        existing = read_json(path)
        if existing.get("fingerprint") != fingerprint:
            raise ValueError(
                f"Existing run config at {path} does not match this invocation. "
                "Choose a new OUTPUT_DIR so completed steps are never mixed or overwritten."
            )
        return fingerprint
    atomic_json(path, {"fingerprint": fingerprint, **dict(config)})
    return fingerprint


def step_dir(output_dir: Path, step: int) -> Path:
    return output_dir / f"step_{step:03d}"


def step_is_complete(output_dir: Path, step: int, fingerprint: str) -> bool:
    marker = step_dir(output_dir, step) / COMPLETION_FILE
    if not marker.exists():
        return False
    try:
        payload = read_json(marker)
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("fingerprint") != fingerprint or int(payload.get("edit_count", -1)) != step:
        return False
    required_files = payload.get("required_files", [])
    return all((step_dir(output_dir, step) / relative).is_file() for relative in required_files)


def component_is_complete(directory: Path, fingerprint: str, edit_count: int) -> bool:
    marker = directory / COMPLETION_FILE
    if not marker.exists():
        return False
    try:
        payload = read_json(marker)
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("fingerprint") != fingerprint or int(payload.get("edit_count", -1)) != edit_count:
        return False
    return all((directory / relative).is_file() for relative in payload.get("required_files", []))


def save_feature_map(path: Path, feature_map: Mapping[Tuple[int, int, str], np.ndarray]) -> None:
    arrays = {
        safe_key(f"P{prompt}", f"L{layer}", kind): np.asarray(value)
        for (prompt, layer, kind), value in feature_map.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_feature_map(
    path: Path,
    n_prompts: int,
    layers: Sequence[int],
) -> Dict[Tuple[int, int, str], np.ndarray]:
    output: Dict[Tuple[int, int, str], np.ndarray] = {}
    with np.load(path) as loaded:
        for prompt in range(n_prompts):
            for layer in layers:
                for kind in ("hidden", "attention"):
                    key = safe_key(f"P{prompt}", f"L{layer}", kind)
                    if key not in loaded:
                        raise KeyError(f"Base feature cache {path} has no array {key!r}")
                    output[(prompt, layer, kind)] = np.asarray(loaded[key]).copy()
    return output


def save_downproj_features(
    path: Path,
    features: Mapping[Tuple[str, int, str], np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{
            safe_key(position, f"L{layer}", kind): np.asarray(values)
            for (position, layer, kind), values in features.items()
        },
    )


def save_downproj_feature_deltas(
    path: Path,
    current_features: Mapping[Tuple[str, int, str], np.ndarray],
    base_features: Mapping[Tuple[str, int, str], np.ndarray],
) -> None:
    """Persist explicit fixed-probe ``delta k`` and ``delta output`` arrays."""

    arrays: Dict[str, np.ndarray] = {}
    for (position, layer, kind), current in current_features.items():
        identity = (position, layer, kind)
        if identity not in base_features:
            raise KeyError(f"Base down-projection features have no {identity}")
        current_array = np.asarray(current, dtype=np.float32)
        base_array = np.asarray(base_features[identity], dtype=np.float32)
        if current_array.shape != base_array.shape:
            raise ValueError(
                f"Feature shape mismatch for {identity}: "
                f"{current_array.shape} != {base_array.shape}"
            )
        arrays[
            safe_key(position, f"L{layer}", f"delta_{kind}")
        ] = current_array - base_array
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_downproj_features(
    path: Path,
    positions: Sequence[str],
    layers: Sequence[int],
) -> Dict[Tuple[str, int, str], np.ndarray]:
    output: Dict[Tuple[str, int, str], np.ndarray] = {}
    with np.load(path) as loaded:
        for position in positions:
            for layer in layers:
                for kind in ("key", "output"):
                    key = safe_key(position, f"L{layer}", kind)
                    if key not in loaded:
                        raise KeyError(
                            f"Down-projection feature cache {path} has no {key!r}"
                        )
                    output[(position, layer, kind)] = np.asarray(
                        loaded[key], dtype=np.float32
                    ).copy()
    return output


@torch.inference_mode()
def collect_downproj_features(
    model: Any,
    probes: Sequence[Any],
    layers: Sequence[int],
    positions: Sequence[str],
) -> Dict[Tuple[str, int, str], np.ndarray]:
    """Capture MLP down-projection inputs k and outputs Wk."""

    decoder = model_layers(model)
    device = model_input_device(model)
    values: Dict[Tuple[str, int, str], List[np.ndarray]] = defaultdict(list)
    for prompt_index, probe in enumerate(probes):
        captured: Dict[Tuple[int, str], torch.Tensor] = {}
        handles = []

        def pre_hook(layer: int):
            def hook(_module: Any, inputs: Tuple[Any, ...]) -> None:
                captured[(layer, "key")] = inputs[0].detach()

            return hook

        def output_hook(layer: int):
            def hook(
                _module: Any, _inputs: Tuple[Any, ...], output: torch.Tensor
            ) -> None:
                captured[(layer, "output")] = output.detach()

            return hook

        for layer in layers:
            module = decoder[layer].mlp.down_proj
            handles.extend(
                [
                    module.register_forward_pre_hook(pre_hook(layer)),
                    module.register_forward_hook(output_hook(layer)),
                ]
            )
        try:
            outputs = model(
                **probe_inputs(probe, device),
                use_cache=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        for position in positions:
            token_pos = int(probe.positions[position])
            for layer in layers:
                for kind in ("key", "output"):
                    selected = captured[(layer, kind)][0, token_pos]
                    values[(position, layer, kind)].append(
                        selected.float().cpu().numpy()
                    )
        del outputs, captured
        print(
            f"[feature-update] captured {prompt_index + 1}/{len(probes)} prompts"
        )
    return {
        key: np.stack(selected).astype(np.float32, copy=False)
        for key, selected in values.items()
    }


def decompose_feature_update(
    *,
    model: Any,
    positions: Sequence[str],
    layers: Sequence[int],
    base_features: Mapping[Tuple[str, int, str], np.ndarray],
    current_features: Mapping[Tuple[str, int, str], np.ndarray],
    base_weights: Mapping[int, torch.Tensor],
    state: str,
    edit_count: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compute W0Δk + ΔWk0 + ΔWΔk at a cumulative checkpoint."""

    decoder = model_layers(model)
    prompt_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    component_vectors: Dict[Tuple[str, int, str], np.ndarray] = {}

    for layer in layers:
        current_weight = decoder[layer].mlp.down_proj.weight.detach()
        device = current_weight.device
        compute_dtype = torch.float32
        w_current = current_weight.to(dtype=compute_dtype)
        w_base = base_weights[layer].to(
            device=device, dtype=compute_dtype
        )
        delta_weight = w_current - w_base
        for position in positions:
            k0 = torch.from_numpy(
                np.asarray(base_features[(position, layer, "key")])
            ).to(device=device, dtype=compute_dtype)
            kt = torch.from_numpy(
                np.asarray(current_features[(position, layer, "key")])
            ).to(device=device, dtype=compute_dtype)
            f0 = torch.from_numpy(
                np.asarray(base_features[(position, layer, "output")])
            ).to(device=device, dtype=compute_dtype)
            ft = torch.from_numpy(
                np.asarray(current_features[(position, layer, "output")])
            ).to(device=device, dtype=compute_dtype)
            delta_key = kt - k0
            components = {
                "feature_response": delta_key @ w_base.T,
                "fixed_update": k0 @ delta_weight.T,
                "interaction": delta_key @ delta_weight.T,
                "observed_total": ft - f0,
            }
            components["reconstructed_total"] = (
                components["feature_response"]
                + components["fixed_update"]
                + components["interaction"]
            )
            closure = (
                components["observed_total"]
                - components["reconstructed_total"]
            )
            observed_norm = components["observed_total"].norm(dim=1)
            for component, tensor in components.items():
                cpu = tensor.detach().float().cpu().numpy()
                component_vectors[(position, layer, component)] = cpu
                norms = tensor.norm(dim=1)
                rms = tensor.square().mean(dim=1).sqrt()
                for prompt_index in range(tensor.size(0)):
                    prompt_rows.append(
                        {
                            "state": state,
                            "edit_count": edit_count,
                            "position": position,
                            "layer": layer,
                            "prompt_index": prompt_index,
                            "component": component,
                            "l2": float(norms[prompt_index].item()),
                            "rms": float(rms[prompt_index].item()),
                            "relative_l2_to_observed": float(
                                norms[prompt_index].item()
                                / (observed_norm[prompt_index].item() + EPS)
                            ),
                        }
                    )
            for component, tensor in components.items():
                summary_rows.append(
                    {
                        "state": state,
                        "edit_count": edit_count,
                        "position": position,
                        "layer": layer,
                        "component": component,
                        **branch_vector_summary(
                            list(
                                component_vectors[
                                    (position, layer, component)
                                ]
                            )
                        ),
                    }
                )
            summary_rows.append(
                {
                    "state": state,
                    "edit_count": edit_count,
                    "position": position,
                    "layer": layer,
                    "component": "closure_error",
                    **branch_vector_summary(
                        list(closure.detach().float().cpu().numpy())
                    ),
                }
            )
        del w_current, w_base, delta_weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return prompt_rows, summary_rows


def save_projected_matrices(
    path: Path,
    label: str,
    matrices: Mapping[Tuple[str, int], Sequence[np.ndarray]],
) -> None:
    arrays = {
        safe_key(label, position, f"L{layer}", f"P{prompt}"): matrix
        for (position, layer), values in matrices.items()
        for prompt, matrix in enumerate(values)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_projected_matrices(
    path: Path,
    label: str,
    positions: Sequence[str],
    layers: Sequence[int],
    n_prompts: int,
) -> Dict[Tuple[str, int], List[np.ndarray]]:
    output: Dict[Tuple[str, int], List[np.ndarray]] = {
        (position, layer): [] for position in positions for layer in layers
    }
    with np.load(path) as loaded:
        for position in positions:
            for layer in layers:
                for prompt in range(n_prompts):
                    key = safe_key(label, position, f"L{layer}", f"P{prompt}")
                    if key not in loaded:
                        raise KeyError(f"Projected-Jacobian cache {path} has no array {key!r}")
                    output[(position, layer)].append(np.asarray(loaded[key]).copy())
    return output


def context_layout(contexts: Sequence[str]) -> Dict[str, List[str]]:
    unknown = sorted(set(contexts) - CONTEXT_CHOICES)
    if unknown:
        raise ValueError(f"Unknown contexts: {unknown}; choices are {sorted(CONTEXT_CHOICES)}")
    layout: Dict[str, List[str]] = {}
    edit_positions = []
    if "edit_subject_last" in contexts:
        edit_positions.append("subject_last")
    if "edit_prompt_last" in contexts:
        edit_positions.append("prompt_last")
    if edit_positions:
        layout["edit"] = edit_positions
    heldout_positions = []
    if "heldout_subject_last" in contexts:
        heldout_positions.append("subject_last")
    if "heldout_prompt_last" in contexts:
        heldout_positions.append("prompt_last")
    if heldout_positions:
        layout["heldout"] = heldout_positions
    if "locality_prompt_last" in contexts:
        layout["locality"] = ["prompt_last"]
    return layout


def load_context_probes(
    layout: Mapping[str, Sequence[str]],
    tokenizer: Any,
    args: argparse.Namespace,
    requests_path: Path,
    locality_requests_path: Path,
) -> Dict[str, List[Any]]:
    probes: Dict[str, List[Any]] = {}
    if "edit" in layout:
        probes["edit"] = load_probes(
            str(requests_path),
            tokenizer,
            layout["edit"],
            min(args.probe_prompts, args.sample_size),
            args.max_length,
            probe_source="rewrite",
            selection="prefix",
            seed=args.analysis_seed,
        )
    if "heldout" in layout:
        probes["heldout"] = load_probes(
            args.data_path,
            tokenizer,
            layout["heldout"],
            args.probe_prompts,
            args.max_length,
            probe_source="rewrite",
            exclude_requests_paths=[str(requests_path)],
            selection="random",
            seed=args.analysis_seed,
        )
    if "locality" in layout:
        probes["locality"] = load_probes(
            str(locality_requests_path),
            tokenizer,
            ["prompt_last"],
            min(args.probe_prompts, args.locality_eval_prompts, args.sample_size),
            args.max_length,
            probe_source="locality",
            selection="prefix",
            seed=args.analysis_seed,
        )
    return probes


def prepare_bases_and_features(
    model: Any,
    probes_by_group: Mapping[str, Sequence[Any]],
    layout: Mapping[str, Sequence[str]],
    layers: Sequence[int],
    pca_rank: int,
    output_dir: Path,
) -> Tuple[
    Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    Dict[str, Dict[str, Dict[Tuple[int, int, str], np.ndarray]]],
]:
    bases_by_group: Dict[str, Dict[str, Dict[int, torch.Tensor]]] = {}
    features_by_group: Dict[str, Dict[str, Dict[Tuple[int, int, str], np.ndarray]]] = {}
    hidden_size = int(model.config.hidden_size)

    for group, probes in probes_by_group.items():
        reference_dir = output_dir / "base_reference" / group
        basis_path = reference_dir / "shared_pca_bases.npz"
        if basis_path.exists():
            bases = load_external_bases(
                str(basis_path), layout[group], layers, pca_rank, hidden_size
            )
        else:
            print(f"[base-reference] fitting {group} PCA bases")
            states = collect_residual_states(model, probes, layers, layout[group])
            bases, _, basis_rows = fit_shared_bases(
                states, layout[group], layers, pca_rank
            )
            reference_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                basis_path,
                **{
                    safe_key(position, f"L{layer}"): basis.numpy()
                    for position, layer_bases in bases.items()
                    for layer, basis in layer_bases.items()
                },
            )
            write_csv(reference_dir / "shared_pca_basis_metrics.csv", basis_rows)
            del states
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        bases_by_group[group] = bases

        features_by_group[group] = {}
        for position in layout[group]:
            feature_path = reference_dir / f"base_features_{position}.npz"
            if feature_path.exists():
                feature_map = load_feature_map(feature_path, len(probes), layers)
                print(f"[base-reference] loaded {group}/{position} forward features")
            else:
                print(f"[base-reference] collecting {group}/{position} forward features")
                feature_map = collect_base_features(model, probes, layers, position)
                save_feature_map(feature_path, feature_map)
            features_by_group[group][position] = feature_map

    return bases_by_group, features_by_group


def flatten_locality_outputs(
    outputs: Mapping[int, Mapping[str, Sequence[Any]]]
) -> Dict[str, Any]:
    return {str(index): value for index, value in outputs.items()}


def unflatten_locality_outputs(payload: Mapping[str, Any]) -> Dict[int, Dict[str, List[Any]]]:
    return {int(index): value for index, value in payload.items()}


CONTINUOUS_BEHAVIOR_SUMMARY_FIELDS = (
    "rewrite_target_new_mean_logprob",
    "rewrite_target_new_nll",
    "rewrite_target_new_geomean_prob",
    "rewrite_target_old_mean_logprob",
    "rewrite_target_old_nll",
    "rewrite_target_old_geomean_prob",
    "rewrite_new_vs_old_logprob_margin",
    "rewrite_new_vs_old_geomean_prob_margin",
    "rephrase_target_new_mean_logprob",
    "rephrase_target_new_nll",
    "rephrase_target_new_geomean_prob",
    "rephrase_target_old_mean_logprob",
    "rephrase_target_old_nll",
    "rephrase_target_old_geomean_prob",
    "rephrase_new_vs_old_logprob_margin",
    "rephrase_new_vs_old_geomean_prob_margin",
    "locality_target_mean_logprob",
    "locality_target_nll",
    "locality_target_geomean_prob",
)
CONTINUOUS_BEHAVIOR_COUNT_FIELDS = (
    "rewrite_target_old_count",
    "rephrase_target_old_count",
    "locality_target_case_count",
)


def _finite_mean(values: Iterable[Any]) -> float:
    finite: List[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    return float(np.mean(finite)) if finite else float("nan")


def _target_suffix_text(target: Any) -> str | None:
    """Normalize one teacher-forced target without changing its tokens."""

    if isinstance(target, Mapping):
        target = target.get("str")
    if target is None:
        return None
    text = str(target).strip()
    if not text or text in {"<|endoftext|>", "<|end_of_text|>"}:
        return None
    return text


def _comparable_old_target(
    request: Mapping[str, Any],
    tokenizer: Any,
) -> str | None:
    """Return the original target with the edited target's EOS convention.

    Sequential trajectories append the tokenizer EOS token to ``target_new``.
    Adding the same suffix to the old target keeps the new-vs-old mean-token
    likelihood comparison symmetric.  Missing EasyEdit placeholder targets
    remain unavailable rather than being scored as literal text.
    """

    old_target = _target_suffix_text(request.get("ground_truth"))
    if old_target is None:
        return None
    new_target = _target_suffix_text(request.get("target_new"))
    eos_token = getattr(tokenizer, "eos_token", None)
    if (
        eos_token
        and new_target is not None
        and new_target.endswith(str(eos_token))
        and not old_target.endswith(str(eos_token))
    ):
        old_target = old_target + str(eos_token)
    return old_target


@torch.inference_mode()
def compute_target_likelihoods(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    targets: Sequence[Any],
    batch_size: int,
) -> List[Dict[str, Any]]:
    """Score target suffixes by mean teacher-forced token log probability.

    ``geomean_prob`` is ``exp(mean_logprob)``.  It is the geometric mean of
    the probabilities assigned to the target tokens, not the generally tiny
    full-sequence probability.  Rows for unavailable targets contain NaNs so
    callers retain one-to-one alignment with their requests.
    """

    if len(prompts) != len(targets):
        raise ValueError("Prompts and targets must have equal lengths")
    results: List[Dict[str, Any]] = [
        {
            "mean_logprob": float("nan"),
            "nll": float("nan"),
            "geomean_prob": float("nan"),
            "token_count": 0,
        }
        for _ in prompts
    ]
    encoded_rows: List[Dict[str, Any]] = []
    for index, (prompt, raw_target) in enumerate(zip(prompts, targets)):
        target = _target_suffix_text(raw_target)
        if target is None:
            continue
        suffix_text = " " + target
        suffix_ids = tokenizer(
            suffix_text,
            add_special_tokens=False,
        )["input_ids"]
        if not suffix_ids:
            continue
        full_ids = tokenizer(
            str(prompt).rstrip() + suffix_text,
            add_special_tokens=True,
        )["input_ids"]
        if len(suffix_ids) >= len(full_ids):
            raise ValueError(
                "Cannot locate a target suffix after its prompt: "
                f"prompt={prompt!r} target={target!r}"
            )
        encoded_rows.append(
            {
                "index": index,
                "input_ids": list(full_ids),
                "suffix_length": len(suffix_ids),
            }
        )

    device = model_input_device(model)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError("Tokenizer has neither a pad token nor an EOS token")
    chunk_size = max(int(batch_size), 1)
    for start in range(0, len(encoded_rows), chunk_size):
        chunk = encoded_rows[start : start + chunk_size]
        max_length = max(len(row["input_ids"]) for row in chunk)
        input_ids = torch.full(
            (len(chunk), max_length),
            int(pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_index, row in enumerate(chunk):
            ids = torch.as_tensor(
                row["input_ids"], dtype=torch.long, device=device
            )
            input_ids[row_index, : ids.numel()] = ids
            attention_mask[row_index, : ids.numel()] = 1
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits
        for row_index, row in enumerate(chunk):
            sequence_length = len(row["input_ids"])
            suffix_length = int(row["suffix_length"])
            suffix_start = sequence_length - suffix_length
            target_ids = input_ids[
                row_index, suffix_start:sequence_length
            ]
            target_logits = logits[
                row_index, suffix_start - 1 : sequence_length - 1
            ].float()
            token_logprobs = torch.log_softmax(
                target_logits, dim=-1
            ).gather(1, target_ids.unsqueeze(1)).squeeze(1)
            mean_logprob = float(token_logprobs.mean().item())
            results[int(row["index"])] = {
                "mean_logprob": mean_logprob,
                "nll": -mean_logprob,
                "geomean_prob": math.exp(mean_logprob),
                "token_count": suffix_length,
            }
        del outputs, logits, input_ids, attention_mask
    return results


def _attach_edit_target_likelihoods(
    *,
    rows: List[Dict[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    prompts: Sequence[str],
    model: Any,
    tokenizer: Any,
    batch_size: int,
    prefix: str,
) -> None:
    new_scores = compute_target_likelihoods(
        model,
        tokenizer,
        prompts,
        [request.get("target_new") for request in requests],
        batch_size,
    )
    old_scores = compute_target_likelihoods(
        model,
        tokenizer,
        prompts,
        [_comparable_old_target(request, tokenizer) for request in requests],
        batch_size,
    )
    for row, new_score, old_score in zip(rows, new_scores, old_scores):
        for target_name, score in (("new", new_score), ("old", old_score)):
            row[f"{prefix}_target_{target_name}_mean_logprob"] = score[
                "mean_logprob"
            ]
            row[f"{prefix}_target_{target_name}_nll"] = score["nll"]
            row[f"{prefix}_target_{target_name}_geomean_prob"] = score[
                "geomean_prob"
            ]
            row[f"{prefix}_target_{target_name}_token_count"] = score[
                "token_count"
            ]
        row[f"{prefix}_new_vs_old_logprob_margin"] = (
            float(new_score["mean_logprob"])
            - float(old_score["mean_logprob"])
        )
        row[f"{prefix}_new_vs_old_geomean_prob_margin"] = (
            float(new_score["geomean_prob"])
            - float(old_score["geomean_prob"])
        )


def compute_locality_evaluation(
    model: Any,
    model_name: str,
    hparams: Any,
    tokenizer: Any,
    requests: Sequence[Dict[str, Any]],
    base_outputs: Mapping[int, Mapping[str, Sequence[Any]]],
    batch_size: int,
    continuous_behavior: bool = True,
) -> Tuple[List[Dict[str, Any]], float]:
    locality_items = collect_group_items(list(requests), "locality")
    current_outputs = compute_locality_outputs(
        model=model,
        model_name=model_name,
        hparams=hparams,
        tokenizer=tokenizer,
        locality_items=locality_items,
        device=hparams.device,
        batch_size=batch_size,
    )
    likelihoods_by_index: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    if continuous_behavior and locality_items:
        likelihoods = compute_target_likelihoods(
            model,
            tokenizer,
            [prompt for _, _, prompt, _ in locality_items],
            [target for _, _, _, target in locality_items],
            batch_size,
        )
        for item, likelihood in zip(locality_items, likelihoods):
            likelihoods_by_index[int(item[0])].append(likelihood)

    rows: List[Dict[str, Any]] = []
    for index, request in enumerate(requests):
        scores: List[float] = []
        for locality_key, post_outputs in current_outputs.get(index, {}).items():
            pre_outputs = base_outputs.get(index, {}).get(locality_key)
            if pre_outputs is None:
                continue
            pair_scores = [
                float(np.mean(np.equal(post, pre)))
                for post, pre in zip(post_outputs, pre_outputs)
            ]
            scores.extend(pair_scores)
        if scores:
            row: Dict[str, Any] = {
                "case_id": request.get("case_id", index),
                "locality_acc": float(np.mean(scores)),
                "n_locality_pairs": len(scores),
            }
            if continuous_behavior:
                likelihood_rows = likelihoods_by_index.get(index, [])
                row.update(
                    {
                        "locality_target_mean_logprob": _finite_mean(
                            item["mean_logprob"] for item in likelihood_rows
                        ),
                        "locality_target_nll": _finite_mean(
                            item["nll"] for item in likelihood_rows
                        ),
                        "locality_target_geomean_prob": _finite_mean(
                            item["geomean_prob"] for item in likelihood_rows
                        ),
                        "locality_target_token_count": sum(
                            int(item["token_count"])
                            for item in likelihood_rows
                        ),
                    }
                )
            rows.append(row)
    mean_score = float(np.mean([row["locality_acc"] for row in rows])) if rows else float("nan")
    return rows, mean_score


def compute_rewrite_evaluation(
    model: Any,
    model_name: str,
    hparams: Any,
    tokenizer: Any,
    requests: Sequence[Dict[str, Any]],
    batch_size: int,
    continuous_behavior: bool = True,
) -> Tuple[List[Dict[str, Any]], float]:
    if not requests:
        return [], float("nan")
    scores = compute_rewrite_scores(
        model=model,
        model_name=model_name,
        hparams=hparams,
        tokenizer=tokenizer,
        prompts=[request["prompt"] for request in requests],
        targets=[request["target_new"] for request in requests],
        device=hparams.device,
        batch_size=batch_size,
        test_rephrase=False,
    )
    rows = [
        {"case_id": request.get("case_id", index), "rewrite_acc": float(score)}
        for index, (request, score) in enumerate(zip(requests, scores))
    ]
    if continuous_behavior:
        _attach_edit_target_likelihoods(
            rows=rows,
            requests=requests,
            prompts=[request["prompt"] for request in requests],
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            prefix="rewrite",
        )
    return rows, float(np.mean(scores)) if scores else float("nan")


def compute_rephrase_evaluation(
    model: Any,
    model_name: str,
    hparams: Any,
    tokenizer: Any,
    requests: Sequence[Dict[str, Any]],
    batch_size: int,
    continuous_behavior: bool = True,
) -> Tuple[List[Dict[str, Any]], float]:
    if not requests:
        return [], float("nan")
    prompts = [
        request.get("rephrase_prompt")
        or request.get("rephrase")
        or request["prompt"]
        for request in requests
    ]
    scores = compute_rewrite_scores(
        model=model,
        model_name=model_name,
        hparams=hparams,
        tokenizer=tokenizer,
        prompts=prompts,
        targets=[request["target_new"] for request in requests],
        device=hparams.device,
        batch_size=batch_size,
        test_rephrase=False,
    )
    rows = [
        {"case_id": request.get("case_id", index), "rephrase_acc": float(score)}
        for index, (request, score) in enumerate(zip(requests, scores))
    ]
    if continuous_behavior:
        _attach_edit_target_likelihoods(
            rows=rows,
            requests=requests,
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            prefix="rephrase",
        )
    return rows, float(np.mean(scores)) if scores else float("nan")


def evaluate_step(
    step: int,
    model: Any,
    model_name: str,
    hparams: Any,
    tokenizer: Any,
    all_requests: Sequence[Dict[str, Any]],
    locality_requests: Sequence[Dict[str, Any]],
    base_locality_outputs: Mapping[int, Mapping[str, Sequence[Any]]],
    eval_batch_size: int,
    continuous_behavior: bool = True,
) -> Dict[str, Any]:
    locality_rows, locality_acc = compute_locality_evaluation(
        model,
        model_name,
        hparams,
        tokenizer,
        locality_requests,
        base_locality_outputs,
        eval_batch_size,
        continuous_behavior,
    )
    rewrite_rows, rewrite_acc = compute_rewrite_evaluation(
        model,
        model_name,
        hparams,
        tokenizer,
        all_requests[:step],
        eval_batch_size,
        continuous_behavior,
    )
    rephrase_rows, rephrase_acc = compute_rephrase_evaluation(
        model,
        model_name,
        hparams,
        tokenizer,
        all_requests[:step],
        eval_batch_size,
        continuous_behavior,
    )
    summary: Dict[str, Any] = {
        "locality_acc": locality_acc,
        "locality_count": len(locality_rows),
        "rewrite_acc": rewrite_acc,
        "rewrite_count": len(rewrite_rows),
        "rephrase_acc": rephrase_acc,
        "rephrase_count": len(rephrase_rows),
    }
    if continuous_behavior:
        all_rows = {
            "rewrite": rewrite_rows,
            "rephrase": rephrase_rows,
            "locality": locality_rows,
        }
        for field in CONTINUOUS_BEHAVIOR_SUMMARY_FIELDS:
            group = field.split("_", 1)[0]
            summary[field] = _finite_mean(
                row.get(field, float("nan")) for row in all_rows[group]
            )
        summary.update(
            {
                "rewrite_target_old_count": sum(
                    math.isfinite(
                        float(
                            row.get(
                                "rewrite_target_old_mean_logprob",
                                float("nan"),
                            )
                        )
                    )
                    for row in rewrite_rows
                ),
                "rephrase_target_old_count": sum(
                    math.isfinite(
                        float(
                            row.get(
                                "rephrase_target_old_mean_logprob",
                                float("nan"),
                            )
                        )
                    )
                    for row in rephrase_rows
                ),
                "locality_target_case_count": sum(
                    math.isfinite(
                        float(
                            row.get(
                                "locality_target_mean_logprob",
                                float("nan"),
                            )
                        )
                    )
                    for row in locality_rows
                ),
            }
        )
    return {
        "edit_count": step,
        "summary": summary,
        "locality": locality_rows,
        "rewrite": rewrite_rows,
        "rephrase": rephrase_rows,
    }


def index_sample_rows(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, int, int], Mapping[str, Any]]:
    return {
        (str(row["position"]), int(row["layer"]), int(row["prompt_index"])): row
        for row in rows
    }


def build_proxy_rows(
    label: str,
    step: int,
    positions: Sequence[str],
    layers: Sequence[int],
    probes: Sequence[Any],
    model: Any,
    base_features: Mapping[str, Mapping[Tuple[int, int, str], np.ndarray]],
    base_matrices: Mapping[Tuple[str, int], Sequence[np.ndarray]],
    state_matrices: Mapping[Tuple[str, int], Sequence[np.ndarray]],
    state_sample_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    forward_rows: List[MutableMapping[str, Any]] = []
    for position in positions:
        forward_rows.extend(
            compare_forward_features(
                label,
                model,
                probes,
                layers,
                position,
                base_features[position],
            )
        )
    sample_lookup = index_sample_rows(state_sample_rows)
    output: List[Dict[str, Any]] = []
    for row in forward_rows:
        position = str(row["position"])
        layer = int(row["layer"])
        prompt = int(row["prompt_index"])
        state_j = np.asarray(state_matrices[(position, layer)][prompt])
        base_j = np.asarray(base_matrices[(position, layer)][prompt])
        current_gain = float(sample_lookup[(position, layer, prompt)]["R_frobenius"])
        base_r = base_j - np.eye(base_j.shape[0], dtype=np.float64)
        base_gain = float(np.linalg.norm(base_r, ord="fro"))
        item = dict(row)
        item.update(jacobian_divergences(base_j, state_j))
        item.update(
            {
                "edit_count": step,
                "projected_residual_gain": current_gain,
                "base_projected_residual_gain": base_gain,
                "projected_residual_gain_ratio": current_gain / (base_gain + EPS),
                "projected_residual_gain_attenuation": math.log(
                    (base_gain + EPS) / (current_gain + EPS)
                ),
            }
        )
        output.append(item)
    return output


def base_proxy_rows(
    label: str,
    step: int,
    positions: Sequence[str],
    layers: Sequence[int],
    probes: Sequence[Any],
    base_features: Mapping[str, Mapping[Tuple[int, int, str], np.ndarray]],
    base_matrices: Mapping[Tuple[str, int], Sequence[np.ndarray]],
    base_sample_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    sample_lookup = index_sample_rows(base_sample_rows)
    output: List[Dict[str, Any]] = []
    for position in positions:
        for prompt, probe in enumerate(probes):
            for layer in layers:
                base_j = np.asarray(base_matrices[(position, layer)][prompt])
                gain = float(sample_lookup[(position, layer, prompt)]["R_frobenius"])
                base_hidden = np.asarray(
                    base_features[position][(prompt, layer, "hidden")],
                    dtype=np.float64,
                )
                base_hidden_rms = float(
                    np.sqrt(np.mean(base_hidden * base_hidden))
                )
                item: Dict[str, Any] = {
                    "state": label,
                    "edit_count": step,
                    "position": position,
                    "layer": layer,
                    "prompt_index": prompt,
                    "case_id": probe.case_id,
                    "token_position": probe.positions[position],
                    "projected_residual_gain": gain,
                    "base_projected_residual_gain": gain,
                    "projected_residual_gain_ratio": 1.0,
                    "projected_residual_gain_attenuation": 0.0,
                    "base_hidden_rms": base_hidden_rms,
                    "hidden_rms": base_hidden_rms,
                    "hidden_rms_ratio": 1.0,
                    "hidden_energy_ratio": 1.0,
                }
                item.update({proxy: 0.0 for proxy in ALL_PROXIES})
                item.update(jacobian_divergences(base_j, base_j))
                output.append(item)
    return output


def spectral_summary(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.maximum(np.asarray(values, dtype=np.float64).reshape(-1), 0.0)
    total = float(values.sum())
    if total > EPS:
        probabilities = values / total
        nonzero = probabilities[probabilities > 0]
        effective_rank = float(np.exp(-np.sum(nonzero * np.log(nonzero))))
        participation_ratio = float(total * total / (np.sum(values**2) + EPS))
        top_fraction = float(values.max() / total)
    else:
        effective_rank = 0.0
        participation_ratio = 0.0
        top_fraction = 0.0
    return {
        f"{prefix}_trace": total,
        f"{prefix}_effective_rank": effective_rank,
        f"{prefix}_participation_ratio": participation_ratio,
        f"{prefix}_top_eigenvalue_fraction": top_fraction,
    }


def covariance_rows_for_state(
    label: str,
    step: int,
    states: Mapping[str, Mapping[int, torch.Tensor]],
    positions: Sequence[str],
    layers: Sequence[int],
    bases: Mapping[str, Mapping[int, torch.Tensor]],
    base_rows: Mapping[Tuple[str, int], Mapping[str, Any]] | None,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, int, str], np.ndarray]]:
    rows: List[Dict[str, Any]] = []
    spectra: Dict[Tuple[str, int, str], np.ndarray] = {}
    for position in positions:
        for layer in layers:
            layer_states = states[position][layer].float()
            full_spectrum, full_metrics, _ = spectrum_metrics(layer_states)
            centered = layer_states - layer_states.mean(dim=0, keepdim=True)
            basis = bases[position][layer].float()
            projected = centered @ basis
            if projected.size(0) <= 1:
                projected_spectrum = np.zeros(projected.size(1), dtype=np.float64)
            else:
                singular = torch.linalg.svdvals(projected)
                projected_spectrum = (
                    singular.square() / max(projected.size(0) - 1, 1)
                ).double().cpu().numpy()
                if projected_spectrum.size < projected.size(1):
                    projected_spectrum = np.pad(
                        projected_spectrum,
                        (0, projected.size(1) - projected_spectrum.size),
                    )

            full_summary = spectral_summary(full_spectrum, "full_covariance")
            projected_summary = spectral_summary(
                projected_spectrum, "projected_covariance"
            )
            base_row = base_rows.get((position, layer)) if base_rows is not None else None
            if base_row is None:
                projected_mean = projected_summary["projected_covariance_trace"] / max(
                    len(projected_spectrum), 1
                )
                log_floor = max(projected_mean * 1e-8, EPS)
                full_nonzero_rank = max(min(layer_states.size(0) - 1, layer_states.size(1)), 1)
                full_mean = full_summary["full_covariance_trace"] / full_nonzero_rank
                full_log_floor = max(full_mean * 1e-8, EPS)
            else:
                log_floor = float(base_row["projected_covariance_log_floor"])
                full_log_floor = float(base_row["full_covariance_log_floor"])

            projected_logdet = float(
                np.sum(np.log(np.maximum(projected_spectrum, log_floor)))
            )
            full_nonzero = np.asarray(full_spectrum, dtype=np.float64)
            full_nonzero = full_nonzero[full_nonzero > full_log_floor]
            full_log_pseudodet = float(
                np.sum(np.log(np.maximum(full_nonzero, full_log_floor)))
            )
            row: Dict[str, Any] = {
                "state": label,
                "edit_count": step,
                "position": position,
                "layer": layer,
                "n_prompts": int(layer_states.size(0)),
                "hidden_size": int(layer_states.size(1)),
                "pca_rank": int(projected.size(1)),
                **full_summary,
                **projected_summary,
                "full_covariance_variance_per_hidden_dim": float(
                    full_metrics["variance_per_dimension"]
                ),
                "projected_covariance_variance_per_dimension": float(
                    projected_summary["projected_covariance_trace"]
                    / max(projected.size(1), 1)
                ),
                "full_covariance_log_floor": full_log_floor,
                "full_covariance_log_pseudodet": full_log_pseudodet,
                "projected_covariance_log_floor": log_floor,
                "projected_covariance_logdet": projected_logdet,
                "projected_covariance_logdet_per_dimension": projected_logdet
                / max(projected.size(1), 1),
            }
            if base_row is None:
                for name in (
                    "full_covariance_trace_ratio_to_base",
                    "full_covariance_effective_rank_ratio_to_base",
                    "projected_covariance_trace_ratio_to_base",
                    "projected_covariance_effective_rank_ratio_to_base",
                    "projected_covariance_geometric_variance_ratio_to_base",
                ):
                    row[name] = 1.0
            else:
                row.update(
                    {
                        "full_covariance_trace_ratio_to_base": row["full_covariance_trace"]
                        / (float(base_row["full_covariance_trace"]) + EPS),
                        "full_covariance_effective_rank_ratio_to_base": row[
                            "full_covariance_effective_rank"
                        ]
                        / (float(base_row["full_covariance_effective_rank"]) + EPS),
                        "projected_covariance_trace_ratio_to_base": row[
                            "projected_covariance_trace"
                        ]
                        / (float(base_row["projected_covariance_trace"]) + EPS),
                        "projected_covariance_effective_rank_ratio_to_base": row[
                            "projected_covariance_effective_rank"
                        ]
                        / (
                            float(base_row["projected_covariance_effective_rank"])
                            + EPS
                        ),
                        "projected_covariance_geometric_variance_ratio_to_base": math.exp(
                            (
                                projected_logdet
                                - float(base_row["projected_covariance_logdet"])
                            )
                            / max(projected.size(1), 1)
                        ),
                    }
                )
            rows.append(row)
            spectra[(position, layer, "full")] = np.asarray(full_spectrum)
            spectra[(position, layer, "projected")] = np.asarray(projected_spectrum)
    return rows, spectra


def load_base_covariance_rows(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    return {
        (str(row["position"]), int(row["layer"])): row
        for row in read_csv_rows(path)
    }


def analyze_group(
    step: int,
    label: str,
    group: str,
    positions: Sequence[str],
    model: Any,
    probes: Sequence[Any],
    layers: Sequence[int],
    bases: Mapping[str, Mapping[int, torch.Tensor]],
    base_features: Mapping[str, Mapping[Tuple[int, int, str], np.ndarray]],
    base_matrices: Mapping[Tuple[str, int], Sequence[np.ndarray]] | None,
    interaction_analysis: bool,
    interaction_layers: Sequence[int],
    base_downproj_features: Mapping[Tuple[str, int, str], np.ndarray],
    base_downproj_weights: Mapping[int, torch.Tensor],
    branch_analysis: bool,
    covariance_analysis: bool,
    base_covariance_rows: Mapping[Tuple[str, int], Mapping[str, Any]] | None,
    jacobian_prompts: int,
    group_output_dir: Path,
) -> Tuple[
    Dict[Tuple[str, int], List[np.ndarray]],
    Dict[Tuple[str, int], Dict[str, Any]],
]:
    group_output_dir.mkdir(parents=True, exist_ok=True)
    covariance_lookup: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if branch_analysis:
        (
            branch_prompt_rows,
            branch_summary_rows,
            _branch_inputs,
            _branch_sequence_rms,
            branch_write_vectors,
        ) = collect_forward_writes(model, probes, layers, positions)
        for row in branch_prompt_rows:
            row["state"] = label
            row["edit_count"] = step
        for row in branch_summary_rows:
            row["state"] = label
            row["edit_count"] = step
        write_csv(
            group_output_dir / "branch_prompt_metrics.csv",
            branch_prompt_rows,
        )
        write_csv(
            group_output_dir / "branch_summary_metrics.csv",
            branch_summary_rows,
        )
        save_write_vectors(
            group_output_dir / "branch_write_vectors.npz",
            branch_write_vectors,
        )
        del _branch_inputs, _branch_sequence_rms, branch_write_vectors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if interaction_analysis:
        current_downproj_features = collect_downproj_features(
            model,
            probes,
            interaction_layers,
            positions,
        )
        interaction_prompt_rows, interaction_summary_rows = (
            decompose_feature_update(
                model=model,
                positions=positions,
                layers=interaction_layers,
                base_features=base_downproj_features,
                current_features=current_downproj_features,
                base_weights=base_downproj_weights,
                state=label,
                edit_count=step,
            )
        )
        write_csv(
            group_output_dir / "feature_update_prompt_metrics.csv",
            interaction_prompt_rows,
        )
        write_csv(
            group_output_dir / "feature_update_summary_metrics.csv",
            interaction_summary_rows,
        )
        save_downproj_features(
            group_output_dir / "downproj_features.npz",
            current_downproj_features,
        )
        save_downproj_feature_deltas(
            group_output_dir / "downproj_feature_deltas.npz",
            current_downproj_features,
            base_downproj_features,
        )
        del current_downproj_features
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if covariance_analysis:
        residual_states = collect_residual_states(
            model, probes, layers, positions
        )
        covariance_rows, covariance_spectra = covariance_rows_for_state(
            label,
            step,
            residual_states,
            positions,
            layers,
            bases,
            base_covariance_rows,
        )
        write_csv(group_output_dir / "covariance_metrics.csv", covariance_rows)
        np.savez(
            group_output_dir / "covariance_spectra.npz",
            **{
                safe_key(label, position, f"L{layer}", kind): values
                for (position, layer, kind), values in covariance_spectra.items()
            },
        )
        covariance_lookup = {
            (str(row["position"]), int(row["layer"])): row
            for row in covariance_rows
        }
        del residual_states
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sample_rows, matrices = analyze_projected_jacobians(
        model,
        probes,
        layers,
        positions,
        bases,
        jacobian_prompts,
    )
    for row in sample_rows:
        row["state"] = label
        row["edit_count"] = step
    summary_rows, _ = aggregate_jacobians(
        label, positions, layers, sample_rows, matrices
    )
    for row in summary_rows:
        row["edit_count"] = step

    group_output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(group_output_dir / "jacobian_sample_metrics.csv", sample_rows)
    write_csv(group_output_dir / "jacobian_summary_metrics.csv", summary_rows)
    save_projected_matrices(
        group_output_dir / "projected_jacobians.npz", label, matrices
    )

    if step == 0:
        proxy_rows = base_proxy_rows(
            label,
            step,
            positions,
            layers,
            probes[:jacobian_prompts],
            base_features,
            matrices,
            sample_rows,
        )
    else:
        if base_matrices is None:
            raise RuntimeError("Edited-state analysis requires cached Base projected Jacobians")
        proxy_rows = build_proxy_rows(
            label,
            step,
            positions,
            layers,
            probes[:jacobian_prompts],
            model,
            base_features,
            base_matrices,
            matrices,
            sample_rows,
        )
    write_csv(group_output_dir / "per_layer_proxy_metrics.csv", proxy_rows)
    return (
        {key: list(value) for key, value in matrices.items()},
        covariance_lookup,
    )


def mean_finite(values: Iterable[Any]) -> float:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def completed_steps(output_dir: Path, steps: Sequence[int], fingerprint: str) -> List[int]:
    return [step for step in steps if step_is_complete(output_dir, step, fingerprint)]


def aggregate_covariance_outputs(
    output_dir: Path,
    steps: Sequence[int],
    fingerprint: str,
    layout: Mapping[str, Sequence[str]],
    window_layers: Sequence[int],
    editing_method: str,
) -> None:
    evaluation_by_step: Dict[int, Mapping[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    for step in completed_steps(output_dir, steps, fingerprint):
        evaluation_path = step_dir(output_dir, step) / "evaluation.json"
        if evaluation_path.exists():
            evaluation_by_step[step] = read_json(evaluation_path)["summary"]
        for group in layout:
            covariance_path = step_dir(output_dir, step) / group / "covariance_metrics.csv"
            if not covariance_path.exists():
                continue
            for row in read_csv_rows(covariance_path):
                row["context"] = group
                rows.append(row)
    if not rows:
        return
    write_csv(output_dir / "trajectory_covariance_layer_metrics.csv", rows)

    window_set = set(window_layers)
    grouped: Dict[Tuple[int, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["layer"]) in window_set:
            grouped[
                (int(row["edit_count"]), str(row["context"]), str(row["position"]))
            ].append(row)
    metrics = [
        "full_covariance_trace_ratio_to_base",
        "full_covariance_effective_rank_ratio_to_base",
        "projected_covariance_trace_ratio_to_base",
        "projected_covariance_effective_rank_ratio_to_base",
        "projected_covariance_geometric_variance_ratio_to_base",
        "full_covariance_trace",
        "full_covariance_effective_rank",
        "projected_covariance_trace",
        "projected_covariance_effective_rank",
        "projected_covariance_logdet_per_dimension",
    ]
    summary_rows: List[Dict[str, Any]] = []
    for (step, context, position), selected in sorted(grouped.items()):
        evaluation = evaluation_by_step.get(step, {})
        item: Dict[str, Any] = {
            "editing_method": editing_method,
            "edit_count": step,
            "context": context,
            "position": position,
            "window_layers": ",".join(str(layer) for layer in window_layers),
            "n_layers": len(selected),
            "locality_acc": evaluation.get("locality_acc", float("nan")),
            "locality_loss": 1.0
            - float(evaluation.get("locality_acc", float("nan"))),
            "rewrite_acc": evaluation.get("rewrite_acc", float("nan")),
            "rephrase_acc": evaluation.get("rephrase_acc", float("nan")),
        }
        for metric in metrics:
            item[metric] = mean_finite(
                row.get(metric, float("nan")) for row in selected
            )
        summary_rows.append(item)
    write_csv(output_dir / "trajectory_covariance_window_summary.csv", summary_rows)

    correlation_specs = [
        ("full_covariance_trace_ratio_to_base", "locality_acc"),
        ("full_covariance_effective_rank_ratio_to_base", "locality_acc"),
        ("projected_covariance_trace_ratio_to_base", "locality_acc"),
        ("projected_covariance_effective_rank_ratio_to_base", "locality_acc"),
        ("projected_covariance_geometric_variance_ratio_to_base", "locality_acc"),
    ]
    correlation_rows: List[Dict[str, Any]] = []
    pairs = sorted({(row["context"], row["position"]) for row in summary_rows})
    for context, position in pairs:
        selected = [
            row
            for row in summary_rows
            if row["context"] == context
            and row["position"] == position
            and int(row["edit_count"]) > 0
        ]
        for metric, outcome in correlation_specs:
            valid = [
                row
                for row in selected
                if math.isfinite(float(row[metric]))
                and math.isfinite(float(row[outcome]))
            ]
            x = [float(row[metric]) for row in valid]
            y = [float(row[outcome]) for row in valid]
            correlation_rows.append(
                {
                    "editing_method": editing_method,
                    "context": context,
                    "position": position,
                    "proxy": metric,
                    "outcome": outcome,
                    "n_checkpoints": len(valid),
                    "pearson": pearson(x, y),
                    "spearman": spearman(x, y),
                }
            )
    write_csv(output_dir / "trajectory_covariance_correlations.csv", correlation_rows)
    plot_covariance_trajectory(
        output_dir,
        summary_rows,
        editing_method,
        window_layers,
    )


def plot_covariance_trajectory(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    editing_method: str,
    window_layers: Sequence[int],
) -> None:
    pairs = sorted({(str(row["context"]), str(row["position"])) for row in rows})
    if not pairs:
        return
    specs = [
        ("full_covariance_trace_ratio_to_base", "full covariance trace / Base"),
        ("projected_covariance_trace_ratio_to_base", "Base-PCA trace / Base"),
        (
            "projected_covariance_geometric_variance_ratio_to_base",
            "Base-PCA geometric variance / Base",
        ),
        (
            "projected_covariance_effective_rank_ratio_to_base",
            "Base-PCA effective rank / Base",
        ),
    ]
    fig, axes = plt.subplots(
        len(pairs), len(specs), figsize=(5.0 * len(specs), 4.2 * len(pairs)), squeeze=False
    )
    for row_index, (context, position) in enumerate(pairs):
        selected = sorted(
            [
                row
                for row in rows
                if row["context"] == context and row["position"] == position
            ],
            key=lambda row: int(row["edit_count"]),
        )
        for column, (metric, title) in enumerate(specs):
            axis = axes[row_index, column]
            axis.plot(
                [int(row["edit_count"]) for row in selected],
                [float(row[metric]) for row in selected],
                marker="o",
                color="#6F2DBD",
            )
            axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
            axis.set_title(f"{context}/{position}\n{title}")
            axis.set_xlabel("cumulative edits")
            axis.grid(alpha=0.24)
    layer_label = (
        f"L{min(window_layers)}–L{max(window_layers)}"
        if window_layers
        else "configured window"
    )
    fig.suptitle(
        f"{editing_method} residual covariance trajectory ({layer_label} mean)",
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "trajectory_covariance_information.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def aggregate_branch_outputs(
    output_dir: Path,
    steps: Sequence[int],
    fingerprint: str,
    layout: Mapping[str, Sequence[str]],
    layers: Sequence[int],
) -> bool:
    rows: List[Dict[str, Any]] = []
    completed = completed_steps(output_dir, steps, fingerprint)
    for step in completed:
        for group in layout:
            path = step_dir(output_dir, step) / group / "branch_summary_metrics.csv"
            if not path.exists():
                continue
            for row in read_csv_rows(path):
                row["edit_count"] = step
                row["context"] = group
                rows.append(row)
    if not rows:
        return False

    base_lookup = {
        (
            str(row["context"]),
            str(row["position"]),
            int(row["layer"]),
            str(row["write_kind"]),
        ): row
        for row in rows
        if int(row["edit_count"]) == 0
    }
    ratio_metrics = (
        "rms_l2",
        "centroid_l2",
        "centered_trace_population",
        "covariance_effective_rank",
        "covariance_participation_ratio",
        "direction_concentration",
        "write_variability_fraction",
        "constant_component_fraction",
    )
    for row in rows:
        reference = base_lookup.get(
            (
                str(row["context"]),
                str(row["position"]),
                int(row["layer"]),
                str(row["write_kind"]),
            )
        )
        for metric in ratio_metrics:
            row[f"{metric}_ratio_to_base"] = (
                float(row[metric]) / (float(reference[metric]) + EPS)
                if reference is not None
                else float("nan")
            )
    write_csv(output_dir / "trajectory_branch_summary.csv", rows)

    delta_rows: List[Dict[str, Any]] = []
    for group, positions in layout.items():
        base_path = step_dir(output_dir, 0) / group / "branch_write_vectors.npz"
        if not base_path.exists():
            continue
        base_vectors = load_write_vectors(base_path, positions, layers)
        for step in completed:
            current_path = (
                step_dir(output_dir, step) / group / "branch_write_vectors.npz"
            )
            if not current_path.exists():
                continue
            current_vectors = load_write_vectors(current_path, positions, layers)
            for row in summarize_delta_writes(current_vectors, base_vectors):
                row["edit_count"] = step
                row["context"] = group
                delta_rows.append(row)
            del current_vectors
        del base_vectors
    if delta_rows:
        write_csv(
            output_dir / "trajectory_delta_branch_summary.csv",
            delta_rows,
        )
    return True


def aggregate_interaction_outputs(
    output_dir: Path,
    steps: Sequence[int],
    fingerprint: str,
    layout: Mapping[str, Sequence[str]],
) -> None:
    summary_rows: List[Dict[str, Any]] = []
    prompt_rows: List[Dict[str, Any]] = []
    for step in completed_steps(output_dir, steps, fingerprint):
        for group in layout:
            summary_path = (
                step_dir(output_dir, step)
                / group
                / "feature_update_summary_metrics.csv"
            )
            prompt_path = (
                step_dir(output_dir, step)
                / group
                / "feature_update_prompt_metrics.csv"
            )
            if summary_path.exists():
                for row in read_csv_rows(summary_path):
                    row["edit_count"] = step
                    row["context"] = group
                    summary_rows.append(row)
            if prompt_path.exists():
                for row in read_csv_rows(prompt_path):
                    row["edit_count"] = step
                    row["context"] = group
                    prompt_rows.append(row)
    if not summary_rows:
        return

    observed = {
        (
            int(row["edit_count"]),
            str(row["context"]),
            str(row["position"]),
            int(row["layer"]),
        ): float(row["mean_squared_l2"])
        for row in summary_rows
        if str(row["component"]) == "observed_total"
    }
    interaction_totals: Dict[Tuple[int, str, str], float] = defaultdict(float)
    for row in summary_rows:
        if str(row["component"]) == "interaction":
            interaction_totals[
                (
                    int(row["edit_count"]),
                    str(row["context"]),
                    str(row["position"]),
                )
            ] += float(row["mean_squared_l2"])

    for row in summary_rows:
        identity = (
            int(row["edit_count"]),
            str(row["context"]),
            str(row["position"]),
            int(row["layer"]),
        )
        component_energy = float(row["mean_squared_l2"])
        row["component_energy_fraction_of_observed"] = (
            component_energy / (observed.get(identity, 0.0) + EPS)
        )
        if str(row["component"]) == "interaction":
            total = interaction_totals[
                (
                    int(row["edit_count"]),
                    str(row["context"]),
                    str(row["position"]),
                )
            ]
            row["interaction_energy_fraction_across_edited_layers"] = (
                component_energy / (total + EPS)
            )
        else:
            row["interaction_energy_fraction_across_edited_layers"] = float(
                "nan"
            )

    write_csv(
        output_dir / "trajectory_feature_update_summary.csv",
        summary_rows,
    )
    if prompt_rows:
        write_csv(
            output_dir / "trajectory_feature_update_prompt_metrics.csv",
            prompt_rows,
        )


def aggregate_outputs(
    output_dir: Path,
    steps: Sequence[int],
    fingerprint: str,
    layout: Mapping[str, Sequence[str]],
    window_layers: Sequence[int],
    editing_method: str,
    covariance_analysis: bool = False,
) -> None:
    layer_rows: List[Dict[str, Any]] = []
    evaluation_by_step: Dict[int, Mapping[str, Any]] = {}
    prompt_locality_by_step: Dict[int, Dict[str, float]] = {}
    for step in completed_steps(output_dir, steps, fingerprint):
        evaluation_path = step_dir(output_dir, step) / "evaluation.json"
        if evaluation_path.exists():
            evaluation_payload = read_json(evaluation_path)
            evaluation_by_step[step] = evaluation_payload["summary"]
            prompt_locality_by_step[step] = {
                str(row["case_id"]): float(row["locality_acc"])
                for row in evaluation_payload.get("locality", [])
            }
        for group in layout:
            path = step_dir(output_dir, step) / group / "per_layer_proxy_metrics.csv"
            if not path.exists():
                continue
            for row in read_csv_rows(path):
                row["context"] = group
                case_locality = prompt_locality_by_step.get(step, {}).get(str(row.get("case_id")))
                row["prompt_locality_acc"] = (
                    case_locality if group == "locality" and case_locality is not None else float("nan")
                )
                row["prompt_locality_loss"] = (
                    1.0 - case_locality
                    if group == "locality" and case_locality is not None
                    else float("nan")
                )
                layer_rows.append(row)
    if not layer_rows:
        return

    write_csv(output_dir / "trajectory_per_layer_prompt_metrics.csv", layer_rows)
    grouped: Dict[Tuple[int, str, str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in layer_rows:
        grouped[
            (
                int(row["edit_count"]),
                str(row["context"]),
                str(row["position"]),
                int(row["layer"]),
            )
        ].append(row)

    layer_summary: List[Dict[str, Any]] = []
    metric_names = [
        "projected_residual_gain",
        "projected_residual_gain_ratio",
        "projected_residual_gain_attenuation",
        "base_hidden_rms",
        "hidden_rms",
        "hidden_rms_ratio",
        "hidden_energy_ratio",
        *ALL_PROXIES,
    ]
    for (step, context, position, layer), rows in sorted(grouped.items()):
        evaluation = evaluation_by_step.get(step, {})
        item: Dict[str, Any] = {
            "edit_count": step,
            "context": context,
            "position": position,
            "layer": layer,
            "n_prompts": len(rows),
            "locality_acc": evaluation.get("locality_acc", float("nan")),
            "rewrite_acc": evaluation.get("rewrite_acc", float("nan")),
            "rephrase_acc": evaluation.get("rephrase_acc", float("nan")),
        }
        for metric in metric_names:
            item[metric] = mean_finite(row.get(metric, float("nan")) for row in rows)
        layer_summary.append(item)
    write_csv(output_dir / "trajectory_layer_summary.csv", layer_summary)

    window_set = set(window_layers)
    window_groups: Dict[Tuple[int, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in layer_rows:
        if int(row["layer"]) in window_set:
            window_groups[
                (int(row["edit_count"]), str(row["context"]), str(row["position"]))
            ].append(row)
    window_summary: List[Dict[str, Any]] = []
    for (step, context, position), rows in sorted(window_groups.items()):
        evaluation = evaluation_by_step.get(step, {})
        item = {
            "edit_count": step,
            "context": context,
            "position": position,
            "window_layers": ",".join(str(layer) for layer in window_layers),
            "n_prompt_layers": len(rows),
            "locality_acc": evaluation.get("locality_acc", float("nan")),
            "locality_loss": 1.0 - float(evaluation.get("locality_acc", float("nan"))),
            "rewrite_acc": evaluation.get("rewrite_acc", float("nan")),
            "rephrase_acc": evaluation.get("rephrase_acc", float("nan")),
        }
        for metric in metric_names:
            item[metric] = mean_finite(row.get(metric, float("nan")) for row in rows)
        window_summary.append(item)
    write_csv(output_dir / "trajectory_window_summary.csv", window_summary)
    write_csv(
        output_dir / "trajectory_checkpoint_outcomes.csv",
        [
            {
                "edit_count": step,
                "rewrite_acc": evaluation_by_step[step].get(
                    "rewrite_acc", float("nan")
                ),
                "rewrite_count": evaluation_by_step[step].get(
                    "rewrite_count", 0
                ),
                "rephrase_acc": evaluation_by_step[step].get(
                    "rephrase_acc", float("nan")
                ),
                "rephrase_count": evaluation_by_step[step].get(
                    "rephrase_count", 0
                ),
                "locality_acc": evaluation_by_step[step].get(
                    "locality_acc", float("nan")
                ),
                "locality_loss": (
                    1.0
                    - float(
                        evaluation_by_step[step].get(
                            "locality_acc", float("nan")
                        )
                    )
                ),
                "locality_count": evaluation_by_step[step].get(
                    "locality_count", 0
                ),
                **{
                    field: evaluation_by_step[step][field]
                    for field in (
                        *CONTINUOUS_BEHAVIOR_SUMMARY_FIELDS,
                        *CONTINUOUS_BEHAVIOR_COUNT_FIELDS,
                    )
                    if field in evaluation_by_step[step]
                },
            }
            for step in sorted(evaluation_by_step)
        ],
    )

    prompt_groups: Dict[Tuple[int, str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in layer_rows:
        if int(row["layer"]) in window_set and math.isfinite(float(row["prompt_locality_acc"])):
            prompt_groups[
                (
                    int(row["edit_count"]),
                    str(row["context"]),
                    str(row["position"]),
                    str(row["case_id"]),
                )
            ].append(row)
    prompt_window_rows: List[Dict[str, Any]] = []
    for (step, context, position, case_id), rows in sorted(prompt_groups.items()):
        item: Dict[str, Any] = {
            "edit_count": step,
            "context": context,
            "position": position,
            "case_id": case_id,
            "window_layers": ",".join(str(layer) for layer in window_layers),
            "n_layers": len(rows),
            "locality_acc": float(rows[0]["prompt_locality_acc"]),
            "locality_loss": float(rows[0]["prompt_locality_loss"]),
        }
        for metric in metric_names:
            item[metric] = mean_finite(row.get(metric, float("nan")) for row in rows)
        prompt_window_rows.append(item)
    write_csv(output_dir / "trajectory_per_prompt_window_metrics.csv", prompt_window_rows)

    correlation_rows: List[Dict[str, Any]] = []
    correlation_specs = [
        ("projected_residual_gain", "locality_acc"),
        ("projected_residual_gain_ratio", "locality_acc"),
        ("projected_residual_gain_attenuation", "locality_loss"),
        ("hidden_rms", "locality_acc"),
        ("hidden_rms_ratio", "locality_acc"),
        ("jacobian_rhat_delta_frobenius", "locality_loss"),
        ("jacobian_eig_contraction", "locality_loss"),
        ("hidden_delta_relative_l2", "locality_loss"),
        ("attention_symmetric_kl", "locality_loss"),
    ]
    contexts = sorted({(row["context"], row["position"]) for row in window_summary})
    for context, position in contexts:
        selected = [
            row
            for row in window_summary
            if row["context"] == context and row["position"] == position and int(row["edit_count"]) > 0
        ]
        for metric, outcome in correlation_specs:
            valid = [
                row
                for row in selected
                if math.isfinite(float(row[metric])) and math.isfinite(float(row[outcome]))
            ]
            x = [float(row[metric]) for row in valid]
            y = [float(row[outcome]) for row in valid]
            correlation_rows.append(
                {
                    "context": context,
                    "position": position,
                    "window_layers": ",".join(str(layer) for layer in window_layers),
                    "proxy": metric,
                    "outcome": outcome,
                    "n_checkpoints": len(valid),
                    "pearson": pearson(x, y),
                    "spearman": spearman(x, y),
                }
            )
    write_csv(output_dir / "trajectory_proxy_correlations.csv", correlation_rows)

    prompt_correlation_rows: List[Dict[str, Any]] = []
    prompt_specs = [
        ("projected_residual_gain", "locality_acc"),
        ("projected_residual_gain_ratio", "locality_acc"),
        ("projected_residual_gain_attenuation", "locality_loss"),
        ("hidden_rms", "locality_acc"),
        ("hidden_rms_ratio", "locality_acc"),
        ("jacobian_rhat_delta_frobenius", "locality_loss"),
        ("jacobian_eig_contraction", "locality_loss"),
        ("hidden_delta_relative_l2", "locality_loss"),
        ("attention_symmetric_kl", "locality_loss"),
    ]
    edited_prompt_rows = [row for row in prompt_window_rows if int(row["edit_count"]) > 0]
    for step in sorted({int(row["edit_count"]) for row in edited_prompt_rows}):
        selected = [row for row in edited_prompt_rows if int(row["edit_count"]) == step]
        for metric, outcome in prompt_specs:
            x = [float(row[metric]) for row in selected]
            y = [float(row[outcome]) for row in selected]
            prompt_correlation_rows.append(
                {
                    "analysis": "within_checkpoint",
                    "edit_count": step,
                    "proxy": metric,
                    "outcome": outcome,
                    "n_prompts": len(selected),
                    "pearson": pearson(x, y),
                    "spearman": spearman(x, y),
                }
            )
    for metric, outcome in prompt_specs:
        pooled_x = [float(row[metric]) for row in edited_prompt_rows]
        pooled_y = [float(row[outcome]) for row in edited_prompt_rows]
        demeaned_x: List[float] = []
        demeaned_y: List[float] = []
        for step in sorted({int(row["edit_count"]) for row in edited_prompt_rows}):
            selected = [row for row in edited_prompt_rows if int(row["edit_count"]) == step]
            state_x = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            state_y = np.asarray([float(row[outcome]) for row in selected], dtype=np.float64)
            demeaned_x.extend((state_x - state_x.mean()).tolist())
            demeaned_y.extend((state_y - state_y.mean()).tolist())
        for analysis, x, y in (
            ("pooled", pooled_x, pooled_y),
            ("checkpoint_demeaned", demeaned_x, demeaned_y),
        ):
            prompt_correlation_rows.append(
                {
                    "analysis": analysis,
                    "edit_count": "all",
                    "proxy": metric,
                    "outcome": outcome,
                    "n_prompts": len(x),
                    "pearson": pearson(x, y),
                    "spearman": spearman(x, y),
                }
            )
    write_csv(output_dir / "trajectory_prompt_proxy_correlations.csv", prompt_correlation_rows)
    plot_trajectory(output_dir, layer_summary, window_summary, editing_method)
    branch_summary_ready = aggregate_branch_outputs(
        output_dir,
        steps,
        fingerprint,
        layout,
        sorted({int(row["layer"]) for row in layer_rows}),
    )
    if branch_summary_ready:
        plot_f_trajectory(output_dir, editing_method)
        plot_unified_branch_rms_trajectory(
            output_dir,
            editing_method=editing_method,
        )
    aggregate_interaction_outputs(
        output_dir,
        steps,
        fingerprint,
        layout,
    )
    if covariance_analysis:
        aggregate_covariance_outputs(
            output_dir,
            steps,
            fingerprint,
            layout,
            window_layers,
            editing_method,
        )


def plot_trajectory(
    output_dir: Path,
    layer_rows: Sequence[Mapping[str, Any]],
    window_rows: Sequence[Mapping[str, Any]],
    editing_method: str,
) -> None:
    panel_specs = [
        ("edit", "subject_last", "Edit prompts — subject last"),
        ("locality", "prompt_last", "Locality prompts — prompt last"),
    ]
    available = {
        (str(row["context"]), str(row["position"])) for row in layer_rows
    }
    if any((context, position) not in available for context, position, _ in panel_specs):
        print(
            "[plot] skipped canonical three-panel trajectory: "
            "edit/subject_last and locality/prompt_last are both required"
        )
        return
    completed = sorted({int(row["edit_count"]) for row in layer_rows})
    color_map = plt.get_cmap("viridis")
    color_by_step = {
        step: color_map(index / max(len(completed) - 1, 1))
        for index, step in enumerate(completed)
    }

    locality_rows = sorted(
        [
            row
            for row in window_rows
            if row["context"] == "locality"
            and row["position"] == "prompt_last"
        ],
        key=lambda row: int(row["edit_count"]),
    )

    def plot_layer_profiles(
        axes: Sequence[Any],
        metric: str,
        ylabel: str,
        *,
        base_line: bool,
    ) -> None:
        for axis, (context, position, title) in zip(axes[:2], panel_specs):
            selected_layers = [
                row
                for row in layer_rows
                if row["context"] == context and row["position"] == position
            ]
            for step in completed:
                current = sorted(
                    [
                        row
                        for row in selected_layers
                        if int(row["edit_count"]) == step
                    ],
                    key=lambda row: int(row["layer"]),
                )
                if not current:
                    continue
                axis.plot(
                    [int(row["layer"]) for row in current],
                    [float(row[metric]) for row in current],
                    marker="o",
                    color=color_by_step[step],
                    label=f"edit {step}",
                )
            if base_line:
                axis.axhline(
                    1.0,
                    color="#666666",
                    linestyle="--",
                    linewidth=1,
                )
            axis.set_title(title)
            axis.set_xlabel("decoder layer")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)

        axes[2].plot(
            [int(row["edit_count"]) for row in locality_rows],
            [float(row["locality_acc"]) for row in locality_rows],
            color="#D81B60",
            marker="o",
            linewidth=2,
        )
        for row in locality_rows:
            axes[2].annotate(
                str(row["edit_count"]),
                (int(row["edit_count"]), float(row["locality_acc"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axes[2].set_title("Locality preservation by edit count")
        axes[2].set_xlabel("cumulative edits")
        axes[2].set_ylabel("Base-output token agreement")
        axes[2].grid(alpha=0.25)

    def gain_break_limits() -> Tuple[float, float, float, float] | None:
        """Return shared lower/upper y ranges when a real empty gap exists."""
        panel_pairs = {
            (context, position) for context, position, _ in panel_specs
        }
        values = np.asarray(
            [
                float(row["projected_residual_gain"])
                for row in layer_rows
                if (str(row["context"]), str(row["position"])) in panel_pairs
                and math.isfinite(float(row["projected_residual_gain"]))
            ],
            dtype=np.float64,
        )
        if values.size < 20:
            return None
        lower_min = min(0.0, float(values.min()))
        lower_max = max(1.5, 1.10 * float(np.quantile(values, 0.975)))
        upper_candidates = values[values > 1.40 * lower_max]
        if upper_candidates.size == 0 or float(values.max()) <= 2.0 * lower_max:
            return None
        upper_min = 0.95 * float(upper_candidates.min())
        if upper_min <= 1.15 * lower_max:
            return None
        upper_max = 1.04 * float(values.max())
        return lower_min, lower_max, upper_min, upper_max

    def rms_break_limits() -> Tuple[float, float, float, float] | None:
        """Use a linear early-flow panel and a log-scale explosion panel."""
        panel_pairs = {
            (context, position) for context, position, _ in panel_specs
        }
        selected = [
            row
            for row in layer_rows
            if (str(row["context"]), str(row["position"])) in panel_pairs
            and math.isfinite(float(row["hidden_rms_ratio"]))
            and float(row["hidden_rms_ratio"]) > 0.0
        ]
        if len(selected) < 20:
            return None
        values = np.asarray(
            [float(row["hidden_rms_ratio"]) for row in selected],
            dtype=np.float64,
        )
        early_values = np.asarray(
            [
                float(row["hidden_rms_ratio"])
                for row in selected
                if int(row["edit_count"]) <= 200
            ],
            dtype=np.float64,
        )
        if early_values.size == 0:
            return None
        lower_min = max(0.0, 0.95 * float(early_values.min()))
        lower_max = 1.10 * float(early_values.max())
        if float(values.max()) <= 100.0 * lower_max:
            return None
        return lower_min, lower_max, lower_max, 1.10 * float(values.max())

    def plot_broken_profile_trajectory(
        limits: Tuple[float, float, float, float],
        *,
        metric: str,
        ylabel: str,
        figure_title: str,
        output_name: str,
        upper_log: bool = False,
        base_line: bool = False,
    ) -> None:
        lower_min, lower_max, upper_min, upper_max = limits
        fig = plt.figure(figsize=(20.0, 6.2))
        outer = fig.add_gridspec(
            1,
            3,
            width_ratios=[1.0, 1.0, 1.0],
            wspace=0.26,
        )
        broken_axes: List[Tuple[Any, Any]] = []
        for column in range(2):
            nested = outer[column].subgridspec(
                2,
                1,
                height_ratios=[1.0, 3.2],
                hspace=0.045,
            )
            upper = fig.add_subplot(nested[0])
            lower = fig.add_subplot(nested[1], sharex=upper)
            broken_axes.append((upper, lower))
        locality_axis = fig.add_subplot(outer[2])

        for (upper, lower), (context, position, title) in zip(
            broken_axes, panel_specs
        ):
            selected_layers = [
                row
                for row in layer_rows
                if row["context"] == context and row["position"] == position
            ]
            for step in completed:
                current = sorted(
                    [
                        row
                        for row in selected_layers
                        if int(row["edit_count"]) == step
                    ],
                    key=lambda row: int(row["layer"]),
                )
                if not current:
                    continue
                x_values = [int(row["layer"]) for row in current]
                y_values = [
                    float(row[metric]) for row in current
                ]
                for axis in (upper, lower):
                    axis.plot(
                        x_values,
                        y_values,
                        marker="o",
                        color=color_by_step[step],
                        label=f"edit {step}",
                    )

            if upper_log:
                upper.set_yscale("log")
            upper.set_ylim(upper_min, upper_max)
            lower.set_ylim(lower_min, lower_max)
            upper.spines["bottom"].set_visible(False)
            lower.spines["top"].set_visible(False)
            upper.tick_params(
                axis="x", which="both", bottom=False, labelbottom=False
            )
            lower.xaxis.tick_bottom()
            upper.set_title(title)
            lower.set_xlabel("decoder layer")
            lower.set_ylabel(ylabel)
            if base_line:
                lower.axhline(1.0, color="#666666", linestyle="--", linewidth=1)
            upper.grid(alpha=0.25)
            lower.grid(alpha=0.25)

            marker_kwargs = {
                "marker": [(-1, -0.55), (1, 0.55)],
                "markersize": 8,
                "linestyle": "none",
                "color": "k",
                "mec": "k",
                "mew": 1.0,
                "clip_on": False,
            }
            upper.plot(
                [0, 1], [0, 0], transform=upper.transAxes, **marker_kwargs
            )
            lower.plot(
                [0, 1], [1, 1], transform=lower.transAxes, **marker_kwargs
            )

        locality_axis.plot(
            [int(row["edit_count"]) for row in locality_rows],
            [float(row["locality_acc"]) for row in locality_rows],
            color="#D81B60",
            marker="o",
            linewidth=2,
        )
        for row in locality_rows:
            locality_axis.annotate(
                str(row["edit_count"]),
                (int(row["edit_count"]), float(row["locality_acc"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        locality_axis.set_title("Locality preservation by edit count")
        locality_axis.set_xlabel("cumulative edits")
        locality_axis.set_ylabel("Base-output token agreement")
        locality_axis.grid(alpha=0.25)

        handles, labels = broken_axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.015),
            ncol=min(len(labels), 9),
            frameon=False,
        )
        fig.suptitle(
            figure_title,
            fontweight="bold",
            y=1.075,
        )
        fig.savefig(
            output_dir / output_name,
            dpi=190,
            bbox_inches="tight",
        )
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(20.0, 5.0), squeeze=False)
    flat_axes = axes[0]
    plot_layer_profiles(
        flat_axes,
        "projected_residual_gain",
        r"mean projected residual gain $\|\widehat R\|_F$",
        base_line=False,
    )
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=min(len(labels), 9),
        frameon=False,
    )
    fig.suptitle(
        f"{editing_method} edit-count gain and locality trajectory",
        fontweight="bold",
        y=1.11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    break_limits = gain_break_limits()
    fig.savefig(
        output_dir
        / (
            "trajectory_gain_locality_linear.png"
            if break_limits is not None
            else "trajectory_gain_locality.png"
        ),
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)
    if break_limits is not None:
        plot_broken_profile_trajectory(
            break_limits,
            metric="projected_residual_gain",
            ylabel=r"mean projected residual gain $\|\widehat R\|_F$",
            figure_title=f"{editing_method} edit-count gain and locality trajectory",
            output_name="trajectory_gain_locality.png",
        )

    fig, axes = plt.subplots(1, 3, figsize=(20.0, 5.0), squeeze=False)
    flat_axes = axes[0]
    plot_layer_profiles(
        flat_axes,
        "hidden_rms_ratio",
        "residual-input RMS / Base",
        base_line=True,
    )
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=min(len(labels), 9),
        frameon=False,
    )
    fig.suptitle(
        f"{editing_method} edit-count residual-input RMS trajectory",
        fontweight="bold",
        y=1.11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    rms_limits = rms_break_limits()
    fig.savefig(
        output_dir
        / (
            "trajectory_rms_locality_linear.png"
            if rms_limits is not None
            else "trajectory_rms_locality.png"
        ),
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)
    if rms_limits is not None:
        plot_broken_profile_trajectory(
            rms_limits,
            metric="hidden_rms_ratio",
            ylabel="residual-input RMS / Base",
            figure_title=(
                f"{editing_method} edit-count residual-input RMS trajectory"
            ),
            output_name="trajectory_rms_locality.png",
            upper_log=True,
            base_line=True,
        )


def plot_f_trajectory(output_dir: Path, editing_method: str) -> None:
    """Plot complete-block residual-write RMS and locality trajectories.

    Unlike ``trajectory_rms_locality.png``, whose layer profiles contain the
    residual-input state ``H_l``, this figure uses the complete block write

        ``F_l = H_(l+1) - H_l``

    and reports its pooled RMS relative to the same probe set in the Base
    model.  Only ``write_kind=total`` is used; attention, MLP, and input rows
    remain available in ``trajectory_branch_summary.csv`` for decomposition.
    """

    branch_path = output_dir / "trajectory_branch_summary.csv"
    outcomes_path = output_dir / "trajectory_checkpoint_outcomes.csv"
    if not branch_path.exists() or not outcomes_path.exists():
        print(
            "[plot] skipped complete-block F trajectory: branch summary or "
            "checkpoint outcomes are missing"
        )
        return

    panel_specs = [
        ("edit", "subject_last", "Edit prompts — subject last"),
        ("locality", "prompt_last", "Locality prompts — prompt last"),
    ]
    panel_pairs = {(context, position) for context, position, _ in panel_specs}
    branch_rows = [
        row
        for row in read_csv_rows(branch_path)
        if str(row.get("write_kind")) == "total"
        and (str(row.get("context")), str(row.get("position"))) in panel_pairs
        and math.isfinite(float(row.get("rms_l2_ratio_to_base", float("nan"))))
    ]
    available = {
        (str(row["context"]), str(row["position"])) for row in branch_rows
    }
    if any(pair not in available for pair in panel_pairs):
        print(
            "[plot] skipped complete-block F trajectory: "
            "edit/subject_last and locality/prompt_last are both required"
        )
        return

    completed = sorted({int(row["edit_count"]) for row in branch_rows})
    color_map = plt.get_cmap("viridis")
    color_by_step = {
        step: color_map(index / max(len(completed) - 1, 1))
        for index, step in enumerate(completed)
    }
    outcome_lookup = {
        int(row["edit_count"]): float(row["locality_acc"])
        for row in read_csv_rows(outcomes_path)
        if math.isfinite(float(row.get("locality_acc", float("nan"))))
    }

    locality_steps = [step for step in completed if step in outcome_lookup]

    def draw_profiles(profile_axes: Sequence[Any]) -> None:
        for axis, (context, position, title) in zip(profile_axes, panel_specs):
            selected = [
                row
                for row in branch_rows
                if str(row["context"]) == context
                and str(row["position"]) == position
            ]
            for step in completed:
                current = sorted(
                    [row for row in selected if int(row["edit_count"]) == step],
                    key=lambda row: int(row["layer"]),
                )
                if not current:
                    continue
                axis.plot(
                    [int(row["layer"]) for row in current],
                    [float(row["rms_l2_ratio_to_base"]) for row in current],
                    marker="o",
                    color=color_by_step[step],
                    label=f"edit {step}",
                )
            axis.axhline(1.0, color="#666666", linestyle="--", linewidth=1)
            axis.set_title(title)
            axis.set_xlabel("decoder layer")
            axis.set_ylabel(
                r"total-write RMS $\mathrm{RMS}(F_l)/\mathrm{RMS}(F_l^{Base})$"
            )
            axis.grid(alpha=0.25)

    def draw_locality(axis: Any) -> None:
        axis.plot(
            locality_steps,
            [outcome_lookup[step] for step in locality_steps],
            color="#D81B60",
            marker="o",
            linewidth=2,
        )
        for step in locality_steps:
            axis.annotate(
                str(step),
                (step, outcome_lookup[step]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_title("Locality preservation by edit count")
        axis.set_xlabel("cumulative edits")
        axis.set_ylabel("Base-output token agreement")
        axis.grid(alpha=0.25)

    all_values = np.asarray(
        [float(row["rms_l2_ratio_to_base"]) for row in branch_rows],
        dtype=np.float64,
    )
    early_values = np.asarray(
        [
            float(row["rms_l2_ratio_to_base"])
            for row in branch_rows
            if int(row["edit_count"]) <= 200
        ],
        dtype=np.float64,
    )
    break_limits: Tuple[float, float, float, float] | None = None
    if all_values.size >= 20 and early_values.size > 0:
        lower_min = min(0.0, 0.95 * float(early_values.min()))
        lower_max = max(1.5, 1.10 * float(early_values.max()))
        upper_candidates = all_values[all_values > 1.40 * lower_max]
        if (
            upper_candidates.size > 0
            and float(all_values.max()) > 100.0 * lower_max
        ):
            upper_min = 0.95 * float(upper_candidates.min())
            if upper_min > 1.15 * lower_max:
                break_limits = (
                    lower_min,
                    lower_max,
                    upper_min,
                    1.10 * float(all_values.max()),
                )

    fig, axes = plt.subplots(1, 3, figsize=(20.0, 5.0), squeeze=False)
    flat_axes = axes[0]
    draw_profiles(flat_axes[:2])
    draw_locality(flat_axes[2])

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=min(len(labels), 9),
        frameon=False,
    )
    run_name = output_dir.name.lower()
    if "rgr_expcos_" in run_name:
        expcos_tokens = run_name.split("rgr_expcos_", 1)[1].split("_")
        cosine_lambda = expcos_tokens[0][1:].replace("p", ".")
        cosine_gamma = expcos_tokens[1][1:].replace("p", ".")
        method_label = (
            f"RGR+ExpCos lambda={cosine_lambda}, gamma={cosine_gamma}"
        )
    else:
        method_label = next(
            (
                label
                for marker, label in (
                    ("baseline", "Baseline"),
                    ("sadr", "SADR"),
                    ("encore", "ENCORE"),
                    ("nas", "NAS"),
                    ("rgr", "RGR"),
                )
                if marker in run_name
            ),
            "",
        )
    figure_title = " ".join(
        value
        for value in (
            editing_method,
            method_label,
            "edit-count complete-block write trajectory",
        )
        if value
    )
    fig.suptitle(figure_title, fontweight="bold", y=1.11)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(
        output_dir
        / (
            "trajectory_f_locality_linear.png"
            if break_limits is not None
            else "trajectory_f_locality.png"
        ),
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)

    if break_limits is None:
        return

    lower_min, lower_max, upper_min, upper_max = break_limits
    fig = plt.figure(figsize=(20.0, 6.2))
    outer = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.26)
    broken_axes: List[Tuple[Any, Any]] = []
    for column in range(2):
        nested = outer[column].subgridspec(
            2,
            1,
            height_ratios=[1.0, 3.2],
            hspace=0.045,
        )
        upper = fig.add_subplot(nested[0])
        lower = fig.add_subplot(nested[1], sharex=upper)
        broken_axes.append((upper, lower))
    locality_axis = fig.add_subplot(outer[2])

    for (upper, lower), (context, position, title) in zip(
        broken_axes, panel_specs
    ):
        selected = [
            row
            for row in branch_rows
            if str(row["context"]) == context
            and str(row["position"]) == position
        ]
        for step in completed:
            current = sorted(
                [row for row in selected if int(row["edit_count"]) == step],
                key=lambda row: int(row["layer"]),
            )
            if not current:
                continue
            x_values = [int(row["layer"]) for row in current]
            y_values = [float(row["rms_l2_ratio_to_base"]) for row in current]
            for axis in (upper, lower):
                axis.plot(
                    x_values,
                    y_values,
                    marker="o",
                    color=color_by_step[step],
                    label=f"edit {step}",
                )
        upper.set_yscale("log")
        upper.set_ylim(upper_min, upper_max)
        lower.set_ylim(lower_min, lower_max)
        upper.spines["bottom"].set_visible(False)
        lower.spines["top"].set_visible(False)
        upper.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        lower.axhline(1.0, color="#666666", linestyle="--", linewidth=1)
        upper.set_title(title)
        lower.set_xlabel("decoder layer")
        lower.set_ylabel(
            r"total-write RMS $\mathrm{RMS}(F_l)/\mathrm{RMS}(F_l^{Base})$"
        )
        upper.grid(alpha=0.25)
        lower.grid(alpha=0.25)
        marker_kwargs = {
            "marker": [(-1, -0.55), (1, 0.55)],
            "markersize": 8,
            "linestyle": "none",
            "color": "k",
            "mec": "k",
            "mew": 1.0,
            "clip_on": False,
        }
        upper.plot([0, 1], [0, 0], transform=upper.transAxes, **marker_kwargs)
        lower.plot([0, 1], [1, 1], transform=lower.transAxes, **marker_kwargs)

    draw_locality(locality_axis)
    handles, labels = broken_axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=min(len(labels), 9),
        frameon=False,
    )
    fig.suptitle(figure_title, fontweight="bold", y=1.075)
    fig.savefig(
        output_dir / "trajectory_f_locality.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--editing-method",
        choices=["MEMIT", "AlphaEdit"],
        default="MEMIT",
    )
    parser.add_argument(
        "--hparams-path",
        default=None,
        help="Defaults to hparams/<editing-method>/llama3-8b.yaml",
    )
    parser.add_argument("--data-path", default="data/zsre/zsre_3k.json")
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--steps",
        type=parse_steps,
        default=parse_steps("0,10,20,50,100,150,200,250,300"),
    )
    parser.add_argument(
        "--layers",
        type=parse_int_list,
        default=parse_int_list("4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"),
    )
    parser.add_argument(
        "--window-layers",
        type=parse_int_list,
        default=parse_int_list("8,9,10,11,12,13,14,15,16,17,18"),
    )
    parser.add_argument(
        "--contexts",
        type=parse_csv_strings,
        default=parse_csv_strings("edit_subject_last,locality_prompt_last"),
    )
    parser.add_argument("--probe-prompts", type=int, default=50)
    parser.add_argument("--jacobian-prompts", type=int, default=50)
    parser.add_argument("--locality-eval-prompts", type=int, default=50)
    parser.add_argument("--pca-rank", type=int, default=32)
    parser.add_argument(
        "--branch-analysis",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "At every edit count, also save residual-input and "
            "total/attention/MLP write RMS, diversity, and common-mode metrics."
        ),
    )
    parser.add_argument(
        "--interaction-analysis",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "At every edit count, decompose cumulative down-projection drift "
            "as W0Δk + ΔWk0 + ΔWΔk."
        ),
    )
    parser.add_argument(
        "--interaction-layers",
        type=parse_int_list,
        default=parse_int_list("4,5,6,7,8"),
        help="Edited MLP down-projection layers used for ΔW–Δk decomposition.",
    )
    parser.add_argument(
        "--covariance-analysis",
        type=int,
        choices=[0, 1],
        default=0,
        help="Also measure full and fixed-Base-PCA residual covariance spectra.",
    )
    parser.add_argument(
        "--residual-gain-regularization",
        type=int,
        choices=[0, 1],
        default=0,
        help="Run the trajectory with RGR enabled instead of the stock editor.",
    )
    parser.add_argument(
        "--analysis-condition",
        default=None,
        help=(
            "Method condition for the full diagnostic trajectory. When omitted, "
            "--residual-gain-regularization=1 selects rgr and otherwise baseline. "
            "Components may be combined with '+' (e.g. rgr+nas) to test whether "
            "RGR plugs into an existing method; 'baseline' cannot be combined, "
            "and nse is excluded because it replaces compute_z outright."
        ),
    )
    parser.add_argument("--sadr-lambda", type=float, default=0.01)
    parser.add_argument(
        "--sadr-attn-layers",
        type=parse_int_list,
        default=list(range(32)),
    )
    parser.add_argument("--sadr-efficacy-threshold", type=float, default=0.5)
    parser.add_argument("--encore-mpes-top1-steps", type=int, default=2)
    parser.add_argument(
        "--encore-mpes-exclude-first-context",
        type=int,
        choices=[0, 1],
        default=1,
    )
    parser.add_argument("--encore-norm-lambda-memit", type=float, default=20.0)
    parser.add_argument(
        "--sphere-beta",
        type=float,
        default=0.5,
        help="SPHERE retained spectral-energy fraction (official Llama-3: 0.5).",
    )
    parser.add_argument(
        "--sphere-alpha",
        type=float,
        default=None,
        help=(
            "SPHERE soft projection strength. Defaults to the official "
            "Llama-3 value: 0.8 for MEMIT and 0.5 for AlphaEdit."
        ),
    )
    parser.add_argument("--nas-anchor-path", default=None)
    parser.add_argument("--nas-outlier-factor", type=float, default=2.0)
    parser.add_argument(
        "--nas-outlier-mode",
        choices=["skip_delta", "scale"],
        default="skip_delta",
    )
    parser.add_argument("--residual-gain-lambda", type=float, default=0.1)
    parser.add_argument(
        "--residual-gain-subject-layers",
        type=parse_int_list,
        default=parse_int_list("8"),
    )
    parser.add_argument(
        "--residual-gain-margin",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--residual-gain-loss-type",
        choices=["positive_l1", "positive_squared", "absolute_l1"],
        default="positive_l1",
    )
    parser.add_argument(
        "--residual-gain-objective",
        choices=[
            "gain",
            "delta_relative_energy",
            "write_relative_energy",
            "cosine_drift",
            "alignment_drift",
            "cosine_amplified_gain",
            "shifted_cosine_energy",
            "rho_minus_one",
            "log_rho",
            "norm_difference",
            "write_energy",
            "tangential_deficit",
        ],
        default="gain",
        help=(
            "RGR-family objective relative to the per-edit zero-delta "
            "reference. The default gain is the original RGR; "
            "write_relative_energy controls r^2 = ||F||^2/||H||^2; "
            "cosine_amplified_gain increases the input--write alignment "
            "term inside that gain decomposition; shifted_cosine_energy "
            "controls r^2 + 2r(cos(H,F) + 1); rho_minus_one and "
            "log_rho are scale-softened monotone transforms of gain. "
            "tangential_deficit is the only floor rather than a ceiling: "
            "it penalises the drop of ||P_u F||^2/||H||^2 below the "
            "zero-delta reference, i.e. the block losing its ability to "
            "rotate the residual direction."
        ),
    )
    parser.add_argument(
        "--residual-gain-alignment-weight",
        type=float,
        default=1.0,
        help=(
            "Multiplier gamma on 2<H,F>/||H||^2 when objective is "
            "cosine_amplified_gain. gamma=1 exactly recovers ordinary gain."
        ),
    )
    parser.add_argument(
        "--residual-gain-cosine-aux-lambda",
        type=float,
        default=0.0,
        help=(
            "Weight of the independent absolute exponential cosine target "
            "phi(cos(H,F)); zero preserves ordinary RGR."
        ),
    )
    parser.add_argument(
        "--residual-gain-cosine-aux-sharpness",
        type=float,
        default=1.0,
        help=(
            "Positive gamma in expm1(gamma*(cos(H,F)+1))/expm1(2*gamma)."
        ),
    )
    parser.add_argument("--residual-gain-efficacy-threshold", type=float, default=0.0)
    add_oedit_arguments(parser)
    add_early_attention_arguments(parser)
    add_o0_axis_arguments(parser)
    parser.add_argument(
        "--inner-margin-schedule",
        choices=["legacy", "fixed_stop", "delayed_rgr", "two_stage"],
        default="legacy",
        help=(
            "AlphaEdit causal ablation. fixed_stop ends target formation once "
            "every supervised rewrite token clears the margin; delayed_rgr "
            "then runs ordinary NLL+RGR for a fixed update budget; two_stage "
            "uses a margin-floor hinge plus RGR during that refinement."
        ),
    )
    parser.add_argument(
        "--inner-target-margin-threshold",
        type=float,
        default=0.0,
        help="Target-vs-best-alternative logit-margin floor for the inner schedule.",
    )
    parser.add_argument(
        "--inner-margin-refinement-steps",
        type=int,
        default=0,
        help="Number of optimizer updates after the margin floor is first reached.",
    )
    parser.add_argument(
        "--inner-margin-hinge-weight",
        type=float,
        default=1.0,
        help="Weight on the target-margin floor hinge in two_stage refinement.",
    )
    parser.add_argument(
        "--context-multikey-enabled",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "AlphaEdit only: retain the already-generated original/prefix "
            "context keys as an exact same-target Gram in the outer solve "
            "and cumulative cache. No additional prompts are introduced."
        ),
    )
    parser.add_argument(
        "--key-gaussian-noise-enabled",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "MEMIT/AlphaEdit: replace the original+five-prefix mean key "
            "with one canonical prompt key plus deterministic Gaussian noise."
        ),
    )
    parser.add_argument(
        "--key-gaussian-noise-relative-std",
        type=float,
        default=0.05,
        help=(
            "Gaussian key coordinate std as a fraction of ||k||/sqrt(d); "
            "the expected noise/key L2 ratio is approximately this value."
        ),
    )
    parser.add_argument(
        "--key-gaussian-noise-seed",
        type=int,
        default=None,
        help=(
            "Base seed for deterministic per-request/per-layer key noise. "
            "Defaults to --seed when the mode is enabled."
        ),
    )
    parser.add_argument(
        "--tangent-layer-allocation-enabled",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "AlphaEdit only: allocate the initial z* error jointly across "
            "clean per-layer local tangent spaces under J_l ~= I. This is "
            "independent of --context-multikey-enabled and requires bs=1."
        ),
    )
    parser.add_argument(
        "--tangent-layer-allocation-rcond",
        type=float,
        default=None,
        help="Optional relative eigenvalue cutoff for the joint tangent solve.",
    )
    parser.add_argument(
        "--tangent-layer-allocation-max-relative-slack",
        type=float,
        default=None,
        help=(
            "Optional fail-fast maximum ||tangent-null slack||/||target||. "
            "Slack is reported and never assigned to the terminal layer."
        ),
    )
    parser.add_argument(
        "--save-layer-checkpoints",
        type=parse_int_list,
        default=[],
        help=(
            "Edit-count checkpoints at which to save only the parameters "
            "modified in --save-layer-ids. Example: 300 or 100,300."
        ),
    )
    parser.add_argument(
        "--save-layer-ids",
        type=parse_int_list,
        default=parse_int_list("4,5,6,7,8"),
        help=(
            "Rewrite layers whose changed parameters are stored in each "
            "compact checkpoint."
        ),
    )
    parser.add_argument(
        "--resume-layer-checkpoint",
        default=None,
        help=(
            "Compact edited-parameter checkpoint to apply to the freshly "
            "loaded Base model before continuing. Its metadata edit_count "
            "determines the first new request (edit_count + 1). Completed "
            "analyses at or below that count must already exist."
        ),
    )
    parser.add_argument(
        "--resume-context-templates-json",
        default=None,
        help=(
            "JSON containing the editor context_templates used before the "
            "resume checkpoint. This avoids regenerating stochastic MEMIT/"
            "AlphaEdit templates in a fresh process."
        ),
    )
    parser.add_argument(
        "--save-edit-artifacts",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Save per-edit v*, delta*, keys, outer realization statistics, "
            "and record-only inner residual-gain trajectories."
        ),
    )
    parser.add_argument(
        "--track-post-update",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "For every edit, re-forward the current edit request immediately "
            "before and after apply_algo and save paired boundary vectors plus "
            "continuous target likelihoods. This is an actual post-update "
            "measurement, unlike an outer-solve prediction."
        ),
    )
    parser.add_argument(
        "--post-update-layers",
        type=parse_int_list,
        default=parse_int_list("8,9,10"),
        help=(
            "Decoder layers permitted in --post-update-nodes. Kept explicit "
            "in the run fingerprint so the capture scope is reproducible."
        ),
    )
    parser.add_argument(
        "--capture-virtual-actual", type=int, choices=[0, 1], default=0,
        help="AlphaEdit: capture all-layer O/A/M on current raw rewrite prompt with selected virtual z and committed actual weights.",
    )
    parser.add_argument(
        "--virtual-actual-save-vectors", type=int, choices=[0, 1], default=1,
        help="Retain all raw O/A/M vectors alongside paired cosine matrices and norms when capture is enabled.",
    )
    parser.add_argument(
        "--post-update-nodes",
        type=parse_post_update_nodes,
        default=parse_post_update_nodes("H8,M8,H9,A9,H10"),
        help=(
            "Comma-separated boundary nodes: H=input, A=attention write, "
            "M=MLP write, F=total block write, followed by decoder layer."
        ),
    )
    parser.add_argument(
        "--continuous-behavior",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "At each analysis checkpoint, retain token accuracy and also "
            "score teacher-forced target mean log-probability, NLL, "
            "geometric-mean probability, and new-vs-old margins."
        ),
    )
    parser.add_argument(
        "--fixed-probe-path",
        default=None,
        help=(
            "Deterministic disjoint request JSON used by the lightweight "
            "per-edit fixed-panel observer. Empty disables the observer."
        ),
    )
    parser.add_argument(
        "--fixed-probe-manifest",
        default=None,
        help="Optional selection manifest copied into the run for auditability.",
    )
    parser.add_argument("--fixed-probe-prompts", type=int, default=200)
    parser.add_argument("--fixed-probe-interval", type=int, default=1)
    parser.add_argument("--fixed-probe-batch-size", type=int, default=32)
    parser.add_argument(
        "--fixed-probe-contexts",
        type=parse_fixed_probe_contexts,
        default=FIXED_PROBE_DEFAULT_CONTEXTS,
    )
    parser.add_argument(
        "--fixed-probe-vector-interval",
        type=int,
        default=10,
        help=(
            "Save H8/F8/H9/F9/H10 vectors every N edits; H9 temporal state "
            "and scalar metrics are always saved at every observed edit."
        ),
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--analysis-seed",
        type=int,
        default=None,
        help="Seed for held-out diagnostic prompt selection; defaults to --seed.",
    )
    parser.add_argument(
        "--edit-order",
        choices=["prefix", "shuffle"],
        default="prefix",
        help="Order of the fixed prefix edit pool. Shuffle changes order, not membership.",
    )
    parser.add_argument(
        "--edit-order-file",
        default=None,
        help=(
            "Validated deterministic fixed-prefix order manifest. This changes "
            "only request order; --seed and all model/training RNG streams stay "
            "fixed. Mutually exclusive with --edit-order shuffle."
        ),
    )
    parser.add_argument("--selection", choices=["prefix"], default="prefix")
    parser.add_argument("--append-eos-to-target", type=int, choices=[0, 1], default=1)
    return parser


#: Components that may be switched on independently in one trajectory run.  ``nse`` is
#: absent on purpose: it replaces compute_z with a cached target, so anything that
#: shapes the latent objective (RGR, SADR) would be silently discarded alongside it.
CONDITION_COMPONENTS = ("sadr", "encore", "sphere", "nas", "rgr", "oedit")


def condition_components(condition: str) -> frozenset:
    """Parse a ``+``-joined condition into the set of components to enable.

    A single name behaves exactly as before, so existing callers are unaffected.
    ``rgr+nas`` enables both, which is how the plug-in question -- does RGR compose
    with an existing method, or do the two fight over the same degree of freedom --
    gets asked experimentally.
    """

    if condition in O0_AXIS_CONDITION_COMPONENTS:
        return O0_AXIS_CONDITION_COMPONENTS[condition]
    parts = [p for p in str(condition).split("+") if p]
    if not parts:
        raise ValueError("--analysis-condition must not be empty")
    if "baseline" in parts:
        if len(parts) > 1:
            raise ValueError(
                f"'baseline' means no components and cannot be combined: {condition!r}"
            )
        return frozenset()
    unknown = [p for p in parts if p not in CONDITION_COMPONENTS]
    if unknown:
        raise ValueError(
            f"unknown condition component(s) {unknown} in {condition!r}; "
            f"choose from {list(CONDITION_COMPONENTS)} or 'baseline'"
        )
    if len(set(parts)) != len(parts):
        raise ValueError(f"repeated component in {condition!r}")
    if "oedit" in parts and len(parts) != 1:
        raise ValueError("OEdit must be the standalone 'oedit' condition")
    return frozenset(parts)


def validate_args(args: argparse.Namespace) -> None:
    validate_o0_axis_args(args)
    if args.batch_size != 1:
        raise ValueError("This trajectory experiment requires --batch-size 1")
    if args.sample_size < max(args.steps):
        raise ValueError(
            f"sample size {args.sample_size} is smaller than the last requested step {max(args.steps)}"
        )
    if args.probe_prompts <= 1:
        raise ValueError("--probe-prompts must be greater than one")
    if args.jacobian_prompts <= 0 or args.jacobian_prompts > args.probe_prompts:
        raise ValueError("--jacobian-prompts must be in [1, probe-prompts]")
    if args.locality_eval_prompts <= 0:
        raise ValueError("--locality-eval-prompts must be positive")
    if args.edit_order_file:
        edit_order_path = Path(args.edit_order_file).expanduser()
        if not edit_order_path.is_file():
            raise FileNotFoundError(
                f"--edit-order-file does not exist: {edit_order_path}"
            )
        if args.edit_order != "prefix":
            raise ValueError(
                "--edit-order-file is mutually exclusive with "
                "--edit-order shuffle"
            )
    if args.fixed_probe_path:
        fixed_path = Path(args.fixed_probe_path).expanduser()
        if not fixed_path.is_file():
            raise FileNotFoundError(f"--fixed-probe-path does not exist: {fixed_path}")
        if args.fixed_probe_prompts <= 1:
            raise ValueError("--fixed-probe-prompts must exceed one")
        if args.fixed_probe_interval <= 0:
            raise ValueError("--fixed-probe-interval must be positive")
        if args.fixed_probe_batch_size <= 0:
            raise ValueError("--fixed-probe-batch-size must be positive")
        if args.fixed_probe_vector_interval < 0:
            raise ValueError("--fixed-probe-vector-interval cannot be negative")
        if args.fixed_probe_manifest and not Path(
            args.fixed_probe_manifest
        ).expanduser().is_file():
            raise FileNotFoundError(
                f"--fixed-probe-manifest does not exist: {args.fixed_probe_manifest}"
            )
    elif args.fixed_probe_manifest:
        raise ValueError("--fixed-probe-manifest requires --fixed-probe-path")
    if args.pca_rank <= 0:
        raise ValueError("--pca-rank must be positive")
    if not set(args.window_layers).issubset(set(args.layers)):
        raise ValueError("--window-layers must be a subset of --layers")
    if not set(args.interaction_layers).issubset(set(args.layers)):
        raise ValueError("--interaction-layers must be a subset of --layers")
    if not set(args.save_layer_checkpoints).issubset(set(args.steps)):
        raise ValueError("--save-layer-checkpoints must be a subset of --steps")
    if args.save_layer_checkpoints and not args.save_layer_ids:
        raise ValueError("--save-layer-ids cannot be empty when saving checkpoints")
    post_node_layers = {
        int(str(node)[1:]) for node in args.post_update_nodes
    }
    if bool(args.track_post_update) and not post_node_layers.issubset(
        set(args.post_update_layers)
    ):
        raise ValueError(
            "Every --post-update-node layer must be included in "
            f"--post-update-layers; nodes use {sorted(post_node_layers)}, "
            f"allowed layers are {list(args.post_update_layers)}"
        )
    if args.resume_layer_checkpoint:
        checkpoint_path = Path(args.resume_layer_checkpoint).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"--resume-layer-checkpoint does not exist: {checkpoint_path}"
            )
        manifest_path = checkpoint_path.with_suffix(".json")
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "Resume checkpoint is missing its JSON manifest: "
                f"{manifest_path}"
            )
    if args.resume_context_templates_json:
        template_path = Path(args.resume_context_templates_json).expanduser()
        if not template_path.is_file():
            raise FileNotFoundError(
                "--resume-context-templates-json does not exist: "
                f"{template_path}"
            )
        if not args.resume_layer_checkpoint:
            raise ValueError(
                "--resume-context-templates-json requires "
                "--resume-layer-checkpoint"
            )
    if bool(args.context_multikey_enabled):
        if args.editing_method != "AlphaEdit":
            raise ValueError(
                "--context-multikey-enabled=1 is only valid for AlphaEdit"
            )
        if args.resume_layer_checkpoint:
            raise ValueError(
                "Context multi-key checkpoint resume requires the cumulative "
                "AlphaEdit cache_c state, but compact layer checkpoints store "
                "only parameter deltas. Resume by replaying the completed "
                "prefix (omit --resume-layer-checkpoint)."
            )
    if bool(args.key_gaussian_noise_enabled):
        if not math.isfinite(float(args.key_gaussian_noise_relative_std)) or (
            float(args.key_gaussian_noise_relative_std) <= 0.0
        ):
            raise ValueError(
                "--key-gaussian-noise-relative-std must be finite and > 0"
            )
        if bool(args.context_multikey_enabled):
            raise ValueError(
                "--key-gaussian-noise-enabled and --context-multikey-enabled "
                "are mutually exclusive"
            )
        if args.editing_method == "AlphaEdit" and args.resume_layer_checkpoint:
            raise ValueError(
                "Gaussian-key checkpoint resume requires the cumulative "
                "AlphaEdit cache_c state, but compact layer checkpoints store "
                "only parameter deltas. Resume by replaying the completed "
                "prefix (omit --resume-layer-checkpoint)."
            )
    if bool(args.tangent_layer_allocation_enabled):
        if args.editing_method != "AlphaEdit":
            raise ValueError(
                "--tangent-layer-allocation-enabled=1 is only valid for AlphaEdit"
            )
        if args.batch_size != 1:
            raise ValueError(
                "Strict AlphaEdit tangent layer allocation requires --batch-size 1"
            )
        if args.resume_layer_checkpoint:
            raise ValueError(
                "A strict tangent-allocation trajectory must start at edit 1. "
                "A parameter-only checkpoint cannot prove that its completed "
                "prefix used the same layer-allocation intervention."
            )
    if (
        args.tangent_layer_allocation_rcond is not None
        and not 0.0 <= float(args.tangent_layer_allocation_rcond) < 1.0
    ):
        raise ValueError("--tangent-layer-allocation-rcond must be in [0, 1)")
    if (
        args.tangent_layer_allocation_max_relative_slack is not None
        and float(args.tangent_layer_allocation_max_relative_slack) < 0.0
    ):
        raise ValueError(
            "--tangent-layer-allocation-max-relative-slack must be non-negative"
        )
    inferred_condition = (
        args.analysis_condition
        or ("rgr" if bool(args.residual_gain_regularization) else "baseline")
    )
    inferred_active = condition_components(inferred_condition)
    validate_oedit_args(args, inferred_condition)
    if args.inner_margin_schedule != "legacy":
        if args.editing_method != "AlphaEdit":
            raise ValueError(
                "The initial inner-margin causal ablation is AlphaEdit-only"
            )
        if args.resume_layer_checkpoint:
            raise ValueError(
                "Inner-margin trajectories must start at edit 1; compact "
                "checkpoints do not preserve optimizer-stage provenance"
            )
        if args.inner_margin_refinement_steps < 0:
            raise ValueError("--inner-margin-refinement-steps must be non-negative")
        if not math.isfinite(float(args.inner_target_margin_threshold)):
            raise ValueError("--inner-target-margin-threshold must be finite")
        if not math.isfinite(float(args.inner_margin_hinge_weight)):
            raise ValueError("--inner-margin-hinge-weight must be finite")
        if args.inner_margin_hinge_weight < 0.0:
            raise ValueError("--inner-margin-hinge-weight must be non-negative")
        if args.inner_margin_schedule == "fixed_stop":
            if "rgr" in inferred_active:
                raise ValueError("fixed_stop is the no-RGR target-formation arm")
            if args.inner_margin_refinement_steps != 0:
                raise ValueError("fixed_stop requires zero refinement steps")
        else:
            if "rgr" not in inferred_active:
                raise ValueError(
                    f"{args.inner_margin_schedule} requires an RGR condition"
                )
            if args.inner_margin_refinement_steps <= 0:
                raise ValueError(
                    f"{args.inner_margin_schedule} requires positive refinement steps"
                )
        incompatible = inferred_active.difference({"rgr"})
        if incompatible:
            raise ValueError(
                "Inner-margin causal arms cannot be mixed with other method "
                f"components: {sorted(incompatible)}"
            )
        if (
            bool(args.context_multikey_enabled)
            or bool(args.key_gaussian_noise_enabled)
            or bool(args.tangent_layer_allocation_enabled)
        ):
            raise ValueError(
                "Inner-margin causal arms keep the native outer solve; disable "
                "context multi-key, Gaussian key noise, and tangent allocation"
            )
    if "nas" in inferred_active and not args.nas_anchor_path:
        raise ValueError("NAS requires --nas-anchor-path")
    if "sphere" in inferred_active:
        if not 0.0 < float(args.sphere_beta) <= 1.0:
            raise ValueError("SPHERE --sphere-beta must be in (0, 1]")
        if args.sphere_alpha is not None and not 0.0 <= float(
            args.sphere_alpha
        ) <= 1.0:
            raise ValueError("SPHERE --sphere-alpha must be in [0, 1]")
    context_layout(args.contexts)


def main() -> None:
    args = build_parser().parse_args()
    if args.hparams_path is None:
        args.hparams_path = f"hparams/{args.editing_method}/llama3-8b.yaml"
    explicit_analysis_seed = args.analysis_seed is not None
    if args.analysis_seed is None:
        args.analysis_seed = args.seed
    validate_args(args)
    key_gaussian_noise_seed = int(
        args.seed
        if args.key_gaussian_noise_seed is None
        else args.key_gaussian_noise_seed
    )
    analysis_condition = (
        args.analysis_condition
        or ("rgr" if bool(args.residual_gain_regularization) else "baseline")
    )
    # --residual-gain-regularization used to be mutually exclusive with the named
    # conditions.  Now that components compose, fold the flag in rather than letting
    # it be silently overridden by the condition string.
    if (
        bool(args.residual_gain_regularization)
        and "rgr" not in condition_components(analysis_condition)
    ):
        analysis_condition = (
            "rgr" if analysis_condition == "baseline"
            else f"{analysis_condition}+rgr"
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    layout = context_layout(args.contexts)

    edit_order_permutation: List[int] | None = None
    edit_order_metadata: Dict[str, Any] | None = None
    if args.edit_order_file:
        edit_order_manifest_path = (
            Path(args.edit_order_file).expanduser().resolve()
        )
        edit_order_permutation, edit_order_metadata = (
            validate_edit_order_manifest(
                edit_order_manifest_path,
                data_path=Path(args.data_path),
                sample_size=args.sample_size,
            )
        )
        archived_manifest_path = output_dir / "edit_order_manifest.json"
        if archived_manifest_path.exists():
            if edit_order_file_sha256(archived_manifest_path) != str(
                edit_order_metadata["sha256"]
            ):
                raise ValueError(
                    "Existing archived edit-order manifest differs from "
                    f"--edit-order-file: {archived_manifest_path}"
                )
        elif archived_manifest_path != edit_order_manifest_path:
            shutil.copy2(edit_order_manifest_path, archived_manifest_path)
        # Fingerprint the immutable archived copy rather than the caller's
        # possibly relocated manifest path. This keeps resumes portable while
        # preserving the byte-level manifest hash in run_config.json.
        edit_order_metadata["path"] = str(archived_manifest_path)

    edit_runner.add_easyedit_to_syspath(None)
    edit_runner.init_easyedit_imports()
    edit_runner.fix_seed(args.seed)
    hparams = edit_runner.HPARAMS_REGISTRY[args.editing_method].from_hparams(args.hparams_path)
    hparams.batch_size = 1
    hparams.model_name = args.model_name
    # One component per run by default; a '+'-joined condition activates several so
    # that RGR can be measured as a plug-in on top of an existing method.
    active_components = condition_components(analysis_condition)
    hparams.residual_gain_regularization = "rgr" in active_components
    hparams.sadr_regularization = "sadr" in active_components
    hparams.encore_enabled = False
    hparams.nse_enabled = False
    hparams.sphere_enabled = False
    hparams.nas_enabled = False
    hparams.nas_collect_stats = False
    if hasattr(hparams, "context_multikey_enabled"):
        hparams.context_multikey_enabled = bool(
            args.context_multikey_enabled
        )
        hparams.context_multikey_expected_group_sizes = (
            [1, 5] if bool(args.context_multikey_enabled) else None
        )
        hparams.context_multikey_log_path = (
            str(output_dir / "context_multikey.jsonl")
            if bool(args.context_multikey_enabled)
            else None
        )
    if hasattr(hparams, "key_gaussian_noise_enabled"):
        hparams.key_gaussian_noise_enabled = bool(
            args.key_gaussian_noise_enabled
        )
        if bool(args.key_gaussian_noise_enabled):
            hparams.key_gaussian_noise_relative_std = float(
                args.key_gaussian_noise_relative_std
            )
            hparams.key_gaussian_noise_seed = key_gaussian_noise_seed
            hparams.key_gaussian_noise_log_path = str(
                output_dir / "key_gaussian_noise.jsonl"
            )
        else:
            hparams.key_gaussian_noise_log_path = None
    if hasattr(hparams, "tangent_layer_allocation_enabled"):
        hparams.tangent_layer_allocation_enabled = bool(
            args.tangent_layer_allocation_enabled
        )
        hparams.tangent_layer_allocation_rcond = (
            float(args.tangent_layer_allocation_rcond)
            if args.tangent_layer_allocation_rcond is not None
            else None
        )
        hparams.tangent_layer_allocation_max_relative_slack = (
            float(args.tangent_layer_allocation_max_relative_slack)
            if args.tangent_layer_allocation_max_relative_slack is not None
            else None
        )
        hparams.tangent_layer_allocation_log_path = (
            str(output_dir / "tangent_layer_allocation.jsonl")
            if bool(args.tangent_layer_allocation_enabled)
            else None
        )
    hparams.encore_mpes_top1_steps = int(args.encore_mpes_top1_steps)
    hparams.encore_mpes_exclude_first_context = bool(
        args.encore_mpes_exclude_first_context
    )
    hparams.encore_norm_lambda = (
        float(args.encore_norm_lambda_memit)
        if args.editing_method == "MEMIT"
        else 0.0
    )
    hparams.official_baseline_log_path = str(
        output_dir / "official_baseline.jsonl"
    )
    if "encore" in active_components:
        hparams.encore_enabled = True
    sphere_alpha = (
        float(args.sphere_alpha)
        if args.sphere_alpha is not None
        else (0.8 if args.editing_method == "MEMIT" else 0.5)
    )
    hparams.sphere_beta = float(args.sphere_beta)
    hparams.sphere_alpha = sphere_alpha
    if "sphere" in active_components:
        hparams.sphere_enabled = True
    if "nas" in active_components:
        hparams.nas_enabled = True
        hparams.nas_anchor_path = str(Path(args.nas_anchor_path).resolve())
        hparams.nas_outlier_factor = float(args.nas_outlier_factor)
        hparams.nas_outlier_mode = args.nas_outlier_mode
        hparams.nas_log_path = str(output_dir / "nas_scaling.jsonl")
    hparams.residual_gain_lambda = float(args.residual_gain_lambda)
    hparams.residual_gain_subject_layers = list(
        args.residual_gain_subject_layers
    )
    hparams.residual_gain_token_scope = "subject_last"
    hparams.residual_gain_prompt_layers = []
    hparams.residual_gain_margin = float(args.residual_gain_margin)
    hparams.residual_gain_loss_type = args.residual_gain_loss_type
    hparams.residual_gain_objective = args.residual_gain_objective
    hparams.residual_gain_alignment_weight = float(
        args.residual_gain_alignment_weight
    )
    hparams.residual_gain_cosine_aux_lambda = float(
        args.residual_gain_cosine_aux_lambda
    )
    hparams.residual_gain_cosine_aux_sharpness = float(
        args.residual_gain_cosine_aux_sharpness
    )
    hparams.residual_gain_efficacy_threshold = float(args.residual_gain_efficacy_threshold)
    hparams.residual_gain_select_best = False
    hparams.residual_gain_selection_threshold = 0.5
    hparams.residual_gain_early_stop_mode = "base"
    hparams.residual_gain_log_path = str(output_dir / "rgr_optimization.jsonl")
    configure_oedit_hparams(args, hparams, condition=analysis_condition, output_dir=output_dir)
    configure_early_attention_hparams(
        args, hparams, method=args.editing_method, output_dir=output_dir
    )
    if bool(getattr(hparams, "early_attention_preservation_enabled", False)):
        Path(hparams.early_attention_preservation_log_path).touch()
    configure_o0_axis_hparams(args, hparams, output_dir=output_dir)
    if bool(getattr(hparams, "o0_axis_preservation_enabled", False)):
        Path(hparams.o0_axis_preservation_log_path).touch()
    if hasattr(hparams, "inner_margin_schedule"):
        hparams.inner_margin_schedule = str(args.inner_margin_schedule)
        hparams.inner_target_margin_threshold = float(
            args.inner_target_margin_threshold
        )
        hparams.inner_margin_refinement_steps = int(
            args.inner_margin_refinement_steps
        )
        hparams.inner_margin_hinge_weight = float(
            args.inner_margin_hinge_weight
        )
        hparams.inner_margin_schedule_log_path = (
            str(output_dir / "inner_margin_schedule.jsonl")
            if args.inner_margin_schedule != "legacy"
            else None
        )
    hparams.sadr_lambda = float(args.sadr_lambda)
    hparams.sadr_attn_layers = list(args.sadr_attn_layers)
    hparams.sadr_efficacy_threshold = float(args.sadr_efficacy_threshold)
    hparams.sadr_log_path = str(output_dir / "sadr_optimization.jsonl")
    active_log_paths = {
        "sadr": hparams.sadr_log_path,
        "encore": hparams.official_baseline_log_path,
        "sphere": hparams.official_baseline_log_path,
        "nas": getattr(hparams, "nas_log_path", None),
        "rgr": hparams.residual_gain_log_path,
        "oedit": getattr(hparams, "oedit_log_path", None),
    }
    for component in sorted(active_components):
        active_log_path = active_log_paths.get(component)
        if active_log_path:
            Path(active_log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(active_log_path).touch()
    if bool(args.context_multikey_enabled):
        Path(hparams.context_multikey_log_path).parent.mkdir(
            parents=True, exist_ok=True
        )
        Path(hparams.context_multikey_log_path).touch()
    if bool(args.key_gaussian_noise_enabled):
        Path(hparams.key_gaussian_noise_log_path).parent.mkdir(
            parents=True, exist_ok=True
        )
        Path(hparams.key_gaussian_noise_log_path).touch()
    if bool(args.tangent_layer_allocation_enabled):
        Path(hparams.tangent_layer_allocation_log_path).parent.mkdir(
            parents=True, exist_ok=True
        )
        Path(hparams.tangent_layer_allocation_log_path).touch()
    if args.inner_margin_schedule != "legacy":
        Path(hparams.inner_margin_schedule_log_path).parent.mkdir(
            parents=True, exist_ok=True
        )
        Path(hparams.inner_margin_schedule_log_path).touch()
    hparams.analysis_artifact_dir = str(output_dir / "edit_artifacts")
    hparams.analysis_capture_inner_gain = bool(args.save_edit_artifacts)
    hparams.analysis_capture_latents = bool(args.save_edit_artifacts)
    if bool(args.capture_virtual_actual):
        if args.editing_method != "AlphaEdit" or list(hparams.layers) != [4, 5, 6, 7, 8] or hparams.fact_token != "subject_last":
            raise ValueError("virtual/actual suite requires AlphaEdit L4-L8 subject_last editing")
        hparams.analysis_capture_virtual_actual = True
    elif hasattr(hparams, "analysis_capture_virtual_actual"):
        hparams.analysis_capture_virtual_actual = False
    hparams.analysis_capture_outer = bool(args.save_edit_artifacts)
    hparams.analysis_gain_subject_layers = list(
        args.residual_gain_subject_layers
    )
    hparams.analysis_gain_prompt_layers = []
    edit_runner.normalize_hparam_paths(hparams, PROJECT_ROOT)
    if args.save_layer_checkpoints and not set(args.save_layer_ids).issubset(
        set(int(layer) for layer in hparams.layers)
    ):
        raise ValueError(
            "--save-layer-ids must be a subset of the editor rewrite layers "
            f"{list(hparams.layers)}"
        )
    compact_parameter_names = (
        rewrite_parameter_names(hparams, args.save_layer_ids)
        if args.save_layer_checkpoints
        else []
    )
    effective_editing_hparams = {
        name: getattr(hparams, name, None)
        for name in (
            "layers",
            "fact_token",
            "clamp_norm_factor",
            "v_num_grad_steps",
            "v_lr",
            "v_loss_layer",
            "v_weight_decay",
            "kl_factor",
            "mom2_adjustment",
            "mom2_update_weight",
            "mom2_dataset",
            "mom2_n_samples",
            "mom2_dtype",
            "nullspace_threshold",
            "L2",
            "model_parallel",
            "bf16",
            "residual_gain_objective",
        )
    }
    effective_editing_hparams["batch_size"] = int(hparams.batch_size)
    effective_editing_hparams.update(oedit_config(hparams))
    effective_editing_hparams.update(early_attention_config(hparams))
    effective_editing_hparams.update(o0_axis_config(hparams))
    if bool(args.context_multikey_enabled):
        effective_editing_hparams.update(
            {
                "context_multikey_enabled": True,
                "context_multikey_source": "existing_context_templates",
                "context_multikey_compression": (
                    "exact_same_target_mean_plus_deviation"
                ),
                "context_multikey_variant_policy": (
                    "all_existing_context_templates_no_subsampling"
                ),
                "context_multikey_expected_group_sizes": [1, 5],
                "context_multikey_expected_num_variants": 6,
                "context_multikey_subsampling": False,
            }
        )
    if bool(args.key_gaussian_noise_enabled):
        effective_editing_hparams.update(
            {
                "key_gaussian_noise_enabled": True,
                "key_gaussian_noise_mode": (
                    "one_canonical_key_plus_relative_gaussian_noise"
                ),
                "key_gaussian_noise_source": "canonical_unprefixed_prompt",
                "key_gaussian_noise_keys_per_fact": 1,
                "key_gaussian_noise_context_key_averaging": False,
                "key_gaussian_noise_generated_prefix_key_forwards": 0,
                "key_gaussian_noise_relative_std": float(
                    args.key_gaussian_noise_relative_std
                ),
                "key_gaussian_noise_seed": key_gaussian_noise_seed,
            }
        )
    if bool(args.tangent_layer_allocation_enabled):
        effective_editing_hparams.update(
            {
                "tangent_layer_allocation_enabled": True,
                "tangent_layer_allocation_mode": (
                    "strict_joint_local_tangent_jacobian_identity"
                ),
                "tangent_layer_allocation_rcond": (
                    args.tangent_layer_allocation_rcond
                ),
                "tangent_layer_allocation_max_relative_slack": (
                    args.tangent_layer_allocation_max_relative_slack
                ),
                "tangent_layer_allocation_batch_size": 1,
            }
        )
    if args.inner_margin_schedule != "legacy":
        effective_editing_hparams.update(
            {
                "inner_margin_schedule": str(args.inner_margin_schedule),
                "inner_target_margin_threshold": float(
                    args.inner_target_margin_threshold
                ),
                "inner_margin_refinement_steps": int(
                    args.inner_margin_refinement_steps
                ),
                "inner_margin_hinge_weight": float(
                    args.inner_margin_hinge_weight
                ),
                "inner_margin_metric": (
                    "minimum_target_vs_best_alternative_margin_across_"
                    "rewrite_contexts_and_target_tokens"
                ),
            }
        )
    if "sphere" in active_components:
        effective_editing_hparams.update(
            {
                "sphere_enabled": True,
                "sphere_beta": float(hparams.sphere_beta),
                "sphere_alpha": float(hparams.sphere_alpha),
            }
        )

    requests_path = output_dir / "requests.json"
    analysis_requests_path = output_dir / "analysis_requests.json"
    source_requests: List[Dict[str, Any]] | None = None
    if edit_order_permutation is not None:
        source_requests, source_record_count = edit_runner.load_requests(
            args.data_path
        )
        if source_record_count != int(edit_order_metadata["source_record_count"]):
            raise ValueError(
                "Edit-order source normalization changed the record count: "
                f"manifest={edit_order_metadata['source_record_count']}, "
                f"runner={source_record_count}"
            )
        if len(source_requests) != source_record_count:
            raise ValueError(
                "Edit-order manifests require every source JSON record to be "
                "accepted by the trajectory request loader"
            )
        expected_ordered_requests = apply_manifest_order(
            list(source_requests[: args.sample_size]),
            edit_order_permutation,
        )
    if requests_path.exists():
        requests, _ = edit_runner.load_requests(str(requests_path))
        if len(requests) != args.sample_size:
            raise ValueError(
                f"Existing {requests_path} contains {len(requests)} requests, expected {args.sample_size}"
            )
        if edit_order_permutation is not None and analysis_request_fingerprint(
            requests
        ) != analysis_request_fingerprint(expected_ordered_requests):
            raise ValueError(
                f"Existing {requests_path} does not match --edit-order-file"
            )
        if analysis_requests_path.exists():
            analysis_requests, _ = edit_runner.load_requests(str(analysis_requests_path))
        elif source_requests is not None:
            analysis_requests = list(
                source_requests[
                    : min(args.locality_eval_prompts, len(source_requests))
                ]
            )
        else:
            analysis_requests = requests[: min(args.locality_eval_prompts, len(requests))]
    else:
        if source_requests is None:
            source_requests, _ = edit_runner.load_requests(args.data_path)
        all_requests = source_requests
        requests = list(all_requests[: args.sample_size])
        analysis_requests = list(
            all_requests[: min(args.locality_eval_prompts, len(all_requests))]
        )
        if edit_order_permutation is not None:
            requests = expected_ordered_requests
        elif args.edit_order == "shuffle":
            random.Random(args.seed).shuffle(requests)

    # Runs started before online post-update/continuous scoring existed must
    # remain resumable with their original fingerprint.  New diagnostics are
    # intentionally not retrofitted into a partially completed trajectory;
    # compact checkpoints support those post-hoc analyses separately.
    preexisting_config_path = output_dir / "run_config.json"
    preexisting_config: Dict[str, Any] | None = None
    if preexisting_config_path.is_file():
        preexisting_config = read_json(preexisting_config_path)
        if "track_post_update" not in preexisting_config and bool(
            args.track_post_update
        ):
            print(
                "[compat] existing trajectory predates paired post-update "
                "tracking; disabling it for this resume"
            )
            args.track_post_update = 0
        if "continuous_behavior" not in preexisting_config:
            if bool(args.continuous_behavior):
                print(
                    "[compat] existing trajectory predates continuous "
                    "behavior scoring; disabling it for this resume"
                )
            args.continuous_behavior = 0

    existing_effective_hparams = (
        preexisting_config.get("effective_editing_hparams", {})
        if preexisting_config is not None
        else {}
    )
    record_alignment_weight = (
        float(args.residual_gain_alignment_weight) != 1.0
        or (
            preexisting_config is not None
            and "residual_gain_alignment_weight" in preexisting_config
        )
        or "residual_gain_alignment_weight" in existing_effective_hparams
    )
    if record_alignment_weight:
        effective_editing_hparams["residual_gain_alignment_weight"] = float(
            args.residual_gain_alignment_weight
        )
    record_cosine_aux = (
        float(args.residual_gain_cosine_aux_lambda) != 0.0
        or float(args.residual_gain_cosine_aux_sharpness) != 1.0
        or (
            preexisting_config is not None
            and "residual_gain_cosine_aux_lambda" in preexisting_config
        )
        or "residual_gain_cosine_aux_lambda" in existing_effective_hparams
    )
    if record_cosine_aux:
        effective_editing_hparams.update(
            {
                "residual_gain_cosine_aux_lambda": float(
                    args.residual_gain_cosine_aux_lambda
                ),
                "residual_gain_cosine_aux_sharpness": float(
                    args.residual_gain_cosine_aux_sharpness
                ),
            }
        )

    # Resolve EOS before fingerprinting.  On a resumed run this is idempotent.
    # The tokenizer is loaded with the editor below; until then use the saved
    # requests as the exact trajectory definition if they already exist.
    preliminary_config = {
        # Preserve the original MEMIT fingerprint schema so an experiment
        # started before AlphaEdit support remains exactly resumable.
        "experiment": (
            "memit_projected_residual_gain_trajectory"
            if args.editing_method == "MEMIT"
            else "alphaedit_projected_residual_gain_trajectory"
        ),
        "hparams_path": str(Path(args.hparams_path).resolve()),
        "hparams_sha256": hashlib.sha256(
            Path(args.hparams_path).read_bytes()
        ).hexdigest(),
        "data_path": str(Path(args.data_path).resolve()),
        "model_name": args.model_name,
        "sample_size": args.sample_size,
        "batch_size": 1,
        "steps": args.steps,
        "layers": args.layers,
        "window_layers": args.window_layers,
        "contexts": args.contexts,
        "probe_prompts": args.probe_prompts,
        "jacobian_prompts": args.jacobian_prompts,
        "locality_eval_prompts": args.locality_eval_prompts,
        "pca_rank": args.pca_rank,
        "branch_analysis": bool(args.branch_analysis),
        "interaction_analysis": bool(args.interaction_analysis),
        "interaction_layers": list(args.interaction_layers),
        "max_length": args.max_length,
        "eval_batch_size": args.eval_batch_size,
        "seed": args.seed,
        "append_eos_to_target": bool(args.append_eos_to_target),
        "residual_gain_regularization": "rgr" in active_components,
        "residual_gain_lambda": float(args.residual_gain_lambda),
        "residual_gain_subject_layers": list(
            args.residual_gain_subject_layers
        ),
        "residual_gain_margin": float(args.residual_gain_margin),
        "residual_gain_loss_type": args.residual_gain_loss_type,
        "residual_gain_objective": args.residual_gain_objective,
        "save_layer_checkpoints": list(args.save_layer_checkpoints),
        "save_layer_ids": list(args.save_layer_ids),
        "save_edit_artifacts": bool(args.save_edit_artifacts),
        "effective_editing_hparams": effective_editing_hparams,
    }
    preliminary_config.update(early_attention_config(hparams))
    preliminary_config.update(o0_axis_config(hparams))
    if bool(args.capture_virtual_actual):
        preliminary_config["capture_virtual_actual"] = {
            "enabled": True, "schema_version": 1, "positions": ["subject_last", "prompt_last"],
            "layers": list(range(32)), "target_layer": 8,
            "save_vectors": bool(args.virtual_actual_save_vectors),
            "virtual": "pre-edit weights plus selected returned target replacing subject O8",
            "actual": "committed post-edit weights without intervention",
            "prompt": "canonical current-edit raw rewrite prompt without generated prefix or target suffix",
            "artifact_directory": "edit_artifacts/virtual_actual",
        }
    # ``rho_minus_one`` with the squared one-sided loss is the project's
    # HiddenNorm (HN) formulation.  Older RGR trajectories intentionally keep
    # their byte-compatible config schema; new HN runs receive an explicit
    # formulation contract so a plug-in result cannot be mislabeled from the
    # historical ``rgr`` implementation component alone.
    is_hidden_norm_formulation = (
        "rgr" in active_components
        and args.residual_gain_objective == "rho_minus_one"
        and args.residual_gain_loss_type == "positive_squared"
    )
    if is_hidden_norm_formulation:
        preliminary_config["hidden_norm_formulation"] = {
            "name": "hidden_output_norm_one_sided_squared_hinge",
            "implementation_component": "rgr",
            "lambda": float(args.residual_gain_lambda),
            "objective": args.residual_gain_objective,
            "loss_type": args.residual_gain_loss_type,
            "token_scope": "subject_last",
            "subject_layers": list(args.residual_gain_subject_layers),
            "margin": float(args.residual_gain_margin),
            "efficacy_threshold": float(
                args.residual_gain_efficacy_threshold
            ),
            "select_best": False,
            "selection_threshold": 0.5,
            "early_stop_mode": "base",
        }
    if bool(args.context_multikey_enabled):
        preliminary_config["context_multikey"] = {
            "enabled": True,
            "source": "existing_context_templates",
            "new_prompts_added": False,
            "compression": "exact_same_target_mean_plus_deviation",
            "cache_semantics": "weighted_raw_context_key_gram",
            "variant_policy": "all_existing_context_templates_no_subsampling",
            "context_group_sizes": [1, 5],
            "num_original_contexts": 1,
            "num_generated_prefix_contexts": 5,
            "num_context_variants": 6,
            "subsampling": False,
            "log_path": "context_multikey.jsonl",
        }
    if bool(args.key_gaussian_noise_enabled):
        preliminary_config["key_gaussian_noise"] = {
            "enabled": True,
            "mode": "one_canonical_key_plus_relative_gaussian_noise",
            "key_source": "canonical_unprefixed_prompt",
            "canonical_prompt_forwards_per_fact_per_layer": 1,
            "generated_prefix_key_forwards": 0,
            "keys_per_fact": 1,
            "context_key_averaging": False,
            "perturbation_distribution": "isotropic_gaussian",
            "relative_std": float(args.key_gaussian_noise_relative_std),
            "base_seed": key_gaussian_noise_seed,
            "seed_derivation": (
                "sha256(base_seed,layer,case_id,edit_index,prompt_template,subject)"
            ),
            "target_formation": (
                f"native_{args.editing_method.lower()}_compute_z_unchanged"
            ),
            "log_path": "key_gaussian_noise.jsonl",
        }
    if bool(args.tangent_layer_allocation_enabled):
        preliminary_config["tangent_layer_allocation"] = {
            "enabled": True,
            "mode": "strict_joint_local_tangent_jacobian_identity",
            "reference": "clean_subject_token_block_output_per_edited_layer",
            "target": "initial_z_star_minus_current_z",
            "transport_assumption": "J_l ~= I",
            "terminal_slack_cleanup": False,
            "batch_size": 1,
            "rcond": args.tangent_layer_allocation_rcond,
            "max_relative_slack": (
                args.tangent_layer_allocation_max_relative_slack
            ),
            "log_path": "tangent_layer_allocation.jsonl",
        }
    if args.fixed_probe_path:
        fixed_probe_path = Path(args.fixed_probe_path).expanduser().resolve()
        preliminary_config["fixed_probe_tracking"] = {
            "measurement": "observer_only_ordinary_model_forward",
            "path": str(fixed_probe_path),
            "sha256": fixed_probe_file_sha256(fixed_probe_path),
            "manifest_path": (
                str(Path(args.fixed_probe_manifest).expanduser().resolve())
                if args.fixed_probe_manifest
                else None
            ),
            "manifest_sha256": (
                fixed_probe_file_sha256(
                    Path(args.fixed_probe_manifest).expanduser().resolve()
                )
                if args.fixed_probe_manifest
                else None
            ),
            "prompts_per_family": int(args.fixed_probe_prompts),
            "interval": int(args.fixed_probe_interval),
            "batch_size": int(args.fixed_probe_batch_size),
            "contexts": list(args.fixed_probe_contexts),
            "vector_interval": int(args.fixed_probe_vector_interval),
            "boundary_layers": [8, 9],
        }
    if record_alignment_weight:
        preliminary_config["residual_gain_alignment_weight"] = float(
            args.residual_gain_alignment_weight
        )
    if record_cosine_aux:
        preliminary_config.update(
            {
                "residual_gain_cosine_aux_lambda": float(
                    args.residual_gain_cosine_aux_lambda
                ),
                "residual_gain_cosine_aux_sharpness": float(
                    args.residual_gain_cosine_aux_sharpness
                ),
            }
        )
    if preexisting_config is None or "continuous_behavior" in preexisting_config:
        preliminary_config["continuous_behavior"] = bool(args.continuous_behavior)
    if bool(args.track_post_update):
        preliminary_config.update(
            {
                "track_post_update": True,
                "post_update_layers": list(args.post_update_layers),
                "post_update_nodes": list(args.post_update_nodes),
                "post_update_behavior": (
                    "current_request_teacher_forced_target_likelihood"
                ),
            }
        )
    if args.analysis_condition is not None:
        preliminary_config["analysis_condition"] = analysis_condition
    if "sadr" in active_components:
        preliminary_config.update(
            {
                "sadr_regularization": True,
                "sadr_lambda": float(args.sadr_lambda),
                "sadr_attn_layers": list(args.sadr_attn_layers),
                "sadr_efficacy_threshold": float(args.sadr_efficacy_threshold),
            }
        )
    if "encore" in active_components:
        preliminary_config.update(
            {
                "encore_enabled": True,
                "encore_mpes_top1_steps": int(args.encore_mpes_top1_steps),
                "encore_mpes_exclude_first_context": bool(
                    args.encore_mpes_exclude_first_context
                ),
                "encore_norm_lambda": float(hparams.encore_norm_lambda),
            }
        )
    if "sphere" in active_components:
        preliminary_config.update(
            {
                "sphere_enabled": True,
                "sphere_beta": float(hparams.sphere_beta),
                "sphere_alpha": float(hparams.sphere_alpha),
            }
        )
    if "nas" in active_components:
        preliminary_config.update(
            {
                "nas_enabled": True,
                "nas_anchor_path": str(Path(args.nas_anchor_path).resolve()),
                "nas_outlier_factor": float(args.nas_outlier_factor),
                "nas_outlier_mode": args.nas_outlier_mode,
            }
        )
    if "rgr" in active_components and args.analysis_condition is not None:
        preliminary_config.update(
            {
                "residual_gain_efficacy_threshold": float(
                    args.residual_gain_efficacy_threshold
                ),
            }
        )
    if args.editing_method != "MEMIT":
        preliminary_config["editing_method"] = args.editing_method
    if edit_order_metadata is not None:
        preliminary_config["edit_order"] = "manifest"
        preliminary_config["edit_order_manifest"] = edit_order_metadata
    elif args.edit_order != "prefix":
        preliminary_config["edit_order"] = args.edit_order
    if explicit_analysis_seed:
        preliminary_config["analysis_seed"] = args.analysis_seed
    if edit_order_metadata is not None or args.edit_order != "prefix" or explicit_analysis_seed:
        preliminary_config["analysis_request_fingerprint"] = analysis_request_fingerprint(
            analysis_requests
        )
    if bool(args.covariance_analysis):
        preliminary_config["covariance_analysis"] = True

    # If every step is complete, config validation and aggregation do not need
    # to load an 8B model.  The final request fingerprint is stored in config.
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        existing_config = read_json(config_path)
        expected_without_request = dict(existing_config)
        expected_without_request.pop("fingerprint", None)
        expected_request_fingerprint = expected_without_request.pop("request_fingerprint", None)
        if expected_without_request != preliminary_config:
            raise ValueError(
                f"Existing run config at {config_path} does not match this invocation. "
                "Choose a new OUTPUT_DIR."
            )
        if expected_request_fingerprint != request_fingerprint(requests):
            raise ValueError("Existing requests.json does not match the run-config request fingerprint")
        fingerprint = str(existing_config["fingerprint"])
        if all(step_is_complete(output_dir, step, fingerprint) for step in args.steps):
            print("[resume] every requested step is already complete; no model is loaded")
            aggregate_outputs(
                output_dir,
                args.steps,
                fingerprint,
                layout,
                args.window_layers,
                args.editing_method,
                bool(args.covariance_analysis),
            )
            if bool(args.save_edit_artifacts) or bool(args.track_post_update):
                aggregate_edit_artifacts(output_dir)
            return

    print("[model] loading Base model through EasyEdit")
    editor = edit_runner.BaseEditor.from_hparams(hparams)
    editor.model.eval()
    editor.model.config.use_cache = False
    if bool(args.capture_virtual_actual) and int(editor.model.config.num_hidden_layers) != 32:
        raise ValueError("virtual/actual suite requires the configured 32-layer model")
    base_checkpoint_parameters = (
        snapshot_parameters(
            editor.model,
            compact_parameter_names,
        )
        if args.save_layer_checkpoints
        else {}
    )
    if bool(args.append_eos_to_target):
        requests, eos_token, appended = edit_runner.append_eos_to_request_targets(
            requests, editor.tok
        )
        analysis_requests, _, analysis_appended = edit_runner.append_eos_to_request_targets(
            analysis_requests, editor.tok
        )
        print(f"[data] EOS token {eos_token!r}, appended to {appended} targets")
        if analysis_appended:
            print(f"[data] EOS appended to {analysis_appended} fixed analysis requests")
    verify_oedit_reference_requests(hparams, requests, analysis_requests)
    if not requests_path.exists():
        atomic_json(requests_path, requests)
    if (
        edit_order_metadata is not None
        or args.edit_order != "prefix"
        or explicit_analysis_seed
    ) and not analysis_requests_path.exists():
        atomic_json(analysis_requests_path, analysis_requests)

    config = {**preliminary_config, "request_fingerprint": request_fingerprint(requests)}
    fingerprint = validate_or_write_config(config_path, config)
    missing_steps = [
        step for step in args.steps if not step_is_complete(output_dir, step, fingerprint)
    ]
    if not missing_steps:
        aggregate_outputs(
            output_dir,
            args.steps,
            fingerprint,
            layout,
            args.window_layers,
            args.editing_method,
            bool(args.covariance_analysis),
        )
        if bool(args.save_edit_artifacts) or bool(args.track_post_update):
            aggregate_edit_artifacts(output_dir)
        return
    last_needed_step = max(missing_steps)
    resume_checkpoint_path: Path | None = None
    resume_edit_count = 0
    resume_manifest: Dict[str, Any] | None = None
    if args.resume_layer_checkpoint:
        resume_checkpoint_path = (
            Path(args.resume_layer_checkpoint).expanduser().resolve()
        )
        resume_manifest = read_json(resume_checkpoint_path.with_suffix(".json"))
        resume_edit_count = int(resume_manifest.get("edit_count", -1))
        if resume_edit_count <= 0:
            raise ValueError(
                "Resume checkpoint manifest has no positive edit_count: "
                f"{resume_checkpoint_path.with_suffix('.json')}"
            )
        if resume_edit_count >= last_needed_step:
            raise ValueError(
                f"Resume edit count {resume_edit_count} must be smaller than "
                f"the last missing step {last_needed_step}"
            )
        if resume_edit_count not in set(args.steps):
            raise ValueError(
                f"Resume edit count {resume_edit_count} is absent from --steps"
            )
        incomplete_prefix = [
            step for step in args.steps
            if step <= resume_edit_count
            and not step_is_complete(output_dir, step, fingerprint)
        ]
        if incomplete_prefix:
            raise ValueError(
                "Cannot start from a parameter checkpoint while earlier "
                f"analyses are incomplete: {incomplete_prefix}"
            )
        expected_metadata = {
            "editing_method": args.editing_method,
            "base_model": args.model_name,
            "run_fingerprint": fingerprint,
            "request_fingerprint": request_fingerprint(requests),
        }
        mismatched_metadata = {
            key: {
                "checkpoint": resume_manifest.get(key),
                "expected": expected,
            }
            for key, expected in expected_metadata.items()
            if resume_manifest.get(key) != expected
        }
        checkpoint_names = list(resume_manifest.get("parameter_names", []))
        if set(checkpoint_names) != set(compact_parameter_names):
            mismatched_metadata["parameter_names"] = {
                "checkpoint": checkpoint_names,
                "expected": compact_parameter_names,
            }
        if mismatched_metadata:
            raise ValueError(
                "Resume checkpoint metadata does not match this trajectory: "
                f"{mismatched_metadata}"
            )
        print(
            "[resume-checkpoint] validated "
            f"edit_count={resume_edit_count} path={resume_checkpoint_path}"
        )
        if active_log_path:
            prepare_append_log_for_checkpoint_resume(
                Path(active_log_path).expanduser().resolve(),
                requests,
                resume_edit_count,
            )
    if (
        resume_checkpoint_path is None
        and preexisting_config is not None
        and active_log_path
    ):
        # A replay resume reconstructs the edited model from Base, so its
        # append-only optimizer trace must also be reconstructed from edit 1.
        # Back up and clear any completed/partial prefix rather than appending
        # duplicate inner-loop records for the replayed requests.
        prepare_append_log_for_checkpoint_resume(
            Path(active_log_path).expanduser().resolve(),
            requests,
            0,
        )
    if resume_checkpoint_path is None:
        print(
            f"[resume] missing analyses={missing_steps}; edits will be replayed "
            f"through {last_needed_step}"
        )
    else:
        print(
            f"[resume] missing analyses={missing_steps}; loading edit "
            f"{resume_edit_count} and continuing through {last_needed_step}"
        )

    locality_probe_path = analysis_requests_path if analysis_requests_path.exists() else requests_path
    probes_by_group = load_context_probes(
        layout,
        editor.tok,
        args,
        requests_path,
        locality_probe_path,
    )
    fixed_probe_tracker: FixedProbeTracker | None = None
    if args.fixed_probe_path:
        fixed_panel_dir = output_dir / "fixed_probe_panel"
        fixed_panel_dir.mkdir(parents=True, exist_ok=True)
        fixed_probe_path = Path(args.fixed_probe_path).expanduser().resolve()
        copied_requests = fixed_panel_dir / "fixed_probe_requests.json"
        if not copied_requests.exists():
            shutil.copy2(fixed_probe_path, copied_requests)
        if args.fixed_probe_manifest:
            copied_manifest = fixed_panel_dir / "fixed_probe_manifest.json"
            if not copied_manifest.exists():
                shutil.copy2(
                    Path(args.fixed_probe_manifest).expanduser().resolve(),
                    copied_manifest,
                )
        fixed_probe_tracker = FixedProbeTracker(
            model=editor.model,
            tokenizer=editor.tok,
            probe_path=fixed_probe_path,
            output_root=output_dir / "edit_artifacts" / "fixed_probe",
            count=args.fixed_probe_prompts,
            max_length=args.max_length,
            batch_size=args.fixed_probe_batch_size,
            contexts=args.fixed_probe_contexts,
            vector_interval=args.fixed_probe_vector_interval,
        )
        print(
            "[fixed-probe] observing Base state: "
            f"families={list(fixed_probe_tracker.panel)} "
            f"n={args.fixed_probe_prompts} interval={args.fixed_probe_interval}"
        )
        fixed_probe_tracker.observe(0, force_vector_save=True)
    bases_by_group, base_features_by_group = prepare_bases_and_features(
        editor.model,
        probes_by_group,
        layout,
        args.layers,
        args.pca_rank,
        output_dir,
    )
    base_downproj_weights: Dict[int, torch.Tensor] = {}
    base_downproj_features_by_group: Dict[
        str, Dict[Tuple[str, int, str], np.ndarray]
    ] = {}
    if bool(args.interaction_analysis):
        decoder = model_layers(editor.model)
        for layer in args.interaction_layers:
            parameter_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            if parameter_name in base_checkpoint_parameters:
                base_downproj_weights[layer] = base_checkpoint_parameters[
                    parameter_name
                ]
            else:
                base_downproj_weights[layer] = (
                    decoder[layer]
                    .mlp.down_proj.weight.detach()
                    .to(device="cpu")
                    .clone()
                )
        for group, probes in probes_by_group.items():
            feature_path = (
                output_dir
                / "base_reference"
                / group
                / "base_downproj_features.npz"
            )
            if feature_path.exists():
                base_downproj_features_by_group[group] = (
                    load_downproj_features(
                        feature_path,
                        layout[group],
                        args.interaction_layers,
                    )
                )
                print(
                    f"[base-reference] loaded {group} down-projection features"
                )
            else:
                print(
                    f"[base-reference] collecting {group} down-projection features"
                )
                captured = collect_downproj_features(
                    editor.model,
                    probes,
                    args.interaction_layers,
                    layout[group],
                )
                save_downproj_features(feature_path, captured)
                base_downproj_features_by_group[group] = captured

    locality_requests = analysis_requests[: min(args.locality_eval_prompts, len(analysis_requests))]
    base_locality_path = output_dir / "base_reference" / "base_locality_outputs.json"
    if base_locality_path.exists():
        base_locality_outputs = unflatten_locality_outputs(read_json(base_locality_path))
        print("[base-reference] loaded cached locality outputs")
    else:
        locality_items = collect_group_items(locality_requests, "locality")
        base_locality_outputs = compute_locality_outputs(
            model=editor.model,
            model_name=editor.model_name,
            hparams=editor.hparams,
            tokenizer=editor.tok,
            locality_items=locality_items,
            device=editor.hparams.device,
            batch_size=args.eval_batch_size,
        )
        atomic_json(base_locality_path, flatten_locality_outputs(base_locality_outputs))

    if resume_checkpoint_path is not None:
        # MEMIT/AlphaEdit generate and cache their context templates on the
        # first edit. Prime that cache on the unedited Base model before
        # applying the checkpoint, matching an uninterrupted trajectory.
        if args.editing_method not in {"MEMIT", "AlphaEdit"}:
            raise ValueError(
                "Checkpoint resume has no context-cache handler for "
                f"{args.editing_method}"
            )
        # This workspace can load EasyEdit through either ``easyeditor`` or
        # ``EasyEdit.easyeditor``. Inject into the exact module globals used by
        # this BaseEditor instance instead of guessing the import alias.
        editor_globals = editor.apply_algo.__globals__
        get_context_templates = editor_globals.get("get_context_templates")
        if get_context_templates is None:
            raise RuntimeError(
                "Active editor handler exposes no get_context_templates"
            )
        if args.resume_context_templates_json:
            template_payload = read_json(
                Path(args.resume_context_templates_json).expanduser().resolve()
            )
            context_templates = template_payload.get(
                "context_templates",
                template_payload,
            )
            if (
                not isinstance(context_templates, list)
                or not context_templates
                or any(not isinstance(group, list) for group in context_templates)
                or any(
                    not isinstance(template, str) or "{}" not in template
                    for group in context_templates
                    for template in group
                )
            ):
                raise ValueError(
                    "Resume context-template JSON has an invalid structure"
                )
            editor_globals["CONTEXT_TEMPLATES_CACHE"] = context_templates
            print(
                "[resume-checkpoint] restored context templates from "
                f"{Path(args.resume_context_templates_json).resolve()} "
                f"into {editor.apply_algo.__module__}"
            )
        else:
            editor_globals["CONTEXT_TEMPLATES_CACHE"] = None
            get_context_templates(editor.model, editor.tok)
            print(
                "[resume-checkpoint] regenerated context templates on Base; "
                "provide --resume-context-templates-json for exact template "
                "identity"
            )
        loaded_manifest = load_edited_parameter_checkpoint(
            editor.model,
            resume_checkpoint_path,
        )
        if int(loaded_manifest.get("edit_count", -1)) != resume_edit_count:
            raise ValueError(
                "Loaded checkpoint edit_count changed between manifest and "
                f"payload: manifest={resume_edit_count}, "
                f"payload={loaded_manifest.get('edit_count')}"
            )
        editor.model.eval()
        editor.model.config.use_cache = False
        print(
            "[resume-checkpoint] applied cumulative parameter delta; "
            f"next edit={resume_edit_count + 1}"
        )
        if fixed_probe_tracker is not None:
            fixed_probe_tracker.observe(
                resume_edit_count,
                force_vector_save=resume_edit_count in set(args.steps),
            )

    base_matrices_by_group: Dict[str, Dict[Tuple[str, int], List[np.ndarray]]] = {}
    base_covariance_by_group: Dict[
        str, Dict[Tuple[str, int], Dict[str, Any]]
    ] = {}

    def run_analysis(edit_count: int) -> None:
        if step_is_complete(output_dir, edit_count, fingerprint):
            print(f"[resume] step {edit_count}: completed analysis preserved")
            return
        label = f"{args.editing_method}-edit-{edit_count}"
        current_dir = step_dir(output_dir, edit_count)
        compact_checkpoint_files: List[str] = []
        if edit_count in set(args.save_layer_checkpoints):
            checkpoint_path = current_dir / "edited_parameter_deltas.pt"
            print(
                f"[save] step {edit_count}: compact edited parameters "
                f"for layers {args.save_layer_ids}"
            )
            save_parameter_delta_checkpoint(
                editor.model,
                checkpoint_path,
                base_parameters=base_checkpoint_parameters,
                parameter_names=compact_parameter_names,
                metadata={
                    "editing_method": args.editing_method,
                    "base_model": args.model_name,
                    "edit_count": edit_count,
                    "rewrite_layers": list(args.save_layer_ids),
                    "rewrite_module_tmp": hparams.rewrite_module_tmp,
                    "run_fingerprint": fingerprint,
                    "request_fingerprint": request_fingerprint(requests),
                    "hparams_path": str(Path(args.hparams_path).resolve()),
                    "checkpoint_semantics": (
                        "Add these cumulative deltas to the named Base-model "
                        "parameters."
                    ),
                },
            )
            compact_checkpoint_files = [
                checkpoint_path.name,
                checkpoint_path.with_suffix(".json").name,
            ]
        evaluation_path = current_dir / "evaluation.json"
        evaluation_complete = False
        if evaluation_path.exists():
            try:
                evaluation_complete = read_json(evaluation_path).get("fingerprint") == fingerprint
            except (OSError, json.JSONDecodeError):
                evaluation_complete = False
        if evaluation_complete:
            print(f"[resume] step {edit_count}: completed evaluation preserved")
        else:
            print(f"[analysis] step {edit_count}: evaluating locality and rewrite efficacy")
            evaluation = evaluate_step(
                edit_count,
                editor.model,
                editor.model_name,
                editor.hparams,
                editor.tok,
                requests,
                locality_requests,
                base_locality_outputs,
                args.eval_batch_size,
                bool(args.continuous_behavior),
            )
            evaluation["fingerprint"] = fingerprint
            atomic_json(evaluation_path, evaluation)

        for group, positions in layout.items():
            group_dir = current_dir / group
            group_prompts = min(args.jacobian_prompts, len(probes_by_group[group]))
            if component_is_complete(group_dir, fingerprint, edit_count):
                print(f"[resume] step {edit_count}: completed {group} analysis preserved")
                matrices = (
                    load_projected_matrices(
                        group_dir / "projected_jacobians.npz",
                        label,
                        positions,
                        args.layers,
                        group_prompts,
                    )
                    if edit_count == 0
                    else {}
                )
                covariance_lookup = (
                    load_base_covariance_rows(group_dir / "covariance_metrics.csv")
                    if edit_count == 0 and bool(args.covariance_analysis)
                    else {}
                )
            else:
                print(f"[analysis] step {edit_count}: {group}/{','.join(positions)}")
                matrices, covariance_lookup = analyze_group(
                    edit_count,
                    label,
                    group,
                    positions,
                    editor.model,
                    probes_by_group[group],
                    args.layers,
                    bases_by_group[group],
                    base_features_by_group[group],
                    base_matrices_by_group.get(group),
                    bool(args.interaction_analysis),
                    args.interaction_layers,
                    base_downproj_features_by_group.get(group, {}),
                    base_downproj_weights,
                    bool(args.branch_analysis),
                    bool(args.covariance_analysis),
                    base_covariance_by_group.get(group),
                    group_prompts,
                    group_dir,
                )
                required_group_files = [
                    "jacobian_sample_metrics.csv",
                    "jacobian_summary_metrics.csv",
                    "projected_jacobians.npz",
                    "per_layer_proxy_metrics.csv",
                ]
                if bool(args.branch_analysis):
                    required_group_files.extend(
                        [
                            "branch_prompt_metrics.csv",
                            "branch_summary_metrics.csv",
                            "branch_write_vectors.npz",
                        ]
                    )
                if bool(args.interaction_analysis):
                    required_group_files.extend(
                        [
                            "feature_update_prompt_metrics.csv",
                            "feature_update_summary_metrics.csv",
                            "downproj_features.npz",
                            "downproj_feature_deltas.npz",
                        ]
                    )
                if bool(args.covariance_analysis):
                    required_group_files.extend(
                        ["covariance_metrics.csv", "covariance_spectra.npz"]
                    )
                atomic_json(
                    group_dir / COMPLETION_FILE,
                    {
                        "fingerprint": fingerprint,
                        "edit_count": edit_count,
                        "context": group,
                        "required_files": required_group_files,
                    },
                )
            if edit_count == 0:
                base_matrices_by_group[group] = matrices
                if bool(args.covariance_analysis):
                    base_covariance_by_group[group] = covariance_lookup
        required_step_filenames = [
            "complete.json",
            "jacobian_sample_metrics.csv",
            "jacobian_summary_metrics.csv",
            "projected_jacobians.npz",
            "per_layer_proxy_metrics.csv",
        ]
        if bool(args.branch_analysis):
            required_step_filenames.extend(
                [
                    "branch_prompt_metrics.csv",
                    "branch_summary_metrics.csv",
                    "branch_write_vectors.npz",
                ]
            )
        if bool(args.interaction_analysis):
            required_step_filenames.extend(
                [
                    "feature_update_prompt_metrics.csv",
                    "feature_update_summary_metrics.csv",
                    "downproj_features.npz",
                    "downproj_feature_deltas.npz",
                ]
            )
        if bool(args.covariance_analysis):
            required_step_filenames.extend(
                ["covariance_metrics.csv", "covariance_spectra.npz"]
            )
        atomic_json(
            current_dir / COMPLETION_FILE,
            {
                "fingerprint": fingerprint,
                "edit_count": edit_count,
                "label": label,
                "completed_at_unix": time.time(),
                "model_checkpoint_saved": False,
                "compact_edited_parameter_checkpoint_saved": bool(
                    compact_checkpoint_files
                ),
                "required_files": [
                    "evaluation.json",
                    *compact_checkpoint_files,
                    *[
                        f"{group}/{filename}"
                        for group in layout
                        for filename in required_step_filenames
                    ],
                ],
            },
        )
        aggregate_outputs(
            output_dir,
            args.steps,
            fingerprint,
            layout,
            args.window_layers,
            args.editing_method,
            bool(args.covariance_analysis),
        )
        if bool(args.save_edit_artifacts) or bool(args.track_post_update):
            aggregate_edit_artifacts(output_dir)

    # Base matrices are required even when step 0 was completed by an earlier
    # invocation.  Load them without touching any completed output.
    if step_is_complete(output_dir, 0, fingerprint):
        for group, positions in layout.items():
            base_matrices_by_group[group] = load_projected_matrices(
                step_dir(output_dir, 0) / group / "projected_jacobians.npz",
                f"{args.editing_method}-edit-0",
                positions,
                args.layers,
                min(args.jacobian_prompts, len(probes_by_group[group])),
            )
            if bool(args.covariance_analysis):
                base_covariance_by_group[group] = load_base_covariance_rows(
                    step_dir(output_dir, 0) / group / "covariance_metrics.csv"
                )
        print("[resume] loaded Base projected Jacobians from completed step 0")
    else:
        run_analysis(0)

    requested_steps = set(args.steps)
    edit_start = resume_edit_count + 1
    for edit_count in range(edit_start, last_needed_step + 1):
        request = requests[edit_count - 1]
        # Match the one-based request identity used by run_rgr_batch.py so a
        # given Gaussian-key arm realizes the same noise in every entrypoint.
        if bool(args.key_gaussian_noise_enabled):
            request["edit_index"] = int(edit_count)
        print(f"[edit {edit_count}/{last_needed_step}] case_id={request.get('case_id')}")
        editor.hparams.analysis_current_edit_index = int(edit_count)
        before_post_update_nodes: Dict[str, np.ndarray] = {}
        before_post_update_behavior: Dict[str, Any] = {}
        post_update_metadata: Dict[str, Any] = {}
        if bool(args.track_post_update):
            before_post_update_nodes, post_update_metadata = capture_boundary_nodes(
                editor.model,
                editor.tok,
                request,
                nodes=args.post_update_nodes,
                max_length=args.max_length,
            )
            before_post_update_behavior = score_request_behavior(
                editor.model,
                editor.tok,
                request,
                max_length=args.max_length,
            )
        virtual_actual_recorder = None
        callback_name = "_analysis_virtual_actual_selected_target_callback"
        old_callback = getattr(editor.hparams, callback_name, None)
        if bool(args.capture_virtual_actual):
            virtual_actual_recorder = VirtualActualEditRecorder(
                output_dir / "edit_artifacts", request=request, edit_index=edit_count,
                condition=analysis_condition, max_length=args.max_length,
                order_manifest_sha256=(edit_order_metadata or {}).get("sha256"),
                run_fingerprint=fingerprint, save_vectors=bool(args.virtual_actual_save_vectors),
            )
            setattr(editor.hparams, callback_name, virtual_actual_recorder.selected_target)
        try:
            edited_model, _ = editor.apply_algo(
                editor.model,
                editor.tok,
                [request],
                editor.hparams,
                copy=False,
                return_orig_weights=False,
                keep_original_weight=False,
            )
        finally:
            if bool(args.capture_virtual_actual):
                if old_callback is None:
                    delattr(editor.hparams, callback_name)
                else:
                    setattr(editor.hparams, callback_name, old_callback)
        editor.model = edited_model
        record_oedit_state(editor.hparams, output_dir, edit_count)
        editor.model.eval()
        editor.model.config.use_cache = False
        if virtual_actual_recorder is not None:
            virtual_actual_recorder.actual(editor.model, editor.tok)
        if bool(args.track_post_update):
            after_post_update_nodes, after_metadata = capture_boundary_nodes(
                editor.model,
                editor.tok,
                request,
                nodes=args.post_update_nodes,
                max_length=args.max_length,
            )
            after_post_update_behavior = score_request_behavior(
                editor.model,
                editor.tok,
                request,
                max_length=args.max_length,
            )
            if after_metadata.get("token_positions") != post_update_metadata.get(
                "token_positions"
            ):
                raise RuntimeError(
                    "Tokenizer positions changed across a parameter-only update"
                )
            save_paired_post_update_artifact(
                output_dir / "edit_artifacts",
                edit_index=edit_count,
                request=request,
                before_nodes=before_post_update_nodes,
                after_nodes=after_post_update_nodes,
                node_metadata=post_update_metadata,
                before_behavior=before_post_update_behavior,
                after_behavior=after_post_update_behavior,
            )
        if fixed_probe_tracker is not None and (
            edit_count % args.fixed_probe_interval == 0
            or edit_count == last_needed_step
        ):
            print(
                f"[fixed-probe] edit={edit_count} "
                f"n={args.fixed_probe_prompts}"
            )
            fixed_probe_tracker.observe(
                edit_count,
                force_vector_save=edit_count in requested_steps,
            )
        if edit_count in requested_steps:
            run_analysis(edit_count)

    aggregate_outputs(
        output_dir,
        args.steps,
        fingerprint,
        layout,
        args.window_layers,
        args.editing_method,
        bool(args.covariance_analysis),
    )
    if bool(args.save_edit_artifacts) or bool(args.track_post_update):
        aggregate_edit_artifacts(output_dir)
    print(f"[done] trajectory outputs written to {output_dir}")
    print("[done] no full model checkpoint was saved")
    if args.save_layer_checkpoints:
        print(
            "[done] compact edited-parameter checkpoints saved at edits "
            f"{args.save_layer_checkpoints} for layers {args.save_layer_ids}"
        )


if __name__ == "__main__":
    main()
