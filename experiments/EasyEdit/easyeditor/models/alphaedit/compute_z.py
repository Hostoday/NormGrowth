from __future__ import annotations

import hashlib
import json
import math
import os
import re
from copy import copy
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome import repr_tools
from ...util import nethook
from ...util.oedit import get_oedit_regularizer
from ...util.residual_gain_regularization import (
    ResidualGainRegularizer,
    append_rgr_record,
)
from ...util.edit_analysis_artifacts import (
    edit_identity,
    record_inner_gain_step,
    save_latent_artifact,
)
from ...util.early_attention_preservation import (
    EarlyAttentionPreservationRegularizer,
    early_attention_projection_loss,
)
from ...util.o0_axis_preservation import O0AxisPreserver
from ...util.official_editing_baselines import (
    MPESState,
    append_official_baseline_record,
    mpes_observe,
)
from ...util.sadr_regularization import (
    detach_attention_tensors,
    official_sadr_attention_kl_loss,
)
from ...util.norm_anchor_scaling import apply_norm_anchor_scaling
from ...util.inner_margin_schedule import (
    target_margin_stats,
    validate_inner_margin_schedule,
)

from .AlphaEdit_hparams import AlphaEditHyperParams


_DEFAULT_ADAPTIVE_RHO_LADDER = (1.0, 1.05, 1.10, 1.15, 1.25)
_HN_RECOVERY_LOGGED_KEYS: Dict[str, set[Tuple[str, int, str]]] = {}


def _prepare_oedit_regularizer(model, hparams, *, layer=None):
    """Resolve final-writer O-Edit state; observers never advance this state."""
    if not bool(getattr(hparams, "oedit_enabled", False)):
        return None
    if int(getattr(hparams, "batch_size", 1)) != 1:
        raise ValueError("O-Edit currently supports sequential batch_size=1 only")
    if not hparams.layers or hparams.layers[-1] != max(hparams.layers):
        raise ValueError("O-Edit requires an ordered final edited layer")
    final_layer = int(hparams.layers[-1])
    if layer is not None and int(layer) != final_layer:
        raise ValueError("O-Edit penalties are supported only at the final edited layer")
    # Analysis/capture flags intentionally remain allowed. Active interventions
    # change the comparison or invalidate the native actual-update contract.
    conflicts = [name for name in (
        "residual_gain_regularization", "residual_gain_select_best",
        "sadr_regularization", "nse_enabled", "encore_enabled", "sphere_enabled",
        "nas_enabled", "early_attention_preservation_enabled",
        "o0_axis_preservation_enabled", "context_multikey_enabled",
        "key_gaussian_noise_enabled", "tangent_layer_allocation_enabled",
        "hn_recovery_paraphrase_aware", "hn_recovery_adaptive_rho",
    ) if bool(getattr(hparams, name, False))]
    if conflicts:
        raise ValueError(f"Run O-Edit in isolation; disable {conflicts}")
    if (str(getattr(hparams, "inner_margin_schedule", "legacy")) != "legacy"
            or float(getattr(hparams, "residual_gain_target_ratio", 1.0)) != 1.0):
        raise ValueError("O-Edit requires the native inner schedule and no HN recovery")
    if str(getattr(hparams, "residual_gain_early_stop_mode", "base")).lower() not in {"base", "total"}:
        raise ValueError("O-Edit early-stop mode must be base or total")
    module_name = hparams.rewrite_module_tmp.format(final_layer)
    module = nethook.get_module(model, module_name)
    if isinstance(module, torch.nn.Linear):
        output_axis = 0
    elif type(module).__name__ == "Conv1D":
        output_axis = 1
    else:
        raise ValueError("O-Edit requires a Linear or Conv1D rewrite module")
    weight = nethook.get_parameter(model, f"{module_name}.weight")
    return get_oedit_regularizer(
        hparams, model_name=hparams.model_name,
        weight_name=f"{module_name}.weight", weight_shape=tuple(weight.shape),
        output_axis=output_axis,
    )


@dataclass
class _HNRecoverySolveResult:
    target: torch.Tensor
    target_init: torch.Tensor
    optimizer_delta: torch.Tensor
    final_nll_canonical: float
    p_target_canonical: float
    final_nll_weighted: float
    p_target_weighted: float
    optimization_context_count: int
    optimization_para_count: int
    optimization_para_ids: List[str]
    solver_failed: bool
    canonical_subject_token_index: Optional[int] = None
    canonical_subject_prefix_token_ids: Optional[List[int]] = None


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prompt_template(text: str, subject: str, *, label: str) -> str:
    value = str(text).strip()
    if not value:
        raise ValueError(f"{label} must be a non-empty string")
    if "{}" in value:
        try:
            value.format(subject)
        except (IndexError, KeyError, ValueError) as error:
            raise ValueError(f"{label} has an invalid format template") from error
        return value
    if subject not in value:
        raise ValueError(
            f"{label} must contain the subject {subject!r} or one '{{}}' slot"
        )
    return value.replace(subject, "{}", 1)


def _normalized_prompt_text(value: str, subject: str) -> str:
    text = str(value).strip()
    if "{}" in text:
        try:
            text = text.format(subject)
        except (IndexError, KeyError, ValueError) as error:
            raise ValueError("prompt has an invalid format template") from error
    return " ".join(text.casefold().split())


def _normalized_prompt_text_outside_subject(value: str, subject: str) -> str:
    """Normalize a prompt while masking its required subject occurrence.

    Target-blind paraphrases necessarily contain the subject.  A target string
    can also be a lexical substring of that subject (for example, target
    ``Jupiter`` and subject ``Jupiter-Avia``).  Such an occurrence was visible
    to the target-blind generator and is not answer leakage.  Keep the inner
    trust boundary aligned with the frozen sidecar audit by searching only the
    text outside the subject slot.
    """

    text = str(value).strip()
    if "{}" in text:
        try:
            text = text.format(" __SUBJECT__ ")
        except (IndexError, KeyError, ValueError) as error:
            raise ValueError("prompt has an invalid format template") from error
    else:
        text = text.replace(subject, " __SUBJECT__ ", 1)
    return " ".join(text.casefold().split())


def _optimization_paraphrases(
    request: Mapping[str, Any],
    hparams: AlphaEditHyperParams,
) -> Tuple[List[str], List[str]]:
    """Return audited optimization-only prompt templates and stable IDs.

    The runner owns the sidecar coverage/contamination audit.  This inner
    trust boundary nevertheless rejects the two catastrophic wiring errors:
    feeding the official evaluation rephrase back into optimization or adding
    a hidden target answer outside the subject/source-visible text.
    """

    if not bool(getattr(hparams, "hn_recovery_paraphrase_aware", False)):
        return [], []
    if "optimization_paraphrase" not in request:
        raise ValueError(
            "Paraphrase-aware HN requires runner-injected "
            "request['optimization_paraphrase']; official rephrase_prompt is "
            "evaluation-only"
        )
    raw = request["optimization_paraphrase"]
    values = [raw] if isinstance(raw, str) else list(raw)
    if not values:
        raise ValueError("optimization_paraphrase cannot be empty")
    subject = str(request["subject"])
    official_values: List[str] = []
    for key in ("rephrase_prompt", "rephrase"):
        official = request.get(key)
        if official is None:
            continue
        official_values.extend(
            [official] if isinstance(official, str) else list(official)
        )
    normalized_official = {
        _normalized_prompt_text(str(value), subject)
        for value in official_values
        if str(value).strip()
    }
    # Runner targets may include a training terminator such as <|eot_id|>.
    # Remove special-token spellings before the verbatim answer leak guard.
    target_without_specials = re.sub(
        r"<\|[^>]+\|>|</?s>|<pad>",
        " ",
        str(request["target_new"]),
        flags=re.IGNORECASE,
    )
    target_text = " ".join(target_without_specials.casefold().split()).strip()
    source_outside_subject = _normalized_prompt_text_outside_subject(
        str(request["prompt"]), subject
    )
    target_pattern = (
        re.compile(rf"(?<!\w){re.escape(target_text)}(?!\w)")
        if target_text
        else None
    )
    target_visible_in_source = bool(
        target_pattern and target_pattern.search(source_outside_subject)
    )
    templates: List[str] = []
    normalized_seen: set[str] = set()
    for index, value in enumerate(values):
        template = _prompt_template(
            str(value), subject, label=f"optimization_paraphrase[{index}]"
        )
        normalized = _normalized_prompt_text(template, subject)
        normalized_outside_subject = _normalized_prompt_text_outside_subject(
            template, subject
        )
        if normalized in normalized_official:
            raise ValueError(
                "Official evaluation rephrase was reused as an optimization "
                f"paraphrase for case {request.get('case_id')}"
            )
        if (
            target_pattern
            and target_pattern.search(normalized_outside_subject)
            and not target_visible_in_source
        ):
            raise ValueError(
                "Target answer appears verbatim in optimization paraphrase "
                f"for case {request.get('case_id')}"
            )
        if normalized in normalized_seen:
            raise ValueError("optimization_paraphrase contains duplicates")
        normalized_seen.add(normalized)
        templates.append(template)
    raw_ids = request.get("optimization_para_ids")
    ids = (
        [str(value) for value in raw_ids]
        if raw_ids is not None
        else [f"case_{request.get('case_id')}_opt_para_{i}" for i in range(len(templates))]
    )
    if len(ids) != len(templates) or len(set(ids)) != len(ids):
        raise ValueError(
            "optimization_para_ids must be unique and match the paraphrase count"
        )
    expected_sha256 = request.get("optimization_para_sha256")
    if expected_sha256 is not None:
        actual_sha256 = (
            hashlib.sha256(str(values[0]).encode("utf-8")).hexdigest()
            if len(values) == 1
            else _canonical_json_sha256([str(value) for value in values])
        )
        if str(expected_sha256) != actual_sha256:
            raise ValueError(
                "optimization_paraphrase payload SHA does not match the "
                "audited runner metadata"
            )
    return templates, ids


def _family_weighted_nll(
    nll_loss_each: torch.Tensor,
    *,
    canonical_context_count: int,
    para_context_count: int,
    edit_weight: float,
    para_weight: float,
) -> torch.Tensor:
    """Average within prompt families, then apply the configured family mix."""

    if canonical_context_count < 1:
        raise ValueError("canonical_context_count must be positive")
    if canonical_context_count + para_context_count != int(nll_loss_each.numel()):
        raise ValueError("NLL family sizes do not match the context tensor")
    edit_weight = float(edit_weight)
    para_weight = float(para_weight)
    if not math.isfinite(edit_weight) or edit_weight <= 0.0:
        raise ValueError("edit family weight must be finite and positive")
    if not math.isfinite(para_weight) or para_weight < 0.0:
        raise ValueError("paraphrase family weight must be finite and nonnegative")
    canonical_mean = nll_loss_each[:canonical_context_count].mean()
    if para_context_count == 0:
        # Keep the feature-off reducer byte-for-byte identical.
        return nll_loss_each.mean()
    if para_weight == 0.0:
        return canonical_mean
    para_mean = nll_loss_each[canonical_context_count:].mean()
    return (
        edit_weight * canonical_mean + para_weight * para_mean
    ) / (edit_weight + para_weight)


def _semantic_target_token_count(
    target_ids: torch.Tensor,
    tok: AutoTokenizer,
) -> int:
    """Exclude only known trailing training terminators from P(o*).

    Optimization continues to use the complete historical target tensor.  The
    adaptive selector and common final-NLL log measure the semantic answer,
    not the probability of an appended EOS/EOT token.
    """

    special_ids: set[int] = set()
    for attribute in ("eos_token_id", "sep_token_id"):
        raw = getattr(tok, attribute, None)
        if raw is None:
            continue
        if isinstance(raw, (list, tuple, set)):
            special_ids.update(int(value) for value in raw)
        else:
            special_ids.add(int(raw))
    count = int(target_ids.numel())
    flattened = target_ids.detach().reshape(-1)
    while count > 1 and int(flattened[count - 1].item()) in special_ids:
        count -= 1
    return count


def _semantic_rewriting_mask(
    rewriting_targets: torch.Tensor,
    *,
    total_target_token_count: int,
    semantic_target_token_count: int,
    device: torch.device,
) -> torch.Tensor:
    if not 0 < semantic_target_token_count <= total_target_token_count:
        raise ValueError("invalid semantic/total target token counts")
    mask = (rewriting_targets != -100).float().to(device)
    trailing_terminators = total_target_token_count - semantic_target_token_count
    for row in range(mask.shape[0]):
        positions = torch.nonzero(mask[row], as_tuple=False).flatten()
        if int(positions.numel()) != total_target_token_count:
            raise RuntimeError(
                "rewrite target mask does not contain the expected number of "
                "supervised tokens"
            )
        if trailing_terminators:
            mask[row, positions[-trailing_terminators:]] = 0.0
    return mask


def _latent_geometry(
    target_init: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> Dict[str, float]:
    h = target_init.detach().float().reshape(-1)
    d = target.detach().to(target_init.device).float().reshape(-1) - h
    h_sq = float(torch.dot(h, h).item())
    if not math.isfinite(h_sq) or h_sq <= eps:
        raise ValueError("target_init norm is zero or non-finite")
    h_norm = math.sqrt(h_sq)
    a = float(torch.dot(h, d).item()) / h_sq
    d_orthogonal = d - a * h
    b = float(torch.linalg.vector_norm(d_orthogonal).item()) / h_norm
    q = float(torch.linalg.vector_norm(d).item()) / h_norm
    rho = float(torch.linalg.vector_norm(h + d).item()) / h_norm
    return {
        "a": a,
        "b": b,
        "q_realized": q,
        "rho_realized": rho,
        "q_identity_abs_error": abs(q * q - (a * a + b * b)),
        "rho_identity_abs_error": abs(
            rho * rho - ((1.0 + a) * (1.0 + a) + b * b)
        ),
        "target_init_norm": h_norm,
        "delta_norm": float(torch.linalg.vector_norm(d).item()),
        "target_norm": float(torch.linalg.vector_norm(h + d).item()),
    }


def _adaptive_rho_settings(
    hparams: AlphaEditHyperParams,
) -> Tuple[Tuple[float, ...], float]:
    raw_ladder = getattr(hparams, "hn_recovery_adaptive_ladder", None)
    ladder = tuple(
        float(value)
        for value in (
            _DEFAULT_ADAPTIVE_RHO_LADDER if raw_ladder is None else raw_ladder
        )
    )
    if not ladder or any(
        not math.isfinite(value) or value < 1.0 for value in ladder
    ):
        raise ValueError("adaptive rho ladder must contain finite values >= 1")
    if any(right <= left for left, right in zip(ladder, ladder[1:])):
        raise ValueError("adaptive rho ladder must be strictly increasing")
    threshold = float(getattr(hparams, "hn_recovery_adaptive_threshold", 0.5))
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("adaptive target-probability threshold must be in [0, 1]")
    return ladder, threshold


def _attempt_sha256(result: _HNRecoverySolveResult, ratio: float) -> str:
    digest = hashlib.sha256()
    digest.update(f"{float(ratio):.17g}".encode("ascii"))
    digest.update(result.target.detach().float().cpu().numpy().tobytes())
    digest.update(f"{result.final_nll_canonical:.17g}".encode("ascii"))
    return digest.hexdigest()


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[2] / "util" / "residual_gain_regularization.py",
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _append_unique_hn_recovery_record(
    path_value: str,
    record: Mapping[str, Any],
) -> None:
    """Append one edit exactly once, including across resumed processes."""

    destination = Path(path_value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    path_key = str(destination.resolve())
    if path_key not in _HN_RECOVERY_LOGGED_KEYS:
        existing: set[Tuple[str, int, str]] = set()
        if destination.exists():
            with destination.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    existing.add(
                        (
                            str(payload.get("run_id")),
                            int(payload.get("edit_index", -1)),
                            str(payload.get("case_id")),
                        )
                    )
        _HN_RECOVERY_LOGGED_KEYS[path_key] = existing
    identity = (
        str(record["run_id"]),
        int(record["edit_index"]),
        str(record["case_id"]),
    )
    if identity in _HN_RECOVERY_LOGGED_KEYS[path_key]:
        raise FileExistsError(
            "HN recovery scalar row already exists; refusing to overwrite or "
            f"duplicate {identity} in {destination}"
        )
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _HN_RECOVERY_LOGGED_KEYS[path_key].add(identity)


def _recovery_identity(
    request: Mapping[str, Any],
    hparams: AlphaEditHyperParams,
    name: str,
    default: Any = None,
) -> Any:
    if name in request:
        return request[name]
    prefixed = f"hn_recovery_{name}"
    if prefixed in request:
        return request[prefixed]
    return getattr(hparams, prefixed, default)


def _selected_scalar_record(
    *,
    request: Mapping[str, Any],
    hparams: AlphaEditHyperParams,
    result: _HNRecoverySolveResult,
    requested_ratio: float,
    selected_ratio: float,
    threshold: Optional[float],
    relaxation_level: int,
    attempts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    geometry = _latent_geometry(result.target_init, result.target)
    q_cap = float(hparams.clamp_norm_factor)
    ceiling_violation = max(
        0.0, geometry["rho_realized"] - float(selected_ratio)
    )
    solver_failed = bool(
        result.solver_failed
        or any(not math.isfinite(value) for value in geometry.values())
    )
    threshold_met = (
        None
        if threshold is None
        else bool(result.p_target_canonical >= threshold and not solver_failed)
    )
    official_rephrase = request.get(
        "rephrase_prompt", request.get("rephrase", "")
    )
    optimization_para_sha256 = request.get("optimization_para_sha256")
    if optimization_para_sha256 is None and result.optimization_para_count:
        optimization_para_sha256 = _canonical_json_sha256(
            request.get("optimization_paraphrase")
        )
    experiment_id = _recovery_identity(request, hparams, "experiment_id")
    arm_id = _recovery_identity(request, hparams, "arm_id")
    order_id = _recovery_identity(request, hparams, "order_id")
    run_id = _recovery_identity(request, hparams, "run_id")
    if run_id is None and all(
        value is not None for value in (experiment_id, arm_id, order_id)
    ):
        run_id = f"{experiment_id}__{arm_id}__{order_id}"
    identity_values = {
        "experiment_id": experiment_id,
        "arm_id": arm_id,
        "run_id": run_id,
        "order_id": order_id,
        "order_seed": _recovery_identity(request, hparams, "order_seed"),
        "model_seed": _recovery_identity(request, hparams, "model_seed"),
        "edit_index": _recovery_identity(request, hparams, "edit_index"),
    }
    missing = [key for key, value in identity_values.items() if value is None]
    if missing:
        raise ValueError(
            "HN recovery scalar logging requires runner-injected identity "
            f"fields; missing {missing}"
        )
    constraint_violation = bool(
        ceiling_violation > 1e-4
        or geometry["q_realized"] > q_cap + 1e-4
    )
    geometric_infeasible = False
    record: Dict[str, Any] = {
        "schema_version": "alphaedit-hn-recovery-per-edit-v1",
        "protocol_id": _recovery_identity(
            request,
            hparams,
            "protocol_id",
            "alphaedit_hn_recovery_v1",
        ),
        **identity_values,
        "editor": "AlphaEdit",
        "dataset": "ZsRE",
        "case_id": request.get("case_id"),
        "source_index": request.get("source_index"),
        "rho_ceiling_requested": float(requested_ratio),
        "rho_ceiling_selected": float(selected_ratio),
        "q_cap": q_cap,
        **geometry,
        "q_contact": bool(geometry["q_realized"] >= q_cap - 1e-4),
        "rho_ceiling_violation": ceiling_violation,
        # R >= 1 always admits delta=0 under a q ball.  Keep this separate
        # from numerical/optimizer failure as required by the protocol.
        "geometric_infeasible": geometric_infeasible,
        "solver_failed": solver_failed,
        "constraint_violation": constraint_violation,
        "infeasible": bool(
            geometric_infeasible or solver_failed or constraint_violation
        ),
        "final_nll_canonical": result.final_nll_canonical,
        "p_target_canonical": result.p_target_canonical,
        "final_nll_weighted": result.final_nll_weighted,
        "p_target_weighted": result.p_target_weighted,
        "optimization_context_count": result.optimization_context_count,
        "optimization_para_count": result.optimization_para_count,
        "official_rephrase_used_in_optimization": False,
        "threshold_tau": threshold,
        "threshold_met": threshold_met,
        "relaxed": bool(relaxation_level > 0),
        "relaxation_level": int(relaxation_level),
        "adaptive_attempts": attempts,
        "optimization_para_ids": list(result.optimization_para_ids),
        "optimization_para_sha256": optimization_para_sha256,
        "evaluation_rephrase_sha256": request.get(
            "evaluation_rephrase_sha256",
            hashlib.sha256(str(official_rephrase).encode("utf-8")).hexdigest(),
        ),
        "committed_post_edit": None,
        "request_sha256": request.get(
            "request_sha256",
            _canonical_json_sha256(
                {
                    "case_id": request.get("case_id"),
                    "prompt": request.get("prompt"),
                    "subject": request.get("subject"),
                    "target_new": request.get("target_new"),
                    "edit_index": identity_values["edit_index"],
                }
            ),
        ),
        "implementation_sha256": _implementation_sha256(),
    }
    return record


def _validate_recovery_configuration(
    hparams: AlphaEditHyperParams,
) -> None:
    ratio = float(getattr(hparams, "residual_gain_target_ratio", 1.0))
    para_enabled = bool(
        getattr(hparams, "hn_recovery_paraphrase_aware", False)
    )
    adaptive = bool(getattr(hparams, "hn_recovery_adaptive_rho", False))
    scalar_log_path = getattr(hparams, "hn_recovery_scalar_log_path", None)
    if ratio != 1.0 or para_enabled or adaptive or scalar_log_path:
        if not bool(getattr(hparams, "residual_gain_regularization", False)):
            raise ValueError("HN recovery features require RGR to be enabled")
        if str(getattr(hparams, "residual_gain_objective", "gain")).lower() != "rho_minus_one":
            raise ValueError(
                "HN recovery features require residual_gain_objective="
                "'rho_minus_one'"
            )
    if ratio != 1.0 and adaptive:
        raise ValueError(
            "Adaptive HN owns the candidate ceiling; keep the top-level "
            "residual_gain_target_ratio at 1"
        )
    if adaptive and bool(getattr(hparams, "nas_enabled", False)):
        raise ValueError("Adaptive HN cannot select pre-NAS candidates")
    if scalar_log_path and bool(
        getattr(hparams, "nas_enabled", False)
        or getattr(hparams, "nas_collect_stats", False)
    ):
        raise ValueError(
            "The HN recovery scalar logger cannot be combined with NAS: NAS "
            "changes the target after inner-score measurement. Evaluate NAS "
            "anchors through the matched checkpoint evaluator instead."
        )


def _publish_selected_latent(
    model: Any, tok: Any, request: Mapping[str, Any], hparams: Any,
    layer: int, target_init: torch.Tensor, optimizer_delta: torch.Tensor,
    target: torch.Tensor, *, subject_index: Optional[int] = None,
    subject_prefix_ids: Optional[List[int]] = None,
    o0_axis_preservation_state: Optional[Mapping[str, Any]] = None,
) -> None:
    if not bool(getattr(hparams, "analysis_capture_virtual_actual", False)):
        return
    callback = getattr(hparams, "_analysis_virtual_actual_selected_target_callback", None)
    if not callable(callback):
        raise RuntimeError("virtual/actual capture requires the runner's selected-target callback")
    if subject_index is None or subject_prefix_ids is None:
        subject_index = find_fact_lookup_idx(
            str(request["prompt"]), str(request["subject"]), tok,
            hparams.fact_token, verbose=False,
        )
        raw_prompt = str(request["prompt"]).format(request["subject"])
        subject_prefix_ids = tok(raw_prompt, add_special_tokens=True)["input_ids"][:subject_index + 1]
    # Detached copies prevent an observer from modifying the solver's return.
    extra_observations = {}
    if o0_axis_preservation_state is not None:
        extra_observations["o0_axis_preservation_state"] = o0_axis_preservation_state
    callback(
        model=model, tokenizer=tok, request=dict(request), write_layer=int(layer),
        target_init=target_init.detach().clone(),
        optimizer_delta=optimizer_delta.detach().clone(), target=target.detach().clone(),
        canonical_subject_token_index=int(subject_index),
        canonical_subject_prefix_token_ids=list(subject_prefix_ids),
        **extra_observations,
    )


def _validate_o0_axis_configuration(hparams: Any, layer: int) -> None:
    if not bool(getattr(hparams, "o0_axis_preservation_enabled", False)):
        return
    if (
        int(layer) != 8 or list(hparams.layers) != [4, 5, 6, 7, 8]
        or hparams.fact_token != "subject_last"
        or getattr(hparams, "o0_axis_preservation_reference", "current_pre_edit") != "current_pre_edit"
        or getattr(hparams, "o0_axis_preservation_scope", "all_injected_contexts") != "all_injected_contexts"
    ):
        raise ValueError("O0-axis experiment requires L4-L8 subject_last edits, current_pre_edit reference, and all injected contexts")
    rtol = float(getattr(hparams, "o0_axis_preservation_svd_rtol", 1.0e-6))
    if not math.isfinite(rtol) or not 0.0 < rtol < 1.0:
        raise ValueError("O0-axis SVD rtol must be finite and between zero and one")
    axis_layers = getattr(hparams, "o0_axis_preservation_layers", None)
    if axis_layers is not None and list(axis_layers) not in ([0], [0, 1, 2, 3]):
        raise ValueError("Controlled early-O preservation supports axis layers [0] or [0,1,2,3]")
    incompatible = (
        "early_attention_preservation_enabled", "sadr_regularization",
        "nas_enabled", "nas_collect_stats", "sphere_enabled", "encore_enabled",
        "nse_enabled", "context_multikey_enabled", "key_gaussian_noise_enabled",
        "tangent_layer_allocation_enabled", "hn_recovery_adaptive_rho",
        "hn_recovery_paraphrase_aware",
    )
    if any(bool(getattr(hparams, name, False)) for name in incompatible):
        raise ValueError("O0-axis controlled experiment supports native AlphaEdit with optional historical HN only")
    if (getattr(hparams, "hn_recovery_scalar_log_path", None)
        or float(getattr(hparams, "residual_gain_target_ratio", 1.0)) != 1.0
        or getattr(hparams, "inner_margin_schedule", "legacy") != "legacy"):
        raise ValueError("O0-axis experiment does not combine with HN recovery or margin-schedule modes")


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: AlphaEditHyperParams,
    layer: int,
    context_templates: List[str],
) -> torch.Tensor:
    """Compute AlphaEdit's target, with optional HN recovery controls."""

    _validate_recovery_configuration(hparams)
    _validate_o0_axis_configuration(hparams, layer)
    ratio = float(getattr(hparams, "residual_gain_target_ratio", 1.0))
    para_enabled = bool(
        getattr(hparams, "hn_recovery_paraphrase_aware", False)
    )
    adaptive = bool(getattr(hparams, "hn_recovery_adaptive_rho", False))
    scalar_log_path = getattr(hparams, "hn_recovery_scalar_log_path", None)
    recovery_metadata = bool(
        adaptive or para_enabled or ratio != 1.0 or scalar_log_path
    )
    if not recovery_metadata:
        # Feature-off path calls the historical implementation directly.
        return _compute_z_once(
            model, tok, request, hparams, layer, context_templates
        )

    if not adaptive:
        result = _compute_z_once(
            model,
            tok,
            request,
            hparams,
            layer,
            context_templates,
            return_recovery_result=True,
        )
        assert isinstance(result, _HNRecoverySolveResult)
        if result.solver_failed:
            raise RuntimeError(
                "HN recovery's selected non-adaptive solve is non-finite; "
                "refusing to commit it to the outer AlphaEdit solve"
            )
        attempt = {
            "attempt_index": 0,
            "rho_ceiling_candidate": ratio,
            "final_nll_canonical": result.final_nll_canonical,
            "p_target_canonical": result.p_target_canonical,
            "threshold_met": True,
            "selected": True,
            "solver_failed": result.solver_failed,
            "artifact_sha256": _attempt_sha256(result, ratio),
        }
        if scalar_log_path:
            _append_unique_hn_recovery_record(
                scalar_log_path,
                _selected_scalar_record(
                    request=request,
                    hparams=hparams,
                    result=result,
                    requested_ratio=ratio,
                    selected_ratio=ratio,
                    threshold=None,
                    relaxation_level=0,
                    attempts=[attempt],
                ),
            )
        _publish_selected_latent(
            model, tok, request, hparams, layer,
            result.target_init, result.optimizer_delta, result.target,
            subject_index=result.canonical_subject_token_index,
            subject_prefix_ids=result.canonical_subject_prefix_token_ids,
        )
        return result.target

    ladder, threshold = _adaptive_rho_settings(hparams)
    results: List[_HNRecoverySolveResult] = []
    attempts: List[Dict[str, Any]] = []
    selected_index: Optional[int] = None
    for attempt_index, candidate_ratio in enumerate(ladder):
        attempt_hparams = copy(hparams)
        attempt_hparams.hn_recovery_adaptive_rho = False
        attempt_hparams.residual_gain_target_ratio = candidate_ratio
        attempt_hparams.hn_recovery_scalar_log_path = None
        attempt_hparams._hn_recovery_attempt_index = attempt_index
        attempt_hparams._hn_recovery_suppress_side_artifacts = True
        result = _compute_z_once(
            model,
            tok,
            request,
            attempt_hparams,
            layer,
            context_templates,
            return_recovery_result=True,
            persist_latent_artifact=False,
        )
        assert isinstance(result, _HNRecoverySolveResult)
        results.append(result)
        met = bool(
            not result.solver_failed
            and result.p_target_canonical >= threshold
        )
        attempts.append(
            {
                "attempt_index": attempt_index,
                "rho_ceiling_candidate": candidate_ratio,
                "final_nll_canonical": result.final_nll_canonical,
                "p_target_canonical": result.p_target_canonical,
                "threshold_met": met,
                "selected": False,
                "solver_failed": result.solver_failed,
                "artifact_sha256": _attempt_sha256(result, candidate_ratio),
            }
        )
        if met:
            selected_index = attempt_index
            break
    if selected_index is None:
        selected_index = len(results) - 1
    attempts[selected_index]["selected"] = True
    selected = results[selected_index]
    selected_ratio = ladder[selected_index]
    if selected.solver_failed:
        raise RuntimeError(
            "Adaptive HN's selected candidate is non-finite; refusing to "
            "commit it to the outer AlphaEdit solve"
        )
    save_latent_artifact(
        hparams,
        method="AlphaEdit",
        request=request,
        write_layer=layer,
        target_init=selected.target_init,
        optimizer_delta=selected.optimizer_delta,
        target=selected.target,
    )
    if scalar_log_path:
        _append_unique_hn_recovery_record(
            scalar_log_path,
            _selected_scalar_record(
                request=request,
                hparams=hparams,
                result=selected,
                requested_ratio=ladder[0],
                selected_ratio=selected_ratio,
                threshold=threshold,
                relaxation_level=selected_index,
                attempts=attempts,
            ),
        )
    _publish_selected_latent(
        model, tok, request, hparams, layer,
        selected.target_init, selected.optimizer_delta, selected.target,
        subject_index=selected.canonical_subject_token_index,
        subject_prefix_ids=selected.canonical_subject_prefix_token_ids,
    )
    return selected.target


def _get_model_input_device(model: AutoModelForCausalLM) -> torch.device:
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    return next(model.parameters()).device


def _get_module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _save_early_attention_artifact(
    hparams: Any,
    request: Mapping[str, Any],
    regularizer: EarlyAttentionPreservationRegularizer,
    delta: torch.Tensor,
    target_dtype: torch.dtype,
) -> None:
    """Persist the frozen basis and latent target independently of outer writes."""
    root_value = getattr(hparams, "analysis_artifact_dir", None)
    if not root_value:
        return
    edit_index, case_id, stem = edit_identity(hparams, request)
    destination = Path(root_value) / "early_attention"
    destination.mkdir(parents=True, exist_ok=True)
    payload = regularizer.export_state()
    payload.update({
        "method": "AlphaEdit",
        "edit_index": edit_index,
        "case_id": request.get("case_id", case_id),
        "write_layer": 8,
        "lambda": float(hparams.early_attention_preservation_lambda),
        "optimizer_delta": delta.detach().float().cpu().clone(),
    })
    payload["mathematical_target_contexts"] = payload["reference"] + payload["optimizer_delta"]
    payload["forward_dtype_target_contexts"] = payload["mathematical_target_contexts"].to(target_dtype).float()
    payload["forward_dtype"] = str(target_dtype)
    with torch.no_grad():
        for label, key in (
            ("final_mathematical_target", "mathematical_target_contexts"),
            ("final_forward_dtype_target", "forward_dtype_target_contexts"),
        ):
            _, final_diagnostics = early_attention_projection_loss(
                payload[key], payload["reference"], payload["basis"]
            )
            payload[label] = final_diagnostics
    payload["target_semantics"] = (
        "current pre-edit context reference plus shared optimizer delta in FP32; "
        "inner forward additionally rounds the in-place write to the model dtype; "
        "this is a latent target, not the realized outer parameter update"
    )
    tensor_path = destination / f"{stem}.pt"
    temporary = tensor_path.with_name(f".{tensor_path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, tensor_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    metadata = {key: value for key, value in payload.items() if not torch.is_tensor(value)}
    metadata["tensor_shapes"] = {
        key: list(value.shape) for key, value in payload.items() if torch.is_tensor(value)
    }
    metadata["ranks"] = payload["ranks"].tolist()
    metadata["artifact_file"] = tensor_path.name
    metadata["artifact_sha256"] = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    json_path = destination / f"{stem}.json"
    temporary_json = json_path.with_name(f".{json_path.name}.{os.getpid()}.tmp")
    try:
        temporary_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary_json, json_path)
    finally:
        if temporary_json.exists():
            temporary_json.unlink()


def _save_o0_axis_artifact(
    hparams: Any, request: Mapping[str, Any], preserver: O0AxisPreserver,
    raw_delta: torch.Tensor, delta: torch.Tensor, target: torch.Tensor,
    target_dtype: torch.dtype,
) -> None:
    """Freeze context axes, anchors and effective target before the outer solve."""
    root_value = getattr(hparams, "analysis_artifact_dir", None)
    if not root_value:
        return
    edit_index, case_id, stem = edit_identity(hparams, request)
    destination = Path(root_value) / "o0_axis"
    destination.mkdir(parents=True, exist_ok=True)
    payload = preserver.export_state()
    payload.update({
        "method": "AlphaEdit", "edit_index": edit_index,
        "case_id": request.get("case_id", case_id), "write_layer": 8,
        "optimizer_raw_delta": raw_delta.detach().float().cpu().clone(),
        "optimizer_delta": delta.detach().float().cpu().clone(),
        "returned_target": target.detach().float().cpu().clone(),
        "forward_dtype": str(target_dtype),
    })
    reference = payload["reference"]
    axes = payload["context_unit_axes"]
    payload["mathematical_target_contexts"] = reference + payload["optimizer_delta"]
    payload["forward_dtype_target_contexts"] = payload["mathematical_target_contexts"].to(target_dtype).float()
    for name in ("mathematical", "forward_dtype"):
        difference = payload[f"{name}_target_contexts"] - reference
        drifts = (difference * axes).sum(dim=-1)
        payload[f"{name}_coefficient_drift"] = drifts
        payload[f"{name}_coefficient_drift_max_abs"] = float(drifts.abs().max())
        if "context_layer_unit_axes" in payload:
            all_drifts = (difference[:, None, :] * payload["context_layer_unit_axes"]).sum(dim=-1)
            payload[f"{name}_context_layer_coefficient_drift"] = all_drifts
            payload[f"{name}_context_layer_coefficient_drift_max_abs"] = float(all_drifts.abs().max())
    payload["target_semantics"] = (
        "All selected early O axes at every injected rewrite and KL context and pre-edit O8 anchors are frozen. "
        "The shared effective FP32 delta is projected onto their orthogonal complement; "
        "native BF16 injection rounding is recorded separately. Outer weight updates are unconstrained."
    )
    tensor_path = destination / f"{stem}.pt"
    temporary = tensor_path.with_name(f".{tensor_path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, tensor_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    metadata = {k: v for k, v in payload.items() if not torch.is_tensor(v)}
    metadata["tensor_shapes"] = {k: list(v.shape) for k, v in payload.items() if torch.is_tensor(v)}
    metadata["artifact_file"] = tensor_path.name
    metadata["artifact_sha256"] = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    json_path = tensor_path.with_suffix(".json")
    temporary_json = json_path.with_name(f".{json_path.name}.{os.getpid()}.tmp")
    try:
        temporary_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary_json, json_path)
    finally:
        if temporary_json.exists():
            temporary_json.unlink()


def _compute_z_once(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: AlphaEditHyperParams,
    layer: int,
    context_templates: List[str],
    *,
    return_recovery_result: bool = False,
    persist_latent_artifact: bool = True,
) -> Union[torch.Tensor, _HNRecoverySolveResult]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    oedit_regularizer = _prepare_oedit_regularizer(model, hparams, layer=layer)

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_module(model, f"{hparams.lm_head_module}").weight.T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")
    input_device = _get_model_input_device(model)
    rewrite_device = _get_module_device(
        nethook.get_module(model, hparams.layer_module_tmp.format(layer))
    )

    # Tokenize target into list of int token IDs
    target_ids = tok.encode(
        request["target_new"],
        return_tensors="pt",
        add_special_tokens=False,
    ).to(input_device)[0]

    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    if target_ids.numel() == 0:
        raise ValueError("target_new tokenized to an empty target")
    semantic_target_token_count = _semantic_target_token_count(target_ids, tok)
    # Preserve the original prompt endpoint for prompt_last RGR.
    canonical_rewriting_original_prompts = [
        context.format(request["prompt"])
        for context_types in context_templates
        for context in context_types
    ]
    optimization_para_templates, optimization_para_ids = (
        _optimization_paraphrases(request, hparams)
    )
    para_rewriting_original_prompts = [
        context.format(para_template)
        for para_template in optimization_para_templates
        for context_types in context_templates
        for context in context_types
    ]
    rewriting_original_prompts = (
        canonical_rewriting_original_prompts
        + para_rewriting_original_prompts
    )
    canonical_context_count = len(canonical_rewriting_original_prompts)
    para_context_count = len(para_rewriting_original_prompts)
    rewriting_prompts, kl_prompts = [
        prompt + tok.decode(target_ids[:-1])
        for prompt in rewriting_original_prompts
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(input_device)

    # Compute rewriting targets
    rewriting_targets = torch.full(
        (len(rewriting_prompts), input_tok["input_ids"].shape[1]),
        -100,
        device=input_device,
        dtype=input_tok["input_ids"].dtype,
    )

    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    prompt_last_idxs = [
        len(
            tok(
                prompt.format(request["subject"]),
                add_special_tokens=True,
            )["input_ids"]
        )
        - 1
        for prompt in rewriting_original_prompts
    ]
    rgr = ResidualGainRegularizer(
        hparams=hparams,
        write_layer=layer,
        num_hidden_layers=int(
            getattr(
                model.config,
                "num_hidden_layers",
                getattr(model.config, "n_layer", 0),
            )
        ),
        attention_mask=input_tok["attention_mask"],
        # HN observes canonical contexts only.  Optimization paraphrases
        # rotate the target-fitting direction without silently expanding the
        # state/radius constraint's context scope.
        rewrite_batch_indices=list(range(canonical_context_count)),
        subject_last_indices=lookup_idxs[:canonical_context_count],
        prompt_last_indices=prompt_last_idxs[:canonical_context_count],
        reduction_device=rewrite_device,
    )
    if rgr.optimization_enabled:
        print(
            "[RGR][AlphaEdit] "
            f"layers={rgr.layers} scope={rgr.token_scope} "
            f"lambda={rgr.lambda_} position_lambdas={rgr.position_lambdas} "
            f"margin={rgr.margin} type={rgr.loss_type} "
            f"target_ratio={rgr.target_ratio} "
            f"efficacy_gate={rgr.efficacy_threshold}"
        )
    if optimization_para_templates:
        print(
            "[HN-RECOVERY][AlphaEdit] paraphrase-aware "
            f"canonical_contexts={canonical_context_count} "
            f"para_prompts={len(optimization_para_templates)} "
            f"para_contexts={para_context_count} "
            f"family_weights="
            f"{getattr(hparams, 'hn_recovery_edit_family_weight', 1.0)}:"
            f"{getattr(hparams, 'hn_recovery_para_family_weight', 0.5)} "
            "rgr_scope=canonical_only"
        )
    sadr_enabled = bool(getattr(hparams, "sadr_regularization", False))
    sadr_layers = list(
        getattr(hparams, "sadr_attn_layers", None)
        or range(
            int(
                getattr(
                    model.config,
                    "num_hidden_layers",
                    getattr(model.config, "n_layer", 0),
                )
            )
        )
    )
    sadr_reference = None
    sadr_lambda = float(getattr(hparams, "sadr_lambda", 0.01))
    if sadr_enabled:
        if not sadr_layers:
            raise ValueError("SADR requires at least one attention layer")
        print(
            "[SADR][AlphaEdit] "
            f"layers={sadr_layers} lambda={sadr_lambda} "
            f"efficacy_gate={getattr(hparams, 'sadr_efficacy_threshold', 0.5)}"
        )

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros(
            (model.config.n_embd,),
            requires_grad=True,
            device=rewrite_device,
        )
    elif hasattr(model.config, 'hidden_size'):
        delta = torch.zeros(
            (model.config.hidden_size,),
            requires_grad=True,
            device=rewrite_device,
        )
    else:
        raise NotImplementedError
    target_init, kl_distr_init = None, None
    # Keep Adam's leaf parameter separate from the effective projected delta.
    # With the opt-in disabled these are exactly the original same tensor.
    delta_parameter = delta
    o0_axis = None
    if bool(getattr(hparams, "o0_axis_preservation_enabled", False)):
        _validate_o0_axis_configuration(hparams, layer)
        if (canonical_context_count != 6 or para_context_count != 0
            or getattr(model.config, "model_type", None) != "llama"):
            raise ValueError("O0-axis controlled run requires the original six Llama rewrite contexts and no extra paraphrases")
        o0_axis = O0AxisPreserver(
            model, enabled=True,
            context_batch_indices=list(range(len(all_prompts))),
            lookup_indices=lookup_idxs,
            o0_module_name=hparams.layer_module_tmp.format(0),
            module_template=hparams.layer_module_tmp,
            axis_layers=(getattr(hparams, "o0_axis_preservation_layers", None) or [0]),
            attention_mask=input_tok["attention_mask"], batch_first=True,
            svd_rtol=float(getattr(hparams, "o0_axis_preservation_svd_rtol", 1.0e-6)),
        )

    early_attention = None
    early_attention_weight = 0.0
    if bool(getattr(hparams, "early_attention_preservation_enabled", False)):
        early_attention_weight = float(
            getattr(hparams, "early_attention_preservation_lambda", 1.0)
        )
        early_layers = list(
            getattr(hparams, "early_attention_preservation_layers", None)
            or range(5)
        )
        if (
            layer != 8
            or list(hparams.layers) != [4, 5, 6, 7, 8]
            or hparams.fact_token != "subject_last"
            or early_layers != list(range(5))
            or canonical_context_count != 6
            or para_context_count != 0
            or not math.isfinite(early_attention_weight)
            or early_attention_weight <= 0.0
            or getattr(hparams, "early_attention_preservation_reference", "current_pre_edit") != "current_pre_edit"
        ):
            raise ValueError("HN+attn requires L4-L8 subject editing, A0-A4, six canonical contexts and current_pre_edit reference")
        if (
            not rgr.optimization_enabled
            or rgr.objective != "rho_minus_one"
            or rgr.loss_type != "positive_squared"
            or rgr.lambda_ != 1.0
            or rgr.target_ratio != 1.0
            or rgr.select_best
            or str(getattr(hparams, "inner_margin_schedule", "legacy")) != "legacy"
            or rgr.cosine_aux_lambda != 0.0
            or rgr.margin != 0.0
            or rgr.efficacy_threshold != 0.0
            or rgr.position_layers != {"subject_last": [8]}
            or bool(getattr(hparams, "hn_recovery_adaptive_rho", False))
            or bool(getattr(hparams, "sadr_regularization", False))
            or bool(getattr(hparams, "nas_enabled", False))
            or bool(getattr(hparams, "encore_enabled", False))
        ):
            raise ValueError("HN+attn must add only early-attention preservation to the unchanged HN control")
        early_attention = EarlyAttentionPreservationRegularizer(
            model,
            enabled=True,
            layers=early_layers,
            rewrite_batch_indices=list(range(canonical_context_count)),
            subject_last_indices=lookup_idxs[:canonical_context_count],
            attention_module_template=hparams.attn_module_tmp,
            svd_rtol=1.0e-5,
            attention_mask=input_tok["attention_mask"],
        )

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init, delta

        if cur_layer == hparams.layer_module_tmp.format(layer):
            if o0_axis is not None:
                o0_axis.initialize_reference(cur_out)
                delta = o0_axis.effective_delta(delta_parameter)
            if early_attention is not None:
                # Must run before the in-place delta insertion. The utility
                # clones the zero-delta target and freezes the early-A basis.
                early_attention.initialize_reference(cur_out)
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                if isinstance(cur_out, tuple):
                    target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()
                else:
                    target_init = cur_out[0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):
                if isinstance(cur_out, tuple):
                    if len(lookup_idxs)!=len(cur_out[0]):
                        cur_out[0][idx, i, :] += delta.to(cur_out[0].device)
                    else:
                        cur_out[0][i, idx, :] += delta.to(cur_out[0].device)
                else:
                    if len(lookup_idxs)!=len(cur_out):
                        cur_out[idx, i, :] += delta.to(cur_out.device)
                    else:
                        cur_out[i, idx, :] += delta.to(cur_out.device)

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta_parameter], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)
    best_rgr_delta = None
    best_rgr_key = None
    best_efficacy_delta = None
    best_efficacy = float("-inf")
    encore_enabled = bool(getattr(hparams, "encore_enabled", False))
    mpes_state = MPESState()
    margin_schedule = validate_inner_margin_schedule(
        getattr(hparams, "inner_margin_schedule", "legacy"),
        getattr(hparams, "inner_target_margin_threshold", 0.0),
        getattr(hparams, "inner_margin_refinement_steps", 0),
        getattr(hparams, "inner_margin_hinge_weight", 1.0),
    )
    margin_schedule_mode = str(margin_schedule["mode"])
    margin_schedule_enabled = margin_schedule_mode != "legacy"
    margin_reached_step = None
    margin_refinement_updates = 0
    formation_iteration_budget = int(hparams.v_num_grad_steps)
    refinement_iteration_budget = (
        int(margin_schedule["refinement_steps"])
        if margin_schedule_mode in {"delayed_rgr", "two_stage"}
        else 0
    )
    # A refinement arm promises K optimizer updates *after* target formation.
    # Keep the historical formation budget intact, but reserve K additional
    # forward/update slots so a margin first reached near the old final step
    # does not silently receive fewer than K refinement updates.  Legacy and
    # fixed_stop retain exactly the historical loop bound.
    total_iteration_budget = (
        formation_iteration_budget + refinement_iteration_budget
    )
    if margin_schedule_enabled:
        print(
            "[INNER-MARGIN][AlphaEdit] "
            f"mode={margin_schedule_mode} "
            f"threshold={margin_schedule['threshold']:.6g} "
            f"refinement_steps={margin_schedule['refinement_steps']} "
            f"hinge_weight={margin_schedule['hinge_weight']:.6g}"
        )

    # Execute optimization
    for it in range(total_iteration_budget):
        opt.zero_grad()
        rgr_loss = None
        rgr_diagnostics = {}
        sadr_loss = None
        sadr_diagnostics = {}
        early_attention_loss = None
        early_attention_diagnostics = {}
        o0_axis_diagnostics = {}

        # Forward propagation
        with (o0_axis if o0_axis is not None else nullcontext()), (early_attention if early_attention is not None else nullcontext()), nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            if rgr.enabled or sadr_enabled:
                model_output = model(
                    **input_tok,
                    output_hidden_states=rgr.enabled,
                    output_attentions=sadr_enabled,
                    use_cache=False,
                )
            else:
                model_output = model(**input_tok)
            logits = model_output.logits
            if o0_axis is not None:
                o0_axis_diagnostics = o0_axis.coefficient_diagnostics(
                    tr[hparams.layer_module_tmp.format(layer)].output
                )
                o0_axis_diagnostics.update(o0_axis.delta_diagnostics(delta_parameter))
            if early_attention is not None:
                early_attention_loss, early_attention_diagnostics = early_attention.loss(
                    tr[hparams.layer_module_tmp.format(layer)].output
                )
            if rgr.enabled:
                rgr_loss, rgr_diagnostics = rgr.observe(
                    model_output.hidden_states
                )
            if sadr_enabled:
                if model_output.attentions is None:
                    raise RuntimeError(
                        "SADR was enabled but the model returned no attentions"
                    )
                if sadr_reference is None:
                    sadr_reference = detach_attention_tensors(
                        model_output.attentions,
                        sadr_layers,
                    )
                else:
                    raw_sadr_loss, sadr_diagnostics = (
                        official_sadr_attention_kl_loss(
                            model_output.attentions,
                            sadr_reference,
                            input_tok["attention_mask"],
                            list(range(len(rewriting_prompts))),
                            prompt_last_idxs,
                            lookup_idxs[: len(rewriting_prompts)],
                            layers=sadr_layers,
                            reduction_device=rewrite_device,
                        )
                    )
                    sadr_loss = sadr_lambda * raw_sadr_loss
                    sadr_diagnostics.update(
                        {
                            "sadr_raw_loss": float(
                                raw_sadr_loss.detach().item()
                            ),
                            "sadr_lambda": sadr_lambda,
                            "weighted_sadr_loss": float(
                                sadr_loss.detach().item()
                            ),
                        }
                    )

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        if isinstance(tr[hparams.layer_module_tmp.format(loss_layer)].output, tuple):
            output=tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
        else:
            output=tr[hparams.layer_module_tmp.format(loss_layer)].output
        if output.shape[1]!=rewriting_targets.shape[1]:
            output=torch.transpose(output, 0, 1)
        full_repr = output[:len(rewriting_prompts)]

        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / target_ids.size(0)
        if para_context_count:
            nll_loss = _family_weighted_nll(
                nll_loss_each,
                canonical_context_count=canonical_context_count,
                para_context_count=para_context_count,
                edit_weight=float(
                    getattr(hparams, "hn_recovery_edit_family_weight", 1.0)
                ),
                para_weight=float(
                    getattr(hparams, "hn_recovery_para_family_weight", 0.5)
                ),
            )
        else:
            # Exact legacy reducer when paraphrase-aware optimization is off.
            nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss.to(nll_loss.device) + weight_decay.to(nll_loss.device)
        efficacy_score = float(
            torch.exp(
                -nll_loss_each[:canonical_context_count].detach()
            ).mean().item()
        )
        margin_stats = None
        margin_floor_loss = None
        margin_score = float("nan")
        margin_mean = float("nan")
        margin_reached = False
        if margin_schedule_enabled:
            margin_stats = target_margin_stats(log_probs, rewriting_targets)
            margin_floor_loss = margin_stats.hinge_floor(
                float(margin_schedule["threshold"])
            )
            margin_score = float(margin_stats.minimum.detach().item())
            margin_mean = float(margin_stats.mean.detach().item())
            margin_reached = bool(
                margin_score >= float(margin_schedule["threshold"])
            )
            if margin_reached and margin_reached_step is None:
                margin_reached_step = int(it)
        schedule_rgr_ready = bool(
            not margin_schedule_enabled
            or (
                margin_schedule_mode in {"delayed_rgr", "two_stage"}
                and margin_reached_step is not None
            )
        )
        rgr_gate_active = bool(
            rgr.optimization_enabled
            and rgr_loss is not None
            and efficacy_score >= rgr.efficacy_threshold
            and schedule_rgr_ready
        )
        if margin_schedule_mode == "two_stage" and margin_reached_step is not None:
            if margin_floor_loss is None:
                raise RuntimeError("two_stage schedule has no margin-floor loss")
            optimization_loss = (
                float(margin_schedule["hinge_weight"]) * margin_floor_loss
                + kl_loss.to(nll_loss.device)
                + weight_decay.to(nll_loss.device)
            )
            if rgr_gate_active:
                optimization_loss = optimization_loss + rgr_loss.to(loss.device)
        else:
            optimization_loss = (
                loss + rgr_loss.to(loss.device)
                if rgr_gate_active
                else loss
            )
        sadr_gate_active = bool(
            sadr_loss is not None
            and efficacy_score
            >= float(getattr(hparams, "sadr_efficacy_threshold", 0.5))
        )
        if sadr_gate_active:
            optimization_loss = optimization_loss + sadr_loss.to(loss.device)
        oedit_diagnostics = None
        if oedit_regularizer is not None:
            oedit_loss, oedit_diagnostics = oedit_regularizer.loss(delta)
            optimization_loss = optimization_loss + oedit_loss.to(loss.device)
        if early_attention_loss is not None:
            weighted_early_attention_loss = early_attention_weight * early_attention_loss.to(loss.device)
            optimization_loss = optimization_loss + weighted_early_attention_loss
            early_attention_diagnostics.update({
                "early_attention_preservation_lambda": early_attention_weight,
                "weighted_early_attention_preservation_loss": float(weighted_early_attention_loss.detach().item()),
            })
            rgr_diagnostics.update(early_attention_diagnostics)
            append_rgr_record(
                getattr(hparams, "early_attention_preservation_log_path", None),
                {
                    "method": "AlphaEdit",
                    "edit_index": getattr(hparams, "analysis_current_edit_index", None),
                    "case_id": request.get("case_id"),
                    "write_layer": layer,
                    "inner_step": it,
                    "efficacy_score": efficacy_score,
                    "base_edit_loss": float(loss.detach().item()),
                    "optimization_loss": float(optimization_loss.detach().item()),
                    **early_attention_diagnostics,
                },
            )
        if o0_axis is not None:
            rgr_diagnostics.update(o0_axis_diagnostics)
            append_rgr_record(
                getattr(hparams, "o0_axis_preservation_log_path", None),
                {
                    "method": "AlphaEdit", "record_type": "inner",
                    "edit_index": getattr(hparams, "analysis_current_edit_index", None),
                    "case_id": request.get("case_id"), "write_layer": layer,
                    "inner_step": it, "efficacy_score": efficacy_score,
                    "nll_loss": float(nll_loss.detach().item()),
                    "kl_loss": float(kl_loss.detach().item()),
                    "weight_decay": float(weight_decay.detach().item()),
                    "base_edit_loss": float(loss.detach().item()),
                    "optimization_loss": float(optimization_loss.detach().item()),
                    **o0_axis_diagnostics,
                },
            )
        if rgr.select_best:
            if efficacy_score > best_efficacy:
                best_efficacy = efficacy_score
                best_efficacy_delta = delta.detach().clone()
            if (
                rgr_gate_active
                and efficacy_score >= rgr.selection_threshold
                and rgr_diagnostics.get("residual_gain_loss") is not None
            ):
                selection_loss = (
                    rgr_diagnostics["weighted_residual_gain_loss"]
                    if rgr.cosine_aux_lambda > 0.0
                    else rgr_diagnostics["residual_gain_loss"]
                )
                candidate_key = (
                    float(selection_loss),
                    float(loss.detach().item()),
                )
                if best_rgr_key is None or candidate_key < best_rgr_key:
                    best_rgr_key = candidate_key
                    best_rgr_delta = delta.detach().clone()
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if rgr.optimization_enabled:
            weighted = float(
                rgr_diagnostics.get("weighted_residual_gain_loss", 0.0)
            )
            print(
                "[RGR][AlphaEdit] "
                f"step={it} efficacy={efficacy_score:.6f} "
                f"gate={int(rgr_gate_active)} "
                f"objective={rgr_diagnostics.get('residual_gain_objective', 'gain')} "
                f"alignment_weight={rgr_diagnostics.get('residual_gain_alignment_weight', 1.0):.3g} "
                f"cos_aux_lambda={rgr_diagnostics.get('residual_gain_cosine_aux_lambda', 0.0):.3g} "
                f"cosine={rgr_diagnostics.get('residual_gain_current_input_write_cosine_mean', 0.0):.6e} "
                f"cosine_ref={rgr_diagnostics.get('residual_gain_reference_input_write_cosine_mean', 0.0):.6e} "
                f"cos_aux={rgr_diagnostics.get('residual_gain_cosine_aux_loss', 0.0):.6e} "
                f"weighted_cos_aux={rgr_diagnostics.get('weighted_residual_gain_cosine_aux_loss', 0.0):.6e} "
                f"gain={rgr_diagnostics.get('residual_gain_current_gain_mean', 0.0):.6e} "
                f"reference={rgr_diagnostics.get('residual_gain_reference_gain_mean', 0.0):.6e} "
                f"excess={rgr_diagnostics.get('residual_gain_excess_gain_mean', 0.0):.6e} "
                f"objective_excess={rgr_diagnostics.get('residual_gain_excess_objective_mean', 0.0):.6e} "
                f"weighted_base={rgr_diagnostics.get('weighted_residual_gain_base_loss', 0.0):.6e} "
                f"weighted={weighted:.6e}"
            )
            append_rgr_record(
                rgr.log_path,
                {
                    "method": "AlphaEdit",
                    "case_id": request.get("case_id"),
                    "hn_recovery_attempt_index": getattr(
                        hparams, "_hn_recovery_attempt_index", None
                    ),
                    "rho_ceiling_candidate": float(
                        getattr(hparams, "residual_gain_target_ratio", 1.0)
                    ),
                    "write_layer": layer,
                    "inner_step": it,
                    "efficacy_score": efficacy_score,
                    "efficacy_gate_active": rgr_gate_active,
                    "base_edit_loss": float(loss.detach().item()),
                    "optimization_loss": float(optimization_loss.detach().item()),
                    **rgr_diagnostics,
                },
            )
        if rgr.enabled and not bool(
            getattr(hparams, "_hn_recovery_suppress_side_artifacts", False)
        ):
            record_inner_gain_step(
                hparams,
                method="AlphaEdit",
                request=request,
                write_layer=layer,
                inner_step=it,
                efficacy_score=efficacy_score,
                base_edit_loss=float(loss.detach().item()),
                optimization_loss=float(optimization_loss.detach().item()),
                diagnostics=rgr_diagnostics,
            )
        if margin_schedule_enabled:
            margin_stage = (
                "refinement" if margin_reached_step is not None else "formation"
            )
            print(
                "[INNER-MARGIN][AlphaEdit] "
                f"step={it} stage={margin_stage} "
                f"minimum={margin_score:.6e} mean={margin_mean:.6e} "
                f"floor_loss={float(margin_floor_loss.detach().item()):.6e} "
                f"rgr_gate={int(rgr_gate_active)} "
                f"refinement_updates={margin_refinement_updates}"
            )
            append_rgr_record(
                getattr(hparams, "inner_margin_schedule_log_path", None),
                {
                    "method": "AlphaEdit",
                    "case_id": request.get("case_id"),
                    "write_layer": layer,
                    "inner_step": it,
                    "schedule_mode": margin_schedule_mode,
                    "stage": margin_stage,
                    "target_margin_threshold": float(
                        margin_schedule["threshold"]
                    ),
                    "target_margin_minimum": margin_score,
                    "target_margin_mean": margin_mean,
                    "target_margin_floor_loss": float(
                        margin_floor_loss.detach().item()
                    ),
                    "efficacy_score": efficacy_score,
                    "rgr_gate_active": rgr_gate_active,
                    "margin_reached_step": margin_reached_step,
                    "refinement_updates": margin_refinement_updates,
                },
            )
        if sadr_enabled:
            print(
                "[SADR][AlphaEdit] "
                f"step={it} efficacy={efficacy_score:.6f} "
                f"gate={int(sadr_gate_active)} "
                f"selected_heads={int(sadr_diagnostics.get('sadr_selected_heads', 0))} "
                f"raw={sadr_diagnostics.get('sadr_raw_loss', 0.0):.6e} "
                f"weighted={sadr_diagnostics.get('weighted_sadr_loss', 0.0):.6e}"
            )
            append_official_baseline_record(
                getattr(hparams, "sadr_log_path", None),
                {
                    "method": "AlphaEdit",
                    "baseline": "SADR",
                    "case_id": request.get("case_id"),
                    "write_layer": layer,
                    "inner_step": it,
                    "efficacy_score": efficacy_score,
                    "efficacy_gate_active": sadr_gate_active,
                    "base_edit_loss": float(loss.detach().item()),
                    "optimization_loss": float(
                        optimization_loss.detach().item()
                    ),
                    **sadr_diagnostics,
                },
            )
        if encore_enabled:
            mpes_status = mpes_observe(
                log_probs=log_probs,
                rewriting_targets=rewriting_targets,
                state=mpes_state,
                required_top1_steps=int(
                    getattr(hparams, "encore_mpes_top1_steps", 2)
                ),
                step=it,
                exclude_first_context=bool(
                    getattr(
                        hparams,
                        "encore_mpes_exclude_first_context",
                        True,
                    )
                ),
            )
            print(
                "[ENCORE-MPES][AlphaEdit] "
                f"step={it} top1={mpes_status['num_top1']}/"
                f"{mpes_status['num_targets']} "
                f"qualifying={mpes_status['qualifying_steps']}/"
                f"{getattr(hparams, 'encore_mpes_top1_steps', 2)} "
                f"stop={int(mpes_status['should_stop'])}"
            )
            append_official_baseline_record(
                getattr(hparams, "official_baseline_log_path", None),
                {
                    "method": "AlphaEdit",
                    "baseline": "ENCORE",
                    "stage": "mpes",
                    "case_id": request.get("case_id"),
                    "inner_step": it,
                    **mpes_status,
                },
            )
            if mpes_status["should_stop"]:
                break
        if margin_schedule_mode == "fixed_stop" and margin_reached:
            break
        if (
            margin_schedule_mode in {"delayed_rgr", "two_stage"}
            and margin_reached_step is not None
            and margin_refinement_updates
            >= int(margin_schedule["refinement_steps"])
        ):
            break

        # Extra iterations are reserved exclusively for refinement.  If the
        # floor was not reached within the original formation budget, stop
        # instead of covertly giving these arms more target-fitting updates.
        if (
            margin_schedule_mode in {"delayed_rgr", "two_stage"}
            and margin_reached_step is None
            and it == formation_iteration_budget - 1
        ):
            break

        stop_loss = (
            optimization_loss
            if (
                sadr_enabled
                or str(
                    getattr(hparams, "residual_gain_early_stop_mode", "base")
                ).lower()
                == "total"
            )
            else loss
        )
        if oedit_diagnostics is not None:
            append_rgr_record(
                getattr(hparams, "oedit_log_path", None),
                {
                    "method": "AlphaEdit", "stage": "oedit_inner",
                    "case_id": request.get("case_id"), "inner_step": it,
                    "edit_count": oedit_regularizer.edit_count,
                    "write_layer": int(layer),
                    "base_edit_loss": float(loss.detach().item()),
                    "optimization_loss": float(optimization_loss.detach().item()),
                    "mechanism_early_stop_mode": str(getattr(
                        hparams, "residual_gain_early_stop_mode", "base")).lower(),
                    "mechanism_early_stop_loss": float(stop_loss.detach().item()),
                    **oedit_diagnostics,
                },
            )
        if not margin_schedule_enabled and stop_loss < 5e-2:
            break

        if it == total_iteration_budget - 1:
            break

        # Backpropagate
        optimization_loss.backward()
        opt.step()
        if (
            margin_schedule_mode in {"delayed_rgr", "two_stage"}
            and margin_reached_step is not None
        ):
            margin_refinement_updates += 1

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        clamp_delta = (
            o0_axis.effective_delta(delta_parameter)
            if o0_axis is not None else delta_parameter
        )
        if clamp_delta.norm() > max_norm:
            with torch.no_grad():
                delta_parameter[...] = delta_parameter * max_norm / clamp_delta.norm()

    if rgr.select_best:
        selected_delta = (
            best_rgr_delta
            if best_rgr_delta is not None
            else best_efficacy_delta
        )
        if selected_delta is not None:
            with torch.no_grad():
                delta_parameter.copy_(selected_delta.to(delta_parameter))

    if o0_axis is not None:
        delta = o0_axis.effective_delta(delta_parameter)

    final_nll_canonical = float("nan")
    p_target_canonical = float("nan")
    final_nll_weighted = float("nan")
    p_target_weighted = float("nan")
    if return_recovery_result:
        try:
            canonical_unprefixed_index = (
                canonical_rewriting_original_prompts.index(request["prompt"])
            )
        except ValueError as error:
            raise RuntimeError(
                "HN recovery requires the canonical unprefixed prompt in the "
                "existing AlphaEdit context family"
            ) from error
        # Re-forward the actually selected delta.  The historical inner loop
        # can stop before an update or restore a best-RGR delta, so reusing the
        # last loop scalar would not be a valid adaptive selection statistic.
        with torch.no_grad():
            with nethook.TraceDict(
                module=model,
                layers=[hparams.layer_module_tmp.format(loss_layer)],
                retain_input=False,
                retain_output=True,
                edit_output=edit_output_fn,
            ) as final_trace:
                model(**input_tok)
            final_output = final_trace[
                hparams.layer_module_tmp.format(loss_layer)
            ].output
            if isinstance(final_output, tuple):
                final_output = final_output[0]
            if final_output.shape[1] != rewriting_targets.shape[1]:
                final_output = torch.transpose(final_output, 0, 1)
            final_repr = final_output[: len(rewriting_prompts)]
            final_log_probs = torch.log_softmax(
                ln_f(final_repr) @ lm_w.to(final_repr.device)
                + lm_b.to(final_repr.device),
                dim=2,
            )
            final_token_log_probs = torch.gather(
                final_log_probs,
                2,
                torch.where(
                    rewriting_targets != -100,
                    rewriting_targets,
                    0,
                )
                .unsqueeze(2)
                .to(final_log_probs.device),
            ).squeeze(2)
            final_semantic_mask = _semantic_rewriting_mask(
                rewriting_targets,
                total_target_token_count=int(target_ids.size(0)),
                semantic_target_token_count=semantic_target_token_count,
                device=final_token_log_probs.device,
            )
            final_nll_each = -(
                final_token_log_probs * final_semantic_mask
            ).sum(1) / semantic_target_token_count
            final_nll_canonical_tensor = final_nll_each[
                canonical_unprefixed_index
            ]
            if para_context_count:
                final_nll_weighted_tensor = _family_weighted_nll(
                    final_nll_each,
                    canonical_context_count=canonical_context_count,
                    para_context_count=para_context_count,
                    edit_weight=float(
                        getattr(
                            hparams,
                            "hn_recovery_edit_family_weight",
                            1.0,
                        )
                    ),
                    para_weight=float(
                        getattr(
                            hparams,
                            "hn_recovery_para_family_weight",
                            0.5,
                        )
                    ),
                )
            else:
                final_nll_weighted_tensor = final_nll_each.mean()
            final_nll_canonical = float(
                final_nll_canonical_tensor.detach().float().item()
            )
            final_nll_weighted = float(
                final_nll_weighted_tensor.detach().float().item()
            )
            p_target_canonical = math.exp(-final_nll_canonical)
            p_target_weighted = math.exp(-final_nll_weighted)
    nas_active = bool(
        getattr(hparams, "nas_enabled", False)
        or getattr(hparams, "nas_collect_stats", False)
    )
    if nas_active:
        _, v0 = get_module_input_output_at_words(
            model,
            tok,
            layer,
            context_templates=[all_prompts[0]],
            words=[request["subject"]],
            module_template=hparams.mlp_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )
        nas_result = apply_norm_anchor_scaling(
            target_init=target_init,
            delta=delta,
            v0=v0[0],
            hparams=hparams,
            method="AlphaEdit",
            request=request,
            layer=layer,
        )
        target = nas_result.target
        print(
            "[NAS][AlphaEdit] "
            f"case={request.get('case_id')} L{layer} "
            f"anchor={nas_result.anchor_norm} "
            f"v*={nas_result.vstar_norm_before:.6f}"
            f"->{nas_result.vstar_norm_after:.6f} "
            f"scale={nas_result.scale_factor:.6f} "
            f"safeguard={int(nas_result.safeguard_triggered)}"
        )
    else:
        target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )
    if persist_latent_artifact:
        save_latent_artifact(
            hparams,
            method="AlphaEdit",
            request=request,
            write_layer=layer,
            target_init=target_init,
            optimizer_delta=delta,
            target=target,
        )
        if early_attention is not None and bool(getattr(hparams, "analysis_capture_latents", False)):
            _save_early_attention_artifact(
                hparams, request, early_attention, delta, target_init.dtype
            )
        if o0_axis is not None and bool(getattr(hparams, "analysis_capture_latents", False)):
            _save_o0_axis_artifact(
                hparams, request, o0_axis, delta_parameter, delta, target, target_init.dtype
            )

    if o0_axis is not None:
        final_o0_diagnostics = o0_axis.delta_diagnostics(delta_parameter)
        final_o0_diagnostics.update(o0_axis.coefficient_diagnostics(o0_axis.reference + delta))
        append_rgr_record(
            getattr(hparams, "o0_axis_preservation_log_path", None),
            {"method": "AlphaEdit", "record_type": "selected",
             "edit_index": getattr(hparams, "analysis_current_edit_index", None),
             "case_id": request.get("case_id"), "write_layer": layer,
             "target_norm": float(target.detach().norm()), **final_o0_diagnostics},
        )

    if return_recovery_result:
        finite_values = (
            target.detach().isfinite().all().item(),
            delta.detach().isfinite().all().item(),
            math.isfinite(final_nll_canonical),
            math.isfinite(final_nll_weighted),
            math.isfinite(p_target_canonical),
            math.isfinite(p_target_weighted),
        )
        return _HNRecoverySolveResult(
            target=target,
            target_init=target_init.detach().clone(),
            optimizer_delta=delta.detach().clone(),
            final_nll_canonical=final_nll_canonical,
            p_target_canonical=p_target_canonical,
            final_nll_weighted=final_nll_weighted,
            p_target_weighted=p_target_weighted,
            optimization_context_count=len(rewriting_prompts),
            optimization_para_count=len(optimization_para_templates),
            optimization_para_ids=list(optimization_para_ids),
            solver_failed=not all(bool(value) for value in finite_values),
            canonical_subject_token_index=(
                int(lookup_idxs[0]) if bool(getattr(hparams, "analysis_capture_virtual_actual", False)) else None
            ),
            canonical_subject_prefix_token_ids=(
                input_tok["input_ids"][0, :lookup_idxs[0] + 1].detach().cpu().tolist()
                if bool(getattr(hparams, "analysis_capture_virtual_actual", False)) else None
            ),
        )
    if bool(getattr(hparams, "analysis_capture_virtual_actual", False)):
        _publish_selected_latent(
            model, tok, request, hparams, layer, target_init, delta, target,
            subject_index=int(lookup_idxs[0]),
            subject_prefix_ids=input_tok["input_ids"][0, :lookup_idxs[0] + 1].detach().cpu().tolist(),
            o0_axis_preservation_state=(o0_axis.export_state() if o0_axis is not None else None),
        )
    return target


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
