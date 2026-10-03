
# Release-only filesystem defaults; computation below follows the source snapshot.
import sys as _bng_sys
from pathlib import Path as _BNGPath
_bng_sys.path.insert(0, str(_BNGPath(__file__).resolve().parents[1]))
from model_code.paths import RESEARCH_ROOT, OUTPUT_ROOT, DATA_ROOT, LLAMA_MODEL
#!/usr/bin/env python3

import argparse
import ast
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml


BaseEditor = None
BatchEditor = None
compute_edit_quality = None
HPARAMS_REGISTRY: Dict[str, Any] = {}
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.edit_order_manifest import (  # noqa: E402
    apply_manifest_order,
    file_sha256 as edit_order_file_sha256,
    validate_manifest as validate_edit_order_manifest,
)
from diagnostics.target_text_contract import strip_training_terminators  # noqa: E402

ALG_ALIASES = {
    "alphaedit": "AlphaEdit",
    "memit": "MEMIT",
}

def add_easyedit_to_syspath(easyedit_path: Optional[str]) -> None:
    if easyedit_path is None:
        candidate_root = PROJECT_ROOT
    else:
        candidate_root = Path(easyedit_path).expanduser().resolve()

    if (candidate_root / "EasyEdit" / "easyeditor").is_dir():
        sys.path.insert(0, str(candidate_root))
        return

    if candidate_root.name == "EasyEdit" and (candidate_root / "easyeditor").is_dir():
        sys.path.insert(0, str(candidate_root.parent))
        return

    raise FileNotFoundError(
        f"Could not locate EasyEdit package from --easyedit_path={candidate_root}. "
        "Pass either the RGR repository root or the EasyEdit directory itself."
    )


def init_easyedit_imports() -> None:
    global BaseEditor, BatchEditor, HPARAMS_REGISTRY, compute_edit_quality

    from EasyEdit.easyeditor import (
        BaseEditor as _BaseEditor,
        AlphaEditHyperParams,
        MEMITHyperParams,
    )
    from EasyEdit.easyeditor.editors.batch_editor import BatchEditor as _BatchEditor
    from EasyEdit.easyeditor.evaluate import compute_edit_quality as _compute_edit_quality

    BaseEditor = _BaseEditor
    BatchEditor = _BatchEditor
    compute_edit_quality = _compute_edit_quality

    HPARAMS_REGISTRY = {
        "AlphaEdit": AlphaEditHyperParams,
        "MEMIT": MEMITHyperParams,
    }


def is_runner_supported_method(alg_name: str) -> bool:
    return BatchEditor.is_batchable_method(alg_name)


def fix_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except Exception:
        pass


def canonicalize_alg_name(name: str) -> str:
    if name in HPARAMS_REGISTRY:
        return name
    lowered = name.lower()
    if lowered in ALG_ALIASES:
        return ALG_ALIASES[lowered]
    raise ValueError(
        f"Unsupported editing method: {name}. "
        f"Supported methods: {', '.join(sorted(HPARAMS_REGISTRY))}"
    )


def infer_alg_name_from_yaml(hparams_path: str) -> str:
    with open(hparams_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    alg_name = config.get("alg_name") or config.get("alg")
    if alg_name is None:
        raise ValueError(f"Could not infer editing method from {hparams_path}")
    return canonicalize_alg_name(str(alg_name))


def resolve_device_arg(device_arg: Optional[str], default_device: int) -> int:
    if device_arg is None:
        return int(default_device)
    if device_arg == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, so --device auto cannot be resolved.")
        return torch.cuda.current_device()
    return int(device_arg)


def maybe_parse_literal_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    parsed = ast.literal_eval(value)
    if isinstance(parsed, int):
        return [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list literal, but got: {value}")
    return parsed


def maybe_parse_float_list(value: Optional[str]) -> Optional[List[float]]:
    """Parse either ``1,1.05`` or a Python-list literal into finite floats."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise ValueError("Expected a non-empty float list")
    parsed: Any
    if text.startswith("["):
        parsed = ast.literal_eval(text)
    else:
        parsed = [item.strip() for item in text.split(",")]
    if not isinstance(parsed, (list, tuple)):
        parsed = [parsed]
    result = [float(item) for item in parsed]
    if not result or any(not np.isfinite(item) for item in result):
        raise ValueError(f"Expected a non-empty finite float list, got: {value}")
    return result


def normalize_hn_recovery_adaptive_ladder(value: Any) -> List[float]:
    """Materialize the protocol default when the dataclass default is None."""

    return list([1.0, 1.05, 1.1, 1.15, 1.25] if value is None else value)


def apply_hparam_overrides(hparams: Any, args: argparse.Namespace) -> None:
    if hasattr(hparams, "model_parallel") and hparams.model_parallel and args.device is not None:
        print(
            f"[config] Ignoring --device={args.device} because this hparams file uses model_parallel=True."
        )
    elif hasattr(hparams, "device"):
        hparams.device = resolve_device_arg(args.device, getattr(hparams, "device", 0))

    if args.batch_size is not None:
        hparams.batch_size = int(args.batch_size)

    if args.model_name is not None:
        hparams.model_name = args.model_name

    if args.layers is not None and hasattr(hparams, "layers"):
        hparams.layers = args.layers

    if args.append_eos_to_target is not None and hasattr(hparams, "append_eos_to_target"):
        hparams.append_eos_to_target = bool(args.append_eos_to_target)

    if args.context_num is not None and hasattr(hparams, "context_template_length_params"):
        if args.context_num == 0:
            hparams.context_template_length_params = "None"
        else:
            half = int(args.context_num // 2)
            hparams.context_template_length_params = [[5, half], [10, half]]

    if args.p_loc is not None and hasattr(hparams, "P_loc"):
        hparams.P_loc = args.p_loc
    if args.v_num_grad_steps is not None:
        hparams.v_num_grad_steps = int(args.v_num_grad_steps)
    if args.v_lr is not None:
        hparams.v_lr = float(args.v_lr)
    if args.mom2_n_samples is not None:
        hparams.mom2_n_samples = int(args.mom2_n_samples)
    if args.alphaedit_l2 is not None:
        if not hasattr(hparams, "L2"):
            raise ValueError("--alphaedit_l2 is only valid for AlphaEdit")
        hparams.L2 = float(args.alphaedit_l2)
    if args.context_multikey_enabled is not None:
        if not hasattr(hparams, "context_multikey_enabled"):
            raise ValueError(
                "--context_multikey_enabled is only valid for AlphaEdit"
            )
        hparams.context_multikey_enabled = bool(
            args.context_multikey_enabled
        )
        hparams.context_multikey_expected_group_sizes = (
            [1, 5] if bool(args.context_multikey_enabled) else None
        )
    if args.key_gaussian_noise_enabled is not None:
        if not hasattr(hparams, "key_gaussian_noise_enabled"):
            raise ValueError(
                "--key_gaussian_noise_enabled is only valid for MEMIT/AlphaEdit"
            )
        hparams.key_gaussian_noise_enabled = bool(
            args.key_gaussian_noise_enabled
        )
        if bool(args.key_gaussian_noise_enabled) and (
            args.key_gaussian_noise_seed is None
        ):
            hparams.key_gaussian_noise_seed = int(args.seed)
    if args.key_gaussian_noise_relative_std is not None:
        if not hasattr(hparams, "key_gaussian_noise_relative_std"):
            raise ValueError(
                "--key_gaussian_noise_relative_std is only valid for MEMIT/AlphaEdit"
            )
        hparams.key_gaussian_noise_relative_std = float(
            args.key_gaussian_noise_relative_std
        )
    if args.key_gaussian_noise_seed is not None:
        if not hasattr(hparams, "key_gaussian_noise_seed"):
            raise ValueError(
                "--key_gaussian_noise_seed is only valid for MEMIT/AlphaEdit"
            )
        hparams.key_gaussian_noise_seed = int(args.key_gaussian_noise_seed)
    if bool(getattr(hparams, "key_gaussian_noise_enabled", False)):
        if bool(getattr(hparams, "context_multikey_enabled", False)):
            raise ValueError(
                "--key_gaussian_noise_enabled and "
                "--context_multikey_enabled are mutually exclusive"
            )
        relative_std = float(hparams.key_gaussian_noise_relative_std)
        if not np.isfinite(relative_std) or relative_std <= 0.0:
            raise ValueError(
                "--key_gaussian_noise_relative_std must be finite and > 0"
            )
    endogenous_overrides = {
        "endogenous_pivot_enabled": args.endogenous_pivot_enabled,
        "endogenous_pivot_mode": args.endogenous_pivot_mode,
        "endogenous_pivot_trainable_layers": (
            args.endogenous_pivot_trainable_layers
        ),
        "endogenous_pivot_num_steps": args.endogenous_pivot_num_steps,
        "endogenous_pivot_num_sweeps": args.endogenous_pivot_num_sweeps,
        "endogenous_pivot_lr": args.endogenous_pivot_lr,
        "endogenous_pivot_l2_lambda": args.endogenous_pivot_l2_lambda,
        "endogenous_pivot_grad_clip_norm": args.endogenous_pivot_grad_clip_norm,
        "endogenous_pivot_capture_trajectory": (
            args.endogenous_pivot_capture_trajectory
        ),
    }
    supplied_endogenous = {
        name: value for name, value in endogenous_overrides.items() if value is not None
    }
    if supplied_endogenous and not hasattr(hparams, "endogenous_pivot_enabled"):
        raise ValueError("Endogenous-pivot refinement is only valid for MEMIT")
    for name, value in supplied_endogenous.items():
        if name in {"endogenous_pivot_enabled", "endogenous_pivot_capture_trajectory"}:
            value = bool(value)
        elif name in {"endogenous_pivot_num_steps", "endogenous_pivot_num_sweeps"}:
            value = int(value)
        elif name == "endogenous_pivot_trainable_layers":
            value = [int(layer) for layer in value]
        elif name in {
            "endogenous_pivot_lr",
            "endogenous_pivot_l2_lambda",
            "endogenous_pivot_grad_clip_norm",
        }:
            value = float(value)
        setattr(hparams, name, value)
    non_enable_endogenous = {
        name: value
        for name, value in supplied_endogenous.items()
        if name != "endogenous_pivot_enabled"
    }
    if non_enable_endogenous and not bool(
        getattr(hparams, "endogenous_pivot_enabled", False)
    ):
        raise ValueError(
            "Endogenous-pivot options were supplied while refinement is disabled; "
            "also pass --endogenous-pivot-enabled 1"
        )
    if args.residual_gain_regularization is not None:
        hparams.residual_gain_regularization = bool(
            args.residual_gain_regularization
        )
    if args.residual_gain_lambda is not None:
        hparams.residual_gain_lambda = float(args.residual_gain_lambda)
    if args.residual_gain_layers is not None:
        hparams.residual_gain_layers = args.residual_gain_layers
    if args.residual_gain_token_scope is not None:
        hparams.residual_gain_token_scope = args.residual_gain_token_scope
    if args.residual_gain_subject_layers is not None:
        hparams.residual_gain_subject_layers = args.residual_gain_subject_layers
    if args.residual_gain_prompt_layers is not None:
        hparams.residual_gain_prompt_layers = args.residual_gain_prompt_layers
    if args.residual_gain_subject_lambda is not None:
        hparams.residual_gain_subject_lambda = float(
            args.residual_gain_subject_lambda
        )
    if args.residual_gain_prompt_lambda is not None:
        hparams.residual_gain_prompt_lambda = float(
            args.residual_gain_prompt_lambda
        )
    if args.residual_gain_margin is not None:
        hparams.residual_gain_margin = float(args.residual_gain_margin)
    if args.residual_gain_loss_type is not None:
        hparams.residual_gain_loss_type = args.residual_gain_loss_type
    if args.residual_gain_objective is not None:
        hparams.residual_gain_objective = args.residual_gain_objective
    if args.residual_gain_alignment_weight is not None:
        hparams.residual_gain_alignment_weight = float(
            args.residual_gain_alignment_weight
        )
    if args.residual_gain_cosine_aux_lambda is not None:
        hparams.residual_gain_cosine_aux_lambda = float(
            args.residual_gain_cosine_aux_lambda
        )
    if args.residual_gain_cosine_aux_sharpness is not None:
        hparams.residual_gain_cosine_aux_sharpness = float(
            args.residual_gain_cosine_aux_sharpness
        )
    if args.residual_gain_efficacy_threshold is not None:
        hparams.residual_gain_efficacy_threshold = float(
            args.residual_gain_efficacy_threshold
        )
    if args.residual_gain_select_best is not None:
        hparams.residual_gain_select_best = bool(args.residual_gain_select_best)
    if args.residual_gain_selection_threshold is not None:
        hparams.residual_gain_selection_threshold = float(
            args.residual_gain_selection_threshold
        )
    if args.residual_gain_early_stop_mode is not None:
        hparams.residual_gain_early_stop_mode = (
            args.residual_gain_early_stop_mode
        )
    if args.residual_gain_target_ratio is not None:
        hparams.residual_gain_target_ratio = float(
            args.residual_gain_target_ratio
        )
    if args.hn_recovery_paraphrase_aware is not None:
        hparams.hn_recovery_paraphrase_aware = bool(
            args.hn_recovery_paraphrase_aware
        )
    if args.hn_recovery_edit_family_weight is not None:
        hparams.hn_recovery_edit_family_weight = float(
            args.hn_recovery_edit_family_weight
        )
    if args.hn_recovery_para_family_weight is not None:
        hparams.hn_recovery_para_family_weight = float(
            args.hn_recovery_para_family_weight
        )
    if args.hn_recovery_adaptive_rho is not None:
        hparams.hn_recovery_adaptive_rho = bool(
            args.hn_recovery_adaptive_rho
        )
    if args.hn_recovery_adaptive_threshold is not None:
        hparams.hn_recovery_adaptive_threshold = float(
            args.hn_recovery_adaptive_threshold
        )
    if args.hn_recovery_adaptive_ladder is not None:
        hparams.hn_recovery_adaptive_ladder = list(
            args.hn_recovery_adaptive_ladder
        )
    if args.sadr_regularization is not None:
        hparams.sadr_regularization = bool(args.sadr_regularization)
    if args.sadr_lambda is not None:
        hparams.sadr_lambda = float(args.sadr_lambda)
    if args.sadr_attn_layers is not None:
        hparams.sadr_attn_layers = args.sadr_attn_layers
    if args.sadr_efficacy_threshold is not None:
        hparams.sadr_efficacy_threshold = float(
            args.sadr_efficacy_threshold
        )
    if args.encore_enabled is not None:
        hparams.encore_enabled = bool(args.encore_enabled)
    if args.encore_mpes_top1_steps is not None:
        hparams.encore_mpes_top1_steps = int(
            args.encore_mpes_top1_steps
        )
    if args.encore_mpes_exclude_first_context is not None:
        hparams.encore_mpes_exclude_first_context = bool(
            args.encore_mpes_exclude_first_context
        )
    if args.encore_norm_lambda is not None:
        hparams.encore_norm_lambda = float(args.encore_norm_lambda)
    if args.nse_enabled is not None:
        hparams.nse_enabled = bool(args.nse_enabled)
    if args.nse_alpha is not None:
        hparams.nse_alpha = float(args.nse_alpha)
    if args.nse_upper_bound is not None:
        hparams.nse_upper_bound = int(args.nse_upper_bound)
    if args.nse_max_iterations is not None:
        hparams.nse_max_iterations = int(args.nse_max_iterations)
    if args.nse_neuron_threshold is not None:
        hparams.nse_neuron_threshold = float(args.nse_neuron_threshold)
    if args.nse_target_cache_dir is not None:
        hparams.nse_target_cache_dir = args.nse_target_cache_dir
    if args.sphere_enabled is not None:
        hparams.sphere_enabled = bool(args.sphere_enabled)
    if args.sphere_beta is not None:
        hparams.sphere_beta = float(args.sphere_beta)
    if args.sphere_alpha is not None:
        hparams.sphere_alpha = float(args.sphere_alpha)
    if args.nas_enabled is not None:
        hparams.nas_enabled = bool(args.nas_enabled)
    if args.nas_collect_stats is not None:
        hparams.nas_collect_stats = bool(args.nas_collect_stats)
    if args.nas_anchor_path is not None:
        hparams.nas_anchor_path = args.nas_anchor_path
    if args.nas_anchor_norm is not None:
        hparams.nas_anchor_norm = float(args.nas_anchor_norm)
    if args.nas_outlier_factor is not None:
        hparams.nas_outlier_factor = float(args.nas_outlier_factor)
    if args.nas_outlier_mode is not None:
        hparams.nas_outlier_mode = args.nas_outlier_mode


def normalize_hparam_paths(hparams: Any, project_root: Path) -> None:
    for attr in (
        "stats_dir",
        "P_loc",
        "save_path",
        "load_path",
        "nse_target_cache_dir",
        "nas_anchor_path",
    ):
        if not hasattr(hparams, attr):
            continue
        raw_value = getattr(hparams, attr)
        if raw_value is None:
            continue
        path = Path(os.path.expandvars(str(raw_value))).expanduser()
        if path.is_absolute():
            setattr(hparams, attr, str(path.resolve()))
            continue
        setattr(hparams, attr, str((project_root / path).resolve()))


def first_present(record: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def normalize_ground_truth(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("str", "text", "value"):
            if key in value and value[key] is not None:
                return str(value[key])
        return "<|endoftext|>"
    if isinstance(value, list):
        return str(value[0]) if value else "<|endoftext|>"
    if value is None:
        return "<|endoftext|>"
    return str(value)


def normalize_target_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("str", "text", "value"):
            if key in value and value[key] is not None:
                return str(value[key])
        raise ValueError(f"Unsupported target_new dict format: {value}")
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def resolve_tokenizer_eos_token(tokenizer: Any) -> str:
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token:
        return str(eos_token)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0] if eos_token_id else None
    if eos_token_id is not None and hasattr(tokenizer, "decode"):
        return str(tokenizer.decode([int(eos_token_id)], skip_special_tokens=False))

    raise ValueError(
        "--append_eos_to_target=1 was requested, but the tokenizer has no eos_token or eos_token_id."
    )


def append_eos_to_request_targets(
    requests: List[Dict[str, Any]],
    tokenizer: Any,
) -> Tuple[List[Dict[str, Any]], str, int]:
    eos_token = resolve_tokenizer_eos_token(tokenizer)
    appended_count = 0

    for request in requests:
        target_new = str(request["target_new"])
        if not target_new.endswith(eos_token):
            request["target_new"] = target_new + eos_token
            appended_count += 1

    return requests, eos_token, appended_count


def prepare_nse_original_targets(
    *,
    editor: Any,
    requests: List[Dict[str, Any]],
    hparams: Any,
    alg_name: str,
    output_dir: str,
) -> None:
    """Cache every NSE v* before the first sequential model update."""

    from EasyEdit.easyeditor.util.official_editing_baselines import (
        nse_target_cache_file,
        reset_nse_runtime,
        store_nse_target,
    )

    if alg_name == "MEMIT":
        from EasyEdit.easyeditor.models.memit.compute_z import compute_z
        from EasyEdit.easyeditor.models.memit.memit_main import (
            get_context_templates,
        )
    elif alg_name == "AlphaEdit":
        from EasyEdit.easyeditor.models.alphaedit.compute_z import compute_z
        from EasyEdit.easyeditor.models.alphaedit.AlphaEdit_main import (
            get_context_templates,
        )
    else:
        raise ValueError("NSE precomputation supports MEMIT/AlphaEdit only")

    reset_nse_runtime(hparams, clear_targets=True, clear_grams=True)
    cache_dir = (
        str(hparams.nse_target_cache_dir)
        if getattr(hparams, "nse_target_cache_dir", None)
        else os.path.join(output_dir, "nse_original_targets")
    )
    os.makedirs(cache_dir, exist_ok=True)
    hparams.nse_target_cache_dir = cache_dir
    context_templates = get_context_templates(editor.model, editor.tok)
    z_layer = int(hparams.layers[-1])

    for index, raw_request in enumerate(requests, start=1):
        request = dict(raw_request)
        if request["target_new"][0] != " ":
            request["target_new"] = " " + request["target_new"]
        if "{}" not in request["prompt"]:
            if request["subject"] not in request["prompt"]:
                raise ValueError(
                    f"NSE cannot template case {request['case_id']}: "
                    f"subject {request['subject']!r} is absent"
                )
            request["prompt"] = request["prompt"].replace(
                request["subject"], "{}", 1
            )
        cache_file = nse_target_cache_file(
            cache_dir,
            layer=z_layer,
            clamp_norm_factor=float(hparams.clamp_norm_factor),
            case_id=request["case_id"],
        )
        target = None
        source = "computed"
        if os.path.isfile(cache_file):
            try:
                payload = torch.load(
                    cache_file,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                payload = torch.load(cache_file, map_location="cpu")
            if (
                payload.get("method") == alg_name
                and payload.get("target_new") == request["target_new"]
                and payload.get("model_name") == str(hparams.model_name)
            ):
                target = payload["v_star"]
                source = "cache"
        if target is None:
            target = compute_z(
                editor.model,
                editor.tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )
            torch.save(
                {
                    "v_star": target.detach().float().cpu(),
                    "case_id": request["case_id"],
                    "z_layer": z_layer,
                    "clamp_norm_factor": float(
                        hparams.clamp_norm_factor
                    ),
                    "method": alg_name,
                    "target_new": request["target_new"],
                    "model_name": str(hparams.model_name),
                },
                cache_file,
            )
        store_nse_target(hparams, request["case_id"], target)
        print(
            f"[NSE][precompute] {index}/{len(requests)} "
            f"case={request['case_id']} source={source}"
        )


def normalize_eval_group(raw_group: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw_group, dict):
        return {}

    if "prompt" in raw_group and ("ground_truth" in raw_group or "answer" in raw_group):
        ground_truth = first_present(raw_group, ("ground_truth", "answer"))
        return {
            "default": {
                "prompt": raw_group["prompt"],
                "ground_truth": ground_truth,
            }
        }

    normalized: Dict[str, Dict[str, Any]] = {}
    for key, value in raw_group.items():
        if not isinstance(value, dict):
            continue
        prompt = first_present(value, ("prompt", "src"))
        ground_truth = first_present(value, ("ground_truth", "answer", "answers"))
        if prompt is None or ground_truth is None:
            continue
        normalized[key] = {
            "prompt": prompt,
            "ground_truth": ground_truth,
        }
    return normalized


def build_request(record: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    official_rewrite = record.get("requested_rewrite")
    if isinstance(official_rewrite, dict):
        prompt = official_rewrite.get("prompt")
        target_new = official_rewrite.get("target_new")
        subject = official_rewrite.get("subject")
        target_true = official_rewrite.get("target_true")
        if prompt is None or target_new is None or subject is None:
            return None

        target_true_text = normalize_ground_truth(target_true)
        paraphrases = [
            str(item)
            for item in record.get("paraphrase_prompts", [])
            if item is not None and str(item).strip()
        ]
        neighborhoods = [
            str(item)
            for item in record.get("neighborhood_prompts", [])
            if item is not None and str(item).strip()
        ]
        normalized_target = normalize_target_text(target_new)
        if not strip_training_terminators(normalized_target):
            return None
        request = {
            "case_id": first_present(record, ("case_id", "id"), idx),
            "prompt": str(prompt),
            "target_new": normalized_target,
            "ground_truth": target_true_text,
            "subject": str(subject),
            "loc_prompt": str(subject),
            "portability": {},
            "locality": {},
            # Preserve the full official prompt collections for post-hoc
            # AlphaEdit/MEMIT evaluation while keeping EasyEdit's scalar
            # rephrase field compatible with existing code.
            "paraphrase_prompts": paraphrases,
            "neighborhood_prompts": neighborhoods,
            "generation_prompts": [
                str(item)
                for item in record.get("generation_prompts", [])
                if item is not None and str(item).strip()
            ],
        }
        if paraphrases:
            request["rephrase_prompt"] = paraphrases[0]
        if neighborhoods and target_true_text:
            request["locality"]["neighborhood"] = {
                "prompt": neighborhoods,
                "ground_truth": [target_true_text] * len(neighborhoods),
            }
        return request

    prompt = first_present(record, ("prompt", "src"))
    target_new = first_present(record, ("target_new", "alt"))

    if prompt is None or target_new is None:
        return None

    normalized_target = normalize_target_text(target_new)
    if not strip_training_terminators(normalized_target):
        return None
    request: Dict[str, Any] = {
        "case_id": first_present(record, ("case_id", "id"), idx),
        "prompt": str(prompt),
        "target_new": normalized_target,
        "ground_truth": normalize_ground_truth(
            first_present(record, ("ground_truth", "pred", "answers"))
        ),
        "portability": {},
        "locality": {},
    }

    subject = first_present(record, ("subject",))
    if subject is not None:
        request["subject"] = str(subject)
        request.setdefault("loc_prompt", str(subject))

    loc_prompt = first_present(record, ("loc_prompt",))
    if loc_prompt is not None:
        request["loc_prompt"] = str(loc_prompt)

    rephrase_prompt = first_present(record, ("rephrase_prompt", "rephrase"))
    if rephrase_prompt is not None:
        request["rephrase_prompt"] = str(rephrase_prompt)

    locality_prompt = first_present(record, ("locality_prompt", "loc"))
    locality_ground_truth = first_present(record, ("locality_ground_truth", "loc_ans"))
    if locality_prompt is not None and locality_ground_truth is not None:
        request["locality"]["neighborhood"] = {
            "prompt": locality_prompt,
            "ground_truth": locality_ground_truth,
        }

    request["locality"].update(normalize_eval_group(record.get("locality")))
    request["portability"].update(normalize_eval_group(record.get("portability")))

    return request


def load_requests(data_path: str) -> Tuple[List[Dict[str, Any]], int]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "data" not in data or not isinstance(data["data"], list):
            raise ValueError(f"Unsupported JSON structure in {data_path}")
        data = data["data"]

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {data_path}")

    requests: List[Dict[str, Any]] = []
    for idx, record in enumerate(data):
        request = build_request(record, idx)
        if request is not None:
            requests.append(request)

    return requests, len(data)


def maybe_sample_requests(
    requests: List[Dict[str, Any]],
    sample_size: Optional[int],
    seed: int,
    selection: str = "prefix",
) -> List[Dict[str, Any]]:
    if sample_size is None or sample_size <= 0:
        return requests
    if sample_size > len(requests):
        raise ValueError(
            f"Requested {sample_size} edits but the dataset contains only "
            f"{len(requests)} usable requests. Supply enough data or set --sample_size explicitly."
        )

    if selection == "prefix":
        return requests[:sample_size]
    if selection != "random":
        raise ValueError(
            f"selection must be 'prefix' or 'random', got {selection!r}"
        )
    rng = random.Random(seed)
    sampled = rng.sample(requests, sample_size)
    sampled.sort(key=lambda x: int(x["case_id"]) if str(x["case_id"]).isdigit() else str(x["case_id"]))
    return sampled


def resolve_request_order(
    requests: List[Dict[str, Any]],
    *,
    raw_count: int,
    data_path: str,
    sample_size: Optional[int],
    seed: int,
    selection: str,
    edit_order_file: Optional[str],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Select requests and optionally apply a frozen fixed-prefix order.

    The order manifest is defined over raw source records.  Applying its
    permutation to normalized requests is sound only when normalization is a
    one-to-one, order-preserving mapping, so manifest-backed runs fail closed
    if even one raw record was filtered.  The manifest implementation uses a
    local ``random.Random`` instance and therefore does not advance the model
    RNG initialized from ``seed``.
    """

    if not edit_order_file:
        return (
            maybe_sample_requests(requests, sample_size, seed, selection),
            None,
        )

    if selection != "prefix":
        raise ValueError(
            "--edit_order_file is only valid with --selection prefix; "
            "the manifest already declares the exact request order."
        )
    if len(requests) != int(raw_count):
        raise ValueError(
            "Cannot apply a raw-record edit-order manifest because request "
            "normalization filtered records: "
            f"raw_records={raw_count}, usable_requests={len(requests)}."
        )

    requested_sample_size = (
        int(sample_size)
        if sample_size is not None and int(sample_size) > 0
        else None
    )
    permutation, metadata = validate_edit_order_manifest(
        Path(edit_order_file),
        data_path=Path(data_path),
        sample_size=requested_sample_size,
    )
    manifest_sample_size = int(metadata["sample_size"])
    if manifest_sample_size > len(requests):
        raise ValueError(
            "Edit-order manifest selects more records than normalization "
            f"produced: manifest={manifest_sample_size}, usable={len(requests)}."
        )

    selected_prefix = requests[:manifest_sample_size]
    ordered_requests = apply_manifest_order(selected_prefix, permutation)
    return ordered_requests, metadata


def archive_edit_order_manifest(
    *,
    edit_order_file: Optional[str],
    metadata: Optional[Dict[str, Any]],
    output_dir: str,
) -> Optional[Dict[str, Any]]:
    """Archive the validated manifest byte-for-byte in the run directory."""

    if edit_order_file is None:
        if metadata is not None:
            raise ValueError("Edit-order metadata exists without a manifest path")
        return None
    if metadata is None:
        raise ValueError("Edit-order manifest path exists without validated metadata")

    source = Path(edit_order_file).expanduser().resolve()
    expected_sha256 = str(metadata["sha256"])
    actual_source_sha256 = edit_order_file_sha256(source)
    if actual_source_sha256 != expected_sha256:
        raise ValueError(
            "Edit-order manifest changed after validation: "
            f"expected={expected_sha256}, actual={actual_source_sha256}."
        )

    archived = Path(output_dir).resolve() / "edit_order_manifest.json"
    if archived.exists():
        archived_sha256 = edit_order_file_sha256(archived)
        if archived_sha256 != expected_sha256:
            raise FileExistsError(
                "Refusing to overwrite a different archived edit-order "
                f"manifest: {archived}"
            )
    elif archived != source:
        shutil.copy2(source, archived)

    archived_sha256 = edit_order_file_sha256(archived)
    if archived_sha256 != expected_sha256:
        raise RuntimeError(
            "Archived edit-order manifest hash mismatch: "
            f"expected={expected_sha256}, actual={archived_sha256}."
        )
    archived_metadata = dict(metadata)
    archived_metadata.update(
        {
            "archived_path": str(archived),
            "archived_sha256": archived_sha256,
        }
    )
    return archived_metadata


HN_RECOVERY_SIDECAR_SCHEMA = "alphaedit-hn-optimization-paraphrase-sidecar-v1"


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def attach_hn_recovery_metadata(
    requests: List[Dict[str, Any]],
    *,
    args: argparse.Namespace,
    data_path: str,
) -> Optional[Dict[str, Any]]:
    """Bind run identity and an audited optimization-only paraphrase sidecar.

    This loader deliberately recognizes one strict sidecar schema.  It never
    falls back to ``rephrase_prompt`` because that field is the official ZsRE
    generalization evaluation prompt.
    """

    # Gaussian-key seeding shares HN recovery's one-based stream identity.
    # Attach it before the recovery-only early return, while leaving ordinary
    # feature-off requests exactly as they were before this option existed.
    if bool(args.key_gaussian_noise_enabled):
        for edit_index, request in enumerate(requests, start=1):
            request["edit_index"] = edit_index

    identity_values = {
        "protocol_id": args.hn_recovery_protocol_id,
        "experiment_id": args.hn_recovery_experiment_id,
        "arm_id": args.hn_recovery_arm_id,
        "order_id": args.hn_recovery_order_id,
        "order_seed": args.hn_recovery_order_seed,
    }
    supplied = {key for key, value in identity_values.items() if value is not None}
    if supplied and supplied != set(identity_values):
        missing = sorted(set(identity_values) - supplied)
        raise ValueError(f"Incomplete HN recovery run identity; missing={missing}")

    recovery_requested = bool(
        supplied
        or args.residual_gain_target_ratio is not None
        or args.hn_recovery_paraphrase_aware
        or args.hn_recovery_adaptive_rho
        or args.hn_recovery_optimization_paraphrase_sidecar
    )
    if not recovery_requested:
        return None

    sidecar_path_text = args.hn_recovery_optimization_paraphrase_sidecar
    para_aware = bool(args.hn_recovery_paraphrase_aware)
    if para_aware and not sidecar_path_text:
        raise ValueError(
            "Paraphrase-aware HN requires an audited "
            "--hn-recovery-optimization-paraphrase-sidecar"
        )
    if sidecar_path_text and not para_aware:
        raise ValueError(
            "An optimization-paraphrase sidecar is only valid with "
            "--hn-recovery-paraphrase-aware=1"
        )

    data_sha256 = edit_order_file_sha256(Path(data_path).expanduser().resolve())
    sidecar_metadata: Optional[Dict[str, Any]] = None
    by_case: Dict[str, Dict[str, Any]] = {}
    if sidecar_path_text:
        sidecar_path = Path(sidecar_path_text).expanduser().resolve()
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"Optimization paraphrase sidecar missing: {sidecar_path}")
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != HN_RECOVERY_SIDECAR_SCHEMA:
            raise ValueError(
                "Unexpected optimization paraphrase sidecar schema: "
                f"{payload.get('schema') if isinstance(payload, dict) else type(payload)}"
            )
        if payload.get("dataset_sha256") != data_sha256:
            raise ValueError("Optimization paraphrase sidecar dataset SHA mismatch")
        records = payload.get("records")
        scope_size = int(payload.get("scope_size", -1))
        if not isinstance(records, list) or scope_size != len(records):
            raise ValueError("Optimization paraphrase sidecar scope/record count mismatch")
        request_indices = [record.get("request_index") for record in records if isinstance(record, dict)]
        if request_indices != list(range(scope_size)):
            raise ValueError("Optimization paraphrase sidecar must cover exact source indices 0..N-1")
        for record in records:
            case_id = str(record.get("case_id"))
            if case_id in by_case:
                raise ValueError(f"Duplicate optimization paraphrase case_id={case_id}")
            variants = record.get("optimization_paraphrases")
            if (
                not isinstance(variants, list)
                or len(variants) != 1
                or not isinstance(variants[0], str)
                or not variants[0].strip()
            ):
                raise ValueError(f"Expected exactly one optimization paraphrase for case_id={case_id}")
            by_case[case_id] = record
        sidecar_metadata = {
            "path": str(sidecar_path),
            "sha256": edit_order_file_sha256(sidecar_path),
            "schema": HN_RECOVERY_SIDECAR_SCHEMA,
            "scope_size": scope_size,
            "input_records_sha256": payload.get("input_records_sha256"),
            "source_semantic_variants_sha256": payload.get(
                "source_semantic_variants_sha256"
            ),
        }

    for edit_index, request in enumerate(requests, start=1):
        request["edit_index"] = edit_index
        request["source_index"] = (
            int(request["case_id"])
            if str(request.get("case_id", "")).isdigit()
            else None
        )
        request["model_seed"] = int(args.seed)
        request["data_sha256"] = data_sha256
        for key, value in identity_values.items():
            if value is not None:
                request[key] = value
        if sidecar_metadata is None:
            continue
        record = by_case.get(str(request.get("case_id")))
        if record is None:
            raise ValueError(
                "Optimization paraphrase sidecar has no row for ordered "
                f"case_id={request.get('case_id')}"
            )
        if record.get("subject") != request.get("subject"):
            raise ValueError(
                f"Optimization paraphrase subject mismatch for case_id={request.get('case_id')}"
            )
        expected_prompt_sha = _text_sha256(str(request.get("prompt", "")))
        if record.get("source_prompt_sha256") != expected_prompt_sha:
            raise ValueError(
                f"Optimization paraphrase source prompt mismatch for case_id={request.get('case_id')}"
            )
        text_value = str(record["optimization_paraphrases"][0])
        official_rephrase = request.get("rephrase_prompt")
        if official_rephrase is not None and text_value == str(official_rephrase):
            raise ValueError(
                f"Official rephrase leaked into optimization for case_id={request.get('case_id')}"
            )
        request["optimization_paraphrase"] = text_value
        request["optimization_para_ids"] = [f"{record['case_id']}:0"]
        request["optimization_para_sha256"] = _text_sha256(text_value)
        request["optimization_paraphrase_sidecar_sha256"] = sidecar_metadata["sha256"]

    if sidecar_metadata is not None and len(requests) > int(sidecar_metadata["scope_size"]):
        raise ValueError("Optimization paraphrase sidecar does not cover every selected request")
    return sidecar_metadata


def validate_method_specific_constraints(
    alg_name: str,
    hparams: Any,
    requests: List[Dict[str, Any]],
) -> None:
    unsupported = [
        req.get("case_id")
        for req in requests
        if "subject" not in req or "prompt" not in req or "target_new" not in req
    ]
    if unsupported:
        raise ValueError(
            f"{alg_name} requires prompt, subject, and target_new for every "
            f"request; invalid case_ids={unsupported[:10]}"
        )


def chunk_list(items: List[Any], size: int) -> Iterable[List[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def collapse_locality_outputs(
    pre_eval: Optional[Dict[str, Any]],
    post_eval: Optional[Dict[str, Any]],
    request: Dict[str, Any],
    evaluation_type: Optional[str],
) -> None:
    if not pre_eval or not post_eval:
        return
    if "locality" not in post_eval or not post_eval["locality"]:
        return

    for locality_key in request.get("locality", {}).keys():
        output_key = f"{locality_key}_output"
        if output_key not in post_eval["locality"] or output_key not in pre_eval.get("locality", {}):
            continue

        post_output = post_eval["locality"][output_key]
        pre_output = pre_eval["locality"][output_key]

        if evaluation_type == "LLM-judge":
            acc = [float(post_output == pre_output)]
        else:
            if not isinstance(post_output, list):
                post_output = [post_output]
            if not isinstance(pre_output, list):
                pre_output = [pre_output]
            acc = [float(np.mean(np.equal(a, b))) for a, b in zip(post_output, pre_output)]

        post_eval["locality"][f"{locality_key}_acc"] = acc
        post_eval["locality"].pop(output_key, None)

    if "locality" in pre_eval:
        pre_eval.pop("locality", None)


def flatten_numeric_metrics(obj: Any, prefix: str = "") -> Dict[str, float]:
    flat: Dict[str, float] = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            flat.update(flatten_numeric_metrics(value, next_prefix))
        return flat

    if isinstance(obj, (bool, np.bool_)):
        flat[prefix] = float(obj)
        return flat

    if isinstance(obj, (int, float, np.integer, np.floating)):
        flat[prefix] = float(obj)
        return flat

    if isinstance(obj, list) and obj and all(
        isinstance(x, (bool, int, float, np.integer, np.floating, np.bool_)) for x in obj
    ):
        flat[prefix] = float(np.mean(obj))
        return flat

    return flat


def summarize_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_requests": len(metrics),
        "num_batches": len({m["batch_index"] for m in metrics}) if metrics else 0,
    }

    if metrics:
        summary["mean_edit_time_sec"] = float(np.mean([m["edit_time_sec"] for m in metrics]))

    for stage in ("pre", "post"):
        collected: Dict[str, List[float]] = {}
        for metric in metrics:
            if stage not in metric or metric[stage] is None:
                continue
            for key, value in flatten_numeric_metrics(metric[stage]).items():
                collected.setdefault(key, []).append(value)
        if collected:
            summary[stage] = {
                key: float(np.mean(values))
                for key, values in sorted(collected.items())
            }
    return summary


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize runner JSON deterministically for artifact hash binding."""

    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=json_default,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def atomic_write_json(path: Any, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(canonical_json_bytes(value))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_delta_checkpoint_interval(
    interval: int,
    *,
    batch_size: int,
    checkpointing_enabled: bool,
) -> int:
    normalized = int(interval)
    if normalized < 0:
        raise ValueError("--delta_checkpoint_interval must be >= 0")
    if normalized > 0 and not checkpointing_enabled:
        raise ValueError(
            "--delta_checkpoint_interval requires --save_delta_checkpoint=1"
        )
    if normalized > 0 and normalized % int(batch_size) != 0:
        raise ValueError(
            "--delta_checkpoint_interval must be divisible by batch_size "
            f"({batch_size}); got {normalized}"
        )
    return normalized


def validate_delta_checkpoint_steps(
    steps: Optional[List[int]],
    *,
    batch_size: int,
    sample_size: int,
    checkpointing_enabled: bool,
    interval: int,
) -> List[int]:
    if steps is None:
        return []
    normalized = [int(step) for step in steps]
    if not checkpointing_enabled:
        raise ValueError("--delta_checkpoint_steps requires --save_delta_checkpoint=1")
    if interval > 0:
        raise ValueError(
            "--delta_checkpoint_steps is mutually exclusive with a nonzero "
            "--delta_checkpoint_interval"
        )
    if not normalized or normalized != sorted(set(normalized)):
        raise ValueError("--delta_checkpoint_steps must be non-empty, sorted, and unique")
    invalid = [
        step
        for step in normalized
        if step <= 0 or step > int(sample_size) or step % int(batch_size) != 0
    ]
    if invalid:
        raise ValueError(
            "Checkpoint steps must be positive batch boundaries no larger "
            f"than the request count ({sample_size}); invalid={invalid}"
        )
    return normalized


def save_model_and_tokenizer(
    model: Any,
    tokenizer: Any,
    save_dir: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    if not hasattr(model, "save_pretrained"):
        raise TypeError(
            f"Edited model of type {type(model)} does not support save_pretrained()."
        )
    model.save_pretrained(save_dir)
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(save_dir)


def determine_output_dir(
    args: argparse.Namespace,
    alg_name: str,
    model_name: str,
    batch_size: int,
) -> str:
    if args.output_dir is not None:
        return os.path.abspath(args.output_dir)

    return determine_default_model_dir(args, alg_name, model_name, batch_size)


def build_default_run_name(args: argparse.Namespace, model_name: str, batch_size: int) -> str:
    if args.run_name is not None:
        return args.run_name

    model_slug = str(model_name).rstrip("/").split("/")[-1]
    data_stem = Path(args.data_path).stem
    sample_tag = args.sample_size if args.sample_size and args.sample_size > 0 else "all"
    return f"{model_slug}_{data_stem}_bs{batch_size}_n{sample_tag}_seed{args.seed}"


def determine_default_model_dir(
    args: argparse.Namespace,
    alg_name: str,
    model_name: str,
    batch_size: int,
) -> str:
    run_name = build_default_run_name(args, model_name, batch_size)
    return str(OUTPUT_ROOT / "Models" / alg_name / run_name)


def determine_save_model_dir(
    args: argparse.Namespace,
    alg_name: str,
    model_name: str,
    batch_size: int,
) -> str:
    if args.save_model_dir is not None:
        return os.path.abspath(args.save_model_dir)
    return determine_default_model_dir(args, alg_name, model_name, batch_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential MEMIT/AlphaEdit runner with optional RGR."
    )
    parser.add_argument(
        "--editing_method",
        type=str,
        default=None,
        help="Editing method: MEMIT or AlphaEdit.",
    )
    parser.add_argument("--hparams_path", type=str, required=True, help="Path to an EasyEdit hparams yaml.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to editing data json.")
    parser.add_argument(
        "--easyedit_path",
        type=str,
        default=None,
        help="RGR repository root or EasyEdit root.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for metrics and manifests. Defaults to outputs/Models/<method>/<run_name>.",
    )
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name when --output_dir is omitted.")
    parser.add_argument(
        "--save_model_dir",
        type=str,
        default=None,
        help="Directory to save the edited model. Defaults to outputs/Models/<method>/<run_name>.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Editing batch size (default: 1).")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Number of edit requests (default: 1000). Set 0 to use all usable requests.",
    )
    parser.add_argument(
        "--selection",
        choices=["prefix", "random"],
        default="prefix",
        help="Use the dataset prefix (paper default) or a seeded random subset.",
    )
    parser.add_argument(
        "--edit_order_file",
        "--edit-order-file",
        dest="edit_order_file",
        type=str,
        default=None,
        help=(
            "Validated fixed-prefix edit-order manifest. It permutes only the "
            "declared prefix and is mutually exclusive with --selection random."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="CUDA device id or 'auto'.")
    parser.add_argument("--layers", type=maybe_parse_literal_list, default=None, help="Override hparams.layers.")
    parser.add_argument("--model_name", type=str, default=None, help="Optional model_name override in hparams.")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default=None)
    parser.add_argument("--context_num", type=int, default=None)
    parser.add_argument(
        "--append_eos_to_target",
        type=int,
        choices=[0, 1],
        default=1,
        help="Append tokenizer EOS token to each request target_new before editing. Set 0 to disable.",
    )
    parser.add_argument("--p_loc", type=str, default=None, help="Optional AlphaEdit projection path override.")
    parser.add_argument("--v_num_grad_steps", type=int, default=None)
    parser.add_argument("--v_lr", type=float, default=None)
    parser.add_argument("--mom2_n_samples", type=int, default=None)
    parser.add_argument("--alphaedit_l2", type=float, default=None)
    parser.add_argument(
        "--context_multikey_enabled",
        "--context-multikey-enabled",
        dest="context_multikey_enabled",
        type=int,
        choices=[0, 1],
        default=None,
        help=(
            "AlphaEdit only: preserve the existing original/prefix context "
            "keys as an exact same-target multi-key Gram instead of keeping "
            "only their weighted mean. No new prompts are generated."
        ),
    )
    parser.add_argument(
        "--key_gaussian_noise_enabled",
        "--key-gaussian-noise-enabled",
        dest="key_gaussian_noise_enabled",
        type=int,
        choices=[0, 1],
        default=None,
        help=(
            "MEMIT/AlphaEdit: replace the original+five-prefix mean key "
            "with one canonical prompt key plus deterministic Gaussian noise."
        ),
    )
    parser.add_argument(
        "--key_gaussian_noise_relative_std",
        "--key-gaussian-noise-relative-std",
        dest="key_gaussian_noise_relative_std",
        type=float,
        default=None,
        help=(
            "Gaussian key coordinate std as a fraction of ||k||/sqrt(d); "
            "the expected noise/key L2 ratio is approximately this value."
        ),
    )
    parser.add_argument(
        "--key_gaussian_noise_seed",
        "--key-gaussian-noise-seed",
        dest="key_gaussian_noise_seed",
        type=int,
        default=None,
        help=(
            "Base seed for deterministic per-request/per-layer key noise. "
            "Defaults to --seed when the mode is enabled."
        ),
    )
    parser.add_argument(
        "--endogenous_pivot_enabled",
        "--endogenous-pivot-enabled",
        dest="endogenous_pivot_enabled",
        type=int,
        choices=[0, 1],
        default=None,
        help=(
            "MEMIT only: refine native value factors through the actual "
            "multi-layer forward dynamics while holding adjusted keys fixed."
        ),
    )
    parser.add_argument(
        "--endogenous_pivot_mode",
        "--endogenous-pivot-mode",
        dest="endogenous_pivot_mode",
        choices=["joint", "coordinate"],
        default=None,
    )
    parser.add_argument(
        "--endogenous_pivot_trainable_layers",
        "--endogenous-pivot-trainable-layers",
        dest="endogenous_pivot_trainable_layers",
        type=maybe_parse_literal_list,
        default=None,
        help=(
            "Optional MEMIT layer subset whose R factors are trainable; "
            "all other edit layers retain their native MEMIT factors."
        ),
    )
    parser.add_argument(
        "--endogenous_pivot_num_steps",
        "--endogenous-pivot-num-steps",
        dest="endogenous_pivot_num_steps",
        type=int,
        default=None,
        help="Adam steps total (joint) or per layer visit (coordinate).",
    )
    parser.add_argument(
        "--endogenous_pivot_num_sweeps",
        "--endogenous-pivot-num-sweeps",
        dest="endogenous_pivot_num_sweeps",
        type=int,
        default=None,
        help="Coordinate sweeps; ignored in joint mode.",
    )
    parser.add_argument(
        "--endogenous_pivot_lr",
        "--endogenous-pivot-lr",
        dest="endogenous_pivot_lr",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--endogenous_pivot_l2_lambda",
        "--endogenous-pivot-l2-lambda",
        dest="endogenous_pivot_l2_lambda",
        type=float,
        default=None,
        help="Weight on covariance-normalized low-rank update energy.",
    )
    parser.add_argument(
        "--endogenous_pivot_grad_clip_norm",
        "--endogenous-pivot-grad-clip-norm",
        dest="endogenous_pivot_grad_clip_norm",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--endogenous_pivot_capture_trajectory",
        "--endogenous-pivot-capture-trajectory",
        dest="endogenous_pivot_capture_trajectory",
        type=int,
        choices=[0, 1],
        default=None,
        help="Save per-request H_l subject vectors before/native/after refinement.",
    )
    parser.add_argument(
        "--residual_gain_regularization",
        "--residual_energy_gain_regularization",
        dest="residual_gain_regularization",
        type=int,
        choices=[0, 1],
        default=None,
        help="Enable Residual Gain Regularization (legacy REGC flag accepted).",
    )
    parser.add_argument(
        "--residual_gain_lambda",
        "--residual_energy_gain_lambda",
        dest="residual_gain_lambda",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--residual_gain_layers",
        "--residual_energy_gain_layers",
        dest="residual_gain_layers",
        type=maybe_parse_literal_list,
        default=None,
    )
    parser.add_argument(
        "--residual_gain_token_scope",
        "--residual_energy_gain_token_scope",
        dest="residual_gain_token_scope",
        choices=["subject_last", "prompt_last", "both"],
        default=None,
    )
    parser.add_argument(
        "--residual_gain_subject_layers",
        "--residual_energy_gain_subject_layers",
        dest="residual_gain_subject_layers",
        type=maybe_parse_literal_list,
        default=None,
    )
    parser.add_argument(
        "--residual_gain_prompt_layers",
        "--residual_energy_gain_prompt_layers",
        dest="residual_gain_prompt_layers",
        type=maybe_parse_literal_list,
        default=None,
    )
    parser.add_argument(
        "--residual_gain_subject_lambda",
        "--residual_energy_gain_subject_lambda",
        dest="residual_gain_subject_lambda",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--residual_gain_prompt_lambda",
        "--residual_energy_gain_prompt_lambda",
        dest="residual_gain_prompt_lambda",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--residual_gain_margin",
        "--residual_energy_gain_margin",
        dest="residual_gain_margin",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--residual_gain_loss_type",
        "--residual_energy_gain_loss_type",
        dest="residual_gain_loss_type",
        choices=["positive_l1", "positive_squared", "absolute_l1"],
        default=None,
    )
    parser.add_argument(
        "--residual_gain_objective",
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
        ],
        default=None,
        help=(
            "Quantity controlled relative to the current edit's zero-delta "
            "reference. Default gain preserves the original RGR behavior."
        ),
    )
    parser.add_argument(
        "--residual_gain_alignment_weight",
        type=float,
        default=None,
        help=(
            "Multiplier gamma on 2<H,F>/||H||^2 for the "
            "cosine_amplified_gain objective."
        ),
    )
    parser.add_argument(
        "--residual_gain_cosine_aux_lambda",
        "--residual-gain-cosine-aux-lambda",
        dest="residual_gain_cosine_aux_lambda",
        type=float,
        default=None,
        help=(
            "Weight of the independent absolute exp-cosine auxiliary that "
            "steers cos(H,F) toward -1. Zero preserves ordinary RGR."
        ),
    )
    parser.add_argument(
        "--residual_gain_cosine_aux_sharpness",
        "--residual-gain-cosine-aux-sharpness",
        dest="residual_gain_cosine_aux_sharpness",
        type=float,
        default=None,
        help=(
            "Positive gamma in expm1(gamma*(cos(H,F)+1))/expm1(2*gamma)."
        ),
    )
    parser.add_argument(
        "--residual_gain_efficacy_threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--residual_gain_select_best",
        "--residual_energy_gain_select_best",
        dest="residual_gain_select_best",
        type=int,
        choices=[0, 1],
        default=None,
    )
    parser.add_argument(
        "--residual_gain_selection_threshold",
        "--residual_energy_gain_selection_threshold",
        dest="residual_gain_selection_threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--residual_gain_early_stop_mode",
        choices=["base", "total"],
        default=None,
    )
    parser.add_argument(
        "--residual_gain_target_ratio",
        "--residual-gain-target-ratio",
        dest="residual_gain_target_ratio",
        type=float,
        default=None,
        help="Opt-in one-sided hidden-norm ceiling ratio; 1.0 is legacy HN.",
    )
    parser.add_argument(
        "--hn_recovery_paraphrase_aware",
        "--hn-recovery-paraphrase-aware",
        dest="hn_recovery_paraphrase_aware",
        type=int,
        choices=[0, 1],
        default=None,
    )
    parser.add_argument(
        "--hn_recovery_edit_family_weight",
        "--hn-recovery-edit-family-weight",
        dest="hn_recovery_edit_family_weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--hn_recovery_para_family_weight",
        "--hn-recovery-para-family-weight",
        dest="hn_recovery_para_family_weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--hn_recovery_optimization_paraphrase_sidecar",
        "--hn-recovery-optimization-paraphrase-sidecar",
        dest="hn_recovery_optimization_paraphrase_sidecar",
        type=str,
        default=None,
        help=(
            "Audited target-blind sidecar. Required for paraphrase-aware HN; "
            "the benchmark rephrase field is never used."
        ),
    )
    parser.add_argument(
        "--hn_recovery_adaptive_rho",
        "--hn-recovery-adaptive-rho",
        dest="hn_recovery_adaptive_rho",
        type=int,
        choices=[0, 1],
        default=None,
    )
    parser.add_argument(
        "--hn_recovery_adaptive_threshold",
        "--hn-recovery-adaptive-threshold",
        dest="hn_recovery_adaptive_threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--hn_recovery_adaptive_ladder",
        "--hn-recovery-adaptive-ladder",
        dest="hn_recovery_adaptive_ladder",
        type=maybe_parse_float_list,
        default=None,
    )
    parser.add_argument("--hn-recovery-protocol-id", default=None)
    parser.add_argument("--hn-recovery-experiment-id", default=None)
    parser.add_argument("--hn-recovery-arm-id", default=None)
    parser.add_argument("--hn-recovery-order-id", default=None)
    parser.add_argument("--hn-recovery-order-seed", type=int, default=None)
    parser.add_argument(
        "--sadr_regularization",
        type=int,
        choices=[0, 1],
        default=None,
    )
    parser.add_argument("--sadr_lambda", type=float, default=None)
    parser.add_argument(
        "--sadr_attn_layers",
        type=maybe_parse_literal_list,
        default=None,
    )
    parser.add_argument(
        "--sadr_efficacy_threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--encore_enabled",
        type=int,
        choices=[0, 1],
        default=None,
    )
    parser.add_argument("--encore_mpes_top1_steps", type=int, default=None)
    parser.add_argument(
        "--encore_mpes_exclude_first_context",
        type=int,
        choices=[0, 1],
        default=None,
    )
    parser.add_argument("--encore_norm_lambda", type=float, default=None)
    parser.add_argument(
        "--nse_enabled",
        type=int,
        choices=[0, 1],
        default=None,
    )
    parser.add_argument("--nse_alpha", type=float, default=None)
    parser.add_argument("--nse_upper_bound", type=int, default=None)
    parser.add_argument("--nse_max_iterations", type=int, default=None)
    parser.add_argument("--nse_neuron_threshold", type=float, default=None)
    parser.add_argument("--nse_target_cache_dir", type=str, default=None)
    parser.add_argument(
        "--sphere_enabled",
        type=int,
        choices=[0, 1],
        default=None,
    )
    parser.add_argument("--sphere_beta", type=float, default=None)
    parser.add_argument("--sphere_alpha", type=float, default=None)
    parser.add_argument(
        "--nas_enabled",
        type=int,
        choices=[0, 1],
        default=None,
        help="Enable Norm Anchor Scaling after MEMIT's v* optimization.",
    )
    parser.add_argument(
        "--nas_collect_stats",
        type=int,
        choices=[0, 1],
        default=None,
        help="Record the NAS decomposition without necessarily enabling scaling.",
    )
    parser.add_argument("--nas_anchor_path", type=str, default=None)
    parser.add_argument("--nas_anchor_norm", type=float, default=None)
    parser.add_argument("--nas_outlier_factor", type=float, default=None)
    parser.add_argument(
        "--nas_outlier_mode",
        choices=["skip_delta", "scale"],
        default=None,
    )
    parser.add_argument(
        "--save_model",
        type=int,
        choices=[0, 1],
        default=1,
    )
    parser.add_argument(
        "--save_delta_checkpoint",
        "--save-delta-checkpoint",
        dest="save_delta_checkpoint",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Save a compact cumulative current-minus-Base checkpoint after "
            "the final batch. This remains usable after a full model purge."
        ),
    )
    parser.add_argument(
        "--delta_checkpoint_layers",
        "--delta-checkpoint-layers",
        dest="delta_checkpoint_layers",
        type=maybe_parse_literal_list,
        default=None,
        help=(
            "Rewrite layers stored by --save_delta_checkpoint. Defaults to "
            "all configured editor rewrite layers."
        ),
    )
    parser.add_argument(
        "--delta_checkpoint_interval",
        "--delta-checkpoint-interval",
        dest="delta_checkpoint_interval",
        type=int,
        default=0,
        help=(
            "When --save_delta_checkpoint=1, additionally save a compact "
            "cumulative checkpoint every N edited requests. Zero preserves "
            "the legacy final-only behavior. Checkpoints are emitted only at "
            "batch boundaries, so N must be divisible by batch_size."
        ),
    )
    parser.add_argument(
        "--delta_checkpoint_steps",
        "--delta-checkpoint-steps",
        dest="delta_checkpoint_steps",
        type=maybe_parse_literal_list,
        default=None,
        help=(
            "Exact positive request counts at which compact cumulative "
            "checkpoints are saved. Mutually exclusive with a nonzero interval."
        ),
    )
    parser.add_argument(
        "--capture_latent_norms",
        "--capture-latent-norms",
        dest="capture_latent_norms",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Append one scalar-only target/init norm record per edit to "
            "edit_artifacts/latent_norms.jsonl. No latent vectors are saved."
        ),
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Evaluate the edited prefix for EFF/GEN and the full fixed request cohort for locality at checkpoints.",
    )
    parser.add_argument(
        "--eval_steps", type=maybe_parse_literal_list,
        default=[50, 100, 150, 200, 250, 300, 500, 750, 1000],
        help="Cumulative evaluation checkpoints for --do_eval; the final edit is always evaluated.",
    )
    parser.add_argument("--eval_batch_size", type=int, default=1,
                        help="Evaluation batch size for --do_eval (default: 1).")
    parser.add_argument(
        "--capture_latents", type=int, choices=[0, 1], default=0,
        help="Save full target/init/delta vectors for each single-request edit.",
    )
    parser.add_argument(
        "--track_post_update", type=int, choices=[0, 1], default=0,
        help="Capture the final edited block boundary before/after each single-request edit.",
    )
    parser.add_argument("--skip_eval", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.edit_order_file is not None and args.selection != "prefix":
        raise ValueError(
            "--edit_order_file is only valid with --selection prefix; "
            "do not combine a frozen order with --selection random."
        )
    project_root = PROJECT_ROOT
    add_easyedit_to_syspath(args.easyedit_path)
    init_easyedit_imports()
    fix_seed(args.seed)

    alg_name = (
        canonicalize_alg_name(args.editing_method)
        if args.editing_method is not None
        else infer_alg_name_from_yaml(args.hparams_path)
    )
    hparams_cls = HPARAMS_REGISTRY[alg_name]
    hparams = hparams_cls.from_hparams(args.hparams_path)

    if not is_runner_supported_method(alg_name):
        raise ValueError(f"{alg_name} is not supported by this runner.")

    apply_hparam_overrides(hparams, args)
    if args.attn_implementation is not None:
        hparams.attn_implementation = args.attn_implementation
    normalize_hparam_paths(hparams, project_root)
    enabled_methods = {
        "RGR": bool(
            getattr(hparams, "residual_gain_regularization", False)
        ),
        "SADR": bool(getattr(hparams, "sadr_regularization", False)),
        "NSE": bool(getattr(hparams, "nse_enabled", False)),
        "ENCORE": bool(getattr(hparams, "encore_enabled", False)),
        "SPHERE": bool(getattr(hparams, "sphere_enabled", False)),
        "NAS": bool(getattr(hparams, "nas_enabled", False)),
        "EndogenousPivot": bool(
            getattr(hparams, "endogenous_pivot_enabled", False)
        ),
    }
    active_methods = [
        name for name, enabled in enabled_methods.items() if enabled
    ]
    if bool(
        getattr(hparams, "endogenous_pivot_capture_trajectory", False)
    ) and not enabled_methods["EndogenousPivot"]:
        raise ValueError(
            "--endogenous-pivot-capture-trajectory=1 requires "
            "--endogenous-pivot-enabled=1"
        )
    if len(active_methods) > 1:
        raise ValueError(
            "Comparison methods are isolated; enable at most one of "
            "RGR/SADR/NSE/ENCORE/SPHERE/NAS/EndogenousPivot, "
            f"got {active_methods}"
        )
    if enabled_methods["EndogenousPivot"]:
        if alg_name != "MEMIT":
            raise ValueError("Endogenous-pivot refinement is MEMIT-only")
        if bool(getattr(hparams, "nas_collect_stats", False)):
            raise ValueError(
                "Endogenous-pivot MEMIT and NAS statistics collection are "
                "isolated; enable only one"
            )
        if bool(getattr(hparams, "key_gaussian_noise_enabled", False)):
            raise ValueError(
                "Endogenous-pivot MEMIT and Gaussian-key ablation are isolated; "
                "enable only one"
            )
        from EasyEdit.easyeditor.models.memit.endogenous_pivots import (
            validate_endogenous_pivot_hparams,
        )

        validate_endogenous_pivot_hparams(hparams)
    if enabled_methods["NAS"] and (
        getattr(hparams, "nas_anchor_path", None) is None
        and getattr(hparams, "nas_anchor_norm", None) is None
    ):
        raise ValueError(
            "--nas_enabled=1 requires --nas_anchor_path or --nas_anchor_norm"
        )
    if float(getattr(hparams, "nas_outlier_factor", 2.0)) <= 0.0:
        raise ValueError("--nas_outlier_factor must be positive")
    if (
        alg_name == "AlphaEdit"
        and enabled_methods["ENCORE"]
        and float(getattr(hparams, "encore_norm_lambda", 0.0)) != 0.0
    ):
        raise ValueError(
            "AlphaEdit ENCORE uses MPES only; set --encore_norm_lambda 0"
        )
    hn_feature_enabled = bool(
        args.residual_gain_target_ratio is not None
        or args.hn_recovery_paraphrase_aware
        or args.hn_recovery_adaptive_rho
        or args.hn_recovery_optimization_paraphrase_sidecar
    )
    if hn_feature_enabled and alg_name != "AlphaEdit":
        raise ValueError("HN recovery interventions are currently AlphaEdit-only")
    if hn_feature_enabled and not enabled_methods["RGR"]:
        raise ValueError("HN recovery interventions require RGR to be enabled")
    target_ratio = float(getattr(hparams, "residual_gain_target_ratio", 1.0))
    if not np.isfinite(target_ratio) or target_ratio < 1.0:
        raise ValueError("--residual_gain_target_ratio must be finite and >= 1")
    if float(getattr(hparams, "hn_recovery_edit_family_weight", 1.0)) <= 0.0:
        raise ValueError("HN recovery edit-family weight must be positive")
    if float(getattr(hparams, "hn_recovery_para_family_weight", 0.5)) < 0.0:
        raise ValueError("HN recovery paraphrase-family weight cannot be negative")
    adaptive_threshold = float(
        getattr(hparams, "hn_recovery_adaptive_threshold", 0.5)
    )
    if not 0.0 <= adaptive_threshold <= 1.0:
        raise ValueError("HN recovery adaptive threshold must be in [0, 1]")
    raw_adaptive_ladder = getattr(hparams, "hn_recovery_adaptive_ladder", None)
    adaptive_ladder = normalize_hn_recovery_adaptive_ladder(raw_adaptive_ladder)
    if (
        not adaptive_ladder
        or adaptive_ladder != sorted(set(float(value) for value in adaptive_ladder))
        or any(not np.isfinite(float(value)) or float(value) < 1.0 for value in adaptive_ladder)
    ):
        raise ValueError("HN recovery adaptive ladder must be finite, sorted, unique, and >= 1")

    if not hasattr(hparams, "batch_size"):
        raise ValueError(f"{alg_name} hparams does not define batch_size.")
    if hparams.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {hparams.batch_size}")
    delta_checkpoint_interval = validate_delta_checkpoint_interval(
        args.delta_checkpoint_interval,
        batch_size=int(hparams.batch_size),
        checkpointing_enabled=bool(args.save_delta_checkpoint),
    )

    requests, raw_count = load_requests(args.data_path)
    if not requests:
        raise ValueError(f"No valid edit requests found in {args.data_path}")

    requests, edit_order_metadata = resolve_request_order(
        requests,
        raw_count=raw_count,
        data_path=args.data_path,
        sample_size=args.sample_size,
        seed=args.seed,
        selection=args.selection,
        edit_order_file=args.edit_order_file,
    )
    delta_checkpoint_steps = validate_delta_checkpoint_steps(
        args.delta_checkpoint_steps,
        batch_size=int(hparams.batch_size),
        sample_size=len(requests),
        checkpointing_enabled=bool(args.save_delta_checkpoint),
        interval=delta_checkpoint_interval,
    )
    hn_recovery_sidecar = attach_hn_recovery_metadata(
        requests,
        args=args,
        data_path=args.data_path,
    )
    validate_method_specific_constraints(alg_name, hparams, requests)
    hparams_model_name = getattr(hparams, "model_name", "unknown_model")
    output_dir = determine_output_dir(
        args,
        alg_name,
        hparams_model_name,
        hparams.batch_size,
    )
    save_model_enabled = bool(args.save_model)
    save_model_dir = (
        determine_save_model_dir(
            args,
            alg_name,
            hparams_model_name,
            hparams.batch_size,
        )
        if save_model_enabled
        else None
    )
    setattr(hparams, "save_model_dir", save_model_dir)
    os.makedirs(output_dir, exist_ok=True)
    if hn_feature_enabled:
        hparams.hn_recovery_scalar_log_path = os.path.join(
            output_dir, "hn_recovery_per_edit.jsonl"
        )
        scalar_log_path = Path(hparams.hn_recovery_scalar_log_path)
        if scalar_log_path.exists() and scalar_log_path.stat().st_size > 0:
            raise FileExistsError(
                "Refusing a non-empty HN recovery scalar log without an "
                f"explicit resume contract: {scalar_log_path}"
            )
        scalar_log_path.touch(exist_ok=True)
    edit_order_metadata = archive_edit_order_manifest(
        edit_order_file=args.edit_order_file,
        metadata=edit_order_metadata,
        output_dir=output_dir,
    )
    capture_latent_norms = bool(args.capture_latent_norms)
    capture_latents = bool(args.capture_latents)
    track_post_update = bool(args.track_post_update)
    if (capture_latents or track_post_update) and int(hparams.batch_size) != 1:
        raise ValueError("Latent-vector and post-update tracking require batch_size=1")
    if capture_latent_norms or capture_latents or track_post_update:
        hparams.analysis_artifact_dir = os.path.join(
            output_dir,
            "edit_artifacts",
        )
        hparams.analysis_capture_latent_norms = capture_latent_norms
        hparams.analysis_capture_latents = capture_latents
        os.makedirs(hparams.analysis_artifact_dir, exist_ok=True)
        latent_norm_log = os.path.join(
            hparams.analysis_artifact_dir,
            "latent_norms.jsonl",
        )
        with open(latent_norm_log, "w", encoding="utf-8"):
            pass
    if bool(getattr(hparams, "residual_gain_regularization", False)):
        hparams.residual_gain_log_path = os.path.join(
            output_dir, "rgr_optimization.jsonl"
        )
        with open(hparams.residual_gain_log_path, "w", encoding="utf-8"):
            pass
    if bool(getattr(hparams, "endogenous_pivot_enabled", False)):
        hparams.endogenous_pivot_log_path = os.path.join(
            output_dir, "endogenous_pivot_optimization.jsonl"
        )
        with open(hparams.endogenous_pivot_log_path, "w", encoding="utf-8"):
            pass
        if bool(
            getattr(hparams, "endogenous_pivot_capture_trajectory", False)
        ):
            hparams.endogenous_pivot_artifact_dir = os.path.join(
                output_dir, "endogenous_pivot_trajectories"
            )
            os.makedirs(hparams.endogenous_pivot_artifact_dir, exist_ok=True)
    if bool(getattr(hparams, "context_multikey_enabled", False)):
        hparams.context_multikey_log_path = os.path.join(
            output_dir, "context_multikey.jsonl"
        )
        with open(
            hparams.context_multikey_log_path,
            "w",
            encoding="utf-8",
        ):
            pass
    if bool(getattr(hparams, "key_gaussian_noise_enabled", False)):
        hparams.key_gaussian_noise_log_path = os.path.join(
            output_dir, "key_gaussian_noise.jsonl"
        )
        with open(
            hparams.key_gaussian_noise_log_path,
            "w",
            encoding="utf-8",
        ):
            pass
    if bool(getattr(hparams, "sadr_regularization", False)):
        hparams.sadr_log_path = os.path.join(
            output_dir, "sadr_optimization.jsonl"
        )
        with open(hparams.sadr_log_path, "w", encoding="utf-8"):
            pass
    if any(
        bool(getattr(hparams, name, False))
        for name in ("encore_enabled", "nse_enabled", "sphere_enabled")
    ):
        hparams.official_baseline_log_path = os.path.join(
            output_dir, "official_baseline.jsonl"
        )
        with open(
            hparams.official_baseline_log_path,
            "w",
            encoding="utf-8",
        ):
            pass
    if bool(
        getattr(hparams, "nas_enabled", False)
        or getattr(hparams, "nas_collect_stats", False)
    ):
        hparams.nas_log_path = os.path.join(
            output_dir, "nas_scaling.jsonl"
        )
        with open(hparams.nas_log_path, "w", encoding="utf-8"):
            pass
    run_eval = bool(args.do_eval and not args.skip_eval)
    editor = BaseEditor.from_hparams(hparams)
    if track_post_update:
        from diagnostics.post_update_tracking import (
            capture_boundary_nodes, save_paired_post_update_artifact,
        )
        final_write_layer = int(hparams.layers[-1])
        post_update_nodes = (
            f"H{final_write_layer}", f"M{final_write_layer}",
            f"H{final_write_layer + 1}", f"A{final_write_layer + 1}",
            f"H{final_write_layer + 2}",
        )
    compact_parameter_names: List[str] = []
    base_checkpoint_parameters: Dict[str, torch.Tensor] = {}
    compact_rewrite_layers: List[int] = []
    compact_checkpoint_manifests: Dict[int, Dict[str, Any]] = {}
    if bool(args.save_delta_checkpoint):
        from EasyEdit.easyeditor.util.edited_layer_checkpoint import (
            rewrite_parameter_names,
            save_parameter_delta_checkpoint,
            snapshot_parameters,
        )

        configured_rewrite_layers = [int(layer) for layer in hparams.layers]
        compact_rewrite_layers = (
            [int(layer) for layer in args.delta_checkpoint_layers]
            if args.delta_checkpoint_layers is not None
            else configured_rewrite_layers
        )
        if not compact_rewrite_layers:
            raise ValueError("--delta_checkpoint_layers cannot be empty")
        if not set(compact_rewrite_layers).issubset(
            set(configured_rewrite_layers)
        ):
            raise ValueError(
                "--delta_checkpoint_layers must be a subset of configured "
                f"rewrite layers {configured_rewrite_layers}; got "
                f"{compact_rewrite_layers}."
            )
        compact_parameter_names = rewrite_parameter_names(
            hparams,
            compact_rewrite_layers,
        )
        base_checkpoint_parameters = snapshot_parameters(
            editor.model,
            compact_parameter_names,
        )
        print(
            "[checkpoint] snapshotted Base parameters for cumulative deltas: "
            f"layers={compact_rewrite_layers} interval={delta_checkpoint_interval}"
        )
    append_eos_enabled = bool(args.append_eos_to_target)
    eos_token = None
    eos_appended_count = 0
    if append_eos_enabled:
        requests, eos_token, eos_appended_count = append_eos_to_request_targets(requests, editor.tok)
    if bool(getattr(hparams, "nse_enabled", False)):
        prepare_nse_original_targets(
            editor=editor,
            requests=requests,
            hparams=hparams,
            alg_name=alg_name,
            output_dir=output_dir,
        )

    # Publish the immutable, fully normalized request list before the first
    # edit. Periodic compact checkpoints bind both this full list and the
    # exact prefix that had been committed when the checkpoint was written.
    requests_path = os.path.join(output_dir, "requests.json")
    atomic_write_json(requests_path, requests)
    requests_sha256 = edit_order_file_sha256(Path(requests_path))
    if requests_sha256 != canonical_json_sha256(requests):
        raise RuntimeError("requests.json serialization/hash contract mismatch")

    def save_compact_checkpoint(edit_count: int) -> Dict[str, Any]:
        if not bool(args.save_delta_checkpoint):
            raise RuntimeError("compact checkpointing is disabled")
        if edit_count < 1 or edit_count > len(requests):
            raise ValueError(f"invalid compact checkpoint edit_count={edit_count}")
        existing = compact_checkpoint_manifests.get(edit_count)
        if existing is not None:
            return existing

        checkpoint_dir = Path(output_dir) / f"step_{edit_count:03d}"
        checkpoint_path = checkpoint_dir / "edited_parameter_deltas.pt"
        prefix_sha256 = canonical_json_sha256(requests[:edit_count])
        print(
            "[checkpoint] saving cumulative parameter deltas: "
            f"{checkpoint_path}"
        )
        manifest = save_parameter_delta_checkpoint(
            editor.model,
            checkpoint_path,
            base_parameters=base_checkpoint_parameters,
            parameter_names=compact_parameter_names,
            metadata={
                "editing_method": alg_name,
                "base_model": str(hparams.model_name),
                "edit_count": edit_count,
                "batch_size": int(hparams.batch_size),
                "model_seed": args.seed,
                "rewrite_layers": compact_rewrite_layers,
                "rewrite_module_tmp": hparams.rewrite_module_tmp,
                "hparams_path": os.path.abspath(args.hparams_path),
                "hparams_sha256": edit_order_file_sha256(
                    Path(args.hparams_path).expanduser().resolve()
                ),
                "requests_path": os.path.abspath(requests_path),
                "requests_sha256": requests_sha256,
                "requests_prefix_count": edit_count,
                "requests_prefix_sha256": prefix_sha256,
                "edit_order_id": (
                    edit_order_metadata.get("order_id")
                    if edit_order_metadata is not None
                    else None
                ),
                "edit_order_shuffle_seed": (
                    edit_order_metadata.get("shuffle_seed")
                    if edit_order_metadata is not None
                    else None
                ),
                "edit_order_manifest_sha256": (
                    edit_order_metadata.get("sha256")
                    if edit_order_metadata is not None
                    else None
                ),
                "residual_gain_lambda": getattr(
                    hparams, "residual_gain_lambda", None
                ),
                "residual_gain_objective": getattr(
                    hparams, "residual_gain_objective", None
                ),
                "residual_gain_loss_type": getattr(
                    hparams, "residual_gain_loss_type", None
                ),
                "residual_gain_margin": getattr(
                    hparams, "residual_gain_margin", None
                ),
                "residual_gain_token_scope": getattr(
                    hparams, "residual_gain_token_scope", None
                ),
                "residual_gain_subject_layers": getattr(
                    hparams, "residual_gain_subject_layers", None
                ),
                "residual_gain_target_ratio": getattr(
                    hparams, "residual_gain_target_ratio", 1.0
                ),
                "hn_recovery_paraphrase_aware": bool(
                    getattr(hparams, "hn_recovery_paraphrase_aware", False)
                ),
                "hn_recovery_edit_family_weight": getattr(
                    hparams, "hn_recovery_edit_family_weight", 1.0
                ),
                "hn_recovery_para_family_weight": getattr(
                    hparams, "hn_recovery_para_family_weight", 0.5
                ),
                "hn_recovery_adaptive_rho": bool(
                    getattr(hparams, "hn_recovery_adaptive_rho", False)
                ),
                "hn_recovery_adaptive_threshold": getattr(
                    hparams, "hn_recovery_adaptive_threshold", 0.5
                ),
                "hn_recovery_adaptive_ladder": getattr(
                    hparams, "hn_recovery_adaptive_ladder", None
                ),
                "nas_enabled": bool(getattr(hparams, "nas_enabled", False)),
                "nas_anchor_path": getattr(hparams, "nas_anchor_path", None),
                "nas_anchor_sha256": (
                    edit_order_file_sha256(
                        Path(str(hparams.nas_anchor_path)).expanduser().resolve()
                    )
                    if getattr(hparams, "nas_anchor_path", None)
                    else None
                ),
                "nas_outlier_factor": getattr(hparams, "nas_outlier_factor", None),
                "nas_outlier_mode": getattr(hparams, "nas_outlier_mode", None),
                "endogenous_pivot_enabled": bool(
                    getattr(hparams, "endogenous_pivot_enabled", False)
                ),
                "endogenous_pivot_mode": getattr(
                    hparams, "endogenous_pivot_mode", None
                ),
                "endogenous_pivot_trainable_layers": getattr(
                    hparams, "endogenous_pivot_trainable_layers", None
                ),
                "endogenous_pivot_num_steps": getattr(
                    hparams, "endogenous_pivot_num_steps", None
                ),
                "endogenous_pivot_num_sweeps": getattr(
                    hparams, "endogenous_pivot_num_sweeps", None
                ),
                "endogenous_pivot_lr": getattr(
                    hparams, "endogenous_pivot_lr", None
                ),
                "endogenous_pivot_l2_lambda": getattr(
                    hparams, "endogenous_pivot_l2_lambda", None
                ),
                "endogenous_pivot_grad_clip_norm": getattr(
                    hparams, "endogenous_pivot_grad_clip_norm", None
                ),
                "endogenous_pivot_capture_trajectory": bool(
                    getattr(
                        hparams,
                        "endogenous_pivot_capture_trajectory",
                        False,
                    )
                ),
                "delta_checkpoint_interval": delta_checkpoint_interval,
                "delta_checkpoint_steps": delta_checkpoint_steps,
                "hn_recovery_protocol_id": args.hn_recovery_protocol_id,
                "hn_recovery_experiment_id": args.hn_recovery_experiment_id,
                "hn_recovery_arm_id": args.hn_recovery_arm_id,
                "hn_recovery_order_id": args.hn_recovery_order_id,
                "hn_recovery_order_seed": args.hn_recovery_order_seed,
                "hn_recovery_sidecar_sha256": (
                    hn_recovery_sidecar.get("sha256")
                    if hn_recovery_sidecar is not None
                    else None
                ),
                "checkpoint_semantics": (
                    "Add these cumulative deltas to the named Base-model "
                    f"parameters to reconstruct the state after {edit_count} "
                    "ordered edits."
                ),
            },
        )
        compact_checkpoint_manifests[edit_count] = manifest
        return manifest

    print(f"[config] method={alg_name}")
    print(f"[config] model={hparams.model_name}")
    print(f"[config] data_path={args.data_path}")
    print(f"[config] requests={len(requests)} / raw_records={raw_count}")
    print(f"[config] batch_size={hparams.batch_size}")
    print(f"[config] model_seed={args.seed}")
    if edit_order_metadata is not None:
        print(
            "[config] edit_order="
            f"{edit_order_metadata['order_id']} "
            f"shuffle_seed={edit_order_metadata['shuffle_seed']} "
            f"manifest_sha256={edit_order_metadata['sha256']}"
        )
    print(
        "[config] append_eos_to_target="
        f"{append_eos_enabled}"
        + (f" eos_token={eos_token!r} appended={eos_appended_count}" if append_eos_enabled else "")
    )
    print(f"[config] output_dir={output_dir}")
    print(f"[config] save_model={int(save_model_enabled)}")
    print(f"[config] save_model_dir={save_model_dir}")
    print(
        "[config] save_delta_checkpoint="
        f"{int(bool(args.save_delta_checkpoint))} "
        f"layers={compact_rewrite_layers} interval={delta_checkpoint_interval} "
        f"steps={delta_checkpoint_steps}"
    )
    print(f"[config] capture_latent_norms={int(capture_latent_norms)}")
    print(
        "[config] RGR="
        f"{int(bool(getattr(hparams, 'residual_gain_regularization', False)))} "
        f"layers={getattr(hparams, 'residual_gain_layers', None)} "
        f"subject_layers={getattr(hparams, 'residual_gain_subject_layers', None)} "
        f"prompt_layers={getattr(hparams, 'residual_gain_prompt_layers', None)} "
        f"lambda={getattr(hparams, 'residual_gain_lambda', None)} "
        f"loss={getattr(hparams, 'residual_gain_loss_type', None)} "
        f"objective={getattr(hparams, 'residual_gain_objective', 'gain')} "
        "alignment_weight="
        f"{getattr(hparams, 'residual_gain_alignment_weight', 1.0)} "
        "cosine_aux_lambda="
        f"{getattr(hparams, 'residual_gain_cosine_aux_lambda', 0.0)} "
        "cosine_aux_sharpness="
        f"{getattr(hparams, 'residual_gain_cosine_aux_sharpness', 1.0)}"
    )
    print(
        "[config] comparison_method="
        f"{active_methods[0] if active_methods else 'baseline'} "
        f"SADR={int(enabled_methods['SADR'])} "
        f"NSE={int(enabled_methods['NSE'])} "
        f"ENCORE={int(enabled_methods['ENCORE'])} "
        f"SPHERE={int(enabled_methods['SPHERE'])} "
        f"NAS={int(enabled_methods['NAS'])}"
    )
    print(
        "[config] endogenous_pivot="
        f"{int(enabled_methods['EndogenousPivot'])} "
        f"mode={getattr(hparams, 'endogenous_pivot_mode', None)} "
        f"trainable_layers="
        f"{getattr(hparams, 'endogenous_pivot_trainable_layers', None)} "
        f"steps={getattr(hparams, 'endogenous_pivot_num_steps', None)} "
        f"sweeps={getattr(hparams, 'endogenous_pivot_num_sweeps', None)} "
        f"lr={getattr(hparams, 'endogenous_pivot_lr', None)} "
        f"cost_lambda={getattr(hparams, 'endogenous_pivot_l2_lambda', None)} "
        "cost=covariance_update_energy target=edit_layer_output "
        f"trajectory={int(bool(getattr(hparams, 'endogenous_pivot_capture_trajectory', False)))}"
    )
    print(
        "[config] context_multikey="
        f"{int(bool(getattr(hparams, 'context_multikey_enabled', False)))} "
        "source=existing_context_templates "
        "compression=exact_same_target_mean_plus_deviation "
        "contexts=original_1+all_generated_prefixes_5 variants=6 subsampling=0"
    )
    print(
        "[config] key_gaussian_noise="
        f"{int(bool(getattr(hparams, 'key_gaussian_noise_enabled', False)))} "
        "source=canonical_unprefixed_prompt keys_per_fact=1 "
        "context_average=0 generated_prefix_keys=0 "
        "relative_std="
        f"{getattr(hparams, 'key_gaussian_noise_relative_std', None)} "
        f"seed={getattr(hparams, 'key_gaussian_noise_seed', None)}"
    )
    if bool(
        getattr(hparams, "nas_enabled", False)
        or getattr(hparams, "nas_collect_stats", False)
    ):
        print(
            "[config] NAS="
            f"{int(enabled_methods['NAS'])} "
            f"anchor_path={getattr(hparams, 'nas_anchor_path', None)} "
            f"anchor_norm={getattr(hparams, 'nas_anchor_norm', None)} "
            f"outlier_factor={getattr(hparams, 'nas_outlier_factor', 2.0)} "
            f"outlier_mode={getattr(hparams, 'nas_outlier_mode', 'skip_delta')}"
        )
    print(f"[config] do_eval={run_eval}")

    cumulative_evaluations: List[Dict[str, Any]] = []
    evaluation_steps = set()
    if run_eval:
        from diagnostics.analyze_edit_count_gain_trajectory import (
            collect_group_items, compute_locality_outputs, evaluate_step,
        )
        from evaluate.eval_hf_easyedit import maybe_format_prompt
        # Editing algorithms use subject placeholders; evaluation uses rendered text.
        evaluation_requests = [
            dict(request, prompt=maybe_format_prompt(request["prompt"], request.get("subject")),
                 rephrase_prompt=maybe_format_prompt(request.get("rephrase_prompt"), request.get("subject")))
            for request in requests
        ]
        if args.eval_batch_size < 1 or any(step < 1 for step in args.eval_steps):
            raise ValueError("Evaluation batch size and checkpoint edit counts must be positive")
        evaluation_steps = {step for step in args.eval_steps if step <= len(requests)}
        evaluation_steps.add(len(requests))
        invalid_steps = [step for step in evaluation_steps
                         if step != len(requests) and step % int(hparams.batch_size)]
        if invalid_steps:
            raise ValueError(f"Evaluation checkpoints must coincide with editing batch boundaries: {invalid_steps}")
        locality_items = collect_group_items(requests, "locality")
        missing_locality = set(range(len(requests))) - {int(item[0]) for item in locality_items}
        if missing_locality:
            raise ValueError(f"The fixed locality panel is missing prompts for {len(missing_locality)} selected requests")
        # The reference is captured once from the unedited model, on all selected requests.
        editor.model.eval()
        base_locality_outputs = compute_locality_outputs(
            model=editor.model, model_name=editor.model_name, hparams=editor.hparams,
            tokenizer=editor.tok, locality_items=locality_items,
            device=editor.hparams.device, batch_size=args.eval_batch_size,
        )

    all_metrics: List[Dict[str, Any]] = []
    started_at = time.time()
    configured_batch_size = int(hparams.batch_size)
    total_batches = (len(requests) + configured_batch_size - 1) // configured_batch_size
    for batch_index, batch_requests in enumerate(chunk_list(requests, configured_batch_size), start=1):
        case_ids = [req["case_id"] for req in batch_requests]
        if len(batch_requests) == 1:
            hparams.analysis_current_edit_index = int(
                batch_requests[0].get("edit_index", len(all_metrics) + 1)
            )
        print(
            f"[batch {batch_index}/{total_batches}] editing {len(batch_requests)} request(s) "
            f"case_ids={case_ids[:3]}{'...' if len(case_ids) > 3 else ''}"
        )

        if track_post_update:
            before_nodes, node_metadata = capture_boundary_nodes(
                editor.model, editor.tok, batch_requests[0],
                nodes=post_update_nodes, max_length=512,
            )
        edit_start = time.perf_counter()
        edited_model, _ = editor.apply_algo(
            editor.model,
            editor.tok,
            batch_requests,
            editor.hparams,
            copy=False,
            return_orig_weights=False,
            keep_original_weight=False,
        )
        edit_time = time.perf_counter() - edit_start
        editor.model = edited_model
        if track_post_update:
            after_nodes, after_node_metadata = capture_boundary_nodes(
                editor.model, editor.tok, batch_requests[0],
                nodes=post_update_nodes, max_length=512,
            )
            if node_metadata != after_node_metadata:
                raise RuntimeError("Post-update probe token coordinates changed")
            save_paired_post_update_artifact(
                Path(hparams.analysis_artifact_dir),
                edit_index=int(hparams.analysis_current_edit_index),
                request=batch_requests[0], before_nodes=before_nodes,
                after_nodes=after_nodes, node_metadata=node_metadata,
                before_behavior={}, after_behavior={},
                array_dtype="float32",
            )

        for request_index, request in enumerate(batch_requests):
            all_metrics.append(
                {
                    "case_id": request["case_id"],
                    "batch_index": batch_index,
                    "index_in_batch": request_index,
                    "requested_rewrite": request,
                    "edit_time_sec": edit_time,
                    "pre": None,
                    "post": None,
                }
            )

        print(
            f"[batch {batch_index}/{total_batches}] done in {edit_time:.2f}s "
            f"(cumulative requests={len(all_metrics)})"
        )
        committed_requests = len(all_metrics)
        if committed_requests in evaluation_steps:
            editor.model.eval()
            evaluation = evaluate_step(
                committed_requests, editor.model, editor.model_name, editor.hparams,
                editor.tok, evaluation_requests, requests, base_locality_outputs,
                args.eval_batch_size, continuous_behavior=False,
            )
            evaluation["evaluation_scope"] = {
                "editing_requests": committed_requests,
                "locality_requests": len(requests),
                "locality_reference": "unedited Base predictions on the fixed selected request cohort",
            }
            atomic_write_json(
                os.path.join(output_dir, "evaluations", f"step_{committed_requests:04d}.json"),
                evaluation,
            )
            cumulative_evaluations.append(dict(edit_count=committed_requests, **evaluation["summary"]))
        if (
            bool(args.save_delta_checkpoint)
            and (
                (
                    delta_checkpoint_interval > 0
                    and committed_requests % delta_checkpoint_interval == 0
                )
                or committed_requests in set(delta_checkpoint_steps)
            )
        ):
            save_compact_checkpoint(committed_requests)

    hparams.batch_size = configured_batch_size
    total_elapsed = time.time() - started_at
    summary = summarize_metrics(all_metrics)
    summary["total_elapsed_sec"] = total_elapsed
    if cumulative_evaluations:
        summary["cumulative_evaluations"] = cumulative_evaluations
        summary["final_cumulative_evaluation"] = cumulative_evaluations[-1]
        summary["evaluation_protocol"] = "EFF/GEN on requests[:edit_count]; locality on the fixed full request cohort versus Base"

    metrics_path = os.path.join(output_dir, "metrics.json")
    requests_path = os.path.join(output_dir, "requests.json")
    summary_path = os.path.join(output_dir, "summary.json")
    manifest_path = os.path.join(output_dir, "run_manifest.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2, default=json_default)
    if edit_order_file_sha256(Path(requests_path)) != requests_sha256:
        raise RuntimeError("immutable requests.json changed during editing")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=json_default)

    compact_checkpoint_manifest: Optional[Dict[str, Any]] = None
    if bool(args.save_delta_checkpoint):
        compact_checkpoint_manifest = save_compact_checkpoint(len(requests))

    manifest = {
        "editing_method": alg_name,
        "hparams_path": os.path.abspath(args.hparams_path),
        "hparams_sha256": edit_order_file_sha256(
            Path(args.hparams_path).expanduser().resolve()
        ),
        "data_path": os.path.abspath(args.data_path),
        "raw_record_count": raw_count,
        "num_requests_used": len(requests),
        "batch_size": hparams.batch_size,
        "seed": args.seed,
        "model_seed": args.seed,
        "selection": args.selection,
        "edit_order_manifest": edit_order_metadata,
        "append_eos_to_target": append_eos_enabled,
        "eos_token": eos_token,
        "eos_appended_count": eos_appended_count,
        "do_eval": run_eval,
        "eval_steps": sorted(evaluation_steps),
        "eval_batch_size": args.eval_batch_size,
        "locality_panel_requests": len(requests) if run_eval else None,
        "evaluation_metric": "complete-target teacher-forced accuracy; fixed-panel Base prediction agreement",
        "output_dir": output_dir,
        "save_model": save_model_enabled,
        "save_model_dir": save_model_dir,
        "save_delta_checkpoint": bool(args.save_delta_checkpoint),
        "delta_checkpoint_interval": delta_checkpoint_interval,
        "delta_checkpoint_steps": delta_checkpoint_steps,
        "delta_checkpoint": compact_checkpoint_manifest,
        "delta_checkpoints": [
            compact_checkpoint_manifests[edit_count]
            for edit_count in sorted(compact_checkpoint_manifests)
        ],
        "capture_latent_norms": capture_latent_norms,
        "capture_latents": capture_latents,
        "track_post_update": track_post_update,
        "attn_implementation": args.attn_implementation,
        "hn_recovery": {
            "protocol_id": args.hn_recovery_protocol_id,
            "experiment_id": args.hn_recovery_experiment_id,
            "arm_id": args.hn_recovery_arm_id,
            "order_id": args.hn_recovery_order_id,
            "order_seed": args.hn_recovery_order_seed,
            "sidecar": hn_recovery_sidecar,
            "scalar_log_path": getattr(
                hparams, "hn_recovery_scalar_log_path", None
            ),
        },
        "total_elapsed_sec": total_elapsed,
        "effective_hparams": vars(hparams),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=json_default)

    if save_model_enabled:
        print(f"[save] writing edited model to {save_model_dir}")
        save_model_and_tokenizer(
            editor.model,
            editor.tok,
            save_model_dir,
        )
    else:
        print("[save] skipped edited model (--save_model=0)")

    print(f"[done] metrics: {metrics_path}")
    print(f"[done] summary: {summary_path}")
    print(f"[done] manifest: {manifest_path}")
    if compact_checkpoint_manifest is not None:
        print(
            "[done] delta_checkpoint: "
            f"{Path(output_dir) / f'step_{len(requests):03d}' / 'edited_parameter_deltas.pt'}"
        )
    print(
        f"[done] saved_model: "
        f"{save_model_dir if save_model_enabled else 'disabled'}"
    )


if __name__ == "__main__":
    main()
