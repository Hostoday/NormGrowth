import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome.layer_stats import layer_stats
from ...util import nethook
from ...util.oedit import reset_oedit_state
from ...util.residual_gain_regularization import append_rgr_record
from ...util.generate import generate_fast
from ...util.globals import *
from ...util.official_editing_baselines import (
    append_official_baseline_record,
    get_nse_target,
    nse_restricted_solve,
    nse_select_neurons,
    project_updates_with_sphere,
)
from ...util.edit_analysis_artifacts import (
    save_committed_outer_layer_artifact,
    save_outer_layer_artifact,
)
from ...util.key_gaussian_noise import canonical_prompt_template
from ...util.semantic_multikey import (
    protect_accumulator_owner_from_inplace_add,
    same_target_multikey_terms,
)
from ...util.strict_tangent_layer_plan import (
    build_strict_joint_tangent_plan,
    collect_clean_local_output_references,
    tangent_plan_diagnostics,
)

from .compute_ks import (
    compute_ks,
    compute_ks_context_components,
    compute_ks_gaussian_noise,
)
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx, _validate_o0_axis_configuration
from .compute_z import _prepare_oedit_regularizer
from .AlphaEdit_hparams import AlphaEditHyperParams
# import compute_z_kl

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

P = None
cache_c = None
P_loaded = False
cache_c_new = False
cache_c_outer_key_signature = None
_CONTEXT_MULTIKEY_PROVENANCE_LOGGED_PATHS = set()
_KEY_GAUSSIAN_NOISE_PROVENANCE_LOGGED_PATHS = set()


def _context_multikey_enabled(hparams: AlphaEditHyperParams) -> bool:
    return bool(getattr(hparams, "context_multikey_enabled", False))


def _key_gaussian_noise_enabled(hparams: AlphaEditHyperParams) -> bool:
    return bool(getattr(hparams, "key_gaussian_noise_enabled", False))


def _outer_key_mode(hparams: AlphaEditHyperParams) -> str:
    """Validate and name the mutually exclusive AlphaEdit key semantics."""

    context_multikey = _context_multikey_enabled(hparams)
    gaussian_noise = _key_gaussian_noise_enabled(hparams)
    if context_multikey and gaussian_noise:
        raise ValueError(
            "context_multikey_enabled and key_gaussian_noise_enabled are "
            "mutually exclusive: the former extracts original+prefix keys, "
            "whereas the latter extracts exactly one canonical prompt key."
        )
    if gaussian_noise:
        relative_std = float(
            getattr(hparams, "key_gaussian_noise_relative_std", 0.05)
        )
        if not np.isfinite(relative_std) or relative_std <= 0.0:
            raise ValueError(
                "key_gaussian_noise_relative_std must be finite and > 0; "
                f"got {relative_std!r}"
            )
        return "canonical_gaussian_noise"
    if context_multikey:
        return "exact_context_multikey"
    return "legacy_context_mean"


def _outer_key_cache_signature(hparams: AlphaEditHyperParams) -> Tuple[Any, ...]:
    """Return every setting that changes the cumulative key Gram."""

    mode = _outer_key_mode(hparams)
    if mode == "canonical_gaussian_noise":
        return (
            mode,
            float(hparams.key_gaussian_noise_relative_std),
            int(hparams.key_gaussian_noise_seed),
        )
    if mode == "exact_context_multikey":
        expected = getattr(
            hparams, "context_multikey_expected_group_sizes", None
        )
        return (mode, tuple(expected) if expected is not None else None)
    return (mode,)


def _tangent_layer_allocation_enabled(hparams: AlphaEditHyperParams) -> bool:
    return bool(getattr(hparams, "tangent_layer_allocation_enabled", False))


def _append_tangent_layer_allocation_record(
    hparams: AlphaEditHyperParams,
    record: Dict[str, Any],
) -> None:
    """Append provenance/geometry for the opt-in strict layer allocation."""

    path_value = getattr(hparams, "tangent_layer_allocation_log_path", None)
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "AlphaEdit",
        "feature": "tangent_layer_allocation",
        "mode": "strict_joint_local_tangent_jacobian_identity",
        **record,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _append_context_multikey_record(
    hparams: AlphaEditHyperParams,
    record: Dict[str, Any],
) -> None:
    """Append compact provenance/diagnostics for the opt-in outer solve."""

    path_value = getattr(hparams, "context_multikey_log_path", None)
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "AlphaEdit",
        "feature": "context_multikey",
        "mode": "exact_same_target_context_components",
        **record,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _append_key_gaussian_noise_record(
    hparams: AlphaEditHyperParams,
    record: Dict[str, Any],
) -> None:
    """Append reproducibility and realization data for noisy canonical keys."""

    path_value = getattr(hparams, "key_gaussian_noise_log_path", None)
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "AlphaEdit",
        "feature": "key_gaussian_noise",
        "mode": "one_canonical_key_plus_relative_gaussian_noise",
        **record,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _ensure_key_gaussian_noise_provenance_record(
    hparams: AlphaEditHyperParams,
) -> Dict[str, Any]:
    """Validate and persist the prefix-free noisy-key contract once."""

    if _outer_key_mode(hparams) != "canonical_gaussian_noise":
        raise ValueError(
            "Gaussian-key provenance requested while its mode is disabled"
        )
    provenance = {
        "key_source": "canonical_unprefixed_prompt",
        "canonical_prompt_forwards_per_fact_per_layer": 1,
        "generated_prefix_key_forwards": 0,
        "keys_per_fact": 1,
        "context_key_averaging": False,
        "perturbation_distribution": "isotropic_gaussian",
        "perturbation_formula": (
            "epsilon ~ N(0, (relative_std * ||k||_2 / sqrt(d))^2 I); "
            "k_update = k + epsilon"
        ),
        "relative_std": float(hparams.key_gaussian_noise_relative_std),
        "base_seed": int(hparams.key_gaussian_noise_seed),
        "seed_derivation": (
            "sha256(base_seed,layer,case_id,edit_index,prompt_template,subject)"
        ),
        "target_formation": "native_alphaedit_compute_z_unchanged",
    }

    path_value = getattr(hparams, "key_gaussian_noise_log_path", None)
    if not path_value:
        return provenance
    path = Path(path_value).resolve()
    path_key = str(path)
    if path_key in _KEY_GAUSSIAN_NOISE_PROVENANCE_LOGGED_PATHS:
        return provenance
    already_logged = False
    if path.exists() and path.stat().st_size:
        with path.open("r", encoding="utf-8") as handle:
            already_logged = any(
                '"stage": "key_construction_provenance"' in line
                for line in handle
            )
    if not already_logged:
        _append_key_gaussian_noise_record(
            hparams,
            {
                "stage": "key_construction_provenance",
                **provenance,
            },
        )
    _KEY_GAUSSIAN_NOISE_PROVENANCE_LOGGED_PATHS.add(path_key)
    return provenance


def _context_multikey_template_provenance(
    context_templates: List[List[str]],
) -> Dict[str, Any]:
    """Describe the exact context set consumed by the multi-key solve."""

    group_sizes = [len(group) for group in context_templates]
    flat_templates = [template for group in context_templates for template in group]
    serialized_groups = json.dumps(
        context_templates,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "context_variant_policy": "all_existing_context_templates_no_subsampling",
        "context_subsampling": False,
        "context_group_sizes": group_sizes,
        "num_original_contexts": group_sizes[0] if group_sizes else 0,
        "num_generated_prefix_contexts": sum(group_sizes[1:]),
        "num_context_variants": len(flat_templates),
        "current_expected_layout": "original_1_plus_generated_prefix_5",
        "matches_current_expected_layout": group_sizes == [1, 5],
        "ordered_context_templates": flat_templates,
        "ordered_context_template_sha256": [
            hashlib.sha256(template.encode("utf-8")).hexdigest()
            for template in flat_templates
        ],
        "ordered_context_template_groups_sha256": hashlib.sha256(
            serialized_groups.encode("utf-8")
        ).hexdigest(),
    }


def _ensure_context_multikey_provenance_record(
    hparams: AlphaEditHyperParams,
    context_templates: List[List[str]],
) -> Dict[str, Any]:
    """Validate and persist the ordered context set once per JSONL file."""

    provenance = _context_multikey_template_provenance(context_templates)
    expected_group_sizes = getattr(
        hparams, "context_multikey_expected_group_sizes", None
    )
    if (
        expected_group_sizes is not None
        and provenance["context_group_sizes"]
        != [int(size) for size in expected_group_sizes]
    ):
        raise RuntimeError(
            "AlphaEdit context multi-key layout changed before the solve: "
            f"got {provenance['context_group_sizes']}, expected "
            f"{list(expected_group_sizes)}. Refusing to subsample or silently "
            "drop an existing context."
        )

    path_value = getattr(hparams, "context_multikey_log_path", None)
    if not path_value:
        return provenance
    path = Path(path_value).resolve()
    path_key = str(path)
    if path_key in _CONTEXT_MULTIKEY_PROVENANCE_LOGGED_PATHS:
        return provenance

    already_logged = False
    if path.exists() and path.stat().st_size:
        with path.open("r", encoding="utf-8") as handle:
            already_logged = any(
                '"stage": "context_set_provenance"' in line
                for line in handle
            )
    if not already_logged:
        _append_context_multikey_record(
            hparams,
            {
                "stage": "context_set_provenance",
                **provenance,
            },
        )
    _CONTEXT_MULTIKEY_PROVENANCE_LOGGED_PATHS.add(path_key)
    return provenance


def _context_multikey_diagnostics(
    raw_keys: torch.Tensor,
    mean_keys: torch.Tensor,
    weights: torch.Tensor,
) -> Dict[str, Any]:
    """Return trace-level diagnostics without retaining a dense extra Gram."""

    normalized_weights = weights.to(
        device=raw_keys.device,
        dtype=raw_keys.dtype,
    )
    normalized_weights = normalized_weights / normalized_weights.sum()
    total_trace = torch.einsum(
        "v,bvd->", normalized_weights, raw_keys.square()
    )
    mean_trace = mean_keys.square().sum()
    deviation_trace = (total_trace - mean_trace).clamp_min(0)
    return {
        "num_facts": int(raw_keys.shape[0]),
        "num_context_variants": int(raw_keys.shape[1]),
        "context_weights": [
            float(value)
            for value in normalized_weights.detach().float().cpu().tolist()
        ],
        "mean_key_gram_trace": float(mean_trace.detach().float().item()),
        "deviation_gram_trace": float(
            deviation_trace.detach().float().item()
        ),
        "deviation_trace_fraction": float(
            (
                deviation_trace
                / total_trace.clamp_min(torch.finfo(total_trace.dtype).eps)
            )
            .detach()
            .float()
            .item()
        ),
    }


def _context_multikey_realization_diagnostics(
    update: torch.Tensor,
    raw_keys: torch.Tensor,
    shared_targets: torch.Tensor,
    weights: torch.Tensor,
) -> Dict[str, float]:
    """Measure whether all retained contexts realize the shared response."""

    normalized_weights = weights.to(
        device=raw_keys.device,
        dtype=raw_keys.dtype,
    )
    normalized_weights = normalized_weights / normalized_weights.sum()
    realized = torch.einsum("od,bvd->bvo", update, raw_keys)
    target_by_context = shared_targets[:, None, :].to(
        device=realized.device,
        dtype=realized.dtype,
    )
    target_errors = (realized - target_by_context).norm(dim=-1)
    realized_mean = torch.einsum(
        "v,bvo->bo", normalized_weights, realized
    )
    response_spread = (realized - realized_mean[:, None, :]).norm(dim=-1)
    weighted_target_error = (
        target_errors * normalized_weights.unsqueeze(0)
    ).sum(dim=1)
    weighted_response_spread = (
        response_spread * normalized_weights.unsqueeze(0)
    ).sum(dim=1)
    mean_target_error = (
        realized_mean - target_by_context[:, 0, :]
    ).norm(dim=-1)
    return {
        "context_target_l2_mean": float(
            weighted_target_error.detach().float().mean().item()
        ),
        "context_target_l2_max": float(
            target_errors.detach().float().max().item()
        ),
        "context_response_spread_l2_mean": float(
            weighted_response_spread.detach().float().mean().item()
        ),
        "mean_key_target_l2_mean": float(
            mean_target_error.detach().float().mean().item()
        ),
    }


def _key_gaussian_noise_diagnostics(
    clean_keys: torch.Tensor,
    noisy_keys: torch.Tensor,
    derived_seeds: List[int],
) -> Dict[str, Any]:
    """Summarize the actual perturbation drawn around each canonical key."""

    if clean_keys.shape != noisy_keys.shape or clean_keys.ndim != 2:
        raise ValueError(
            "clean/noisy key diagnostics require matching [B, D] tensors; "
            f"got {tuple(clean_keys.shape)} and {tuple(noisy_keys.shape)}"
        )
    clean = clean_keys.to(device=noisy_keys.device, dtype=noisy_keys.dtype)
    noise = noisy_keys - clean
    clean_norm = clean.norm(dim=-1)
    noise_norm = noise.norm(dim=-1)
    eps = torch.finfo(noisy_keys.dtype).eps
    relative_norm = noise_norm / clean_norm.clamp_min(eps)
    cosine = torch.nn.functional.cosine_similarity(clean, noisy_keys, dim=-1)
    return {
        "num_facts": int(clean.shape[0]),
        "key_dim": int(clean.shape[1]),
        "derived_noise_seeds": [int(seed) for seed in derived_seeds],
        "clean_key_l2_mean": float(clean_norm.detach().float().mean().item()),
        "noise_l2_mean": float(noise_norm.detach().float().mean().item()),
        "actual_relative_noise_l2_mean": float(
            relative_norm.detach().float().mean().item()
        ),
        "actual_relative_noise_l2_max": float(
            relative_norm.detach().float().max().item()
        ),
        "clean_noisy_cosine_mean": float(
            cosine.detach().float().mean().item()
        ),
    }


def _key_gaussian_noise_realization_diagnostics(
    update: torch.Tensor,
    clean_keys: torch.Tensor,
    noisy_keys: torch.Tensor,
    shared_targets: torch.Tensor,
) -> Dict[str, float]:
    """Compare the requested response at the noisy and canonical keys."""

    clean = clean_keys.to(device=update.device, dtype=update.dtype)
    noisy = noisy_keys.to(device=update.device, dtype=update.dtype)
    targets = shared_targets.to(device=update.device, dtype=update.dtype)
    clean_realized = torch.einsum("od,bd->bo", update, clean)
    noisy_realized = torch.einsum("od,bd->bo", update, noisy)
    clean_error = (clean_realized - targets).norm(dim=-1)
    noisy_error = (noisy_realized - targets).norm(dim=-1)
    response_shift = (noisy_realized - clean_realized).norm(dim=-1)
    return {
        "noisy_key_target_l2_mean": float(
            noisy_error.detach().float().mean().item()
        ),
        "canonical_key_target_l2_mean": float(
            clean_error.detach().float().mean().item()
        ),
        "noisy_vs_canonical_response_l2_mean": float(
            response_shift.detach().float().mean().item()
        ),
    }


def _get_module_device(model: AutoModelForCausalLM, module_name: str) -> torch.device:
    module = nethook.get_module(model, module_name)
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _template_requests_for_cache(requests: List[Dict]) -> List[Dict]:
    cache_requests = deepcopy(requests)
    for request in cache_requests:
        if "{}" not in request["prompt"]:
            request["prompt"] = request["prompt"].replace(
                request["subject"], "{}", 1
            )
    return cache_requests


def apply_AlphaEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    reset_cache=False,
    reset_oedit=False,
    **kwargs
) -> Dict[str, Tuple[torch.Tensor]]:
  #-> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :param reset_cache: If true, will reset cache_c_new to False, forcing re-initialization of cache_c.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    global P, P_loaded, cache_c, cache_c_new
    global cache_c_outer_key_signature
    
    # Reset cache if requested
    if reset_cache:
        cache_c_new = False
        cache_c_outer_key_signature = None

    if bool(getattr(hparams, "oedit_enabled", False)) and len(requests) != 1:
        raise ValueError("O-Edit requires exactly one request per sequential apply")
    if reset_oedit or reset_cache:
        reset_oedit_state(hparams)

    requested_key_mode = _outer_key_mode(hparams)
    requested_cache_signature = _outer_key_cache_signature(hparams)
    if (
        cache_c_new
        and cache_c_outer_key_signature is not None
        and cache_c_outer_key_signature != requested_cache_signature
    ):
        raise RuntimeError(
            "AlphaEdit cache_c was initialized with outer-key signature "
            f"{cache_c_outer_key_signature!r}, but this call requests "
            f"{requested_cache_signature!r}. Pass reset_cache=True and "
            "restart from the Base model before changing key mode, Gaussian "
            "noise scale, or Gaussian noise seed."
        )

    weights_copy = {}
    if copy:
        model = deepcopy(model)
    oedit_regularizer = _prepare_oedit_regularizer(model, hparams)
    
    # Calculate the null-space projection matrix P
    # Please ensure that you have downloaded "null_space_project.pt" to the easyedit folder beforehand, or get the P by following calculation
    if not os.path.exists(hparams.P_loc):
        print(os.path.abspath(hparams.P_loc))
        print(f"The null-space projection matrix P does not exist and now calculate.")
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        if (
            "llama" in hparams.model_name.lower()
            or "qwen" in hparams.model_name.lower()
            or "gpt-j-6b" in hparams.model_name.lower()
        ):
            P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        elif "gpt2-xl" in hparams.model_name.lower():
            P = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
        else:
            raise ValueError(
                f"Unsupported AlphaEdit P initialization for model {hparams.model_name}"
            )
        del W_out
        for i, layer in enumerate(hparams.layers):
            P[i,:,:] = get_project(model, tok, layer, hparams)
        torch.save(P, hparams.P_loc)
        P_loaded = True
    elif P_loaded == False:
        P = torch.load(hparams.P_loc)
        P_loaded = True

    # Maintain the global variable cache_c to avoid redundant computations.
    # If this is the first calculation (i.e., cache_c_new == false), then initialize cache_c first
    if not cache_c_new:
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        if "llama" in hparams.model_name.lower() or "qwen" in hparams.model_name.lower():
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        elif "gpt2-xl" in hparams.model_name.lower():
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
        else:
            raise ValueError(
                f"Unsupported AlphaEdit cache_c initialization for model {hparams.model_name}"
            )
        del W_out
        cache_c_new = True
        cache_c_outer_key_signature = requested_cache_signature
    
    deltas = execute_AlphaEdit(model, tok, requests, hparams, cache_template=cache_template)

    update_matrices = {}
    for w_name, upd_m in deltas.items():
        w = nethook.get_parameter(model, w_name)
        update_matrices[w_name] = upd_matrix_match_shape(
            upd_m.to(w.device), w.shape
        ).detach()
    # execute_AlphaEdit already applies SPHERE to each layer before computing
    # the next layer's keys/residual.  Its returned deltas are the projected
    # updates; projecting them again here would apply SPHERE twice.
    if bool(getattr(hparams, "sphere_enabled", False)):
        for layer in hparams.layers:
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            save_committed_outer_layer_artifact(
                hparams,
                method="AlphaEdit",
                requests=requests,
                layer=int(layer),
                committed_update=update_matrices[weight_name],
            )

    with torch.no_grad():
        for w_name, upd_matrix in update_matrices.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix.to(w.device).float()

    if oedit_regularizer is not None:
        final_name = f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight"
        # Commit only after all real outer writes. Temporary per-layer writes
        # inside execute_AlphaEdit do not advance the O-Edit history.
        oedit_commit = oedit_regularizer.commit_update(update_matrices[final_name])
        append_rgr_record(
            getattr(hparams, "oedit_log_path", None),
            {"method": "AlphaEdit", "stage": "oedit_commit",
             "case_id": requests[0].get("case_id"),
             "write_layer": int(hparams.layers[-1]), **oedit_commit},
        )

    # Preserve the keys produced by the update that was actually committed.
    context_templates = get_context_templates(model, tok)
    cache_requests = _template_requests_for_cache(requests)
    for i, layer in enumerate(hparams.layers):
        if requested_key_mode == "exact_context_multikey":
            raw_context_keys, context_weights = compute_ks_context_components(
                model,
                tok,
                cache_requests,
                hparams,
                layer,
                context_templates,
            )
            # cache_c is deliberately a CPU accumulator.  Passing it directly
            # to the compression helper adds mean + within-fact deviation
            # Grams in place and avoids allocating another dense D x D matrix.
            raw_context_keys = raw_context_keys.detach().to(
                device="cpu",
                dtype=cache_c[i].dtype,
            )
            context_weights = context_weights.to(
                device="cpu",
                dtype=cache_c[i].dtype,
            )
            cache_terms = same_target_multikey_terms(
                raw_context_keys,
                torch.zeros(
                    (raw_context_keys.shape[0], 1),
                    device="cpu",
                    dtype=cache_c[i].dtype,
                ),
                weights=context_weights,
                key_gram=cache_c[i],
                compute_target_key_cross=False,
            )
            cache_diagnostics = _context_multikey_diagnostics(
                raw_context_keys,
                cache_terms.mean_keys,
                context_weights,
            )
            _append_context_multikey_record(
                hparams,
                {
                    "stage": "committed_cache_update",
                    "edit_index": getattr(
                        hparams, "analysis_current_edit_index", None
                    ),
                    "layer": int(layer),
                    "case_ids": [request.get("case_id") for request in requests],
                    **cache_diagnostics,
                },
            )
            del raw_context_keys, context_weights, cache_terms
        elif requested_key_mode == "canonical_gaussian_noise":
            noisy_keys, clean_keys, derived_seeds = compute_ks_gaussian_noise(
                model,
                tok,
                cache_requests,
                hparams,
                layer,
            )
            noisy_keys = noisy_keys.detach().to(
                device="cpu",
                dtype=cache_c[i].dtype,
            )
            clean_keys = clean_keys.detach().to(
                device="cpu",
                dtype=cache_c[i].dtype,
            )
            layer_ks = noisy_keys.T
            cache_c[i, :, :] += layer_ks @ layer_ks.T
            _append_key_gaussian_noise_record(
                hparams,
                {
                    "stage": "committed_cache_update",
                    "edit_index": getattr(
                        hparams, "analysis_current_edit_index", None
                    ),
                    "layer": int(layer),
                    "case_ids": [request.get("case_id") for request in requests],
                    "relative_std": float(
                        hparams.key_gaussian_noise_relative_std
                    ),
                    "base_seed": int(hparams.key_gaussian_noise_seed),
                    **_key_gaussian_noise_diagnostics(
                        clean_keys,
                        noisy_keys,
                        derived_seeds,
                    ),
                },
            )
            del noisy_keys, clean_keys, layer_ks
        else:
            # Keep the native path byte-for-byte equivalent when the feature
            # flag is disabled.
            layer_ks = compute_ks(
                model,
                tok,
                cache_requests,
                hparams,
                layer,
                context_templates,
            ).T.to(dtype=cache_c[i].dtype)
            cache_c[i, :, :] += layer_ks.cpu() @ layer_ks.cpu().T

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_AlphaEdit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """Execute native AlphaEdit or NSE over its projected solve."""

    if bool(getattr(hparams, "oedit_enabled", False)):
        if len(requests) != 1:
            raise ValueError("O-Edit requires exactly one request per sequential execute")
        _prepare_oedit_regularizer(model, hparams)
        # v* depends on the current accumulated weight history. Neither old
        # target-cache reads nor writes are valid for this sequential state.
        cache_template = None

    if not bool(getattr(hparams, "nse_enabled", False)):
        return _execute_AlphaEdit_single_pass(
            model,
            tok,
            requests,
            hparams,
            cache_template=cache_template,
        )

    max_iterations = int(getattr(hparams, "nse_max_iterations", 3))
    if max_iterations < 2:
        raise ValueError("NSE nse_max_iterations must be >= 2")
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    sequence_start = {
        name: weight.detach().clone() for name, weight in weights.items()
    }
    accumulated: Dict[str, torch.Tensor] = {}
    context_templates = get_context_templates(model, tok)
    normalized_requests = deepcopy(requests)
    for request in normalized_requests:
        if request["target_new"][0] != " ":
            request["target_new"] = " " + request["target_new"]
        if "{}" not in request["prompt"]:
            request["prompt"] = request["prompt"].replace(
                request["subject"], "{}", 1
            )

    try:
        for iteration in range(max_iterations):
            z_layer = hparams.layers[-1]
            current_zs = get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=[
                    request["prompt"] for request in normalized_requests
                ],
                words=[
                    request["subject"] for request in normalized_requests
                ],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1].T
            target_zs = torch.stack(
                [
                    get_nse_target(
                        hparams,
                        request["case_id"],
                        device=current_zs.device,
                        dtype=current_zs.dtype,
                    )
                    for request in normalized_requests
                ],
                dim=1,
            )
            errors = torch.linalg.norm(target_zs - current_zs, dim=0)
            mask = errors.gt(
                float(getattr(hparams, "nse_alpha", 2.5))
            ) & errors.lt(float(getattr(hparams, "nse_upper_bound", 50)))
            unresolved = [
                normalized_requests[index]
                for index in torch.nonzero(mask, as_tuple=True)[0].tolist()
            ]
            print(
                f"[NSE][AlphaEdit] iteration={iteration} unresolved="
                f"{len(unresolved)}/{len(normalized_requests)} "
                f"mean_error={float(errors.mean().item()):.4f}"
            )
            append_official_baseline_record(
                getattr(hparams, "official_baseline_log_path", None),
                {
                    "method": "AlphaEdit",
                    "baseline": "NSE",
                    "stage": "iteration_filter",
                    "iteration": iteration,
                    "num_requests": len(normalized_requests),
                    "num_unresolved": len(unresolved),
                    "mean_target_error": float(errors.mean().item()),
                    "max_target_error": float(errors.max().item()),
                },
            )
            if not unresolved or iteration == max_iterations - 1:
                break
            round_deltas = _execute_AlphaEdit_single_pass(
                model,
                tok,
                unresolved,
                hparams,
                cache_template=None,
            )
            with torch.no_grad():
                for name, round_update_cpu in round_deltas.items():
                    accumulated[name] = (
                        accumulated.get(
                            name,
                            torch.zeros_like(round_update_cpu),
                        )
                        + round_update_cpu
                    )
                    weights[name][...] += round_update_cpu.to(
                        weights[name].device
                    ).float()
    finally:
        with torch.no_grad():
            for name, weight in weights.items():
                weight[...] = sequence_start[name]

    for name, weight in weights.items():
        if name not in accumulated:
            accumulated[name] = torch.zeros_like(weight, device="cpu")
    return accumulated


def _execute_AlphaEdit_single_pass(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the AlphaEdit update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    _validate_o0_axis_configuration(hparams, hparams.layers[-1])
    deltas = {}
    rgr_enabled = bool(
        getattr(hparams, "residual_gain_regularization", False)
    )
    nse_enabled = bool(getattr(hparams, "nse_enabled", False))
    encore_enabled = bool(getattr(hparams, "encore_enabled", False))
    sadr_enabled = bool(getattr(hparams, "sadr_regularization", False))
    outer_key_mode = _outer_key_mode(hparams)
    context_multikey_enabled = outer_key_mode == "exact_context_multikey"
    key_gaussian_noise_enabled = (
        outer_key_mode == "canonical_gaussian_noise"
    )
    tangent_layer_allocation_enabled = _tangent_layer_allocation_enabled(
        hparams
    )
    if tangent_layer_allocation_enabled and len(requests) != 1:
        raise ValueError(
            "Strict AlphaEdit tangent layer allocation currently requires "
            f"batch size 1; received {len(requests)} requests."
        )
    if tangent_layer_allocation_enabled and nse_enabled:
        raise ValueError(
            "Strict tangent layer allocation cannot be combined with NSE's "
            "neuron-restricted outer solve, which may rotate the requested "
            "output component."
        )
    if tangent_layer_allocation_enabled and bool(
        getattr(hparams, "sphere_enabled", False)
    ):
        raise ValueError(
            "Strict tangent layer allocation cannot be combined with SPHERE "
            "post-processing, which changes the realized update after the "
            "local tangency-constrained solve."
        )
    analysis_capture_enabled = any(
        bool(getattr(hparams, name, False))
        for name in (
            "analysis_capture_inner_gain",
            "analysis_capture_latents",
            "analysis_capture_outer",
        )
    )

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"] = " " + request["target_new"]
        if key_gaussian_noise_enabled:
            requests[i]["prompt"] = canonical_prompt_template(request)
        else:
            if '{}' not in request['prompt']:
                assert request['subject'] in request['prompt'] or \
                       print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")
            requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')
        print(
            f"Executing AlphaEdit algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    if context_multikey_enabled:
        _ensure_context_multikey_provenance_record(hparams, context_templates)
    elif key_gaussian_noise_enabled:
        _ensure_key_gaussian_noise_provenance_record(hparams)
    # context_templates = [["{}"]] 
    z_layer = hparams.layers[-1]
    z_device = _get_module_device(model, hparams.layer_module_tmp.format(z_layer))
    z_list = []

    for request in requests:
        # The recovery runner supplies the one-based stream index explicitly.
        # Publish it before each per-request inner solve so latent/scalar
        # artifacts cannot fall back to case_id (which is not an edit order).
        if request.get("edit_index") is not None:
            hparams.analysis_current_edit_index = int(request["edit_index"])
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
            and not rgr_enabled
            and not nse_enabled
            and not encore_enabled
            and not sadr_enabled
            and not analysis_capture_enabled
            and not bool(getattr(hparams, "analysis_capture_virtual_actual", False))
            and not bool(getattr(hparams, "o0_axis_preservation_enabled", False))
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to(z_device))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        
        if not data_loaded:
            if nse_enabled:
                cur_z = get_nse_target(
                    hparams,
                    request["case_id"],
                    device=z_device,
                    dtype=torch.float32,
                )
            else:
                cur_z = compute_z(
                    model,
                    tok,
                    request,
                    hparams,
                    z_layer,
                    context_templates,
                )

            # else:
            #     from .compute_z_kl import compute_z
            #     cur_z = compute_z(
            #         model,
            #         tok,
            #         request,
            #         hparams,
            #         z_layer,
            #         context_templates,
            #     )

            z_list.append(cur_z)

            if (
                cache_fname is not None
                and not rgr_enabled
                and not nse_enabled
                and not encore_enabled
                and not sadr_enabled
                and not analysis_capture_enabled
                and not bool(getattr(hparams, "o0_axis_preservation_enabled", False))
            ):
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1).to(dtype=torch.float32)

    # The feature-on path plans all layer components together from the clean
    # pre-edit state.  It must happen before the temporary per-layer writes in
    # the loop below; planning greedily from each remaining error would restore
    # the same terminal clean-up mechanism as AlphaEdit's native `/remaining`
    # rule.  Feature-off performs none of these extra forwards and preserves
    # the native operation/reduction order below.
    tangent_plan = None
    if tangent_layer_allocation_enabled:
        local_references = collect_clean_local_output_references(
            model,
            tok,
            requests,
            hparams,
            get_module_input_output_at_words,
        )
        # The terminal reference is exactly the current z-layer output for
        # these same prompts/tokens, so reuse it instead of a sixth forward.
        initial_cur_zs = local_references[:, -1, :].T.to(
            device=zs.device, dtype=zs.dtype
        )
        initial_targets = zs - initial_cur_zs
        tangent_plan = build_strict_joint_tangent_plan(
            initial_targets,
            local_references,
            rcond=getattr(
                hparams, "tangent_layer_allocation_rcond", None
            ),
            max_relative_slack=getattr(
                hparams,
                "tangent_layer_allocation_max_relative_slack",
                None,
            ),
            require_batch_size_one=True,
        )
        plan_diagnostics = tangent_plan_diagnostics(tangent_plan)
        _append_tangent_layer_allocation_record(
            hparams,
            {
                "stage": "joint_plan",
                "edit_index": getattr(
                    hparams, "analysis_current_edit_index", None
                ),
                "case_ids": [request.get("case_id") for request in requests],
                "layers": [int(layer) for layer in hparams.layers],
                "prompt_subject_sha256": [
                    hashlib.sha256(
                        json.dumps(
                            {
                                "prompt_template": request["prompt"],
                                "subject": request["subject"],
                                "formatted_prompt": request["prompt"].format(
                                    request["subject"]
                                ),
                                "fact_token": hparams.fact_token,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    for request in requests
                ],
                "context_multikey_enabled": context_multikey_enabled,
                "key_gaussian_noise_enabled": key_gaussian_noise_enabled,
                "outer_key_mode": outer_key_mode,
                "configured_rcond": getattr(
                    hparams, "tangent_layer_allocation_rcond", None
                ),
                "configured_max_relative_slack": getattr(
                    hparams,
                    "tangent_layer_allocation_max_relative_slack",
                    None,
                ),
                **plan_diagnostics,
            },
        )
        print(
            "[TANGENT][AlphaEdit] strict joint plan "
            f"relative_slack_max={plan_diagnostics['relative_slack_max']:.6g} "
            f"max_abs_dot={plan_diagnostics['max_abs_tangency_dot']:.6g}"
        )

    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations.  The feature-off branch deliberately
        # calls the unchanged legacy reducer.  The opt-in branch retains the
        # same already-forwarded raw contexts until the shared target response
        # for this layer is known.
        raw_context_keys = None
        context_weights = None
        multikey_diagnostics = None
        clean_gaussian_keys = None
        noisy_gaussian_keys = None
        gaussian_noise_seeds = None
        if context_multikey_enabled:
            raw_context_keys, context_weights = compute_ks_context_components(
                model,
                tok,
                requests,
                hparams,
                layer,
                context_templates,
            )
            num_key_pairs = raw_context_keys.size(0)
        elif key_gaussian_noise_enabled:
            (
                noisy_gaussian_keys,
                clean_gaussian_keys,
                gaussian_noise_seeds,
            ) = compute_ks_gaussian_noise(
                model,
                tok,
                requests,
                hparams,
                layer,
            )
            # One noisy key per fact.  There is no context dimension and no
            # averaging: this tensor enters the native AlphaEdit solve as-is.
            layer_ks = noisy_gaussian_keys.T
            num_key_pairs = layer_ks.size(1)
        else:
            layer_ks = compute_ks(
                model, tok, requests, hparams, layer, context_templates
            ).T
            num_key_pairs = layer_ks.size(1)
        print(f"Writing {num_key_pairs} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T.to(device=zs.device, dtype=zs.dtype)
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        if tangent_layer_allocation_enabled:
            # Consume the immutable B-column joint plan.  In particular, do
            # not repeat it over raw context variants: exact context multi-key
            # compression represents those variants in the Gram while sharing
            # this single response per fact.
            resid = tangent_plan.layer_residuals[i]
            if resid.size(1) != targets.size(1):
                raise RuntimeError(
                    "Tangent allocation fact columns changed before the "
                    f"outer solve: resid={tuple(resid.shape)}, "
                    f"targets={tuple(targets.shape)}"
                )
        else:
            # Native AlphaEdit rule.  Keep the exact expression and reduction
            # order when tangent allocation is disabled.
            repeat_factor = (num_key_pairs // targets.size(1))
            targets = targets.repeat_interleave(repeat_factor, dim=1)
            resid = targets / (len(hparams.layers) - i)
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        target_device = weights[weight_name].device
        work_dtype = cache_c[i].dtype
        resid = resid.to(device=target_device, dtype=work_dtype)
        proj = P[i, :, :].to(device=target_device, dtype=work_dtype)
        cov_cache = cache_c[i, :, :].to(device=target_device, dtype=work_dtype)
        if context_multikey_enabled:
            raw_context_keys = raw_context_keys.to(
                device=target_device,
                dtype=work_dtype,
            )
            context_weights = context_weights.to(
                device=target_device,
                dtype=work_dtype,
            )
            # Start from cache_c and add this edit's compressed multi-key Gram
            # in place.  This avoids retaining both a dense data Gram and a
            # dense cache+data Gram. All V raw contexts constrain the system,
            # while the shared response remains one B-column target per fact.
            # `.to()` may alias the CPU global cache when device/dtype already
            # match.  Clone only in that case: on GPU, cov_cache is already a
            # private ~D x D transfer and a second clone would waste ~0.8 GiB
            # for Llama-3's 14336-dimensional key space.
            solve_gram = protect_accumulator_owner_from_inplace_add(
                cov_cache,
                cache_c[i],
            )
            multikey_terms = same_target_multikey_terms(
                raw_context_keys,
                resid.T,
                weights=context_weights,
                key_gram=solve_gram,
                compute_target_key_cross=False,
            )
            layer_ks = multikey_terms.mean_keys
            multikey_diagnostics = _context_multikey_diagnostics(
                raw_context_keys,
                layer_ks,
                context_weights,
            )
            system = (
                proj @ multikey_terms.key_gram
                + hparams.L2
                * torch.eye(
                    layer_ks.shape[0],
                    dtype=work_dtype,
                    device=target_device,
                )
            )
            # For shared targets, the exact weighted target/key cross is
            # resid @ mean_keys.T.  Use the native AlphaEdit multiplication
            # order instead of materializing that [H, D] tensor (about
            # 224 MiB in fp32 for Llama-3-8B).
            rhs = proj @ layer_ks @ resid.T
        else:
            layer_ks = layer_ks.to(device=target_device, dtype=work_dtype)
            # Native AlphaEdit path.  Keep the exact expression and reduction
            # order for both the legacy mean key and the single noisy key.
            system = (
                proj @ (layer_ks @ layer_ks.T + cov_cache)
                + hparams.L2
                * torch.eye(
                    layer_ks.shape[0],
                    dtype=work_dtype,
                    device=target_device,
                )
            )
            rhs = proj @ layer_ks @ resid.T
        if nse_enabled:
            selected, selection_stats = nse_select_neurons(
                layer_ks,
                float(getattr(hparams, "nse_neuron_threshold", 1.0)),
            )
            upd_matrix = nse_restricted_solve(
                system,
                rhs,
                selected,
            )
            print(
                f"[NSE][AlphaEdit] L{layer} neurons="
                f"{int(selection_stats['selected_neurons'])}/"
                f"{int(selection_stats['total_neurons'])} "
                f"activation={selection_stats['activation_fraction']:.4f}"
            )
            append_official_baseline_record(
                getattr(hparams, "official_baseline_log_path", None),
                {
                    "method": "AlphaEdit",
                    "baseline": "NSE",
                    "stage": "neuron_restricted_solve",
                    "layer": int(layer),
                    "case_ids": [
                        request.get("case_id") for request in requests
                    ],
                    **selection_stats,
                },
            )
        else:
            upd_matrix = torch.linalg.solve(system, rhs)

        # Adjust update matrix shape
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        from ...util.rewrite_update_action import rewrite_update_action
        realized_update = rewrite_update_action(
            nethook.get_module(model, hparams.rewrite_module_tmp.format(layer)),
            upd_matrix, layer_ks,
        )
        if tangent_layer_allocation_enabled:
            reference = tangent_plan.local_references[:, i, :].to(
                device=realized_update.device,
                dtype=realized_update.dtype,
            )
            requested_component = resid.T.to(
                device=realized_update.device,
                dtype=realized_update.dtype,
            )
            unit_reference = torch.nn.functional.normalize(
                reference, dim=-1
            )
            requested_dot = torch.einsum(
                "bh,bh->b", unit_reference, requested_component
            )
            realized_dot = torch.einsum(
                "bh,bh->b", unit_reference, realized_update.T
            )
            _append_tangent_layer_allocation_record(
                hparams,
                {
                    "stage": "outer_realization",
                    "edit_index": getattr(
                        hparams, "analysis_current_edit_index", None
                    ),
                    "case_ids": [
                        request.get("case_id") for request in requests
                    ],
                    "layer": int(layer),
                    "layer_index": int(i),
                    "context_multikey_enabled": context_multikey_enabled,
                    "key_gaussian_noise_enabled": key_gaussian_noise_enabled,
                    "outer_key_mode": outer_key_mode,
                    "requested_component_l2_mean": float(
                        requested_component.float().norm(dim=-1).mean().item()
                    ),
                    "realized_mean_key_write_l2_mean": float(
                        realized_update.T.float().norm(dim=-1).mean().item()
                    ),
                    "requested_tangency_dot_abs_max": float(
                        requested_dot.float().abs().max().item()
                    ),
                    "realized_tangency_dot_abs_max": float(
                        realized_dot.float().abs().max().item()
                    ),
                },
            )
        if context_multikey_enabled:
            realization_diagnostics = _context_multikey_realization_diagnostics(
                upd_matrix.to(
                    device=raw_context_keys.device,
                    dtype=raw_context_keys.dtype,
                ),
                raw_context_keys,
                resid.T,
                context_weights,
            )
            _append_context_multikey_record(
                hparams,
                {
                    "stage": "outer_solve",
                    "edit_index": getattr(
                        hparams, "analysis_current_edit_index", None
                    ),
                    "layer": int(layer),
                    "case_ids": [request.get("case_id") for request in requests],
                    **multikey_diagnostics,
                    **realization_diagnostics,
                },
            )
        elif key_gaussian_noise_enabled:
            gaussian_diagnostics = _key_gaussian_noise_diagnostics(
                clean_gaussian_keys,
                noisy_gaussian_keys,
                gaussian_noise_seeds,
            )
            realization_diagnostics = (
                _key_gaussian_noise_realization_diagnostics(
                    upd_matrix,
                    clean_gaussian_keys,
                    noisy_gaussian_keys,
                    resid.T,
                )
            )
            _append_key_gaussian_noise_record(
                hparams,
                {
                    "stage": "outer_solve",
                    "edit_index": getattr(
                        hparams, "analysis_current_edit_index", None
                    ),
                    "layer": int(layer),
                    "case_ids": [request.get("case_id") for request in requests],
                    "relative_std": float(
                        hparams.key_gaussian_noise_relative_std
                    ),
                    "base_seed": int(hparams.key_gaussian_noise_seed),
                    **gaussian_diagnostics,
                    **realization_diagnostics,
                },
            )

        print("orig norm", torch.linalg.norm(weights[weight_name].float()))
        print("upd norm", torch.linalg.norm(upd_matrix.float()))
        save_outer_layer_artifact(
            hparams,
            method="AlphaEdit",
            requests=requests,
            layer=int(layer),
            keys=layer_ks,
            target_error=targets,
            distributed_residual=resid,
            realized_update=realized_update,
            update_frobenius=float(upd_matrix.float().norm().item()),
        )

        if bool(getattr(hparams, "sphere_enabled", False)):
            # Match the official AlphaEdit+SPHERE update order: project this
            # layer now, then let all later key/residual forwards observe the
            # projected temporary write.  Retain the native-solve diagnostics
            # above; apply_AlphaEdit_to_model records the committed realization.
            # Returned deltas already contain this projection.
            upd_matrix = project_updates_with_sphere(
                model,
                hparams,
                {weight_name: upd_matrix},
                method="AlphaEdit",
                parameter_getter=nethook.get_parameter,
            )[weight_name]

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix.float()
            deltas[weight_name] = (
                upd_matrix.detach().cpu()
            )
        
        # Clear GPU memory
        #del U,S,cov
        for x in [layer_ks, cur_zs, targets, realized_update]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    
    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]
    
    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
    hparams=None,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            hparams.stats_dir,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            hparams=hparams,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    target_device = _get_module_device(model, layer_name)
    return torch.inverse(COV_CACHE[key].to(target_device)) if inv else COV_CACHE[key].to(target_device)


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by AlphaEdit does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE

def get_project(model, tok, layer, hparams):
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
        hparams=hparams
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T
