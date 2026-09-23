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
from ...util.generate import generate_fast
from ...util.globals import *
from ...util.official_editing_baselines import (
    append_official_baseline_record,
    get_encore_gram,
    get_nse_gram,
    get_nse_target,
    nse_restricted_solve,
    nse_select_neurons,
    project_updates_with_sphere,
    update_encore_gram,
    update_nse_gram,
)
from ...util.edit_analysis_artifacts import (
    save_committed_outer_layer_artifact,
    save_outer_layer_artifact,
)
from ...util.key_gaussian_noise import canonical_prompt_template

from .compute_ks import compute_ks, compute_ks_gaussian_noise
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .memit_hparams import MEMITHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}
_KEY_GAUSSIAN_NOISE_PROVENANCE_LOGGED_PATHS = set()


def _key_gaussian_noise_enabled(hparams: MEMITHyperParams) -> bool:
    enabled = bool(getattr(hparams, "key_gaussian_noise_enabled", False))
    if enabled:
        relative_std = float(
            getattr(hparams, "key_gaussian_noise_relative_std", 0.05)
        )
        if not np.isfinite(relative_std) or relative_std <= 0.0:
            raise ValueError(
                "key_gaussian_noise_relative_std must be finite and > 0; "
                f"got {relative_std!r}"
            )
    return enabled


def _append_key_gaussian_noise_record(
    hparams: MEMITHyperParams,
    record: Dict[str, Any],
) -> None:
    path_value = getattr(hparams, "key_gaussian_noise_log_path", None)
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "MEMIT",
        "feature": "key_gaussian_noise",
        "mode": "one_canonical_key_plus_relative_gaussian_noise",
        **record,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _ensure_key_gaussian_noise_provenance_record(
    hparams: MEMITHyperParams,
) -> Dict[str, Any]:
    if not _key_gaussian_noise_enabled(hparams):
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
        "target_formation": "native_memit_compute_z_unchanged",
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


def _key_gaussian_noise_diagnostics(
    clean_keys: torch.Tensor,
    noisy_keys: torch.Tensor,
    derived_seeds: List[int],
) -> Dict[str, Any]:
    if clean_keys.shape != noisy_keys.shape or clean_keys.ndim != 2:
        raise ValueError(
            "clean/noisy key diagnostics require matching [B, D] tensors; "
            f"got {tuple(clean_keys.shape)} and {tuple(noisy_keys.shape)}"
        )
    clean = clean_keys.to(device=noisy_keys.device, dtype=noisy_keys.dtype)
    noise = noisy_keys - clean
    clean_norm = clean.norm(dim=-1)
    noise_norm = noise.norm(dim=-1)
    relative_norm = noise_norm / clean_norm.clamp_min(
        torch.finfo(noisy_keys.dtype).eps
    )
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


def apply_memit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    if _key_gaussian_noise_enabled(hparams):
        # Guarantee provenance even when NSE decides that every request is
        # already resolved and therefore performs no inner single-pass solve.
        _ensure_key_gaussian_noise_provenance_record(hparams)

    deltas = execute_memit(model, tok, requests, hparams, cache_template=cache_template)

    update_matrices = {}
    for w_name, (key_mat, val_mat) in deltas.items():
        w = nethook.get_parameter(model, w_name)
        upd_matrix = key_mat.to(w.device) @ val_mat.to(w.device).T
        update_matrices[w_name] = upd_matrix_match_shape(
            upd_matrix, w.shape
        ).detach()
    update_matrices = project_updates_with_sphere(
        model,
        hparams,
        update_matrices,
        method="MEMIT",
        parameter_getter=nethook.get_parameter,
    )
    if bool(getattr(hparams, "sphere_enabled", False)):
        for layer in hparams.layers:
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            save_committed_outer_layer_artifact(
                hparams,
                method="MEMIT",
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

    if bool(getattr(hparams, "nse_enabled", False)):
        context_templates = get_context_templates(model, tok)
        normalized_requests = deepcopy(requests)
        for request in normalized_requests:
            if _key_gaussian_noise_enabled(hparams):
                request["prompt"] = canonical_prompt_template(request)
            elif "{}" not in request["prompt"]:
                request["prompt"] = request["prompt"].replace(
                    request["subject"], "{}", 1
                )
        for layer in hparams.layers:
            if _key_gaussian_noise_enabled(hparams):
                noisy_keys, clean_keys, derived_seeds = (
                    compute_ks_gaussian_noise(
                        model,
                        tok,
                        normalized_requests,
                        hparams,
                        layer,
                    )
                )
                layer_keys = noisy_keys.T
                _append_key_gaussian_noise_record(
                    hparams,
                    {
                        "stage": "committed_nse_cache_update",
                        "edit_index": getattr(
                            hparams, "analysis_current_edit_index", None
                        ),
                        "layer": int(layer),
                        "case_ids": [
                            request.get("case_id") for request in requests
                        ],
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
            else:
                layer_keys = compute_ks(
                    model,
                    tok,
                    normalized_requests,
                    hparams,
                    layer,
                    context_templates,
                ).T
            update_nse_gram(
                hparams,
                layer=int(layer),
                layer_keys=layer_keys,
            )

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_memit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """Execute native MEMIT or NSE's iterative wrapper."""

    if not bool(getattr(hparams, "nse_enabled", False)):
        return _execute_memit_single_pass(
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
    accumulated: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    context_templates = get_context_templates(model, tok)
    normalized_requests = deepcopy(requests)
    for request in normalized_requests:
        if request["target_new"][0] != " ":
            request["target_new"] = " " + request["target_new"]
        if _key_gaussian_noise_enabled(hparams):
            request["prompt"] = canonical_prompt_template(request)
        elif "{}" not in request["prompt"]:
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
                track="out",
            ).T
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
                f"[NSE][MEMIT] iteration={iteration} unresolved="
                f"{len(unresolved)}/{len(normalized_requests)} "
                f"mean_error={float(errors.mean().item()):.4f}"
            )
            append_official_baseline_record(
                getattr(hparams, "official_baseline_log_path", None),
                {
                    "method": "MEMIT",
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
            round_deltas = _execute_memit_single_pass(
                model,
                tok,
                unresolved,
                hparams,
                cache_template=None,
            )
            with torch.no_grad():
                for name, (key_factor, value_factor) in round_deltas.items():
                    if name in accumulated:
                        old_key, old_value = accumulated[name]
                        accumulated[name] = (
                            torch.cat([old_key, key_factor], dim=1),
                            torch.cat([old_value, value_factor], dim=1),
                        )
                    else:
                        accumulated[name] = (key_factor, value_factor)
                    weight = weights[name]
                    update = upd_matrix_match_shape(
                        key_factor.to(weight.device)
                        @ value_factor.to(weight.device).T,
                        weight.shape,
                    )
                    weight[...] += update.float()
    finally:
        with torch.no_grad():
            for name, weight in weights.items():
                weight[...] = sequence_start[name]

    if not accumulated:
        for name, weight in weights.items():
            key_dim = (
                weight.shape[1]
                if "llama" in hparams.model_name.lower()
                or "qwen" in hparams.model_name.lower()
                else weight.shape[0]
            )
            output_dim = weight.numel() // key_dim
            accumulated[name] = (
                torch.zeros((key_dim, 0), dtype=torch.float32),
                torch.zeros((output_dim, 0), dtype=torch.float32),
            )
    return accumulated


def _execute_memit_single_pass(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}
    rgr_enabled = bool(
        getattr(hparams, "residual_gain_regularization", False)
    )
    nse_enabled = bool(getattr(hparams, "nse_enabled", False))
    encore_enabled = bool(getattr(hparams, "encore_enabled", False))
    sadr_enabled = bool(getattr(hparams, "sadr_regularization", False))
    key_gaussian_noise_enabled = _key_gaussian_noise_enabled(hparams)
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
        elif '{}' not in request['prompt']:
            assert request['subject'] in request['prompt'] or \
                   print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")

            requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')

        if request.get("edit_index") is not None:
            hparams.analysis_current_edit_index = int(request["edit_index"])

    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
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
    if key_gaussian_noise_enabled:
        _ensure_key_gaussian_noise_provenance_record(hparams)
    z_layer = hparams.layers[-1]
    z_device = _get_module_device(model, hparams.layer_module_tmp.format(z_layer))
    z_list = []

    for request in requests:
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

            z_list.append(cur_z)

            if (
                cache_fname is not None
                and not rgr_enabled
                and not nse_enabled
                and not encore_enabled
                and not sadr_enabled
                and not analysis_capture_enabled
            ):
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        clean_gaussian_keys = None
        noisy_gaussian_keys = None
        gaussian_noise_seeds = None
        if key_gaussian_noise_enabled:
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
            # One noisy key per fact enters MEMIT's native closed-form solve;
            # there is no context axis and no original/prefix key average.
            layer_ks = noisy_gaussian_keys.T
        else:
            layer_ks = compute_ks(
                model,
                tok,
                requests,
                hparams,
                layer,
                context_templates,
            ).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
            track='out'
        ).T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        # Load covariance matrix
        force_recompute = False
        # force_recompute = layer != hparams.layers[0]
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
        )

        # Compute update in double precision
        layer_ks, targets = (
            layer_ks.double(),
            targets.double(),
        )
        
        system = (
            hparams.mom2_update_weight * cov.double()
            + (layer_ks @ layer_ks.T).to(cov.device)
        )
        if encore_enabled:
            norm_lambda = float(getattr(hparams, "encore_norm_lambda", 20.0))
            prior_gram = get_encore_gram(
                hparams,
                layer=int(layer),
                dimension=int(system.shape[0]),
                device=system.device,
                dtype=system.dtype,
            )
            prior_active = prior_gram is not None
            if prior_active:
                system.add_(prior_gram)
            system.diagonal().add_(norm_lambda)
            cumulative_keys = update_encore_gram(
                hparams,
                layer=int(layer),
                layer_keys=layer_ks,
            )
            print(
                f"[ENCORE-NC][MEMIT] L{layer} lambda_n={norm_lambda:g} "
                f"prior_key_gram={int(prior_active)} "
                f"cumulative_keys={cumulative_keys}"
            )
            append_official_baseline_record(
                getattr(hparams, "official_baseline_log_path", None),
                {
                    "method": "MEMIT",
                    "baseline": "ENCORE",
                    "stage": "sequential_closed_form",
                    "layer": int(layer),
                    "norm_lambda": norm_lambda,
                    "prior_key_gram_active": prior_active,
                    "cumulative_key_count": cumulative_keys,
                },
            )
        if nse_enabled:
            system = system + get_nse_gram(
                hparams,
                layer=int(layer),
                dimension=int(system.shape[0]),
                device=system.device,
                dtype=system.dtype,
            )
            selected, selection_stats = nse_select_neurons(
                layer_ks,
                float(getattr(hparams, "nse_neuron_threshold", 1.0)),
            )
            adj_k = nse_restricted_solve(
                system,
                layer_ks.to(cov.device),
                selected,
            )
            print(
                f"[NSE][MEMIT] L{layer} neurons="
                f"{int(selection_stats['selected_neurons'])}/"
                f"{int(selection_stats['total_neurons'])} "
                f"activation={selection_stats['activation_fraction']:.4f}"
            )
            append_official_baseline_record(
                getattr(hparams, "official_baseline_log_path", None),
                {
                    "method": "MEMIT",
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
            adj_k = torch.linalg.solve(
                system,
                layer_ks.to(cov.device),
            )
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = resid @ adj_k.T.to(resid.device)

        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        realized_update = (
            upd_matrix.to(device=layer_ks.device, dtype=layer_ks.dtype)
            @ layer_ks
        )

        if key_gaussian_noise_enabled:
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
                    **_key_gaussian_noise_diagnostics(
                        clean_gaussian_keys,
                        noisy_gaussian_keys,
                        gaussian_noise_seeds,
                    ),
                    **_key_gaussian_noise_realization_diagnostics(
                        upd_matrix,
                        clean_gaussian_keys,
                        noisy_gaussian_keys,
                        resid.T,
                    ),
                },
            )

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))
        save_outer_layer_artifact(
            hparams,
            method="MEMIT",
            requests=requests,
            layer=int(layer),
            keys=layer_ks,
            target_error=targets,
            distributed_residual=resid,
            realized_update=realized_update,
            update_frobenius=float(upd_matrix.float().norm().item()),
        )

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float().to(
                weights[weight_name].device)
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                resid.detach().cpu(),
            )

        # Clear GPU memory
        cov.cpu()
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
            "Update matrix computed by MEMIT does not match original weight shape. "
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
