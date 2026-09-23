from typing import Dict, List, Tuple

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
    record_inner_gain_step,
    save_latent_artifact,
)
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

from .memit_hparams import MEMITHyperParams


def _prepare_oedit_regularizer(model, hparams, *, layer=None):
    """Resolve final-writer O-Edit state, leaving analysis observers enabled."""
    if not bool(getattr(hparams, "oedit_enabled", False)):
        return None
    if int(getattr(hparams, "batch_size", 1)) != 1:
        raise ValueError("O-Edit currently supports sequential batch_size=1 only")
    if not hparams.layers or hparams.layers[-1] != max(hparams.layers):
        raise ValueError("O-Edit requires an ordered final edited layer")
    final_layer = int(hparams.layers[-1])
    if layer is not None and int(layer) != final_layer:
        raise ValueError("O-Edit penalties are supported only at the final edited layer")
    conflicts = [name for name in (
        "residual_gain_regularization", "residual_gain_select_best",
        "sadr_regularization", "nse_enabled", "encore_enabled", "sphere_enabled",
        "nas_enabled", "key_gaussian_noise_enabled", "endogenous_pivot_enabled",
        "early_attention_preservation_enabled", "o0_axis_preservation_enabled",
        "context_multikey_enabled", "tangent_layer_allocation_enabled",
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


def _get_lm_head_weight(model: AutoModelForCausalLM, hparams: MEMITHyperParams) -> torch.Tensor:
    lm_head = nethook.get_module(model, hparams.lm_head_module)
    input_embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    output_embeddings = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None

    if (
        input_embeddings is not None
        and output_embeddings is not None
        and lm_head is input_embeddings
        and output_embeddings is not input_embeddings
    ):
        model_name = getattr(model.config, "_name_or_path", type(model).__name__)
        raise ValueError(
            "MEMIT hparams lm_head_module points to the input embedding module, "
            f"but {model_name} uses untied output embeddings. "
            "Set lm_head_module to the model's output head, e.g. 'lm_head'."
        )

    return lm_head.weight


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    oedit_regularizer = _prepare_oedit_regularizer(model, hparams, layer=layer)

    # Get model parameters
    lm_w, ln_f = (
        _get_lm_head_weight(model, hparams).T,
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
    # Keep the original prompt endpoint: prompt_last RGR must not point into
    # the target prefix appended for the rewrite objective.
    rewriting_original_prompts = [
        context.format(request["prompt"])
        for context_types in context_templates
        for context in context_types
    ]
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
        rewrite_batch_indices=list(range(len(rewriting_prompts))),
        subject_last_indices=lookup_idxs[: len(rewriting_prompts)],
        prompt_last_indices=prompt_last_idxs,
        reduction_device=rewrite_device,
    )
    if rgr.optimization_enabled:
        print(
            "[RGR][MEMIT] "
            f"layers={rgr.layers} scope={rgr.token_scope} "
            f"lambda={rgr.lambda_} position_lambdas={rgr.position_lambdas} "
            f"margin={rgr.margin} type={rgr.loss_type} "
            f"efficacy_gate={rgr.efficacy_threshold}"
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
            "[SADR][MEMIT] "
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

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        def _unwrap_output(output):
            if isinstance(output, torch.Tensor):
                return output, None
            if isinstance(output, (list, tuple)):
                if len(output) == 0:
                    raise ValueError("Layer output container is empty.")
                return output[0], output
            raise TypeError(
                f"Unsupported layer output type {type(output)} encountered in MEMIT."
            )

        def _rewrap_output(updated, original):
            if original is None:
                return updated
            if isinstance(original, list):
                new_out = list(original)
            elif isinstance(original, tuple):
                new_out = list(original)
            else:
                raise TypeError(
                    f"Unsupported layer output container {type(original)} in MEMIT."
                )
            new_out[0] = updated
            return type(original)(new_out) if isinstance(original, tuple) else new_out

        if cur_layer == hparams.layer_module_tmp.format(layer):
            layer_output, original_container = _unwrap_output(cur_out)

            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = layer_output[0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):

                if len(lookup_idxs)!=layer_output.shape[0]:
                    layer_output[idx, i, :] += delta.to(layer_output.device)
                else:
                    layer_output[i, idx, :] += delta.to(layer_output.device)

            return _rewrap_output(layer_output, original_container)

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)
    best_rgr_delta = None
    best_rgr_key = None
    best_efficacy_delta = None
    best_efficacy = float("-inf")
    encore_enabled = bool(getattr(hparams, "encore_enabled", False))
    mpes_state = MPESState()

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()
        rgr_loss = None
        rgr_diagnostics = {}
        sadr_loss = None
        sadr_diagnostics = {}

        # Forward propagation
        with nethook.TraceDict(
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

        loss_layer_out = tr[hparams.layer_module_tmp.format(loss_layer)].output
        if isinstance(loss_layer_out, (list, tuple)):
            output = loss_layer_out[0]
        else:
            output = loss_layer_out

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
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta.to(target_init.device)) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss.to(nll_loss.device) + weight_decay.to(nll_loss.device)
        efficacy_score = float(torch.exp(-nll_loss_each.detach()).mean().item())
        rgr_gate_active = bool(
            rgr.optimization_enabled
            and rgr_loss is not None
            and efficacy_score >= rgr.efficacy_threshold
        )
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
                "[RGR][MEMIT] "
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
                    "method": "MEMIT",
                    "case_id": request.get("case_id"),
                    "write_layer": layer,
                    "inner_step": it,
                    "efficacy_score": efficacy_score,
                    "efficacy_gate_active": rgr_gate_active,
                    "base_edit_loss": float(loss.detach().item()),
                    "optimization_loss": float(optimization_loss.detach().item()),
                    **rgr_diagnostics,
                },
            )
        if rgr.enabled:
            record_inner_gain_step(
                hparams,
                method="MEMIT",
                request=request,
                write_layer=layer,
                inner_step=it,
                efficacy_score=efficacy_score,
                base_edit_loss=float(loss.detach().item()),
                optimization_loss=float(optimization_loss.detach().item()),
                diagnostics=rgr_diagnostics,
            )
        if sadr_enabled:
            print(
                "[SADR][MEMIT] "
                f"step={it} efficacy={efficacy_score:.6f} "
                f"gate={int(sadr_gate_active)} "
                f"selected_heads={int(sadr_diagnostics.get('sadr_selected_heads', 0))} "
                f"raw={sadr_diagnostics.get('sadr_raw_loss', 0.0):.6e} "
                f"weighted={sadr_diagnostics.get('weighted_sadr_loss', 0.0):.6e}"
            )
            append_official_baseline_record(
                getattr(hparams, "sadr_log_path", None),
                {
                    "method": "MEMIT",
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
                "[ENCORE-MPES][MEMIT] "
                f"step={it} top1={mpes_status['num_top1']}/"
                f"{mpes_status['num_targets']} "
                f"qualifying={mpes_status['qualifying_steps']}/"
                f"{getattr(hparams, 'encore_mpes_top1_steps', 2)} "
                f"stop={int(mpes_status['should_stop'])}"
            )
            append_official_baseline_record(
                getattr(hparams, "official_baseline_log_path", None),
                {
                    "method": "MEMIT",
                    "baseline": "ENCORE",
                    "stage": "mpes",
                    "case_id": request.get("case_id"),
                    "inner_step": it,
                    **mpes_status,
                },
            )
            if mpes_status["should_stop"]:
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
                    "method": "MEMIT", "stage": "oedit_inner",
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
        if stop_loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        optimization_loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.to(max_norm.device).norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta.to(max_norm.device) * max_norm / delta.to(max_norm.device).norm()

    if rgr.select_best:
        selected_delta = (
            best_rgr_delta
            if best_rgr_delta is not None
            else best_efficacy_delta
        )
        if selected_delta is not None:
            with torch.no_grad():
                delta.copy_(selected_delta.to(delta))
    nas_active = bool(
        getattr(hparams, "nas_enabled", False)
        or getattr(hparams, "nas_collect_stats", False)
    )
    if nas_active:
        # target_init is the edited block output at the fact lookup token.
        # Recover the MLP output v0 from the same layer/token so NAS can scale
        # only v*=v0+delta and preserve the pre-MLP residual component.
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
            method="MEMIT",
            request=request,
            layer=layer,
        )
        target = nas_result.target
        print(
            "[NAS][MEMIT] "
            f"case={request.get('case_id')} L{layer} "
            f"anchor={nas_result.anchor_norm} "
            f"v*={nas_result.vstar_norm_before:.6f}"
            f"->{nas_result.vstar_norm_after:.6f} "
            f"scale={nas_result.scale_factor:.6f} "
            f"safeguard={int(nas_result.safeguard_triggered)}"
        )
    else:
        target = target_init + delta.to(target_init.device)
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )
    save_latent_artifact(
        hparams,
        method="MEMIT",
        request=request,
        write_layer=layer,
        target_init=target_init,
        optimizer_delta=delta,
        target=target,
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
    track=None,
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
        if track == 'out' or track == 'in':
            return repr_tools.get_reprs_at_word_tokens(
                track=track, subtoken=subtoken, **context_info, **word_repr_args
            )
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
        if track == 'out' or track == 'in':
            return repr_tools.get_reprs_at_word_tokens(
                track=track, subtoken=subtoken, **context_info, **word_repr_args
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
