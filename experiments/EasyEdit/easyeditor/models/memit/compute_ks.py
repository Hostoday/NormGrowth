from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compute_z import get_module_input_output_at_words
from .memit_hparams import MEMITHyperParams
from ...util.key_gaussian_noise import (
    add_relative_gaussian_noise_to_keys,
    canonical_prompt_template,
)


def compute_ks_gaussian_noise(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    layer: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Return one canonical ``k + epsilon`` key per MEMIT fact.

    Generated-prefix keys are not forwarded and no context average is taken.
    The returned noisy and clean tensors both have shape
    ``[num_facts, key_dim]``.
    """

    clean_keys = get_module_input_output_at_words(
        model,
        tok,
        layer,
        context_templates=[
            canonical_prompt_template(request) for request in requests
        ],
        words=[str(request["subject"]) for request in requests],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )[0]
    if clean_keys.ndim != 2 or clean_keys.shape[0] != len(requests):
        raise RuntimeError(
            "MEMIT canonical key extraction returned an unexpected shape: "
            f"got {tuple(clean_keys.shape)}, expected "
            f"[{len(requests)}, key_dim]"
        )
    noisy_keys, seeds = add_relative_gaussian_noise_to_keys(
        clean_keys,
        requests,
        relative_std=float(hparams.key_gaussian_noise_relative_std),
        base_seed=int(hparams.key_gaussian_noise_seed),
        layer=int(layer),
    )
    return noisy_keys, clean_keys.to(dtype=noisy_keys.dtype), seeds


def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
):
    layer_ks = get_module_input_output_at_words(
        model,
        tok,
        layer,
        context_templates=[
            context.format(request["prompt"])
            for request in requests
            for context_type in context_templates
            for context in context_type
        ],
        words=[
            request["subject"]
            for request in requests
            for context_type in context_templates
            for _ in context_type
        ],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )[0]

    context_type_lens = [0] + [len(context_type) for context_type in context_templates]
    context_len = sum(context_type_lens)
    context_type_csum = np.cumsum(context_type_lens).tolist()

    ans = []
    for i in range(0, layer_ks.size(0), context_len):
        tmp = []
        for j in range(len(context_type_csum) - 1):
            start, end = context_type_csum[j], context_type_csum[j + 1]
            tmp.append(layer_ks[i + start : i + end].mean(0))
        ans.append(torch.stack(tmp, 0).mean(0))
    return torch.stack(ans, dim=0)
