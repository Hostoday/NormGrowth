from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compute_z import get_module_input_output_at_words
from .AlphaEdit_hparams import AlphaEditHyperParams
from ...util.key_gaussian_noise import (
    add_relative_gaussian_noise_to_keys,
    canonical_prompt_template as _canonical_prompt_template,
    gaussian_key_noise_seed,
)


def compute_ks_gaussian_noise(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    layer: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Extract one canonical key per fact and perturb it with Gaussian noise.

    No generated-prefix prompt is forwarded and no context mean is taken.
    Returns ``(noisy_keys, clean_keys, derived_seeds)`` with both key tensors
    shaped ``[num_facts, key_dim]``.
    """

    clean_keys = get_module_input_output_at_words(
        model,
        tok,
        layer,
        context_templates=[
            _canonical_prompt_template(request) for request in requests
        ],
        words=[str(request["subject"]) for request in requests],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )[0]
    if clean_keys.ndim != 2 or clean_keys.shape[0] != len(requests):
        raise RuntimeError(
            "AlphaEdit canonical key extraction returned an unexpected "
            f"shape: got {tuple(clean_keys.shape)}, expected "
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


def context_template_weights(
    context_templates: List[List[str]],
    *,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Return the raw-context weights implicit in AlphaEdit's legacy mean.

    AlphaEdit first averages the keys inside each context-template group and
    then averages those group means.  Therefore a context in group ``g`` has
    weight ``1 / (num_groups * len(group_g))``.  For the current Llama setup,
    ``[["{}"], [five generated prefixes]]``, this is exactly
    ``[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]``.
    """

    if not context_templates:
        raise ValueError("context_templates must contain at least one group")
    if any(not group for group in context_templates):
        raise ValueError("Every context-template group must be non-empty")
    num_groups = float(len(context_templates))
    weights = [
        1.0 / (num_groups * float(len(group)))
        for group in context_templates
        for _ in group
    ]
    return torch.tensor(weights, device=device, dtype=dtype or torch.float32)


def compute_ks_context_components(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: Dict,
    hparams: AlphaEditHyperParams,
    layer: int,
    context_templates: List[List[str]],
):
    """Return every already-computed context key instead of discarding them.

    The returned key tensor is shaped ``[num_facts, num_contexts, key_dim]``.
    No prompt is added: these are the same raw original/prefix contexts used
    by :func:`compute_ks`.  The second return value contains the hierarchical
    weights whose weighted mean equals the legacy AlphaEdit key.
    """

    weights = context_template_weights(context_templates)
    context_len = int(weights.numel())
    expected_group_sizes = getattr(
        hparams, "context_multikey_expected_group_sizes", None
    )
    actual_group_sizes = [len(group) for group in context_templates]
    if (
        expected_group_sizes is not None
        and actual_group_sizes != [int(size) for size in expected_group_sizes]
    ):
        raise RuntimeError(
            "AlphaEdit context multi-key layout changed: "
            f"got group sizes {actual_group_sizes}, expected "
            f"{list(expected_group_sizes)}. Refusing to subsample or silently "
            "drop an existing context."
        )
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
    expected = len(requests) * context_len
    if layer_ks.ndim != 2 or layer_ks.size(0) != expected:
        raise RuntimeError(
            "AlphaEdit context-key extraction returned an unexpected shape: "
            f"got {tuple(layer_ks.shape)}, expected [{expected}, key_dim]"
        )
    return layer_ks.reshape(len(requests), context_len, layer_ks.size(-1)), weights


def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: Dict,
    hparams: AlphaEditHyperParams,
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
