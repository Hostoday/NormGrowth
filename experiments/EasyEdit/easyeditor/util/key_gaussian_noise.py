"""Deterministic, scale-safe Gaussian perturbations for editing keys."""

import hashlib
import json
import math
from typing import Dict, List, Tuple

import torch


def canonical_prompt_template(request: Dict) -> str:
    """Return the unprefixed prompt template used for one canonical key."""

    prompt = str(request["prompt"])
    subject = str(request["subject"])
    if "{}" not in prompt:
        if subject not in prompt:
            raise ValueError(
                "Cannot construct a canonical editing key: the prompt has "
                "no '{}' placeholder and does not contain its subject."
            )
        prompt = prompt.replace(subject, "{}", 1)
    return prompt


def gaussian_key_noise_seed(
    request: Dict,
    *,
    base_seed: int,
    layer: int,
) -> int:
    """Derive a stable per-request/per-layer key-noise seed.

    The derivation is independent of batch grouping and PyTorch's global RNG
    state.  Recomputing a key after committing an edit therefore reuses the
    same noise direction around the updated activation.
    """

    payload = {
        "base_seed": int(base_seed),
        "layer": int(layer),
        "case_id": request.get("case_id"),
        "edit_index": request.get("edit_index"),
        "prompt_template": canonical_prompt_template(request),
        "subject": str(request["subject"]),
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def add_relative_gaussian_noise_to_keys(
    clean_keys: torch.Tensor,
    requests: List[Dict],
    *,
    relative_std: float,
    base_seed: int,
    layer: int,
) -> Tuple[torch.Tensor, List[int]]:
    """Return one ``k + epsilon`` key per canonical prompt, without averaging.

    For every key ``k`` of dimension ``d`` this samples

    ``epsilon ~ N(0, (relative_std * ||k||_2 / sqrt(d))^2 I)``.

    Low-precision activations are promoted to float32 before perturbation.
    Independent CPU generators make the result insensitive to device mapping,
    global RNG consumption, request batching, and diagnostic forwards.
    """

    relative_std = float(relative_std)
    if not math.isfinite(relative_std) or relative_std <= 0.0:
        raise ValueError(
            "key_gaussian_noise_relative_std must be finite and > 0; "
            f"got {relative_std!r}"
        )
    if clean_keys.ndim != 2:
        raise ValueError(
            "clean_keys must have shape [num_facts, key_dim]; "
            f"got {tuple(clean_keys.shape)}"
        )
    if clean_keys.shape[0] != len(requests):
        raise ValueError(
            "The number of clean keys and requests must match; "
            f"got {clean_keys.shape[0]} and {len(requests)}"
        )
    if clean_keys.shape[1] <= 0:
        raise ValueError("The key dimension must be positive")

    work_dtype = (
        clean_keys.dtype
        if clean_keys.dtype in (torch.float32, torch.float64)
        else torch.float32
    )
    clean_work = clean_keys.to(dtype=work_dtype)
    noisy_rows = []
    seeds = []
    for clean_key, request in zip(clean_work, requests):
        noise_seed = gaussian_key_noise_seed(
            request,
            base_seed=int(base_seed),
            layer=int(layer),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(noise_seed)
        unit_noise = torch.randn(
            clean_key.numel(),
            generator=generator,
            device="cpu",
            dtype=work_dtype,
        ).to(device=clean_key.device)
        coordinate_rms = clean_key.square().mean().sqrt()
        noisy_rows.append(
            clean_key + unit_noise * (relative_std * coordinate_rms)
        )
        seeds.append(noise_seed)

    return torch.stack(noisy_rows, dim=0), seeds
