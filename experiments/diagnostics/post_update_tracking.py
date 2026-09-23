"""Paired pre/post-update boundary states and continuous behavior scores.

This module is deliberately editor agnostic.  A trajectory runner captures one
request immediately before and after ``apply_algo`` and writes a compact paired
artifact.  Unlike the editor-side ``outer/delta_w_k`` diagnostics, these values
come from an actual re-forward through the updated model.

Node notation follows the residual-block convention

``H_l``
    residual stream entering decoder block ``l``;
``A_l`` / ``M_l``
    attention/MLP residual writes produced by block ``l``;
``F_l``
    the complete block write, ``H_{l+1} - H_l``.

The default nodes ``H8,M8,H9,A9,H10`` therefore cover the final edited MLP,
the first unedited attention block, and the propagated downstream state for
the usual L4--L8 editing setup.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from diagnostics.analyze_residual_spectrum import (
    Probe,
    _layer_hidden_output,
    build_probe,
    layer_attention,
    model_input_device,
    model_layers,
    probe_inputs,
)


NODE_PATTERN = re.compile(r"^([HAFM])(\d+)$")


def parse_nodes(text: str | Sequence[str]) -> Tuple[str, ...]:
    raw = text.split(",") if isinstance(text, str) else text
    nodes = tuple(str(value).strip().upper() for value in raw if str(value).strip())
    if not nodes:
        raise ValueError("at least one post-update node is required")
    invalid = [node for node in nodes if NODE_PATTERN.fullmatch(node) is None]
    if invalid:
        raise ValueError(f"invalid post-update node(s): {invalid}")
    if len(nodes) != len(set(nodes)):
        raise ValueError("post-update nodes contain duplicates")
    return nodes


def required_layers(nodes: Sequence[str]) -> Tuple[int, ...]:
    return tuple(sorted({int(NODE_PATTERN.fullmatch(node).group(2)) for node in nodes}))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **dict(arrays))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_case_id(request: Mapping[str, Any]) -> str:
    raw = str(request.get("case_id", "unknown"))
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return safe or "unknown"


def artifact_stem(edit_index: int, request: Mapping[str, Any]) -> str:
    return f"edit_{int(edit_index):06d}_case_{_safe_case_id(request)}"


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy().astype(np.float32, copy=False)


def capture_boundary_nodes(
    model: Any,
    tokenizer: Any,
    request: Mapping[str, Any],
    *,
    nodes: Sequence[str],
    max_length: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Capture selected nodes for one request without changing model state."""

    nodes = parse_nodes(nodes)
    positions = ("subject_last", "prompt_last")
    probe: Probe = build_probe(
        request,
        tokenizer,
        positions,
        max_length,
        probe_source="rewrite",
        source_index=int(request.get("case_id", 0)),
    )
    layers = required_layers(nodes)
    decoder_layers = model_layers(model)
    arrays: Dict[str, np.ndarray] = {}
    requested = set(nodes)
    handles = []

    def capture_selected(node: str, value: torch.Tensor) -> None:
        if node not in requested:
            return
        for position in positions:
            token_index = int(probe.positions[position])
            arrays[f"{position}__{node}"] = _to_numpy(value[0, token_index])

    for layer_id in layers:
        layer = decoder_layers[layer_id]

        def capture_input(
            _module: Any,
            args: Tuple[Any, ...],
            kwargs: Mapping[str, Any],
            *,
            key: int = layer_id,
        ) -> None:
            hidden = args[0] if args else kwargs.get("hidden_states")
            if not torch.is_tensor(hidden):
                raise TypeError(f"Could not capture H{key} from decoder input")
            capture_selected(f"H{key}", hidden)

        def capture_total(
            _module: Any,
            args: Tuple[Any, ...],
            output: Any,
            *,
            key: int = layer_id,
        ) -> None:
            if f"F{key}" not in requested:
                return
            if not args or not torch.is_tensor(args[0]):
                raise TypeError(f"Could not capture F{key} decoder input")
            hidden_out = _layer_hidden_output(output)
            capture_selected(f"F{key}", hidden_out - args[0].to(hidden_out.device))

        def capture_attention(
            _module: Any,
            _args: Tuple[Any, ...],
            output: Any,
            *,
            key: int = layer_id,
        ) -> None:
            capture_selected(f"A{key}", _layer_hidden_output(output))

        def capture_mlp(
            _module: Any,
            _args: Tuple[Any, ...],
            output: Any,
            *,
            key: int = layer_id,
        ) -> None:
            capture_selected(f"M{key}", _layer_hidden_output(output))

        handles.append(layer.register_forward_pre_hook(capture_input, with_kwargs=True))
        handles.append(layer.register_forward_hook(capture_total))
        if f"A{layer_id}" in requested:
            handles.append(layer_attention(layer).register_forward_hook(capture_attention))
        if f"M{layer_id}" in requested:
            handles.append(layer.mlp.register_forward_hook(capture_mlp))
    try:
        with torch.no_grad():
            model(
                **probe_inputs(probe, model_input_device(model)),
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    expected = {
        f"{position}__{node}" for position in positions for node in nodes
    }
    missing = sorted(expected - set(arrays))
    if missing:
        raise RuntimeError(f"Post-update hooks did not capture nodes: {missing}")
    metadata = {
        "case_id": request.get("case_id"),
        "prompt": probe.prompt,
        "subject": probe.subject,
        "token_count": len(probe.input_ids),
        "token_positions": {key: int(value) for key, value in probe.positions.items()},
        "nodes": list(nodes),
        "layers": list(layers),
    }
    return arrays, metadata


def _target_text(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("str", "")
    text = "" if value is None else str(value).strip()
    if text in {"<|endoftext|>", "<|end_of_text|>"}:
        return ""
    return text


def _request_behavior_items(
    request: Mapping[str, Any], tokenizer: Any
) -> Sequence[Tuple[str, str, str]]:
    prompt = str(request.get("prompt", ""))
    rephrase = str(
        request.get("rephrase_prompt")
        or request.get("rephrase")
        or prompt
    )
    target_new = _target_text(request.get("target_new"))
    target_old = _target_text(request.get("ground_truth"))
    eos_token = getattr(tokenizer, "eos_token", None)
    if (
        target_old
        and target_new
        and eos_token
        and target_new.endswith(str(eos_token))
        and not target_old.endswith(str(eos_token))
    ):
        target_old = target_old + str(eos_token)
    items = []
    if prompt and target_new:
        items.append(("rewrite_new", prompt, target_new))
    if prompt and target_old:
        items.append(("rewrite_old", prompt, target_old))
    if rephrase and target_new:
        items.append(("rephrase_new", rephrase, target_new))
    if rephrase and target_old:
        items.append(("rephrase_old", rephrase, target_old))

    locality = request.get("locality")
    if isinstance(locality, Mapping):
        for group_name, group in locality.items():
            if not isinstance(group, Mapping):
                continue
            prompts = group.get("prompt", [])
            targets = group.get("ground_truth", [])
            if not isinstance(prompts, list):
                prompts = [prompts]
            if not isinstance(targets, list):
                targets = [targets]
            for index, (local_prompt, local_target) in enumerate(zip(prompts, targets)):
                target = _target_text(local_target)
                if local_prompt and target:
                    items.append(
                        (f"locality_{group_name}_{index}", str(local_prompt), target)
                    )
    return items


def _forward_target_logprobs(
    model: Any,
    tokenizer: Any,
    items: Sequence[Tuple[str, str, str]],
    *,
    max_length: int,
) -> Dict[str, Dict[str, float | int]]:
    """Teacher-forced target likelihoods, matching EasyEdit's ``prompt + ' '``."""

    if not items:
        return {}
    prompts = [prompt for _, prompt, _ in items]
    targets = [target for _, _, target in items]
    full_texts = [f"{prompt} {target}" for prompt, target in zip(prompts, targets)]
    before_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        # Do not truncate away the target merely because the configured probe
        # max length is shorter than this particular prompt-target pair.
        encoded_lengths = [len(tokenizer.encode(text)) for text in full_texts]
        effective_max = max(int(max_length), max(encoded_lengths) + 1)
        full = tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=effective_max,
            return_tensors="pt",
        )
        prompt_tokens = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=effective_max,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = before_padding

    device = model_input_device(model)
    full = {key: value.to(device) for key, value in full.items()}
    with torch.no_grad():
        output = model(**full)
        logits = output if isinstance(output, torch.Tensor) else output.logits
        token_logprobs = torch.log_softmax(logits.float(), dim=-1)
    nonpad_prompt = prompt_tokens["attention_mask"].sum(dim=1).tolist()
    left_padding = (full["attention_mask"] == 0).sum(dim=1).tolist()
    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    results: Dict[str, Dict[str, float | int]] = {}
    for row, (name, _, _) in enumerate(items):
        prompt_start = int(left_padding[row] + nonpad_prompt[row])
        valid_end = int(attention_mask[row].sum().item() + left_padding[row])
        # Label token j is predicted by logits at j-1.
        label_positions = torch.arange(prompt_start, valid_end, device=device)
        if label_positions.numel() == 0:
            continue
        labels = input_ids[row, label_positions]
        selected = token_logprobs[row, label_positions - 1, labels]
        mean_logprob = float(selected.mean().item())
        results[name] = {
            "mean_logprob": mean_logprob,
            "nll": -mean_logprob,
            "geomean_probability": float(math.exp(mean_logprob)),
            "min_token_logprob": float(selected.min().item()),
            "token_count": int(selected.numel()),
        }
    return results


def score_request_behavior(
    model: Any,
    tokenizer: Any,
    request: Mapping[str, Any],
    *,
    max_length: int,
) -> Dict[str, Any]:
    results: Dict[str, Any] = dict(
        _forward_target_logprobs(
            model,
            tokenizer,
            _request_behavior_items(request, tokenizer),
            max_length=max_length,
        )
    )
    for prefix in ("rewrite", "rephrase"):
        new = results.get(f"{prefix}_new")
        old = results.get(f"{prefix}_old")
        if new is not None and old is not None:
            results[f"{prefix}_new_minus_old_margin"] = float(
                new["mean_logprob"] - old["mean_logprob"]
            )
    locality_values = [
        value["mean_logprob"]
        for key, value in results.items()
        if key.startswith("locality_") and isinstance(value, Mapping)
    ]
    if locality_values:
        results["locality_mean_logprob"] = float(np.mean(locality_values))
        results["locality_nll"] = float(-np.mean(locality_values))
        results["locality_geomean_probability"] = float(
            math.exp(float(np.mean(locality_values)))
        )
    return results


def _vector_pair_metrics(before: np.ndarray, after: np.ndarray) -> Dict[str, float]:
    before64 = np.asarray(before, dtype=np.float64)
    after64 = np.asarray(after, dtype=np.float64)
    delta = after64 - before64
    before_l2 = float(np.linalg.norm(before64))
    after_l2 = float(np.linalg.norm(after64))
    delta_l2 = float(np.linalg.norm(delta))
    denom = before_l2 * after_l2
    return {
        "pre_l2": before_l2,
        "post_l2": after_l2,
        "pre_rms": float(np.sqrt(np.mean(before64 * before64))),
        "post_rms": float(np.sqrt(np.mean(after64 * after64))),
        "delta_l2": delta_l2,
        "delta_rms": float(np.sqrt(np.mean(delta * delta))),
        "post_over_pre_l2": after_l2 / (before_l2 + 1e-12),
        "delta_over_pre_l2": delta_l2 / (before_l2 + 1e-12),
        "pre_post_cosine": float(np.dot(before64, after64) / (denom + 1e-12)),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    return float(
        np.dot(left64, right64)
        / (np.linalg.norm(left64) * np.linalg.norm(right64) + 1e-12)
    )


def _flatten_behavior_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> Dict[str, Any]:
    output: Dict[str, Any] = {"pre": dict(before), "post": dict(after)}
    deltas: Dict[str, float] = {}
    for key in sorted(set(before) & set(after)):
        left, right = before[key], after[key]
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for metric in sorted(set(left) & set(right)):
                if isinstance(left[metric], (int, float)) and isinstance(
                    right[metric], (int, float)
                ):
                    deltas[f"{key}.{metric}"] = float(right[metric] - left[metric])
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            deltas[key] = float(right - left)
    output["delta"] = deltas
    return output


def save_paired_post_update_artifact(
    output_root: Path,
    *,
    edit_index: int,
    request: Mapping[str, Any],
    before_nodes: Mapping[str, np.ndarray],
    after_nodes: Mapping[str, np.ndarray],
    node_metadata: Mapping[str, Any],
    before_behavior: Mapping[str, Any],
    after_behavior: Mapping[str, Any],
    array_dtype: str = "float16",
) -> Tuple[Path, Path]:
    """Write one crash-safe artifact after a successful editor update."""

    if array_dtype not in {"float16", "float32"}:
        raise ValueError("post-update array_dtype must be float16 or float32")
    root = Path(output_root) / "post_update"
    stem = artifact_stem(edit_index, request)
    npz_path = root / f"{stem}.npz"
    json_path = root / f"{stem}.json"
    common_keys = sorted(set(before_nodes) & set(after_nodes))
    arrays: Dict[str, np.ndarray] = {}
    vector_metrics: Dict[str, Any] = {}
    deltas: Dict[str, np.ndarray] = {}
    for key in common_keys:
        before = np.asarray(before_nodes[key], dtype=np.float32)
        after = np.asarray(after_nodes[key], dtype=np.float32)
        arrays[f"pre__{key}"] = before.astype(array_dtype)
        arrays[f"post__{key}"] = after.astype(array_dtype)
        deltas[key] = after - before
        vector_metrics[key] = _vector_pair_metrics(before, after)

    # Follow each measured MLP write through the next block. This preserves
    # the Llama M8 -> H9 -> A9 -> H10 bridge and supports GPT-2's M17 -> H18
    # -> A18 -> H19 boundary without changing the vector definitions.
    bridge_metrics: Dict[str, float] = {}
    for position in ("subject_last", "prompt_last"):
        mlp_layers = sorted({
            int(key.split("__", 1)[1][1:])
            for key in deltas
            if key.startswith(f"{position}__M")
        })
        chain = [
            pair
            for layer in mlp_layers
            for pair in (
                (f"M{layer}", f"H{layer + 1}"),
                (f"H{layer + 1}", f"A{layer + 1}"),
                (f"A{layer + 1}", f"H{layer + 2}"),
                (f"M{layer}", f"H{layer + 2}"),
            )
        ]
        for left_node, right_node in chain:
            left_key = f"{position}__{left_node}"
            right_key = f"{position}__{right_node}"
            if left_key in deltas and right_key in deltas:
                bridge_metrics[f"{position}__delta_{left_node}_delta_{right_node}_cosine"] = _cosine(
                    deltas[left_key], deltas[right_key]
                )

    _atomic_npz(npz_path, arrays)
    payload = {
        "schema_version": 1,
        "measurement": "actual_paired_pre_post_reforward",
        "edit_index": int(edit_index),
        "case_id": request.get("case_id"),
        "node_metadata": dict(node_metadata),
        "vector_metrics": vector_metrics,
        "bridge_metrics": bridge_metrics,
        "behavior": _flatten_behavior_delta(before_behavior, after_behavior),
        "arrays_path": npz_path.name,
        "array_dtype": array_dtype,
        "delta_arrays": "derive as post - pre",
    }
    _atomic_json(json_path, payload)
    return npz_path, json_path
