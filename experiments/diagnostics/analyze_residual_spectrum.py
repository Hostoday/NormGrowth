#!/usr/bin/env python3
"""Compare residual-stream geometry and local dynamics across edited LLMs.

Two deliberately different spectral objects are reported:

1. The covariance spectrum of residual vectors collected from a fixed prompt
   set.  These eigenvalues are real and non-negative.
2. Complex eigenvalues of a token-local residual-block Jacobian projected into
   a shared PCA subspace, ``J_k = U.T @ (dh_{l+1}/dh_l) @ U``.

The PCA bases are fitted only on the first state (normally the base model) and
then reused for every edited state.  This keeps the projected coordinates
comparable across conditions.  Full 4096 x 4096 Llama Jacobians are never
materialized.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from EasyEdit.easyeditor.util.edited_layer_checkpoint import (
    FORMAT_NAME as EDITED_PARAMETER_FORMAT,
    load_edited_parameter_checkpoint,
)


EPS = 1e-12

# Use deliberately separated hues: the previous Base and AlphaEdit-s0 colors
# were both dark blue and became almost indistinguishable in dense overlays.
# Warm colors denote MEMIT, cool colors denote AlphaEdit, and Base stays
# achromatic.  Markers provide a second cue in metric plots.
BASE_BACKGROUND_COLOR = "#3A3A3A"
METHOD_COLORS = {
    "memit-s0": "#FF9D00",      # bright orange
    "memit-s1": "#D81B60",      # crimson
    "alphaedit-s0": "#00A6D6",  # cyan-blue
    "alphaedit-s1": "#6F2DBD",  # violet
    "alphaedit-n50-s0": "#00A6D6",
    "alphaedit-n100-s0": "#6F2DBD",
}
METHOD_MARKERS = {
    "memit-s0": "^",
    "memit-s1": "s",
    "alphaedit-s0": "D",
    "alphaedit-s1": "P",
    "alphaedit-n50-s0": "D",
    "alphaedit-n100-s0": "P",
}


@dataclass(frozen=True)
class StateSpec:
    label: str
    path: str


@dataclass(frozen=True)
class Probe:
    case_id: Any
    prompt: str
    subject: str
    input_ids: Tuple[int, ...]
    positions: Mapping[str, int]


def _strip_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("str", "")
    return "" if value is None else str(value).strip()


def _format_prompt(prompt: str, subject: str) -> str:
    if "{}" not in prompt:
        return prompt
    try:
        return prompt.format(subject)
    except (IndexError, KeyError, ValueError):
        return prompt.replace("{}", subject)


def _rewrite_prompt_subject(record: Mapping[str, Any]) -> Tuple[str, str]:
    rewrite = record.get("requested_rewrite")
    rewrite = rewrite if isinstance(rewrite, Mapping) else {}
    subject = _strip_text(record.get("subject") or rewrite.get("subject"))
    prompt = _strip_text(record.get("prompt") or record.get("src") or rewrite.get("prompt"))
    prompt = _format_prompt(prompt, subject)
    if not prompt:
        raise ValueError("record has no usable prompt")
    return prompt, subject


def _rephrase_prompt_subject(record: Mapping[str, Any]) -> Tuple[str, str]:
    rewrite = record.get("requested_rewrite")
    rewrite = rewrite if isinstance(rewrite, Mapping) else {}
    subject = _strip_text(record.get("subject") or rewrite.get("subject"))
    prompt: Any = record.get("rephrase_prompt") or record.get("rephrase")
    if not prompt:
        prompts = record.get("paraphrase_prompts")
        if isinstance(prompts, Sequence) and not isinstance(prompts, (str, bytes)):
            prompt = prompts[0] if prompts else ""
    prompt = _format_prompt(_strip_text(prompt), subject)
    if not prompt:
        raise ValueError("record has no usable rephrase prompt")
    return prompt, subject


def _locality_prompt(record: Mapping[str, Any]) -> str:
    locality = record.get("locality")
    locality = locality if isinstance(locality, Mapping) else {}
    neighborhood = locality.get("neighborhood")
    neighborhood = neighborhood if isinstance(neighborhood, Mapping) else {}
    prompt = (
        neighborhood.get("prompt")
        or record.get("locality_prompt")
        or record.get("loc")
    )
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""
    prompt = _strip_text(prompt)
    if not prompt:
        raise ValueError("record has no usable locality neighborhood prompt")
    return prompt


def _record_probe_text(record: Mapping[str, Any], probe_source: str) -> Tuple[str, str]:
    if probe_source == "rewrite":
        return _rewrite_prompt_subject(record)
    if probe_source == "rephrase":
        return _rephrase_prompt_subject(record)
    if probe_source == "locality":
        return _locality_prompt(record), ""
    raise ValueError(f"Unknown probe source: {probe_source}")


def _normalized_signature(prompt: str, subject: str) -> Tuple[str, str]:
    return " ".join(prompt.split()), " ".join(subject.split())


def load_exclusion_signatures(paths: Sequence[str]) -> set:
    signatures = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON list in exclusion file {path}")
        for record in records:
            if not isinstance(record, Mapping):
                continue
            try:
                prompt, subject = _rewrite_prompt_subject(record)
            except ValueError:
                continue
            signatures.add(_normalized_signature(prompt, subject))
    return signatures


def _subject_last_from_offsets(prompt: str, subject: str, offsets: Sequence[Sequence[int]]) -> int:
    if not subject:
        raise ValueError("subject_last requires a non-empty subject")
    char_start = prompt.rfind(subject)
    if char_start < 0:
        raise ValueError(f"subject {subject!r} is not present in prompt {prompt!r}")
    char_end = char_start + len(subject)
    hits = [
        index
        for index, pair in enumerate(offsets)
        if len(pair) == 2 and int(pair[1]) > char_start and int(pair[0]) < char_end
    ]
    if not hits:
        raise ValueError(f"tokenizer offsets do not overlap subject {subject!r}")
    return int(hits[-1])


def _fallback_subject_last(tokenizer: Any, prompt: str, subject: str) -> int:
    if not subject:
        raise ValueError("subject_last requires a non-empty subject")
    char_start = prompt.rfind(subject)
    if char_start < 0:
        raise ValueError(f"subject {subject!r} is not present in prompt {prompt!r}")
    through_subject = prompt[: char_start + len(subject)]
    token_ids = tokenizer(through_subject, add_special_tokens=True)["input_ids"]
    if not token_ids:
        raise ValueError(f"subject prefix produced no tokens for prompt {prompt!r}")
    return len(token_ids) - 1


def build_probe(
    record: Mapping[str, Any],
    tokenizer: Any,
    positions: Sequence[str],
    max_length: int,
    *,
    probe_source: str = "rewrite",
    source_index: int = 0,
) -> Probe:
    """Materialize one prompt/token-position probe from an in-memory record."""

    prompt, subject = _record_probe_text(record, probe_source)
    tokenization = dict(add_special_tokens=True, truncation=True, max_length=max_length)
    try:
        encoded = tokenizer(prompt, return_offsets_mapping=True, **tokenization)
    except NotImplementedError:
        # EasyEdit loads GPT2Tokenizer (the slow Python implementation), which
        # intentionally has no offset mapping. Keep its exact tokenization and
        # use the established prefix-based subject lookup below.
        encoded = tokenizer(prompt, **tokenization)
    token_ids = encoded["input_ids"]
    if not token_ids:
        raise ValueError("prompt produced no tokens")
    token_positions: Dict[str, int] = {}
    if "prompt_last" in positions:
        token_positions["prompt_last"] = len(token_ids) - 1
    if "subject_last" in positions:
        offsets = encoded.get("offset_mapping")
        if offsets is not None:
            token_positions["subject_last"] = _subject_last_from_offsets(
                prompt, subject, offsets
            )
        else:
            token_positions["subject_last"] = _fallback_subject_last(
                tokenizer, prompt, subject
            )
        if token_positions["subject_last"] >= len(token_ids):
            raise ValueError("subject was truncated out of the tokenized prompt")
    return Probe(
        case_id=record.get("case_id", source_index),
        prompt=prompt,
        subject=subject,
        input_ids=tuple(int(value) for value in token_ids),
        positions=token_positions,
    )


def load_probes(
    requests_path: str,
    tokenizer: Any,
    positions: Sequence[str],
    max_prompts: int,
    max_length: int,
    probe_source: str = "rewrite",
    exclude_requests_paths: Sequence[str] = (),
    selection: str = "prefix",
    seed: int = 42,
) -> List[Probe]:
    with open(requests_path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {requests_path}")

    excluded_signatures = load_exclusion_signatures(exclude_requests_paths)
    indexed_records = list(enumerate(records))
    if selection == "random":
        random.Random(seed).shuffle(indexed_records)
    elif selection != "prefix":
        raise ValueError(f"Unknown probe selection: {selection}")

    probes: List[Probe] = []
    failures: List[str] = []
    excluded_count = 0
    for source_index, record in indexed_records:
        if max_prompts > 0 and len(probes) >= max_prompts:
            break
        if not isinstance(record, Mapping):
            failures.append(f"record {source_index}: not a JSON object")
            continue
        try:
            prompt, subject = _record_probe_text(record, probe_source)
            if _normalized_signature(prompt, subject) in excluded_signatures:
                excluded_count += 1
                continue
            probes.append(
                build_probe(
                    record,
                    tokenizer,
                    positions,
                    max_length,
                    probe_source=probe_source,
                    source_index=source_index,
                )
            )
        except (TypeError, ValueError) as exc:
            failures.append(f"record {source_index}: {exc}")

    if not probes:
        detail = "; ".join(failures[:5])
        raise RuntimeError(f"No usable probes were loaded from {requests_path}. {detail}")
    if failures:
        print(f"[data] skipped {len(failures)} unusable records; first: {failures[0]}")
    print(
        f"[data] loaded {len(probes)} {probe_source} probes from {requests_path} "
        f"(selection={selection}, excluded={excluded_count})"
    )
    return probes


def parse_state_spec(text: str) -> StateSpec:
    if "=" not in text:
        raise argparse.ArgumentTypeError("state must be LABEL=MODEL_PATH")
    label, path = text.split("=", 1)
    label, path = label.strip(), path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("state must contain a non-empty label and path")
    return StateSpec(label=label, path=path)


def parse_int_list(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("layer list contains duplicates")
    return values


def parse_str_list(text: str) -> List[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def model_layers(model: Any) -> Sequence[torch.nn.Module]:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise TypeError(f"Unsupported model architecture: {type(model).__name__}; decoder layers were not found")


def model_input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def layer_attention(layer: torch.nn.Module) -> torch.nn.Module:
    """Return the attention residual-write module for Llama or GPT-2 blocks."""
    for name in ("self_attn", "attn"):
        module = getattr(layer, name, None)
        if isinstance(module, torch.nn.Module):
            return module
    raise TypeError(f"Unsupported decoder block attention: {type(layer).__name__}")


def probe_inputs(probe: Probe, device: torch.device) -> Dict[str, torch.Tensor]:
    ids = torch.tensor([probe.input_ids], dtype=torch.long, device=device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def compact_checkpoint_metadata(checkpoint_path: str) -> Dict[str, Any]:
    """Read the sidecar manifest for a compact edited-parameter checkpoint.

    The tensor payload is intentionally not opened here: trajectory deltas can
    be hundreds of MiB, while the adjacent JSON is sufficient to resolve the
    Base model before model loading begins.
    """

    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        return {}
    manifest_path = path.with_suffix(".json")
    if not manifest_path.is_file():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Compact-checkpoint manifest is not an object: {manifest_path}")
    if metadata.get("format") != EDITED_PARAMETER_FORMAT:
        raise ValueError(
            f"File state {path} has an incompatible manifest format: "
            f"{metadata.get('format')!r}"
        )
    return metadata


def resolve_model_load_spec(
    model_path: str,
    base_model_path: str | None = None,
) -> Tuple[str, str | None, Dict[str, Any]]:
    """Resolve a Hugging Face state or Base-plus-compact-delta state.

    A ``--state`` path that names a regular directory/repository is loaded as
    before.  A path that names a file is treated as a compact checkpoint.  Its
    Base can be supplied explicitly or inferred from the adjacent manifest.
    """

    expanded = Path(model_path).expanduser()
    if not expanded.is_file():
        return model_path, None, {}
    metadata = compact_checkpoint_metadata(str(expanded))
    inferred_base = metadata.get("base_model")
    source = base_model_path or (
        str(inferred_base).strip() if inferred_base is not None else ""
    )
    if not source:
        raise ValueError(
            f"Compact checkpoint {expanded} needs --base-model-path because "
            "its sidecar manifest has no base_model"
        )
    return source, str(expanded.resolve()), metadata


def load_model(
    model_path: str,
    torch_dtype: str,
    device_map: str,
    trust_remote_code: bool,
    attn_implementation: str,
    base_model_path: str | None = None,
) -> Any:
    source, delta_checkpoint, checkpoint_metadata = resolve_model_load_spec(
        model_path,
        base_model_path,
    )
    if torch_dtype == "auto":
        dtype: Any = "auto"
    else:
        dtype = getattr(torch, torch_dtype)
    kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }
    if torch.cuda.is_available() and device_map != "none":
        kwargs["device_map"] = device_map
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    if delta_checkpoint is not None:
        loaded_metadata = load_edited_parameter_checkpoint(model, delta_checkpoint)
        if checkpoint_metadata and loaded_metadata != checkpoint_metadata:
            raise ValueError(
                "Compact-checkpoint payload metadata differs from its JSON manifest: "
                f"{delta_checkpoint}"
            )
        print(
            "[checkpoint] applied compact parameter delta "
            f"edit_count={loaded_metadata.get('edit_count')} "
            f"parameters={len(loaded_metadata.get('parameter_names', []))} "
            f"to Base={source}"
        )
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def collect_residual_states(
    model: Any,
    probes: Sequence[Probe],
    layers: Sequence[int],
    positions: Sequence[str],
) -> Dict[str, Dict[int, torch.Tensor]]:
    rows: Dict[str, Dict[int, List[torch.Tensor]]] = {
        position: {layer: [] for layer in layers} for position in positions
    }
    device = model_input_device(model)
    for probe_index, probe in enumerate(probes):
        outputs = model(
            **probe_inputs(probe, device),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        for layer in layers:
            if layer < 0 or layer >= len(hidden_states) - 1:
                raise ValueError(
                    f"Layer {layer} is invalid for {len(hidden_states) - 1} decoder blocks"
                )
            for position in positions:
                token_pos = probe.positions[position]
                vector = hidden_states[layer][0, token_pos].detach().float().cpu()
                rows[position][layer].append(vector)
        if (probe_index + 1) % 10 == 0 or probe_index + 1 == len(probes):
            print(f"[covariance] collected {probe_index + 1}/{len(probes)} prompts")
        del outputs

    return {
        position: {layer: torch.stack(vectors) for layer, vectors in layer_rows.items()}
        for position, layer_rows in rows.items()
    }


def spectrum_metrics(states: torch.Tensor) -> Tuple[np.ndarray, Dict[str, float], torch.Tensor]:
    states = states.float()
    centered = states - states.mean(dim=0, keepdim=True)
    if centered.size(0) <= 1:
        singular = torch.zeros(1, dtype=torch.float32)
        basis = torch.eye(centered.size(1), dtype=torch.float32)[:, :1]
    else:
        _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh.T.contiguous()
    spectrum = singular.square() / max(centered.size(0) - 1, 1)
    total = float(spectrum.sum())
    if total > EPS:
        probabilities = spectrum.double() / total
        nonzero = probabilities[probabilities > 0]
        effective_rank = float(torch.exp(-(nonzero * nonzero.log()).sum()))
        participation_ratio = float(total * total / (spectrum.double().square().sum() + EPS))
        top_fraction = float(spectrum.max() / total)
    else:
        effective_rank = 0.0
        participation_ratio = 0.0
        top_fraction = 0.0
    metrics = {
        "n_prompts": int(states.size(0)),
        "hidden_size": int(states.size(1)),
        "trace": total,
        "variance_per_dimension": total / max(states.size(1), 1),
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "top_eigenvalue_fraction": top_fraction,
    }
    return spectrum.double().numpy(), metrics, basis


def fit_shared_bases(
    base_states: Mapping[str, Mapping[int, torch.Tensor]],
    positions: Sequence[str],
    layers: Sequence[int],
    requested_rank: int,
) -> Tuple[Dict[str, Dict[int, torch.Tensor]], Dict[str, Dict[int, np.ndarray]], List[Dict[str, Any]]]:
    bases: Dict[str, Dict[int, torch.Tensor]] = {position: {} for position in positions}
    spectra: Dict[str, Dict[int, np.ndarray]] = {position: {} for position in positions}
    rows: List[Dict[str, Any]] = []
    for position in positions:
        for layer in layers:
            spectrum, metrics, full_basis = spectrum_metrics(base_states[position][layer])
            rank = min(requested_rank, max(base_states[position][layer].size(0) - 1, 1), full_basis.size(1))
            bases[position][layer] = full_basis[:, :rank].contiguous()
            spectra[position][layer] = spectrum
            rows.append({"position": position, "layer": layer, "pca_rank": rank, **metrics})
            print(
                f"[basis] {position} L{layer}: rank={rank}, "
                f"eRank={metrics['effective_rank']:.2f}, trace={metrics['trace']:.4g}"
            )
    return bases, spectra, rows


def _detach_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_tree(item) for key, item in value.items()}
    return value


LayerContext = Tuple[Tuple[Any, ...], Dict[str, Any]]


def capture_layer_contexts(
    model: Any,
    inputs: Mapping[str, torch.Tensor],
    layers: Sequence[int],
) -> Dict[int, LayerContext]:
    decoder_layers = model_layers(model)
    contexts: Dict[int, LayerContext] = {}
    handles = []

    for layer_id in layers:
        if layer_id < 0 or layer_id >= len(decoder_layers):
            raise ValueError(f"Layer {layer_id} is out of range for {len(decoder_layers)} blocks")

        def capture(_module: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any], *, key: int = layer_id) -> None:
            contexts[key] = (_detach_tree(tuple(args)), _detach_tree(dict(kwargs)))

        handles.append(decoder_layers[layer_id].register_forward_pre_hook(capture, with_kwargs=True))

    try:
        with torch.no_grad():
            model(**inputs, output_hidden_states=False, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    missing = [layer for layer in layers if layer not in contexts]
    if missing:
        raise RuntimeError(f"Forward hooks did not capture layers: {missing}")
    return contexts


def _layer_hidden_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"Could not find hidden-state tensor in layer output {type(output).__name__}")


def projected_local_jacobian(
    layer: torch.nn.Module,
    context: LayerContext,
    token_pos: int,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Return U.T J U without materializing the full token-local Jacobian.

    All residual positions except ``token_pos`` are held fixed.  Rows are
    computed with first-order vector-Jacobian products, avoiding unsupported
    second derivatives through attention kernels.
    """

    args, kwargs = context
    if not args or not torch.is_tensor(args[0]):
        raise TypeError("Expected decoder layer hidden_states as its first positional argument")
    hidden = args[0]
    if hidden.size(0) != 1:
        raise ValueError("projected_local_jacobian currently expects batch size 1")
    if token_pos < 0 or token_pos >= hidden.size(1):
        raise IndexError(f"token position {token_pos} is out of range for sequence length {hidden.size(1)}")

    # Under Accelerate model parallelism a decoder block can sit on the next
    # device boundary: the captured input remains on the previous device while
    # the module hook moves the patched input and produces ``y`` elsewhere.
    # The input-side projection and output-side VJP vector therefore need
    # separate device copies of the same fixed PCA basis.
    input_basis = basis.to(device=hidden.device, dtype=hidden.dtype)
    x0 = hidden[0, token_pos].detach().clone().requires_grad_(True)

    def block_output(x: torch.Tensor) -> torch.Tensor:
        patched = torch.cat(
            [hidden[:, :token_pos], x.view(1, 1, -1), hidden[:, token_pos + 1 :]],
            dim=1,
        )
        patched_args = (patched,) + tuple(args[1:])
        output = layer(*patched_args, **kwargs)
        return _layer_hidden_output(output)[0, token_pos]

    with torch.enable_grad():
        y = block_output(x0)
        output_basis = basis.to(device=y.device, dtype=y.dtype)
        rows = []
        rank = input_basis.size(1)
        for index in range(rank):
            grad = torch.autograd.grad(
                y,
                x0,
                grad_outputs=output_basis[:, index],
                retain_graph=index + 1 < rank,
                create_graph=False,
                allow_unused=False,
            )[0]
            rows.append(grad @ input_basis)
    return torch.stack(rows).detach().float().cpu()


def eigen_metrics(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.complex128).reshape(-1)
    if values.size == 0:
        return {
            "eig_spread": float("nan"),
            "eig_var_real": float("nan"),
            "eig_var_imag": float("nan"),
            "spectral_radius": float("nan"),
            "mean_abs_imag": float("nan"),
        }
    var_real = float(np.var(values.real))
    var_imag = float(np.var(values.imag))
    return {
        "eig_spread": var_real + var_imag,
        "eig_var_real": var_real,
        "eig_var_imag": var_imag,
        "spectral_radius": float(np.max(np.abs(values))),
        "mean_abs_imag": float(np.mean(np.abs(values.imag))),
    }


def matrix_metrics(matrix: np.ndarray, prefix: str) -> Tuple[np.ndarray, Dict[str, float]]:
    matrix = np.asarray(matrix, dtype=np.float64)
    eigvals = np.linalg.eigvals(matrix)
    singular = np.linalg.svd(matrix, compute_uv=False)
    squared = singular**2
    if squared.sum() > EPS:
        probs = squared / squared.sum()
        nonzero = probs[probs > 0]
        energy_erank = float(np.exp(-(nonzero * np.log(nonzero)).sum()))
        stable_rank = float(squared.sum() / (squared.max() + EPS))
    else:
        energy_erank = 0.0
        stable_rank = 0.0
    frobenius = float(np.linalg.norm(matrix, ord="fro"))
    commutator = matrix.T @ matrix - matrix @ matrix.T
    metrics = {
        f"{prefix}_frobenius": frobenius,
        f"{prefix}_energy_erank": energy_erank,
        f"{prefix}_stable_rank": stable_rank,
        f"{prefix}_commutator": float(np.linalg.norm(commutator, ord="fro") / (frobenius**2 + EPS)),
    }
    metrics.update({f"{prefix}_{key}": value for key, value in eigen_metrics(eigvals).items()})
    return eigvals, metrics


def analyze_projected_jacobians(
    model: Any,
    probes: Sequence[Probe],
    layers: Sequence[int],
    positions: Sequence[str],
    bases: Mapping[str, Mapping[int, torch.Tensor]],
    n_prompts: int,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, int], List[np.ndarray]]]:
    decoder_layers = model_layers(model)
    device = model_input_device(model)
    sample_rows: List[Dict[str, Any]] = []
    matrices: Dict[Tuple[str, int], List[np.ndarray]] = {
        (position, layer): [] for position in positions for layer in layers
    }
    selected = list(probes[: min(n_prompts, len(probes))])

    for prompt_index, probe in enumerate(selected):
        contexts = capture_layer_contexts(model, probe_inputs(probe, device), layers)
        for position in positions:
            token_pos = probe.positions[position]
            for layer_id in layers:
                projected_j = projected_local_jacobian(
                    decoder_layers[layer_id], contexts[layer_id], token_pos, bases[position][layer_id]
                ).double().numpy()
                projected_r = projected_j - np.eye(projected_j.shape[0], dtype=np.float64)
                r_norm = float(np.linalg.norm(projected_r, ord="fro"))
                projected_rhat = projected_r / (r_norm + EPS)
                _, j_metrics = matrix_metrics(projected_j, "J")
                _, r_metrics = matrix_metrics(projected_r, "R")
                _, rhat_metrics = matrix_metrics(projected_rhat, "Rhat")
                sample_rows.append(
                    {
                        "position": position,
                        "layer": layer_id,
                        "prompt_index": prompt_index,
                        "case_id": probe.case_id,
                        "token_position": token_pos,
                        "pca_rank": projected_j.shape[0],
                        **j_metrics,
                        **r_metrics,
                        **rhat_metrics,
                    }
                )
                matrices[(position, layer_id)].append(projected_j)
        print(f"[jacobian] completed {prompt_index + 1}/{len(selected)} prompts")
        del contexts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return sample_rows, matrices


def aggregate_jacobians(
    label: str,
    positions: Sequence[str],
    layers: Sequence[int],
    sample_rows: Sequence[Mapping[str, Any]],
    matrices: Mapping[Tuple[str, int], Sequence[np.ndarray]],
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, int, str], np.ndarray]]:
    summary_rows: List[Dict[str, Any]] = []
    pooled_eigs: Dict[Tuple[str, int, str], np.ndarray] = {}
    metric_names = [
        "J_frobenius",
        "J_eig_spread",
        "J_spectral_radius",
        "J_mean_abs_imag",
        "R_frobenius",
        "R_eig_spread",
        "R_spectral_radius",
        "R_mean_abs_imag",
        "R_energy_erank",
        "R_stable_rank",
        "R_commutator",
        "Rhat_eig_spread",
        "Rhat_spectral_radius",
        "Rhat_mean_abs_imag",
        "Rhat_energy_erank",
        "Rhat_stable_rank",
        "Rhat_commutator",
    ]

    for position in positions:
        for layer in layers:
            layer_matrices = list(matrices[(position, layer)])
            if not layer_matrices:
                continue
            selected_rows = [
                row for row in sample_rows if row["position"] == position and int(row["layer"]) == layer
            ]
            row: Dict[str, Any] = {
                "state": label,
                "position": position,
                "layer": layer,
                "n_prompts": len(layer_matrices),
                "pca_rank": layer_matrices[0].shape[0],
            }
            for name in metric_names:
                row[f"per_sample_mean_{name}"] = float(np.mean([float(item[name]) for item in selected_rows]))

            j_values: List[np.ndarray] = []
            r_values: List[np.ndarray] = []
            rhat_values: List[np.ndarray] = []
            r_matrices = []
            rhat_matrices = []
            for projected_j in layer_matrices:
                projected_r = projected_j - np.eye(projected_j.shape[0])
                projected_rhat = projected_r / (np.linalg.norm(projected_r, ord="fro") + EPS)
                j_values.append(np.linalg.eigvals(projected_j))
                r_values.append(np.linalg.eigvals(projected_r))
                rhat_values.append(np.linalg.eigvals(projected_rhat))
                r_matrices.append(projected_r)
                rhat_matrices.append(projected_rhat)

            for kind, values in (("J", j_values), ("R", r_values), ("Rhat", rhat_values)):
                pooled = np.concatenate(values)
                pooled_eigs[(position, layer, kind)] = pooled
                row.update({f"pooled_{kind}_{key}": value for key, value in eigen_metrics(pooled).items()})

            mean_j = np.mean(layer_matrices, axis=0)
            mean_r = np.mean(r_matrices, axis=0)
            # Normalize after averaging and also average individually normalized R.
            mean_rhat_after = mean_r / (np.linalg.norm(mean_r, ord="fro") + EPS)
            mean_of_rhat = np.mean(rhat_matrices, axis=0)
            for prefix, matrix in (
                ("mean_J", mean_j),
                ("mean_R", mean_r),
                ("mean_Rhat_after_mean", mean_rhat_after),
                ("mean_of_Rhat", mean_of_rhat),
            ):
                _, metrics = matrix_metrics(matrix, prefix)
                row.update(metrics)
            summary_rows.append(row)
    return summary_rows, pooled_eigs


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_key(*parts: Any) -> str:
    return "__".join(re.sub(r"[^A-Za-z0-9_.-]+", "_", str(part)) for part in parts)


def load_external_bases(
    path: str,
    positions: Sequence[str],
    layers: Sequence[int],
    requested_rank: int,
    hidden_size: int,
) -> Dict[str, Dict[int, torch.Tensor]]:
    loaded = np.load(path)
    bases: Dict[str, Dict[int, torch.Tensor]] = {position: {} for position in positions}
    for position in positions:
        for layer in layers:
            key = safe_key(position, f"L{layer}")
            if key not in loaded:
                raise KeyError(f"External PCA basis {path} has no array named {key!r}")
            array = np.asarray(loaded[key], dtype=np.float32)
            if array.ndim != 2 or array.shape[0] != hidden_size:
                raise ValueError(
                    f"External basis {key} has shape {array.shape}; expected ({hidden_size}, rank)"
                )
            rank = min(requested_rank, array.shape[1])
            basis = torch.from_numpy(array[:, :rank].copy()).float()
            gram = basis.T @ basis
            if not torch.allclose(gram, torch.eye(rank), rtol=2e-3, atol=2e-3):
                raise ValueError(f"External basis {key} is not approximately orthonormal")
            bases[position][layer] = basis
            print(f"[basis] loaded external {position} L{layer}: rank={rank}")
    loaded.close()
    return bases


def _state_colors(labels: Sequence[str]) -> Dict[str, Any]:
    fallback = plt.get_cmap("tab10")
    colors: Dict[str, Any] = {}
    fallback_index = 0
    for label in labels:
        normalized = label.strip().lower()
        if normalized in {"base", "base reference"}:
            colors[label] = BASE_BACKGROUND_COLOR
        elif normalized in METHOD_COLORS:
            colors[label] = METHOD_COLORS[normalized]
        else:
            colors[label] = fallback(fallback_index % 10)
            fallback_index += 1
    return colors


def _line_style(label: str, base_label: str) -> Dict[str, Any]:
    if label == base_label:
        return {"linestyle": "--", "linewidth": 2.6, "alpha": 0.72, "zorder": 1}
    return {"linestyle": "-", "linewidth": 1.8, "alpha": 0.95, "zorder": 2}


def _state_marker(label: str, base_label: str) -> str:
    if label == base_label:
        return "o"
    return METHOD_MARKERS.get(label.strip().lower(), "o")


def plot_covariance(
    output_dir: Path,
    labels: Sequence[str],
    positions: Sequence[str],
    layers: Sequence[int],
    spectra: Mapping[Tuple[str, str, int], np.ndarray],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    colors = _state_colors(labels)
    base_label = labels[0]
    for position in positions:
        ncols = min(4, len(layers))
        nrows = math.ceil(len(layers) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.1 * nrows), squeeze=False)
        for axis, layer in zip(axes.flat, layers):
            for label in labels:
                values = spectra[(label, position, layer)]
                total = float(values.sum())
                fraction = values / total if total > EPS else values
                positive = np.maximum(fraction, 1e-12)
                axis.plot(
                    np.arange(1, len(values) + 1),
                    positive,
                    label=label,
                    color=colors[label],
                    **_line_style(label, base_label),
                )
            axis.set_title(f"Layer {layer}")
            axis.set_yscale("log")
            axis.set_xlabel("covariance eigenvalue index")
            axis.set_ylabel("explained-variance fraction")
            axis.grid(alpha=0.2)
        for axis in axes.flat[len(layers) :]:
            axis.axis("off")
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", ncol=min(len(labels), 5), frameon=False)
        fig.suptitle(f"Residual covariance spectrum — {position}", y=1.01, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / f"covariance_spectrum_{position}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
        metric_specs = [
            ("variance_per_dimension", "variance / hidden dimension"),
            ("effective_rank", "covariance effective rank"),
            ("top_eigenvalue_fraction", "top eigenvalue fraction"),
        ]
        for axis, (metric, title) in zip(axes, metric_specs):
            for label in labels:
                selected = sorted(
                    (
                        row for row in rows if row["state"] == label and row["position"] == position
                    ),
                    key=lambda item: int(item["layer"]),
                )
                axis.plot(
                    [int(item["layer"]) for item in selected],
                    [float(item[metric]) for item in selected],
                    marker=_state_marker(label, base_label),
                    markersize=5.5,
                    label=label,
                    color=colors[label],
                    **_line_style(label, base_label),
                )
            axis.set_title(title)
            axis.set_xlabel("layer")
            axis.grid(alpha=0.25)
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle(f"Residual covariance summary — {position}", fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / f"covariance_metrics_{position}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def _complex_limits(arrays: Iterable[np.ndarray], center_real: float = 0.0) -> Tuple[float, float, float]:
    values = [np.asarray(array).reshape(-1) for array in arrays if np.asarray(array).size]
    if not values:
        return center_real - 1.0, center_real + 1.0, 1.0
    joined = np.concatenate(values)
    re_lo, re_hi = np.percentile(joined.real, [0.3, 99.7])
    im_q = float(np.percentile(np.abs(joined.imag), 99.7))
    re_span = max(float(re_hi - re_lo), 1e-4)
    re_lo = min(float(re_lo - 0.15 * re_span), center_real)
    re_hi = max(float(re_hi + 0.15 * re_span), center_real)
    im_q = max(im_q * 1.15, 0.15 * (re_hi - re_lo), 1e-4)
    return re_lo, re_hi, im_q


def plot_jacobian_eigenvalues(
    output_dir: Path,
    labels: Sequence[str],
    positions: Sequence[str],
    layers: Sequence[int],
    eigs: Mapping[Tuple[str, str, int, str], np.ndarray],
) -> None:
    base_label = labels[0]
    colors = _state_colors(labels)
    comparison_labels = list(labels[1:]) if len(labels) > 1 else [base_label]
    for position in positions:
        for kind in ("J", "Rhat"):
            re_lo, re_hi, im_q = _complex_limits(
                (eigs[(label, position, layer, kind)] for label in labels for layer in layers),
                center_real=1.0 if kind == "J" else 0.0,
            )
            fig, axes = plt.subplots(
                len(comparison_labels),
                len(layers),
                figsize=(2.65 * len(layers), 2.55 * len(comparison_labels)),
                squeeze=False,
            )
            for row_index, label in enumerate(comparison_labels):
                for column_index, layer in enumerate(layers):
                    axis = axes[row_index, column_index]
                    values = eigs[(label, position, layer, kind)]
                    base_values = eigs[(base_label, position, layer, kind)]
                    axis.axhline(0, color="#d1d5db", lw=0.7)
                    if kind == "J":
                        axis.axvline(1, color="#fca5a5", ls=":", lw=0.8)
                        axis.add_patch(Circle((0, 0), 1.0, fill=False, ls="--", ec="#9ca3af", lw=0.7))
                    else:
                        axis.axvline(0, color="#d1d5db", lw=0.7)
                    # Always draw the base cloud first.  Every edited row is
                    # therefore an absolute overlay rather than a ratio plot.
                    axis.scatter(
                        base_values.real,
                        base_values.imag,
                        s=11,
                        alpha=0.14,
                        color=colors[base_label],
                        edgecolors="none",
                        zorder=1,
                    )
                    axis.scatter(
                        values.real,
                        values.imag,
                        s=9,
                        alpha=0.62,
                        marker=_state_marker(label, base_label),
                        color=colors[label],
                        edgecolors="none",
                        zorder=2,
                    )
                    axis.set_xlim(re_lo, re_hi)
                    axis.set_ylim(-im_q, im_q)
                    axis.set_aspect("equal", adjustable="box")
                    if row_index == 0:
                        axis.set_title(f"L{layer}")
                    if column_index == 0:
                        axis.set_ylabel(f"{label} over Base\nIm(λ)", fontsize=8)
                    if row_index == len(comparison_labels) - 1:
                        axis.set_xlabel("Re(λ)", fontsize=8)
                    axis.tick_params(labelsize=6)
            title = "projected J" if kind == "J" else "scale-normalized projected residual operator R̂"
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markersize=6,
                    markerfacecolor=colors[base_label],
                    markeredgecolor="none",
                    alpha=0.45,
                    label=f"{base_label} reference (background)",
                )
            ] + [
                Line2D(
                    [0],
                    [0],
                    marker=_state_marker(label, base_label),
                    linestyle="none",
                    markersize=6,
                    markerfacecolor=colors[label],
                    markeredgecolor="none",
                    label=label,
                )
                for label in comparison_labels
            ]
            fig.legend(
                handles=legend_handles,
                loc="upper center",
                ncol=min(len(legend_handles), 5),
                frameon=False,
                bbox_to_anchor=(0.5, 0.975),
            )
            fig.suptitle(
                f"Complex eigenvalue clouds: {title} — {position} (Base-reference overlay)",
                y=0.995,
                fontweight="bold",
            )
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            fig.savefig(output_dir / f"jacobian_eigenvalues_{kind}_{position}.png", dpi=180)
            plt.close(fig)


def plot_jacobian_metrics(
    output_dir: Path,
    labels: Sequence[str],
    positions: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    colors = _state_colors(labels)
    base_label = labels[0]
    specs = [
        ("pooled_R_eig_spread", "pooled raw R eigen spread"),
        ("pooled_Rhat_eig_spread", "pooled normalized R̂ eigen spread"),
        ("per_sample_mean_R_frobenius", "mean projected residual gain ‖R‖F"),
    ]
    for position in positions:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
        for axis, (metric, title) in zip(axes, specs):
            for label in labels:
                selected = sorted(
                    (row for row in rows if row["state"] == label and row["position"] == position),
                    key=lambda item: int(item["layer"]),
                )
                axis.plot(
                    [int(item["layer"]) for item in selected],
                    [float(item[metric]) for item in selected],
                    marker=_state_marker(label, base_label),
                    markersize=5.5,
                    label=label,
                    color=colors[label],
                    **_line_style(label, base_label),
                )
            axis.set_title(title)
            axis.set_xlabel("layer")
            axis.grid(alpha=0.25)
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle(f"Projected residual-dynamics metrics — {position}", fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / f"jacobian_metrics_{position}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def replot_existing_outputs(output_dir: Path) -> None:
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest for replot: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    labels = [str(state["label"]) for state in manifest["states"]]
    positions = [str(value) for value in manifest["positions"]]
    layers = [int(value) for value in manifest["layers"]]

    covariance_csv = output_dir / "covariance_metrics.csv"
    covariance_npz = output_dir / "covariance_spectra.npz"
    if covariance_csv.exists() and covariance_npz.exists():
        rows = read_csv_rows(covariance_csv)
        loaded = np.load(covariance_npz)
        spectra = {
            (label, position, layer): np.asarray(
                loaded[safe_key(label, position, f"L{layer}")]
            )
            for label in labels
            for position in positions
            for layer in layers
        }
        loaded.close()
        plot_covariance(output_dir, labels, positions, layers, spectra, rows)
        print(f"[replot] covariance figures refreshed in {output_dir}")

    jacobian_csv = output_dir / "jacobian_summary_metrics.csv"
    jacobian_npz = output_dir / "jacobian_eigenvalues.npz"
    if jacobian_csv.exists() and jacobian_npz.exists():
        rows = read_csv_rows(jacobian_csv)
        loaded = np.load(jacobian_npz)
        eigs = {
            (label, position, layer, kind): np.asarray(
                loaded[safe_key(label, position, f"L{layer}", kind)]
            )
            for label in labels
            for position in positions
            for layer in layers
            for kind in ("J", "Rhat")
        }
        loaded.close()
        plot_jacobian_eigenvalues(output_dir, labels, positions, layers, eigs)
        plot_jacobian_metrics(output_dir, labels, positions, rows)
        print(f"[replot] Jacobian figures refreshed in {output_dir}")

    if not covariance_csv.exists() and not jacobian_csv.exists():
        raise FileNotFoundError(f"No covariance or Jacobian result CSV was found in {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--state",
        type=parse_state_spec,
        action="append",
        default=[],
        help="Repeated LABEL=MODEL_PATH. The first state supplies every shared PCA basis.",
    )
    parser.add_argument("--requests-path", default=None, help="Saved ZsRE requests.json used for all states")
    parser.add_argument(
        "--exclude-requests-path",
        action="append",
        default=[],
        help="Rewrite request JSON to exclude by normalized (prompt, subject); repeatable",
    )
    parser.add_argument(
        "--probe-source",
        choices=["rewrite", "rephrase", "locality"],
        default="rewrite",
    )
    parser.add_argument("--probe-selection", choices=["prefix", "random"], default="prefix")
    parser.add_argument("--tokenizer-path", default=None, help="Defaults to the first model path")
    parser.add_argument(
        "--base-model-path",
        default=None,
        help=(
            "Base Hugging Face model used for any --state whose path is a "
            "compact edited_parameter_deltas.pt file. If omitted, base_model "
            "is read from the checkpoint's adjacent JSON manifest."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Regenerate figures from CSV/NPZ files in output-dir without loading any model",
    )
    parser.add_argument("--layers", type=parse_int_list, default=parse_int_list("4,5,6,7,8,12,16,20,24,28,31"))
    parser.add_argument(
        "--positions",
        type=parse_str_list,
        default=parse_str_list("subject_last"),
        help="Comma list from subject_last,prompt_last",
    )
    parser.add_argument("--max-prompts", type=int, default=50, help="Prompts used for covariance and PCA")
    parser.add_argument("--jacobian-prompts", type=int, default=12, help="Prefix prompts used for projected Jacobians")
    parser.add_argument("--pca-rank", type=int, default=32)
    parser.add_argument(
        "--pca-bases-path",
        default=None,
        help="Optional shared_pca_bases.npz from another run; useful for fixed-coordinate controls",
    )
    parser.add_argument(
        "--pca-bases-missing",
        choices=["error", "fallback"],
        default="error",
        help="If an external basis is incompatible, error or fit the current probe Base basis",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--analysis", choices=["all", "covariance", "jacobian"], default="all")
    parser.add_argument(
        "--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--device-map", default="auto", help="Hugging Face device_map or 'none'")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="eager")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.replot_only:
        return
    if not args.state:
        raise ValueError("At least one --state LABEL=MODEL_PATH is required")
    if not args.requests_path:
        raise ValueError("--requests-path is required unless --replot-only is used")
    bad_positions = sorted(set(args.positions) - {"subject_last", "prompt_last"})
    if bad_positions:
        raise ValueError(f"Unknown positions: {bad_positions}")
    if args.probe_source == "locality" and "subject_last" in args.positions:
        raise ValueError("locality probes have no edited subject; use --positions prompt_last")
    if args.max_prompts <= 1:
        raise ValueError("--max-prompts must be greater than 1")
    if args.pca_rank <= 0:
        raise ValueError("--pca-rank must be positive")
    if args.jacobian_prompts <= 0 and args.analysis in {"all", "jacobian"}:
        raise ValueError("--jacobian-prompts must be positive when Jacobian analysis is enabled")
    labels = [state.label for state in args.state]
    if len(labels) != len(set(labels)):
        raise ValueError("state labels must be unique")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.replot_only:
        replot_existing_outputs(output_dir)
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    first_model_source, _, _ = resolve_model_load_spec(
        args.state[0].path,
        args.base_model_path,
    )
    tokenizer_path = args.tokenizer_path or first_model_source
    print(f"[tokenizer] {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    probes = load_probes(
        args.requests_path,
        tokenizer,
        args.positions,
        args.max_prompts,
        args.max_length,
        probe_source=args.probe_source,
        exclude_requests_paths=args.exclude_requests_path,
        selection=args.probe_selection,
        seed=args.seed,
    )

    labels = [state.label for state in args.state]
    covariance_rows: List[Dict[str, Any]] = []
    covariance_spectra: Dict[Tuple[str, str, int], np.ndarray] = {}
    jacobian_sample_rows: List[Dict[str, Any]] = []
    jacobian_summary_rows: List[Dict[str, Any]] = []
    pooled_eigenvalues: Dict[Tuple[str, str, int, str], np.ndarray] = {}
    projected_matrices: Dict[Tuple[str, str, int, int], np.ndarray] = {}

    # PCA bases always come from the first state, including analysis=jacobian.
    base_spec = args.state[0]
    print(f"[model] loading PCA reference {base_spec.label}: {base_spec.path}")
    base_model = load_model(
        base_spec.path,
        args.torch_dtype,
        args.device_map,
        args.trust_remote_code,
        args.attn_implementation,
        args.base_model_path,
    )
    base_states = collect_residual_states(base_model, probes, args.layers, args.positions)
    fitted_bases, _, basis_rows = fit_shared_bases(
        base_states, args.positions, args.layers, args.pca_rank
    )
    external_bases_used = False
    if args.pca_bases_path:
        hidden_size = next(iter(next(iter(base_states.values())).values())).size(1)
        try:
            bases = load_external_bases(
                args.pca_bases_path, args.positions, args.layers, args.pca_rank, hidden_size
            )
            external_bases_used = True
        except (KeyError, ValueError) as exc:
            if args.pca_bases_missing == "error":
                raise
            print(
                f"[basis] WARNING: external PCA basis is incompatible ({exc}); "
                "falling back to the current probe's Base-state PCA basis"
            )
            bases = fitted_bases
        for row in basis_rows:
            if external_bases_used:
                row["basis_source"] = str(Path(args.pca_bases_path).expanduser().resolve())
                row["pca_rank"] = bases[str(row["position"])][int(row["layer"])].size(1)
            else:
                row["basis_source"] = "fallback_current_probe_base_state"
    else:
        bases = fitted_bases
        for row in basis_rows:
            row["basis_source"] = "current_probe_base_state"
    basis_arrays = {
        safe_key(position, f"L{layer}"): basis.numpy()
        for position, layer_bases in bases.items()
        for layer, basis in layer_bases.items()
    }
    np.savez(output_dir / "shared_pca_bases.npz", **basis_arrays)
    for position, layer_bases in bases.items():
        np.savez(
            output_dir / f"shared_pca_bases_{position}.npz",
            **{
                safe_key(position, f"L{layer}"): basis.numpy()
                for layer, basis in layer_bases.items()
            },
        )
    write_csv(output_dir / "shared_pca_basis_metrics.csv", basis_rows)

    for state_index, state in enumerate(args.state):
        if state_index == 0:
            model = base_model
            states = base_states
        else:
            print(f"[model] loading {state.label}: {state.path}")
            model = load_model(
                state.path,
                args.torch_dtype,
                args.device_map,
                args.trust_remote_code,
                args.attn_implementation,
                args.base_model_path,
            )
            states = (
                collect_residual_states(model, probes, args.layers, args.positions)
                if args.analysis in {"all", "covariance"}
                else None
            )

        if args.analysis in {"all", "covariance"}:
            if states is None:
                raise RuntimeError("Residual states were not collected for covariance analysis")
            for position in args.positions:
                for layer in args.layers:
                    spectrum, metrics, _ = spectrum_metrics(states[position][layer])
                    covariance_spectra[(state.label, position, layer)] = spectrum
                    covariance_rows.append(
                        {"state": state.label, "position": position, "layer": layer, **metrics}
                    )

        if args.analysis in {"all", "jacobian"}:
            sample_rows, matrices = analyze_projected_jacobians(
                model,
                probes,
                args.layers,
                args.positions,
                bases,
                args.jacobian_prompts,
            )
            for row in sample_rows:
                row["state"] = state.label
            jacobian_sample_rows.extend(sample_rows)
            summary_rows, state_eigs = aggregate_jacobians(
                state.label, args.positions, args.layers, sample_rows, matrices
            )
            jacobian_summary_rows.extend(summary_rows)
            for (position, layer, kind), values in state_eigs.items():
                pooled_eigenvalues[(state.label, position, layer, kind)] = values
            for (position, layer), values in matrices.items():
                for prompt_index, matrix in enumerate(values):
                    projected_matrices[(state.label, position, layer, prompt_index)] = matrix

        if state_index > 0:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"[model] completed {state.label}")

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if covariance_rows:
        write_csv(output_dir / "covariance_metrics.csv", covariance_rows)
        np.savez(
            output_dir / "covariance_spectra.npz",
            **{
                safe_key(label, position, f"L{layer}"): values
                for (label, position, layer), values in covariance_spectra.items()
            },
        )
        plot_covariance(
            output_dir, labels, args.positions, args.layers, covariance_spectra, covariance_rows
        )

    if jacobian_summary_rows:
        write_csv(output_dir / "jacobian_sample_metrics.csv", jacobian_sample_rows)
        write_csv(output_dir / "jacobian_summary_metrics.csv", jacobian_summary_rows)
        np.savez(
            output_dir / "projected_jacobians.npz",
            **{
                safe_key(label, position, f"L{layer}", f"P{prompt_index}"): matrix
                for (label, position, layer, prompt_index), matrix in projected_matrices.items()
            },
        )
        np.savez(
            output_dir / "jacobian_eigenvalues.npz",
            **{
                safe_key(label, position, f"L{layer}", kind): values
                for (label, position, layer, kind), values in pooled_eigenvalues.items()
            },
        )
        plot_jacobian_eigenvalues(
            output_dir, labels, args.positions, args.layers, pooled_eigenvalues
        )
        plot_jacobian_metrics(output_dir, labels, args.positions, jacobian_summary_rows)

    manifest = {
        "states": [{"label": state.label, "path": state.path} for state in args.state],
        "base_model_path": args.base_model_path,
        "pca_reference_state": args.state[0].label,
        "requests_path": str(Path(args.requests_path).expanduser().resolve()),
        "exclude_requests_paths": [
            str(Path(path).expanduser().resolve()) for path in args.exclude_requests_path
        ],
        "probe_source": args.probe_source,
        "probe_selection": args.probe_selection,
        "n_prompts": len(probes),
        "jacobian_prompts": min(args.jacobian_prompts, len(probes)),
        "layers": args.layers,
        "positions": args.positions,
        "pca_rank_requested": args.pca_rank,
        "pca_bases_path": (
            str(Path(args.pca_bases_path).expanduser().resolve()) if args.pca_bases_path else None
        ),
        "pca_bases_missing_policy": args.pca_bases_missing,
        "external_pca_bases_used": external_bases_used,
        "analysis": args.analysis,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "definitions": {
            "covariance": "covariance of centered residual vectors h_l at the selected token",
            "projected_jacobian": "U_l^T (d h_{l+1,pos} / d h_{l,pos}) U_l, all other positions fixed",
            "residual_operator": "R = J - I",
            "normalized_residual_operator": "Rhat = R / ||R||_F",
            "eigen_spread": "Var(Re(lambda)) + Var(Im(lambda))",
        },
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"[done] outputs written to {output_dir}")


if __name__ == "__main__":
    main()
