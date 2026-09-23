"""Matched current-request virtual/actual observations during sequential editing.

Virtual means W_(t-1) with the selected returned target replacing subject O8.
Actual means the committed W_t without an intervention. Attention and MLP
arrays are native branch writes; an additional effective-M view explicitly
attributes the external virtual injection to the terminal edited MLP.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from diagnostics.analyze_base_component_cosines import LayerComponentCapture
from diagnostics.analyze_residual_spectrum import model_input_device, model_layers
from diagnostics.post_update_tracking import _atomic_json, _atomic_npz, artifact_stem


POSITIONS = ("subject_last", "prompt_last")


def runtime_memory_metadata(model: Any) -> Dict[str, Any]:
    mapping = {str(key): str(value) for key, value in getattr(model, "hf_device_map", {}).items()}
    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    result: Dict[str, Any] = {
        "hf_device_map": mapping, "parameter_devices": parameter_devices,
        "cpu_or_disk_offload": any(value in {"cpu", "disk"} for value in mapping.values())
        or any(device == "cpu" for device in parameter_devices),
        "cuda_peak_memory": [],
    }
    if torch.cuda.is_available():
        result["cuda_peak_memory"] = [
            {"visible_device": index,
             "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
             "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index))}
            for index in range(torch.cuda.device_count())
        ]
    return result


def _unwrap(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, (tuple, list)) else output
    if not torch.is_tensor(value) or value.ndim != 3:
        raise ValueError("virtual/actual capture expects batch-first block outputs")
    return value


def _rewrap(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if isinstance(output, list):
        return [hidden] + output[1:]
    return hidden


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    ln = np.linalg.norm(left, axis=-1)
    rn = np.linalg.norm(right, axis=-1)
    denominator = ln[..., :, None] * rn[..., None, :]
    result = np.full(denominator.shape, np.nan, dtype=np.float32)
    np.divide(np.einsum("pld,prd->plr", left, right), denominator,
              out=result, where=denominator > 1e-12)
    return result


def phase_metrics(vectors: Mapping[str, np.ndarray], prefix: str) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for name in ("o", "a", "m"):
        value = np.asarray(vectors[name], dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite {prefix} {name} vectors")
        result[f"{prefix}_norm_{name}"] = np.linalg.norm(value, axis=-1)
        result[f"{prefix}_cos_o_{name}"] = _cosine(vectors["o"], value)
    return result


def capture_raw_prompt(
    model: Any, tokenizer: Any, raw_prompt: str, *, subject_index: int,
    max_length: int, target_layer: int, target: Optional[torch.Tensor] = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """One no-grad forward; replace only the selected subject output if requested."""
    batch = tokenizer(raw_prompt, add_special_tokens=True, return_tensors="pt")
    ids = batch["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("current-edit observations require exactly one raw prompt")
    length = int(ids.shape[1])
    if length > max_length:
        raise ValueError("raw current-edit prompt exceeds capture max_length; refusing silent truncation")
    if not 0 <= subject_index < length:
        raise ValueError("selected inner subject index is outside the raw prompt")
    decoder = model_layers(model)
    if not 0 <= target_layer < len(decoder):
        raise ValueError("selected target layer is outside the decoder")
    token_positions = [int(subject_index), length - 1]
    capture = LayerComponentCapture(model, list(range(len(decoder))))
    capture.begin_batch(torch.tensor([token_positions], dtype=torch.long))
    injection_handle = None
    native_target = None
    injected_target = None
    native_observed = None
    parameter_versions = [(parameter, parameter._version) for parameter in model.parameters()]
    training_flags = [(module, module.training) for module in model.modules()]
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    cuda_devices = sorted({parameter.device.index for parameter in model.parameters()
                           if parameter.device.type == "cuda"})

    def inject(_module: Any, _inputs: Any, output: Any) -> Any:
        nonlocal native_target, injected_target, native_observed
        hidden = _unwrap(output)
        native_target = hidden[0, subject_index].detach().float().cpu().clone()
        native_observed = hidden[0, token_positions].detach().float().cpu().clone()
        if target.ndim != 1 or target.numel() != hidden.shape[-1] or not bool(torch.isfinite(target).all()):
            raise ValueError("selected target has invalid shape or non-finite values")
        updated = hidden.clone()
        updated[0, subject_index] = target.to(device=hidden.device, dtype=hidden.dtype)
        injected_target = updated[0, subject_index].detach().float().cpu().clone()
        return _rewrap(output, updated)

    try:
        model.eval()
        # Register replacement first: the block output collector must see z.
        if target is not None:
            injection_handle = decoder[target_layer].register_forward_hook(inject)
        capture.install()
        device = model_input_device(model)
        inputs = {key: value.to(device) for key, value in batch.items()
                  if key in {"input_ids", "attention_mask", "position_ids"}}
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            model(**inputs, use_cache=False, output_hidden_states=False,
                  output_attentions=False, return_dict=True)
        capture.validate()
        vectors = {
            key: torch.stack([
                capture.values[(layer, source)][0].detach().float().cpu()
                for layer in range(len(decoder))
            ], dim=1).numpy().copy()
            for key, source in (("o", "output"), ("a", "attention"), ("m", "mlp"))
        }
    finally:
        capture.close()
        if injection_handle is not None:
            injection_handle.remove()
        for module, training in training_flags:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    if any(parameter._version != version for parameter, version in parameter_versions):
        raise RuntimeError("observation unexpectedly mutated model parameters")
    metadata: Dict[str, Any] = {
        "raw_prompt": raw_prompt,
        "raw_prompt_sha256": hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest(),
        "token_positions": token_positions,
        "input_ids": ids[0].tolist(),
        "token_count": length,
        "layers": list(range(len(decoder))),
        "hidden_size": int(vectors["o"].shape[-1]),
        "parameter_versions_unchanged": True,
    }
    if target is not None:
        metadata["native_subject_output"] = native_target.numpy()
        metadata["injected_subject_output"] = injected_target.numpy()
        metadata["native_target_layer_outputs"] = native_observed.numpy()
    return vectors, metadata


class VirtualActualEditRecorder:
    """Keep one virtual vector capture until the corresponding actual commit."""

    def __init__(self, artifact_root: Path, *, request: Mapping[str, Any],
                 edit_index: int, condition: str, max_length: int,
                 order_manifest_sha256: Optional[str] = None,
                 run_fingerprint: Optional[str] = None,
                 save_vectors: bool = True) -> None:
        self.root = Path(artifact_root) / "virtual_actual"
        self.request = dict(request)
        self.edit_index = int(edit_index)
        self.condition = str(condition)
        self.max_length = int(max_length)
        self.order_manifest_sha256 = order_manifest_sha256
        self.run_fingerprint = run_fingerprint
        self.save_vectors = bool(save_vectors)
        self.virtual = None
        self.latent = None
        self.metadata = None
        self.completed = False
        self.o0_axis_state = None

    def selected_target(self, *, model: Any, tokenizer: Any, request: Mapping[str, Any],
                        write_layer: int, target_init: torch.Tensor,
                        optimizer_delta: torch.Tensor, target: torch.Tensor,
                        canonical_subject_token_index: int,
                        canonical_subject_prefix_token_ids: list[int],
                        o0_axis_preservation_state: Optional[Mapping[str, Any]] = None) -> None:
        if self.virtual is not None:
            raise RuntimeError("more than one selected target published for the same edit")
        if str(request.get("case_id")) != str(self.request.get("case_id")):
            raise ValueError("selected target belongs to a different current edit")
        raw_prompt = str(self.request["prompt"]).format(self.request["subject"])
        virtual, metadata = capture_raw_prompt(
            model, tokenizer, raw_prompt, subject_index=canonical_subject_token_index,
            max_length=self.max_length, target_layer=int(write_layer), target=target,
        )
        if metadata["input_ids"][:canonical_subject_token_index + 1] != canonical_subject_prefix_token_ids:
            raise ValueError("raw prompt token prefix differs from the selected inner target context")
        self.virtual = virtual
        self.metadata = metadata
        self.latent = {
            "target_layer": int(write_layer),
            "target_init": target_init.detach().float().cpu().numpy().copy(),
            "optimizer_delta": optimizer_delta.detach().float().cpu().numpy().copy(),
            "optimized_z": target.detach().float().cpu().numpy().copy(),
        }
        if o0_axis_preservation_state is not None:
            if not self.save_vectors:
                raise ValueError("O0-axis observations require retained virtual/actual raw vectors")
            from diagnostics.o0_axis_observation_metrics import state_arrays
            cpu_state = {key: value.detach().float().cpu().numpy() if torch.is_tensor(value) and value.is_floating_point()
                         else value.detach().cpu().numpy() if torch.is_tensor(value) else value
                         for key, value in o0_axis_preservation_state.items()}
            self.o0_axis_state = state_arrays(cpu_state, self.latent["optimized_z"].shape[0])

    def actual(self, model: Any, tokenizer: Any) -> tuple[Path, Path]:
        if self.virtual is None or self.metadata is None or self.latent is None:
            raise RuntimeError("actual observation has no selected virtual target")
        if self.completed:
            raise RuntimeError("actual observation was already saved")
        layer = self.latent["target_layer"]
        actual, metadata = capture_raw_prompt(
            model, tokenizer, self.metadata["raw_prompt"],
            subject_index=self.metadata["token_positions"][0],
            max_length=self.max_length, target_layer=layer,
        )
        for field in ("input_ids", "token_positions", "raw_prompt_sha256", "layers"):
            if metadata[field] != self.metadata[field]:
                raise RuntimeError(f"virtual and actual observations disagree on {field}")
        arrays = {**phase_metrics(self.virtual, "virtual"), **phase_metrics(actual, "actual")}
        if self.save_vectors:
            for phase, vectors in (("virtual", self.virtual), ("actual", actual)):
                for component in ("o", "a", "m"):
                    arrays[f"{phase}_{component}"] = vectors[component]
        virtual_o, actual_o = self.virtual["o"], actual["o"]
        invariants = {
            "virtual_subject_target_exact": np.array_equal(
                virtual_o[0, layer], self.metadata["injected_subject_output"]),
            "prompt_last_passive_or_same_token": (
                metadata["token_positions"][0] == metadata["token_positions"][1]
                or np.array_equal(virtual_o[1, layer], self.metadata["native_target_layer_outputs"][1])
            ),
        }
        if layer == 8:
            invariants.update({
                "O0_to_O3_unchanged": np.array_equal(virtual_o[:, :4], actual_o[:, :4]),
                "A0_to_A4_unchanged": np.array_equal(self.virtual["a"][:, :5], actual["a"][:, :5]),
            })
        if not all(invariants.values()):
            raise RuntimeError(f"virtual/actual structural invariants failed: {invariants}")
        gap = np.linalg.norm(actual_o - virtual_o, axis=-1)
        arrays["output_gap_l2"] = gap
        arrays["output_gap_relative_l2"] = gap / np.maximum(np.linalg.norm(virtual_o, axis=-1), 1e-12)
        arrays["output_cos_virtual_actual"] = np.sum(virtual_o * actual_o, axis=-1) / np.maximum(
            np.linalg.norm(virtual_o, axis=-1) * np.linalg.norm(actual_o, axis=-1), 1e-12)
        native = self.metadata["native_subject_output"]
        injected = self.metadata["injected_subject_output"]
        injection_delta = injected - native
        effective_m = self.virtual["m"].copy()
        # The two requested observation positions can be the same token.
        for index, position in enumerate(metadata["token_positions"]):
            if position == metadata["token_positions"][0]:
                effective_m[index, layer] += injection_delta
        arrays["virtual_cos_o_m_effective"] = _cosine(virtual_o, effective_m)
        arrays["virtual_norm_m_effective"] = np.linalg.norm(effective_m, axis=-1)
        arrays.update({
            "optimized_z": self.latent["optimized_z"],
            "target_init": self.latent["target_init"],
            "optimizer_delta": self.latent["optimizer_delta"],
            "virtual_native_subject_output": native,
            "virtual_injected_subject_output": injected,
            "virtual_injection_delta": injection_delta,
            "virtual_native_target_layer_outputs": self.metadata["native_target_layer_outputs"],
            "actual_subject_output": actual_o[0, layer],
            "actual_target_difference": actual_o[0, layer] - self.latent["optimized_z"],
            "z_target_layer": np.asarray(layer, dtype=np.int64),
            "case_id": np.asarray(str(self.request.get("case_id"))),
            "edit_index": np.asarray(self.edit_index, dtype=np.int64),
            "positions": np.asarray(POSITIONS),
            "layers": np.asarray(metadata["layers"], dtype=np.int64),
            "token_positions": np.asarray(metadata["token_positions"], dtype=np.int64),
        })
        o0_observation = None
        if self.o0_axis_state is not None:
            from diagnostics.o0_axis_observation_metrics import observation_metrics
            arrays.update(self.o0_axis_state)
            o0_arrays, o0_observation = observation_metrics(arrays)
            arrays.update(o0_arrays)
        stem = artifact_stem(self.edit_index, self.request)
        npz_path, json_path = self.root / f"{stem}.npz", self.root / f"{stem}.json"
        _atomic_npz(npz_path, arrays)
        sidecar = {
            "schema_version": 1, "complete": True, "editor": "AlphaEdit",
            "condition": self.condition, "edit_index": self.edit_index,
            "case_id": self.request.get("case_id"), "positions": list(POSITIONS),
            "z_target_layer": layer, "subject": self.request["subject"],
            "raw_prompt": metadata["raw_prompt"],
            "raw_prompt_sha256": metadata["raw_prompt_sha256"],
            "input_ids": metadata["input_ids"], "token_positions": metadata["token_positions"],
            "layers": metadata["layers"], "order_manifest_sha256": self.order_manifest_sha256,
            "run_fingerprint": self.run_fingerprint,
            "pre_edit_count": self.edit_index - 1, "post_edit_count": self.edit_index,
            "virtual_semantics": "pre-edit weights W_(t-1), selected returned z replaces only subject_last block output",
            "actual_semantics": "post-edit committed weights W_t, no injection, including any SPHERE projection",
            "virtual_m_semantics": "native MLP branch output excluding external target injection",
            "virtual_m_effective_semantics": "native M plus actual dtype-rounded external injection at subject target layer",
            "observation_prompt_semantics": "canonical current-edit raw rewrite prompt; no generated prefix or teacher-forced target suffix",
            "matrix_layout": "position, output_anchor_layer, reference_layer",
            "undefined_zero_norm_cosines": "NaN, accompanied by raw component norms",
            "save_vectors": self.save_vectors,
            "vector_layout": "position, layer, hidden_dimension",
            "parameter_versions_unchanged_during_observations": True,
            "npz_file": npz_path.name,
            "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
            "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
            "target_init_vs_raw_native_l2": float(np.linalg.norm(self.latent["target_init"] - native)),
            "actual_vs_returned_z_l2": float(np.linalg.norm(arrays["actual_target_difference"])),
            "actual_vs_virtual_injected_l2": float(gap[0, layer]),
            "runtime": runtime_memory_metadata(model),
            "invariants": invariants,
        }
        if o0_observation is not None:
            sidecar["o0_axis_preservation"] = o0_observation
        _atomic_json(json_path, sidecar)
        self.completed = True
        self.virtual = None
        return npz_path, json_path
