#!/usr/bin/env python3
"""Restore saved sequential-edit checkpoints and capture canonical locality H9.

No editing or optimization. Every cumulative BF16 delta is applied to pristine
Base parameters. An input hook stops the ordinary eager forward at layer 9;
the resulting states must exactly reproduce the historical Base and endpoints.
"""
from __future__ import annotations

# Release-only filesystem defaults; computation below follows the source snapshot.
import sys as _bng_sys
from pathlib import Path as _BNGPath
_bng_sys.path.insert(0, str(_BNGPath(__file__).resolve().parents[1]))
from model_code.paths import RESEARCH_ROOT, OUTPUT_ROOT, DATA_ROOT, LLAMA_MODEL

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from transformers import AutoTokenizer
from diagnostics.analyze_base_component_cosines import _atomic_json, _atomic_npz, _model_backbone, _padded_batch
from diagnostics.analyze_residual_spectrum import load_model, model_input_device
from diagnostics.capture_matched_geometry_checkpoints import MODEL, NAMES, make_probes
from diagnostics.assemble_cumulative_prefix_locality1k import file_record, source_run, validate_native_replay

OUT = OUTPUT_ROOT / '_Analysis_Cross_Layer/a1_checkpoint_scatter_step50_20260913'
AUDIT = OUTPUT_ROOT / '_Analysis_Cross_Layer/hn_paper_source_audit_20260912/matched_h9_geometry.csv'
OLD = OUTPUT_ROOT / '_Analysis_Cross_Layer/all_methods_h9_capture_n1000_seed42_v1/captures'
BASE = OLD / "base/step_0000/locality/h_m_reference"
METHODS = ("Native", "HN", "NAS", "ENCORE", "SPHERE", "SADR")
STEPS = (50, 100, 150, 200, 250, 300, 500, 750, 1000)
FIELDS = ("editor", "method", "edit_count", "mean_q", "mean_abs_norm_deviation", "mean_p", "mean_kappa", "n_prompts", "raw_h9_path", "checkpoint_path")


def read(path):
    return json.loads(path.read_text())


def raw(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def checkpoint_for(editor, method, step):
    run = source_run(editor, method)
    path = run / f"step_{step:03d}/edited_parameter_deltas.pt"
    if not path.exists() and method == "Native":
        path = OUTPUT_ROOT / 'paper_a_cumulative_prefix_locality1000_n1000_v1/native_replay/full' / editor / run.name / f"step_{step:03d}/edited_parameter_deltas.pt"
    assert path.is_file(), path
    return path


def restore(model, pristine, checkpoint, run, step):
    metadata = read(checkpoint.with_suffix(".json"))
    assert metadata["edit_count"] == step
    assert metadata["storage_mode"] == "parameter_deltas"
    if "rewrite_layers" in metadata:
        assert metadata["rewrite_layers"] == [4, 5, 6, 7, 8]
    assert metadata["parameter_names"] == NAMES
    original_fingerprint = read(run / "run_config.json")["fingerprint"]
    if "source_run_fingerprint" in metadata:
        assert Path(metadata["source_run"]).resolve() == run.resolve()
        assert metadata["source_run_fingerprint"] == original_fingerprint
        # The enclosing full replay was validated before model loading. Its
        # checkpoint sidecars retain their original provisional labels.
        assert metadata["run_fingerprint"] == read(checkpoint.parents[1] / "provenance.json")["replay_fingerprint"]
    else:
        assert metadata["run_fingerprint"] == original_fingerprint
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["metadata"] == metadata
    assert payload["format"] == "easyedit-edited-parameters" and payload["format_version"] == 1
    assert list(payload["state_dict"]) == NAMES
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, delta in payload["state_dict"].items():
            param = named[name]
            assert delta.dtype == param.dtype == torch.bfloat16
            assert delta.shape == param.shape
            param.copy_(pristine[name].to(param.device) + delta.to(param.device))
    return dict(checkpoint=file_record(checkpoint), checkpoint_metadata=metadata)


class H9Reached(Exception):
    pass


@torch.inference_mode()
def capture(model, tokenizer, probes):
    states = np.empty((1000, 1, 4096), dtype=np.float32)
    positions = np.empty((1000, 1), dtype=np.int32)
    selected, result = None, None
    def hook(module, args):
        nonlocal result
        h = args[0]
        rows = torch.arange(h.shape[0], device=h.device)[:, None]
        result = h[rows, selected.to(h.device)].float().cpu().numpy()
        raise H9Reached
    handle = _model_backbone(model).layers[9].register_forward_pre_hook(hook)
    try:
        for start in range(0, 1000, 16):
            end = min(start + 16, 1000)
            inputs, selected = _padded_batch(probes[start:end], ["prompt_last"],
                pad_token_id=int(tokenizer.pad_token_id), device=model_input_device(model))
            result = None
            try:
                _model_backbone(model)(**inputs, use_cache=False, output_hidden_states=False, return_dict=True)
            except H9Reached:
                pass
            assert result is not None
            states[start:end] = result
            positions[start:end] = selected.numpy()
    finally:
        handle.remove()
    assert np.isfinite(states).all(), "Nonfinite H9 states"
    return states, positions


def validate_coordinates(values, base):
    for key in ("case_ids", "positions", "token_positions", "valid_positions"):
        assert np.array_equal(values[key], base[key]), key
    assert values["h9"].shape == base["h9"].shape == (1000, 1, 4096)
    assert np.isfinite(values["h9"]).all()


def geometry(values, base):
    validate_coordinates(values, base)
    h0, h = base["h9"][:, 0].astype(np.float64), values["h9"][:, 0].astype(np.float64)
    delta = h - h0
    norm0 = np.linalg.norm(h0, axis=1)
    assert (norm0 > 0).all()
    p = np.sum(delta * h0, axis=1) / norm0**2
    q = np.linalg.norm(delta - p[:, None] * h0, axis=1) / norm0
    kappa = np.linalg.norm(h, axis=1) / norm0
    reconstructed = np.sqrt((1 + p)**2 + q**2)
    max_relative_error = float(np.max(np.abs(reconstructed - kappa) / np.maximum(kappa, 1)))
    assert max_relative_error < 1e-12, max_relative_error
    summary = dict(mean_q=float(q.mean()), mean_abs_norm_deviation=float(np.abs(kappa - 1).mean()),
        mean_p=float(p.mean()), mean_kappa=float(kappa.mean()), n_prompts=1000)
    return summary, dict(p=p, q=q, kappa=kappa), max_relative_error


def write_csv(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["editor"], METHODS.index(r["method"]), int(r["edit_count"]))))
    temporary.replace(path)


def merge(out):
    paths = [out / f"geometry_by_checkpoint_{editor}.csv" for editor in ("AlphaEdit", "MEMIT")]
    assert all(path.exists() for path in paths)
    rows = [row for path in paths for row in csv.DictReader(path.open())]
    assert len(rows) == 108
    assert len({(r["editor"], r["method"], r["edit_count"]) for r in rows}) == 108
    write_csv(out / "geometry_by_checkpoint.csv", rows)
    manifests = {editor: read(out / f"capture_manifest_{editor}.json") for editor in ("AlphaEdit", "MEMIT")}
    assert all(m["complete"] for m in manifests.values())
    _atomic_json(out / "geometry_manifest.json", dict(complete=True, n_checkpoints=108, editors=manifests,
        csv=file_record(out / "geometry_by_checkpoint.csv"), formula="p=dot(h-h0,h0)/||h0||^2; q=||(h-h0)-p*h0||/||h0||; kappa=||h||/||h0||; float64 per case then mean"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editor", choices=("AlphaEdit", "MEMIT"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge(out)
        return
    assert args.editor
    editor = args.editor
    torch.set_num_threads(4)
    torch.manual_seed(42)
    started = time.monotonic()
    manifest_path = out / f"capture_manifest_{editor}.json"
    rows, checks = [], []
    manifest = dict(complete=False, editor=editor, pid=os.getpid(), device=args.device, steps=list(STEPS),
        methods=list(METHODS), n_prompts=1000, batch_size=16, padding_side="right", max_length=256,
        positions=["prompt_last"], torch_dtype="bfloat16", attn_implementation="eager", model_path=str(MODEL),
        restore_semantics="pristine BF16 Base + saved cumulative BF16 delta, separately for every checkpoint",
        capture_semantics="ordinary eager model backbone forward stopped at input of decoder layer 9",
        base_raw=file_record(BASE / "raw_h9.npz"), code=file_record(Path(__file__)), checks=checks)
    def status(stage, **kwargs):
        manifest.update(stage=stage, elapsed_seconds=time.monotonic() - started,
                        n_completed=len(rows), **kwargs)
        _atomic_json(manifest_path, manifest)
        print(f"[{editor}] {stage} {kwargs} elapsed={manifest['elapsed_seconds']:.1f}s", flush=True)
    status("source_validation")
    endpoints = {(r["editor"], r["method"]): r for r in csv.DictReader(AUDIT.open()) if r["family"] == "locality" and r["position"] == "prompt_last"}
    requests = read(source_run(editor, "HN") / "requests.json")
    assert len(requests) == 1000
    for method in METHODS:
        assert read(source_run(editor, method) / "requests.json") == requests
        for step in STEPS:
            checkpoint_for(editor, method, step)
    native_run = source_run(editor, "Native")
    replay = OUTPUT_ROOT / 'paper_a_cumulative_prefix_locality1000_n1000_v1/native_replay/full' / editor / native_run.name
    manifest["native_replay_validation"] = validate_native_replay(replay, native_run)
    status("loading_model")
    model = load_model(str(MODEL), "bfloat16", args.device, False, "eager")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, use_fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    probes, positions, valid, missing = make_probes(requests, tokenizer, "locality")
    assert positions == ["prompt_last"] and valid.all() and not missing
    old_probes = read(BASE / "probes.json")["probes"]
    for probe, old in zip(probes, old_probes):
        assert str(probe.case_id) == old["case_id"]
        assert probe.prompt == old["prompt"] and probe.subject == old["subject"]
        assert list(probe.input_ids) == old["input_ids"]
        assert probe.positions == old["positions"]
    base = raw(BASE / "raw_h9.npz")
    status("validating_exact_base")
    captured, tp = capture(model, tokenizer, probes)
    assert np.array_equal(tp, base["token_positions"])
    assert np.array_equal(captured, base["h9"]), float(np.max(np.abs(captured - base["h9"])))
    manifest["base_exact_validation"] = dict(exact=True, n_prompts=1000, n_values=int(captured.size),
        input_ids_exact=True, token_positions_exact=True, maximum_absolute_error=0)
    named = dict(model.named_parameters())
    pristine = {name: named[name].detach().cpu().clone() for name in NAMES}
    for method in METHODS:
        for step in (1000, *STEPS[:-1]):
            destination = out / "captures" / editor / method / f"step_{step:04d}"
            complete_path = destination / "complete.json"
            checkpoint = checkpoint_for(editor, method, step)
            if complete_path.exists():
                done = read(complete_path)
                assert done["complete"] and done["editor"] == editor and done["method"] == method and done["edit_count"] == step
                assert done["identity"]["checkpoint"] == file_record(checkpoint)
                assert done["raw_h9"] == file_record(Path(done["raw_h9"]["path"]))
                values = raw(Path(done["raw_h9"]["path"]))
                summary, _, error = geometry(values, base)
                assert summary == done["summary"]
                checks.append(done)
                rows.append(dict(editor=editor, method=method, edit_count=step, **summary,
                    raw_h9_path=done["raw_h9"]["path"], checkpoint_path=str(checkpoint.resolve())))
                write_csv(out / f"geometry_by_checkpoint_{editor}.csv", rows)
                status("reused_completed_capture", method=method, edit_count=step)
                continue
            status("restoring", method=method, edit_count=step)
            identity = restore(model, pristine, checkpoint, source_run(editor, method), step)
            versions = {name: int(param._version) for name, param in model.named_parameters()}
            status("capturing", method=method, edit_count=step)
            captured, tp = capture(model, tokenizer, probes)
            assert all(int(param._version) == versions[name] for name, param in model.named_parameters())
            assert np.array_equal(tp, base["token_positions"])
            values = dict(h9=captured, **{key: base[key] for key in ("case_ids", "positions", "token_positions", "valid_positions")})
            destination.mkdir(parents=True, exist_ok=True)
            endpoint_validation = None
            if step == 1000:
                endpoint = endpoints[(editor, method)]
                assert identity["checkpoint"]["sha256"] == endpoint["checkpoint_sha256"]
                raw_path = Path(endpoint["raw_h9_path"])
                assert file_record(raw_path)["sha256"] == endpoint["raw_h9_sha256"]
                old = raw(raw_path)
                validate_coordinates(old, base)
                assert np.array_equal(captured, old["h9"]), (editor, method, float(np.max(np.abs(captured - old["h9"]))))
                values = old
                endpoint_validation = dict(exact=True, n_prompts=1000, n_values=int(captured.size),
                    maximum_absolute_error=0, checkpoint_sha256_exact=True, canonical_row=endpoint)
            else:
                raw_path = destination / "raw_h9.npz"
                _atomic_npz(raw_path, values)
            summary, per_case, error = geometry(values, base)
            _atomic_npz(destination / "per_case_geometry.npz", dict(**per_case, case_ids=base["case_ids"]))
            done = dict(complete=True, editor=editor, method=method, edit_count=step, identity=identity,
                raw_h9=file_record(raw_path), summary=summary, endpoint_exact_validation=endpoint_validation,
                maximum_relative_geometry_identity_error=error, input_ids_exact=True, token_positions_exact=True,
                parameters_unchanged_during_inference=True, finite_h9=True,
                per_case_geometry=file_record(destination / "per_case_geometry.npz"))
            _atomic_json(complete_path, done)
            checks.append(done)
            rows.append(dict(editor=editor, method=method, edit_count=step, **summary,
                raw_h9_path=str(raw_path.resolve()), checkpoint_path=str(checkpoint.resolve())))
            write_csv(out / f"geometry_by_checkpoint_{editor}.csv", rows)
            status("checkpoint_complete", method=method, edit_count=step)
    assert len(rows) == 54
    manifest["complete"] = True
    status("complete", csv=file_record(out / f"geometry_by_checkpoint_{editor}.csv"))


if __name__ == "__main__":
    main()
