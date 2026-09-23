#!/usr/bin/env python3
"""GPT-2 XL fixed-panel H18 geometry and performance from cumulative deltas.

Every checkpoint is restored as pristine Base + its saved parameter delta.
The fixed panel is the canonical first 1,000 zsRE cases, including duplicate
prompt occurrences. Geometry uses prompt-only H18 at locality/prompt_last and
rewrite/subject_last, each relative to the matching pretrained Base state.
Generation EFF/Gen follows the paper's normalized target-prefix, greedy,
12-new-token protocol. Corrected teacher-forced EFF/Gen is saved separately;
LOC is case-macro Base/post argmax agreement on the identical target span.
No fitting, editing, bootstrap, or changes to existing Llama analyses occur.
"""
from __future__ import annotations

# Release-only filesystem defaults; computation below follows the source snapshot.
import sys as _bng_sys
from pathlib import Path as _BNGPath
_bng_sys.path.insert(0, str(_BNGPath(__file__).resolve().parents[1]))
from model_code.paths import RESEARCH_ROOT, OUTPUT_ROOT, DATA_ROOT, LLAMA_MODEL

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diagnostics.target_text_contract import strip_training_terminators
from diagnostics.analyze_residual_spectrum import build_probe

STEPS = (50, 100, 150, 200, 250, 300, 500, 750, 1000)
LAYERS = (13, 14, 15, 16, 17)
NAMES = tuple(f"transformer.h.{i}.mlp.c_proj.weight" for i in LAYERS)
SCHEMA = 1


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record(path: Path) -> dict:
    return dict(path=str(path.resolve()), sha256=digest(path), bytes=path.stat().st_size)


def runner_json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, indent=2).encode()).hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **values)
    tmp.replace(path)


def materialize(tokenizer, prompts: list[str], targets: list[str], device="cpu"):
    """Joint tokenization, exact target spans, real EOS retained, absolute positions.

    Genuine EOS shares pad_token_id in GPT-2. Only attention_mask identifies
    padding. Prefix equality fails closed if joint tokenization merges a token
    across the prompt/target boundary instead of silently shifting labels.
    """
    require(len(prompts) == len(targets) > 0, "Empty or unmatched prompt/target batch")
    old = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        batch = tokenizer([p + " " + t for p, t in zip(prompts, targets)],
                          padding=True, return_tensors="pt", add_special_tokens=True)
    finally:
        tokenizer.padding_side = old
    mask = batch["attention_mask"]
    batch["position_ids"] = (mask.cumsum(-1) - 1).clamp_min(0)
    starts, proofs = [], []
    for i, prompt in enumerate(prompts):
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        ids, attention = batch["input_ids"][i].tolist(), mask[i].tolist()
        padding = attention.count(0)
        start = padding + len(prompt_ids)
        require(attention == [0] * padding + [1] * (len(ids) - padding), "Non-contiguous padding")
        require(ids[padding:start] == prompt_ids, "Joint tokenization changes prompt prefix")
        require(padding < start < len(ids), "Empty prompt or target span")
        starts.append(start)
        proofs.append(dict(prompt_token_count=len(prompt_ids), padding_count=padding,
                           label_start=start, logit_start=start - 1,
                           target_token_ids=ids[start:],
                           final_token_is_eos=ids[-1] == tokenizer.eos_token_id,
                           full_nonpadding_ids=ids[padding:]))
    return {k: v.to(device) for k, v in batch.items()}, starts, proofs


@torch.inference_mode()
def teacher_forced(model, tokenizer, prompts, targets, batch_size=8):
    rows = []
    device = next(model.parameters()).device
    for begin in range(0, len(prompts), batch_size):
        inputs, starts, proofs = materialize(tokenizer, prompts[begin:begin + batch_size],
                                            targets[begin:begin + batch_size], device)
        require(inputs["input_ids"].shape[1] <= model.config.n_positions,
                "Teacher-forced sequence exceeds GPT-2 position limit; no truncation is permitted")
        logits = model(**inputs, use_cache=False).logits
        require(bool(torch.isfinite(logits).all()), "Nonfinite teacher-forced logits")
        predictions = logits.argmax(-1).cpu().numpy()
        for i, (start, proof) in enumerate(zip(starts, proofs)):
            gold = np.asarray(proof["target_token_ids"], dtype=np.int64)
            pred = predictions[i, start - 1:-1]
            require(pred.shape == gold.shape and len(gold) > 0, "Target prediction span mismatch")
            content_count = len(gold) - int(proof["final_token_is_eos"])
            require(content_count > 0, "Target contains no semantic tokens")
            rows.append(dict(**proof, predicted_token_ids=pred.tolist(),
                             token_accuracy=float(np.equal(pred, gold).mean()),
                             content_token_accuracy=float(np.equal(pred[:content_count], gold[:content_count]).mean()),
                             all_tokens_correct=bool(np.equal(pred, gold).all())))
        del logits, inputs
    return rows


def target_prefix_match(prediction: str, target: str) -> bool:
    # Identical normalization to eval_cumulative_generation_locality.py.
    normalized = lambda text: " ".join(text.strip().lower().split())
    expected = normalized(target)
    return bool(expected) and normalized(prediction).startswith(expected)


@torch.inference_mode()
def generate_answers(model, tokenizer, prompts, targets, batch_size=8, max_new_tokens=12):
    rows = []
    device = next(model.parameters()).device
    old = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for begin in range(0, len(prompts), batch_size):
            chunk = prompts[begin:begin + batch_size]
            encoded = tokenizer(chunk, padding=True, truncation=True, max_length=512,
                                return_tensors="pt").to(device)
            # GPT2LMHeadModel.prepare_inputs_for_generation derives absolute
            # positions from attention_mask at every decoding step. Passing a
            # frozen position_ids array here would prevent that cache update.
            generated = model.generate(**encoded, max_new_tokens=max_new_tokens,
                                       do_sample=False, num_beams=1, use_cache=True,
                                       pad_token_id=tokenizer.pad_token_id,
                                       eos_token_id=tokenizer.eos_token_id)
            continuation_ids = generated[:, encoded["input_ids"].shape[1]:].cpu().tolist()
            decoded = tokenizer.batch_decode(continuation_ids, skip_special_tokens=True)
            for i, (prediction, ids) in enumerate(zip(decoded, continuation_ids)):
                target = strip_training_terminators(targets[begin + i])
                rows.append(dict(prediction=prediction, continuation_token_ids=ids,
                                 semantic_target=target, target_prefix_match=target_prefix_match(prediction, target)))
            del encoded, generated
    finally:
        tokenizer.padding_side = old
    return rows


class BoundaryReached(Exception):
    pass


@torch.inference_mode()
def capture_h18(model, tokenizer, prompts, batch_size=16, boundary=18, *, token_indices=None):
    """Prompt-only block-18 input = block-17 output, before block-18 LayerNorm."""
    if token_indices is not None:
        require(len(token_indices) == len(prompts), "Geometry token index count differs from prompts")
    states = np.empty((len(prompts), 1, model.config.n_embd), dtype=np.float32)
    positions = np.empty((len(prompts), 1), dtype=np.int32)
    result, selected = None, None

    def hook(_module, args):
        nonlocal result
        h = args[0]
        result = h[torch.arange(h.shape[0], device=h.device), selected].float().cpu().numpy()
        raise BoundaryReached()

    handle = model.transformer.h[boundary].register_forward_pre_hook(hook)
    old = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for start in range(0, len(prompts), batch_size):
            inputs = tokenizer(prompts[start:start + batch_size], padding=True,
                               add_special_tokens=True, return_tensors="pt").to(next(model.parameters()).device)
            require(inputs["input_ids"].shape[1] <= model.config.n_positions, "Geometry prompt exceeds context")
            mask = inputs["attention_mask"]
            inputs["position_ids"] = (mask.cumsum(-1) - 1).clamp_min(0)
            lengths = mask.sum(-1)
            selected = lengths - 1 if token_indices is None else torch.as_tensor(
                token_indices[start:start + len(lengths)], device=mask.device, dtype=torch.long)
            require(bool((selected >= 0).all()), "Empty geometry prompt")
            require(bool((selected < lengths).all()), "Geometry token index is outside the nonpadding prompt")
            result = None
            try:
                model.transformer(**inputs, use_cache=False, return_dict=True)
            except BoundaryReached:
                pass
            require(result is not None, "Boundary hook did not run")
            states[start:start + len(result), 0] = result
            positions[start:start + len(result), 0] = selected.cpu().numpy()
    finally:
        tokenizer.padding_side = old
        handle.remove()
    require(bool(np.isfinite(states).all()), "Nonfinite H18 states")
    return states, positions


def rewrite_subject_indices(tokenizer, prompts, subjects, max_length=1024):
    """Use the same audited subject lookup as online GPT-2 pre/post tracking."""
    require(len(prompts) == len(subjects) > 0, "Empty or unmatched rewrite prompts/subjects")
    positions = []
    for i, (prompt, subject) in enumerate(zip(prompts, subjects)):
        probe = build_probe(dict(case_id=i, prompt=prompt, subject=subject), tokenizer,
                            ("subject_last",), max_length)
        require(probe.prompt == prompt, "Rewrite geometry refuses implicit prompt whitespace changes")
        positions.append(int(probe.positions["subject_last"]))
    return positions


def capture_rewrite_h18(model, tokenizer, prompts, subjects, batch_size=16, boundary=18):
    """Fixed rewrite panel, subject_last H18; no target suffix or z injection."""
    indices = rewrite_subject_indices(tokenizer, prompts, subjects, model.config.n_positions)
    return capture_h18(model, tokenizer, prompts, batch_size=batch_size, boundary=boundary, token_indices=indices)


def geometry(h: np.ndarray, h0: np.ndarray):
    require(h.shape == h0.shape and h.ndim == 3 and h.shape[1] == 1, "H18 shape mismatch")
    a, b = h[:, 0].astype(np.float64), h0[:, 0].astype(np.float64)
    require(bool(np.isfinite(a).all() and np.isfinite(b).all()), "Nonfinite geometry")
    delta = a - b
    norm0 = np.linalg.norm(b, axis=-1)
    require(bool((norm0 > 0).all()), "Zero Base norm")
    p = (delta * b).sum(-1) / norm0**2
    q = np.linalg.norm(delta - p[:, None] * b, axis=-1) / norm0
    kappa = np.linalg.norm(a, axis=-1) / norm0
    error = float(np.max(np.abs(np.sqrt((1 + p)**2 + q**2) - kappa) / np.maximum(1, kappa)))
    require(error < 1e-12, f"p/q/kappa identity failed: {error}")
    raw = dict(p=p, q=q, kappa=kappa, absolute_norm_deviation=np.abs(kappa - 1),
               base_norm=norm0, post_norm=np.linalg.norm(a, axis=-1))
    summary = dict(mean_p=float(p.mean()), mean_q=float(q.mean()), mean_kappa=float(kappa.mean()),
                   mean_abs_norm_deviation=float(np.abs(kappa - 1).mean()), n_prompts=len(p),
                   geometry_identity_max_relative_error=error)
    return summary, raw


def canonical_panel(run_dir: Path, data_path: Path, limit: int):
    ordered = read(run_dir / "requests.json")
    raw = read(data_path)
    require(isinstance(raw, list) and isinstance(ordered, list), "Expected JSON request lists")
    require(len(ordered) == len({str(r["case_id"]) for r in ordered}), "Duplicate edit case IDs")
    require(len(ordered) >= limit, "Run has fewer cases than requested evaluation panel")
    if limit == 1000:
        require(len(ordered) == 1000, "Full replication requires exactly 1,000 edit requests")
    rank = {str(r["case_id"]): i + 1 for i, r in enumerate(ordered)}
    by_id = {str(r["case_id"]): r for r in ordered}
    panel = []
    for index, item in enumerate(raw[:limit]):
        case_id = str(item.get("case_id", item.get("id", index)))
        require(case_id in by_id, f"Canonical case missing from run: {case_id}")
        req = by_id[case_id]
        subject = str(req.get("subject", ""))
        prompt = item.get("src", item.get("prompt"))
        rephrase = item.get("rephrase", item.get("rephrase_prompt"))
        target = strip_training_terminators(item.get("alt", item.get("target_new")))
        locality = req.get("locality", {})
        pairs = [(str(v["prompt"]), str(v["ground_truth"])) for v in locality.values()]
        require(len(pairs) == 1, f"Expected one zsRE locality pair for case {case_id}")
        loc_prompt, loc_target = pairs[0]
        require(req["prompt"] == prompt and req.get("rephrase_prompt") == rephrase,
                f"Canonical edit/rephrase prompt mismatch: {case_id}")
        require(bool(subject) and subject in prompt, f"Canonical rewrite subject missing: {case_id}")
        if item.get("subject") is not None:
            require(subject == str(item["subject"]), f"Canonical subject mismatch: {case_id}")
        require(strip_training_terminators(req["target_new"]) == target and bool(target),
                f"Canonical semantic target mismatch: {case_id}")
        require(loc_prompt == item["loc"] and strip_training_terminators(loc_target) == item["loc_ans"].strip(),
                f"Canonical locality mismatch: {case_id}")
        panel.append(dict(case_id=case_id, source_index=index, edit_rank=rank[case_id], subject=subject,
                          rewrite_prompt=prompt, rephrase_prompt=rephrase,
                          target=str(req["target_new"]), semantic_target=target,
                          locality_prompt=loc_prompt, locality_target=loc_target))
    require(len(panel) == limit, "Canonical raw panel too short")
    return panel, ordered


def checkpoint_path(run_dir: Path, step: int) -> Path:
    candidates = list(run_dir.glob(f"step_*/edited_parameter_deltas.pt"))
    matches = [p for p in candidates if int(p.parent.name.split("_")[-1]) == step]
    require(len(matches) == 1, f"Expected one saved checkpoint for step {step}, got {matches}")
    return matches[0]


def validate_checkpoint(path: Path, run_dir: Path, ordered: list[dict], step: int, model_path: Path):
    meta = read(path.with_suffix(".json"))
    expected = dict(format="easyedit-edited-parameters", format_version=1,
                    storage_mode="parameter_deltas", edit_count=step,
                    rewrite_layers=list(LAYERS), parameter_names=list(NAMES),
                    requests_prefix_count=step,
                    requests_sha256=digest(run_dir / "requests.json"),
                    requests_prefix_sha256=runner_json_hash(ordered[:step]))
    for key, value in expected.items():
        require(meta.get(key) == value, f"Checkpoint {key} mismatch: {path}")
    require(meta["editing_method"] in ("MEMIT", "AlphaEdit"), "Unexpected editor")
    allowed = {"gpt2-xl", "openai-community/gpt2-xl", str(model_path), str(model_path.resolve())}
    require(meta["base_model"] in allowed, "Checkpoint Base model mismatch")
    require(Path(meta["requests_path"]).resolve() == (run_dir / "requests.json").resolve(),
            "Checkpoint request path mismatch")
    hparams = Path(meta["hparams_path"])
    require(hparams.is_file() and digest(hparams) == meta["hparams_sha256"], "Checkpoint hparams binding mismatch")
    cfgpath = run_dir / "run_config.json"
    if cfgpath.exists() and meta.get("run_fingerprint") is not None:
        require(read(cfgpath).get("fingerprint") == meta["run_fingerprint"], "Run fingerprint mismatch")
    return dict(checkpoint=record(path), sidecar=record(path.with_suffix(".json")), metadata=meta)


def restore_checkpoint(model, pristine: dict, path: Path, identity: dict):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload.get("format") == "easyedit-edited-parameters" and payload.get("format_version") == 1,
            "Checkpoint payload format mismatch")
    require(payload.get("metadata") == identity["metadata"], "Payload/sidecar metadata mismatch")
    require(tuple(payload["state_dict"]) == NAMES, "Unexpected checkpoint parameter names/order")
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, delta in payload["state_dict"].items():
            parameter = named[name]
            expected = identity["metadata"]["tensor_metadata"][name]
            require(list(delta.shape) == list(parameter.shape) == expected["shape"], f"Shape mismatch: {name}")
            require(delta.dtype == parameter.dtype == pristine[name].dtype, f"Dtype mismatch: {name}")
            require(str(delta.dtype) == expected["dtype"] and delta.numel() == expected["numel"], "Tensor metadata mismatch")
            require(bool(torch.isfinite(delta).all()), f"Nonfinite checkpoint delta: {name}")
            parameter.copy_(pristine[name].to(parameter.device) + delta.to(parameter.device))


def evaluate_state(model, tokenizer, panel, step, baseline, args):
    teacher = {}
    for family in ("rewrite", "rephrase", "locality"):
        prompts = [r[f"{family}_prompt"] for r in panel]
        targets = [r["locality_target"] if family == "locality" else r["target"] for r in panel]
        teacher[family] = teacher_forced(model, tokenizer, prompts, targets, args.eval_batch_size)
        print(f"[teacher-forced] step={step} family={family} n={len(panel)}", flush=True)
    generation = {}
    if not args.skip_generation and step in args.generation_steps:
        for family in ("rewrite", "rephrase"):
            generation[family] = generate_answers(model, tokenizer,
                [r[f"{family}_prompt"] for r in panel], [r["semantic_target"] for r in panel],
                args.generation_batch_size, args.max_new_tokens)
            print(f"[generation] step={step} family={family} n={len(panel)}", flush=True)
    cases = []
    for i, item in enumerate(panel):
        loc = teacher["locality"][i]
        base = loc if baseline is None else baseline["cases"][i]["teacher_forced"]["locality"]
        require(base["target_token_ids"] == loc["target_token_ids"], "Base/post locality target span differs")
        require(base["full_nonpadding_ids"] == loc["full_nonpadding_ids"], "Base/post locality input differs")
        require(len(base["predicted_token_ids"]) == len(loc["predicted_token_ids"]) > 0, "Locality prediction length mismatch")
        row = dict(case_id=item["case_id"], edit_rank=item["edit_rank"], edited=item["edit_rank"] <= step,
                   tf_eff=teacher["rewrite"][i]["token_accuracy"],
                   tf_gen=teacher["rephrase"][i]["token_accuracy"],
                   loc=float(np.equal(base["predicted_token_ids"], loc["predicted_token_ids"]).mean()),
                   teacher_forced={k: v[i] for k, v in teacher.items()},
                   base_locality_predicted_token_ids=base["predicted_token_ids"])
        if generation:
            row.update(generation={k: v[i] for k, v in generation.items()},
                       generation_eff=float(generation["rewrite"][i]["target_prefix_match"]),
                       generation_gen=float(generation["rephrase"][i]["target_prefix_match"]))
        cases.append(row)
    metrics = ["tf_eff", "tf_gen", "loc", "generation_eff", "generation_gen"]
    summaries = {}
    for name, subset in (("fixed_panel", cases), ("edited_prefix", [r for r in cases if r["edited"]]),
                         ("first100_edited", [r for r in cases if r["edited"] and r["edit_rank"] <= 100])):
        summaries[name] = dict(n_cases=len(subset), **{m: float(np.mean([r[m] for r in subset]))
                                                    if subset and m in subset[0] else None for m in metrics})
    return dict(cases=cases, summary=summaries, generation_evaluated=bool(generation))


def load_model_and_tokenizer(model_path: Path, device="cuda:0", dtype="float32"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        torch_dtype=getattr(torch, dtype), attn_implementation="eager").to(device).eval()
    model.requires_grad_(False)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=DATA_ROOT / 'zsre/zsre_3k.json')
    parser.add_argument("--output-root", type=Path, required=True,
                        help="A dedicated directory for ONE editor/method run; never share between workers")
    parser.add_argument("--steps", type=int, nargs="+", default=list(STEPS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--method", default=None)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--limit", type=int, default=1000, help="Preview only when less than 1,000")
    parser.add_argument("--capture-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--generation-steps", type=int, nargs="+", default=[1000],
                        help="Generation is endpoint-only by default; teacher forcing and LOC cover every saved step")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    require(0 < args.limit <= 1000, "limit must be between 1 and 1,000")
    require(args.steps == sorted(set(args.steps)) and min(args.steps) > 0, "Steps must be positive, sorted, unique")
    require(min(args.capture_batch_size, args.eval_batch_size, args.generation_batch_size, args.max_new_tokens) > 0,
            "Batch sizes and generation length must be positive")
    args.run_dir = args.run_dir.resolve()
    args.model_path = args.model_path.resolve()
    panel, ordered = canonical_panel(args.run_dir, args.data_path, args.limit)
    require(max(args.steps) <= len(ordered), "Checkpoint exceeds source run length")
    paths = {s: checkpoint_path(args.run_dir, s) for s in args.steps}
    identities = {s: validate_checkpoint(p, args.run_dir, ordered, s, args.model_path) for s, p in paths.items()}
    editors = {i["metadata"]["editing_method"] for i in identities.values()}
    require(len(editors) == 1, "Mixed editors in one run")
    editor = next(iter(editors))
    method = args.method or args.run_dir.name
    model_config = read(args.model_path / "config.json")
    require(model_config.get("model_type") == "gpt2" and model_config.get("n_embd") == 1600
            and model_config.get("n_layer") == 48, "Expected GPT-2 XL architecture (48 blocks, 1600 hidden)")
    binding = dict(schema_version=SCHEMA, run_dir=str(args.run_dir), editor=editor, method=method,
                   requests=record(args.run_dir / "requests.json"), data=record(args.data_path),
                   model_path=str(args.model_path), model_config=record(args.model_path / "config.json"),
                   panel=panel, dtype=args.dtype, steps=args.steps, preview=args.limit != 1000,
                   protocol=dict(geometry="prompt-only H18 = block17 output = block18 pre-LayerNorm input; prompt_last",
                                 rewrite_geometry="Fixed canonical rewrite panel, subject_last H18, before block18 LayerNorm; raw prompt only, no target suffix",
                                 rewrite_reference="The same case, prompt and subject token in pretrained Base; not the current pre-edit model or optimized target z",
                                 position_ids="attention_mask.cumsum(-1)-1 clamped to zero for padding",
                                 teacher_forcing="joint prompt + ASCII space + stored target; complete genuine EOS retained",
                                 locality="fixed canonical case-macro same-target-span Base/post argmax agreement",
                                 generation="greedy normalized target-prefix; training terminators removed from target",
                                 generation_max_new_tokens=args.max_new_tokens,
                                 generation_enabled=not args.skip_generation,
                                 generation_steps=[] if args.skip_generation else args.generation_steps,
                                 capture_batch_size=args.capture_batch_size, eval_batch_size=args.eval_batch_size,
                                 generation_batch_size=args.generation_batch_size),
                   checkpoint_identities={str(s): i for s, i in identities.items()},
                   code=record(Path(__file__)))
    fingerprint = json_hash(binding)
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan = args.output_root / "source_audit.json"
    if plan.exists():
        require(read(plan)["fingerprint"] == fingerprint, "Existing output binding differs; use another output root")
    else:
        write_json(plan, dict(fingerprint=fingerprint, **binding))
    if args.audit_only:
        print(json.dumps(dict(audit_complete=True, n_cases=len(panel), checkpoints=len(paths), fingerprint=fingerprint)))
        return
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device, args.dtype)
    named = dict(model.named_parameters())
    pristine = {n: named[n].detach().cpu().clone() for n in NAMES}
    baseline = None
    base_h = None
    base_positions = None
    base_rewrite_h = None
    base_rewrite_positions = None
    summary_rows = []
    for step in [0, *args.steps]:
        destination = args.output_root / f"step_{step:04d}"
        complete_path = destination / "complete.json"
        if complete_path.exists():
            done = read(complete_path)
            require(done["fingerprint"] == fingerprint and done["complete"], "Completed output identity differs")
            for item in done["files"].values():
                require(record(destination / Path(item["path"]).name) == item, "Completed artifact hash differs")
            performance = read(destination / "performance.json")
            if step == 0:
                baseline = performance
                with np.load(destination / "raw_h18.npz", allow_pickle=False) as raw_h:
                    base_h, base_positions = raw_h["h18"].copy(), raw_h["token_positions"].copy()
                with np.load(destination / "raw_rewrite_h18.npz", allow_pickle=False) as raw_h:
                    base_rewrite_h, base_rewrite_positions = raw_h["h18"].copy(), raw_h["token_positions"].copy()
            summary_rows.append(done["summary"])
            print(f"[reuse] step={step}", flush=True)
            continue
        started = time.monotonic()
        if step:
            restore_checkpoint(model, pristine, paths[step], identities[step])
        h, positions = capture_h18(model, tokenizer, [r["locality_prompt"] for r in panel], args.capture_batch_size)
        if step == 0:
            base_h, base_positions = h.copy(), positions.copy()
        require(np.array_equal(positions, base_positions), "Base/post geometry token coordinates differ")
        scalar, per_case = geometry(h, base_h)
        coords = dict(case_ids=np.asarray([r["case_id"] for r in panel]),
                      positions=np.asarray(["prompt_last"]), token_positions=positions,
                      valid_positions=np.ones_like(positions, dtype=bool))
        write_npz(destination / "raw_h18.npz", h18=h, **coords)
        write_npz(destination / "per_case_geometry.npz", **per_case, case_ids=coords["case_ids"])
        rewrite_h, rewrite_positions = capture_rewrite_h18(
            model, tokenizer, [r["rewrite_prompt"] for r in panel],
            [r["subject"] for r in panel], args.capture_batch_size)
        if step == 0:
            base_rewrite_h, base_rewrite_positions = rewrite_h.copy(), rewrite_positions.copy()
        require(np.array_equal(rewrite_positions, base_rewrite_positions), "Base/post rewrite subject coordinates differ")
        rewrite_scalar, rewrite_per_case = geometry(rewrite_h, base_rewrite_h)
        rewrite_coords = dict(case_ids=coords["case_ids"], positions=np.asarray(["subject_last"]),
                              token_positions=rewrite_positions, valid_positions=np.ones_like(rewrite_positions, dtype=bool))
        write_npz(destination / "raw_rewrite_h18.npz", h18=rewrite_h, **rewrite_coords)
        write_npz(destination / "per_case_rewrite_geometry.npz", **rewrite_per_case, case_ids=coords["case_ids"])
        performance = evaluate_state(model, tokenizer, panel, step, baseline, args)
        performance.update(complete=True, fingerprint=fingerprint, step=step, editor=editor, method=method,
                           protocol=binding["protocol"], units="fraction (multiply by 100 for percent)")
        if step == 0:
            baseline = performance
        write_json(destination / "performance.json", performance)
        row = dict(editor=editor, method=method, edit_count=step, **scalar,
                   **{f"rewrite_{k}": v for k, v in rewrite_scalar.items()},
                   **{f"fixed_{k}": v for k, v in performance["summary"]["fixed_panel"].items()},
                   **{f"prefix_{k}": v for k, v in performance["summary"]["edited_prefix"].items()})
        summary_rows.append(row)
        files = {name: record(destination / name) for name in (
            "raw_h18.npz", "per_case_geometry.npz", "raw_rewrite_h18.npz", "per_case_rewrite_geometry.npz", "performance.json")}
        write_json(complete_path, dict(complete=True, fingerprint=fingerprint, summary=row, files=files,
                                       seconds=time.monotonic() - started))
        print(f"[checkpoint-complete] {json.dumps(row)} seconds={time.monotonic() - started:.1f}", flush=True)
    csv_path = args.output_root / "geometry_performance_by_checkpoint.csv"
    tmp = csv_path.with_suffix(".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    tmp.replace(csv_path)
    runtime = dict(torch=torch.__version__, device=str(next(model.parameters()).device),
                   dtype=str(next(model.parameters()).dtype))
    if next(model.parameters()).is_cuda:
        runtime.update(max_memory_allocated_gib=torch.cuda.max_memory_allocated(model.device) / 2**30,
                       max_memory_reserved_gib=torch.cuda.max_memory_reserved(model.device) / 2**30)
    write_json(args.output_root / "complete.json", dict(complete=True, fingerprint=fingerprint,
               preview=args.limit != 1000, n_cases=len(panel), n_checkpoints=len(args.steps),
               steps=args.steps, summary_csv=record(csv_path), runtime=runtime))


if __name__ == "__main__":
    main()
