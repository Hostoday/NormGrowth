#!/usr/bin/env python3
"""Missing-only fixed-checkpoint full1000 teacher-forced locality evaluation."""
from __future__ import annotations

# Release-only filesystem defaults; computation below follows the source snapshot.
import sys as _bng_sys
from pathlib import Path as _BNGPath
_bng_sys.path.insert(0, str(_BNGPath(__file__).resolve().parents[1]))
from model_code.paths import RESEARCH_ROOT, OUTPUT_ROOT, DATA_ROOT, LLAMA_MODEL

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluate"))
RUNS = OUTPUT_ROOT / '_Edited_Model'
STANDARD = RUNS / "method_rms_trajectory_n1000_artifacts_v1/AlphaEdit"
SOURCES = {
    "sphere": STANDARD / "alphaedit_zsre_bs1_n1000_sphere_seed42",
    "hiddennorm": RUNS / "method_rms_trajectory_n1000_hiddennorm_mse_rho_lam1p0_l8_v1/AlphaEdit/alphaedit_zsre_bs1_n1000_rgr_seed42",
    "sadr": STANDARD / "alphaedit_zsre_bs1_n1000_sadr_seed42",
}
MODEL = LLAMA_MODEL
OUT = OUTPUT_ROOT / '_Analysis_Cross_Layer/alphaedit_raw_branch_locality_base_sphere_hn_sadr_step1000_n1000_l32_v1/locality_evaluation'
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


def record(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(8 << 20), b""):
            h.update(b)
    return dict(path=str(path.resolve()), sha256=h.hexdigest(), bytes=path.stat().st_size)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def evaluate(states, output, batch_size):
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from diagnostics.analyze_residual_spectrum import load_model
    from diagnostics.analyze_edited_a_m_o_reference_cosines import validate_checkpoint_metadata, validate_hiddennorm_condition
    from eval_hf_easyedit import collect_group_items, compute_locality_outputs

    torch.set_num_threads(4)
    torch.manual_seed(42)
    request_path = SOURCES["sphere"] / "requests.json"
    requests = json.loads(request_path.read_text())
    if len(requests) != 1000 or [int(r["case_id"]) for r in requests] != list(range(1000)):
        raise ValueError("Expected matched canonical 1000 requests")
    request_record = record(request_path)
    items = collect_group_items(requests, "locality")
    if len(items) != 1000 or [i[0] for i in items] != list(range(1000)):
        raise ValueError("Expected exactly one locality pair per case")
    model = load_model(str(MODEL), "bfloat16", "cuda:0", False, "eager")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, use_fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    hparams = SimpleNamespace(max_length=256, alg_name="AlphaEdit", device=0)
    named = dict(model.named_parameters())
    for state in states:
        destination = output / f"{state}.json"
        if destination.exists():
            raise ValueError(f"Already exists: {destination}")
        started = time.monotonic()
        overlay, checkpoint_record = None, None
        if state != "base":
            folder = SOURCES[state]
            cfg = json.loads((folder / "run_config.json").read_text())
            if state == "hiddennorm":
                validate_hiddennorm_condition(cfg)
            if record(folder / "requests.json")["sha256"] != request_record["sha256"]:
                raise ValueError("Mismatched edit requests")
            checkpoint = folder / "step_1000/edited_parameter_deltas.pt"
            checkpoint_record = record(checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            metadata = payload["metadata"]
            validate_checkpoint_metadata(metadata, checkpoint_path=checkpoint,
                                         expected_edit_count=1000, expected_rewrite_layers=[4, 5, 6, 7, 8])
            if metadata != json.loads(checkpoint.with_suffix(".json").read_text()):
                raise ValueError("Checkpoint/sidecar mismatch")
            if metadata["run_fingerprint"] != cfg["fingerprint"]:
                raise ValueError("Checkpoint/run mismatch")
            with torch.no_grad():
                overlay = {name: (named[name].detach() + delta.to(device=named[name].device,
                            dtype=named[name].dtype)).detach() for name, delta in payload["state_dict"].items()}
            del payload

        class FixedModel:
            def __call__(self, *args, **kwargs):
                if overlay is None:
                    return model(*args, **kwargs)
                return torch.func.functional_call(model, overlay, args, kwargs, strict=False)

        outputs = {}
        with torch.inference_mode():
            for start in range(0, len(items), 64):
                outputs.update(compute_locality_outputs(
                    model=FixedModel(), model_name=MODEL_NAME, hparams=hparams,
                    tokenizer=tokenizer, locality_items=items[start:start+64], device=0,
                    batch_size=batch_size))
                print(f"[{state}] {min(start+64,1000)}/1000 {time.monotonic()-started:.1f}s", flush=True)
        if set(outputs) != set(range(1000)):
            raise ValueError("Missing locality cases")
        cases = []
        for index, request in enumerate(requests):
            predictions = outputs[index]
            if any(not tokens for pairs in predictions.values() for tokens in pairs):
                raise ValueError("Empty target predictions")
            cases.append(dict(case_id=request["case_id"], outputs=predictions))
        write_json(destination, dict(complete=True, state=state, n_cases=1000,
                   checkpoint_edit_count=0 if state == "base" else 1000,
                   checkpoint=checkpoint_record, requests=request_record, batch_size=batch_size,
                   dtype="bfloat16", attn_implementation="eager", cases=cases,
                   protocol="EasyEdit teacher-forced locality target-token argmax; no generation",
                   checkpoint_semantics="Base plus cumulative BF16 parameter delta, via functional_call",
                   code=record(Path(__file__)), evaluator=record(ROOT / "evaluate/eval_hf_easyedit.py"),
                   target_slicing=record(ROOT / "EasyEdit/easyeditor/evaluate/evaluate_utils.py"),
                   seconds=time.monotonic()-started))
        del overlay, outputs
        torch.cuda.empty_cache()
        print(f"[{state}] saved {destination}", flush=True)


def summarize(output):
    import numpy as np
    data = {s: json.loads((output / f"{s}.json").read_text()) for s in ("base", *SOURCES)}
    reference = data["base"]
    methods, comparisons = {}, {}
    for state, result in data.items():
        if not result["complete"] or result["n_cases"] != 1000 or result["requests"]["sha256"] != reference["requests"]["sha256"]:
            raise ValueError("Mismatched locality sources")
        scores = []
        for base, current in zip(reference["cases"], result["cases"]):
            if base["case_id"] != current["case_id"] or base["outputs"].keys() != current["outputs"].keys():
                raise ValueError("Mismatched case/pair coordinates")
            pair_scores = []
            for key, pairs in base["outputs"].items():
                post = current["outputs"][key]
                if len(pairs) != len(post):
                    raise ValueError("Pair count mismatch")
                for x, y in zip(pairs, post):
                    if len(x) != len(y) or not x:
                        raise ValueError("Target-token count mismatch")
                    pair_scores.append(float(np.mean(np.equal(x, y))))
            scores.append(dict(case_id=base["case_id"], locality_acc=float(np.mean(pair_scores))))
        methods[state] = dict(locality_acc=float(np.mean([r["locality_acc"] for r in scores])),
                              n_cases=1000, checkpoint_sha256=result["checkpoint"]["sha256"] if result["checkpoint"] else None,
                              cases=scores, source=record(output / f"{state}.json"))
        if state != "base":
            old = json.loads((SOURCES[state] / "step_1000/evaluation.json").read_text())
            previous = {str(r["case_id"]): r["locality_acc"] for r in old["locality"]}
            new = {str(r["case_id"]): r["locality_acc"] for r in scores}
            before = float(np.mean(list(previous.values())))
            after = float(np.mean([new[k] for k in previous]))
            comparisons[state] = dict(n_cases=len(previous), original_live_state=before,
                                      reconstructed_checkpoint=after, absolute_mean_difference=abs(after-before),
                                      changed_case_count=sum(new[k] != v for k,v in previous.items()))
    result = dict(complete=True, n_cases=1000, requests_sha256=reference["requests"]["sha256"],
                  methods=methods, protocol="Case-macro mean of teacher-forced Base target-token argmax agreement",
                  historical_fixed50_comparison=comparisons)
    write_json(output / "summary.json", result)
    print(json.dumps(dict(scores={k:v['locality_acc'] for k,v in methods.items()}, fixed50=comparisons)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", nargs="+", choices=["base", *SOURCES])
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.output_root)
    elif args.states:
        evaluate(args.states, args.output_root, args.batch_size)
    else:
        parser.error("--states or --summarize is required")
