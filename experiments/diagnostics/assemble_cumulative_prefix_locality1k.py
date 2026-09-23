#!/usr/bin/env python3
"""Assemble audited cumulative EasyEdit rewrite/rephrase and fixed-1K locality.

Rewrite/rephrase are the historical teacher-forced accuracies over every edited
case, not the generated-answer metrics present in the locality source bundles.
All reusable measurements retain per-metric source hashes and checkpoint identity.
Unavailable locality checkpoints are reported explicitly rather than imputed.
"""

from __future__ import annotations

# Release-only filesystem defaults; computation below follows the source snapshot.
import sys as _bng_sys
from pathlib import Path as _BNGPath
_bng_sys.path.insert(0, str(_BNGPath(__file__).resolve().parents[1]))
from model_code.paths import RESEARCH_ROOT, OUTPUT_ROOT, DATA_ROOT, LLAMA_MODEL

import argparse
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EDITORS = ("MEMIT", "AlphaEdit")
METHODS = ("Native", "ENCORE", "NAS", "SADR", "SPHERE", "HN", "RGR")
STEPS = (0, 10, 20, 50, 100, 150, 200, 250, 300, 500, 750, 1000)
REQUESTS_SHA = "8ba609099c9e96ec0683d50ce152034fbd8509ee5aef11d7f77eb36642bf04f7"
DEFAULT_OUTPUT = OUTPUT_ROOT / 'paper_a_cumulative_prefix_locality1000_n1000_v1'
LEGACY_LOCALITY = OUTPUT_ROOT / 'generation_easyedit_locality_fixed1000_n1000_v3'
HN_LOCALITY = OUTPUT_ROOT / 'paper_a_immediate_v1/01_hn_generation'


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_run(editor: str, method: str) -> Path:
    if method == "Native":
        family, variant = "gain_trajectory_n1000_baselines_artifacts_v1", "baseline"
    elif method == "HN":
        family = "method_rms_trajectory_n1000_hiddennorm_mse_rho_lam1p0_l8_v1"
        variant = "rgr"
    else:
        family, variant = "method_rms_trajectory_n1000_artifacts_v1", method.lower()
    return OUTPUT_ROOT / '_Edited_Model' / family / editor / f"{editor.lower()}_zsre_bs1_n1000_{variant}_seed42"


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def checked_score(value: Any, context: str) -> float:
    score = float(value)
    require(math.isfinite(score) and 0 <= score <= 1, f"invalid score {context}: {value}")
    return score


def scores_match(a: float | None, b: float | None, context: str) -> None:
    require((a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-12), context)


def validate_historical(path: Path, step: int, config: dict, requests: list) -> tuple[dict, dict]:
    payload = read_json(path)
    require(payload["edit_count"] == step, f"historical edit count: {path}")
    require(payload["fingerprint"] == config["fingerprint"], f"historical run fingerprint: {path}")
    results, summary = {}, {}
    for kind in ("rewrite", "rephrase"):
        records = payload[kind] if step else []
        require([r["case_id"] for r in records] == [r["case_id"] for r in requests[:step]], f"{kind} prefix IDs: {path}")
        key = f"{kind}_acc"
        values = [checked_score(r[key], f"{path}/{kind}/{r['case_id']}") for r in records]
        score = average(values)
        if step:
            require(payload["summary"][f"{kind}_count"] == step, f"{kind} count: {path}")
            scores_match(score, payload["summary"][key], f"{kind} summary mismatch: {path}")
        results[kind] = [{"case_id": r["case_id"], key: r[key]} for r in records]
        summary[key], summary[f"{kind}_count"] = score, len(records)
    return {**payload, **results}, summary


def validate_locality_protocol(payload: dict, path: Path) -> None:
    protocol = payload["protocol"]["locality"]
    expected = {
        "mode": "easyedit_teacher_forced_base_post_argmax_token_agreement",
        "reference": "initial_unedited_base_model",
        "prompt_transform": "none",
        "separator": "single ASCII space",
        "target_add_special_tokens": False,
        "aggregation": "case_macro_of_positionwise_post_equals_base",
        "batch_size": 8,
        "max_length": 256,
        "padding_side": "left",
        "batch_order": "request_order",
    }
    for key, value in expected.items():
        require(protocol.get(key) == value, f"locality protocol {key}: {path}")
    panel = payload["panel"]["locality"]
    require(panel == {"type": "fixed_prefix", "start": 0, "end": 1000, "num_cases": 1000}, f"locality panel: {path}")


def validate_locality_records(payload: dict, requests: list, reference: list | None, path: Path) -> list[dict]:
    validate_locality_protocol(payload, path)
    records = payload["records"]
    require([r["case_id"] for r in records] == [r["case_id"] for r in requests], f"locality case IDs: {path}")
    output = []
    for index, (record, request) in enumerate(zip(records, requests)):
        require(record["request_index"] == index, f"locality request index: {path}/{index}")
        pairs = record["measurements"]
        # This ZsRE cohort has exactly one scalar neighborhood pair per case.
        require(len(pairs) == len(request["locality"]) == 1, f"locality pair count: {path}/{index}")
        pair = pairs[0]
        expected = request["locality"][pair["key"]]
        require(pair["pair_index"] == 0, f"pair index: {path}/{index}")
        require(pair["prompt"] == pair["stored_prompt"] == expected["prompt"], f"locality prompt: {path}/{index}")
        require(pair["target"] == expected["ground_truth"], f"locality target: {path}/{index}")
        base, post, targets = pair["base_predicted_token_ids"], pair["post_predicted_token_ids"], pair["target_token_ids"]
        require(bool(targets) and len(base) == len(post) == len(targets), f"locality token lengths: {path}/{index}")
        if reference is not None:
            canonical = reference[index]["measurements"][0]
            require(base == canonical["base_predicted_token_ids"], f"locality Base predictions differ: {path}/{index}")
            require(targets == canonical["target_token_ids"], f"locality target tokens differ: {path}/{index}")
        hits = [a == b for a, b in zip(base, post)]
        require(hits == pair["token_correct"], f"locality token correctness: {path}/{index}")
        score = sum(hits) / len(hits)
        scores_match(score, checked_score(pair["token_accuracy"], str(path)), f"locality pair accuracy: {path}/{index}")
        output.append({"case_id": record["case_id"], "locality_acc": score})
    scores_match(average([r["locality_acc"] for r in output]), payload["summary"]["specificity"], f"locality summary mean: {path}")
    require(payload["summary"]["specificity_case_count"] == payload["summary"]["locality_panel_size"] == 1000, f"locality summary count: {path}")
    return output


@lru_cache(maxsize=2)
def validate_native_replay(replay: Path, run: Path) -> dict:
    """Accept only a full replay whose independent parity evidence is complete."""
    report_path = replay / "replay_report.json"
    provenance_path = replay / "provenance.json"
    outer_path = replay / "outer_parity.jsonl"
    report, provenance = read_json(report_path), read_json(provenance_path)
    require(report.get("status") == "exact_endpoint_replay_validated", f"Native replay is not fully validated: {report_path}")
    require(report.get("source_endpoint_equal") is True, f"Native endpoint not equal: {report_path}")
    require(report.get("requested_edits") == report.get("last_attempted_edit") == report.get("last_completed_checkpoint") == 1000,
            f"Native replay is not complete through 1000: {report_path}")
    require(report.get("editor") == run.parent.name, f"Native replay editor mismatch: {report_path}")
    require(Path(report["source_run"]).resolve() == run.resolve(), f"Native replay source mismatch: {report_path}")
    require(not report.get("failure") and not report.get("traceback"), f"Native replay failure: {report_path}")
    names = [f"model.layers.{layer}.mlp.down_proj.weight" for layer in range(4, 9)]
    require(report.get("endpoint_tensor_equal") == {name: True for name in names}, f"Native endpoint tensor equality missing: {report_path}")
    metric_keys = {"locality_acc", "rewrite_acc", "rephrase_acc"}
    require([r["edit_count"] for r in report["steps"]] == list(STEPS[1:]), f"Native replay historical checkpoints missing: {report_path}")
    for row in report["steps"]:
        require(set(row["metric_absolute_errors"]) == metric_keys, f"Native replay metric keys: {report_path}")
        require(row.get("per_case_mismatch_counts") == {"locality": 0, "rewrite": 0, "rephrase": 0}, f"Native replay per-case metric parity missing: {report_path}")
        old = read_json(run / f"step_{row['edit_count']:03d}" / "evaluation.json")
        for key in metric_keys:
            error = row["metric_absolute_errors"][key]
            require(math.isfinite(error) and 0 <= error <= 1e-12, f"Native historical metric error: {report_path}/{row['edit_count']}/{key}")
            scores_match(row["historical_summary"][key], old["summary"][key], f"Native historical metric source changed: {report_path}")
            scores_match(row["replayed_summary"][key], old["summary"][key], f"Native historical metric parity mismatch: {report_path}")
    require(Path(provenance["source_run"]).resolve() == run.resolve(), f"Native provenance source mismatch: {provenance_path}")
    source_config = read_json(run / "run_config.json")
    endpoint_metadata = read_json(run / "step_1000/edited_parameter_deltas.json")
    require(provenance["source_run_fingerprint"] == source_config["fingerprint"] == endpoint_metadata["run_fingerprint"], f"Native provenance run fingerprint: {provenance_path}")
    require(provenance["source_request_fingerprint"] == endpoint_metadata["request_fingerprint"], f"Native provenance request fingerprint: {provenance_path}")
    require(provenance["source_requests_sha256"] == REQUESTS_SHA == file_record(replay / "requests.json")["sha256"], f"Native replay requests mismatch: {provenance_path}")
    require(provenance["source_config_sha256"] == file_record(run / "run_config.json")["sha256"], f"Native replay source config mismatch: {provenance_path}")
    require(provenance["source_endpoint_metadata"] == endpoint_metadata, f"Native source endpoint metadata mismatch: {provenance_path}")
    require(provenance["source_endpoint_sha256"] == file_record(run / "step_1000/edited_parameter_deltas.pt")["sha256"], f"Native source endpoint file changed: {provenance_path}")
    fingerprint = provenance["replay_fingerprint"]
    fingerprint_payload = {key: value for key, value in provenance.items() if key != "replay_fingerprint"}
    require(hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest() == fingerprint,
            f"Native replay provenance fingerprint is invalid: {provenance_path}")
    require(report["replay_fingerprint"] == fingerprint, f"Native replay report/provenance fingerprint mismatch: {report_path}")
    replay_config = read_json(replay / "run_config.json")
    require(replay_config["fingerprint"] == source_config["fingerprint"], f"Native copied source config fingerprint mismatch: {replay}")
    require(file_record(replay / "run_config.json")["sha256"] == provenance["source_config_sha256"], f"Native copied source config hash mismatch: {replay}")
    require((replay / "EXACT_REPLAY_COMPLETE").read_text().strip() == fingerprint, f"Native replay completion marker missing: {replay}")
    outer_rows = [json.loads(line) for line in outer_path.read_text().splitlines() if line.strip()]
    require(len(outer_rows) == 5000, f"Native outer parity row count: {outer_path}")
    require({(r["step"], r["layer"]) for r in outer_rows} == {(step, layer) for step in range(1, 1001) for layer in range(4, 9)},
            f"Native outer parity coverage: {outer_path}")
    for row in outer_rows:
        for key in ("k_star", "target_error", "distributed_residual", "delta_w_k"):
            require(row[key].get("exact") is True, f"Native outer parity failed: {outer_path}/{row['step']}/{row['layer']}/{key}")
            require(row[key].get("mismatched_elements") == 0, f"Native outer parity mismatch count: {outer_path}/{row['step']}/{row['layer']}/{key}")
    return {"recovery": "cached historical target replay with complete outer parity and exact final BF16 delta equality",
            "source_run": str(run.resolve()), "replay_report": file_record(report_path), "replay_provenance": file_record(provenance_path),
            "outer_parity": file_record(outer_path), "outer_parity_rows": len(outer_rows),
            "validated_historical_checkpoints": list(STEPS[1:]), "source_endpoint_tensor_equal": report["endpoint_tensor_equal"],
            "replay_fingerprint": fingerprint}


def validate_checkpoint(payload: dict, run: Path, step: int, config: dict, path: Path,
                        *, method: str, output_root: Path) -> dict:
    checkpoint = run / f"step_{step:03d}" / "edited_parameter_deltas.pt"
    recovery = None
    if not checkpoint.is_file() and method == "Native":
        replay = output_root / "native_replay/full" / run.parent.name / run.name
        recovered = replay / f"step_{step:03d}" / "edited_parameter_deltas.pt"
        require(Path(payload["checkpoint"]["path"]).resolve() == recovered.resolve(), f"Native recovered checkpoint must use the explicit validated replay path: {path}")
        require(Path(payload["run"]["source_run_dir"]).resolve() == replay.resolve(), f"Native recovered evaluator source path mismatch: {path}")
        recovery = validate_native_replay(replay, run)
        checkpoint = recovered
    require(checkpoint.is_file(), f"missing checkpoint for available locality: {checkpoint}")
    sidecar = checkpoint.with_suffix(".json")
    metadata = read_json(sidecar)
    recorded = payload["checkpoint"]
    # The artifacts moved under _Edited_Model after legacy evaluation. Resolve
    # identity through run name, manifest fields, byte size and preserved mtime.
    require(Path(recorded["path"]).parent.parent.name == run.name, f"checkpoint source run: {path}")
    require(Path(recorded["path"]).parent.parent.parent.name == run.parent.name, f"checkpoint editor: {path}")
    for key in ("storage_mode", "edit_count", "request_fingerprint", "parameter_names"):
        require(metadata[key] == recorded[key], f"checkpoint {key}: {path}")
    expected_fingerprint = recovery["replay_fingerprint"] if recovery is not None else config["fingerprint"]
    require(metadata["run_fingerprint"] == expected_fingerprint, f"checkpoint run fingerprint: {path}")
    if recovery is not None:
        endpoint_metadata = read_json(run / "step_1000/edited_parameter_deltas.json")
        require(Path(metadata["source_run"]).resolve() == run.resolve(), f"Native recovered metadata source: {path}")
        require(metadata["source_run_fingerprint"] == config["fingerprint"], f"Native recovered metadata source fingerprint: {path}")
        require(metadata["source_request_fingerprint"] == metadata["request_fingerprint"] == endpoint_metadata["request_fingerprint"], f"Native recovered metadata request fingerprint: {path}")
        require(metadata["source_requests_sha256"] == REQUESTS_SHA, f"Native recovered metadata requests SHA: {path}")
    require(metadata["edit_count"] == step, f"checkpoint step: {path}")
    stat = checkpoint.stat()
    require(stat.st_size == recorded["size"] and stat.st_mtime_ns == recorded["mtime_ns"], f"checkpoint size/mtime identity: {path}")
    return {"path": str(checkpoint.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sidecar": file_record(sidecar), "run_fingerprint": metadata["run_fingerprint"],
            "request_fingerprint": metadata["request_fingerprint"],
            "identity_check": "same run/editor, sidecar fields, byte size, and preserved nanosecond mtime",
            **({"validated_recovery": recovery} if recovery is not None else {})}


def source_index(roots: list[Path]) -> dict[tuple[str, str, int], Path]:
    output = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("locality.json")):
            if path.parent.name == "base":
                continue
            payload = read_json(path)
            editor, method = payload["run"]["label"].split("/", 1)
            method = "Native" if method.lower() == "native" else method
            step = int(payload["checkpoint"]["edit_count"])
            if editor in EDITORS and method in METHODS and step in STEPS:
                key = (editor, method, step)
                if key in output:
                    prior = read_json(output[key])
                    require(prior["records"] == payload["records"], f"conflicting duplicate locality {key}: {output[key]} vs {path}")
                else:
                    output[key] = path
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--additional-locality-root", action="append", type=Path, default=[])
    args = parser.parse_args()
    output = args.output_root.resolve()
    locality_sources = source_index([LEGACY_LOCALITY, HN_LOCALITY, output / "remeasured_hn", output / "remeasured_native", *args.additional_locality_root])
    canonical_path = source_run("MEMIT", "Native") / "requests.json"
    requests = read_json(canonical_path)
    require(file_record(canonical_path)["sha256"] == REQUESTS_SHA, "canonical requests SHA mismatch")
    require(len(requests) == 1000 and [r["case_id"] for r in requests] == list(range(1000)), "canonical request IDs")
    base_path = LEGACY_LOCALITY / "base/locality.json"
    base_payload = read_json(base_path)
    require(base_payload["raw_requests_sha256"] == REQUESTS_SHA, "Base requests SHA mismatch")
    base_locality = validate_locality_records(base_payload, requests, None, base_path)
    require(all(r["locality_acc"] == 1 for r in base_locality), "Base locality must be one")
    provenance_base = file_record(base_path)
    cells, missing, comparisons = [], [], []
    for editor in EDITORS:
        for method in METHODS:
            run = source_run(editor, method)
            request_record = file_record(run / "requests.json")
            require(request_record["sha256"] == REQUESTS_SHA, f"requests SHA mismatch: {run}")
            config = read_json(run / "run_config.json")
            for step in STEPS:
                historical_path = run / f"step_{step:03d}" / "evaluation.json"
                historical, summary = validate_historical(historical_path, step, config, requests)
                path = locality_sources.get((editor, method, step))
                destination = output / f"{editor}_{method}" / f"step_{step:04d}" / "evaluation.json"
                result = {"schema_version": 1, "complete": step == 0 or path is not None,
                          "editor": editor, "method": method, "edit_count": step,
                          "summary": summary, "rewrite": historical["rewrite"], "rephrase": historical["rephrase"],
                          "locality": [], "protocol": {
                              "rewrite": "Original EasyEdit teacher-forced target-token accuracy on requests[:edit_count]",
                              "rephrase": "Original EasyEdit teacher-forced rephrase target-token accuracy on requests[:edit_count]",
                              "locality": "EasyEdit teacher-forced Base/Post target-token argmax agreement on fixed requests[:1000]",
                              "aggregation": "Arithmetic mean of case accuracies",
                          }, "provenance": {"requests": request_record,
                                             "rewrite": file_record(historical_path), "rephrase": file_record(historical_path)}}
                if step == 0:
                    result["locality"] = base_locality
                    result["provenance"]["locality"] = provenance_base
                elif path is not None:
                    payload = read_json(path)
                    identity = validate_checkpoint(payload, run, step, config, path, method=method, output_root=output)
                    result["locality"] = validate_locality_records(payload, requests, base_payload["records"], path)
                    result["provenance"].update({"locality": file_record(path), "checkpoint": identity,
                                                "locality_source_protocol_fingerprint": payload["protocol_fingerprint"]})
                    new_scores = [r["locality_acc"] for r in result["locality"][:50]]
                    old_scores = [r["locality_acc"] for r in historical["locality"]]
                    require([r["case_id"] for r in historical["locality"]] == list(range(50)), f"original locality panel: {historical_path}")
                    comparison = {"editor": editor, "method": method, "edit_count": step,
                                  "historical_live_state_first50": average(old_scores), "checkpoint_replay_first50": average(new_scores),
                                  "absolute_mean_difference": abs(average(old_scores) - average(new_scores)),
                                  "changed_case_count": sum(a != b for a, b in zip(old_scores, new_scores))}
                    comparisons.append(comparison)
                    result["original50_replay_comparison"] = comparison
                else:
                    checkpoint = run / f"step_{step:03d}" / "edited_parameter_deltas.pt"
                    reason = "locality evaluation pending" if checkpoint.exists() else "saved checkpoint missing; original cumulative rewrite/rephrase retained"
                    result["missing_reason"] = reason
                    missing.append({"editor": editor, "method": method, "edit_count": step, "reason": reason,
                                    "checkpoint_path": str(checkpoint), "checkpoint_exists": checkpoint.exists()})
                locality_scores = [r["locality_acc"] for r in result["locality"]]
                result["summary"].update({"locality_acc": average(locality_scores), "locality_count": len(locality_scores)})
                result["available_metrics"] = ["rewrite", "rephrase"] + (["locality"] if locality_scores else [])
                write_json(destination, result)
                cells.append({"editor": editor, "method": method, "edit_count": step, "complete": result["complete"],
                              **result["summary"], "path": str(destination)})
    status = {"complete": not missing, "expected_cells_including_base": len(EDITORS) * len(METHODS) * len(STEPS),
              "complete_cells": sum(r["complete"] for r in cells), "missing_cells": missing,
              "original_cumulative_metrics_verified_cells": len(EDITORS) * len(METHODS) * (len(STEPS) - 1),
              "requests_sha256": REQUESTS_SHA, "cells": cells,
              "locality_sources": [str(p) for p in (LEGACY_LOCALITY, HN_LOCALITY, output / "remeasured_hn", output / "remeasured_native", *args.additional_locality_root)],
              "script": file_record(Path(__file__)),
              "limitations": ["Historical rewrite/rephrase were evaluated on the live edited states; expanded locality uses saved compact checkpoint replay.",
                              "BF16 checkpoint reconstruction and batched inference can change a small number of argmax decisions; first-50 comparisons are reported.",
                              "No generated-answer efficacy/generalization values replace the original teacher-forced metrics."]}
    write_json(output / "assembly_status.json", status)
    write_json(output / "original50_replay_comparison.json", comparisons)
    print(json.dumps({"complete_cells": status["complete_cells"], "expected_cells": status["expected_cells_including_base"],
                      "missing_cells": len(missing), "status": str(output / "assembly_status.json")}))


if __name__ == "__main__":
    main()
