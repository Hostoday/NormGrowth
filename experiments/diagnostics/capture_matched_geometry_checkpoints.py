#!/usr/bin/env python3
"""Matched frozen-checkpoint geometry and teacher-forced EFF/Gen/LOC; no editing.

All states use the same canonical case order, BF16 eager forwards, capture batch
16 and evaluator batch 2. Checkpoint deltas are always added to saved pristine
Base parameters, never to another edited endpoint. The existing H/M reference
collector's nodes and calculations are reused; a hook additionally records the
actual residual U entering post-attention RMSNorm.
"""
from __future__ import annotations

# Release-only filesystem defaults; computation below follows the source snapshot.
import sys as _bng_sys
from pathlib import Path as _BNGPath
_bng_sys.path.insert(0, str(_BNGPath(__file__).resolve().parents[1]))
from model_code.paths import RESEARCH_ROOT, OUTPUT_ROOT, DATA_ROOT, LLAMA_MODEL

import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluate"))
import numpy as np
import torch
from transformers import AutoTokenizer
from diagnostics import analyze_base_h_m_reference_cosines as hm
from diagnostics.analyze_base_component_cosines import (
    LayerComponentCapture, _atomic_json, _atomic_npz, _model_backbone,
    _padded_batch, fingerprint,
)
from diagnostics.analyze_residual_spectrum import build_probe, load_model, model_input_device
from diagnostics.analyze_edited_a_m_o_reference_cosines import validate_checkpoint_metadata
from diagnostics.evaluate_raw_plot_locality_n1000 import record
from eval_hf_easyedit import build_request_eval_items, compute_rewrite_scores, compute_locality_outputs

OUT = OUTPUT_ROOT / '_Analysis_Cross_Layer/alphaedit_matched_geometry_causal_audit_n1000_order20260905_v1'
RUNS = OUTPUT_ROOT / '_Edited_Model'
SOURCES = {s: RUNS / "alphaedit_virtual_actual_n1000_order20260905_v1" / s
           for s in ("hiddennorm", "sadr", "sphere")}
SOURCES["hiddennorm_sadr"] = RUNS / "alphaedit_sadr_hiddennorm_n1000_order20260905_v1/hiddennorm_sadr"
MODEL = LLAMA_MODEL
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
CANONICAL = RUNS / "method_rms_trajectory_n1000_artifacts_v1/AlphaEdit/alphaedit_zsre_bs1_n1000_sphere_seed42/requests.json"
NAMES = [f"model.layers.{i}.mlp.down_proj.weight" for i in range(4, 9)]
LAYERS = tuple(range(32))


def read(path):
    return json.loads(path.read_text())


def audit_sources():
    requests = read(CANONICAL)
    ordered = read(SOURCES["hiddennorm"] / "requests.json")
    assert len(requests) == len(ordered) == 1000
    assert [int(r["case_id"]) for r in requests] == list(range(1000))
    assert sorted(ordered, key=lambda r: int(r["case_id"])) == requests
    configs, sources = {}, {}
    common_keys = ("layers", "fact_token", "clamp_norm_factor", "v_num_grad_steps", "v_lr",
                   "v_loss_layer", "v_weight_decay", "kl_factor", "mom2_adjustment",
                   "mom2_update_weight", "mom2_dataset", "mom2_n_samples", "mom2_dtype",
                   "nullspace_threshold", "L2", "model_parallel", "bf16", "batch_size")
    baseline = read(SOURCES["hiddennorm"] / "run_config.json")
    for state, folder in SOURCES.items():
        cfg = read(folder / "run_config.json")
        assert read(folder / "requests.json") == ordered, state
        assert cfg["model_name"] == MODEL_NAME and cfg["sample_size"] == 1000
        assert cfg["seed"] == 42 and cfg["batch_size"] == 1
        assert cfg["edit_order_manifest"]["shuffle_seed"] == 20260905
        assert all(cfg["effective_editing_hparams"][key] == baseline["effective_editing_hparams"][key]
                   for key in common_keys), state
        configs[state] = cfg
        sources[state] = dict(run_config=record(folder / "run_config.json"),
                              requests=record(folder / "requests.json"),
                              order=cfg["edit_order_manifest"],
                              method_specific_hparams={k:v for k,v in cfg.items()
                                  if k.startswith(("residual_gain_", "sadr_", "sphere_", "hidden_norm_"))})
    return requests, ordered, configs, dict(complete=True, all_training_requests_exactly_equal=True,
        canonical_requests_exactly_equal_after_sort=True, canonical_requests=record(CANONICAL),
        training_order_case_ids=[int(r["case_id"]) for r in ordered],
        common_editing_hparams={k:baseline["effective_editing_hparams"][k] for k in common_keys},
        model_path=str(MODEL), sources=sources,
        scope="Matched data/order/Base/AlphaEdit common hyperparameters; method penalties intentionally differ.")


def make_probes(requests, tokenizer, family):
    positions = ["prompt_last"] if family == "locality" else ["subject_last", "prompt_last"]
    probes, valid, missing = [], [], []
    for i, req in enumerate(requests):
        try:
            probe = build_probe(req, tokenizer, positions, 256, probe_source=family, source_index=i)
            mask = [True] * len(positions)
        except ValueError as exc:
            if family != "rephrase":
                raise
            probe = build_probe(req, tokenizer, ["prompt_last"], 256, probe_source=family, source_index=i)
            probe.positions["subject_last"] = probe.positions["prompt_last"]
            mask = [False, True]
            missing.append(dict(case_id=int(req["case_id"]), reason=str(exc)))
        probes.append(probe)
        valid.append(mask)
    return probes, positions, np.asarray(valid, dtype=bool), missing


class SequentialCapture(LayerComponentCapture):
    def install(self):
        super().install()
        for lid in self.layers:
            def capture_u(_module, args, *, key=lid):
                self._capture(key, "u", args[0])
            self.handles.append(self.decoder_layers[lid].post_attention_layernorm.register_forward_pre_hook(capture_u))


def branch_stats(nodes):
    h, a, u, m, o = [nodes[k].detach().to(device="cpu", dtype=torch.float64)
                     for k in ("input", "attention", "u", "mlp", "output")]
    norm = lambda x: torch.linalg.vector_norm(x, dim=-1)
    dot = lambda x,y: torch.sum(x*y, dim=-1)
    result = {f"{key}_norm": norm(val) for key,val in zip("haumo", (h,a,u,m,o))}
    result.update({f"dot_{key}": dot(x,y) for key,x,y in
                  (("ha",h,a),("hm",h,m),("am",a,m),("um",u,m),("ho",h,o),("uo",u,o))})
    result.update(residual_ha_norm=norm(u-h-a), residual_um_norm=norm(o-u-m),
                  residual_ham_norm=norm(o-h-a-m), sum_ha_norm=norm(h+a),
                  sum_um_norm=norm(u+m), sum_ham_norm=norm(h+a+m))
    return {k:v.numpy() for k,v in result.items()}


@torch.inference_mode()
def capture(model, tokenizer, probes, positions, valid):
    n,p = len(probes),len(positions)
    arrays = {k:np.full((n,p,32,32), np.nan, dtype=np.float32) for k in hm.MATRIX_METRICS}
    arrays.update({k:np.full((n,p,32), np.nan, dtype=np.float32)
                   for k in hm.LAYER_METRICS+hm.NORM_METRICS+hm.DIAGNOSTIC_METRICS})
    arrays["token_positions"] = np.empty((n,p), dtype=np.int32)
    raw = np.empty((n,p,4096), dtype=np.float32)
    stats = {}
    collector = SequentialCapture(model, LAYERS)
    collector.install()
    started = time.monotonic()
    try:
        for start in range(0,n,16):
            end = min(start+16,n)
            inputs, tp = _padded_batch(probes[start:end], positions,
                pad_token_id=int(tokenizer.pad_token_id), device=model_input_device(model))
            collector.begin_batch(tp)
            _model_backbone(model)(**inputs, use_cache=False, output_hidden_states=False, return_dict=True)
            collector.validate()
            device = collector.values[(0,"output")].device
            nodes = {key:hm._stack_capture(collector,LAYERS,key,device)
                     for key in ("input","attention","u","mlp","output")}
            assert torch.equal(nodes["input"][:,:,9], nodes["output"][:,:,8])
            for key,value in hm.reference_cosines_from_nodes(nodes).items():
                arrays[key][start:end] = value.float().cpu().numpy()
            for key,value in branch_stats(nodes).items():
                if key not in stats:
                    stats[key] = np.full((n,p,32),np.nan,dtype=np.float64)
                stats[key][start:end] = value
            raw[start:end] = nodes["input"][:,:,9].float().cpu().numpy()
            arrays["token_positions"][start:end] = tp.numpy()
            if start == 0 or end%112 == 0 or end == n:
                print(f"[capture] {end}/{n} {time.monotonic()-started:.1f}s",flush=True)
            del nodes
    finally:
        collector.close()
    for values in [arrays,stats]:
        for key,array in values.items():
            if key != "token_positions":
                array[~valid] = np.nan
    raw[~valid] = np.nan
    arrays["token_positions"][~valid] = -1
    return arrays,stats,raw


def save_capture(model,tokenizer,requests,state,step,family,out,identity,source_audit):
    destination = out / "captures" / state / f"step_{step:04d}" / family / "h_m_reference"
    if (destination / "complete.json").exists():
        done = read(destination / "complete.json")
        assert done.get("complete") and done.get("identity") == identity
        for item in done["files"].values():
            actual = record(destination / item["path"])
            assert actual["sha256"] == item["sha256"] and actual["bytes"] == item["bytes"]
        print(f"[reuse capture] {state} {step} {family}",flush=True)
        return
    probes,positions,valid,missing = make_probes(requests,tokenizer,family)
    arrays,stats,raw = capture(model,tokenizer,probes,positions,valid)
    cfg = dict(schema_version=1, identity=identity, state=state,checkpoint_edit_count=step,
        measurement="matched_frozen_checkpoint_geometry", n_prompts=len(requests),
        positions=positions,layers=list(LAYERS),batch_size=16,max_length=256,
        torch_dtype="bfloat16",attn_implementation="eager",model_name=str(MODEL),
        probe_source=family,prompt_family=family,case_ids=[str(r["case_id"]) for r in requests],
        requests_sha256=source_audit["canonical_requests"]["sha256"],
        prompt_fingerprint=fingerprint([dict(case_id=str(p.case_id),prompt=p.prompt,subject=p.subject) for p in probes]),
        token_semantics="Prompt text only, add_special_tokens=True, right padding, no target or chat template",
        missing_subject_positions=missing, valid_count_by_position=dict(zip(positions,valid.sum(0).tolist())),
        u_semantics="Actual BF16 residual entering post_attention_layernorm, before RMSNorm",
        decomposition_precision="float64 norms/dots of captured BF16 node values; actual residual additions retained",
        code=record(Path(__file__)),collector=record(Path(hm.__file__)))
    basepath = out / "captures/base/step_0000" / family / "h_m_reference/h_m_reference_cosines.npz"
    if state != "base":
        with np.load(basepath,allow_pickle=False) as base:
            assert np.array_equal(arrays["token_positions"],base["token_positions"])
            errors = {}
            for key in hm.MATRIX_METRICS+hm.LAYER_METRICS+hm.NORM_METRICS:
                before,after=base[key],arrays[key]
                before=before[...,:4,:4] if key in hm.MATRIX_METRICS else before[...,:4]
                after=after[...,:4,:4] if key in hm.MATRIX_METRICS else after[...,:4]
                assert np.array_equal(np.isnan(before),np.isnan(after))
                errors[key]=float(np.nanmax(np.abs(before.astype(np.float64)-after.astype(np.float64))))
            assert all(v == 0 for v in errors.values()), errors
            cfg["early_base_validation"] = dict(exact=True,layers=[0,1,2,3],max_abs_errors=errors)
    coords=dict(case_ids=np.asarray(cfg["case_ids"]),positions=np.asarray(positions),
        layers=np.asarray(LAYERS,dtype=np.int32),token_positions=arrays["token_positions"],valid_positions=valid)
    destination.mkdir(parents=True,exist_ok=True)
    _atomic_npz(destination/"decomposition.npz",dict(**stats,**coords))
    _atomic_npz(destination/"raw_h9.npz",dict(h9=raw,**coords))
    _atomic_json(destination/"probes.json",dict(probes=[dict(case_id=str(p.case_id),prompt=p.prompt,
        subject=p.subject,input_ids=list(p.input_ids),positions={k:(v if valid[i,j] else None)
        for j,(k,v) in enumerate((x,p.positions[x]) for x in positions)}) for i,p in enumerate(probes)]))
    manifest=hm.write_outputs(destination,probes=probes,layers=LAYERS,positions=positions,arrays=arrays,config=cfg)
    for key in ("decomposition","raw_h9","probes"):
        filename=key+(".json" if key=="probes" else ".npz")
        item=record(destination/filename)
        item["path"]=filename
        manifest["files"][key]=item
    manifest.update(measurement=cfg["measurement"],identity=identity,state=state,checkpoint_edit_count=step,
                    prompt_family=family,valid_count_by_position=cfg["valid_count_by_position"])
    _atomic_json(destination/"complete.json",manifest)
    print(f"[capture complete] {destination}",flush=True)


@torch.inference_mode()
def evaluate(model,tokenizer,requests,ordered,state,step,out,identity):
    target=out/"performance"/state/f"step_{step:04d}.json"
    if target.exists():
        done=read(target)
        assert done.get("complete") and done.get("identity")==identity
        print(f"[reuse performance] {state} {step}",flush=True)
        return
    prompts,targets,rephrases,locality,_=build_request_eval_items(requests)
    assert len(rephrases)==len(locality)==len(requests)
    assert [x[0] for x in rephrases] == [x[0] for x in locality] == list(range(len(requests)))
    eval_device=int(model_input_device(model).index)
    hparams=SimpleNamespace(max_length=256,alg_name="AlphaEdit",device=eval_device)
    started=time.monotonic()
    scores={}
    for key,ps,ts in (("eff",prompts,targets),("gen",[x[1] for x in rephrases],[x[2] for x in rephrases])):
        values=[]
        for start in range(0,len(requests),100):
            values.extend(compute_rewrite_scores(model,MODEL_NAME,hparams,tokenizer,
                ps[start:start+100],ts[start:start+100],device=eval_device,
                batch_size=2,test_rephrase=(key=="gen")))
            print(f"[{state} {step} {key}] {len(values)}/{len(requests)} {time.monotonic()-started:.1f}s",flush=True)
        assert len(values)==len(requests)
        scores[key]=values
    outputs={}
    for start in range(0,len(locality),100):
        outputs.update(compute_locality_outputs(model,MODEL_NAME,hparams,tokenizer,
            locality[start:start+100],device=eval_device,batch_size=2))
        print(f"[{state} {step} loc] {len(outputs)}/{len(requests)} {time.monotonic()-started:.1f}s",flush=True)
    assert set(outputs)==set(range(len(requests)))
    reference=outputs if state=="base" else {i:row["locality_outputs"] for i,row in enumerate(
        read(out/"performance/base/step_0000.json")["cases"])}
    rank={int(r["case_id"]):i+1 for i,r in enumerate(ordered)}
    cases=[]
    for i,req in enumerate(requests):
        pair_scores=[]
        assert outputs[i].keys()==reference[i].keys()
        for key,pairs in reference[i].items():
            assert len(pairs)==len(outputs[i][key])
            for before,after in zip(pairs,outputs[i][key]):
                assert len(before)==len(after) and len(before)>0
                pair_scores.append(float(np.mean(np.equal(before,after))))
        cases.append(dict(case_id=int(req["case_id"]),edit_rank=rank[int(req["case_id"])],
            edited=rank[int(req["case_id"])]<=step,eff=scores["eff"][i],gen=scores["gen"][i],
            loc=float(np.mean(pair_scores)),locality_outputs=outputs[i]))
    _atomic_json(target,dict(complete=True,state=state,checkpoint_edit_count=step,identity=identity,
        n_cases=len(cases),batch_size=2,max_length=256,dtype="bfloat16",attn_implementation="eager",
        evaluation_order="canonical case_id ascending",cases=cases,
        summary={key:float(np.mean([row[key] for row in cases])) for key in ("eff","gen","loc")},
        protocol="Original EasyEdit teacher-forced target-token accuracy for EFF/Gen; case-macro Base argmax agreement for LOC",
        locality_base_reference=None if state=="base" else record(out/"performance/base/step_0000.json"),
        evaluator=record(ROOT/"evaluate/eval_hf_easyedit.py"),
        target_slicing=record(ROOT/"EasyEdit/easyeditor/evaluate/evaluate_utils.py"),
        seconds=time.monotonic()-started))
    print(f"[performance complete] {target}",flush=True)


def restore_checkpoint(model,pristine,state,step,configs):
    named=dict(model.named_parameters())
    if state=="base":
        with torch.no_grad():
            for name in NAMES:
                named[name].copy_(pristine[name])
        return dict(state="base",checkpoint_edit_count=0,base_model=str(MODEL.resolve()))
    # Source experiment directories are unpadded (step_100, step_250, ...).
    # Output directories remain padded for lexicographic chronological order.
    checkpoint=SOURCES[state]/f"step_{step}"/"edited_parameter_deltas.pt"
    metadata=read(checkpoint.with_suffix(".json"))
    validate_checkpoint_metadata(metadata,checkpoint_path=checkpoint,
        expected_edit_count=step,expected_rewrite_layers=[4,5,6,7,8])
    assert metadata["run_fingerprint"]==configs[state]["fingerprint"]
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    assert payload.get("format")=="easyedit-edited-parameters" and payload.get("format_version")==1
    assert payload["metadata"]==metadata and list(payload["state_dict"])==NAMES
    with torch.no_grad():
        for name,delta in payload["state_dict"].items():
            assert delta.dtype==torch.bfloat16 and tuple(delta.shape)==tuple(named[name].shape)
            base=pristine[name].to(named[name].device)
            named[name].copy_(base+delta.to(device=named[name].device,dtype=named[name].dtype))
    del payload
    return dict(state=state,checkpoint_edit_count=step,base_model=str(MODEL.resolve()),
                checkpoint=record(checkpoint),checkpoint_metadata=metadata)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--states",nargs="+",choices=["base",*SOURCES],default=["base","sadr","hiddennorm_sadr","hiddennorm","sphere"])
    p.add_argument("--steps",nargs="+",type=int,default=[1000,100,250,500,750])
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--output-root",type=Path,default=OUT)
    p.add_argument("--limit",type=int,default=1000,help="Only for smoke in a separate output directory")
    p.add_argument("--audit-only",action="store_true")
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    requests,ordered,configs,audit=audit_sources()
    if a.limit!=1000 and a.output_root.resolve()==OUT.resolve():
        raise ValueError("Smoke requires a separate output root")
    requests=requests[:a.limit]
    a.output_root.mkdir(parents=True,exist_ok=True)
    audit_path=a.output_root/"source_audit.json"
    if audit_path.exists():
        assert read(audit_path)==audit
    else:
        _atomic_json(audit_path,audit)
    if a.audit_only:
        print(json.dumps(audit,ensure_ascii=False),flush=True)
        return
    status_path=a.output_root/f"capture_status_gpu{a.device.split(':')[-1]}.json"
    started=time.monotonic()
    def status(stage,**kwargs):
        _atomic_json(status_path,dict(complete=False,pid=os.getpid(),device=a.device,stage=stage,
            elapsed_seconds=time.monotonic()-started,states=a.states,steps=a.steps,**kwargs))
    if "base" not in a.states:
        base_performance=a.output_root/"performance/base/step_0000.json"
        while not base_performance.exists():
            status("waiting_for_shared_base",required=str(base_performance))
            time.sleep(5)
        assert read(base_performance).get("complete")
    status("loading_model")
    model=load_model(str(MODEL),"bfloat16",a.device,False,"eager")
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True)
    tokenizer.pad_token_id=tokenizer.eos_token_id
    named=dict(model.named_parameters())
    pristine={name:named[name].detach().cpu().clone() for name in NAMES}
    schedule=([("base",0)] if "base" in a.states else [])+[(state,step) for step in a.steps for state in a.states if state!="base"]
    for state,step in schedule:
        status("restoring",state=state,step=step)
        identity=restore_checkpoint(model,pristine,state,step,configs)
        versions={name:int(param._version) for name,param in model.named_parameters()}
        for family in ("rewrite","rephrase","locality"):
            status("capturing",state=state,step=step,family=family)
            save_capture(model,tokenizer,requests,state,step,family,a.output_root,identity,audit)
        status("evaluating",state=state,step=step)
        evaluate(model,tokenizer,requests,ordered,state,step,a.output_root,identity)
        assert all(int(param._version)==versions[name] for name,param in model.named_parameters()), "Inference mutated parameters"
        status("checkpoint_complete",state=state,step=step,parameters_unchanged_during_inference=True)
    _atomic_json(status_path,dict(complete=True,pid=os.getpid(),device=a.device,stage="complete",
        elapsed_seconds=time.monotonic()-started,states=a.states,steps=a.steps,
        parameters_unchanged_during_inference=True,n_cases=len(requests)))


if __name__=="__main__":
    main()
