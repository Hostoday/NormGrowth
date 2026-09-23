#!/usr/bin/env python3
"""Supplement the original benchmark with correct target spans, without editing.

The original evaluator counts genuine target EOT tokens as padding when
pad_token_id == eos_token_id. This supplementary evaluator uses attention_mask
for padding and asserts the complete prompt token prefix before scoring every
target token, including EOT. Legacy scores are independently derived from the
same logits and required to equal the saved original benchmark for every case.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"evaluate"))
import numpy as np
import torch
from transformers import AutoTokenizer
from diagnostics.capture_matched_geometry_checkpoints import (
    OUT, MODEL, MODEL_NAME, NAMES, SOURCES, audit_sources, restore_checkpoint,
)
from diagnostics.analyze_residual_spectrum import load_model, model_input_device
from diagnostics.analyze_base_component_cosines import _atomic_json
from diagnostics.evaluate_raw_plot_locality_n1000 import record
from eval_hf_easyedit import compute_rewrite_scores


def read(path):
    return json.loads(path.read_text())


def materialize(tokenizer,prompts,targets):
    """Exactly preserve original full-text tokenization and left-padding batches."""
    texts=[prompt+" "+target for prompt,target in zip(prompts,targets)]
    max_length=max(256,max(len(tokenizer.encode(text)) for text in texts)+1)
    previous=tokenizer.padding_side
    tokenizer.padding_side="left"
    try:
        tokens=tokenizer(texts,padding=True,truncation=True,max_length=max_length,return_tensors="pt")
        prompt_tokens=tokenizer(prompts,padding=True,truncation=True,max_length=max_length,return_tensors="pt")
    finally:
        tokenizer.padding_side=previous
    starts,legacy_starts,proofs=[],[],[]
    for i,(prompt,target) in enumerate(zip(prompts,targets)):
        ids=tokens["input_ids"][i].tolist()
        mask=tokens["attention_mask"][i].tolist()
        prompt_ids=tokenizer(prompt)["input_ids"]
        padding=mask.count(0)
        assert mask==[0]*padding+[1]*(len(mask)-padding)
        start=padding+len(prompt_ids)
        assert ids[padding:start]==prompt_ids, "Prompt/target tokenization prefix differs"
        assert start>padding and start<len(ids), "Empty prompt or target span"
        assert all(mask[j]==1 for j in range(start,len(ids)))
        legacy_prompt_count=int((prompt_tokens["input_ids"][i]!=tokenizer.pad_token_id).sum())
        assert legacy_prompt_count==len(prompt_ids), "Genuine padding-ID token inside prompt"
        legacy_start=ids.count(tokenizer.pad_token_id)+legacy_prompt_count
        genuine_padding_id_tokens=ids[padding:].count(tokenizer.pad_token_id)
        assert legacy_start==start+genuine_padding_id_tokens
        # Causal prediction of labels j uses logits j-1, including the very first
        # target token at the prompt's final non-padding position.
        assert start-1==padding+len(prompt_ids)-1
        label_indices=list(range(start,len(ids)))
        logit_indices=list(range(start-1,len(ids)-1))
        assert len(label_indices)==len(logit_indices)>0
        assert all(j==i+1 for i,j in zip(logit_indices,label_indices))
        starts.append(start)
        legacy_starts.append(legacy_start)
        proofs.append(dict(label_start=start,logit_start=start-1,sequence_length=len(ids),
            n_prompt_tokens=len(prompt_ids),n_padding_tokens=padding,legacy_label_start=legacy_start,
            first_target_token_id=ids[start],first_target_token=tokenizer.decode([ids[start]]),
            final_target_token_id=ids[-1],genuine_eot_scored=ids[-1]==tokenizer.eos_token_id,
            n_genuine_padding_id_tokens=genuine_padding_id_tokens,prompt_prefix_exact=True,
            full_nonpadding_ids=ids[padding:],label_indices=label_indices,logit_indices=logit_indices))
    return tokens,starts,legacy_starts,proofs


@torch.inference_mode()
def measure(model,tokenizer,prompts,targets,expected_legacy,metric):
    corrected,legacy,details=[],[],[]
    device=model_input_device(model)
    started=time.monotonic()
    for start in range(0,len(prompts),2):
        ps,ts=prompts[start:start+2],targets[start:start+2]
        tokens,starts,old_starts,proofs=materialize(tokenizer,ps,ts)
        labels=tokens["input_ids"].numpy()
        result=model(**tokens.to(device))
        prediction=result.logits.argmax(dim=-1).cpu().numpy()
        assert prediction.shape==labels.shape
        del result
        for row,(new_start,old_start,proof) in enumerate(zip(starts,old_starts,proofs)):
            gold=labels[row,new_start:]
            pred=prediction[row,new_start-1:-1]
            assert gold.shape==pred.shape and len(gold)>0
            old_gold=labels[row,old_start:]
            old_pred=prediction[row,old_start-1:-1]
            assert old_gold.shape==old_pred.shape and len(old_gold)>0
            value=float(np.mean(np.equal(gold,pred)))
            old_value=float(np.mean(np.equal(old_gold,old_pred)))
            assert old_value==expected_legacy[start+row], (metric,start+row,old_value,expected_legacy[start+row])
            matches=np.equal(gold,pred)
            # Actual requests contain answer content plus one terminal EOT.
            # Keep content-only diagnostics separate from the primary full span.
            assert gold[-1]==tokenizer.eos_token_id and len(gold)>=2
            assert old_start==new_start+1
            proof.update(label_ids=gold.tolist(),prediction_ids=pred.tolist(),
                         legacy_label_ids=old_gold.tolist(),legacy_prediction_ids=old_pred.tolist(),
                         target_token_count=len(gold),legacy_target_token_count=len(old_gold),
                         first_target_token_acc=float(matches[0]),
                         without_terminal_eot_acc=float(np.mean(matches[:-1])),
                         terminal_eot_acc=float(matches[-1]))
            corrected.append(value)
            legacy.append(old_value)
            details.append(proof)
        # Direct call to the unchanged evaluator verifies its real implementation
        # as well as saved-score parity on the first four cases of each endpoint.
        if start<4:
            hp=SimpleNamespace(max_length=256,alg_name="AlphaEdit",device=device.index)
            reference=compute_rewrite_scores(model,MODEL_NAME,hp,tokenizer,ps,ts,
                device=device.index,batch_size=2,test_rephrase=(metric=="gen"))
            assert reference==legacy[start:start+len(ps)], (metric,"original evaluator parity")
        if start==0 or (start+2)%100==0 or start+2>=len(prompts):
            print(f"[{metric}] {min(start+2,len(prompts))}/{len(prompts)} {time.monotonic()-started:.1f}s",flush=True)
    return corrected,legacy,details


def evaluate(model,tokenizer,requests,state,step,output,identity):
    path=output/state/f"step_{step:04d}.json"
    original_path=OUT/"performance"/state/f"step_{step:04d}.json"
    original=read(original_path)
    assert original["complete"] and original["identity"]==identity
    parent_record=record(original_path)
    if path.exists():
        old=read(path)
        assert old["complete"] and old["identity"]==identity and old["parent_original"]==parent_record
        assert old["n_cases"]==len(requests)
        print(f"[reuse corrected] {path}",flush=True)
        return
    n=len(requests)
    parent_cases=original["cases"][:n]
    assert [r["case_id"] for r in parent_cases]==[r["case_id"] for r in requests]
    scores={}
    for metric,key in (("eff","prompt"),("gen","rephrase_prompt")):
        scores[metric]=measure(model,tokenizer,[r[key] for r in requests],
            [r["target_new"] for r in requests],[r[metric] for r in parent_cases],metric)
    cases=[]
    for i,parent in enumerate(parent_cases):
        case=dict(parent)
        for metric in ("eff","gen"):
            case[metric]=scores[metric][0][i]
            case[f"legacy_{metric}"]=scores[metric][1][i]
            case[f"{metric}_tokens"]=scores[metric][2][i]
            for diagnostic in ("first_target_token_acc","without_terminal_eot_acc","terminal_eot_acc"):
                case[f"{metric}_{diagnostic}"]=scores[metric][2][i][diagnostic]
        cases.append(case)
    summary={metric:float(np.mean([r[metric] for r in cases])) for metric in ("eff","gen","loc")}
    legacy_summary={metric:float(np.mean([r[metric] for r in parent_cases])) for metric in ("eff","gen","loc")}
    diagnostic_summary={f"{metric}_{diagnostic}":float(np.mean([r[f"{metric}_{diagnostic}"] for r in cases]))
        for metric in ("eff","gen")
        for diagnostic in ("first_target_token_acc","without_terminal_eot_acc","terminal_eot_acc")}
    _atomic_json(path,dict(complete=True,state=state,checkpoint_edit_count=step,identity=identity,
        n_cases=n,cases=cases,summary=summary,legacy_summary=legacy_summary,diagnostic_summary=diagnostic_summary,
        summary_difference={k:summary[k]-legacy_summary[k] for k in summary},
        batch_size=2,max_length=256,dtype="bfloat16",attn_implementation="eager",
        evaluation_order="canonical case_id ascending",parent_original=parent_record,
        locality_reused_from_parent_original=True,locality_outputs_unchanged=True,
        protocol="Corrected teacher-forced all-target-token macro accuracy, including genuine EOT; LOC reused from identical checkpoint original benchmark",
        target_start_rule="count(attention_mask==0)+len(tokenizer(prompt)); assert exact full-text prompt prefix",
        causal_alignment="label j paired with logit j-1, for every target token including final EOT",
        legacy_parity_all_cases=True,original_evaluator_direct_parity_cases=4,
        pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,
        code=record(Path(__file__)),original_evaluator=record(ROOT/"EasyEdit/easyeditor/evaluate/evaluate_utils.py")))
    print(f"[corrected complete] {state} {step} {summary}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--states",nargs="+",choices=["base",*SOURCES],required=True)
    p.add_argument("--steps",nargs="+",type=int,default=[1000,100,250,500,750])
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--output-root",type=Path,default=OUT/"performance_corrected")
    p.add_argument("--limit",type=int,default=1000)
    a=p.parse_args()
    if a.limit!=1000 and a.output_root.resolve()==(OUT/"performance_corrected").resolve():
        raise ValueError("Use a separate smoke output root")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    requests,ordered,configs,audit=audit_sources()
    requests=requests[:a.limit]
    a.output_root.mkdir(parents=True,exist_ok=True)
    status_path=a.output_root/f"status_gpu{a.device.split(':')[-1]}.json"
    started=time.monotonic()
    def status(stage,**extra):
        _atomic_json(status_path,dict(complete=False,pid=os.getpid(),stage=stage,device=a.device,
            states=a.states,steps=a.steps,elapsed_seconds=time.monotonic()-started,**extra))
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
        versions={name:int(p._version) for name,p in model.named_parameters()}
        status("evaluating_corrected",state=state,step=step)
        evaluate(model,tokenizer,requests,state,step,a.output_root,identity)
        assert all(int(p._version)==versions[name] for name,p in model.named_parameters())
    _atomic_json(status_path,dict(complete=True,pid=os.getpid(),stage="complete",device=a.device,
        states=a.states,steps=a.steps,n_cases=len(requests),parameters_unchanged_during_inference=True,
        elapsed_seconds=time.monotonic()-started))


if __name__=="__main__":
    main()
