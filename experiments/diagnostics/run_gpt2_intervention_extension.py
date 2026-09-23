#!/usr/bin/env python3
"""Replay GPT-2 XL H18 interventions for five methods and both editors/datasets.

Protocol v2 uses actual capped partial-removal magnitude for random controls.
All original checkpoints and analysis artifacts are read-only.
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
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from diagnostics import gpt2_checkpoint_analysis as core
from diagnostics import gpt2_intervention_analysis as old

CROSS = OUTPUT_ROOT / '_Analysis_Cross_Layer'
OUTPUT = CROSS / 'intervention_extension_20260915/gpt2'
METHODS = ('Native', 'NAS', 'ENCORE', 'SPHERE', 'SADR')
EDITORS = ('AlphaEdit', 'MEMIT')
FAMILIES = old.FAMILIES
RANDOM = tuple(f'randperp_add_s{s}' for s in old.RANDOM_SEEDS)
GEOMETRY = old.GEOMETRY_FIELDS
require, read, record, write_json = core.require, core.read, core.record, core.write_json
write_csv = old.write_csv


def intervention_target(cond, h, h0, axis, case_ids):
    if cond['kind'] != 'random':
        return ORIGINAL_TARGET(cond, h, h0, axis, case_ids)
    u, delta, _, perp = old.decompose(h, h0)
    magnitude = np.minimum(np.abs(delta @ axis)[:, None], np.linalg.norm(perp, axis=-1, keepdims=True))
    return (np.asarray(h, np.float64) + magnitude * old.random_perpendicular(u, case_ids, cond['seed'])).astype(np.float32)


ORIGINAL_TARGET = old.intervention_target
old.intervention_target = intervention_target
old.STATES = METHODS


def source_rows(dataset):
    source = CROSS / 'eff_gen_correlations_20260915/gpt2_eff_gen_mapping.csv'
    rows = [r for r in csv.DictReader(source.open()) if r['dataset'] == dataset and r['edit_count'] == '1000']
    require(len(rows) == 10, 'Expected ten endpoint checkpoints per dataset')
    result = []
    panel = None
    for editor in EDITORS:
        for method in METHODS:
            row = next(r for r in rows if r['editor'] == editor and r['method'] == method)
            audit_path = Path(row['source_analysis_audit'])
            audit = read(audit_path)
            require(audit['dtype'] == 'float32', 'Expected FP32 restored source')
            require(row['order_id'] == 'canonical', 'Expected canonical order')
            require(len(audit['panel']) == 1000 and [r['edit_rank'] for r in audit['panel']] == list(range(1,1001)), 'Invalid panel order')
            if panel is None:
                panel = audit['panel']
            require(panel == audit['panel'], 'All methods/editors must share the exact dataset panel')
            identity = audit['checkpoint_identities']['1000']
            require(identity['checkpoint']['path'] == row['checkpoint_path'], 'Checkpoint mapping mismatch')
            require(record(Path(row['checkpoint_path'])) == identity['checkpoint'], 'Checkpoint SHA mismatch')
            require(record(Path(identity['sidecar']['path'])) == identity['sidecar'], 'Sidecar SHA mismatch')
            result.append(dict(editor=editor, method=method, path=row['checkpoint_path'], identity=identity,
                               audit=record(audit_path), model_path=audit['model_path']))
    return result, panel, record(source)


def split_panel(panel, dataset):
    split = read(old.DEFAULT_SPLIT)
    # Original zsRE IDs are canonical zero-based source positions. Reuse these
    # positions, not identities, for the separate CounterFact dataset.
    positions = {kind: [int(x) for x in split[key]] for kind,key in [('fit','fit_case_ids'),('evaluation','pilot_case_ids')]}
    require(len(positions['fit']) == len(set(positions['fit'])) == 500, 'Fit must have 500 distinct positions')
    require(len(positions['evaluation']) == len(set(positions['evaluation'])) == 100, 'Evaluation must have 100 distinct positions')
    require(set(positions['fit']).isdisjoint(positions['evaluation']), 'Split overlaps')
    if dataset == 'zsRE':
        require(all(str(panel[i]['case_id']) == str(i) for i in range(1000)), 'Original zsRE ID/position mapping changed')
    groups = {key: [panel[i] for i in values] for key, values in positions.items()}
    require(set(str(r['case_id']) for r in groups['fit']).isdisjoint(str(r['case_id']) for r in groups['evaluation']), 'Case IDs overlap')
    return groups['fit'], groups['evaluation'], dict(source=record(old.DEFAULT_SPLIT), positions=positions,
        mapping='Original zsRE split positions in canonical first-1000 panel; dataset-specific case IDs are stored explicitly',
        fit_case_ids=[r['case_id'] for r in groups['fit']], evaluation_case_ids=[r['case_id'] for r in groups['evaluation']])


def summarize(rows, n_bootstrap=10000, seed=20260912):
    ids = list(dict.fromkeys(str(r['case_id']) for r in rows))
    states = [s for s in METHODS if any(r['state'] == s for r in rows)]
    condition_ids = [c['id'] for c in old.conditions()]
    lookup = {(r['state'],r['family'],r['condition'],str(r['case_id'])):r for r in rows}
    require(len(lookup) == len(rows) == len(states)*3*len(condition_ids)*len(ids), 'Incomplete case grid')
    index = np.random.default_rng(seed).integers(0,len(ids),size=(n_bootstrap,len(ids)))
    def estimate(values):
        lo,hi = np.percentile(values[index].mean(axis=1),[2.5,97.5])
        return float(values.mean()),float(lo),float(hi)
    summaries, contrasts, matching, doses = [],[],[],[]
    for state in states:
        for family in FAMILIES:
            by = {c:[lookup[state,family,c,i] for i in ids] for c in condition_ids}
            scores = {c:np.array([float(r['endpoint_value']) for r in rs])*100 for c,rs in by.items()}
            scores['randperp_mean3'] = np.mean([scores[c] for c in RANDOM],axis=0)
            for lhs,rhs,field,mandatory in [
                ('perp_f100','radial_match_perp100','realized_h18_norm',True),
                ('axis_f100','perp_match_axis','intervention_l2',False),
                *[('perp_match_axis',c,'intervention_l2',True) for c in RANDOM],
            ]:
                a,b = [np.array([float(r[field]) for r in by[c]]) for c in [lhs,rhs]]
                error=np.abs(a-b); scale=np.maximum(np.maximum(np.abs(a),np.abs(b)),1.0)
                ok=error<=1e-5*scale
                if mandatory: require(bool(ok.all()),f'Matching failed {state} {family} {lhs} {rhs}')
                matching.append(dict(state=state,family=family,lhs=lhs,rhs=rhs,field=field,n_cases=len(ids),
                    max_absolute_error=float(error.max()),max_scaled_error=float((error/scale).max()),
                    n_mismatched=int((~ok).sum()),all_cases_matched=bool(ok.all()),mandatory=mandatory))
            clipped = any(bool(r['axis_match_clipped']) for r in by['native'])
            for cond,values in scores.items():
                mean,lo,hi=estimate(values-scores['native'])
                sources=RANDOM if cond=='randperp_mean3' else [cond]
                summaries.append(dict(state=state,label=state,family=family,endpoint=old.ENDPOINTS[family],condition=cond,
                    n_cases=len(ids),endpoint_value=float(values.mean()),endpoint_minus_native=mean,endpoint_ci_low=lo,endpoint_ci_high=hi,
                    **{field:float(np.mean([float(r[field]) for c in sources for r in by[c]])) for field in GEOMETRY}))
            specs=[('matched_perp_minus_random_mean3','perp_match_axis','randperp_mean3','matched_intervention_l2'),
                   ('perp100_minus_radial','perp_f100','radial_match_perp100','matched_result_h18_norm'),
                   ('axis_minus_matched_perp','axis_f100','perp_match_axis','capped_size_match_incomplete' if clipped else 'matched_intervention_l2'),
                   *[(f'dose_{b}_minus_{a}',b,a,'adjacent_orthogonal_dose') for a,b in zip(old.DOSES,old.DOSES[1:])]]
            for name,lhs,rhs,kind in specs:
                mean,lo,hi=estimate(scores[lhs]-scores[rhs])
                contrasts.append(dict(state=state,label=state,family=family,metric=old.ENDPOINTS[family],unit='pp',contrast=name,
                    lhs=lhs,rhs=rhs,matching=kind,n_cases=len(ids),mean_difference=mean,ci_low=lo,ci_high=hi,
                    pointwise_ci_excludes_zero=bool(lo>0 or hi<0)))
            values=[float(scores[c].mean()) for c in old.DOSES]
            doses.append(dict(state=state,family=family,endpoint=old.ENDPOINTS[family],**dict(zip(old.DOSES,values)),
                              nondecreasing_point_estimates=bool((np.diff(values)>=-1e-12).all())))
    return summaries,contrasts,matching,doses


def legacy_audit(output, dataset, all_rows):
    if dataset != 'zsRE': return dict(applicable=False)
    prior = CROSS/'gpt2_xl_replication_20260914/interventions'
    source = list(csv.DictReader((prior/'per_case_scores.csv').open()))
    current={(r['editor'],r['state'],r['family'],r['condition'],str(r['case_id'])):r for r in all_rows}
    differences=[]; compared=0
    for row in source:
        if row['state'] not in ['SPHERE','SADR'] or row['condition'] in RANDOM: continue
        key=('AlphaEdit',row['state'],row['family'],row['condition'],str(row['case_id']))
        now=current[key]; compared+=1
        if float(row['endpoint_value']) != float(now['endpoint_value']):
            differences.append(dict(key=key,old=float(row['endpoint_value']),new=float(now['endpoint_value'])))
    return dict(applicable=True,source=record(prior/'per_case_scores.csv'),n_compared=compared,
                endpoint_values_exact=not differences,n_differences=len(differences),differences=differences,
                old_batch_size=2,new_batch_size=16,random_excluded='Protocol v2 changes random magnitude to capped partial removal L2')


def render(output, summaries, contrasts, dataset):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    out=output/'figures';out.mkdir(exist_ok=True)
    for editor in EDITORS:
        data=[r for r in summaries if r['editor']==editor]
        table={(r['state'],r['family'],r['condition']):r for r in data}
        fig,axs=plt.subplots(1,5,figsize=(18,4.3),sharey=True)
        for ax,state in zip(axs,METHODS):
            ax.axhline(0,color='.5',ls='--',lw=.8)
            for family,color in [('locality','#1f5fa8'),('rewrite','#5a5a5a'),('rephrase','#c1502e')]:
                rr=[table[state,family,c] for c in old.DOSES]
                x=[0,25,50,75,100]
                ax.plot(x,[r['endpoint_minus_native'] for r in rr],'o-',color=color,label=old.ENDPOINTS[family])
                ax.fill_between(x,[r['endpoint_ci_low'] for r in rr],[r['endpoint_ci_high'] for r in rr],color=color,alpha=.13)
            ax.set_title(state);ax.set_xlabel('Orthogonal removal (%)');ax.set_xticks([0,25,50,75,100]);ax.grid(axis='y',alpha=.2)
        axs[0].set_ylabel('Change from unpatched checkpoint (pp)');axs[-1].legend(frameon=False)
        fig.suptitle(f'GPT-2 XL / {dataset} / {editor} / 1,000 edits / canonical order')
        fig.text(.5,.025,'Held-out 100 cases; corrected complete-target teacher forcing; pointwise paired case-bootstrap 95% intervals',ha='center',fontsize=10)
        fig.subplots_adjust(left=.055,right=.99,bottom=.21,top=.8,wspace=.1)
        for ext in ['png','pdf','svg']:fig.savefig(out/f'{editor}_orthogonal_dose.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)
        names=[('matched_perp_minus_random_mean3','Partial removal − size-matched random'),('perp100_minus_radial','Full removal − matched-result-norm radial'),('axis_minus_matched_perp','Shared-axis − capped partial removal')]
        loc={(r['state'],r['contrast']):r for r in contrasts if r['editor']==editor and r['family']=='locality'}
        fig,axs=plt.subplots(1,3,figsize=(14,4.2),sharey=True)
        for ax,(name,title) in zip(axs,names):
            ax.axvline(0,color='.4',ls='--',lw=1)
            for j,state in enumerate(METHODS):
                r=loc[state,name];value=r['mean_difference'];color='#1f5fa8' if r['pointwise_ci_excludes_zero'] else '#5a5a5a'
                ax.errorbar(value,j,xerr=[[value-r['ci_low']],[r['ci_high']-value]],fmt='o',color=color,capsize=4)
            ax.set_title(title,fontsize=11);ax.set_yticks(range(5),METHODS);ax.set_ylim(4.55,-.55);ax.set_xlabel('Paired Δ LOC (pp)');ax.grid(axis='x',alpha=.2)
        fig.suptitle(f'GPT-2 XL / {dataset} / {editor} / 1,000 edits / canonical order')
        fig.text(.5,.025,'100 paired cases; pointwise 95% case-bootstrap intervals; third contrast may retain cap-induced size mismatch',ha='center',fontsize=10)
        fig.subplots_adjust(left=.07,right=.99,bottom=.2,top=.77,wspace=.1)
        for ext in ['png','pdf','svg']:fig.savefig(out/f'{editor}_paired_controls.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['zsRE','CounterFact'],required=True)
    p.add_argument('--device',required=True);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--fit-batch-size',type=int,default=16)
    args=p.parse_args(); started=time.monotonic()
    output=OUTPUT/('zsre' if args.dataset=='zsRE' else 'counterfact');output.mkdir(parents=True,exist_ok=True)
    sources,panel,mapping=source_rows(args.dataset);fitted,evaluated,split=split_panel(panel,args.dataset)
    protocol=dict(schema_version=2,model='GPT-2 XL',dataset=args.dataset,editors=list(EDITORS),methods=list(METHODS),
        step=1000,order_id='canonical',model_seed=42,dtype='float32',batch_size=args.batch_size,fit_batch_size=args.fit_batch_size,
        n_fit=500,n_cases=100,split=split,panel=panel,conditions=old.conditions(),sources=sources,mapping=mapping,
        target_contract='Joint prompt + space + target tokenization; explicit absolute position_ids; retain genuine EOS; no truncation',
        endpoints='EFF/Gen: case-macro complete-target teacher-forced token accuracy including real EOS. LOC: same-span Base argmax token agreement.',
        boundary='H18 block17 full output/block18 pre-LayerNorm input; original prompt_last only',
        shared_axis='Unit-normalized mean raw edited-minus-Base displacement of fit500 rewrite prompt_last, fitted independently per checkpoint',
        random_control='For each case use min(abs(delta dot axis), norm(delta_perp)) times unit random Base-orthogonal vector. SHA256 gpt2-h18|seed|case_id.',
        random_seeds=list(old.RANDOM_SEEDS),random_aggregation='Within-case mean of three directions; statistical n=100',
        uncertainty='10,000 paired case percentile bootstrap resamples; seed20260912; pointwise95%; conditional on one checkpoint/order',
        no_training=True,no_weight_changes_during_intervention=True,no_norm_layer_intervention=True,
        code=record(Path(__file__)),shared_core=record(Path(core.__file__)),intervention_core=record(Path(old.__file__)))
    if (output/'complete.json').exists():
        done=read(output/'complete.json');require(done['protocol_hash']==core.json_hash(protocol),'Completed protocol differs')
        for item in done['artifacts']:require(record(Path(item['path']))==item,'Completed artifact changed')
        print('[reuse] complete',output,flush=True);return
    if (output/'protocol.json').exists(): require(read(output/'protocol.json')==protocol,'Existing protocol differs')
    else:write_json(output/'protocol.json',protocol)
    torch.set_num_threads(4)
    model,tokenizer=core.load_model_and_tokenizer(Path(sources[0]['model_path']),device=args.device,dtype='float32')
    pristine={name:dict(model.named_parameters())[name].detach().cpu().clone() for name in core.NAMES}
    base_fit,_=core.capture_h18(model,tokenizer,[r['rewrite_prompt'] for r in fitted],args.fit_batch_size)
    core.write_npz(output/'base/fit_rewrite_h18.npz',h18=base_fit,case_ids=np.asarray([r['case_id'] for r in fitted]))
    baseline=old.capture_base(model,tokenizer,evaluated,args.batch_size,output)
    all_rows=[];all_summaries=[];all_contrasts=[];all_matching=[];all_doses=[]
    for source in sources:
        editor,state=source['editor'],source['method'];state_dir=output/editor/state
        core.restore_checkpoint(model,pristine,Path(source['path']),source['identity'])
        versions=tuple((n,v._version) for n,v in model.named_parameters())
        fit,_=core.capture_h18(model,tokenizer,[r['rewrite_prompt'] for r in fitted],args.fit_batch_size)
        axis,mean=old.fit_axis(fit[:,0],base_fit[:,0])
        core.write_npz(state_dir/'axis.npz',axis=axis,mean_raw_displacement=mean,base_h18=base_fit,edited_h18=fit,
                       fit_case_ids=np.asarray([r['case_id'] for r in fitted]))
        rows=old.run_state(model,tokenizer,evaluated,baseline,state,axis,args.batch_size,output/editor)
        require(versions==tuple((n,v._version) for n,v in model.named_parameters()),'Intervention changed weights')
        metadata=dict(model='GPT-2 XL',dataset=args.dataset,editor=editor,method=state,order_id='canonical',edit_count=1000)
        summaries,contrasts,matching,doses=summarize(rows)
        for source_rows_,destination in [(rows,all_rows),(summaries,all_summaries),(contrasts,all_contrasts),(matching,all_matching),(doses,all_doses)]:
            destination.extend([{**metadata,**r} for r in source_rows_])
        write_json(state_dir/'complete.json',dict(complete=True,checkpoint=source['identity']['checkpoint'],n_rows=len(rows),
            seconds_total=time.monotonic()-started,peak_memory_gb=torch.cuda.max_memory_allocated(args.device)/1e9))
        print(f'[state complete] {args.dataset} {editor} {state} {time.monotonic()-started:.1f}s',flush=True)
    rename={'realized_h18_norm':'realized_hidden_norm','realized_h18_norm_ratio':'realized_hidden_norm_ratio'}
    for filename,rows in [('per_case_scores.csv',all_rows),('condition_summary.csv',all_summaries),('paired_contrasts.csv',all_contrasts),('matching_audit.csv',all_matching),('dose_monotonicity.csv',all_doses)]:
        renamed=[{rename.get(k,k):rename.get(v,v) if k=='field' else v for k,v in r.items()} for r in rows]
        write_csv(output/filename,renamed)
    write_json(output/'legacy_reproduction_audit.json',legacy_audit(output,args.dataset,all_rows))
    render(output,all_summaries,all_contrasts,args.dataset)
    artifacts=[record(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ['complete.json','status.json']]
    write_json(output/'complete.json',dict(complete=True,protocol_hash=core.json_hash(protocol),artifacts=artifacts,n_states=10,
        n_cases=100,n_fit=500,n_rows=len(all_rows),no_training=True,seconds=time.monotonic()-started,
        mandatory_matching_all_pass=all(r['all_cases_matched'] for r in all_matching if r['mandatory'])))
    print('[complete]',output,flush=True)


if __name__=='__main__':main()
