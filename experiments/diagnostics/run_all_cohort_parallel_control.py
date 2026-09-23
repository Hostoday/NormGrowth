#!/usr/bin/env python3
"""Extend frozen Figure 7 parallel controls to the remaining Figure 6 cohorts.

Preserves corrected AlphaEdit-SPHERE sources, native batching/dtype/token spans,
and the exact-radicand same-target-norm comparison. Original files are read-only.
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
from pathlib import Path
import sys
import time
import numpy as np
import torch
from transformers import AutoTokenizer
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics import gpt2_checkpoint_analysis as core
from diagnostics import gpt2_intervention_analysis as gpt
from diagnostics import run_llama_intervention_extension as llama
from diagnostics import run_llama_counterfact_parallel_control as prior
CROSS=OUTPUT_ROOT / '_Analysis_Cross_Layer'
OUTPUT=CROSS/'parallel_control_all_cohorts_20260923'
EXTENSION=CROSS/'intervention_extension_20260915'
CORRECTED=CROSS/'paper_revision_figures_20260922/interventions/corrected'
FIGURE6=RESEARCH_ROOT / 'paper_results_reorganization_20260922/results_rewrite_20260923_rq3_extended/data/intervention_partial_40.csv'
COHORTS={'llama_zsre':('Llama','zsRE','llama'), 'gpt2_zsre':('GPT-2 XL','zsRE','gpt2/zsre'), 'gpt2_counterfact':('GPT-2 XL','CounterFact','gpt2/counterfact')}
FAMILIES=prior.FAMILIES
METHODS=prior.METHODS
EDITORS=prior.EDITORS
FRACTIONS=prior.FRACTIONS
KEYS=prior.KEYS
read, require, write_json, write_npz=core.read,core.require,core.write_json,core.write_npz
write_csv=llama.write_csv
OUT=None
BATCH=None
EXEC_DTYPE=None
IS_LLAMA=None
legacy=None

def record(path):
    return core.record(Path(path))

def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))

def original_state(source):
    return Path(source['source_state'])

def label(prefix, fraction):
    return f'{prefix}_f{round(fraction*100):03d}'

# Retain the original independently audited F64 construction exactly.
construct=prior.construct

class Backend:
    ENDPOINTS=llama.ENDPOINTS
    geometry=staticmethod(llama.geometry)
    prediction_rows=staticmethod(gpt.prediction_rows)
    @staticmethod
    def materialize(tok, panel, family, device):
        if IS_LLAMA:
            return llama.materialize(tok,panel,family,device)
        pairs=[gpt.family_pair(row,family) for row in panel]
        return core.materialize(tok,[p[0] for p in pairs],[p[1] for p in pairs],device)
    @staticmethod
    def forward(model, inputs, positions, replacement=None):
        fn=llama.forward if IS_LLAMA else gpt.forward_with_patch
        return fn(model,inputs,positions,replacement)

def load_frozen_hidden(source, family):
    directory=original_state(source)
    if IS_LLAMA:
        stored=np.load(directory/f'hidden_{family}.npz')
        return {key:stored[key].copy() for key in ('case_ids','native','perp_f025','perp_f050')}
    result={}
    for condition in ('native','perp_f025','perp_f050'):
        stored=np.load(directory/'geometry'/family/f'{condition}.npz')
        if 'case_ids' in result:
            require(np.array_equal(result['case_ids'],stored['case_ids']),'Frozen GPT case order differs')
        result['case_ids']=stored['case_ids'].copy()
        result[condition]=stored['h18'].copy()
    return result

def audit_sources(cohort):
    model_name,dataset,subdir=COHORTS[cohort]
    original=EXTENSION/subdir
    old=read(original/'protocol.json')
    corrected=CORRECTED/cohort
    correction=read(corrected/'protocol.json')
    sealed={}
    def seal(path):
        value=record(path);sealed[value['path']]=value;return value
    def check_completion(directory):
        completion=read(directory/'complete.json')
        require(completion['complete'],'Original experiment incomplete')
        for item in completion.get('artifacts',[])+completion.get('state_completions',[]):
            require(record(item['path'])==item,f'Original sealed artifact changed: {item["path"]}')
        seal(directory/'complete.json');seal(directory/'protocol.json')
    check_completion(original);check_completion(corrected)
    require(read(corrected/'complete.json')['protocol_hash']==core.json_hash(correction),'Corrected protocol hash differs')
    require(correction['original_protocol']==record(original/'protocol.json'),'Correction references another protocol')
    is_llama=model_name=='Llama'
    if is_llama:
        old_sources=old['checkpoints']
        full_panel=read(Path(old_sources[0]['requests']['path']))
        by_id={str(row['case_id']):row for row in full_panel}
        panel=[by_id[str(i)] for i in old['evaluation_case_ids']]
        batch=old['inference_batch_size'];model_path=str(llama.MODEL)
    else:
        old_sources=old['sources'];full_panel=old['panel']
        panel=[full_panel[i] for i in old['split']['positions']['evaluation']]
        batch=old['batch_size'];model_path=old_sources[0]['model_path']
    ids=[str(r['case_id']) for r in panel]
    require(len(ids)==len(set(ids))==100,'Expected frozen 100 evaluation cases')
    require(ids==[str(i) for i in correction['evaluation_case_ids']],'Corrected case IDs differ')
    require(batch==(2 if is_llama else 16),'Unexpected frozen batch')
    states=[]
    for editor in EDITORS:
        for method in METHODS:
            previous=next(s for s in old_sources if s['editor']==editor and s['method']==method)
            replace=editor=='AlphaEdit' and method=='SPHERE'
            if replace:
                identity=correction['checkpoint'];state_dir=corrected/editor/method
            elif is_llama:
                path=Path(previous['checkpoint_path'])
                require(path.stat().st_size==previous['checkpoint_bytes'] and path.stat().st_mtime_ns==previous['checkpoint_mtime_ns'],'Old checkpoint file changed')
                require(record(path.with_suffix('.json'))==previous['checkpoint_sidecar'],'Llama sidecar changed')
                identity=dict(checkpoint=record(path),sidecar=record(path.with_suffix('.json')),metadata=read(path.with_suffix('.json')))
                state_dir=original/('memit_worker' if editor=='MEMIT' else '')/editor/method
            else:
                identity=previous['identity'];state_dir=original/editor/method
            for field in ('checkpoint','sidecar'):
                require(record(identity[field]['path'])==identity[field],'Checkpoint identity changed')
                seal(identity[field]['path'])
            require(read(Path(identity['sidecar']['path']))==identity['metadata'],'Metadata differs')
            require(identity['metadata']['edit_count']==1000 and identity['metadata']['editing_method']==editor,'Wrong checkpoint endpoint')
            if is_llama:
                require(record(previous['requests']['path'])==previous['requests'],'Request source changed')
                require(read(Path(previous['requests']['path']))==full_panel,'Per-state request panels differ')
                seal(previous['requests']['path'])
                if replace:
                    require(identity['metadata']['request_fingerprint']==read(Path(previous['checkpoint_sidecar']['path']))['request_fingerprint'],'Corrected requests differ')
            elif replace:
                for key in ('requests_sha256','requests_prefix_sha256','model_seed','base_model','parameter_names','rewrite_layers'):
                    require(identity['metadata'][key]==previous['identity']['metadata'][key],'Corrected GPT metadata differs: '+key)
            if (state_dir/'complete.json').exists():
                seal(state_dir/'complete.json')
            seal(state_dir/'scores.csv');seal(state_dir/'sham_checks.json')
            for family in FAMILIES:
                seal(state_dir/f'predictions_{family}.json')
                if is_llama:
                    seal(state_dir/f'hidden_{family}.npz')
                else:
                    for condition in ('native','perp_f025','perp_f050'):
                        seal(state_dir/'geometry'/family/f'{condition}.npz')
            states.append(dict(model=model_name,dataset=dataset,editor=editor,method=method,order_id='canonical',edit_count=1000,
                checkpoint_path=identity['checkpoint']['path'],checkpoint=identity,source_state=str(state_dir.resolve()),corrected_sphere=replace))
    for family in FAMILIES:
        seal(original/'base'/f'{family}.json');seal(original/'base'/f'{family}.npz')
    # Validate against the actual published Figure 6, including corrected SPHERE.
    figure_rows=read_csv(FIGURE6); figure_checks=[]
    for source in states:
        frozen=read_csv(original_state(source)/'scores.csv')
        ref=next(r for r in figure_rows if all(str(r[k])==str(source[k]) for k in ('model','dataset','editor','method')))
        for family,metric in [('locality','LOC'),('rewrite','EFF_TF'),('rephrase','GEN_TF')]:
            values={(r['condition'],str(r['case_id'])):float(r['endpoint_value']) for r in frozen if r['family']==family}
            for fraction in FRACTIONS:
                delta=float(np.mean([(values[label('perp',fraction),cid]-values['native',cid])*100 for cid in ids]))
                expected=float(ref[f'delta_{metric}_{round(fraction*100)}'])
                require(abs(delta-expected)<1e-12,'Frozen source does not reproduce Figure 6')
                figure_checks.append(dict(editor=source['editor'],method=source['method'],family=family,dose_fraction=fraction,delta_pp=delta,figure6_delta_pp=expected))
    seal(FIGURE6)
    protocol=dict(schema_version=1,purpose='Figure 7 extension to all Figure 6 cohorts; old Llama-CounterFact reused separately',
        cohort=cohort,model=model_name,dataset=dataset,dtype='bfloat16' if is_llama else 'float32',
        model_path=model_path,attn_implementation='eager',inference_batch_size=batch,
        state_count=10,n_cases_per_family=100,families=list(FAMILIES),fractions=list(FRACTIONS),
        evaluation_case_ids=[r['case_id'] for r in panel],panel=panel,sources=states,original_base=str((original/'base').resolve()),
        boundary='Original prompt-last '+('H9 pre-RMSNorm' if is_llama else 'H18 pre-LayerNorm')+'; downstream live',
        control_A='h-f*Delta_parallel; edit-displacement attenuation, not norm-matched',
        control_B='b+sign(a)*sqrt(a^2-(2f-f^2)||b||^2)*u; preserve orthogonal component and sign; shrink total Base-axis projection to target norm',
        B_target_norm='Intended F64 norm of h-f*Delta_perp; f=0.25/0.50',
        feasibility='Exact F64 radicand >=0; no clamping; infeasible batch rows retain native and are excluded from B results',
        comparison='Same case-ID subset for native/perp/A/B; B_common_all_families intersects rewrite/rephrase/locality independently by state and dose',
        metrics='Case-macro target-token TF EFF/GEN including genuine EOS/EOT; LOC same-span Base argmax agreement',
        bootstrap_seed=20260912,n_bootstrap=10000,bootstrap='Pointwise paired case percentile intervals',
        no_new_editing=True,no_original_artifact_modification=True,no_axis_fit=True,
        preserved_figure6_checks=figure_checks,original_artifacts=list(sealed.values()),
        code=record(Path(__file__)),helper_sources=[record(Path(module.__file__)) for module in (core,gpt,llama,prior)])
    return states,panel,protocol


def ensure_protocol(protocol):
    path=OUT/'protocol.json'
    if path.exists():
        require(read(path)==protocol,'Existing protocol differs')
    else:
        write_json(path,protocol)


def verify_base(baseline, protocol):
    reference=Path(protocol['original_base'])
    for family,current in baseline.items():
        previous=read(reference/f'{family}.json')
        require(current['proofs']==previous['proofs'] and current['rows']==previous['rows'],'Regenerated Base predictions/tokenization differ')
        key='hidden' if IS_LLAMA else 'h18'
        require(np.array_equal(current['h'],np.load(reference/f'{family}.npz')[key]),'Regenerated Base hidden differs')

def score_row(source, family, condition, dose, case_id, measured, proof, provenance):
    gold = np.asarray(proof['target_token_ids'])
    predicted = np.asarray(measured['predicted_token_ids'])
    require(gold.shape == predicted.shape, 'Prediction span changed')
    correct = gold == predicted
    semantic, terminator = None, None
    if family != 'locality':
        require(gold[-1] == (128009 if IS_LLAMA else 50256) and (gold == (128009 if IS_LLAMA else 50256)).sum() == 1, 'Invalid genuine EOT')
        semantic, terminator = float(correct[:-1].mean()), float(correct[-1])
    return dict(**{k: source[k] for k in KEYS}, family=family, endpoint=legacy.ENDPOINTS[family],
                condition=condition, dose_fraction=dose, case_id=case_id,
                endpoint_value=measured['base_agreement'] if family == 'locality' else measured['accuracy'],
                accuracy=measured['accuracy'], base_agreement=measured['base_agreement'],
                semantic_accuracy=semantic, terminator_accuracy=terminator, n_target_tokens=len(gold),
                source=provenance)

@torch.inference_mode()
def run_state(model, tok, panel, baseline, source, dest):
    old_dir = original_state(source)
    old_scores = {(r['family'], r['condition'], str(r['case_id'])): r for r in read_csv(old_dir/'scores.csv')}
    all_scores, geometry_rows, feasibility, checks = [], [], [], []
    for family in FAMILIES:
        old_hidden = load_frozen_hidden(source, family)
        require([str(i) for i in old_hidden['case_ids']]==[str(r['case_id']) for r in panel], 'Frozen case order differs')
        old_predictions = {(r['condition'], str(r['case_id'])): r for r in read(old_dir/f'predictions_{family}.json')}
        saved_hidden = {c: old_hidden[c].copy() for c in ('native', 'perp_f025', 'perp_f050')}
        for fraction in FRACTIONS:
            for kind in ('delta_parallel', 'projection_match'):
                saved_hidden[label(kind, fraction)] = old_hidden['native'].copy()
            saved_hidden[label('eligible_projection_match', fraction)] = np.zeros(100, dtype=bool)
        predictions = []
        for start in range(0, 100, BATCH):
            chunk = panel[start:start+BATCH]
            ids = [r['case_id'] for r in chunk]
            inputs, positions, proofs = legacy.materialize(tok, chunk, family, next(model.parameters()).device)
            require(proofs == baseline[family]['proofs'][start:start+len(chunk)], 'Family tokenization differs')
            native_logits, h, _ = legacy.forward(model, inputs, positions)
            h0 = baseline[family]['h'][start:start+len(chunk)]
            require(np.array_equal(h, old_hidden['native'][start:start+len(chunk)]), 'New native capture differs from saved boundary')
            base_predictions = [r['predicted_token_ids'] for r in baseline[family]['rows'][start:start+len(chunk)]]
            native_rows = legacy.prediction_rows(native_logits, positions, proofs, base_predictions)
            sham_logits, sham_before, sham_after = legacy.forward(model, inputs, positions, h)
            sham_error = float((native_logits-sham_logits).abs().max().item())
            require(sham_error == 0 and np.array_equal(sham_before, sham_after), 'Sham changed logits/hidden')
            checks.append(dict(family=family, case_ids=ids, saved_native_hidden_exact=True,
                               native_predictions_exact=True, sham_max_logit_difference=sham_error))
            del native_logits, sham_logits
            for i, case_id in enumerate(ids):
                require(native_rows[i] == {k: v for k, v in old_predictions['native', str(case_id)].items()
                                           if k not in ('case_id', 'condition')}, 'New native prediction differs from saved prediction')
                for condition, fraction in [('native', 0.), ('perp_f025', .25), ('perp_f050', .5)]:
                    row = old_predictions[condition, str(case_id)]
                    measured = {k: row[k] for k in ('accuracy', 'base_agreement', 'predicted_token_ids')}
                    frozen = old_scores[family, condition, str(case_id)]
                    require(measured['accuracy'] == float(frozen['accuracy']) and
                            measured['base_agreement'] == float(frozen['base_agreement']), 'Frozen prediction/score differs')
                    all_scores.append(score_row(source, family, condition, fraction, case_id, measured, proofs[i], 'frozen_main_rq3_reused'))
                    predictions.append(dict(case_id=case_id, condition=condition, **measured))
            for fraction in FRACTIONS:
                construction = construct(h, h0, fraction)
                ref_condition = label('perp', fraction)
                ref_actual=old_hidden[ref_condition][start:start+len(chunk)]
                require(np.array_equal(ref_actual,torch.from_numpy(construction['desired_perp']).to(EXEC_DTYPE).float().numpy()),'Frozen perpendicular hidden does not match reconstruction')
                ref_geo=legacy.geometry(h,h0,ref_actual,construction['desired_perp'])
                for i, case_id in enumerate(ids):
                    feasibility.append(dict(**{k: source[k] for k in KEYS}, family=family, dose_fraction=fraction, case_id=case_id,
                                            feasible=bool(construction['feasible'][i]), radicand=float(construction['radicand'][i]),
                                            normalized_radicand=float(construction['normalized_radicand'][i]),
                                            signed_total_projection=float(construction['signed_projection'][i]),
                                            perpendicular_norm=float(construction['perpendicular_norm'][i]),
                                            target_norm=float(construction['desired_norm'][i]),
                                            projection_scale=float(construction['projection_scale'][i]) if construction['feasible'][i] else None))
                saved_hidden[label('eligible_projection_match', fraction)][start:start+len(chunk)] = construction['feasible']
                for kind, key in [('delta_parallel', 'A'), ('projection_match', 'B')]:
                    condition = label(kind, fraction)
                    eligible = np.ones(len(chunk), dtype=bool) if key == 'A' else construction['feasible']
                    if not eligible.any():
                        continue
                    intended = construction[key]
                    logits, before, actual = legacy.forward(model, inputs, positions, intended)
                    require(np.array_equal(before, h), 'Checkpoint native boundary changed')
                    require(np.array_equal(actual, torch.from_numpy(intended).to(EXEC_DTYPE).float().numpy()),
                            'Actual execution-dtype replacement differs from intended rounding')
                    measured = legacy.prediction_rows(logits, positions, proofs, base_predictions)
                    del logits
                    saved_hidden[condition][start:start+len(chunk)] = actual
                    geo = legacy.geometry(h, h0, actual, intended)
                    for i, case_id in enumerate(ids):
                        if not eligible[i]:
                            continue
                        frozen = old_scores[family, ref_condition, str(case_id)]
                        desired_norm = float(ref_geo['intended_hidden_norm'][i])
                        realized_ref_norm = float(ref_geo['realized_hidden_norm'][i])
                        intended_norm_error = abs(float(geo['intended_hidden_norm'][i]) - desired_norm)
                        norm_error = abs(float(geo['realized_hidden_norm'][i]) - realized_ref_norm)
                        rounding_bound = float(geo['rounding_l2'][i]) + float(ref_geo['rounding_l2'][i])
                        scale = max(float(geo['realized_hidden_norm'][i]), realized_ref_norm, 1.)
                        if key == 'B':
                            require(intended_norm_error <= 1e-9 * max(desired_norm, 1.), 'B target norm differs from frozen perp')
                            require(norm_error <= intended_norm_error + rounding_bound + 1e-8*scale,
                                    'B realized norm mismatch exceeds execution-dtype rounding bound')
                        all_scores.append(score_row(source, family, condition, fraction, case_id, measured[i], proofs[i], 'new_parallel_control_inference'))
                        predictions.append(dict(case_id=case_id, condition=condition, **measured[i]))
                        geometry_rows.append(dict(**{k: source[k] for k in KEYS}, family=family, condition=condition,
                            dose_fraction=fraction, case_id=case_id, norm_matching_required=key == 'B',
                            reference_condition=ref_condition, reference_intended_norm=desired_norm, reference_realized_norm=realized_ref_norm,
                            intended_norm_difference=intended_norm_error, realized_norm_difference=norm_error,
                            realized_norm_scaled_error=norm_error/scale, rounding_l2_bound=rounding_bound,
                            rounding_bound_holds=norm_error <= intended_norm_error+rounding_bound+1e-8*scale,
                            **{field: float(values[i]) for field, values in geo.items()}))
            if (start+len(chunk)) % 20 == 0:
                print(f'[parallel] {source["editor"]} {source["method"]} {family} {start+len(chunk)}/100', flush=True)
        write_json(dest/f'predictions_{family}.json', predictions)
        write_npz(dest/f'hidden_{family}.npz', case_ids=np.asarray([r['case_id'] for r in panel]), **saved_hidden)
    write_csv(dest/'per_case_scores.csv', all_scores)
    write_csv(dest/'geometry.csv', geometry_rows)
    write_csv(dest/'feasibility.csv', feasibility)
    write_json(dest/'runtime_checks.json', checks)
    return dict(score_rows=len(all_scores), geometry_rows=len(geometry_rows), feasibility_rows=len(feasibility), sham_batches=len(checks))

def aggregate(states, protocol, shard_count):
    protocol_hash = core.json_hash(protocol)
    ids = [str(i) for i in protocol['evaluation_case_ids']]
    all_scores, all_geo, all_feasible = [], [], []
    completions = []
    for index, source in enumerate(states):
        dest = OUT / f'worker_{index % shard_count}' / source['editor'] / source['method']
        done = read(dest/'complete.json')
        require(done['complete'] and done['protocol_hash'] == protocol_hash and done['weights_unchanged'], 'Incomplete pilot state')
        for artifact in done['artifacts']:
            require(record(artifact['path']) == artifact, 'Pilot state artifact changed')
        completions.append(record(dest/'complete.json'))
        all_scores.extend(read_csv(dest/'per_case_scores.csv'))
        all_geo.extend(read_csv(dest/'geometry.csv'))
        all_feasible.extend(read_csv(dest/'feasibility.csv'))
    require(len(all_feasible) == 6000, 'Incomplete feasibility grid')
    table = {(r['editor'], r['method'], r['family'], r['condition'], str(r['case_id'])): r for r in all_scores}
    require(len(table) == len(all_scores), 'Duplicate scored cases')
    feasible = {(r['editor'], r['method'], r['family'], float(r['dose_fraction']), str(r['case_id'])): r['feasible'] == 'True'
                for r in all_feasible}
    summary, contrasts, availability, common_wide, a_wide = [], [], [], [], []
    def estimate(values):
        values = np.asarray(values, np.float64)
        if not len(values):
            return None, None, None
        indices = np.random.default_rng(20260912).integers(0, len(values), size=(10000, len(values)))
        low, high = np.percentile(values[indices].mean(axis=1), [2.5, 97.5])
        return float(values.mean()), float(low), float(high)
    for source in states:
        editor, method = source['editor'], source['method']
        identity = {k: source[k] for k in KEYS}
        for fraction in FRACTIONS:
            masks = {f: [i for i in ids if feasible[editor, method, f, fraction, i]] for f in FAMILIES}
            common = [i for i in ids if all(i in masks[f] for f in FAMILIES)]
            cw = dict(**identity, dose_fraction=fraction, n_common_cases=len(common))
            aw = dict(**identity, dose_fraction=fraction, n_cases=100)
            perp, a, b = [label(prefix, fraction) for prefix in ('perp', 'delta_parallel', 'projection_match')]
            for family in FAMILIES:
                availability.append(dict(**identity, dose_fraction=fraction, family=family,
                                         n_total=100, n_feasible=len(masks[family]), n_infeasible=100-len(masks[family]),
                                         n_common_cases=len(common), family_case_ids='|'.join(masks[family]), common_case_ids='|'.join(common)))
                for scope, selected, conditions in [('A_all100', ids, ('native', perp, a)),
                                                    ('B_family_feasible', masks[family], ('native', perp, a, b)),
                                                    ('B_common_all_families', common, ('native', perp, a, b))]:
                    values = {c: np.asarray([float(table[editor, method, family, c, i]['endpoint_value'])*100 for i in selected])
                              for c in conditions}
                    for condition in conditions:
                        avg, low, high = estimate(values[condition] - values['native'])
                        summary.append(dict(**identity, dose_fraction=fraction, scope=scope, family=family,
                                            endpoint=legacy.ENDPOINTS[family], condition=condition, n_cases=len(selected),
                                            baseline_pct=float(values['native'].mean()) if selected else None,
                                            endpoint_pct=float(values[condition].mean()) if selected else None,
                                            delta_pp=avg, ci_low_pp=low, ci_high_pp=high))
                        if scope == 'B_common_all_families':
                            prefix = {'native': 'native', perp: 'perp', a: 'delta_parallel', b: 'projection_match'}[condition]
                            cw[f'{prefix}_{legacy.ENDPOINTS[family]}_pct'] = float(values[condition].mean()) if selected else None
                            cw[f'{prefix}_{legacy.ENDPOINTS[family]}_delta_pp'] = avg
                        if scope == 'A_all100':
                            prefix = {'native': 'native', perp: 'perp', a: 'delta_parallel'}[condition]
                            aw[f'{prefix}_{legacy.ENDPOINTS[family]}_delta_pp'] = avg
                    comparison_pairs = [(perp, 'native'), (a, 'native'), (a, perp)]
                    if b in conditions:
                        comparison_pairs += [(b, 'native'), (b, perp)]
                    for lhs, rhs in comparison_pairs:
                        avg, low, high = estimate(values[lhs] - values[rhs])
                        contrasts.append(dict(**identity, dose_fraction=fraction, scope=scope, family=family,
                            endpoint=legacy.ENDPOINTS[family], lhs=lhs, rhs=rhs, n_cases=len(selected),
                            mean_difference_pp=avg, ci_low_pp=low, ci_high_pp=high))
                        if scope == 'B_common_all_families' and lhs == b and rhs == perp:
                            cw[f'projection_minus_perp_{legacy.ENDPOINTS[family]}_pp'] = avg
                            cw[f'projection_minus_perp_{legacy.ENDPOINTS[family]}_ci_low_pp'] = low
                            cw[f'projection_minus_perp_{legacy.ENDPOINTS[family]}_ci_high_pp'] = high
            common_wide.append(cw); a_wide.append(aw)
    for filename, rows in [('per_case_scores.csv', all_scores), ('geometry.csv', all_geo), ('feasibility.csv', all_feasible),
                           ('availability_summary.csv', availability), ('condition_summary.csv', summary), ('paired_contrasts.csv', contrasts),
                           ('pilot_common_case_comparison.csv', common_wide), ('auxiliary_delta_parallel_all100.csv', a_wide)]:
        write_csv(OUT/filename, rows)
    bgeo = [r for r in all_geo if r['norm_matching_required'] == 'True']
    require(all(r['rounding_bound_holds'] == 'True' for r in bgeo), 'A B-control rounding bound failed')
    audit = dict(n_states=10, n_feasibility_rows=6000, n_valid_B_family_cases=len(bgeo),
                 max_B_intended_scaled_norm_error=max(float(r['intended_norm_difference'])/max(float(r['reference_intended_norm']),1.) for r in bgeo),
                 max_B_realized_scaled_norm_error=max(float(r['realized_norm_scaled_error']) for r in bgeo),
                 all_B_rounding_bounds_hold=True, common_case_counts=[dict(editor=r['editor'], method=r['method'],
                     dose_fraction=r['dose_fraction'], n_cases=r['n_common_cases']) for r in common_wide],
                 note='B only compares feasible cases; zero-feasibility states have blank scores and intervals. A is not norm-matched.')
    write_json(OUT/'aggregate_validation.json', audit)
    artifacts = [record(p) for p in sorted(OUT.glob('*.csv'))] + [record(OUT/'aggregate_validation.json')]
    write_json(OUT/'complete.json', dict(complete=True, n_states=10, protocol_hash=protocol_hash,
        no_original_artifact_modification=True, state_completions=completions, artifacts=artifacts))
    print('[parallel aggregate complete]', json.dumps(audit), flush=True)


def main():
    global OUT,BATCH,EXEC_DTYPE,IS_LLAMA,legacy
    parser=argparse.ArgumentParser()
    parser.add_argument('--cohort',required=True,choices=COHORTS)
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--aggregate-only',action='store_true')
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--shard-count',type=int,default=1)
    args=parser.parse_args()
    require(args.shard_count>0 and 0<=args.shard_index<args.shard_count,'Invalid shard')
    torch.set_num_threads(4)
    IS_LLAMA=args.cohort.startswith('llama')
    EXEC_DTYPE=torch.bfloat16 if IS_LLAMA else torch.float32
    OUT=OUTPUT/args.cohort;legacy=Backend()
    states,panel,protocol=audit_sources(args.cohort)
    BATCH=protocol['inference_batch_size']
    ensure_protocol(protocol)
    if args.audit_only:
        print(json.dumps(dict(audit_complete=True,cohort=args.cohort,n_states=10,n_cases=100,figure6_values_verified=len(protocol['preserved_figure6_checks']),protocol_hash=core.json_hash(protocol))))
        return
    if args.aggregate_only:
        aggregate(states,protocol,args.shard_count);return
    require(torch.cuda.is_available(),'CUDA unavailable')
    torch.manual_seed(42)
    worker=OUT/f'worker_{args.shard_index}'
    if IS_LLAMA:
        llama.OUT=worker
        model=llama.load_model(protocol['model_path'],'bfloat16','cuda:0',False,'eager')
        tok=AutoTokenizer.from_pretrained(protocol['model_path'],local_files_only=True)
        if tok.pad_token_id is None:tok.pad_token=tok.eos_token
        baseline=llama.capture_base(model,tok,panel,BATCH)
    else:
        model,tok=core.load_model_and_tokenizer(Path(protocol['model_path']),device='cuda:0',dtype='float32')
        baseline=gpt.capture_base(model,tok,panel,BATCH,worker)
    verify_base(baseline,protocol)
    names=states[0]['checkpoint']['metadata']['parameter_names']
    require(all(s['checkpoint']['metadata']['parameter_names']==names for s in states),'Checkpoint parameter sets differ')
    parameters=dict(model.named_parameters())
    pristine={name:parameters[name].detach().cpu().clone() for name in names}
    for index,source in enumerate(states):
        if index%args.shard_count!=args.shard_index:continue
        dest=worker/source['editor']/source['method']
        if (dest/'complete.json').exists():
            done=read(dest/'complete.json')
            require(done['complete'] and done['protocol_hash']==core.json_hash(protocol),'Completed state protocol differs')
            for artifact in done['artifacts']:require(record(artifact['path'])==artifact,'Completed artifact changed')
            continue
        began=time.monotonic()
        if IS_LLAMA:
            with torch.no_grad():
                for name,tensor in pristine.items():parameters[name].copy_(tensor)
            metadata=llama.ev.load_edited_parameter_checkpoint(model,Path(source['checkpoint_path']))
            require(metadata==source['checkpoint']['metadata'],'Loaded checkpoint metadata differs')
        else:
            core.restore_checkpoint(model,pristine,Path(source['checkpoint_path']),source['checkpoint'])
        versions=tuple((name,tensor._version) for name,tensor in model.named_parameters())
        counts=run_state(model,tok,panel,baseline,source,dest)
        require(versions==tuple((name,tensor._version) for name,tensor in model.named_parameters()),'Intervention modified weights')
        artifacts=[record(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='complete.json']
        write_json(dest/'complete.json',dict(complete=True,**{k:source[k] for k in KEYS},protocol_hash=core.json_hash(protocol),original_native_reproduced_exactly=True,weights_unchanged=True,**counts,elapsed_seconds=time.monotonic()-began,artifacts=artifacts))
        print('[parallel state complete]',args.cohort,source['editor'],source['method'],f'{time.monotonic()-began:.1f}s',flush=True)
    write_json(worker/'complete.json',dict(complete=True,shard_index=args.shard_index,shard_count=args.shard_count,protocol_hash=core.json_hash(protocol),frozen_Base_reproduced_exactly=True))

if __name__=='__main__':main()
