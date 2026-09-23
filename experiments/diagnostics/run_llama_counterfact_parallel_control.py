#!/usr/bin/env python3
"""Standalone parallel-component controls for frozen Llama/CounterFact RQ3.

A attenuates only edited-minus-Base parallel displacement by 25/50 percent.
B attenuates the entire Base-axis projection to match the intended norm of
perpendicular removal at the same dose. B is evaluated only where its exact
nonnegative radicand permits this operation. Original RQ3 files are read-only.
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
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from transformers import AutoTokenizer
from diagnostics import run_llama_intervention_extension as legacy
from diagnostics import gpt2_checkpoint_analysis as core

CROSS = OUTPUT_ROOT / '_Analysis_Cross_Layer'
ORIGINAL = CROSS / 'llama_counterfact_intervention_20260923'
OUT = CROSS / 'llama_counterfact_parallel_control_20260923'
FAMILIES = ('rewrite', 'rephrase', 'locality')
METHODS = ('Native', 'NAS', 'ENCORE', 'SPHERE', 'SADR')
EDITORS = ('AlphaEdit', 'MEMIT')
FRACTIONS = (.25, .5)
KEYS = ('model', 'dataset', 'editor', 'method', 'order_id', 'edit_count')
read, record, require, write_json, write_npz = core.read, core.record, core.require, core.write_json, core.write_npz
write_csv = legacy.write_csv


def record(path):
    return core.record(Path(path))


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def original_state(source):
    found = list(ORIGINAL.glob(f'worker_*/{source["editor"]}/{source["method"]}/complete.json'))
    require(len(found) == 1, 'Expected exactly one frozen source state')
    return found[0].parent


def label(prefix, fraction):
    return f'{prefix}_f{round(fraction*100):03d}'


def construct(h, h0, fraction):
    """No radicand clipping: infeasible B rows receive native batch placeholders."""
    h, h0 = np.asarray(h, np.float64), np.asarray(h0, np.float64)
    n0 = np.linalg.norm(h0, axis=-1)
    require((n0 > 0).all(), 'Zero Base hidden norm')
    u = h0 / n0[:, None]
    delta = h - h0
    delta_parallel = np.sum(delta * u, axis=-1, keepdims=True) * u
    delta_perpendicular = delta - delta_parallel
    a = np.sum(h * u, axis=-1)
    b = h - a[:, None] * u
    b_squared = np.sum(b * b, axis=-1)
    a_squared = a * a
    radicand = a_squared - (2 * fraction - fraction * fraction) * b_squared
    feasible = radicand >= 0
    desired_perp = h - fraction * delta_perpendicular
    desired_norm = np.linalg.norm(desired_perp, axis=-1)
    target_a = h - fraction * delta_parallel
    target_b = h.copy()
    signed_projection = np.copysign(np.sqrt(radicand[feasible]), a[feasible])
    target_b[feasible] = b[feasible] + signed_projection[:, None] * u[feasible]
    scales = np.full(len(h), np.nan)
    nonzero = feasible & (a != 0)
    scales[nonzero] = np.sqrt(radicand[nonzero]) / np.abs(a[nonzero])
    scales[feasible & (a == 0)] = 1.
    require((scales[feasible] >= 0).all() and (scales[feasible] <= 1 + 1e-12).all(),
            'B did not attenuate the total projection')
    # A preserves the perpendicular component before BF16 rounding, but its
    # resulting total norm is not constrained and can increase when p < 0.
    _, _, _, a_perpendicular = legacy.decompose(target_a, h0)
    require(np.allclose(a_perpendicular, delta_perpendicular, rtol=1e-10, atol=1e-8),
            'A changed intended perpendicular displacement')
    if feasible.any():
        error = np.abs(np.linalg.norm(target_b[feasible], axis=-1) - desired_norm[feasible])
        require((error <= 1e-9 * np.maximum(desired_norm[feasible], 1.)).all(), 'B intended norm mismatch')
    return dict(A=target_a, B=target_b, feasible=feasible, radicand=radicand,
                normalized_radicand=radicand / np.maximum(a_squared + b_squared, 1e-300),
                signed_projection=a, perpendicular_norm=np.sqrt(b_squared), projection_scale=scales,
                desired_perp=desired_perp, desired_norm=desired_norm)


def audit_sources():
    old_protocol = read(ORIGINAL / 'protocol.json')
    old_complete = read(ORIGINAL / 'complete.json')
    validated = read(ORIGINAL / 'validation.json')
    require(old_complete['complete'] and validated['passed'], 'Original experiment is not validated and sealed')
    require(old_complete['protocol_hash'] == core.json_hash(old_protocol), 'Original protocol hash mismatch')
    for artifact in old_complete['artifacts'] + old_complete['state_completions']:
        require(record(artifact['path']) == artifact, 'Frozen original artifact changed')
    states = read(ORIGINAL / 'source_manifest.json')['states']
    states.sort(key=lambda r: (EDITORS.index(r['editor']), METHODS.index(r['method'])))
    require(len(states) == 10, 'Expected ten endpoint states')
    requests = read(Path(states[0]['requests_source']))
    panel = [requests[int(i)] for i in old_protocol['split']['positions']['evaluation']]
    require([str(r['case_id']) for r in panel] == [str(i) for i in old_protocol['split']['evaluation_case_ids']],
            'Frozen evaluation case order changed')
    identities = []
    for source in states:
        require(read(Path(source['requests_source'])) == requests, 'Source requests differ')
        old_source = next(s for s in old_protocol['sources']
                          if s['editor'] == source['editor'] and s['method'] == source['method'])
        require(read(Path(source['checkpoint_path']).with_suffix('.json')) == old_source['metadata'],
                'Checkpoint metadata changed')
        state_dir = original_state(source)
        completion = read(state_dir / 'complete.json')
        require(completion['protocol_hash'] == core.json_hash(old_protocol), 'Frozen state protocol differs')
        identities.append(dict(**{k: source[k] for k in KEYS}, checkpoint=old_source,
                               source_state=str(state_dir), source_completion=record(state_dir/'complete.json')))
    protocol = dict(schema_version=1, purpose='Standalone parallel-component pilot, kept separate from frozen main RQ3',
                    model='Llama', dataset='CounterFact', dtype='bfloat16', attn_implementation='eager',
                    model_path=str(legacy.MODEL), inference_batch_size=2, fractions=list(FRACTIONS),
                    state_count=10, n_cases_per_family=100, families=list(FAMILIES), evaluation_case_ids=[r['case_id'] for r in panel],
                    original_protocol=record(ORIGINAL/'protocol.json'), original_complete=record(ORIGINAL/'complete.json'),
                    original_validation=record(ORIGINAL/'validation.json'), sources=identities,
                    boundary='Original prompt-last H9, family-specific Base and edited hidden states; downstream computation unchanged',
                    control_A='h_A=h-f*Delta_parallel; attenuates edited-minus-Base parallel displacement; NOT norm-matched and may increase total norm',
                    control_B='h_B=b+sign(a)*sqrt(a^2-(2f-f^2)||b||^2)*u, u=h0/||h0||, a=h dot u, b=h-a*u',
                    B_target_norm='Same intended final hidden norm as perpendicular removal h-f*Delta_perp at f=0.25 or 0.5',
                    B_feasibility='Exact float64 radicand >= 0; no negative-radicand clamp. All radicands and masks retained.',
                    B_infeasible_batch_rows='Native hidden placeholders preserve frozen batch=2; excluded from B scores and every B comparison',
                    comparisons='A: all100. B: family-feasible subset plus rewrite/rephrase/locality common case-ID intersection separately by state and dose. Native/perp/A/B compared on exactly the same selected IDs.',
                    metrics='Case-macro target-token TF EFF/GEN with genuine EOT; LOC is case-macro Base argmax agreement without appended EOT',
                    bootstrap_seed=20260912, n_bootstrap=10000, bootstrap='Pointwise paired case percentile intervals; shared case IDs/index across endpoints on common subsets; no noninferiority claim',
                    no_new_editing=True, no_original_artifact_modification=True, no_new_axis_fit=True,
                    rounding='Intended float64 norm match plus realized BF16 norm difference bounded by both interventions rounding L2',
                    code=record(Path(__file__)), helper_sources=[record(Path(legacy.__file__)), record(Path(legacy.legacy_materialize.__code__.co_filename)),
                                                              record(ROOT/'diagnostics/gpt2_intervention_analysis.py')])
    return states, panel, protocol


def ensure_protocol(protocol):
    path = OUT / 'protocol.json'
    if path.exists():
        require(read(path) == protocol, 'Existing pilot protocol differs')
    else:
        write_json(path, protocol)


def verify_base(baseline):
    reference = ORIGINAL / 'worker_0/base'
    for family, current in baseline.items():
        previous = read(reference / f'{family}.json')
        require(current['proofs'] == previous['proofs'] and current['rows'] == previous['rows'],
                'Regenerated Base tokenization or predictions differ from frozen main RQ3')
        require(np.array_equal(current['h'], np.load(reference/f'{family}.npz')['hidden']),
                'Regenerated Base hidden differs from frozen main RQ3')


def score_row(source, family, condition, dose, case_id, measured, proof, provenance):
    gold = np.asarray(proof['target_token_ids'])
    predicted = np.asarray(measured['predicted_token_ids'])
    require(gold.shape == predicted.shape, 'Prediction span changed')
    correct = gold == predicted
    semantic, terminator = None, None
    if family != 'locality':
        require(gold[-1] == 128009 and (gold == 128009).sum() == 1, 'Invalid genuine EOT')
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
        old_hidden = np.load(old_dir / f'hidden_{family}.npz')
        old_predictions = {(r['condition'], str(r['case_id'])): r for r in read(old_dir/f'predictions_{family}.json')}
        saved_hidden = {c: old_hidden[c].copy() for c in ('native', 'perp_f025', 'perp_f050')}
        for fraction in FRACTIONS:
            for kind in ('delta_parallel', 'projection_match'):
                saved_hidden[label(kind, fraction)] = old_hidden['native'].copy()
            saved_hidden[label('eligible_projection_match', fraction)] = np.zeros(100, dtype=bool)
        predictions = []
        for start in range(0, 100, 2):
            chunk = panel[start:start+2]
            ids = [r['case_id'] for r in chunk]
            inputs, positions, proofs = legacy.materialize(tok, chunk, family, next(model.parameters()).device)
            require(proofs == baseline[family]['proofs'][start:start+len(chunk)], 'Family tokenization differs')
            native_logits, h, _ = legacy.forward(model, inputs, positions)
            h0 = baseline[family]['h'][start:start+len(chunk)]
            require(np.array_equal(h, old_hidden['native'][start:start+len(chunk)]), 'New native capture differs from saved H9')
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
                    require(np.array_equal(actual, torch.from_numpy(intended).to(torch.bfloat16).float().numpy()),
                            'Actual BF16 replacement differs from intended rounding')
                    measured = legacy.prediction_rows(logits, positions, proofs, base_predictions)
                    del logits
                    saved_hidden[condition][start:start+len(chunk)] = actual
                    geo = legacy.geometry(h, h0, actual, intended)
                    for i, case_id in enumerate(ids):
                        if not eligible[i]:
                            continue
                        frozen = old_scores[family, ref_condition, str(case_id)]
                        desired_norm = float(frozen['intended_hidden_norm'])
                        realized_ref_norm = float(frozen['realized_hidden_norm'])
                        intended_norm_error = abs(float(geo['intended_hidden_norm'][i]) - desired_norm)
                        norm_error = abs(float(geo['realized_hidden_norm'][i]) - realized_ref_norm)
                        rounding_bound = float(geo['rounding_l2'][i]) + float(frozen['rounding_l2'])
                        scale = max(float(geo['realized_hidden_norm'][i]), realized_ref_norm, 1.)
                        if key == 'B':
                            require(intended_norm_error <= 1e-9 * max(desired_norm, 1.), 'B target norm differs from frozen perp')
                            require(norm_error <= intended_norm_error + rounding_bound + 1e-8*scale,
                                    'B realized norm mismatch exceeds BF16 rounding bound')
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=3)
    args = parser.parse_args()
    require(args.shard_count > 0 and 0 <= args.shard_index < args.shard_count, 'Invalid shard')
    torch.set_num_threads(4)
    states, panel, protocol = audit_sources()
    ensure_protocol(protocol)
    if args.audit_only:
        print(json.dumps(dict(audit_complete=True, n_states=10, n_cases=100, protocol_hash=core.json_hash(protocol))))
        return
    if args.aggregate_only:
        aggregate(states, protocol, args.shard_count)
        return
    require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.manual_seed(42)
    worker = OUT / f'worker_{args.shard_index}'
    legacy.OUT = worker
    model = legacy.load_model(str(legacy.MODEL), 'bfloat16', 'cuda:0', False, 'eager')
    tok = AutoTokenizer.from_pretrained(legacy.MODEL, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    names = protocol['sources'][0]['checkpoint']['metadata']['parameter_names']
    pristine = legacy.ev.CHECKPOINT_MODULE.snapshot_parameters(model, names)
    baseline = legacy.capture_base(model, tok, panel, 2)
    verify_base(baseline)
    for index, source in enumerate(states):
        if index % args.shard_count != args.shard_index:
            continue
        dest = worker / source['editor'] / source['method']
        if (dest/'complete.json').exists():
            done = read(dest/'complete.json')
            require(done['protocol_hash'] == core.json_hash(protocol), 'Completed pilot protocol differs')
            for artifact in done['artifacts']:
                require(record(artifact['path']) == artifact, 'Completed pilot artifact changed')
            continue
        began = time.monotonic()
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, tensor in pristine.items():
                parameters[name].copy_(tensor)
        metadata = legacy.ev.load_edited_parameter_checkpoint(model, Path(source['checkpoint_path']))
        require(metadata == protocol['sources'][index]['checkpoint']['metadata'], 'Loaded checkpoint metadata differs')
        versions = tuple((name, tensor._version) for name, tensor in model.named_parameters())
        counts = run_state(model, tok, panel, baseline, source, dest)
        require(versions == tuple((name, tensor._version) for name, tensor in model.named_parameters()), 'Pilot modified weights')
        artifacts = [record(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name != 'complete.json']
        write_json(dest/'complete.json', dict(complete=True, **{k: source[k] for k in KEYS},
            protocol_hash=core.json_hash(protocol), original_native_reproduced_exactly=True, weights_unchanged=True,
            **counts, elapsed_seconds=time.monotonic()-began, artifacts=artifacts))
        print('[parallel state complete]', source['editor'], source['method'], f'{time.monotonic()-began:.1f}s', flush=True)
    write_json(worker/'complete.json', dict(complete=True, shard_index=args.shard_index, shard_count=args.shard_count,
                                         protocol_hash=core.json_hash(protocol), frozen_Base_reproduced_exactly=True))


if __name__ == '__main__':
    main()
