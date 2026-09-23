#!/usr/bin/env python3
"""Canonical Llama H9 interventions, protocol v2; saved states, no editing."""
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
sys.path.insert(0, str(ROOT / 'evaluate'))
import numpy as np
import torch
from transformers import AutoTokenizer
import eval_cumulative_generation_locality as ev
from diagnostics.analyze_residual_spectrum import load_model
from diagnostics.evaluate_matched_geometry_corrected import materialize as legacy_materialize
from diagnostics.gpt2_checkpoint_analysis import read, record, require, write_json, write_npz
from diagnostics.gpt2_intervention_analysis import conditions, decompose, fit_axis, prediction_rows, write_csv

CROSS = OUTPUT_ROOT / '_Analysis_Cross_Layer'
OUT = CROSS / 'intervention_extension_20260915/llama'
SOURCE = CROSS / 'eff_gen_correlations_20260915/llama_eff_gen_sources.csv'
SPLIT = CROSS / 'alphaedit_matched_geometry_causal_audit_n1000_order20260905_v1/delta_common/split.json'
MODEL = LLAMA_MODEL
METHODS = ('NAS', 'ENCORE', 'SPHERE', 'SADR', 'Native')
FAMILIES = ('rewrite', 'rephrase', 'locality')
ENDPOINTS = {'rewrite': 'EFF', 'rephrase': 'Gen', 'locality': 'LOC'}
KEYS = ('model', 'dataset', 'editor', 'method', 'order_id', 'edit_count')
GEO = ('realized_rel_parallel', 'realized_rel_tangential', 'realized_rel_displacement',
       'realized_hidden_norm_ratio', 'realized_hidden_norm', 'intervention_l2',
       'intended_hidden_norm', 'intended_intervention_l2', 'rounding_l2')


def family_pair(req, family):
    if family == 'rewrite':
        return req['prompt'], req['target_new']
    if family == 'rephrase':
        return req['rephrase_prompt'], req['target_new']
    groups = list(req['locality'].values())
    require(len(groups) == 1, 'One locality prompt per case required')
    return groups[0]['prompt'], groups[0]['ground_truth']


def materialize(tok, panel, family, device):
    pairs = [family_pair(r, family) for r in panel]
    inputs, starts, _, proofs = legacy_materialize(tok, [p[0] for p in pairs], [p[1] for p in pairs])
    for i, start in enumerate(starts):
        proofs[i]['target_token_ids'] = inputs['input_ids'][i, start:].tolist()
    return inputs.to(device), starts, proofs


@torch.inference_mode()
def forward(model, inputs, starts, replacement=None):
    captured = {}
    def hook(_module, args):
        h = args[0]
        rows = torch.arange(h.shape[0], device=h.device)
        positions = torch.as_tensor(starts, device=h.device) - 1
        captured['before'] = h[rows, positions].float().cpu().numpy().copy()
        if replacement is None:
            captured['after'] = captured['before'].copy()
            return None
        patched = h.clone()
        patched[rows, positions] = torch.as_tensor(replacement, device=h.device, dtype=h.dtype)
        captured['after'] = patched[rows, positions].float().cpu().numpy().copy()
        return (patched,) + args[1:]
    handle = model.model.layers[9].register_forward_pre_hook(hook)
    try:
        logits = model(**inputs, use_cache=False).logits
    finally:
        handle.remove()
    require(np.isfinite(captured['before']).all() and bool(torch.isfinite(logits).all()), 'Nonfinite forward')
    return logits, captured['before'], captured['after']


class BoundaryReached(Exception):
    pass


@torch.inference_mode()
def capture_fit(model, tok, panel, batch_size):
    pieces = []
    last = None
    def hook(_module, args):
        h = args[0]
        pieces.append(h[torch.arange(len(h), device=h.device), last].float().cpu().numpy().copy())
        raise BoundaryReached()
    handle = model.model.layers[9].register_forward_pre_hook(hook)
    old_padding = tok.padding_side
    tok.padding_side = 'right'
    try:
        for start in range(0, len(panel), batch_size):
            inputs = tok([r['prompt'] for r in panel[start:start + batch_size]],
                         padding=True, return_tensors='pt').to(next(model.parameters()).device)
            last = inputs['attention_mask'].sum(-1) - 1
            try:
                model.model(**inputs, use_cache=False)
            except BoundaryReached:
                pass
    finally:
        handle.remove()
        tok.padding_side = old_padding
    return np.concatenate(pieces)


def intended_target(cond, h, h0, axis, ids):
    u, delta, _, perp = decompose(h, h0)
    h = np.asarray(h, dtype=np.float64)
    amount = np.abs(delta @ axis)[:, None]
    perp_norm = np.linalg.norm(perp, axis=-1, keepdims=True)
    matched = np.minimum(amount, perp_norm)
    kind = cond['kind']
    if kind == 'native':
        result = h.copy()
    elif kind == 'perp':
        result = h - cond['fraction'] * perp
    elif kind == 'perp_match_axis':
        result = h - matched / np.maximum(perp_norm, 1e-12) * perp
    elif kind == 'axis':
        result = h - (delta @ axis)[:, None] * axis
    elif kind == 'random':
        vectors = []
        for ui, case_id in zip(u, ids):
            key = hashlib.sha256(f"llama-h9|{cond['seed']}|{case_id}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(key[:8], 'little'))
            v = rng.standard_normal(ui.shape)
            v -= np.dot(v, ui) * ui
            vectors.append(v / np.linalg.norm(v))
        result = h + matched * np.stack(vectors)
    elif kind == 'radial':
        result = h * np.linalg.norm(h - perp, axis=-1, keepdims=True) / np.linalg.norm(h, axis=-1, keepdims=True)
    else:
        raise ValueError(kind)
    require(np.isfinite(result).all(), 'Nonfinite intended intervention')
    return result


def geometry(h, h0, actual, intended):
    norm0 = np.linalg.norm(h0.astype(np.float64), axis=-1)
    _, delta, parallel, perp = decompose(actual, h0)
    u = h0.astype(np.float64) / norm0[:, None]
    return dict(realized_rel_parallel=np.sum(delta * u, axis=-1) / norm0,
        realized_rel_tangential=np.linalg.norm(perp, axis=-1) / norm0,
        realized_rel_displacement=np.linalg.norm(delta, axis=-1) / norm0,
        realized_hidden_norm_ratio=np.linalg.norm(actual.astype(np.float64), axis=-1) / norm0,
        realized_hidden_norm=np.linalg.norm(actual.astype(np.float64), axis=-1),
        intervention_l2=np.linalg.norm(actual.astype(np.float64) - h, axis=-1),
        intended_hidden_norm=np.linalg.norm(intended, axis=-1),
        intended_intervention_l2=np.linalg.norm(intended - h, axis=-1),
        rounding_l2=np.linalg.norm(actual.astype(np.float64) - intended, axis=-1))


@torch.inference_mode()
def capture_base(model, tok, panel, batch_size):
    result = {}
    for family in FAMILIES:
        hs, rows, proofs = [], [], []
        for start in range(0, len(panel), batch_size):
            inp, positions, proof = materialize(tok, panel[start:start + batch_size], family, next(model.parameters()).device)
            logits, h, _ = forward(model, inp, positions)
            rows.extend(prediction_rows(logits, positions, proof))
            hs.append(h)
            proofs.extend(proof)
            del logits
        result[family] = dict(h=np.concatenate(hs), rows=rows, proofs=proofs)
        write_json(OUT / 'base' / f'{family}.json', dict(rows=rows, proofs=proofs, case_ids=[r['case_id'] for r in panel]))
        write_npz(OUT / 'base' / f'{family}.npz', hidden=result[family]['h'])
    return result


@torch.inference_mode()
def run_state(model, tok, panel, baseline, source, axis, batch_size, dest):
    scores, checks = [], []
    identity = {k: source[k] for k in KEYS}
    state = source['method']
    for family in FAMILIES:
        predictions, raw = [], {c['id']: [] for c in conditions()}
        for start in range(0, len(panel), batch_size):
            chunk = panel[start:start + batch_size]
            end = start + len(chunk)
            ids = [r['case_id'] for r in chunk]
            inp, positions, proofs = materialize(tok, chunk, family, next(model.parameters()).device)
            require(proofs == baseline[family]['proofs'][start:end], 'Base/edited materialization differs')
            native_logits, h, _ = forward(model, inp, positions)
            h0 = baseline[family]['h'][start:end]
            sham_logits, sham_before, sham_after = forward(model, inp, positions, h)
            sham_error = float((native_logits - sham_logits).abs().max().item())
            require(sham_error == 0 and np.array_equal(sham_before, sham_after), 'Sham changes state/logits')
            checks.append(dict(family=family, case_ids=ids, max_logit_difference=sham_error))
            del sham_logits
            base_predictions = [r['predicted_token_ids'] for r in baseline[family]['rows'][start:end]]
            _, delta, _, perp = decompose(h, h0)
            clipped = np.abs(delta @ axis) > np.linalg.norm(perp, axis=-1) + 1e-8
            for cond in conditions():
                intended = intended_target(cond, h, h0, axis, ids)
                if cond['id'] == 'native':
                    logits, actual = native_logits, h
                else:
                    logits, before, actual = forward(model, inp, positions, intended)
                    require(np.array_equal(before, h), 'Native checkpoint boundary changed')
                measured = prediction_rows(logits, positions, proofs, base_predictions)
                geo = geometry(h, h0, actual, intended)
                raw[cond['id']].append(actual)
                for i, row in enumerate(measured):
                    scores.append(dict(**identity, state=state, label=state, family=family,
                        endpoint=ENDPOINTS[family], condition=cond['id'], case_id=ids[i], fraction=cond['fraction'],
                        endpoint_value=row['base_agreement'] if family == 'locality' else row['accuracy'],
                        accuracy=row['accuracy'], base_agreement=row['base_agreement'],
                        axis_match_clipped=bool(clipped[i]), **{k: float(v[i]) for k, v in geo.items()}))
                    predictions.append(dict(case_id=ids[i], condition=cond['id'], **row))
                if cond['id'] != 'native':
                    del logits
            del native_logits
            if end % 20 == 0:
                print(f"[llama] {source['editor']} {state} {family} {end}/100", flush=True)
                write_json(dest / 'status.json', dict(**identity, family=family, completed_cases=end))
        write_json(dest / f'predictions_{family}.json', predictions)
        write_npz(dest / f'hidden_{family}.npz', case_ids=np.asarray([r['case_id'] for r in panel]),
                  **{k: np.concatenate(v) for k, v in raw.items()})
    write_csv(dest / 'scores.csv', scores)
    write_json(dest / 'sham_checks.json', checks)
    return scores


def summarize(rows, n_bootstrap=10000):
    ids = list(dict.fromkeys(str(r['case_id']) for r in rows))
    require(len(ids) == 100, 'Expected 100 paired cases')
    lookup = {(r['family'], r['condition'], str(r['case_id'])): r for r in rows}
    require(len(lookup) == len(rows) == 3300, 'Incomplete per-state grid')
    index = np.random.default_rng(20260912).integers(0, len(ids), size=(n_bootstrap, len(ids)))
    identity = {k: rows[0][k] for k in KEYS}
    identity.update(state=rows[0]['state'], label=rows[0]['label'])
    summaries, contrasts, matching, doses = [], [], [], []
    def estimate(vals):
        lo, hi = np.percentile(vals[index].mean(axis=1), [2.5, 97.5])
        return dict(mean_difference=float(vals.mean()), ci_low=float(lo), ci_high=float(hi),
                    pointwise_ci_excludes_zero=bool(lo > 0 or hi < 0))
    for family in FAMILIES:
        by_c = {c['id']: [lookup[family, c['id'], i] for i in ids] for c in conditions()}
        vals = {c: np.asarray([float(r['endpoint_value']) for r in rr]) * 100 for c, rr in by_c.items()}
        random_ids = [c['id'] for c in conditions() if c['kind'] == 'random']
        vals['randperp_mean3'] = np.mean([vals[c] for c in random_ids], axis=0)
        for lhs, rhs, actual_field, intended_field, must_match in [
            ('perp_f100', 'radial_match_perp100', 'realized_hidden_norm', 'intended_hidden_norm', True),
            *[('perp_match_axis', c, 'intervention_l2', 'intended_intervention_l2', True) for c in random_ids],
            ('axis_f100', 'perp_match_axis', 'intervention_l2', 'intended_intervention_l2', False)]:
            left = np.array([float(r[actual_field]) for r in by_c[lhs]])
            right = np.array([float(r[actual_field]) for r in by_c[rhs]])
            il = np.array([float(r[intended_field]) for r in by_c[lhs]])
            ir = np.array([float(r[intended_field]) for r in by_c[rhs]])
            error = np.abs(left - right)
            intended_error = np.abs(il - ir)
            scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0)
            rounding_bound = np.array([float(l['rounding_l2']) + float(r['rounding_l2']) for l, r in zip(by_c[lhs], by_c[rhs])])
            bound_ok = error <= intended_error + rounding_bound + 1e-8 * scale
            require(bound_ok.all(), 'Realized difference exceeds proven rounding bound')
            if must_match:
                require((intended_error <= 1e-9 * np.maximum(np.maximum(il, ir), 1.0)).all(), 'Intended matched control does not match')
            matching.append(dict(**identity, family=family, lhs=lhs, rhs=rhs, field=actual_field,
                n_cases=100, intended_match_required=must_match,
                max_intended_scaled_error=float((intended_error / np.maximum(np.maximum(il, ir), 1.0)).max()),
                max_absolute_error=float(error.max()), max_scaled_error=float((error / scale).max()),
                mean_scaled_error=float((error / scale).mean()),
                max_rounding_l2_bound=float(rounding_bound.max()), rounding_bound_holds_all_cases=bool(bound_ok.all()),
                n_axis_clipped=sum(bool(r['axis_match_clipped']) for r in by_c[lhs])))
        for cond, value in vals.items():
            e = estimate(value - vals['native'])
            selected = random_ids if cond == 'randperp_mean3' else [cond]
            summaries.append(dict(**identity, family=family, endpoint=ENDPOINTS[family], condition=cond,
                n_cases=100, endpoint_value=float(value.mean()), endpoint_minus_native=e['mean_difference'],
                endpoint_ci_low=e['ci_low'], endpoint_ci_high=e['ci_high'],
                **{field: float(np.mean([float(r[field]) for c in selected for r in by_c[c]])) for field in GEO}))
        dose_ids = ('native', 'perp_f025', 'perp_f050', 'perp_f075', 'perp_f100')
        specs = [('matched_perp_minus_random_mean3', 'perp_match_axis', 'randperp_mean3', 'intended_capped_partial_l2'),
                 ('perp100_minus_radial', 'perp_f100', 'radial_match_perp100', 'intended_result_hidden_norm'),
                 ('axis_minus_matched_perp', 'axis_f100', 'perp_match_axis', 'axis_l2_match_capped_when_needed')]
        specs += [(f'dose_{b}_minus_{a}', b, a, 'adjacent_orthogonal_dose') for a, b in zip(dose_ids, dose_ids[1:])]
        for name, lhs, rhs, match in specs:
            contrasts.append(dict(**identity, family=family, metric=ENDPOINTS[family], unit='pp', contrast=name,
                lhs=lhs, rhs=rhs, matching=match, n_cases=100, **estimate(vals[lhs] - vals[rhs])))
        means = [float(vals[c].mean()) for c in dose_ids]
        doses.append(dict(**identity, family=family, endpoint=ENDPOINTS[family],
            **dict(zip(dose_ids, means)), nondecreasing_point_estimates=bool((np.diff(means) >= -1e-12).all())))
    return summaries, contrasts, matching, doses


def audit_sources():
    rows = list(csv.DictReader(SOURCE.open()))
    rows = [r for r in rows if r['dataset'] == 'zsRE' and r['order_id'] == 'canonical'
            and r['edit_count'] == '1000' and r['method'] in METHODS]
    rows.sort(key=lambda r: (('AlphaEdit', 'MEMIT').index(r['editor']), METHODS.index(r['method'])))
    require(len(rows) == 10, 'Expected ten canonical endpoint states')
    panel = read(Path(rows[0]['requests_source']))
    require([int(r['case_id']) for r in panel] == list(range(1000)), 'Requests not canonical')
    split = read(SPLIT)
    fit_ids, eval_ids = split['fit_case_ids'], split['pilot_case_ids']
    require(len(set(fit_ids)) == 500 and len(set(eval_ids)) == 100 and set(fit_ids).isdisjoint(eval_ids), 'Bad split')
    by_id = {str(r['case_id']): r for r in panel}
    identities = []
    for r in rows:
        require(read(Path(r['requests_source'])) == panel, 'Method requests differ')
        path = Path(r['checkpoint_path'])
        meta = read(path.with_suffix('.json'))
        require(meta['edit_count'] == 1000 and meta['editing_method'] == r['editor'], 'Checkpoint metadata differs')
        cfgpath = path.parent.parent / 'run_config.json'
        cfg = read(cfgpath)
        require(cfg['seed'] == 42 and cfg['batch_size'] == 1, 'Unexpected run seed/batch')
        hp = cfg['effective_editing_hparams']
        flags = {m: bool(cfg.get(k, hp.get(k, False))) for m, k in [('NAS','nas_enabled'),('ENCORE','encore_enabled'),('SPHERE','sphere_enabled'),('SADR','sadr_regularization')]}
        require(cfg['fingerprint'] == meta['run_fingerprint'], 'Checkpoint/run fingerprint mismatch')
        require(sum(flags.values()) == int(r['method'] != 'Native') and (r['method'] == 'Native' or flags[r['method']]), 'Method flag mismatch')
        identities.append(dict(**{k:r[k] for k in KEYS}, checkpoint_path=str(path),
            checkpoint_bytes=path.stat().st_size, checkpoint_mtime_ns=path.stat().st_mtime_ns,
            checkpoint_sidecar=record(path.with_suffix('.json')), requests=record(Path(r['requests_source'])), config=record(cfgpath)))
    return rows, [by_id[str(i)] for i in fit_ids], [by_id[str(i)] for i in eval_ids], identities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--fit-batch-size', type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows, fitted, panel, identities = audit_sources()
    protocol = dict(schema_version=2, model='Llama', dataset='zsRE', hidden_node='H9: input to block9, before RMSNorm',
        checkpoint_edit_count=1000, edit_order='canonical', model_seed=42, dtype='bfloat16', attn_implementation='eager',
        no_new_editing=True, methods=list(METHODS), editors=['AlphaEdit','MEMIT'],
        fit_case_ids=[r['case_id'] for r in fitted], evaluation_case_ids=[r['case_id'] for r in panel],
        source_split=record(SPLIT), source_table=record(SOURCE), checkpoints=identities,
        conditions=conditions(), random_seeds=[20260911,20260912,20260913], bootstrap_seed=20260912, n_bootstrap=10000,
        intervention='Only original prompt_last H9, per-family Base reference, live downstream computation',
        axis_fit='Unit mean raw checkpoint-minus-Base rewrite prompt_last displacement on disjoint fit500',
        paired_partial_random='v2: random addition magnitude=min(abs(delta dot axis), norm(delta_perp)); equals capped partial orthogonal removal L2 before dtype rounding',
        difference_from_legacy='Legacy random magnitude was uncapped abs(delta dot axis). Existing shuffled-order SPHERE/SADR data are not pooled.',
        radial='Edited hidden scaled to full-orthogonal-removal resulting norm before dtype rounding; intervention L2 not matched',
        rounding='Intended F64 matching and realized BF16 norm/L2 errors separately recorded; triangle inequality rounding bound checked',
        eff_gen='Corrected teacher-forced all-target-token case-macro accuracy, genuine EOT included; not free generation EFF/GEN',
        loc='Same target-span case-macro Base argmax agreement; locality target generally has no appended EOT',
        inference_batch_size=args.batch_size, fit_batch_size=args.fit_batch_size,
        bootstrap='Paired case-resampling percentile 95% pointwise intervals, three random seeds averaged within each case; n=100',
        code=record(Path(__file__)), helper_sources=[record(Path(legacy_materialize.__code__.co_filename)),
            record(ROOT/'diagnostics/gpt2_intervention_analysis.py'), record(ROOT/'diagnostics/gpt2_checkpoint_analysis.py')])
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT/'protocol.json', protocol)
    if args.audit_only:
        print(json.dumps(dict(audit_complete=True, n_states=len(rows), n_fit=500, n_eval=100)))
        return
    started = time.monotonic()
    if not args.aggregate_only:
        model = load_model(str(MODEL), 'bfloat16', 'cuda:0', False, 'eager')
        tok = AutoTokenizer.from_pretrained(MODEL)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        names = read(Path(rows[0]['checkpoint_path']).with_suffix('.json'))['parameter_names']
        pristine = ev.CHECKPOINT_MODULE.snapshot_parameters(model, names)
        base_fit = capture_fit(model, tok, fitted, args.fit_batch_size)
        write_npz(OUT/'base'/'fit_rewrite.npz', hidden=base_fit, case_ids=np.asarray([r['case_id'] for r in fitted]))
        baseline = capture_base(model, tok, panel, args.batch_size)
        for source in rows:
            dest = OUT / source['editor'] / source['method']
            if (dest/'complete.json').exists():
                print(f"[reuse] {dest}", flush=True)
                continue
            began = time.monotonic()
            params = dict(model.named_parameters())
            with torch.no_grad():
                for name, tensor in pristine.items():
                    params[name].copy_(tensor)
            metadata = ev.load_edited_parameter_checkpoint(model, Path(source['checkpoint_path']))
            require(metadata == read(Path(source['checkpoint_path']).with_suffix('.json')), 'Payload metadata mismatch')
            fitted_h = capture_fit(model, tok, fitted, args.fit_batch_size)
            axis, mean = fit_axis(fitted_h, base_fit)
            write_npz(dest/'fit_axis.npz', hidden=fitted_h, axis=axis, mean_raw_displacement=mean,
                      case_ids=np.asarray([r['case_id'] for r in fitted]))
            run_state(model, tok, panel, baseline, source, axis, args.batch_size, dest)
            write_json(dest/'complete.json', dict(complete=True, **{k:source[k] for k in KEYS},
                n_rows=3300, seconds=time.monotonic()-began, source_checkpoint=identities[rows.index(source)],
                protocol=record(OUT/'protocol.json'), artifacts=[record(p) for p in sorted(dest.rglob('*')) if p.is_file()]))
            print(f"[complete] {source['editor']} {source['method']} {time.monotonic()-began:.1f}s", flush=True)
    totals = [[],[],[],[]]
    for source in rows:
        dest = OUT / source['editor'] / source['method']
        require(read(dest/'complete.json')['complete'], 'Missing state completion')
        scores = list(csv.DictReader((dest/'scores.csv').open()))
        # CSV bool strings are interpreted explicitly before computing clipping counts.
        for r in scores:
            r['axis_match_clipped'] = r['axis_match_clipped'] == 'True'
        parts = summarize(scores)
        for total, part in zip(totals, parts):
            total.extend(part)
    for name, data in zip(('summary.csv','contrasts.csv','matching_audit.csv','dose_response.csv'), totals):
        write_csv(OUT/name, data)
    write_json(OUT/'complete.json', dict(complete=True, schema_version=2, n_states=10, n_cases=100,
        n_per_case_rows=33000, n_summary_rows=len(totals[0]), n_contrast_rows=len(totals[1]),
        elapsed_seconds=time.monotonic()-started, no_new_editing=True, protocol=record(OUT/'protocol.json'),
        artifacts=[record(OUT/name) for name in ('summary.csv','contrasts.csv','matching_audit.csv','dose_response.csv')],
        state_completions=[record(OUT/r['editor']/r['method']/'complete.json') for r in rows]))
    print(json.dumps(dict(complete=True, n_states=10, seconds=time.monotonic()-started)), flush=True)


if __name__ == '__main__':
    main()
