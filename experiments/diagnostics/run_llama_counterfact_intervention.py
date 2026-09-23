#!/usr/bin/env python3
"""Extend frozen RQ3 protocol v2 to ten Llama/CounterFact endpoint states.

The existing Llama inference, target-token scoring, intervention construction,
and paired bootstrap routines are reused without alteration. Dataset case IDs
are mapped from the frozen canonical split positions, as in GPT-2 CounterFact.
"""

# Release-only filesystem defaults; computation below follows the source snapshot.
import sys as _bng_sys
from pathlib import Path as _BNGPath
_bng_sys.path.insert(0, str(_BNGPath(__file__).resolve().parents[1]))
from model_code.paths import RESEARCH_ROOT, OUTPUT_ROOT, DATA_ROOT, LLAMA_MODEL
from pathlib import Path
import argparse
import csv
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from transformers import AutoTokenizer
from diagnostics import run_llama_intervention_extension as legacy
from diagnostics import gpt2_checkpoint_analysis as core
from diagnostics.run_gpt2_intervention_extension import split_panel

OUT = OUTPUT_ROOT / '_Analysis_Cross_Layer/llama_counterfact_intervention_20260923'
METHODS = ['Native', 'NAS', 'ENCORE', 'SPHERE', 'SADR']
EDITORS = ['AlphaEdit', 'MEMIT']
read, record, require, write_json = core.read, core.record, core.require, core.write_json


def audit_sources():
    manifest_path = OUT / 'source_manifest.json'
    source = read(manifest_path)
    rows = source['states']
    require(len(rows) == 10, 'Ten states are required, including both ENCORE states')
    require({(r['editor'], r['method']) for r in rows} ==
            {(e, m) for e in EDITORS for m in METHODS}, 'Incomplete or duplicate state identities')
    rows.sort(key=lambda r: (EDITORS.index(r['editor']), METHODS.index(r['method'])))
    panel = read(Path(rows[0]['requests_source']))
    require(len(panel) == 1000 and len({str(r['case_id']) for r in panel}) == 1000,
            'Expected 1,000 unique canonical requests')
    fitted, evaluated, split = split_panel(panel, 'CounterFact')
    names = None
    identities = []
    for r in rows:
        require(r['model'] == 'Llama' and r['dataset'] == 'CounterFact' and
                r['order_id'] == 'canonical' and int(r['edit_count']) == 1000, 'Wrong condition')
        checkpoint = Path(r['checkpoint_path'])
        require(checkpoint.is_file(), f'Missing checkpoint: {checkpoint}')
        meta = read(checkpoint.with_suffix('.json'))
        require(meta['edit_count'] == 1000 and meta['editing_method'] == r['editor'],
                'Checkpoint metadata condition mismatch')
        require(meta['storage_mode'] == 'parameter_deltas', 'Expected cumulative parameter deltas')
        if names is None:
            names = meta['parameter_names']
        require(meta['parameter_names'] == names, 'Edited parameter set differs')
        request_path = Path(r['requests_source'])
        require(read(request_path) == panel, 'Canonical requests differ across states')
        cfg = read(Path(r['config_path']))
        hp = cfg.get('effective_hparams', cfg.get('effective_editing_hparams', {}))
        require(bool(hp), 'Missing effective editing hyperparameters')
        flags = {m: bool(hp.get(k, cfg.get(k, False))) for m, k in [
            ('NAS', 'nas_enabled'), ('ENCORE', 'encore_enabled'),
            ('SPHERE', 'sphere_enabled'), ('SADR', 'sadr_regularization')]}
        require(sum(flags.values()) == int(r['method'] != 'Native') and
                (r['method'] == 'Native' or flags[r['method']]), 'Method flags mismatch')
        require(not hp.get('residual_gain_regularization', False), 'Unexpected RGR checkpoint')
        require(cfg.get('model_seed', cfg.get('seed')) == 42, 'Unexpected editing seed')
        if 'run_fingerprint' in meta:
            require(cfg.get('fingerprint') == meta['run_fingerprint'], 'Run fingerprint mismatch')
        else:
            require(meta.get('model_seed') == 42, 'New checkpoint seed mismatch')
            require(meta.get('requests_sha256') == record(request_path)['sha256'],
                    'New checkpoint requests hash mismatch')
        identities.append(dict(**{k:r[k] for k in legacy.KEYS},
            checkpoint_path=str(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
            checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns,
            sidecar=record(checkpoint.with_suffix('.json')), requests=record(request_path),
            config=record(Path(r['config_path'])), metadata=meta))
    protocol = dict(schema_version=2, model='Llama', dataset='CounterFact',
        order_id='canonical', checkpoint_edit_count=1000, model_seed=42,
        dtype='bfloat16', attn_implementation='eager', model_path=str(legacy.MODEL),
        hidden_node='H9: input to block9, before RMSNorm',
        intervention='Only original prompt_last H9, per-family Base reference, live downstream computation',
        split=split, inference_batch_size=2, fit_batch_size=16,
        methods=METHODS, editors=EDITORS, conditions=legacy.conditions(),
        random_seeds=[20260911, 20260912, 20260913], bootstrap_seed=20260912,
        n_bootstrap=10000, n_states=10, n_cases_per_family=100, n_fit=500,
        axis_fit='Unit mean raw edited-minus-Base rewrite prompt_last displacement on disjoint fit500',
        eff_gen='Case-macro teacher-forced all-target-token accuracy including genuine EOT; not free generation',
        locality='Case-macro agreement with Base argmax on the original locality target span, no added EOT',
        radial='Same intended result norm as full orthogonal removal, not Base norm resizing',
        random='Three random seeds; capped magnitude matched to perp_match_axis; same legacy seed mapping',
        scope_extension='CounterFact only; original three-cohort raw artifacts remain untouched',
        no_new_editing=True, new_checkpoint_inference=True,
        sources=identities, source_manifest=record(manifest_path),
        code=record(Path(__file__)), helper_sources=[record(Path(legacy.__file__)),
            record(Path(legacy.legacy_materialize.__code__.co_filename)),
            record(ROOT/'diagnostics/gpt2_intervention_analysis.py'),
            record(ROOT/'diagnostics/run_gpt2_intervention_extension.py')])
    return rows, fitted, evaluated, protocol


def ensure_protocol(protocol):
    path = OUT / 'protocol.json'
    if path.exists():
        require(read(path) == protocol, 'Frozen protocol differs from existing run')
    else:
        write_json(path, protocol)


def aggregate(rows, protocol, shard_count):
    totals = [[], [], [], []]
    scores_all = []
    completions = []
    for i, source in enumerate(rows):
        dest = OUT / f'worker_{i % shard_count}' / source['editor'] / source['method']
        completion = read(dest / 'complete.json')
        require(completion['complete'] and completion['protocol_hash'] == core.json_hash(protocol),
                'Missing or mismatched state completion')
        for item in completion['artifacts']:
            require(record(Path(item['path'])) == item, 'State artifact changed')
        raw = list(csv.DictReader((dest/'scores.csv').open()))
        for r in raw:
            r['axis_match_clipped'] = r['axis_match_clipped'] == 'True'
        scores_all.extend(raw)
        for total, part in zip(totals, legacy.summarize(raw)):
            total.extend(part)
        completions.append(record(dest / 'complete.json'))
    # Independent workers must observe exactly the same Base and tokenization.
    reference = OUT / 'worker_0' / 'base'
    for worker in range(1, shard_count):
        base = OUT / f'worker_{worker}' / 'base'
        for family in legacy.FAMILIES:
            require(read(reference/f'{family}.json') == read(base/f'{family}.json'),
                    'Worker Base tokenization or predictions differ')
            a, b = [np.load(p/f'{family}.npz')['hidden'] for p in (reference, base)]
            require(np.array_equal(a, b), 'Worker Base hidden differs')
        a, b = [np.load(p/'fit_rewrite.npz')['hidden'] for p in (reference, base)]
        require(np.array_equal(a, b), 'Worker Base fit hidden differs')
    for name, data in zip(('condition_summary.csv', 'paired_contrasts.csv',
                           'matching_audit.csv', 'dose_response.csv'), totals):
        legacy.write_csv(OUT/name, data)
    legacy.write_csv(OUT/'per_case_scores.csv', scores_all)
    require(len(scores_all) == 33000 and len(totals[0]) == 360 and len(totals[1]) == 210,
            'Incomplete experiment grid')
    write_json(OUT/'complete.json', dict(complete=True, n_states=10, n_cases=100,
        n_per_case_rows=33000, n_summary_rows=360, n_contrast_rows=210,
        workers_base_bitwise_equal=True, no_new_editing=True, new_checkpoint_inference=True,
        protocol_hash=core.json_hash(protocol), state_completions=completions,
        artifacts=[record(OUT/n) for n in ('condition_summary.csv', 'paired_contrasts.csv',
            'matching_audit.csv', 'dose_response.csv', 'per_case_scores.csv')]))
    print('[aggregate complete] 10 states; 33,000 state-family-condition-case observations', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=3)
    args = parser.parse_args()
    require(args.shard_count > 0 and 0 <= args.shard_index < args.shard_count, 'Invalid shard')
    torch.set_num_threads(4)
    rows, fitted, panel, protocol = audit_sources()
    ensure_protocol(protocol)
    if args.audit_only:
        print(json.dumps(dict(audit_complete=True, n_states=10, n_fit=len(fitted),
                              n_eval=len(panel), protocol_hash=core.json_hash(protocol))))
        return
    if args.aggregate_only:
        aggregate(rows, protocol, args.shard_count)
        return
    require(torch.cuda.is_available(), 'CUDA unavailable; no inference executed')
    torch.manual_seed(42)
    worker = OUT / f'worker_{args.shard_index}'
    worker.mkdir(parents=True, exist_ok=True)
    legacy.OUT = worker
    assigned = [(i, r) for i, r in enumerate(rows) if i % args.shard_count == args.shard_index]
    print('[worker start]', args.shard_index, [(r['editor'], r['method']) for _, r in assigned], flush=True)
    model = legacy.load_model(str(legacy.MODEL), 'bfloat16', 'cuda:0', False, 'eager')
    tok = AutoTokenizer.from_pretrained(legacy.MODEL, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pristine = legacy.ev.CHECKPOINT_MODULE.snapshot_parameters(model, protocol['sources'][0]['metadata']['parameter_names'])
    base_fit = legacy.capture_fit(model, tok, fitted, 16)
    core.write_npz(worker/'base/fit_rewrite.npz', hidden=base_fit,
                   case_ids=np.asarray([r['case_id'] for r in fitted]))
    baseline = legacy.capture_base(model, tok, panel, 2)
    for i, source in assigned:
        dest = worker / source['editor'] / source['method']
        if (dest/'complete.json').exists():
            prior = read(dest/'complete.json')
            require(prior['protocol_hash'] == core.json_hash(protocol), 'Completed state has wrong protocol')
            for artifact in prior['artifacts']:
                require(record(Path(artifact['path'])) == artifact, 'Completed state artifact changed')
            print('[reuse]', source['editor'], source['method'], flush=True)
            continue
        started = time.monotonic()
        params = dict(model.named_parameters())
        with torch.no_grad():
            for name, tensor in pristine.items():
                params[name].copy_(tensor)
        loaded = legacy.ev.load_edited_parameter_checkpoint(model, Path(source['checkpoint_path']))
        require(loaded == protocol['sources'][i]['metadata'], 'Payload metadata differs from sidecar')
        versions = tuple((n, v._version) for n, v in model.named_parameters())
        fit_h = legacy.capture_fit(model, tok, fitted, 16)
        axis, mean = legacy.fit_axis(fit_h, base_fit)
        core.write_npz(dest/'fit_axis.npz', hidden=fit_h, axis=axis, mean_raw_displacement=mean,
                       case_ids=np.asarray([r['case_id'] for r in fitted]))
        scores = legacy.run_state(model, tok, panel, baseline, source, axis, 2, dest)
        require(versions == tuple((n, v._version) for n, v in model.named_parameters()),
                'Weights changed during intervention')
        summary, contrasts, matching, doses = legacy.summarize(scores)
        require(len(scores) == 3300 and len(summary) == 36 and len(contrasts) == 21, 'Incomplete state')
        for name, data in [('condition_summary.csv', summary), ('paired_contrasts.csv', contrasts),
                           ('matching_audit.csv', matching), ('dose_response.csv', doses)]:
            legacy.write_csv(dest/name, data)
        artifacts = [record(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name != 'complete.json']
        write_json(dest/'complete.json', dict(complete=True, **{k:source[k] for k in legacy.KEYS},
            n_rows=3300, seconds=time.monotonic()-started, source_checkpoint=protocol['sources'][i],
            protocol_hash=core.json_hash(protocol), weights_unchanged_during_intervention=True,
            peak_memory_gb=torch.cuda.max_memory_allocated()/1e9, artifacts=artifacts))
        print('[state complete]', source['editor'], source['method'],
              f'{time.monotonic()-started:.1f}s', flush=True)
    write_json(worker/'complete.json', dict(complete=True, shard_index=args.shard_index,
        shard_count=args.shard_count, states=[{k:r[k] for k in legacy.KEYS} for _, r in assigned],
        protocol_hash=core.json_hash(protocol)))
    print('[worker complete]', args.shard_index, flush=True)


if __name__ == '__main__':
    main()
