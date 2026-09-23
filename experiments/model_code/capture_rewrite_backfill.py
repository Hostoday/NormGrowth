#!/usr/bin/env python3
"""Capture missing fixed-panel rewrite/subject-last H9 from saved deltas.

No editing, optimization, or generation. All historical sources are read-only.
Each checkpoint is restored against pristine BF16 parameters independently.
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
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'evaluate'))
import numpy as np
import torch
from transformers import AutoTokenizer
from diagnostics.capture_scalar_checkpoint_h9 import restore, geometry, raw
from diagnostics.capture_matched_geometry_checkpoints import MODEL, NAMES, make_probes
from diagnostics.analyze_base_component_cosines import _atomic_json, _atomic_npz, _model_backbone, _padded_batch
from diagnostics.analyze_residual_spectrum import load_model, model_input_device
from diagnostics.assemble_cumulative_prefix_locality1k import file_record, validate_native_replay


def read(path):
    return json.loads(Path(path).read_text())


class H9Reached(Exception):
    pass


@torch.inference_mode()
def capture(model, tokenizer, probes, batch_size=16):
    states = np.empty((len(probes), 1, 4096), dtype=np.float32)
    positions = np.empty((len(probes), 1), dtype=np.int32)
    selected, result = None, None
    def hook(module, args):
        nonlocal result
        h = args[0]
        rows = torch.arange(h.shape[0], device=h.device)[:, None]
        result = h[rows, selected.to(h.device)].float().cpu().numpy()
        raise H9Reached()
    handle = _model_backbone(model).layers[9].register_forward_pre_hook(hook)
    try:
        for start in range(0, len(probes), batch_size):
            end = min(start + batch_size, len(probes))
            inputs, selected = _padded_batch(probes[start:end], ['subject_last'],
                pad_token_id=int(tokenizer.pad_token_id), device=model_input_device(model))
            result = None
            try:
                _model_backbone(model)(**inputs, use_cache=False, output_hidden_states=False, return_dict=True)
            except H9Reached:
                pass
            assert result is not None
            states[start:end] = result
            positions[start:end] = selected.numpy()
    finally:
        handle.remove()
    assert np.isfinite(states).all()
    return states, positions


def historical_subject(path):
    v = raw(Path(path))
    idx = list(v['positions'].astype(str)).index('subject_last')
    return dict(h9=v['h9'][:, idx:idx+1], case_ids=v['case_ids'].astype(str),
        positions=np.asarray(['subject_last']), token_positions=v['token_positions'][:, idx:idx+1],
        valid_positions=v['valid_positions'][:, idx:idx+1])


def compare_exact(actual, expected):
    for key in ('case_ids', 'positions', 'token_positions', 'valid_positions'):
        assert np.array_equal(actual[key].astype(str), expected[key].astype(str)), key
    diff = float(np.max(np.abs(actual['h9'] - expected['h9'])))
    equal = bool(np.array_equal(actual['h9'], expected['h9']))
    assert equal, f'Historical subject_last H9 mismatch: max_abs_error={diff}'
    return dict(exact=True, maximum_absolute_error=diff, n_cases=len(actual['h9']))


def write_base(path, values, provenance):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        compare_exact(values, raw(path))
    else:
        _atomic_npz(path, values)
    _atomic_json(path.parent / 'complete.json', dict(complete=True, base=True,
        raw_h9=file_record(path), **provenance))


def restore_inventory_checkpoint(model, pristine, row):
    checkpoint = Path(row['checkpoint_path'])
    metadata = read(checkpoint.with_suffix('.json'))
    listed_run = Path(row['source_run'])
    declared_run = Path(metadata.get('source_run', str(listed_run)))
    if not (declared_run/'run_config.json').exists():
        manifest_path = declared_run/'run_manifest.json'
        cfg = read(manifest_path)
        completion = read(declared_run/'complete.json')
        assert completion['status']=='complete' and completion['manifest_sha256']==file_record(manifest_path)['sha256']
        assert cfg['delta_checkpoint']==metadata
        assert metadata['edit_count']==int(row['edit_count'])==1000
        assert metadata['editing_method']==row['editor'] and metadata['base_model']=='meta-llama/Meta-Llama-3-8B-Instruct'
        assert metadata['model_seed']==42 and metadata['batch_size']==1
        assert metadata['rewrite_layers']==[4,5,6,7,8] and metadata['parameter_names']==NAMES
        assert metadata['storage_mode']=='parameter_deltas'
        request_record=file_record(Path(row['requests_path']))
        assert metadata['requests_sha256']==request_record['sha256']
        assert metadata['requests_prefix_sha256']==request_record['sha256'] and metadata['requests_prefix_count']==1000
        hp=cfg['effective_hparams']
        assert row['method']=='ENCORE' and hp['encore_enabled']
        assert not any(hp.get(k,False)for k in ['residual_gain_regularization','nas_enabled','sphere_enabled','sadr_regularization'])
        assert metadata['hparams_sha256']==file_record(Path(metadata['hparams_path']))['sha256']
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert payload['metadata']==metadata
        assert payload['format']=='easyedit-edited-parameters' and payload['format_version']==1
        assert list(payload['state_dict'])==NAMES
        named=dict(model.named_parameters())
        with torch.no_grad():
            for name,delta in payload['state_dict'].items():
                param=named[name]
                assert delta.dtype==param.dtype==torch.bfloat16 and delta.shape==param.shape
                param.copy_(pristine[name].to(param.device)+delta.to(param.device))
        return dict(checkpoint=file_record(checkpoint),checkpoint_metadata=metadata,
            inventory_source_run=str(listed_run.resolve()),declared_original_run=str(declared_run.resolve()),
            new_format_run_manifest=file_record(manifest_path),completion_record=file_record(declared_run/'complete.json'),
            requests=request_record,replay_validation=None)
    recovery = None
    if 'source_run_fingerprint' in metadata:
        recovery = validate_native_replay(checkpoint.parents[1], declared_run)
        assert read(declared_run/'requests.json') == read(row['requests_path'])
        assert read(listed_run/'run_config.json')['fingerprint'] == read(declared_run/'run_config.json')['fingerprint']
    identity = restore(model, pristine, checkpoint, declared_run, int(row['edit_count']))
    identity.update(inventory_source_run=str(listed_run.resolve()),
        declared_original_run=str(declared_run.resolve()), replay_validation=recovery)
    return identity


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inventory', type=Path, default=RESEARCH_ROOT / 'paper_results_reorganization_20260922/results_rewrite_20260923_rq3_all_cohorts/rewrite_geometry_comparison/rewrite_geometry_inventory.csv')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--n-shards', type=int, default=3)
    ap.add_argument('--benchmark-only', action='store_true')
    ap.add_argument('--skip-historical-check', action='store_true')
    args = ap.parse_args()
    assert 0 <= args.shard < args.n_shards
    rows = list(csv.DictReader(args.inventory.open()))
    missing = [r for r in rows if r['model'] == 'Llama' and r['available'] != 'True']
    selected = [r for i, r in enumerate(missing) if i % args.n_shards == args.shard]
    selected.sort(key=lambda r: (int(r['edit_count']) != 1000, r['trajectory_id'], int(r['edit_count'])))
    known = next(r for r in rows if r['trajectory_id'] == 'T29' and r['edit_count'] == '1000')
    started = time.monotonic()
    out = OUTPUT_ROOT / 'rewrite_geometry_capture_backfill'
    out.mkdir(exist_ok=True)
    kind = 'benchmark' if args.benchmark_only else f'shard_{args.shard}'
    manifest_path = out / f'{kind}_manifest.json'
    manifest = dict(complete=False, pid=os.getpid(), device=args.device, shard=args.shard,
        n_shards=args.n_shards, n_selected=len(selected), n_completed=0, checks=[],
        protocol=dict(family='rewrite', position='subject_last', boundary='H9 = block8 output = block9 pre-RMSNorm',
            fixed_n_cases=1000, reference='pristine Base, matching prompt/case/token', dtype='bfloat16',
            attn_implementation='eager', batch_size=16, max_length=256, padding_side='right',
            restore='pristine BF16 Base + saved cumulative BF16 delta independently per checkpoint',
            parameters_unchanged_during_capture=True, optimization_or_editing=False),
        code=file_record(Path(__file__)), inventory=file_record(args.inventory), model_path=str(MODEL))
    def status(stage, **kwargs):
        manifest.update(stage=stage, elapsed_seconds=time.monotonic()-started, **kwargs)
        _atomic_json(manifest_path, manifest)
        print(json.dumps(dict(stage=stage, elapsed=round(manifest['elapsed_seconds'],2),
            n_completed=manifest['n_completed'], n_selected=len(selected), **kwargs), ensure_ascii=False), flush=True)
    try:
        torch.set_num_threads(4)
        torch.manual_seed(42)
        assert torch.cuda.is_available(), 'GPU access required for requested backfill'
        status('loading_model')
        model = load_model(str(MODEL), 'bfloat16', args.device, False, 'eager').eval()
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, use_fast=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'right'
        named = dict(model.named_parameters())
        pristine = {name: named[name].detach().cpu().clone() for name in NAMES}
        def reset_base():
            with torch.no_grad():
                for name in NAMES:
                    named[name].copy_(pristine[name].to(named[name].device))
        panels, bases, base_provenance = {}, {}, {}
        for dataset in sorted({r['dataset'] for r in selected} | {'zsRE'}):
            representative = next(r for r in rows if r['model']=='Llama' and r['dataset']==dataset
                and r['order_id']=='canonical' and r['method']=='Native' and r['editor']=='AlphaEdit')
            requests = read(representative['requests_path'])
            assert len(requests) == 1000
            # Match the historical fixed locality panel case order, independently of edit order.
            with np.load(representative['fixed_panel_reference'], allow_pickle=False) as ref:
                panel_ids = list(ref['case_ids'].astype(str))
            by_id = {str(r['case_id']): r for r in requests}
            assert len(by_id)==1000 and set(panel_ids)==set(by_id)
            requests = [by_id[k] for k in panel_ids]
            probes, positions, valid, missing_probes = make_probes(requests, tokenizer, 'rewrite')
            assert positions == ['subject_last', 'prompt_last'] and valid.all() and not missing_probes
            reset_base()
            status('capturing_base', dataset=dataset)
            tick = time.monotonic()
            hidden, tp = capture(model, tokenizer, probes)
            values = dict(h9=hidden, case_ids=np.asarray(panel_ids), positions=np.asarray(['subject_last']),
                token_positions=tp, valid_positions=np.ones_like(tp, dtype=bool))
            panels[dataset], bases[dataset] = probes, values
            base_provenance[dataset] = dict(dataset=dataset, model_path=str(MODEL), n_cases=1000,
                fixed_requests=file_record(Path(representative['requests_path'])),
                fixed_panel_reference=file_record(Path(representative['fixed_panel_reference'])),
                seconds=time.monotonic()-tick, protocol=manifest['protocol'])
            if dataset=='zsRE':
                base_provenance[dataset]['historical_parity'] = compare_exact(values, historical_subject(known['raw_base_source']))
            _atomic_json(out / f'{kind}_{dataset}_probes.json', dict(probes=[dict(case_id=str(p.case_id),
                prompt=p.prompt, subject=p.subject, input_ids=list(p.input_ids), positions=p.positions) for p in probes]))
        if not args.skip_historical_check:
            status('validating_historical_endpoint', trajectory_id='T29')
            tick = time.monotonic()
            identity = restore_inventory_checkpoint(model, pristine, known)
            hidden, tp = capture(model, tokenizer, panels['zsRE'])
            values = dict(bases['zsRE'], h9=hidden, token_positions=tp)
            parity = compare_exact(values, historical_subject(known['raw_rewrite_source']))
            manifest['historical_endpoint_parity'] = dict(**parity, identity=identity,
                historical_raw=file_record(Path(known['raw_rewrite_source'])), seconds=time.monotonic()-tick)
            status('historical_endpoint_verified')
        manifest['base_provenance'] = base_provenance
        if args.benchmark_only:
            manifest['complete']=True
            status('benchmark_complete')
            return
        for row in selected:
            dataset=row['dataset']
            checkpoint=Path(row['checkpoint_path'])
            destination=Path(row['proposed_backfill_post'])
            destination.parent.mkdir(parents=True, exist_ok=True)
            complete_path=destination.parent/'complete.json'
            write_base(row['proposed_backfill_base'], bases[dataset], base_provenance[dataset])
            if complete_path.exists():
                done=read(complete_path)
                assert done['complete'] and done['identity']['checkpoint']==file_record(checkpoint)
                assert done['raw_h9']==file_record(destination)
                check,_,_=geometry(raw(destination), bases[dataset])
                assert check==done['summary']
                manifest['checks'].append(dict(trajectory_id=row['trajectory_id'],step=int(row['edit_count']),
                    complete_path=str(complete_path), reused=True))
                manifest['n_completed']+=1
                status('reused_checkpoint')
                continue
            status('restoring_checkpoint', trajectory_id=row['trajectory_id'], edit_count=int(row['edit_count']))
            tick=time.monotonic()
            identity=restore_inventory_checkpoint(model, pristine, row)
            train_requests=read(row['requests_path'])
            train_by_id={str(x['case_id']):x for x in train_requests}
            assert set(train_by_id)==set(bases[dataset]['case_ids'])
            for p in panels[dataset]:
                req=train_by_id[str(p.case_id)]
                assert p.prompt==req['prompt'] and p.subject==req['subject']
            versions={n:int(p._version) for n,p in model.named_parameters()}
            hidden,tp=capture(model,tokenizer,panels[dataset])
            assert versions=={n:int(p._version) for n,p in model.named_parameters()}
            assert np.array_equal(tp,bases[dataset]['token_positions'])
            values=dict(bases[dataset],h9=hidden,token_positions=tp)
            summary,per_case,error=geometry(values,bases[dataset])
            _atomic_npz(destination,values)
            per_path=destination.parent/'per_case_rewrite_geometry.npz'
            _atomic_npz(per_path,dict(**per_case,case_ids=values['case_ids']))
            done=dict(complete=True, trajectory_id=row['trajectory_id'], dataset=dataset, editor=row['editor'],
                method=row['method'], version=row['version'], order_id=row['order_id'], edit_count=int(row['edit_count']),
                identity=identity, source_row=row, protocol=manifest['protocol'], n_prompts=1000,
                raw_h9=file_record(destination),per_case_geometry=file_record(per_path),summary=summary,
                base_raw=file_record(Path(row['proposed_backfill_base'])),
                source_requests=file_record(Path(row['requests_path'])),
                parameters_unchanged_during_inference=True, token_positions_exact=True,
                maximum_relative_geometry_identity_error=error, seconds=time.monotonic()-tick)
            _atomic_json(complete_path,done)
            manifest['checks'].append(dict(trajectory_id=row['trajectory_id'],step=int(row['edit_count']),
                complete_path=str(complete_path),seconds=done['seconds']))
            manifest['n_completed']+=1
            status('checkpoint_complete', trajectory_id=row['trajectory_id'],edit_count=int(row['edit_count']),
                checkpoint_seconds=round(done['seconds'],2))
        manifest['complete']=True
        status('complete')
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__=='__main__':
    main()
