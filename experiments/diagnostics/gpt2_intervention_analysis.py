#!/usr/bin/env python3
"""GPT-2 XL H18 interventions at a fixed AlphaEdit checkpoint.

Reuses the paper's exact fit500 / evaluation100 case IDs, but observes the new
canonical-order, model-seed-42 GPT-2 run. Fits a unit mean raw displacement
axis on rewrite/prompt_last only. Every patch is confined to the original
prompt_last H18, with no weight changes or LayerNorm/RMSNorm intervention.
"""
from __future__ import annotations

# Release-only filesystem defaults; computation below follows the source snapshot.
import sys as _bng_sys
from pathlib import Path as _BNGPath
_bng_sys.path.insert(0, str(_BNGPath(__file__).resolve().parents[1]))
from model_code.paths import RESEARCH_ROOT, OUTPUT_ROOT, DATA_ROOT, LLAMA_MODEL

import argparse
from contextlib import AbstractContextManager
import csv
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from diagnostics.gpt2_checkpoint_analysis import (
    NAMES, canonical_panel, capture_h18, checkpoint_path, geometry,
    json_hash, load_model_and_tokenizer, materialize, read, record, require,
    restore_checkpoint, validate_checkpoint, write_json, write_npz,
)

DEFAULT_SPLIT = OUTPUT_ROOT / '_Analysis_Cross_Layer/alphaedit_matched_geometry_causal_audit_n1000_order20260905_v1/delta_common/split.json'
STATES = ("HN", "SPHERE", "SADR")
FAMILIES = ("rewrite", "rephrase", "locality")
ENDPOINTS = {"rewrite": "EFF", "rephrase": "Gen", "locality": "LOC"}
RANDOM_SEEDS = (20260911, 20260912, 20260913)
DOSES = ("native", "perp_f025", "perp_f050", "perp_f075", "perp_f100")
GEOMETRY_FIELDS = (
    "realized_rel_parallel", "realized_rel_tangential", "realized_rel_displacement",
    "realized_h18_norm_ratio", "realized_h18_norm", "intervention_l2",
)


def conditions():
    return [dict(id="native", kind="native", fraction=0.0)] + [
        dict(id=f"perp_f{int(f * 100):03d}", kind="perp", fraction=f)
        for f in (.25, .50, .75, 1.0)
    ] + [
        dict(id="perp_match_axis", kind="perp_match_axis", fraction=1.0),
        dict(id="axis_f100", kind="axis", fraction=1.0),
        *[dict(id=f"randperp_add_s{s}", kind="random", fraction=1.0, seed=s) for s in RANDOM_SEEDS],
        dict(id="radial_match_perp100", kind="radial", fraction=1.0),
    ]


def write_csv(path, rows):
    require(bool(rows), f"No rows for {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_split(path, panel):
    source = read(path)
    by_id = {str(r["case_id"]): r for r in panel}
    fit = [str(i) for i in source["fit_case_ids"]]
    evaluated = [str(i) for i in source["pilot_case_ids"]]
    require(len(fit) == len(set(fit)) == 500, "Expected the fixed paper fit500")
    require(len(evaluated) == len(set(evaluated)) == 100, "Expected the fixed paper evaluation100")
    require(set(fit).isdisjoint(evaluated), "Axis fit/evaluation case IDs overlap")
    require(set(fit + evaluated) <= set(by_id), "Split case IDs absent from canonical panel")
    return [by_id[i] for i in fit], [by_id[i] for i in evaluated]


def validate_method(state, run_dir):
    manifest_path = run_dir / "run_manifest.json"
    manifest = read(manifest_path)
    require(manifest.get("editing_method") == "AlphaEdit", "Method manifest editor mismatch")
    require(manifest.get("model_seed") == 42 and manifest.get("num_requests_used") == 1000,
            "Method manifest must describe a completed 1,000-request seed42 run")
    require(manifest.get("append_eos_to_target") is True, "Expected genuine EOS in rewrite/rephrase targets")
    hp = manifest["effective_hparams"]
    flags = {"HN": bool(hp.get("residual_gain_regularization", False)),
             "SPHERE": bool(hp.get("sphere_enabled", False)),
             "SADR": bool(hp.get("sadr_regularization", False))}
    require(flags[state] and sum(flags.values()) == 1, f"Method label / active objective mismatch: {state}")
    require(not any(hp.get(name, False) for name in ("encore_enabled", "nas_enabled", "oedit_enabled", "nse_enabled")),
            "Unexpected extra editing method in intervention run")
    if state == "HN":
        expected = dict(residual_gain_lambda=1.0, residual_gain_subject_layers=[17],
                        residual_gain_objective="rho_minus_one", residual_gain_loss_type="positive_squared",
                        residual_gain_margin=0.0)
        require(all(hp.get(k) == v for k, v in expected.items()), "HN formulation differs from the paper")
    return dict(manifest=record(manifest_path), effective_hparams=hp)


def family_pair(row, family):
    return row[f"{family}_prompt"], row["locality_target"] if family == "locality" else row["target"]


def fit_axis(h, h0):
    """Normalize the mean raw displacement, not the mean of unit vectors."""
    delta = np.asarray(h, dtype=np.float64) - np.asarray(h0, dtype=np.float64)
    require(delta.ndim == 2 and np.isfinite(delta).all(), "Invalid fit displacement matrix")
    mean = delta.mean(0)
    norm = np.linalg.norm(mean)
    require(norm > 1e-12, "Shared mean displacement axis is undefined")
    return mean / norm, mean


def decompose(h, h0):
    h, h0 = np.asarray(h, dtype=np.float64), np.asarray(h0, dtype=np.float64)
    require(h.shape == h0.shape and h.ndim == 2, "Expected matching batch x hidden states")
    require(bool(np.isfinite(h).all() and np.isfinite(h0).all()), "Nonfinite intervention state")
    norm0 = np.linalg.norm(h0, axis=-1, keepdims=True)
    require(bool((norm0 > 0).all()), "Zero Base norm")
    u = h0 / norm0
    delta = h - h0
    parallel = np.sum(delta * u, axis=-1, keepdims=True) * u
    return u, delta, parallel, delta - parallel


def random_perpendicular(u, case_ids, seed):
    """Stable per-case directions, independent of batch size and worker order."""
    require(len(u) == len(case_ids), "Random direction case IDs mismatch")
    vectors = []
    for row, case_id in zip(u, case_ids):
        key = hashlib.sha256(f"gpt2-h18|{seed}|{case_id}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
        v = rng.standard_normal(row.shape)
        v -= np.dot(v, row) * row
        norm = np.linalg.norm(v)
        require(norm > 1e-12, "Degenerate random orthogonal direction")
        vectors.append(v / norm)
    return np.stack(vectors)


def intervention_target(cond, h, h0, axis, case_ids):
    """Return the intended state; actual dtype-rounded states are measured by hooks."""
    u, delta, parallel, perp = decompose(h, h0)
    h = np.asarray(h, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    require(axis.shape == (h.shape[1],) and abs(np.linalg.norm(axis) - 1) < 1e-6, "Nonunit axis")
    axis_amount = np.abs(delta @ axis)[:, None]
    kind = cond["kind"]
    if kind == "native":
        result = h.copy()
    elif kind == "perp":
        result = h - cond["fraction"] * perp
    elif kind == "perp_match_axis":
        fraction = np.minimum(axis_amount / np.maximum(np.linalg.norm(perp, axis=-1, keepdims=True), 1e-12), 1)
        result = h - fraction * perp
    elif kind == "axis":
        result = h - (delta @ axis)[:, None] * axis
    elif kind == "random":
        result = h + axis_amount * random_perpendicular(u, case_ids, cond["seed"])
    elif kind == "radial":
        desired = np.linalg.norm(h - perp, axis=-1, keepdims=True)
        current = np.linalg.norm(h, axis=-1, keepdims=True)
        require(bool((current > 0).all()), "Cannot radially scale a zero edited hidden")
        result = h * desired / current
    else:
        raise ValueError(f"Unknown intervention kind {kind}")
    require(bool(np.isfinite(result).all()), "Nonfinite intended intervention")
    return result.astype(np.float32)


def realized_geometry(h, h0, achieved):
    _, raw = geometry(np.asarray(achieved)[:, None], np.asarray(h0)[:, None])
    displacement = np.asarray(achieved, dtype=np.float64) - np.asarray(h0, dtype=np.float64)
    return dict(
        realized_rel_parallel=raw["p"], realized_rel_tangential=raw["q"],
        realized_rel_displacement=np.linalg.norm(displacement, axis=-1) / raw["base_norm"],
        realized_h18_norm_ratio=raw["kappa"], realized_h18_norm=raw["post_norm"],
        intervention_l2=np.linalg.norm(np.asarray(achieved, dtype=np.float64) - np.asarray(h, dtype=np.float64), axis=-1),
    )


class H18Intervention(AbstractContextManager):
    """Capture/replace only the original prompt_last input to GPT-2 block 18."""
    def __init__(self, model, positions, replacement=None, boundary=18):
        self.model, self.positions, self.replacement = model, positions, replacement
        self.boundary, self.handle = boundary, None
        self.before = self.after = None

    def __enter__(self):
        def hook(_module, args):
            h = args[0]
            positions = torch.as_tensor(self.positions, dtype=torch.long, device=h.device)
            rows = torch.arange(h.shape[0], device=h.device)
            require(len(positions) == len(rows), "Intervention batch size mismatch")
            self.before = h[rows, positions].detach().float().cpu().numpy().copy()
            if self.replacement is None:
                self.after = self.before.copy()
                return None
            replacement = torch.as_tensor(self.replacement, dtype=h.dtype, device=h.device)
            require(replacement.shape == (len(rows), h.shape[-1]), "Intervention hidden size mismatch")
            updated = h.clone()
            updated[rows, positions] = replacement
            self.after = updated[rows, positions].detach().float().cpu().numpy().copy()
            return (updated,) + args[1:]
        self.handle = self.model.transformer.h[self.boundary].register_forward_pre_hook(hook)
        return self

    def __exit__(self, *_):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False


@torch.inference_mode()
def forward_with_patch(model, inputs, starts, replacement=None, boundary=18):
    require(inputs["input_ids"].shape[1] <= model.config.n_positions, "Sequence exceeds GPT-2 context")
    with H18Intervention(model, np.asarray(starts) - 1, replacement, boundary=boundary) as hook:
        logits = model(**inputs, use_cache=False).logits
    require(hook.before is not None and hook.after is not None, "H18 hook did not fire")
    require(bool(torch.isfinite(logits).all()), "Nonfinite intervention logits")
    return logits, hook.before, hook.after


def prediction_rows(logits, starts, proofs, base_predictions=None):
    predicted = logits.argmax(-1).cpu().numpy()
    results = []
    for i, (start, proof) in enumerate(zip(starts, proofs)):
        gold = np.asarray(proof["target_token_ids"])
        pred = predicted[i, start - 1:-1]
        require(pred.shape == gold.shape and len(gold) > 0, "Prediction/target span mismatch")
        accuracy = float(np.equal(pred, gold).mean())
        base = pred if base_predictions is None else np.asarray(base_predictions[i])
        require(base.shape == pred.shape, "Base prediction span mismatch")
        results.append(dict(accuracy=accuracy, base_agreement=float(np.equal(pred, base).mean()),
                            predicted_token_ids=pred.tolist()))
    return results


@torch.inference_mode()
def capture_base(model, tokenizer, panel, batch_size, output):
    baseline = {}
    for family in FAMILIES:
        h, rows, proofs = [], [], []
        for begin in range(0, len(panel), batch_size):
            chunk = panel[begin:begin + batch_size]
            pairs = [family_pair(r, family) for r in chunk]
            inputs, starts, proof = materialize(tokenizer, [p[0] for p in pairs], [p[1] for p in pairs], next(model.parameters()).device)
            logits, hidden, _ = forward_with_patch(model, inputs, starts)
            h.append(hidden)
            rows.extend(prediction_rows(logits, starts, proof))
            proofs.extend(proof)
        baseline[family] = dict(h=np.concatenate(h), rows=rows, proofs=proofs)
        write_npz(output / "base" / f"{family}.npz", h18=baseline[family]["h"], case_ids=np.asarray([r["case_id"] for r in panel]))
        write_json(output / "base" / f"{family}.json", dict(case_ids=[r["case_id"] for r in panel], rows=rows, proofs=proofs))
    return baseline


@torch.inference_mode()
def run_state(model, tokenizer, panel, baseline, state, axis, batch_size, output):
    score_rows, checks = [], []
    cs = conditions()
    began = time.monotonic()
    for family in FAMILIES:
        geometry_batches = {c["id"]: {} for c in cs}
        predictions = []
        for begin in range(0, len(panel), batch_size):
            chunk = panel[begin:begin + batch_size]
            ids = [r["case_id"] for r in chunk]
            pairs = [family_pair(r, family) for r in chunk]
            inputs, starts, proofs = materialize(tokenizer, [p[0] for p in pairs], [p[1] for p in pairs], next(model.parameters()).device)
            end = begin + len(chunk)
            require(proofs == baseline[family]["proofs"][begin:end], "Base/edited token materialization differs")
            native_logits, h, _ = forward_with_patch(model, inputs, starts)
            h0 = baseline[family]["h"][begin:end]
            base_predictions = [r["predicted_token_ids"] for r in baseline[family]["rows"][begin:end]]
            sham, sham_before, sham_after = forward_with_patch(model, inputs, starts, h)
            sham_error = float((sham - native_logits).abs().max().item())
            require(sham_error == 0 and np.array_equal(sham_before, sham_after), "Zero intervention changes logits or H18")
            checks.append(dict(state=state, family=family, case_ids=ids, zero_delta_max_logit_difference=sham_error))
            del sham
            u, delta, parallel, perp = decompose(h, h0)
            clipped = np.abs(delta @ axis) > np.linalg.norm(perp, axis=-1) + 1e-8
            for cond in cs:
                if cond["id"] == "native":
                    logits, actual = native_logits, h
                else:
                    replacement = intervention_target(cond, h, h0, axis, ids)
                    logits, observed, actual = forward_with_patch(model, inputs, starts, replacement)
                    require(np.array_equal(observed, h), "Intervention input differs from the native same-checkpoint state")
                metrics = prediction_rows(logits, starts, proofs, base_predictions)
                realized = realized_geometry(h, h0, actual)
                geometry_batches[cond["id"]].setdefault("h18", []).append(actual)
                for name, values in realized.items():
                    geometry_batches[cond["id"]].setdefault(name, []).append(values)
                for i, metric in enumerate(metrics):
                    endpoint = metric["base_agreement"] if family == "locality" else metric["accuracy"]
                    score_rows.append(dict(
                        state=state, label=state, family=family, endpoint=ENDPOINTS[family],
                        condition=cond["id"], case_id=ids[i], fraction=cond["fraction"],
                        endpoint_value=endpoint, accuracy=metric["accuracy"], base_agreement=metric["base_agreement"],
                        axis_match_clipped=bool(clipped[i]),
                        **{name: float(values[i]) for name, values in realized.items()},
                    ))
                    predictions.append(dict(condition=cond["id"], case_id=ids[i], **metric))
                if cond["id"] != "native":
                    del logits
            del native_logits
            write_json(output / state / "status.json", dict(state=state, family=family,
                       completed_cases=end, total_cases=len(panel), seconds=time.monotonic() - began))
            print(f"[intervention] {state} {family} {end}/{len(panel)}", flush=True)
        for cond in cs:
            arrays = {k: np.concatenate(v) for k, v in geometry_batches[cond["id"]].items()}
            write_npz(output / state / "geometry" / family / f"{cond['id']}.npz",
                      case_ids=np.asarray([r["case_id"] for r in panel]), **arrays)
        write_json(output / state / f"predictions_{family}.json", predictions)
    write_csv(output / state / "scores.csv", score_rows)
    write_json(output / state / "sham_checks.json", checks)
    return score_rows


def bootstrap_summary(rows, *, n_bootstrap=10000, seed=20260912):
    require(n_bootstrap >= 1000, "Use at least 1,000 bootstrap resamples")
    ids = list(dict.fromkeys(str(r["case_id"]) for r in rows))
    states = [s for s in STATES if any(r["state"] == s for r in rows)]
    condition_ids = [c["id"] for c in conditions()]
    random_ids = [f"randperp_add_s{s}" for s in RANDOM_SEEDS]
    lookup = {}
    for row in rows:
        key = row["state"], row["family"], row["condition"], str(row["case_id"])
        require(key not in lookup, "Duplicate per-case intervention row")
        lookup[key] = row
    require(len(lookup) == len(states) * len(FAMILIES) * len(condition_ids) * len(ids), "Incomplete condition/family/case grid")
    index = np.random.default_rng(seed).integers(0, len(ids), size=(n_bootstrap, len(ids)))
    summaries, contrasts, matching, dose_rows = [], [], [], []
    def estimate(values):
        values = np.asarray(values, dtype=np.float64)
        lo, hi = np.percentile(values[index].mean(axis=1), [2.5, 97.5])
        return float(values.mean()), float(lo), float(hi)
    for state in states:
        for family in FAMILIES:
            by_condition = {c: [lookup[(state, family, c, i)] for i in ids] for c in condition_ids}
            scores = {c: np.array([float(r["endpoint_value"]) for r in data]) * 100 for c, data in by_condition.items()}
            scores["randperp_mean3"] = np.mean([scores[c] for c in random_ids], axis=0)
            match_checks = []
            for lhs, rhs, field, mandatory in [
                ("perp_f100", "radial_match_perp100", "realized_h18_norm", True),
                ("axis_f100", "perp_match_axis", "intervention_l2", False),
                *[("axis_f100", c, "intervention_l2", True) for c in random_ids],
            ]:
                left = np.asarray([r[field] for r in by_condition[lhs]], float)
                right = np.asarray([r[field] for r in by_condition[rhs]], float)
                error = np.abs(left - right)
                # Absolute tolerance allows float32 cancellation at very small interventions.
                agrees = error <= 1e-5 * np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0)
                if mandatory:
                    require(bool(agrees.all()), f"Realized matching failed: {state}/{family}/{lhs}/{rhs}")
                audit = dict(state=state, family=family, lhs=lhs, rhs=rhs, field=field,
                             n_cases=len(ids), max_absolute_error=float(error.max()),
                             max_scaled_error=float((error / np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0)).max()),
                             n_mismatched=int((~agrees).sum()), all_cases_matched=bool(agrees.all()))
                matching.append(audit)
                match_checks.append(audit)
            size_matched = next(r["all_cases_matched"] for r in match_checks if r["rhs"] == "perp_match_axis")
            for condition, values in scores.items():
                mean, lo, hi = estimate(values - scores["native"])
                source = random_ids if condition == "randperp_mean3" else [condition]
                summaries.append(dict(
                    state=state, label=state, family=family, endpoint=ENDPOINTS[family],
                    condition=condition, n_cases=len(ids), endpoint_value=float(values.mean()),
                    endpoint_minus_native=mean, endpoint_ci_low=lo, endpoint_ci_high=hi,
                    **{field: float(np.mean([float(r[field]) for c in source for r in by_condition[c]])) for field in GEOMETRY_FIELDS},
                ))
            specs = [
                ("matched_perp_minus_random_mean3", "perp_match_axis", "randperp_mean3", "matched_intervention_l2" if size_matched else "capped_size_match_incomplete"),
                ("perp100_minus_radial", "perp_f100", "radial_match_perp100", "matched_result_h18_norm"),
                ("axis_minus_matched_perp", "axis_f100", "perp_match_axis", "matched_intervention_l2" if size_matched else "capped_size_match_incomplete"),
                *[(f"dose_{b}_minus_{a}", b, a, "adjacent_orthogonal_dose") for a, b in zip(DOSES, DOSES[1:])],
            ]
            for name, lhs, rhs, matching_kind in specs:
                mean, lo, hi = estimate(scores[lhs] - scores[rhs])
                contrasts.append(dict(state=state, label=state, family=family, metric=ENDPOINTS[family], unit="pp",
                    contrast=name, lhs=lhs, rhs=rhs, matching=matching_kind, n_cases=len(ids),
                    mean_difference=mean, ci_low=lo, ci_high=hi, pointwise_ci_excludes_zero=bool(lo > 0 or hi < 0)))
            means = [float(scores[c].mean()) for c in DOSES]
            dose_rows.append(dict(state=state, family=family, endpoint=ENDPOINTS[family],
                **{c: m for c, m in zip(DOSES, means)},
                nondecreasing_point_estimates=bool((np.diff(means) >= -1e-12).all())))
    return summaries, contrasts, matching, dose_rows


def plot_results(output, summaries, contrasts, *, step, model_seed=42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    states = [s for s in STATES if any(r["state"] == s for r in summaries)]
    table = {(r["state"], r["family"], r["condition"]): r for r in summaries}
    loc = {(r["state"], r["contrast"]): r for r in contrasts if r["family"] == "locality"}
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    footer = (f"GPT-2 XL / AlphaEdit / step {step:,} / canonical edit order / model seed {model_seed}\n"
              "100 paired cases, disjoint from axis-fit 500 cases; pointwise 95% case-bootstrap intervals")
    specs = [
        ("matched_perp_minus_random_mean3", "(a) Partial orthogonal removal\n− random addition"),
        ("perp100_minus_radial", "(b) Full orthogonal removal\n− matched-result-norm radial"),
        ("axis_minus_matched_perp", "(c) Shared-axis removal\n− partial orthogonal removal"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.6), sharey=True)
    limits = [loc[state, name][field] for state in states for name, _ in specs for field in ("ci_low", "ci_high")]
    lower, upper = min(0.0, min(limits)), max(0.0, max(limits))
    padding = max(1.0, .08 * (upper - lower))
    for ax, (name, title) in zip(axes, specs):
        ax.axvline(0, color=".4", linestyle="--", linewidth=1)
        for j, state in enumerate(states):
            row = loc[state, name]
            good = row["pointwise_ci_excludes_zero"]
            color = "#1f5fa8" if good else "#5a5a5a"
            ax.errorbar(row["mean_difference"], j,
                        xerr=[[row["mean_difference"] - row["ci_low"]], [row["ci_high"] - row["mean_difference"]]],
                        fmt="o", color=color, markerfacecolor=color if good else "white", capsize=4, linewidth=2)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlabel("paired Δ LOC (pp)")
        ax.grid(axis="x", alpha=.25)
        ax.set_yticks(range(len(states)), states)
        ax.set_ylim(len(states) - .55, -.55)
        ax.set_xlim(lower - padding, upper + padding)
    fig.text(.5, .025, footer, ha="center", va="bottom", fontsize=10, linespacing=1.6)
    fig.subplots_adjust(left=.075, right=.99, bottom=.25, top=.84, wspace=.15)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"gpt2_h18_paired_interventions.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    colors = {"locality": "#1f5fa8", "rewrite": "#5a5a5a", "rephrase": "#c1502e"}
    fig, axes = plt.subplots(1, len(states), figsize=(5 * len(states), 4.8), squeeze=False, sharey=True)
    for ax, state in zip(axes[0], states):
        ax.axhline(0, color=".45", linestyle="--", linewidth=1)
        for family in ("locality", "rephrase", "rewrite"):
            rows = [table[state, family, c] for c in DOSES]
            means = np.array([r["endpoint_minus_native"] for r in rows])
            lo, hi = np.array([r["endpoint_ci_low"] for r in rows]), np.array([r["endpoint_ci_high"] for r in rows])
            ax.plot([0, 25, 50, 75, 100], means, "o-", color=colors[family], label=ENDPOINTS[family])
            ax.fill_between([0, 25, 50, 75, 100], lo, hi, color=colors[family], alpha=.12)
        ax.set_title(state)
        ax.set_xlabel("Orthogonal displacement removed (%)")
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", alpha=.25)
    axes[0, 0].set_ylabel("Change from unpatched checkpoint (pp)")
    axes[0, -1].legend(frameon=False)
    fig.text(.5, .025, footer, ha="center", va="bottom", fontsize=10, linespacing=1.6)
    fig.subplots_adjust(left=.07, right=.99, bottom=.255, top=.89, wspace=.18)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"gpt2_h18_orthogonal_dose_tradeoff.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, len(states), figsize=(5 * len(states), 4.8), squeeze=False, sharey=True)
    for ax, state in zip(axes[0], states):
        ax.axhline(0, color=".45", linestyle="--", linewidth=1)
        for offset, condition, label, color in [(-.15, "radial_match_perp100", "Radial scaling", "#9b9b9b"),
                                                (.15, "perp_f100", "Full orthogonal removal", "#1f5fa8")]:
            rows = [table[state, f, condition] for f in FAMILIES]
            values = np.asarray([r["endpoint_minus_native"] for r in rows])
            ax.errorbar(np.arange(3) + offset, values,
                        yerr=[values - np.array([r["endpoint_ci_low"] for r in rows]),
                              np.array([r["endpoint_ci_high"] for r in rows]) - values],
                        fmt="o", color=color, capsize=4, label=label)
        ax.set_xticks(range(3), [ENDPOINTS[f] for f in FAMILIES])
        ax.set_title(state)
        ax.grid(axis="y", alpha=.25)
    axes[0, 0].set_ylabel("Change from unpatched checkpoint (pp)")
    axes[0, -1].legend(frameon=False, fontsize=10)
    fig.text(.5, .025, footer, ha="center", va="bottom", fontsize=10, linespacing=1.6)
    fig.subplots_adjust(left=.07, right=.99, bottom=.255, top=.89, wspace=.18)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"gpt2_h18_matched_norm_tradeoff.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="Repeat LABEL=/absolute/run-dir; labels HN, SPHERE, SADR")
    parser.add_argument("--data-path", type=Path, default=DATA_ROOT / 'zsre/zsre_3k.json')
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--fit-batch-size", type=int, default=16)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260912)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    require(args.step == 1000, "The paper-replication intervention checkpoint must be step 1,000")
    require(min(args.batch_size, args.fit_batch_size) > 0, "Batch sizes must be positive")
    runs = {}
    for value in args.run:
        label, path = value.split("=", 1)
        require(label in STATES and label not in runs, "Labels must be unique HN/SPHERE/SADR")
        runs[label] = Path(path).resolve()
    ordered_states = [s for s in STATES if s in runs]
    panel, ordered = canonical_panel(runs[ordered_states[0]], args.data_path, 1000)
    require([str(r["case_id"]) for r in ordered] == [str(r["case_id"]) for r in panel],
            "This replication requires canonical edit order; it is not the Llama shuffled run")
    fitted, evaluated = load_split(args.split_path, panel)
    identities, paths, method_identities = {}, {}, {}
    for state, run_dir in runs.items():
        current, requests = canonical_panel(run_dir, args.data_path, 1000)
        require(current == panel and requests == ordered, "Method panels / edit order differ")
        path = checkpoint_path(run_dir, args.step)
        identity = validate_checkpoint(path, run_dir, requests, args.step, args.model_path)
        require(identity["metadata"]["editing_method"] == "AlphaEdit", "Intervention replication requires AlphaEdit")
        require(identity["metadata"].get("model_seed") == 42, "Expected model seed 42")
        paths[state], identities[state] = path, identity
        method_identities[state] = validate_method(state, run_dir)
    model_config = read(args.model_path / "config.json")
    require(model_config.get("model_type") == "gpt2" and model_config.get("n_embd") == 1600 and model_config.get("n_layer") == 48,
            "Expected GPT-2 XL (48 blocks, 1600 hidden)")
    protocol = dict(
        schema_version=1, experiment="GPT-2 XL H18 fixed-checkpoint intervention replication",
        editor="AlphaEdit", checkpoint_edit_count=1000, model_seed=42, edit_order="canonical",
        order_difference_from_llama="Existing Llama F5 used shuffle seed 20260905; this GPT-2 replication uses canonical order.",
        model_path=str(args.model_path.resolve()), model_config=record(args.model_path / "config.json"),
        data=record(args.data_path), source_split=record(args.split_path), fit_case_ids=[r["case_id"] for r in fitted],
        evaluation_case_ids=[r["case_id"] for r in evaluated], n_fit=500, n_cases=100,
        split_scope="Exact original paper case IDs; disjoint by case, not guaranteed entity/relation disjoint; old-order quartile balance is not claimed for canonical order.",
        axis="Unit-normalized mean raw H18_edited minus H18_Base over fit500 rewrite/prompt_last; reused across all families.",
        boundary="H18 = block17 output = block18 pre-LayerNorm input, original prompt_last only",
        conditions=conditions(), states=ordered_states, identities=identities, method_identities=method_identities,
        primary_metrics="EFF/Gen: complete-target teacher-forced case-macro token accuracy including genuine EOS; LOC: same-span Base argmax agreement.",
        target_contract="Shared checkpoint-analysis materialize(), left padding with explicit absolute position_ids, no truncation, original prompt boundary only.",
        random_seeds=list(RANDOM_SEEDS),
        random_directions="SHA256(gpt2-h18|seed|case_id), first 8 bytes little-endian seed for NumPy default_rng; projected orthogonal to per-family Base h. Stable across batch sizes, unlike legacy batch-restarted draws.",
        random_aggregation="Within-case mean of three seeds; statistical n=100, never 300; no inference over unseen random seeds.",
        partial_matching="perp_match_axis retains original min(axis removal L2 / perp L2,1) cap; actual size matching is audited and clipping is disclosed.",
        uncertainty="Pointwise paired case percentile bootstrap; conditional on this checkpoint/order; no multiplicity adjustment or equivalence claim.",
        n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed, dtype="float32", batch_size=args.batch_size,
        fit_batch_size=args.fit_batch_size, no_training=True, no_norm_layer_intervention=True,
        code=record(Path(__file__)), shared_core=record(ROOT / "diagnostics/gpt2_checkpoint_analysis.py"),
    )
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        require(read(protocol_path) == protocol, "Output protocol changed; choose a separate output directory")
    else:
        write_json(protocol_path, protocol)
    if args.audit_only:
        print(json.dumps(dict(audit="passed", states=ordered_states, fit=500, evaluated=100)))
        return
    if (output / "complete.json").exists():
        done = read(output / "complete.json")
        require(done["protocol_hash"] == json_hash(protocol), "Completion protocol mismatch")
        for item in done["artifacts"]:
            require(record(Path(item["path"])) == item, "Completed artifact changed")
        print("[reuse] completed intervention analysis", flush=True)
        return
    torch.set_num_threads(4)
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device, dtype="float32")
    pristine = {name: dict(model.named_parameters())[name].detach().cpu().clone() for name in NAMES}
    base_fit, _ = capture_h18(model, tokenizer, [r["rewrite_prompt"] for r in fitted], args.fit_batch_size)
    write_npz(output / "base" / "fit_rewrite_h18.npz", h18=base_fit, case_ids=np.asarray([r["case_id"] for r in fitted]))
    baseline = capture_base(model, tokenizer, evaluated, args.batch_size, output)
    rows = []
    for state in ordered_states:
        restore_checkpoint(model, pristine, paths[state], identities[state])
        versions = tuple((name, value._version) for name, value in model.named_parameters())
        current_fit, _ = capture_h18(model, tokenizer, [r["rewrite_prompt"] for r in fitted], args.fit_batch_size)
        axis, mean = fit_axis(current_fit[:, 0], base_fit[:, 0])
        write_npz(output / state / "axis.npz", axis=axis, mean_raw_displacement=mean,
                  fit_case_ids=np.asarray([r["case_id"] for r in fitted]), base_h18=base_fit, edited_h18=current_fit)
        rows.extend(run_state(model, tokenizer, evaluated, baseline, state, axis, args.batch_size, output))
        require(versions == tuple((name, value._version) for name, value in model.named_parameters()), "Inference changed model weights")
    summaries, contrasts, matching, dose_rows = bootstrap_summary(rows, n_bootstrap=args.n_bootstrap, seed=args.bootstrap_seed)
    write_csv(output / "per_case_scores.csv", rows)
    write_csv(output / "condition_summary.csv", summaries)
    write_csv(output / "paired_contrasts.csv", contrasts)
    write_csv(output / "matching_audit.csv", matching)
    write_csv(output / "dose_monotonicity.csv", dose_rows)
    plot_results(output, summaries, contrasts, step=args.step)
    mismatch = [r for r in matching if not r["all_cases_matched"]]
    write_json(output / "matching_status.json", dict(
        all_requested_pairings_realized=not mismatch, exceptions=mismatch,
        interpretation="Capped partial removal can violate size matching. All cases remain in estimates; plots use partial removal labels rather than claiming exact size matching."))
    (output / "README_KO.md").write_text(
        "# GPT-2 XL H18 개입 분석\n\n"
        "AlphaEdit 1,000-edit checkpoint, canonical 편집 순서, model seed 42. 기존 Llama의 shuffle20260905 결과와 편집 순서는 다릅니다.\n\n"
        "기존 분석과 같은 fit500 / 평가100 case IDs를 사용합니다. Shared axis는 fit500 rewrite prompt_last의 평균 raw displacement를 정규화한 벡터입니다. "
        "평가 case는 fit에서 제외되지만 편집 자체에 사용되지 않은 unseen knowledge라는 의미는 아닙니다.\n\n"
        "각 조건은 원래 prompt_last의 H18만 바꿉니다. EFF/Gen은 EOS를 포함한 teacher-forced token accuracy, LOC는 Base argmax와의 token agreement입니다. "
        "세 random seed는 사례 안에서 평균하고 n=100으로 bootstrap합니다. CI는 pointwise paired case 95% 구간입니다.\n\n"
        "`matching_audit.csv`와 `matching_status.json`에서 실제 결과 norm과 개입 L2의 일치를 확인합니다. "
        "perp_match_axis는 기존처럼 전체 직교 성분을 넘게 제거하지 않으므로 clipping 시 완전한 크기 일치가 깨질 수 있습니다. 사례를 제외하지 않고 그대로 보고합니다.\n\n"
        "`figures/gpt2_h18_paired_interventions`는 LOC의 세 paired contrast, `gpt2_h18_orthogonal_dose_tradeoff`는 세 방법의 제거량별 EFF/Gen/LOC, "
        "`gpt2_h18_matched_norm_tradeoff`는 동일 결과 norm의 radial / full orthogonal 제거 비교입니다. PNG/PDF를 함께 저장합니다.\n",
        encoding="utf-8")
    artifacts = [record(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name not in {"complete.json", "status.json"}]
    write_json(output / "complete.json", dict(complete=True, protocol_hash=json_hash(protocol), artifacts=artifacts,
               n_rows=len(rows), states=ordered_states, n_cases=100, n_fit=500, checkpoint_edit_count=1000))
    print(f"[complete] {output}", flush=True)


if __name__ == "__main__":
    main()
