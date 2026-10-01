"""Reaggregate the latest manuscript's RQ1 vectors and RQ3 paired contrasts.

All inputs are portable CSV files under ``data_dir``. Vector geometry is an
archived per-edit measurement: this module reaggregates it, checks the paired
reference definitions and norm ratios, and does not reopen activation tensors.
RQ3 intervals are independently reconstructed from paired case scores using the
original case order, 10,000 draws, seed 20260912, and percentile definition.
No models, original experiment directories, or network access are required.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


GROUP = ["model", "dataset", "editor", "method", "order_id"]
EDIT = GROUP + ["edit_index", "case_id"]
STATE = GROUP + ["edit_count"]
VECTOR_METRICS = [
    "relative_target_realization_error", "target_post_cosine",
    "requested_actual_displacement_cosine", "reference_vector_relative_difference",
    "reference_norm_relative_difference",
]
COMMON = "common_reference_displacement_cosine"
INITIAL = "init_reference_displacement_cosine"
FAMILIES = ["locality", "rewrite", "rephrase"]
ENDPOINTS = {"locality": "LOC", "rewrite": "EFF_TF", "rephrase": "GEN_TF"}
DOSES = [0.25, 0.5]
BOOTSTRAP_SEED = 20260912
BOOTSTRAP_DRAWS = 10000


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bool(values):
    if pd.api.types.is_bool_dtype(values):
        return values
    result = values.map({True: True, False: False, "True": True, "False": False,
                         1: True, 0: False, "1": True, "0": False})
    _require(result.notna().all(), "Invalid boolean column: " + str(values.name))
    return result.astype(bool)


def _close(actual, expected, label, atol=1e-10, rtol=1e-9):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol,
                               equal_nan=True, err_msg=label)


def _vector_summary(frame, identity, scope):
    out = dict(zip(GROUP, identity), scope=scope, n=len(frame))
    for metric in VECTOR_METRICS:
        all_values = frame[metric].to_numpy(dtype=float)
        values = all_values[np.isfinite(all_values)]
        out[metric + "_valid_n"] = len(values)
        out[metric + "_undefined_n"] = len(frame) - len(values)
        if len(values):
            for name, value in zip(["min", "q25", "median", "q75", "max"],
                                   np.quantile(values, [0, .25, .5, .75, 1])):
                out[metric + "_" + name] = float(value)
            out[metric + "_mean"] = float(values.mean())
    out.update(
        target_displacement_zero_n=int(frame.requested_displacement_zero.sum()),
        actual_displacement_zero_n=int(frame.actual_displacement_zero.sum()),
        pre_vector_nonfinite_n=int((~frame.pre_vector_finite).sum()),
        post_vector_nonfinite_n=int((~frame.post_vector_finite).sum()),
        all_required_vectors_finite_n=int((frame.pre_vector_finite & frame.post_vector_finite).sum()),
        pre_norm_reconstruction_max_relative_error=float(frame.pre_norm_reconstruction_relative_error.max()),
        post_norm_reconstruction_max_relative_error=float(frame.post_norm_reconstruction_relative_error.max()),
        reference_norm_above_1_percent_n=int(frame.reference_norm_relative_difference.gt(.01).sum()),
        reference_vector_above_1_percent_n=int(frame.reference_vector_relative_difference.gt(.01).sum()),
        original_target_ratio_median=float(frame.returned_target_ratio.median()),
        common_target_ratio_median=float(frame.common_target_ratio.median()),
        actual_post_pre_ratio_median=float(frame.actual_post_pre_ratio.median()),
        original_norm_ratio_spearman=float(spearmanr(frame.returned_target_ratio, frame.actual_post_pre_ratio)[0]),
        common_norm_ratio_spearman=float(spearmanr(frame.common_target_ratio, frame.actual_post_pre_ratio)[0]),
    )
    return out


def _reference_summary(group, identity):
    paired = group[group.paired_valid]
    common, initial = paired[COMMON].to_numpy(), paired[INITIAL].to_numpy()
    signed = initial - common
    absolute = np.abs(signed)
    return dict(
        zip(GROUP, identity), n_total=len(group),
        source_vectors_finite_n=int(group.source_vectors_finite.sum()),
        paired_valid_n=len(paired),
        init_reference_delta_zero_among_vector_valid_n=int(
            (group.init_reference_delta_zero & group.source_vectors_finite).sum()),
        common_matched_median=float(np.median(common)),
        init_matched_median=float(np.median(initial)),
        paired_signed_difference_median=float(np.median(signed)),
        init_minus_common_difference_of_medians=float(np.median(initial) - np.median(common)),
        paired_absolute_difference_median=float(np.median(absolute)),
        paired_absolute_difference_p95=float(np.quantile(absolute, .95)),
        paired_absolute_difference_max=float(absolute.max()),
    )


def _common_denominator(group, identity):
    target = group.returned_target_ratio.to_numpy()
    post = group.actual_post_pre_ratio.to_numpy()
    common = (group.returned_full_target_norm / group.actual_pre_norm).to_numpy()
    _close(common, group.common_target_ratio, "common target norm ratio", atol=1e-12, rtol=1e-12)
    old_median, common_median, post_median = [float(np.median(v)) for v in (target, common, post)]
    old_rho, common_rho = [float(spearmanr(v, post)[0]) for v in (target, common)]
    return dict(
        zip(GROUP, identity), n=len(group),
        target_ratio_median_original=old_median, target_ratio_median_common=common_median,
        post_ratio_median=post_median, target_median_change=common_median - old_median,
        spearman_original=old_rho, spearman_common=common_rho, spearman_change=common_rho - old_rho,
        target_minus_post_median_gap_original=old_median - post_median,
        target_minus_post_median_gap_common=common_median - post_median,
        paired_gap_median_original=float(np.median(target - post)),
        paired_gap_median_common=float(np.median(common - post)),
        paired_ratio_mae_original=float(np.mean(np.abs(target - post))),
        paired_ratio_mae_common=float(np.mean(np.abs(common - post))),
        per_edit_target_ratio_change_max=float(np.max(np.abs(common - target))),
        target_gt_post_count_original=int(np.sum(target > post)),
        target_gt_post_count_common=int(np.sum(common > post)),
    )


def _rq1(read, compare, write):
    vectors = read("rq1_vectors_per_edit.csv")
    references = read("rq1_reference_sensitivity_per_edit.csv")
    for frame in (vectors, references):
        _require(len(frame) == 40000 and not frame.duplicated(EDIT).any(), "RQ1 edit identity coverage")
        _require(frame.groupby(GROUP).size().eq(1000).all(), "RQ1 edits per configuration")
        _require(frame.order_id.eq("canonical").all(), "Noncanonical RQ1 input")
    for column in ["pre_vector_finite", "post_vector_finite"]:
        vectors[column] = _bool(vectors[column])
    for column in ["source_vectors_finite", "paired_valid", "init_reference_delta_zero"]:
        references[column] = _bool(references[column])
    vectors = vectors.sort_values(EDIT).reset_index(drop=True)
    references = references.sort_values(EDIT).reset_index(drop=True)
    pd.testing.assert_frame_equal(vectors[EDIT], references[EDIT], check_dtype=False)
    finite = vectors.pre_vector_finite & vectors.post_vector_finite
    _require(np.array_equal(finite, references.source_vectors_finite), "RQ1 vector availability differs")
    _require(int(finite.sum()) == 39473, "Expected 39,473 finite vector observations")
    _close(vectors.requested_actual_displacement_cosine, references[COMMON], "common-reference cosine", atol=2e-12)
    paired = np.isfinite(references[COMMON]) & np.isfinite(references[INITIAL])
    _require(np.array_equal(paired, references.paired_valid), "Paired validity does not match defined cosines")
    _require(int(paired.sum()) == 39257, "Expected 39,257 paired displacement cosines")
    _close(references.loc[paired, INITIAL] - references.loc[paired, COMMON],
           references.loc[paired, "paired_signed_difference"], "paired signed cosine difference", atol=2e-12)
    _close(np.abs(references.loc[paired, "paired_signed_difference"]),
           references.loc[paired, "paired_absolute_difference"], "paired absolute cosine difference", atol=2e-12)
    _require(references.loc[~paired, "paired_signed_difference"].isna().all(), "Undefined cosines were imputed")
    _require(np.array_equal(references.init_reference_delta_norm.eq(0), references.init_reference_delta_zero),
             "Zero initial-reference displacement flag differs from norm")
    _require(int((references.init_reference_delta_zero & finite).sum()) == 216,
             "Expected 216 zero initial-reference displacements among finite vectors")

    summary, ratios = [], []
    for identity, group in vectors.groupby(GROUP, sort=True):
        summary.append(_vector_summary(group, identity, "all_edits"))
        summary.append(_vector_summary(group[group.reference_norm_relative_difference.le(.01)], identity,
                                       "reference_norm_difference_at_most_1_percent"))
        ratios.append(_common_denominator(group, identity))
    summary = pd.DataFrame(summary)
    ratios = pd.DataFrame(ratios)
    ref_summary = pd.DataFrame([_reference_summary(group, identity)
                               for identity, group in references.groupby(GROUP, sort=True)])
    compare(summary, "expected/rq1_vectors_by_configuration.csv", GROUP + ["scope"])
    compare(ref_summary, "expected/rq1_reference_sensitivity_manuscript_table.csv", GROUP)
    compare(ratios, "expected/rq1_common_denominator_all40.csv", GROUP)
    write(summary, "rq1_vectors_by_configuration.csv")
    write(ref_summary, "rq1_reference_sensitivity_manuscript_table.csv")
    write(ratios, "rq1_common_denominator_all40.csv")
    all_rows = summary[summary.scope.eq("all_edits")]
    return dict(
        configurations=len(all_rows), edits=len(vectors), finite_vectors=int(finite.sum()),
        configurations_with_all_1000_vectors=int(all_rows.all_required_vectors_finite_n.eq(1000).sum()),
        nonfinite_vectors=int((~finite).sum()), paired_displacement_cosines=int(paired.sum()),
        zero_initial_displacements_on_finite_vectors=216,
        common_origin="cos(z - h_pre, h_post - h_pre)",
        comparison_origin="cos(z - h_init, h_post - h_pre)",
        geometry_scope="Reaggregation of archived per-edit vector measurements; no activation tensors reopened",
        common_denominator_scope="All 40,000 scalar norm records, including edits without finite stored vectors",
    )


def _label(prefix, dose):
    return f"{prefix}_f{int(dose * 100):03d}"


def _rq3(read, compare, write):
    frame = read("rq3_selected_case_scores.csv")
    canonical = read("same_norm_paired_contrasts.csv")
    frame.eligible_at_both_doses = _bool(frame.eligible_at_both_doses)
    case_key = STATE + ["dose_fraction", "family", "condition", "case_id"]
    _require(not frame.duplicated(case_key).any(), "Duplicate RQ3 paired case records")
    _require(frame.order_id.eq("canonical").all() and frame.edit_count.eq(1000).all(), "RQ3 endpoint scope")
    _require(frame.dose_fraction.isin(DOSES).all(), "Unexpected RQ3 reduction fraction")
    _require(frame.endpoint_value.between(0, 1).all(), "Invalid case endpoint score")
    semantic = frame[frame.family.ne("locality")]
    _require(semantic.n_target_tokens.gt(1).all(), "Content-only score requires an answer before EOS/EOT")
    _require(semantic.terminator_accuracy.isin([0, 1]).all(), "Termination accuracy is not a single-token indicator")
    reconstructed = ((semantic.n_target_tokens - 1) * semantic.semantic_accuracy + semantic.terminator_accuracy) / semantic.n_target_tokens
    _close(reconstructed, semantic.endpoint_value, "full/content/termination accuracy decomposition", atol=1e-12)

    # All random index matrices equal a fresh default_rng(SEED) call with this n,
    # exactly as in the archived analysis. Reusing them only saves CPU work.
    draws = {}

    def estimate(values):
        values = np.asarray(values, dtype=np.float64)
        n = len(values)
        _require(n > 0 and np.isfinite(values).all(), "Empty or nonfinite paired contrast")
        if n not in draws:
            draws[n] = np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, size=(BOOTSTRAP_DRAWS, n))
        lo, hi = np.percentile(values[draws[n]].mean(axis=1), [2.5, 97.5])
        return float(values.mean()), float(lo), float(hi)

    groups = {identity: group for identity, group in frame.groupby(STATE, sort=True)}
    _require(len(groups) == 34, "Expected 34 states eligible for norm-matched comparisons")
    results, selections = [], []
    for identity, group in groups.items():
        by_dose = {}
        cases = {}
        for dose in DOSES:
            subset = group[group.dose_fraction.eq(dose)]
            ids = subset[subset.family.eq("locality") & subset.condition.eq("native")].case_id.tolist()
            _require(len(ids) == len(set(ids)) > 0, "Invalid eligible request IDs")
            by_dose[dose] = ids
            for family in FAMILIES:
                for condition in ["native", _label("perp", dose), _label("projection_match", dose)]:
                    current = subset[subset.family.eq(family) & subset.condition.eq(condition)]
                    _require(current.case_id.tolist() == ids, "RQ3 cases/order differ across paired conditions or families")
                    cases[dose, family, condition] = current.set_index("case_id")
        ids25, ids50 = by_dose[.25], by_dose[.5]
        _require(set(ids50) <= set(ids25), "50% eligibility is not a subset of 25% eligibility")
        common = [cid for cid in ids25 if cid in set(ids50)]
        selections.append(dict(zip(STATE, identity), n_25=len(ids25), n_50=len(ids50), n_both=len(common),
                               case_ids_25="|".join(ids25), case_ids_50="|".join(ids50),
                               case_ids_both="|".join(common)))
        _require(np.array_equal(group.case_id.isin(common), group.eligible_at_both_doses),
                 "Saved common-dose eligibility does not equal the ID intersection")
        for family in FAMILIES:
            _close(cases[.25, family, "native"].loc[common, "endpoint_value"],
                   cases[.5, family, "native"].loc[common, "endpoint_value"], "Baseline scores changed between doses", atol=0, rtol=0)
        for dose in DOSES:
            per, par = _label("perp", dose), _label("projection_match", dose)
            for scope, ids in [("dose_specific", by_dose[dose]), ("both_doses", common)]:
                for family in FAMILIES:
                    metrics = [("full", "endpoint_value")]
                    if family != "locality":
                        metrics.append(("content_only", "semantic_accuracy"))
                    for metric, column in metrics:
                        for operation, lhs, rhs in [("orthogonal", per, "native"), ("parallel_only", par, "native"),
                                                    ("parallel_minus_orthogonal", par, per)]:
                            # Preserve the archived operation order, not 100*(lhs-rhs).
                            values = (100 * cases[dose, family, lhs].loc[ids, column].to_numpy()
                                      - 100 * cases[dose, family, rhs].loc[ids, column].to_numpy())
                            point, lo, hi = estimate(values)
                            results.append(dict(zip(STATE, identity), scope=scope, dose_fraction=dose, family=family,
                                                endpoint=ENDPOINTS[family], metric=metric, operation=operation,
                                                n_cases=len(ids), mean_difference_pp=point, ci_low_pp=lo, ci_high_pp=hi))
    sensitivity = pd.DataFrame(results)
    selection = pd.DataFrame(selections)
    _require(len(sensitivity) == 2040, "Expected 2,040 state-specific sensitivity contrasts")
    contrast_key = STATE + ["scope", "dose_fraction", "family", "metric", "operation"]
    compare(selection, "expected/rq3_common_dose_case_ids.csv", STATE)
    compare(sensitivity, "expected/rq3_sensitivity_state_contrasts.csv", contrast_key)

    mean_keys = ["scope", "dose_fraction", "family", "endpoint", "metric", "operation"]
    averages = []
    for identity, group in sensitivity.groupby(mean_keys, sort=True):
        _require(len(group) == 34, "Sensitivity average must weight the same 34 states equally")
        averages.append(dict(zip(mean_keys, identity), n_states=len(group), n_cases_min=int(group.n_cases.min()),
                             n_cases_max=int(group.n_cases.max()), mean_difference_pp=float(group.mean_difference_pp.mean())))
    averages = pd.DataFrame(averages)
    compare(averages, "expected/rq3_sensitivity_overall_means.csv", mean_keys)

    direct = sensitivity[sensitivity.scope.eq("dose_specific") & sensitivity.metric.eq("full")
                         & sensitivity.operation.eq("parallel_minus_orthogonal")].set_index(STATE + ["dose_fraction", "family"])
    direct_rows = []
    _require(len(canonical) == len(direct) == 204, "Expected 204 canonical paired contrasts")
    for row in canonical.to_dict("records"):
        identity = tuple(row[k] for k in STATE + ["dose_fraction", "family"])
        calculated = direct.loc[identity]
        _require(row["scope"] == "B_common_all_families" and row["lhs"] == _label("projection_match", row["dose_fraction"])
                 and row["rhs"] == _label("perp", row["dose_fraction"]), "Canonical contrast direction/scope mismatch")
        direct_rows.append(dict(row, **{key: calculated[key] for key in ["n_cases", "mean_difference_pp", "ci_low_pp", "ci_high_pp"]}))
    direct_frame = pd.DataFrame(direct_rows)
    compare(direct_frame, "same_norm_paired_contrasts.csv", STATE + ["dose_fraction", "family"])

    interval_counts = []
    for (dose, family), group in direct_frame.groupby(["dose_fraction", "family"], sort=True):
        interval_counts.append(dict(dose_fraction=dose, endpoint=ENDPOINTS[family], n_states=len(group),
                                    ci_entirely_negative=int(group.ci_high_pp.lt(0).sum()),
                                    ci_entirely_positive=int(group.ci_low_pp.gt(0).sum()),
                                    ci_includes_zero=int((group.ci_low_pp.le(0) & group.ci_high_pp.ge(0)).sum())))
    write(selection, "rq3_common_dose_case_ids.csv")
    write(sensitivity, "rq3_sensitivity_state_contrasts.csv")
    write(averages, "rq3_sensitivity_overall_means.csv")
    write(direct_frame, "same_norm_paired_contrasts.csv")
    write(pd.DataFrame(interval_counts), "rq3_pointwise_interval_counts.csv")
    return dict(
        endpoint_states=len(groups), selected_case_score_rows=len(frame), direct_contrasts=len(direct_frame),
        sensitivity_state_contrasts=len(sensitivity), equally_weighted_state_mean_rows=len(averages),
        states_with_changed_case_set=int(selection.n_25.ne(selection.n_both).sum()),
        removed_state_case_pairs=int((selection.n_25 - selection.n_both).sum()),
        bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=BOOTSTRAP_DRAWS,
        bootstrap_generator="numpy.random.default_rng", percentile_levels=[2.5, 97.5],
        resampling_unit="paired request ID within a fixed endpoint state and reduction fraction",
        original_case_order_preserved=True, aggregate_state_mean_confidence_interval=False,
        multiplicity_adjustment=False,
    )


def run(data_dir, output_dir):
    """Write independently recomputed summaries and return a JSON-safe report."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    _require(data_dir.resolve() != output_dir.resolve(), "Output directory must differ from input directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs, checks, outputs = {}, [], []

    def read(relative):
        path = data_dir / relative
        inputs[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path, float_precision="round_trip", dtype={"case_id": str})

    def compare(actual, relative, keys):
        expected = read(relative)
        _require(not actual.duplicated(keys).any() and not expected.duplicated(keys).any(), "Duplicate summary keys: " + relative)
        actual = actual.sort_values(keys).reset_index(drop=True)
        expected = expected.sort_values(keys).reset_index(drop=True)
        _require(len(actual) == len(expected), "Row count differs: " + relative)
        largest = 0.0
        for column in expected:
            _require(column in actual, "Missing reconstructed column: " + relative + ":" + column)
            if pd.api.types.is_numeric_dtype(expected[column]):
                _close(actual[column], expected[column], relative + ":" + column)
                differences = np.abs(actual[column].to_numpy(dtype=float) - expected[column].to_numpy(dtype=float))
                finite = differences[np.isfinite(differences)]
                if len(finite):
                    largest = max(largest, float(finite.max()))
            else:
                pd.testing.assert_series_equal(actual[column], expected[column], check_names=False, check_dtype=False)
        checks.append(dict(reference=relative, rows=len(expected), compared_columns=list(expected.columns),
                           max_absolute_numeric_difference=largest, passed=True))

    def write(frame, name):
        frame.to_csv(output_dir / name, index=False, float_format="%.17g")
        outputs.append(name)

    report = dict(scope="CPU reaggregation of archived RQ1 per-edit measurements and RQ3 paired case scores",
                  rq1=_rq1(read, compare, write), rq3=_rq3(read, compare, write),
                  checks=checks, input_sha256=inputs, outputs=outputs, model_inference=False, passed=True)
    (output_dir / "vectors_interventions_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=root / "data/manuscript")
    parser.add_argument("--output-dir", type=Path, default=root / "build/manuscript_vectors_interventions")
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
