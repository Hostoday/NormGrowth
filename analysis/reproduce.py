"""Compute RQ1--RQ3 summaries from externally supplied measurement tables.

Run with --data-dir pointing to a flat directory of experiment measurements.
See analysis/README.md for the input schema. No model inference is performed.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .snapshot import CHECKPOINT, STATE, filename

DATA = None
REQUIRED_INPUTS = (
    "rq1_per_edit.csv", "rq1_vectors_per_edit.csv",
    "rq1_reference_sensitivity_per_edit.csv", "checkpoints.csv",
    "intervention_doses.csv", "same_norm_conditions.csv",
    "rq3_selected_case_scores.csv",
)


def read(name):
    return pd.read_csv(DATA / name, float_precision="round_trip")


def rq1(output):
    frame = read(filename("rq1"))
    assert len(frame) == 40000 and not frame.duplicated(STATE + ["edit_index"]).any()
    rows = []
    for key, group in frame.groupby(STATE, sort=True):
        assert len(group) == 1000
        x = group.returned_target_ratio.to_numpy()
        y = group.actual_post_pre_ratio.to_numpy()
        fit = np.polyfit(x, y, 1)
        residual = y - np.polyval(fit, x)
        rows.append(dict(zip(STATE, key), n=len(group), regression_intercept=fit[1],
            regression_slope=fit[0], regression_r2=1 - np.sum(residual**2) / np.sum((y-y.mean())**2),
            target_actual_spearman=spearmanr(x,y)[0], target_ratio_median=np.median(x),
            actual_ratio_median=np.median(y), actual_lt_target_count=np.sum(y<x),
            target_nonexpands_count=np.sum(x<=1+1e-6), actual_nonexpands_count=np.sum(y<=1+1e-6),
            pi_z=np.mean(x<=1+1e-6), pi_post=np.mean(y<=1+1e-6)))
    result = pd.DataFrame(rows)
    result.to_csv(output / "rq1_condition_summary_40.csv", index=False)
    return dict(edits=len(frame), conditions=len(result))


def rq3(output):
    frame = read(filename("rq3_doses"))
    assert len(frame) == 200 and not frame.duplicated(CHECKPOINT+["dose_fraction"]).any()
    rows = []
    for dose, group in frame[frame.dose_fraction.gt(0)].groupby("dose_fraction"):
        assert len(group) == 40
        # Joint-count classification uses unrounded per-state indicators.
        rows.append(dict(removed_fraction=dose, n_states=len(group),
            mean_delta_LOC=group.delta_LOC.mean(), mean_delta_EFF_TF=group.delta_EFF_TF.mean(),
            mean_delta_GEN_TF=group.delta_GEN_TF.mean(),
            LOC_up_no_edit_loss=int(group.LOC_up_no_edit_loss.sum()),
            LOC_up_both_losses_at_most_1pp=int(group.LOC_up_both_losses_at_most_1pp.sum())))
    result = pd.DataFrame(rows)
    result.to_csv(output / "rq3_orthogonal_dose_summary.csv", index=False)

    conditions = read(filename("rq3_same_norm"))
    conditions["endpoint"] = conditions.endpoint.str.upper()
    assert len(conditions) == 480 and conditions.scope.eq("B_common_all_families").all()
    index = CHECKPOINT+["dose_fraction"]
    operations = ["perp", "projection_match"]
    endpoints = ["LOC", "EFF", "GEN"]
    assert set(conditions.operation) == set(operations)
    assert set(conditions.endpoint) == set(endpoints)
    assert set(conditions.dose_fraction) == {.25, .5}
    assert not conditions.duplicated(index+["operation", "endpoint"]).any()
    assert conditions.groupby(index).size().eq(len(operations)*len(endpoints)).all()
    assert conditions.groupby(index).n_cases.nunique().eq(1).all()
    assert conditions.n_cases.ge(0).all()
    eligible = conditions[conditions.n_cases.gt(0)].copy()
    wide = eligible.pivot(index=index, columns=["operation","endpoint"], values="delta_pp")
    wide = wide.reindex(columns=pd.MultiIndex.from_product([operations, endpoints]))
    assert np.isfinite(wide).all().all()
    counts = eligible.groupby(index).n_cases
    assert counts.nunique().eq(1).all()
    differences = wide.index.to_frame(index=False)
    differences["n_cases"] = counts.first().reindex(wide.index).to_numpy()
    for endpoint, metric in [("LOC","LOC"),("EFF","EFF_TF"),("GEN","GEN_TF")]:
        differences["difference_"+metric+"_pp"] = (wide["projection_match",endpoint]-wide["perp",endpoint]).to_numpy()
    differences.to_csv(output / "rq3_same_norm_state_differences.csv", index=False)
    means = []
    dose_counts = {}
    for dose in [.25,.5]:
        state_values = wide[wide.index.get_level_values("dose_fraction") == dose].droplevel("dose_fraction")
        dose_counts[str(dose)] = dict(eligible_states=len(state_values),
            total_state_case_count=int(differences[differences.dose_fraction.eq(dose)].n_cases.sum()))
        for operation, source in [("orthogonal","perp"),("parallel","projection_match"),("parallel_minus_orthogonal",None)]:
            values = state_values[source] if source else state_values["projection_match"]-state_values["perp"]
            means.append(dict(scope="all_cohorts_equal_eligible_state", dose_fraction=dose,
                dose_percent=round(100*dose),operation=operation,n_states=len(values),
                total_state_case_count=int(differences[differences.dose_fraction.eq(dose)].n_cases.sum()),
                n_joint_condition=int(((values.LOC>1e-10)&(values.EFF>=-1e-10)&(values.GEN>=-1e-10)).sum()),
                mean_delta_LOC_pp=values.LOC.mean(),mean_delta_EFF_TF_pp=values.EFF.mean(),
                mean_delta_GEN_TF_pp=values.GEN.mean()))
    result = pd.DataFrame(means)
    result.to_csv(output / "rq3_same_norm_overall_means.csv", index=False)
    return dict(orthogonal_states=len(frame[CHECKPOINT].drop_duplicates()),
                same_norm_eligible_state_dose_rows=len(wide), same_norm_by_dose=dose_counts)


def main():
    global DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing your measurement CSVs (see analysis/README.md)")
    parser.add_argument("--output-dir", type=Path, default=Path("build/results"))
    args = parser.parse_args()
    DATA = args.data_dir.resolve()
    output = args.output_dir.resolve()
    if output == DATA or DATA in output.parents:
        parser.error("Write outputs outside the input measurement directory")
    missing = [name for name in REQUIRED_INPUTS if not (DATA / name).is_file()]
    if missing:
        parser.error("Missing measurement inputs: " + ", ".join(missing) + ". See analysis/README.md.")
    output.mkdir(parents=True, exist_ok=True)
    rq1_counts = rq1(output)
    rq3_counts = rq3(output)
    from . import manuscript_statistics, manuscript_vectors_interventions
    rq2_report = manuscript_statistics.run(DATA, output / "rq2")
    vector_report = manuscript_vectors_interventions.run(DATA, output / "rq1_rq3")

    # A completed TF evaluation table may be supplied alongside the measurements.
    performance_path = DATA / filename("performance")
    performance_conditions = None
    input_names = list(REQUIRED_INPUTS)
    if performance_path.is_file():
        performance = read(filename("performance"))
        assert len(performance) == 40
        assert performance[["n_rewrite", "n_rephrase", "n_locality"]].eq(1000).all().all()
        assert not performance.duplicated(CHECKPOINT).any()
        assert performance.activation_intervention.eq("none").all()
        assert performance.target_span.str.contains("EOS/EOT").all()
        columns = ["eff_tf_percent", "gen_tf_percent", "locality_percent"]
        assert np.isfinite(performance[columns]).all().all()
        assert performance[columns].ge(0).all().all() and performance[columns].le(100).all().all()
        performance.to_csv(output / performance_path.name, index=False)
        performance_conditions = len(performance)
        input_names.append(performance_path.name)

    report = dict(
        status="completed", model_inference_performed=False,
        rq1_counts=rq1_counts, rq3_counts=rq3_counts,
        analyses=dict(rq2=rq2_report, rq1_rq3=vector_report),
        performance_conditions=performance_conditions,
        input_sha256={name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
                      for name in input_names},
        endpoint_score_source="Optional supplied TF evaluation table; token predictions are not regenerated",
        inferential_scope="Whole-trajectory resampling for RQ2; paired-case resampling for RQ3. Intervals do not represent independent editing runs.",
        numpy_version=np.__version__, pandas_version=pd.__version__,
    )
    (output / "analysis_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Computed RQ1 summaries for {rq1_counts['edits']:,} edits, RQ2 statistics, "
          f"and RQ3 summaries for {rq3_counts['orthogonal_states']} states.")


if __name__ == "__main__":
    main()
