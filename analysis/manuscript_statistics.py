"""Reproduce the manuscript's RQ2 statistics from packaged checkpoint CSVs.

Run ``python -m analysis.manuscript_statistics --output-dir build/manuscript_rq2``.
Only saved numeric measurements are read: no model, network, or NPZ is needed.
The archived reference tables are comparison targets, never analysis inputs.
The formulas and deterministic bootstrap follow the manuscript's archived
trajectory_statistics, rq2_controls, and collapse-exclusion analyses.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import qr
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
KEY = ["model", "dataset", "editor", "method", "order_id"]
TIMES = [50, 100, 150, 200, 250, 300, 500, 750, 1000]
CONTEXTS = ["rewrite", "locality"]
DATASETS = ["zsRE", "CounterFact"]
SPECS = ["unadjusted", "D_H_rank", "T_rank", "T_FE", "D_H_rank+T_rank",
         "D_H_rank+T_FE", "T_FE+trajectory_FE", "D_H_rank+T_FE+trajectory_FE"]
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260928


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 1e-14 else np.nan


def _spearman(x, y):
    return _corr(rankdata(x, method="average"), rankdata(y, method="average"))


def _design(group, context, specification):
    parts = [np.ones((len(group), 1))]
    if "D_H_rank" in specification:
        parts.append(rankdata(group[f"{context}_mean_abs_norm_deviation"])[:, None])
    if "d_rank" in specification:
        parts.append(rankdata(group[f"{context}_mean_d"])[:, None])
    if "T_rank" in specification:
        parts.append(rankdata(group.edit_count)[:, None])
    if "T_FE" in specification:
        parts.append(pd.get_dummies(group.edit_count.astype(str), drop_first=True).to_numpy(dtype=float))
    if "trajectory_FE" in specification:
        parts.append(pd.get_dummies(group.trajectory_id, drop_first=True).to_numpy(dtype=float))
    return np.column_stack(parts)


def _residual_summary(x, y, design):
    """Average ranks, OLS residuals, and an independent pivoted-QR check."""
    rx, ry = rankdata(x), rankdata(y)
    xy = np.column_stack([rx, ry])
    coef, _, design_rank, singular = np.linalg.lstsq(design, xy, rcond=None)
    residual = xy - design @ coef
    residual -= residual.mean(axis=0)
    ex, ey = residual.T
    nx, ny = np.linalg.norm(ex), np.linalg.norm(ey)
    tx, ty = np.linalg.norm(rx - rx.mean()), np.linalg.norm(ry - ry.mean())
    dimension = len(x) - int(design_rank)
    reasons = []
    if tx == 0:
        reasons.append("constant_predictor_rank")
    if ty == 0:
        reasons.append("constant_LOC_rank")
    if nx <= 1e-12 * max(tx, 1.0):
        reasons.append("constant_or_numerically_zero_predictor_residual")
    if ny <= 1e-12 * max(ty, 1.0):
        reasons.append("constant_or_numerically_zero_LOC_residual")
    if dimension < 2:
        reasons.append("fewer_than_two_residual_dimensions")
    rho = float(ex @ ey / (nx * ny)) if not reasons else np.nan
    basis = qr(design, mode="economic", pivoting=True)[0][:, :int(design_rank)]
    qx, qy = rx - basis @ (basis.T @ rx), ry - basis @ (basis.T @ ry)
    qr_rho = _corr(qx, qy) if not reasons else np.nan
    return dict(rho=rho, defined=not reasons, undefined_reason=";".join(reasons),
                n_design_columns=design.shape[1], design_rank=int(design_rank),
                rank_deficient=bool(design_rank < design.shape[1]),
                residual_subspace_dimension=dimension,
                nominal_partial_df_no_inference=dimension - 1,
                predictor_residual_rank_variance_fraction=float(nx * nx / (tx * tx)) if tx else np.nan,
                LOC_residual_rank_variance_fraction=float(ny * ny / (ty * ty)) if ty else np.nan,
                design_nonzero_singular_condition_number=float(singular[0] / singular[int(design_rank)-1]),
                independent_QR_rho=qr_rho,
                independent_QR_absolute_error=abs(rho - qr_rho) if not reasons else np.nan)


def _panels(frame):
    if len(frame) != 360 or frame.trajectory_id.nunique() != 40:
        raise ValueError("Expected 360 checkpoints from 40 canonical trajectories")
    if not frame.order_id.eq("canonical").all() or frame.duplicated(KEY + ["edit_count"]).any():
        raise ValueError("Noncanonical or duplicate checkpoints")
    if frame.groupby("dataset").size().to_dict() != {"CounterFact": 180, "zsRE": 180}:
        raise ValueError("Expected 180 checkpoints per dataset")
    for _, group in frame.groupby("trajectory_id"):
        if sorted(group.edit_count.tolist()) != TIMES or len(group[KEY].drop_duplicates()) != 1:
            raise ValueError("Each trajectory must have all nine stated edit counts")
    suffixes = ["mean_q", "mean_d", "mean_kappa", "mean_abs_norm_deviation", "mean_p",
                "mean_p_squared", "mean_q_squared", "mean_kappa_squared", "mean_angle_degrees"]
    numeric = [f"{context}_{suffix}" for context in CONTEXTS for suffix in suffixes]
    if not np.isfinite(frame[numeric + ["locality_percent"]].to_numpy()).all():
        raise ValueError("Nonfinite RQ2 measurements")
    threshold = [f"{context}_{suffix}" for context in CONTEXTS
                 for suffix in ["mean_q", "mean_kappa", "mean_abs_norm_deviation"]]
    restricted = frame[threshold].le(3).all(axis=1)
    collapsed = frame.model.eq("Llama") & frame.editor.eq("MEMIT") & frame.method.isin(["Native", "SPHERE", "SADR"])
    if frame[restricted].groupby("dataset").size().to_dict() != {"CounterFact": 169, "zsRE": 168}:
        raise ValueError("The geometry-restricted panel must contain 337 checkpoints")
    if int(collapsed.sum()) != 54 or frame[collapsed].trajectory_id.nunique() != 6:
        raise ValueError("The collapse sensitivity must remove six complete trajectories")
    return {"full360": frame, "restricted337": frame[restricted], "excluded306": frame[~collapsed]}, restricted, collapsed


def _rank_controls(panels):
    rows = []
    for panel in ["full360", "restricted337"]:
        for dataset in DATASETS:
            for model in ["Pooled", "Llama", "GPT-2 XL"]:
                group = panels[panel][panels[panel].dataset.eq(dataset)]
                if model != "Pooled":
                    group = group[group.model.eq(model)]
                for context in CONTEXTS:
                    for predictor in ["q", "d"]:
                        for specification in SPECS:
                            result = _residual_summary(group[f"{context}_mean_{predictor}"], group.locality_percent,
                                                       _design(group, context, specification))
                            rows.append(dict(panel=panel, model_scope=model, dataset=dataset, context=context,
                                             predictor=predictor, specification=specification,
                                             n_checkpoints=len(group), n_trajectories=group.trajectory_id.nunique(),
                                             n_edit_counts=group.edit_count.nunique(), **result))
    return pd.DataFrame(rows)


def _composition(panels):
    rows = []
    for panel in ["full360", "restricted337"]:
        for dataset, group in panels[panel].groupby("dataset", sort=True):
            for context in CONTEXTS:
                p = group[f"{context}_mean_p"]
                P = 2 * p + group[f"{context}_mean_p_squared"]
                Q = group[f"{context}_mean_q_squared"]
                if not Q.gt(0).all():
                    raise ValueError("Compensation ratio requires positive orthogonal second moments")
                np.testing.assert_allclose(P + Q, group[f"{context}_mean_kappa_squared"] - 1,
                                           atol=1e-10, rtol=1e-10)
                for control in ["d_rank", "d_rank+T_FE", "d_rank+T_FE+trajectory_FE"]:
                    design = _design(group, context, control)
                    for predictor, x in [("p", p), ("P", P), ("c", -P / Q),
                                         ("angle_degrees", group[f"{context}_mean_angle_degrees"])]:
                        result = _residual_summary(x, group.locality_percent, design)
                        fraction = result["predictor_residual_rank_variance_fraction"]
                        if not result["defined"] or fraction <= 1e-12:
                            raise ValueError("Undefined composition coefficient")
                        rows.append(dict(panel=panel, dataset=dataset, context=context, predictor=predictor,
                                         controls=control, n_checkpoints=len(group),
                                         n_condition_trajectories=group.trajectory_id.nunique(),
                                         design_rank=result["design_rank"],
                                         residual_subspace_dimension=result["residual_subspace_dimension"],
                                         partial_rank_correlation_LOC=result["rho"],
                                         unadjusted_spearman_LOC=_spearman(x, group.locality_percent),
                                         predictor_d_spearman=_spearman(x, group[f"{context}_mean_d"]),
                                         predictor_residual_rank_variance_fraction=fraction,
                                         predictor_rank_VIF=1 / fraction,
                                         LOC_residual_rank_variance_fraction=result["LOC_residual_rank_variance_fraction"]))
    return pd.DataFrame(rows)


def _collapse(panels):
    rows = []
    for panel, name in [("full", "full360"), ("exclude_six_trajectories", "excluded306")]:
        for dataset, group in panels[name].groupby("dataset", sort=True):
            for context in CONTEXTS:
                for metric in ["mean_q", "mean_d", "mean_kappa", "mean_abs_norm_deviation"]:
                    for control in ["unadjusted", "D_H_rank+T_rank", "D_H_rank+T_FE+trajectory_FE"]:
                        if metric in ["mean_kappa", "mean_abs_norm_deviation"] and control != "unadjusted":
                            continue
                        result = _residual_summary(group[f"{context}_{metric}"], group.locality_percent,
                                                   _design(group, context, control))
                        if not result["defined"]:
                            raise ValueError("Undefined collapse-exclusion coefficient")
                        rows.append(dict(panel=panel, dataset=dataset, context=context, metric=metric,
                                         controls=control, n=len(group), trajectories=group.trajectory_id.nunique(),
                                         rho=result["rho"]))
    return pd.DataFrame(rows)


def _within_trajectory(frame):
    rows = []
    for context in CONTEXTS:
        for trajectory, group in frame.groupby("trajectory_id", sort=True):
            group = group.sort_values("edit_count")
            row = {key: group[key].iloc[0] for key in KEY}
            row.update(context=context, trajectory_id=trajectory, n_checkpoints=len(group),
                       n_unique_q=group[f"{context}_mean_q"].nunique(), n_unique_LOC=group.locality_percent.nunique())
            for metric, suffix in [("q", "mean_q"), ("d", "mean_d"), ("D_H", "mean_abs_norm_deviation")]:
                row[f"rho_{metric}_LOC"] = _spearman(group[f"{context}_{suffix}"], group.locality_percent)
            rows.append(row)
    return pd.DataFrame(rows)


def _time_sensitivity(panels):
    rows = []
    for panel in ["full360", "restricted337"]:
        for dataset in DATASETS:
            group = panels[panel][panels[panel].dataset.eq(dataset)]
            for context in CONTEXTS:
                within = [_spearman(g[f"{context}_mean_q"], g.locality_percent)
                          for _, g in group.groupby("edit_count")]
                row = dict(panel=panel, dataset=dataset, geometry_context=context, n=len(group),
                           n_trajectories=group.trajectory_id.nunique(), n_T=group.edit_count.nunique(),
                           pooled_rho=_spearman(group[f"{context}_mean_q"], group.locality_percent),
                           within_T_median_rho=float(np.median(within)), within_T_min_rho=float(np.min(within)),
                           within_T_max_rho=float(np.max(within)), within_T_negative_count=int(np.sum(np.array(within) < 0)))
                for field, spec in [("partial_T_rho", "T_rank"), ("partial_DH_rho", "D_H_rank"),
                                    ("partial_DH_T_rho", "D_H_rank+T_rank")]:
                    row[field] = _residual_summary(group[f"{context}_mean_q"], group.locality_percent,
                                                  _design(group, context, spec))["rho"]
                rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap(frame):
    """One RNG stream; paired whole-trajectory draws shared by context/metric."""
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    rows, differences = [], []
    for dataset in DATASETS:
        ids = sorted(frame.loc[frame.dataset.eq(dataset), "trajectory_id"].unique())
        choices = rng.randint(0, len(ids), size=(BOOTSTRAP_DRAWS, len(ids)))
        group = frame[frame.dataset.eq(dataset)].set_index(["trajectory_id", "edit_count"])
        group = group.reindex(pd.MultiIndex.from_product([ids, TIMES]))
        outcome = group.locality_percent.to_numpy().reshape(len(ids), len(TIMES))
        y = rankdata(outcome[choices].reshape(BOOTSTRAP_DRAWS, -1), axis=1)
        y -= y.mean(axis=1, keepdims=True)
        for context in CONTEXTS:
            replicates, points = {}, {}
            for metric, suffix in [("q", "mean_q"), ("d", "mean_d"), ("D_H", "mean_abs_norm_deviation")]:
                values = group[f"{context}_{suffix}"].to_numpy().reshape(len(ids), len(TIMES))
                x = rankdata(values[choices].reshape(BOOTSTRAP_DRAWS, -1), axis=1)
                x -= x.mean(axis=1, keepdims=True)
                reps = np.sum(x * y, axis=1) / np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y, axis=1))
                if not np.isfinite(reps).all():
                    raise ValueError("Nonfinite trajectory-bootstrap correlation")
                point = _spearman(values.ravel(), outcome.ravel())
                low, high = np.percentile(reps, [2.5, 97.5])
                replicates[metric], points[metric] = reps, point
                rows.append(dict(panel="full360", dataset=dataset, context=context, metric=metric,
                                 n_checkpoints=len(group), n_condition_trajectories=len(ids),
                                 n_checkpoints_per_trajectory=len(TIMES), pooled_spearman=point,
                                 cluster_bootstrap_ci_low=low, cluster_bootstrap_ci_high=high,
                                 bootstrap_replicates=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED,
                                 resampling_unit="condition trajectory", interval="percentile 95%"))
            low, high = np.percentile(replicates["d"] - replicates["D_H"], [2.5, 97.5])
            differences.append(dict(dataset=dataset, context=context,
                                    rho_d_minus_rho_DH=points["d"] - points["D_H"], ci_lower=low, ci_upper=high))
    return pd.DataFrame(rows), pd.DataFrame(differences)


def _compare(actual, expected_path, keys, rounded=False):
    expected = pd.read_csv(expected_path, float_precision="round_trip")
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    if len(actual) != len(expected) or actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError(f"Invalid reference row coverage: {expected_path.name}")
    missing = set(expected.columns) - set(actual.columns)
    if missing:
        raise ValueError(f"Missing computed columns in {expected_path.name}: {sorted(missing)}")
    max_difference = 0.0
    for column in expected:
        a, b = actual[column], expected[column]
        if b.isna().all() and a.fillna("").eq("").all():
            # Empty reason strings are read back as NaN in an all-empty column.
            continue
        if pd.api.types.is_bool_dtype(b):
            pd.testing.assert_series_equal(a, b, check_names=False, check_dtype=False)
        elif pd.api.types.is_numeric_dtype(b):
            # The four-row paired-difference reference is published to five decimals.
            if rounded and column not in keys:
                a = a.round(5)
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10, equal_nan=True,
                                       err_msg=f"{expected_path.name}:{column}")
            finite = np.isfinite(a.to_numpy(dtype=float)) & np.isfinite(b.to_numpy(dtype=float))
            if finite.any():
                max_difference = max(max_difference, float(np.abs(a[finite] - b[finite]).max()))
        else:
            pd.testing.assert_series_equal(a.fillna("").astype(str), b.fillna("").astype(str), check_names=False)
    return dict(reference="expected/" + expected_path.name, rows=len(expected),
                compared_columns=list(expected.columns), maximum_absolute_difference=max_difference,
                comparison="five-decimal published values" if rounded else "full-precision numeric values", passed=True)


def run(data_dir, output_dir):
    """Recompute and validate RQ2; ``data_dir`` is the ``data/manuscript`` folder."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if output_dir.resolve() == data_dir.resolve() or data_dir.resolve() in output_dir.resolve().parents:
        raise ValueError("Write reproduced outputs outside the input data directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "checkpoints.csv"
    frame = pd.read_csv(source, float_precision="round_trip")
    panels, restricted, collapsed = _panels(frame)
    membership = frame[KEY + ["trajectory_id", "edit_count"]].copy()
    membership["full360"], membership["restricted337"], membership["excluded306"] = True, restricted, ~collapsed
    membership.to_csv(output_dir / "rq2_panel_membership.csv", index=False)

    boot, paired = _bootstrap(frame)
    products = [
        ("rq2_partial_rank_all.csv", _rank_controls(panels),
         ["panel", "model_scope", "dataset", "context", "predictor", "specification"], False),
        ("d_controlled_geometry_associations.csv", _composition(panels),
         ["panel", "dataset", "context", "predictor", "controls"], False),
        ("collapse_exclusion_associations.csv", _collapse(panels),
         ["panel", "dataset", "context", "metric", "controls"], False),
        ("trajectory_cluster_bootstrap.csv", boot, ["dataset", "context", "metric"], False),
        ("paired_correlation_differences.csv", paired, ["dataset", "context"], True),
        ("within_trajectory_correlations.csv", _within_trajectory(frame), ["context", "trajectory_id"], False),
        ("rq2_T_sensitivity.csv", _time_sensitivity(panels), ["panel", "dataset", "geometry_context"], False),
    ]
    checks, hashes = [], {"checkpoints.csv": hashlib.sha256(source.read_bytes()).hexdigest()}
    for name, actual, keys, rounded in products:
        reference = data_dir / "expected" / name
        checks.append(_compare(actual, reference, keys, rounded))
        hashes["expected/" + name] = hashlib.sha256(reference.read_bytes()).hexdigest()
        actual.to_csv(output_dir / name, index=False, float_format="%.17g")
    report = dict(
        passed=True, scope="Reaggregation of packaged measured checkpoints; no model inference",
        panels={name: dict(checkpoints=len(panel), trajectories=int(panel.trajectory_id.nunique()),
                           per_dataset=panel.groupby("dataset").size().to_dict()) for name, panel in panels.items()},
        collapse_exclusion="All nine checkpoints of Llama MEMIT Native/SPHERE/SADR in both datasets; descriptive post hoc sensitivity",
        reference_comparisons=checks, input_sha256=hashes,
        bootstrap=dict(draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED, recomputed_correlation_arrays=len(boot),
                       sampling_unit="20 whole trajectories with replacement within each dataset; all nine checkpoints retained",
                       same_draws_for_contexts_and_metrics=True, paired_difference_intervals=len(paired),
                       interval="2.5th and 97.5th percentiles",
                       scope="Full360 unadjusted q/d/D_H correlations and paired d-minus-D_H correlation differences",
                       saved_replicate_arrays_required=False),
        inference_scope="Adjusted coefficients and restricted337/excluded306 sensitivities are descriptive; no p values or confidence intervals for those coefficients",
        versions=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__))
    (output_dir / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "manuscript")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build" / "manuscript_rq2")
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
