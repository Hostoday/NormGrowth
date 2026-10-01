"""Stage a portable canonical snapshot from a completed paper figure bundle.

Run with --source-dir and --output-dir. This never replaces data/ automatically.
Source numeric CSV cells retain their original text, except documented aliases.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from .snapshot import CHECKPOINT, DATA, LEGACY_FILES, STATE, THRESHOLD_COLUMNS


def csv_read(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames), list(reader)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    if output == DATA or DATA in output.parents:
        parser.error("Stage outside data/; review before installing the snapshot")
    if not (source / "data/checkpoints.csv").is_file():
        parser.error("Source bundle lacks data/checkpoints.csv")
    entries = []

    def export(source_name, target_name, columns=None, transform=None):
        path = source / source_name
        fields, rows = csv_read(path)
        original_rows = len(rows)
        if transform:
            rows = transform(rows)
        columns = columns or fields
        destination = output / target_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        entries.append(dict(source_id=source.name + "/" + source_name,
            source_sha256=digest(path), package_file="data/" + target_name,
            package_sha256=digest(destination), rows=len(rows), selected_columns=columns,
            removed_columns=[column for column in fields if column not in columns],
            row_filter_applied=len(rows) != original_rows,
            normalization_applied=bool(transform)))
        return rows

    # Reuse the documented portable fields of the original release where possible.
    old = json.loads((DATA / "provenance/data_sources.json").read_text())
    old_columns = {item["package_file"].removeprefix("data/"): item["selected_columns"]
                   for item in old["files"]}

    def columns_for(role):
        current = DATA / "analysis_snapshot.json"
        mapping = json.loads(current.read_text())["files"] if current.is_file() else LEGACY_FILES
        return old_columns[mapping[role]]

    files = dict(LEGACY_FILES, rq2="rq2_checkpoints.csv", rq2_mask="rq2_threshold_mask.csv",
                 figure4="figure4_rewrite_endpoints.csv", figure5="figure5_prompt_overlay.csv",
                 rq2_drift="rq2_drift_correlations.csv", locality_endpoints="locality_endpoints.csv",
                 endpoint_summary="endpoint_norm_decomposition_summary.csv",
                 rq3_differences="rq3_same_norm_state_differences.csv")
    current_info = json.loads((DATA / "analysis_snapshot.json").read_text()) if (DATA / "analysis_snapshot.json").is_file() else {}
    previous_differences = current_info.get("files", {}).get("rq3_differences", LEGACY_FILES["rq3_differences"])
    copies = [
        ("rq1_per_edit.csv", files["rq1"], columns_for("rq1")),
        ("rq1_summary.csv", "expected/rq1_condition_summary_40.csv", old_columns["expected/rq1_condition_summary_40.csv"]),
        ("intervention_doses.csv", files["rq3_doses"], columns_for("rq3_doses")),
        ("orthogonal_dose_summary.csv", "expected/rq3_orthogonal_dose_summary.csv", old_columns["expected/rq3_orthogonal_dose_summary.csv"]),
        ("same_norm_conditions.csv", files["rq3_same_norm"], columns_for("rq3_same_norm")),
        ("same_norm_state_differences.csv", "expected/" + files["rq3_differences"], old_columns["expected/" + previous_differences]),
        ("same_norm_overall_means.csv", "expected/rq3_same_norm_overall_means.csv", None),
        ("performance_1k.csv", files["performance"], columns_for("performance")),
        ("rewrite_endpoints.csv", files["figure4"], columns_for("figure4")),
    ]
    for source_name, target, columns in copies:
        export("data/" + source_name, target, columns)

    _, endpoint_rows = csv_read(source / "data/endpoint_norm_decomposition.csv")
    locality_keys = {tuple(row[key] for key in CHECKPOINT) for row in endpoint_rows if row["context"] == "locality"}
    endpoint_columns = CHECKPOINT + ["mean_p", "mean_q", "mean_kappa", "mean_p_squared",
                                    "mean_q_squared", "mean_kappa_squared"]
    export("data/locality_endpoints.csv", files["locality_endpoints"], endpoint_columns,
           lambda rows: [row for row in rows if tuple(row[key] for key in CHECKPOINT) in locality_keys])
    export("data/endpoint_norm_decomposition_summary.csv", "expected/" + files["endpoint_summary"])

    checkpoint_columns = list(columns_for("rq2"))
    available_columns, _ = csv_read(source / "data/checkpoints.csv")
    for context in ["locality", "rewrite"]:
        for name in ["mean_d", "mean_p_squared", "mean_q_squared", "mean_kappa_squared"]:
            column = context + "_" + name
            if (column in available_columns or column == "locality_mean_d") and column not in checkpoint_columns:
                checkpoint_columns.append(column)

    def normalize_checkpoints(rows):
        for row in rows:
            if row["order_id"] != "canonical":
                raise ValueError("Source contains an additional edit order")
            row["locality_mean_d"] = row.get("locality_mean_d") or row.get("mean_d", "")
        return rows

    checkpoints = export("data/checkpoints.csv", files["rq2"], checkpoint_columns, normalize_checkpoints)
    mask_columns = CHECKPOINT + ["excluded_above3", "kept_after_threshold3"]
    export("data/threshold_mask.csv", "expected/" + files["rq2_mask"], mask_columns)

    def dataset_correlations(rows):
        return [dict(row, dataset=row["group"]) for row in rows if row["scope"] == "dataset"]

    export("data/correlations.csv", "expected/rq2_correlations.csv",
           old_columns["expected/rq2_correlations.csv"], dataset_correlations)
    export("data/drift_correlations.csv", "expected/" + files["rq2_drift"],
           ["dataset", "geometry_context", "n_checkpoints", "rho_q_LOC", "rho_d_LOC", "rho_DH_LOC", "rho_q_d"])
    plot_columns = CHECKPOINT + ["row_id", "metric", "prompt_context", "geometry_value",
                               "locality_percent", "source_column"]

    def normalize_plot(rows):
        for row in rows:
            row["row_id"] = "|".join(row[column] for column in CHECKPOINT)
            row["source_column"] = row["prompt_context"] + "_" + row["metric"]
        return rows

    export("figures/fig5_geometry_locality_plotdata.csv", files["figure5"], plot_columns, normalize_plot)

    keys = [tuple(row[column] for column in CHECKPOINT) for row in checkpoints]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate checkpoint identifiers in source")
    kept = [row for row in checkpoints if all(float(row[column]) <= 3 for column in THRESHOLD_COLUMNS)]
    trajectories = {tuple(row[column] for column in STATE) for row in checkpoints}
    from itertools import product
    expected = set(product(["Llama", "GPT-2 XL"], ["zsRE", "CounterFact"],
                           ["MEMIT", "AlphaEdit"], ["Native", "NAS", "ENCORE", "SPHERE", "SADR"], ["canonical"]))
    moment_columns = [context + "_" + name for context in ["locality", "rewrite"]
                      for name in ["mean_p_squared", "mean_q_squared", "mean_kappa_squared"]]
    if any(not row.get(column) for row in checkpoints for column in moment_columns):
        raise ValueError("Source lacks second moments required for norm decomposition")
    _, same_norm_rows = csv_read(source / "data/same_norm_conditions.csv")
    eligible_by_dose = {}
    for row in same_norm_rows:
        dose = str(float(row["dose_fraction"]))
        eligible_by_dose.setdefault(dose, set())
        if float(row["n_cases"]) > 0:
            eligible_by_dose[dose].add(tuple(row[key] for key in CHECKPOINT))
    info = dict(schema_version=1, source_bundle=source.name, canonical_orders_only=True,
                complete_second_moments=True,
                design_conditions=40, checkpoints_per_trajectory=9,
                edit_counts=[50,100,150,200,250,300,500,750,1000],
                rq2_available_checkpoints=len(checkpoints), rq2_retained_checkpoints=len(kept),
                rq2_available_trajectories=len(trajectories),
                rq3_same_norm_eligible_states_by_dose={dose: len(states) for dose, states in eligible_by_dose.items()},
                missing_trajectories=[dict(zip(STATE,key)) for key in sorted(expected-trajectories)],
                files=files)
    (output / "analysis_snapshot.json").write_text(json.dumps(info, indent=2)+"\n")
    (output / "provenance").mkdir(exist_ok=True)
    (output / "provenance/data_sources.json").write_text(json.dumps(dict(schema_version=2,
        description="Canonical-order numeric snapshot. Numeric cells preserved; locality mean_d alias normalized. See source_bundle for the experimental revision.",
        files=entries), indent=2)+"\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
