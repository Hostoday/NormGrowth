"""Validate and install a complete canonical paper bundle, preserving a backup.

This is a CPU-only release step. It requires all 40 canonical trajectories and
checks the actual threshold mask; the retained count is never assumed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .snapshot import DATA, LEGACY_FILES, ROOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def managed_destinations():
    names = set(LEGACY_FILES.values()) - {LEGACY_FILES["rq2_mask"], LEGACY_FILES["rq3_differences"]}
    names.update({"analysis_snapshot.json", "rq2_checkpoints.csv", "figure4_rewrite_endpoints.csv",
                  "figure5_prompt_overlay.csv", "locality_endpoints.csv"})
    names.update("expected/" + name for name in ["rq1_condition_summary_40.csv", "rq2_correlations.csv",
        "rq2_threshold_mask_360.csv", "rq2_threshold_mask.csv", "rq2_drift_correlations.csv",
        "rq3_orthogonal_dose_summary.csv", "rq3_same_norm_overall_means.csv",
        "rq3_same_norm_state_differences_68.csv", "rq3_same_norm_state_differences.csv",
        "endpoint_norm_decomposition_summary.csv"])
    names.update("provenance/" + name for name in ["data_sources.json", "cpu_reproduction_validation.json",
        "figure_reproduction_validation.json", "validation_summary.json", "release_manifest.json", "code_sources.json"])
    return sorted(DATA / name for name in names)


def destination_hashes():
    return {str(path.relative_to(ROOT)): sha(path) if path.is_file() else None
            for path in managed_destinations()}


def verify_destination_state(path):
    if not path.is_file():
        raise ValueError("Missing prepared destination state; run --prepare-destination-state first")
    recorded = json.loads(path.read_text())["files"]
    current = destination_hashes()
    changed = sorted(name for name in current.keys() | recorded.keys() if current.get(name) != recorded.get(name))
    if changed:
        raise ValueError("Destination changed after preparation; automatic install refused: " + ", ".join(changed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("build/canonical_release"))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare-destination-state", action="store_true")
    action.add_argument("--install", action="store_true")
    action.add_argument("--verify-only", action="store_true")
    parser.add_argument("--destination-state", type=Path)
    parser.add_argument("--code-provenance", type=Path,
                        help="Reviewed code_sources.json to include in the same guarded installation")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    if work == DATA or DATA in work.parents or work == ROOT:
        parser.error("Work directory must be outside the included source data")
    work.mkdir(parents=True, exist_ok=True)
    state_path = args.destination_state.resolve() if args.destination_state else work / "destination_state.json"
    if args.prepare_destination_state:
        if state_path.exists():
            parser.error("Destination state already exists; review changes before explicitly replacing it")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(dict(prepared_at_utc=datetime.now(timezone.utc).isoformat(),
            files=destination_hashes()), indent=2)+"\n")
        print("Prepared destination hashes; no source data modified")
        return
    if not args.source_dir:
        parser.error("--source-dir is required for verification or installation")
    source = args.source_dir.resolve()
    if args.install:
        verify_destination_state(state_path)
    # Staging/backup trees retain every prior file, including user changes.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    current_report = work / ("install_validation.json" if args.install else "verification_validation.json")
    if current_report.exists():
        current_report.rename(work / ("previous_install_validation_" + stamp + ".json"))
    stage = work / ("staging_" + stamp)
    candidate, results, figures = [stage / name for name in ["data", "results", "figures"]]
    backup = work / ("backup_" + stamp)
    environment = dict(os.environ, MPLCONFIGDIR=str(work / "matplotlib"))

    def run(*arguments):
        subprocess.run([sys.executable, "-m", *arguments], cwd=ROOT,
                       env=environment, check=True)

    run("analysis.import_snapshot", "--source-dir", str(source), "--output-dir", str(candidate))
    info = json.loads((candidate / "analysis_snapshot.json").read_text())
    if (not info["canonical_orders_only"] or info["missing_trajectories"]
            or info["rq2_available_trajectories"] != info["design_conditions"]
            or info["rq2_available_checkpoints"] != info["design_conditions"]*info["checkpoints_per_trajectory"]):
        raise ValueError("Final installation requires all 40 canonical trajectories; source data remain untouched")
    run("analysis.reproduce", "--data-dir", str(candidate), "--output-dir", str(results))
    run("analysis.plot_core_figures", "--data-dir", str(candidate), "--output-dir", str(figures))
    numerical = json.loads((results / "validation.json").read_text())
    graphical = json.loads((figures / "figure_validation.json").read_text())
    if not numerical["passed"] or not graphical["passed"]:
        raise ValueError("Staged validation did not pass")
    if numerical["rq2_counts"]["retained"] != info["rq2_retained_checkpoints"]:
        raise ValueError("Retained panel differs from snapshot metadata")
    if graphical["figures"]["fig5_geometry_locality"]["input_checkpoint_count"] != info["rq2_retained_checkpoints"]:
        raise ValueError("Figure 5 differs from the correlation panel")
    for stem in ["fig4_pq_plane", "fig5_geometry_locality"]:
        for extension in ["pdf", "png", "svg"]:
            path = figures / (stem + "." + extension)
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError("Missing rendered artifact: " + path.name)
    if args.code_provenance:
        code_sources = json.loads(args.code_provenance.read_text())
        for entry in code_sources["files"]:
            if sha(ROOT / entry["release_path"]) != entry["release_sha256"]:
                raise ValueError("Code provenance differs from release file: " + entry["release_path"])
        for entry in code_sources.get("generated_files", []):
            if sha(ROOT / entry["release_path"]) != entry["sha256"]:
                raise ValueError("Generated-code provenance differs from release file: " + entry["release_path"])
        shutil.copy2(args.code_provenance, candidate / "provenance/code_sources.json")
    if not args.install:
        report = dict(passed=True, installed=False, complete_canonical_design=True,
            available_checkpoints=info["rq2_available_checkpoints"],
            retained_checkpoints=info["rq2_retained_checkpoints"], staging_directory=stage.name)
        current_report.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))
        return

    # Reports are part of the installed snapshot, not claims about historical runs.
    provenance = candidate / "provenance"
    shutil.copy2(results / "validation.json", provenance / "cpu_reproduction_validation.json")
    shutil.copy2(figures / "figure_validation.json", provenance / "figure_reproduction_validation.json")
    summary = dict(schema=3, source_bundle=source.name, canonical_orders_only=True,
                   available_checkpoints=info["rq2_available_checkpoints"],
                   retained_checkpoints=info["rq2_retained_checkpoints"],
                   saved_numeric_reference_tables=len(numerical["reference_comparisons"]),
                   model_inference_performed=False, fresh_dependency_install=False,
                   remote_ci_executed=False, current_snapshot_cpu_validation_passed=True,
                   historical_model_source_smoke="model_code_smoke.json")
    summary["rq3_counts"] = numerical["rq3_counts"]
    (provenance / "validation_summary.json").write_text(json.dumps(summary, indent=2)+"\n")

    writes = [(path, DATA / path.relative_to(candidate)) for path in candidate.rglob("*") if path.is_file()]
    obsolete = [DATA / LEGACY_FILES[role] for role in ["rq2", "figure4", "figure5"]]
    obsolete.append(DATA / "expected" / LEGACY_FILES["rq2_mask"])
    obsolete.append(DATA / "expected" / LEGACY_FILES["rq3_differences"])
    obsolete = [path for path in obsolete if path.is_file() and path not in {dest for _,dest in writes}]
    manifest = DATA / "provenance/release_manifest.json"
    affected = {dest for _,dest in writes} | set(obsolete) | {manifest}
    if not affected.issubset(set(managed_destinations())):
        raise ValueError("Unsealed output destination; review installer destination inventory")
    # Repeat immediately before mutation to protect edits made during validation.
    verify_destination_state(state_path)
    preexisting = {path for path in affected if path.exists()}
    for path in preexisting:
        saved = backup / path.relative_to(ROOT)
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, saved)
    try:
        for source_file, destination in writes:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
            if sha(source_file) != sha(destination):
                raise ValueError("Installed file differs from validated staging")
        for path in obsolete:
            path.unlink()
        run("analysis.check_release", "--write-manifest")
        run("analysis.check_release")
    except BaseException:
        for path in affected:
            if path in preexisting:
                shutil.copy2(backup / path.relative_to(ROOT), path)
            elif path.exists():
                path.unlink()
        raise
    report = dict(passed=True, installed=True, complete_canonical_design=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  available_checkpoints=info["rq2_available_checkpoints"],
                  retained_checkpoints=info["rq2_retained_checkpoints"],
                  excluded_checkpoints=numerical["rq2_counts"]["excluded"],
                  reference_tables_checked=len(numerical["reference_comparisons"]),
                  rq3_counts=numerical["rq3_counts"],
                  source_bundle=source.name, installed_snapshot_sha256=sha(DATA / "analysis_snapshot.json"),
                  input_manifest_sha256=sha(DATA / "provenance/data_sources.json"),
                  release_manifest_sha256=sha(manifest),
                  staging_directory=stage.name, backup_directory=backup.name,
                  artifact_sha256={path.name:sha(path) for path in figures.iterdir() if path.is_file()},
                  original_experiment_sources_modified=False)
    current_report.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
