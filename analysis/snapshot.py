"""Shared filenames and sampling constraints for a saved numeric snapshot."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STATE = ["model", "dataset", "editor", "method", "order_id"]
CHECKPOINT = STATE + ["edit_count"]
METRICS = ["mean_q", "mean_kappa", "mean_abs_norm_deviation"]
THRESHOLD_COLUMNS = [context + "_" + metric
                     for context in ["locality", "rewrite"] for metric in METRICS]
LEGACY_FILES = {
    "rq1": "rq1_per_edit_40000.csv",
    "rq2": "rq2_checkpoints_360.csv",
    "rq3_doses": "rq3_orthogonal_states_doses_200.csv",
    "rq3_same_norm": "rq3_same_norm_conditions_480.csv",
    "performance": "performance_1k_40.csv",
    "figure4": "figure4_rewrite_endpoints_34.csv",
    "figure5": "figure5_prompt_overlay_2022.csv",
    "rq2_mask": "rq2_threshold_mask_360.csv",
    "rq3_differences": "rq3_same_norm_state_differences_68.csv",
}


def metadata():
    path = DATA / "analysis_snapshot.json"
    return json.loads(path.read_text()) if path.is_file() else {"files": LEGACY_FILES}


def filename(role):
    return metadata()["files"].get(role, LEGACY_FILES.get(role))


def validate_checkpoints(frame):
    if frame.duplicated(CHECKPOINT).any():
        raise ValueError("Duplicate checkpoint identifiers")
    info = metadata()
    if info.get("canonical_orders_only") and not frame.order_id.eq("canonical").all():
        raise ValueError("Additional edit orders are not part of this snapshot")
    expected = info.get("rq2_available_checkpoints")
    if expected is not None and len(frame) != expected:
        raise ValueError("Checkpoint count differs from the snapshot metadata")
    steps = info.get("edit_counts")
    if steps is not None:
        for key, group in frame.groupby(STATE):
            if sorted(group.edit_count.tolist()) != steps:
                raise ValueError(f"Incomplete checkpoint sequence: {key}")
