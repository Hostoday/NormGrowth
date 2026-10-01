"""Measurement filenames and sampling constraints for the experimental design."""
STATE = ["model", "dataset", "editor", "method", "order_id"]
CHECKPOINT = STATE + ["edit_count"]
METRICS = ["mean_q", "mean_kappa", "mean_abs_norm_deviation"]
THRESHOLD_COLUMNS = [context + "_" + metric
                     for context in ["locality", "rewrite"] for metric in METRICS]
EDIT_COUNTS = [50, 100, 150, 200, 250, 300, 500, 750, 1000]
FILES = {
    "rq1": "rq1_per_edit.csv",
    "rq2": "checkpoints.csv",
    "rq3_doses": "intervention_doses.csv",
    "rq3_same_norm": "same_norm_conditions.csv",
    "performance": "endpoint_tf_1000.csv",
    "figure4": "rewrite_endpoints.csv",
    "figure5": "checkpoints.csv",
}


def filename(role):
    return FILES[role]


def validate_checkpoints(frame):
    if frame.duplicated(CHECKPOINT).any():
        raise ValueError("Duplicate checkpoint identifiers")
    if not frame.order_id.eq("canonical").all():
        raise ValueError("This analysis uses the canonical edit order")
    if len(frame) != 40 * len(EDIT_COUNTS):
        raise ValueError("Expected 40 trajectories with nine checkpoints each")
    for key, group in frame.groupby(STATE):
        if sorted(group.edit_count.tolist()) != EDIT_COUNTS:
            raise ValueError(f"Incomplete checkpoint sequence: {key}")
