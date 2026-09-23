"""Opt-in configuration for signed early-O-axis preservation in AlphaEdit.

This changes the parameterization of the shared inner delta; it adds no loss.
Disabled runs retain their historical configuration/fingerprint schema.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

CONDITION_LABELS = {"o0axis": "O0-axis", "hiddennorm_o0axis": "HN + O0-axis",
                    "o03axis": "O0–O3 axes", "hiddennorm_o03axis": "HN + O0–O3 axes"}
CONDITION_COMPONENTS = {key: frozenset({"rgr"}) if key.startswith("hiddennorm_") else frozenset()
                        for key in CONDITION_LABELS}
CONDITION_LAYERS = {"o0axis": [0], "hiddennorm_o0axis": [0],
                    "o03axis": [0, 1, 2, 3], "hiddennorm_o03axis": [0, 1, 2, 3]}
REFERENCE = "current_pre_edit"
SCOPE = "all_injected_contexts"
SVD_RTOL = 1.0e-6


def _parse_layers(value: str) -> list[int]:
    try:
        layers = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated initial O layer IDs") from error
    if layers not in ([0], [0, 1, 2, 3]):
        raise argparse.ArgumentTypeError("supported preserved O layers are 0 or 0,1,2,3")
    return layers


def add_o0_axis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--o0-axis-preservation-enabled", type=int, choices=[0, 1], default=0,
                        help="Hard-project the shared inner delta away from all supplied early O axes; no extra loss.")
    parser.add_argument("--o0-axis-preservation-layers", type=_parse_layers, default=[0],
                        help="Preserved raw output directions at every injected lookup: 0 (legacy) or 0,1,2,3")


def validate_o0_axis_args(args: argparse.Namespace) -> None:
    enabled = bool(getattr(args, "o0_axis_preservation_enabled", False))
    condition = getattr(args, "analysis_condition", None)
    if enabled != (condition in CONDITION_COMPONENTS):
        raise ValueError("Early-O preservation requires enabled=1 and an explicit o0axis/hiddennorm_o0axis/o03axis/hiddennorm_o03axis condition together")
    if not enabled:
        return
    if args.editing_method != "AlphaEdit":
        raise ValueError("O0-axis preservation is AlphaEdit-only")
    layers = list(getattr(args, "o0_axis_preservation_layers", [0]))
    if layers != CONDITION_LAYERS[condition]:
        raise ValueError(f"Condition {condition} requires preserved O layers {CONDITION_LAYERS[condition]}, received {layers}")
    for key in ("early_attention_preservation_enabled", "context_multikey_enabled", "key_gaussian_noise_enabled", "tangent_layer_allocation_enabled"):
        if bool(getattr(args, key, False)):
            raise ValueError(f"O0-axis pilot excludes additional extension {key}")
    if bool(getattr(args, "residual_gain_regularization", False)) and not CONDITION_COMPONENTS[condition]:
        raise ValueError("Use a hiddennorm_ condition for the HN arm")
    if getattr(args, "inner_margin_schedule", "legacy") != "legacy":
        raise ValueError("O0-axis pilot retains the historical inner optimization schedule")
    if getattr(args, "resume_layer_checkpoint", None):
        raise ValueError("O0-axis pilot starts at edit1; parameter-only resume omits AlphaEdit cumulative state")


def configure_o0_axis_hparams(args: argparse.Namespace, hparams: Any, *, output_dir: Path) -> None:
    validate_o0_axis_args(args)
    enabled = bool(getattr(args, "o0_axis_preservation_enabled", False))
    if not enabled:
        if hasattr(hparams, "o0_axis_preservation_enabled"):
            hparams.o0_axis_preservation_enabled = False
        return
    if list(hparams.layers) != [4, 5, 6, 7, 8] or hparams.fact_token != "subject_last":
        raise ValueError("O0-axis pilot requires AlphaEdit L4-L8 subject_last editing")
    hn = "rgr" in CONDITION_COMPONENTS[args.analysis_condition]
    if bool(hparams.residual_gain_regularization) != hn:
        raise ValueError("O0-axis arm and actual HN enablement disagree")
    if hn:
        expected = dict(residual_gain_lambda=1., residual_gain_subject_layers=[8], residual_gain_margin=0.,
                        residual_gain_loss_type="positive_squared", residual_gain_objective="rho_minus_one",
                        residual_gain_efficacy_threshold=0., residual_gain_token_scope="subject_last",
                        residual_gain_prompt_layers=[], residual_gain_select_best=False, residual_gain_early_stop_mode="base")
        for key, value in expected.items():
            if getattr(hparams, key, None) != value:
                raise ValueError(f"HN+O0 requires historical {key}={value!r}")
    hparams.o0_axis_preservation_enabled = True
    hparams.o0_axis_preservation_layers = list(CONDITION_LAYERS[args.analysis_condition])
    hparams.o0_axis_preservation_reference = REFERENCE
    hparams.o0_axis_preservation_scope = SCOPE
    hparams.o0_axis_preservation_svd_rtol = SVD_RTOL
    hparams.o0_axis_preservation_log_path = str(output_dir / "o0_axis_preservation.jsonl")


def o0_axis_config(hparams: Any) -> dict:
    if not bool(getattr(hparams, "o0_axis_preservation_enabled", False)):
        return {}
    layers = list(getattr(hparams, "o0_axis_preservation_layers", None) or [0])
    payload = dict(o0_axis_preservation_enabled=True, o0_axis_preservation_reference=REFERENCE,
                o0_axis_preservation_scope=SCOPE, o0_axis_preservation_svd_rtol=SVD_RTOL,
                o0_axis_preservation_log_path=str(hparams.o0_axis_preservation_log_path),
                o0_axis_preservation_target_layer=8, o0_axis_preservation_axis_layer=0,
                o0_axis_preservation_parameterization="delta_eff=(I-UU^T)delta_raw; U spans unit O0 at every injected lookup",
                o0_axis_preservation_preserved_quantity="signed raw projection of each context output onto its current pre-edit O0 axis",
                o0_axis_preservation_includes_kl_contexts=True, o0_axis_preservation_additional_loss=False,
                o0_axis_preservation_formulation_version=1)
    if layers != [0]:
        if layers != [0, 1, 2, 3]:
            raise ValueError("Unexpected preserved O layers")
        payload.pop("o0_axis_preservation_axis_layer")
        payload.update(o0_axis_preservation_layers=layers, o0_axis_preservation_axis_layers=layers,
                       o0_axis_preservation_parameterization="delta_eff=(I-UU^T)delta_raw; U spans unit O0,O1,O2,O3 at every injected lookup",
                       o0_axis_preservation_preserved_quantity="signed raw projection of each context output onto its current pre-edit O0,O1,O2,O3 directions",
                       o0_axis_preservation_formulation_version=2)
    return payload


def validate_o0_axis_source(config: Mapping[str, Any], condition: str) -> None:
    """Strict identity guard shared by endpoint collection and verification."""
    if condition not in CONDITION_COMPONENTS or config.get("analysis_condition") != condition:
        raise ValueError("O0-axis source condition mismatch")
    if list(config.get("o0_axis_preservation_layers", [0])) != CONDITION_LAYERS[condition]:
        raise ValueError("Source condition and preserved O layers disagree")
    required = dict(o0_axis_preservation_enabled=True, o0_axis_preservation_reference=REFERENCE,
                    o0_axis_preservation_scope=SCOPE, o0_axis_preservation_svd_rtol=SVD_RTOL,
                    o0_axis_preservation_includes_kl_contexts=True, o0_axis_preservation_additional_loss=False,
                    residual_gain_regularization="rgr" in CONDITION_COMPONENTS[condition])
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"O0-axis source {key}: expected {value!r}, found {config.get(key)!r}")
    for key in ("sphere_enabled", "sadr_regularization", "encore_enabled", "nas_enabled", "early_attention_preservation_enabled"):
        if config.get(key, False):
            raise ValueError(f"Unexpected extension in O0-axis source: {key}")
    if "rgr" in CONDITION_COMPONENTS[condition]:
        for key, value in dict(residual_gain_lambda=1., residual_gain_subject_layers=[8], residual_gain_margin=0.,
                               residual_gain_loss_type="positive_squared", residual_gain_objective="rho_minus_one",
                               residual_gain_efficacy_threshold=0.).items():
            if config.get(key) != value:
                raise ValueError(f"HN+O0 source changed historical {key}")
