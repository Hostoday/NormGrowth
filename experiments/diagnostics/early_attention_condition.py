"""Shared, opt-in configuration for the AlphaEdit HN+attn experiment."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict


def _layers(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated layer IDs") from error
    if result != sorted(set(result)) or not result:
        raise argparse.ArgumentTypeError("layers must be nonempty, sorted and unique")
    return result


def add_early_attention_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--early-attention-preservation-enabled", type=int,
                        choices=[0, 1], default=0)
    parser.add_argument("--early-attention-preservation-lambda", type=float, default=1.0)
    parser.add_argument("--early-attention-preservation-layers", type=_layers,
                        default=list(range(5)))
    parser.add_argument("--early-attention-preservation-reference",
                        choices=["current_pre_edit"], default="current_pre_edit")


def configure_early_attention_hparams(
    args: argparse.Namespace, hparams: Any, *, method: str, output_dir: Path
) -> None:
    enabled = bool(getattr(args, "early_attention_preservation_enabled", False))
    if not enabled:
        if hasattr(hparams, "early_attention_preservation_enabled"):
            hparams.early_attention_preservation_enabled = False
        return
    if method != "AlphaEdit":
        raise ValueError("early-attention preservation currently requires AlphaEdit")
    weight = float(args.early_attention_preservation_lambda)
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("early-attention preservation lambda must be finite and positive")
    layers = list(args.early_attention_preservation_layers)
    if layers != list(range(5)):
        raise ValueError("this HN+attn experiment requires early attention layers 0,1,2,3,4")
    if list(hparams.layers) != [4, 5, 6, 7, 8] or hparams.fact_token != "subject_last":
        raise ValueError("HN+attn requires L4-L8 editing at subject_last")
    if not (
        bool(getattr(hparams, "residual_gain_regularization", False))
        and getattr(hparams, "residual_gain_objective", None) == "rho_minus_one"
        and getattr(hparams, "residual_gain_loss_type", None) == "positive_squared"
        and list(getattr(hparams, "residual_gain_subject_layers", []) or []) == [8]
        and float(getattr(hparams, "residual_gain_lambda", 0.0)) == 1.0
        and float(getattr(hparams, "residual_gain_margin", 0.0)) == 0.0
        and float(getattr(hparams, "residual_gain_efficacy_threshold", 0.0)) == 0.0
    ):
        raise ValueError("HN+attn requires the unchanged HN lambda=1, L8, rho_minus_one squared-hinge control")
    hparams.early_attention_preservation_enabled = True
    hparams.early_attention_preservation_lambda = weight
    hparams.early_attention_preservation_layers = layers
    hparams.early_attention_preservation_reference = str(args.early_attention_preservation_reference)
    hparams.early_attention_preservation_log_path = str(output_dir / "early_attention_preservation.jsonl")


def early_attention_config(hparams: Any) -> Dict[str, Any]:
    """Leave historical disabled fingerprints unchanged."""
    if not bool(getattr(hparams, "early_attention_preservation_enabled", False)):
        return {}
    return {
        "early_attention_preservation_enabled": True,
        "early_attention_preservation_lambda": float(hparams.early_attention_preservation_lambda),
        "early_attention_preservation_layers": list(hparams.early_attention_preservation_layers),
        "early_attention_preservation_reference": hparams.early_attention_preservation_reference,
        "early_attention_preservation_target_layer": 8,
        "early_attention_preservation_token_scope": "subject_last",
        "early_attention_preservation_context_scope": "canonical_rewrite_contexts_excluding_kl",
        "early_attention_preservation_expected_context_count": 6,
        "early_attention_preservation_svd_rtol": 1.0e-5,
        "early_attention_preservation_objective": "mean_context_squared_projected_normalized_output_difference",
        "early_attention_preservation_formulation_version": 1,
    }


def early_attention_cli(args: argparse.Namespace) -> list[str]:
    if not bool(getattr(args, "early_attention_preservation_enabled", False)):
        return []
    return [
        "--early-attention-preservation-enabled", "1",
        "--early-attention-preservation-lambda", str(args.early_attention_preservation_lambda),
        "--early-attention-preservation-layers", ",".join(map(str, args.early_attention_preservation_layers)),
        "--early-attention-preservation-reference", args.early_attention_preservation_reference,
    ]
