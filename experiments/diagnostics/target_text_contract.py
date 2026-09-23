"""Shared edit-target normalization used before training and evaluation.

The cumulative Generate evaluator removes tokenizer training terminators before
scoring a target.  A target containing only one of those tokens is therefore
not a factual target and must be rejected before cohort selection or editing.
"""

from __future__ import annotations

from typing import Any, Mapping


TRAINING_TERMINATORS = ("<|eot_id|>", "<|end_of_text|>", "<|endoftext|>")


def normalize_target_value(value: Any) -> str:
    """Normalize the target container formats used by EasyEdit datasets."""

    if isinstance(value, Mapping):
        for key in ("str", "text", "value"):
            if key in value and value[key] is not None:
                return str(value[key])
        return ""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


def strip_training_terminators(value: Any, *, strip_whitespace: bool = True) -> str:
    """Return the semantic target visible to the Generate evaluator."""

    text = normalize_target_value(value)
    for token in TRAINING_TERMINATORS:
        text = text.replace(token, "")
    return text.strip() if strip_whitespace else text


def raw_edit_target(record: Mapping[str, Any]) -> Any:
    """Extract the edit target using the same nested/flat precedence as training."""

    rewrite = record.get("requested_rewrite")
    if isinstance(rewrite, Mapping):
        return rewrite.get("target_new")
    if record.get("target_new") is not None:
        return record.get("target_new")
    return record.get("alt")


def semantic_edit_target(record: Mapping[str, Any]) -> str:
    """Extract a target and remove terminators that carry no factual content."""

    return strip_training_terminators(raw_edit_target(record))
