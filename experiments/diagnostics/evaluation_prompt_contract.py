"""Fail-closed prompt materialization for generation-based edit evaluation.

Training implementations may internally represent an edit prompt as a Python
format template (for example, ``"{} is in"``).  Free-generation evaluation
must instead receive the prompt with the request subject already inserted.
This module keeps that conversion explicit and opt-in so existing evaluation
protocols that intentionally consume prompts verbatim do not change.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple


SUBJECT_PLACEHOLDER = "{}"
MATERIALIZATION_MODE = "single_literal_empty_braces_to_subject"


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else [value]


def _materialize_one(
    value: Any,
    *,
    subject: str,
    context: str,
) -> Tuple[str, bool, int]:
    text = str(value)
    placeholder_count = text.count(SUBJECT_PLACEHOLDER)
    if placeholder_count == 0:
        return text, False, 0
    if placeholder_count != 1:
        raise ValueError(
            f"{context} has {placeholder_count} literal {SUBJECT_PLACEHOLDER!r} "
            "subject placeholders; exactly one is required"
        )
    if not subject.strip():
        raise ValueError(f"{context} has a subject placeholder but no usable subject")

    materialized = text.replace(SUBJECT_PLACEHOLDER, subject, 1)
    if SUBJECT_PLACEHOLDER in materialized:
        raise AssertionError(f"{context} still has an unresolved subject placeholder")
    return materialized, True, placeholder_count


def prepare_generation_prompts(
    record: Mapping[str, Any],
    index: int,
    *,
    materialize_subject_placeholders: bool,
) -> Tuple[str, List[str], Dict[str, Any]]:
    """Return efficacy/rephrase prompts plus auditable per-request statistics.

    With materialization disabled, prompt strings are returned exactly as the
    evaluator historically consumed them.  With it enabled, every request is
    required to have a non-empty subject and an efficacy prompt containing that
    subject after at most one literal ``{}`` replacement.  Rephrase prompts are
    materialized when needed but are not required to repeat the subject because
    valid paraphrases may use anaphora.
    """

    raw_prompt = record.get("prompt")
    if raw_prompt is None:
        raise ValueError(f"request index {index} lacks prompt")
    prompt = str(raw_prompt)

    raw_rephrases = record.get("rephrase_prompt")
    rephrases = (
        []
        if raw_rephrases is None
        else [str(value) for value in _as_list(raw_rephrases) if str(value)]
    )
    subject_value = record.get("subject")
    subject = "" if subject_value is None else str(subject_value)

    efficacy_placeholders_before = prompt.count(SUBJECT_PLACEHOLDER)
    rephrase_placeholders_before = sum(
        text.count(SUBJECT_PLACEHOLDER) for text in rephrases
    )
    efficacy_materialized = False
    rephrase_materialized = 0

    if materialize_subject_placeholders:
        if not subject.strip():
            raise ValueError(
                f"request index {index} lacks a usable subject required by "
                "--materialize-subject-placeholders"
            )
        prompt, efficacy_materialized, _ = _materialize_one(
            prompt,
            subject=subject,
            context=f"request index {index} efficacy prompt",
        )
        transformed_rephrases = []
        for rephrase_index, rephrase in enumerate(rephrases):
            transformed, changed, _ = _materialize_one(
                rephrase,
                subject=subject,
                context=(
                    f"request index {index} rephrase prompt {rephrase_index}"
                ),
            )
            transformed_rephrases.append(transformed)
            rephrase_materialized += int(changed)
        rephrases = transformed_rephrases

        if SUBJECT_PLACEHOLDER in prompt or any(
            SUBJECT_PLACEHOLDER in rephrase for rephrase in rephrases
        ):
            raise AssertionError(
                f"request index {index} retains an unresolved subject placeholder"
            )
        if subject not in prompt:
            raise ValueError(
                f"request index {index} efficacy prompt does not contain its "
                f"subject after materialization: subject={subject!r}, prompt={prompt!r}"
            )

    unresolved_after = prompt.count(SUBJECT_PLACEHOLDER) + sum(
        text.count(SUBJECT_PLACEHOLDER) for text in rephrases
    )
    return prompt, rephrases, {
        "efficacy_prompt_count": 1,
        "rephrase_prompt_count": len(rephrases),
        "efficacy_placeholders_before": efficacy_placeholders_before,
        "rephrase_placeholders_before": rephrase_placeholders_before,
        "materialized_efficacy_prompts": int(efficacy_materialized),
        "materialized_rephrase_prompts": rephrase_materialized,
        "unresolved_placeholders_after": unresolved_after,
        "efficacy_contains_subject_after": int(bool(subject) and subject in prompt),
    }


def summarize_prompt_contract(
    case_stats: List[Mapping[str, Any]],
    *,
    materialize_subject_placeholders: bool,
) -> Dict[str, Any]:
    keys = (
        "efficacy_prompt_count",
        "rephrase_prompt_count",
        "efficacy_placeholders_before",
        "rephrase_placeholders_before",
        "materialized_efficacy_prompts",
        "materialized_rephrase_prompts",
        "unresolved_placeholders_after",
        "efficacy_contains_subject_after",
    )
    summary: Dict[str, Any] = {
        "enabled": bool(materialize_subject_placeholders),
        "mode": (
            MATERIALIZATION_MODE
            if materialize_subject_placeholders
            else "none"
        ),
        "placeholder": SUBJECT_PLACEHOLDER,
        "strict_at_most_one_placeholder_per_prompt": bool(
            materialize_subject_placeholders
        ),
        "strict_efficacy_subject_presence": bool(
            materialize_subject_placeholders
        ),
        "request_count": len(case_stats),
    }
    for key in keys:
        summary[key] = sum(int(stats.get(key, 0)) for stats in case_stats)

    if materialize_subject_placeholders:
        if summary["unresolved_placeholders_after"] != 0:
            raise AssertionError("strict prompt materialization left unresolved placeholders")
        if summary["efficacy_contains_subject_after"] != len(case_stats):
            raise AssertionError("strict prompt materialization lost an efficacy subject")
    return summary
