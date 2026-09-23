#!/usr/bin/env python3
"""Hybrid cumulative evaluation for the saved ZsRE editing checkpoints.

This evaluator deliberately does *not* run an editor.  It reconstructs each
saved cumulative checkpoint from the Base model and uses two complementary
evaluation protocols:

* efficacy/generalization: free greedy generation followed by normalized
  target-prefix matching (the existing cumulative-argmax protocol);
* locality/specificity: the original EasyEdit teacher-forced preservation
  metric, i.e. next-token argmax agreement between the edited model and the
  initial Base model on a fixed locality panel.

With ``--wild-eff-gen`` it also runs the ACL 2025 WILD-style efficacy and
generalization protocol as an additive measurement: QA-instruction prompting,
greedy autoregressive decoding with natural stops, and either normalized
whole-answer exact match (the deterministic WILD-EM fallback) or the official
GPT-4o-mini judge.  The existing target-prefix scores are retained unchanged
for backward compatibility and direct comparison with completed experiments.

The locality ground-truth answer is used only to define the teacher-forced
suffix/context, exactly as in EasyEdit.  It is not the primary correctness
label: the primary label is the Base model's argmax token at the same position.
Inputs remain in the project's flattened EasyEdit request format.  The legacy
metric applies no chat template; the optional WILD pass applies only its
explicitly selected context wrapper.  Prompt rewriting is disabled by default
and is available only through the explicit, provenance-tracked
``--materialize-subject-placeholders`` compatibility option.

Every evaluated state writes its per-case generated continuations and locality
token predictions, so aggregate scores remain auditable after the run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sqlite3
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import regex as regex_re
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.target_text_contract import (  # noqa: E402
    TRAINING_TERMINATORS,
    strip_training_terminators,
)
from diagnostics.evaluation_prompt_contract import (  # noqa: E402
    MATERIALIZATION_MODE,
    prepare_generation_prompts,
    summarize_prompt_contract,
)

CHECKPOINT_UTIL_PATH = REPO_ROOT / "EasyEdit" / "easyeditor" / "util" / "edited_layer_checkpoint.py"
CHECKPOINT_SPEC = importlib.util.spec_from_file_location(
    "_generation_locality_checkpoint_util",
    CHECKPOINT_UTIL_PATH,
)
if CHECKPOINT_SPEC is None or CHECKPOINT_SPEC.loader is None:
    raise ImportError(f"cannot load checkpoint helper: {CHECKPOINT_UTIL_PATH}")
CHECKPOINT_MODULE = importlib.util.module_from_spec(CHECKPOINT_SPEC)
sys.modules[CHECKPOINT_SPEC.name] = CHECKPOINT_MODULE
CHECKPOINT_SPEC.loader.exec_module(CHECKPOINT_MODULE)
STORAGE_DELTA = CHECKPOINT_MODULE.STORAGE_DELTA
load_edited_parameter_checkpoint = CHECKPOINT_MODULE.load_edited_parameter_checkpoint


PROTOCOL_NAME = "generation_prefix_em_easyedit_locality_v2"
SCHEMA_VERSION = 2
STEP_RE = re.compile(r"^step_(\d+)$")
WILD_IMPLEMENTATION_REVISION = "qaedit_wild_eff_gen_v1"
WILD_SOURCE_COMMIT = "61ccca3c1ecdb3976c9c0924e9352f3e64cc7d48"
WILD_JUDGE_PROMPT_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT"].

The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: Malia and Sasha Obama are the names of Barack Obama's children.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.

The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Malia and Sasha, Malia and Sasha, Malia and Sasha, Malia and Sasha (repeated answer)
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target or contain repeated answer.


Here is a sample. Simply reply with either CORRECT or INCORRECT.

```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

According to the gold target, please grade the predicted answer of this question as one of:
A: CORRECT
B: INCORRECT

Just return the letters "A" or "B", with no text around it.
""".strip()
WILD_JUDGE_PROMPT_SHA256 = hashlib.sha256(
    WILD_JUDGE_PROMPT_TEMPLATE.encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class RunSpec:
    label: str
    source_dir: Path

    @property
    def slug(self) -> str:
        value = re.sub(r"[^A-Za-z0-9._-]+", "_", self.label).strip("._-")
        return value or "run"

    @property
    def editor(self) -> str:
        return self.label.split("/", 1)[0]

    @property
    def variant(self) -> str:
        parts = self.label.split("/", 1)
        return parts[1] if len(parts) == 2 else ""


@dataclass(frozen=True)
class LocalityExample:
    case_index: int
    pair_index: int
    locality_key: str
    stored_prompt: str
    prompt: str
    target: str
    prompt_ids: Tuple[int, ...]
    target_ids: Tuple[int, ...]

    @property
    def input_ids(self) -> Tuple[int, ...]:
        return self.prompt_ids + self.target_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=RUN_DIR",
        help="Saved run to evaluate. Repeat for every method/variant.",
    )
    parser.add_argument(
        "--base-model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument("--requests-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="*", default=None)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument(
        "--wild-eff-gen",
        action="store_true",
        help=(
            "Additionally evaluate efficacy/generalization with the ACL 2025 "
            "WILD autoregressive protocol. Legacy target-prefix metrics remain enabled."
        ),
    )
    parser.add_argument(
        "--wild-context-type",
        choices=("qa_inst", "question-only", "chat_temp"),
        default="qa_inst",
        help=(
            "WILD input wrapper. qa_inst is the paper protocol; question-only "
            "is the released repository's CLI default; chat_temp reproduces its "
            "hard-coded Llama-2 template."
        ),
    )
    parser.add_argument("--wild-max-new-tokens", type=int, default=50)
    parser.add_argument(
        "--wild-generation-batch-size",
        type=int,
        default=1,
        help="WILD generation batch size. Exact reproduction currently requires 1.",
    )
    parser.add_argument(
        "--wild-score-mode",
        choices=("em", "judge"),
        default="em",
        help=(
            "em uses WILD's deterministic normalized whole-answer fallback; "
            "judge uses the official GPT-4o-mini rubric and requires OPENAI_API_KEY."
        ),
    )
    parser.add_argument(
        "--wild-judge-model",
        default="gpt-4o-mini",
        help="Judge model used only with --wild-score-mode judge.",
    )
    parser.add_argument("--locality-batch-size", type=int, default=32)
    parser.add_argument("--easyedit-max-length", type=int, default=256)
    parser.add_argument(
        "--locality-panel-size",
        type=int,
        default=1000,
        help="Fixed requests[:N] panel used for EasyEdit pre/post locality at every step.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--materialize-subject-placeholders",
        action="store_true",
        help=(
            "Replace exactly one literal '{}' in efficacy/rephrase prompts "
            "with request.subject before generation. This is opt-in and "
            "fails closed on missing subjects, repeated placeholders, or an "
            "efficacy prompt that still lacks its subject."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_run_spec(value: str) -> RunSpec:
    if "=" not in value:
        raise ValueError(f"--run must be LABEL=RUN_DIR, got {value!r}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise ValueError(f"--run must be LABEL=RUN_DIR, got {value!r}")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {path}")
    return RunSpec(label=label, source_dir=path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def target_prefix_match(prediction: str, target: str) -> bool:
    pred = normalize_text(prediction)
    expected = normalize_text(target)
    return bool(expected) and pred.startswith(expected)


def wild_normalize_answer(text: str) -> str:
    """Match the released WILD/SQuAD-style whole-answer normalizer exactly."""

    lowered = str(text).lower()
    without_punctuation = "".join(
        character for character in lowered if character not in set(string.punctuation)
    )
    without_articles = regex_re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def wild_exact_match(prediction: str, target: str) -> bool:
    """WILD-EM: normalized *whole-answer* equality, never prefix matching."""

    return wild_normalize_answer(prediction) == wild_normalize_answer(target)


def format_wild_prompt(question: str, context_type: str) -> str:
    """Apply one of the three prompt modes exposed by the official WILD code."""

    question = str(question)
    if context_type == "question-only":
        return question
    if context_type == "qa_inst":
        return f"Please answer the question:\n\nQ: {question}\nA:"
    if context_type == "chat_temp":
        return (
            "<s>[INST] <<SYS>>\n"
            "You are a helpful, respectful and honest assistant.\n"
            "<</SYS>>\n\n"
            f"{question} [/INST]</s>"
        )
    raise ValueError(f"unsupported WILD context type: {context_type!r}")


def wild_stop_strings(tokenizer: Any) -> Tuple[str, ...]:
    eos_token = getattr(tokenizer, "eos_token", None)
    if not isinstance(eos_token, str) or not eos_token:
        raise ValueError("WILD generation requires a non-empty tokenizer.eos_token")
    return (".", "\n", eos_token)


def trim_wild_generation(text: str, stop_strings: Sequence[str]) -> str:
    """Apply the released WILD terminal-suffix loop without extra stripping."""

    prediction = text
    for marker in stop_strings:
        if marker and prediction.endswith(marker):
            prediction = prediction[: -len(marker)]
    return prediction


def wild_stop_event(
    text: str,
    stop_strings: Sequence[str],
) -> Tuple[str, Optional[str], Optional[int]]:
    prediction = text
    last_marker: Optional[str] = None
    last_position: Optional[int] = None
    for marker in stop_strings:
        if marker and prediction.endswith(marker):
            last_position = len(prediction) - len(marker)
            last_marker = marker
            prediction = prediction[:last_position]
    return prediction, last_marker, last_position


def as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else [value]


def paired_strings(prompts: Any, targets: Any, *, context: str) -> List[Tuple[str, str]]:
    prompt_list = as_list(prompts)
    target_list = as_list(targets)
    if len(prompt_list) == 1 and len(target_list) > 1:
        prompt_list *= len(target_list)
    elif len(target_list) == 1 and len(prompt_list) > 1:
        target_list *= len(prompt_list)
    if len(prompt_list) != len(target_list):
        raise ValueError(
            f"mismatched prompt/target counts for {context}: "
            f"{len(prompt_list)} != {len(target_list)}"
        )
    pairs = []
    for prompt, target in zip(prompt_list, target_list):
        prompt_text = str(prompt)
        target_text = strip_training_terminators(target)
        if not prompt_text or not target_text:
            raise ValueError(f"empty prompt or target for {context}")
        pairs.append((prompt_text, target_text))
    return pairs


def _normalize_request_with_prompt_stats(
    record: Dict[str, Any],
    index: int,
    *,
    materialize_subject_placeholders: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    prompt, rephrases, prompt_stats = prepare_generation_prompts(
        record,
        index,
        materialize_subject_placeholders=materialize_subject_placeholders,
    )
    target = strip_training_terminators(record.get("target_new", ""))
    if not prompt or not target:
        raise ValueError(f"request index {index} lacks prompt/target_new")

    locality_pairs: List[Dict[str, str]] = []
    locality = record.get("locality") or {}
    if not isinstance(locality, dict):
        raise ValueError(f"request index {index} locality is not a mapping")
    for locality_key, payload in locality.items():
        if not isinstance(payload, dict):
            continue
        if payload.get("prompt") is None or payload.get("ground_truth") is None:
            continue
        for locality_prompt, locality_target in paired_strings(
            payload["prompt"],
            payload["ground_truth"],
            context=f"request {index} locality {locality_key}",
        ):
            locality_pairs.append(
                {
                    "key": str(locality_key),
                    "prompt": locality_prompt,
                    "target": locality_target,
                }
            )
    if not locality_pairs:
        raise ValueError(f"request index {index} has no usable locality prompt/ground_truth")

    return (
        {
            "case_id": record.get("case_id", index),
            "request_index": index,
            "prompt": str(prompt),
            "target_new": target,
            "rephrase_prompts": rephrases,
            "locality_pairs": locality_pairs,
        },
        prompt_stats,
    )


def normalize_request(
    record: Dict[str, Any],
    index: int,
    *,
    materialize_subject_placeholders: bool = False,
) -> Dict[str, Any]:
    request, _ = _normalize_request_with_prompt_stats(
        record,
        index,
        materialize_subject_placeholders=materialize_subject_placeholders,
    )
    return request


def load_requests_with_provenance(
    path: Path,
    *,
    materialize_subject_placeholders: bool = False,
) -> Tuple[List[Dict[str, Any]], str, str, Dict[str, Any]]:
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    normalized_with_stats = [
        _normalize_request_with_prompt_stats(
            record,
            idx,
            materialize_subject_placeholders=materialize_subject_placeholders,
        )
        for idx, record in enumerate(payload)
    ]
    requests = [request for request, _ in normalized_with_stats]
    prompt_stats = [stats for _, stats in normalized_with_stats]
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    normalized = json.dumps(requests, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    normalized_sha256 = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    prompt_contract = summarize_prompt_contract(
        prompt_stats,
        materialize_subject_placeholders=materialize_subject_placeholders,
    )
    provenance = {
        "path": str(path.resolve()),
        "raw_sha256": raw_sha256,
        "effective_normalized_sha256": normalized_sha256,
        "prompt_materialization": prompt_contract,
    }
    return requests, raw_sha256, normalized_sha256, provenance


def load_requests(
    path: Path,
    *,
    materialize_subject_placeholders: bool = False,
) -> Tuple[List[Dict[str, Any]], str, str]:
    requests, raw_sha256, normalized_sha256, _ = load_requests_with_provenance(
        path,
        materialize_subject_placeholders=materialize_subject_placeholders,
    )
    return requests, raw_sha256, normalized_sha256


def atomic_json(path: Path, payload: Dict[str, Any], *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            if compact:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            else:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class WildJudge:
    """Explicit, cached implementation of the official WILD GPT-4o-mini judge."""

    def __init__(self, *, model: str, cache_path: Path, api_key: str) -> None:
        if not api_key:
            raise ValueError(
                "--wild-score-mode judge requires OPENAI_API_KEY in the environment"
            )
        self.model = str(model)
        self.cache_path = cache_path
        self._api_key = api_key
        self._client: Any = None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._database = sqlite3.connect(cache_path)
        self._database.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._database.execute(
            """
            CREATE TABLE IF NOT EXISTS scores (
                cache_key TEXT PRIMARY KEY,
                score REAL NOT NULL,
                response TEXT
            )
            """
        )
        expected_metadata = {
            "schema_version": "1",
            "model": self.model,
            "temperature": "0.0",
            "judge_prompt_sha256": WILD_JUDGE_PROMPT_SHA256,
        }
        stored_metadata = dict(
            self._database.execute("SELECT key, value FROM metadata").fetchall()
        )
        if stored_metadata and stored_metadata != expected_metadata:
            self._database.close()
            raise ValueError(f"stale WILD judge cache metadata: {cache_path}")
        if not stored_metadata:
            self._database.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                expected_metadata.items(),
            )
            self._database.commit()

    def _cache_key(self, question: str, target: str, prediction: str) -> str:
        payload = {
            "model": self.model,
            "judge_prompt_sha256": WILD_JUDGE_PROMPT_SHA256,
            "question": question,
            "target": target,
            "prediction": prediction,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def score(self, question: str, target: str, prediction: str) -> Dict[str, Any]:
        key = self._cache_key(question, target, prediction)
        cached = self._database.execute(
            "SELECT score, response FROM scores WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if cached is not None:
            return {
                "score": float(cached[0]),
                "response": cached[1],
                "cache_key": key,
                "cache_hit": True,
            }

        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as error:  # pragma: no cover - environment-dependent
                raise RuntimeError("openai is required for --wild-score-mode judge") from error
            self._client = OpenAI(
                api_key=self._api_key,
                max_retries=5,
                timeout=60.0,
            )

        content = WILD_JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            target=target,
            predicted_answer=prediction,
        )
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
        )
        response = completion.choices[0].message.content
        result = {
            "score": float(response == "A"),
            "response": response,
        }
        self._database.execute(
            "INSERT INTO scores(cache_key, score, response) VALUES (?, ?, ?)",
            (key, result["score"], result["response"]),
        )
        self._database.commit()
        time.sleep(1.0)
        return {**result, "cache_key": key, "cache_hit": False}

    def close(self) -> None:
        self._database.close()


def write_measurement_exports(state_dir: Path, payload: Dict[str, Any]) -> None:
    """Write metric-specific raw JSON files alongside the combined item file."""

    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("measurement payload has no cases list")
    shared = {
        key: value
        for key, value in payload.items()
        if key not in {"cases", "prefix_summaries"}
    }
    roles = ["efficacy", "generalization", "locality"]
    if any("wild_efficacy" in case for case in cases):
        roles.extend(["wild_efficacy", "wild_generalization"])
    for role in roles:
        role_payload = {
            **shared,
            "measurement": role,
            "records": [
                {
                    "case_id": case["case_id"],
                    "request_index": case["request_index"],
                    "edit_index": case["edit_index"],
                    "measurements": case[role],
                }
                for case in cases
            ],
        }
        atomic_json(state_dir / f"{role}.json", role_payload, compact=True)

    summary_payload = {key: value for key, value in payload.items() if key != "cases"}
    atomic_json(state_dir / "summary.json", summary_payload)


def protocol_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    materialize_subject_placeholders = bool(
        getattr(args, "materialize_subject_placeholders", False)
    )
    metadata = {
        "name": PROTOCOL_NAME,
        "schema_version": SCHEMA_VERSION,
        "efficacy_generalization": {
            "mode": "greedy_generation_normalized_target_prefix_match",
            "prompt_source": "requests.json prompt/rephrase_prompt",
            "prompt_transform": (
                MATERIALIZATION_MODE
                if materialize_subject_placeholders
                else "none"
            ),
            "subject_placeholder": "{}",
            "subject_placeholder_materialization_opt_in": (
                materialize_subject_placeholders
            ),
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.generation_batch_size,
            "do_sample": False,
            "num_beams": 1,
            "training_terminators_removed": list(TRAINING_TERMINATORS),
        },
        "locality": {
            "mode": "easyedit_teacher_forced_base_post_argmax_token_agreement",
            "prompt_source": "requests.json locality.<key>.prompt",
            "prompt_transform": "none",
            "target_source": "requests.json locality.<key>.ground_truth",
            "target_role": "teacher_forcing_context_and_suffix_length_only",
            "reference": "initial_unedited_base_model",
            "separator": "single ASCII space",
            "target_add_special_tokens": False,
            "panel": f"fixed requests[:{args.locality_panel_size}] at every checkpoint",
            "aggregation": "case_macro_of_positionwise_post_equals_base",
            "batch_size": args.locality_batch_size,
            "max_length": args.easyedit_max_length,
            "padding_side": "left",
            "batch_order": "request_order",
            "implementation": "vendored_EasyEdit_test_prediction_acc_semantics",
        },
        "panels": {
            "efficacy_generalization": "requests[:edit_count]",
            "locality": f"requests[:{args.locality_panel_size}]",
        },
    }
    if bool(getattr(args, "wild_eff_gen", False)):
        score_mode = str(getattr(args, "wild_score_mode", "em"))
        wild_metadata = {
            "implementation_revision": WILD_IMPLEMENTATION_REVISION,
            "source": {
                "paper": "ACL 2025 The Mirage of Model Editing: Revisiting Evaluation in the WILD",
                "repository": "WanliYoung/Revisit-Editing-Evaluation",
                "commit": WILD_SOURCE_COMMIT,
            },
            "roles": {
                "efficacy": "request.prompt",
                "generalization": "request.rephrase_prompt",
                "target": "request.target_new",
            },
            "prompt_transform": (
                MATERIALIZATION_MODE
                if materialize_subject_placeholders
                else "none"
            ),
            "training_terminators_removed": list(TRAINING_TERMINATORS),
            "context_type": str(getattr(args, "wild_context_type", "qa_inst")),
            "qa_instruction": "Please answer the question:\n\nQ: {question}\nA:",
            "paper_default_context_type": "qa_inst",
            "released_cli_default_context_type": "question-only",
            "generation": {
                "mode": "greedy_autoregressive_natural_stop",
                "max_new_tokens": int(getattr(args, "wild_max_new_tokens", 50)),
                "batch_size": int(getattr(args, "wild_generation_batch_size", 1)),
                "official_batch_size": 1,
                "do_sample": False,
                "num_beams": 1,
                "use_cache": False,
                "stop_strings": [".", "\n", "tokenizer.eos_token"],
                "postprocess": "released_terminal_suffix_loop_without_strip",
            },
            "scoring": {
                "selected_mode": score_mode,
                "em": {
                    "name": "WILD-EM",
                    "comparison": "normalized_whole_answer_equality",
                    "normalization": (
                        "lowercase_then_delete_ASCII_punctuation_then_remove_"
                        "standalone_a_an_the_then_collapse_whitespace"
                    ),
                },
                "judge": {
                    "enabled": score_mode == "judge",
                    "provider": "OpenAI",
                    "model": str(getattr(args, "wild_judge_model", "gpt-4o-mini")),
                    "temperature": 0.0,
                    "judge_prompt_sha256": WILD_JUDGE_PROMPT_SHA256,
                    "correct_response": "exactly_A",
                    "credential_source": "OPENAI_API_KEY_environment_only",
                    "silent_em_fallback": False,
                },
            },
            "aggregation": "case_macro_binary_accuracy",
        }
        metadata["wild_efficacy_generalization"] = wild_metadata
    return metadata


def protocol_fingerprint(
    args: argparse.Namespace,
    requests_sha256: str,
    raw_requests_sha256: Optional[str] = None,
) -> str:
    payload = {
        "protocol": protocol_metadata(args),
        "base_model": args.base_model,
        "dtype": args.dtype,
        "effective_normalized_requests_sha256": requests_sha256,
    }
    if raw_requests_sha256 is not None:
        payload["raw_requests_sha256"] = raw_requests_sha256
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def request_provenance_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    provenance = getattr(args, "request_provenance", None)
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("request provenance was not initialized")
    return provenance


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.inference_mode()
def greedy_continuations(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    batch_size: int,
) -> List[str]:
    if not prompts:
        return []
    outputs: List[str] = []
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = list(prompts[start : start + batch_size])
            encoded = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(model_device(model))
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            continuation_ids = generated[:, encoded["input_ids"].shape[1] :]
            outputs.extend(tokenizer.batch_decode(continuation_ids, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = previous_padding_side
    return outputs


@torch.inference_mode()
def wild_greedy_continuations(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 50,
    batch_size: int = 1,
) -> List[Dict[str, Any]]:
    """Generate WILD answers and retain raw and natural-stop-trimmed text."""

    if not prompts:
        return []
    if max_new_tokens <= 0:
        raise ValueError("WILD max_new_tokens must be positive")
    if batch_size != 1:
        raise ValueError("exact WILD generation requires batch_size=1")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("WILD generation requires tokenizer.eos_token_id")
    stops = wild_stop_strings(tokenizer)
    outputs: List[Dict[str, Any]] = []
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = list(prompts[start : start + batch_size])
            encoded = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
            ).to(model_device(model))
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                stop_strings=list(stops),
                tokenizer=tokenizer,
                pad_token_id=eos_token_id,
                do_sample=False,
                num_beams=1,
                use_cache=False,
            )
            continuation_ids = generated[:, encoded["input_ids"].shape[1] :]
            for row in continuation_ids:
                raw_prediction = tokenizer.decode(row.detach().cpu().tolist())
                prediction, stop_marker, stop_character_index = wild_stop_event(
                    raw_prediction,
                    stops,
                )
                outputs.append(
                    {
                        "raw_prediction": raw_prediction,
                        "prediction": prediction,
                        "stop_marker": stop_marker,
                        "stop_character_index": stop_character_index,
                    }
                )
    finally:
        tokenizer.padding_side = previous_padding_side
    if len(outputs) != len(prompts):
        raise RuntimeError(f"WILD generation count mismatch: {len(outputs)} != {len(prompts)}")
    return outputs


def token_labels(tokenizer: Any, token_ids: Sequence[int]) -> Tuple[List[str], List[str]]:
    pieces = tokenizer.convert_ids_to_tokens(list(token_ids))
    if isinstance(pieces, str):
        pieces = [pieces]
    decoded = [tokenizer.decode([int(token_id)], skip_special_tokens=False) for token_id in token_ids]
    return [str(piece) for piece in pieces], decoded


def locality_evaluation_prompt(stored_prompt: str) -> str:
    """Use the flattened EasyEdit request prompt exactly as it was stored."""

    return stored_prompt


def build_locality_examples(
    requests: Sequence[Dict[str, Any]],
    tokenizer: Any,
) -> List[LocalityExample]:
    examples: List[LocalityExample] = []
    for case_index, request in enumerate(requests):
        for pair_index, pair in enumerate(request["locality_pairs"]):
            evaluated_prompt = locality_evaluation_prompt(pair["prompt"])
            prompt_ids = tuple(
                int(x) for x in tokenizer.encode(evaluated_prompt, add_special_tokens=True)
            )
            target_ids = tuple(
                int(x)
                for x in tokenizer.encode(" " + pair["target"], add_special_tokens=False)
            )
            if not prompt_ids:
                raise ValueError(f"empty tokenized locality prompt for request {case_index}")
            if not target_ids:
                raise ValueError(f"empty tokenized locality target for request {case_index}")
            examples.append(
                LocalityExample(
                    case_index=case_index,
                    pair_index=pair_index,
                    locality_key=pair["key"],
                    stored_prompt=pair["prompt"],
                    prompt=evaluated_prompt,
                    target=pair["target"],
                    prompt_ids=prompt_ids,
                    target_ids=target_ids,
                )
            )
    return examples


@torch.inference_mode()
def score_teacher_forced_examples(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: Sequence[LocalityExample],
    *,
    batch_size: int,
    max_length: int = 256,
    reference_predictions: Optional[Sequence[Sequence[int]]] = None,
) -> List[Dict[str, Any]]:
    if not examples:
        return []
    if batch_size <= 0:
        raise ValueError("locality batch size must be positive")
    if reference_predictions is not None and len(reference_predictions) != len(examples):
        raise ValueError(
            "locality reference count mismatch: "
            f"{len(reference_predictions)} != {len(examples)}"
        )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer needs a pad_token_id")

    results: List[Optional[Dict[str, Any]]] = [None] * len(examples)
    device = model_device(model)

    # Preserve EasyEdit's request order, joint ``prompt + ' ' + target``
    # tokenization, left padding, and suffix slicing.  These details can alter
    # BF16 argmax values, so a merely equivalent right-padded implementation is
    # not protocol-compatible in practice.
    for start in range(0, len(examples), batch_size):
        batch_examples = list(examples[start : start + batch_size])
        prompts = [example.prompt for example in batch_examples]
        targets = [example.target for example in batch_examples]
        prompt_targets = [prompt + " " + target for prompt, target in zip(prompts, targets)]
        encoded_max_length = max(
            max_length,
            max(len(tokenizer.encode(text)) for text in prompt_targets) + 1,
        )
        previous_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            prompt_target_tokens = tokenizer(
                prompt_targets,
                padding=True,
                truncation=True,
                max_length=encoded_max_length,
                return_tensors="pt",
            )
            prompt_tokens = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=encoded_max_length,
                return_tensors="pt",
            )
        finally:
            tokenizer.padding_side = previous_padding_side

        num_prompt_tokens = [
            int((row != pad_token_id).sum().item())
            for row in prompt_tokens["input_ids"]
        ]
        num_pad_tokens = [
            int((row == pad_token_id).sum().item())
            for row in prompt_target_tokens["input_ids"]
        ]
        suffix_starts = [
            pad_count + prompt_count
            for pad_count, prompt_count in zip(num_pad_tokens, num_prompt_tokens)
        ]
        prompt_target_tokens = prompt_target_tokens.to(device)

        logits = model(**prompt_target_tokens).logits
        answer_ids = logits.argmax(dim=-1).detach().cpu()
        label_ids = prompt_target_tokens["input_ids"].detach().cpu()

        for row, (example, suffix_start) in enumerate(zip(batch_examples, suffix_starts)):
            result_index = start + row
            predicted_ids = [int(x) for x in answer_ids[row, suffix_start - 1 : -1].tolist()]
            target_ids = [int(x) for x in label_ids[row, suffix_start:].tolist()]
            if not predicted_ids or len(predicted_ids) != len(target_ids):
                raise RuntimeError(
                    "EasyEdit suffix slicing failed for "
                    f"request={example.case_index} pair={example.pair_index}"
                )
            if reference_predictions is None:
                base_predicted_ids = [int(x) for x in predicted_ids]
            else:
                base_predicted_ids = [int(x) for x in reference_predictions[result_index]]
            if len(base_predicted_ids) != len(predicted_ids):
                raise ValueError(
                    "locality reference token count mismatch for "
                    f"request={example.case_index} pair={example.pair_index}: "
                    f"{len(base_predicted_ids)} != {len(predicted_ids)}"
                )
            correct = [
                int(predicted) == int(reference)
                for predicted, reference in zip(predicted_ids, base_predicted_ids)
            ]
            ground_truth_correct = [
                int(predicted) == int(target)
                for predicted, target in zip(predicted_ids, target_ids)
            ]
            target_pieces, target_decoded = token_labels(tokenizer, target_ids)
            predicted_pieces, predicted_decoded = token_labels(tokenizer, predicted_ids)
            base_pieces, base_decoded = token_labels(tokenizer, base_predicted_ids)
            results[result_index] = {
                "key": example.locality_key,
                "pair_index": example.pair_index,
                "stored_prompt": example.stored_prompt,
                "prompt": example.prompt,
                "target": example.target,
                "prompt_token_count": num_prompt_tokens[row],
                "target_token_ids": target_ids,
                "target_tokens": target_pieces,
                "target_token_text": target_decoded,
                "predicted_token_ids": [int(x) for x in predicted_ids],
                "post_predicted_token_ids": [int(x) for x in predicted_ids],
                "predicted_tokens": predicted_pieces,
                "predicted_token_text": predicted_decoded,
                "base_predicted_token_ids": base_predicted_ids,
                "base_predicted_tokens": base_pieces,
                "base_predicted_token_text": base_decoded,
                "token_correct": correct,
                "token_accuracy": sum(correct) / len(correct),
                "all_tokens_correct": all(correct),
                "ground_truth_token_correct": ground_truth_correct,
                "ground_truth_token_accuracy": sum(ground_truth_correct)
                / len(ground_truth_correct),
                "ground_truth_all_tokens_correct": all(ground_truth_correct),
            }

        del logits, answer_ids, label_ids, prompt_target_tokens, prompt_tokens

    if any(result is None for result in results):
        raise RuntimeError("internal error: missing locality result")
    return [result for result in results if result is not None]


def evaluate_state(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Dict[str, Any]],
    *,
    generation_count: int,
    locality_count: int,
    max_new_tokens: int,
    generation_batch_size: int,
    locality_batch_size: int,
    easyedit_max_length: int,
    locality_reference_predictions: Optional[Sequence[Sequence[int]]] = None,
    wild_eff_gen: bool = False,
    wild_context_type: str = "qa_inst",
    wild_max_new_tokens: int = 50,
    wild_generation_batch_size: int = 1,
    wild_score_mode: str = "em",
    wild_judge: Optional[WildJudge] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not 0 <= generation_count <= len(requests):
        raise ValueError(f"invalid generation_count={generation_count} for {len(requests)} requests")
    if not 0 <= locality_count <= len(requests):
        raise ValueError(f"invalid locality_count={locality_count} for {len(requests)} requests")
    item_count = max(generation_count, locality_count)
    items: List[Dict[str, Any]] = [
        {
            "case_id": request["case_id"],
            "request_index": request["request_index"],
            "edit_index": request["request_index"] + 1,
            "efficacy": [],
            "generalization": [],
            "locality": [],
        }
        for request in requests[:item_count]
    ]
    if wild_eff_gen:
        for item in items:
            item["wild_efficacy"] = []
            item["wild_generalization"] = []

    generation_jobs: List[Tuple[int, str, str, str]] = []
    for case_index, request in enumerate(requests[:generation_count]):
        generation_jobs.append((case_index, "efficacy", request["prompt"], request["target_new"]))
        for rephrase in request["rephrase_prompts"]:
            generation_jobs.append((case_index, "generalization", rephrase, request["target_new"]))

    continuations = greedy_continuations(
        model,
        tokenizer,
        [prompt for _, _, prompt, _ in generation_jobs],
        max_new_tokens=max_new_tokens,
        batch_size=generation_batch_size,
    )
    for (case_index, role, prompt, target), prediction in zip(generation_jobs, continuations):
        items[case_index][role].append(
            {
                "prompt": prompt,
                "target": target,
                "prediction": prediction,
                "normalized_target": normalize_text(target),
                "normalized_prediction": normalize_text(prediction),
                "correct": target_prefix_match(prediction, target),
            }
        )

    if wild_eff_gen:
        if wild_score_mode not in {"em", "judge"}:
            raise ValueError(f"unsupported WILD score mode: {wild_score_mode!r}")
        if wild_score_mode == "judge" and wild_judge is None:
            raise ValueError("WILD judge mode requires an initialized WildJudge")
        wild_input_prompts = [
            format_wild_prompt(prompt, wild_context_type)
            for _, _, prompt, _ in generation_jobs
        ]
        wild_outputs = wild_greedy_continuations(
            model,
            tokenizer,
            wild_input_prompts,
            max_new_tokens=wild_max_new_tokens,
            batch_size=wild_generation_batch_size,
        )
        for job, evaluation_prompt, output in zip(
            generation_jobs,
            wild_input_prompts,
            wild_outputs,
        ):
            case_index, legacy_role, raw_prompt, target = job
            wild_role = f"wild_{legacy_role}"
            prediction = str(output["prediction"])
            em_correct = wild_exact_match(prediction, target)
            record: Dict[str, Any] = {
                "raw_prompt": raw_prompt,
                "evaluation_prompt": evaluation_prompt,
                "context_type": wild_context_type,
                "target": target,
                "raw_prediction": output["raw_prediction"],
                "prediction": prediction,
                "stop_marker": output["stop_marker"],
                "stop_character_index": output["stop_character_index"],
                "normalized_target": wild_normalize_answer(target),
                "normalized_prediction": wild_normalize_answer(prediction),
                "em_correct": em_correct,
                "score_mode": wild_score_mode,
            }
            if wild_score_mode == "judge":
                assert wild_judge is not None
                judge_result = wild_judge.score(raw_prompt, target, prediction)
                record["judge"] = judge_result
                record["correct"] = bool(judge_result["score"])
            else:
                record["correct"] = em_correct
            items[case_index][wild_role].append(record)

    locality_examples = build_locality_examples(requests[:locality_count], tokenizer)
    locality_results = score_teacher_forced_examples(
        model,
        tokenizer,
        locality_examples,
        batch_size=locality_batch_size,
        max_length=easyedit_max_length,
        reference_predictions=locality_reference_predictions,
    )
    for example, result in zip(locality_examples, locality_results):
        items[example.case_index]["locality"].append(result)

    return items, summarize_items(
        items,
        generation_count=generation_count,
        locality_count=locality_count,
    )


def locality_reference_predictions(
    base_items: Sequence[Dict[str, Any]],
    locality_count: int,
) -> List[List[int]]:
    """Flatten the Base token predictions in request/pair order for reuse."""

    references: List[List[int]] = []
    for item in base_items:
        if int(item.get("request_index", len(references))) >= locality_count:
            continue
        for entry in item.get("locality", []):
            token_ids = entry.get("base_predicted_token_ids", entry.get("predicted_token_ids"))
            if not isinstance(token_ids, list) or not token_ids:
                raise ValueError("Base locality result has no predicted token IDs")
            references.append([int(token_id) for token_id in token_ids])
    if not references and locality_count > 0:
        raise ValueError("Base result has no locality reference predictions")
    return references


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def population_std(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


def harmonic_mean(values: Sequence[float]) -> float:
    if not values or any(math.isnan(value) for value in values):
        return float("nan")
    if any(value == 0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def summarize_items(
    items: Sequence[Dict[str, Any]],
    *,
    generation_count: Optional[int] = None,
    locality_count: Optional[int] = None,
) -> Dict[str, Any]:
    generation_count = len(items) if generation_count is None else generation_count
    locality_count = len(items) if locality_count is None else locality_count
    efficacy_case_scores: List[float] = []
    generalization_case_scores: List[float] = []
    wild_efficacy_case_scores: List[float] = []
    wild_generalization_case_scores: List[float] = []
    wild_em_efficacy_case_scores: List[float] = []
    wild_em_generalization_case_scores: List[float] = []
    locality_case_scores: List[float] = []
    locality_case_exact: List[float] = []
    locality_token_hits: List[bool] = []
    locality_ground_truth_case_scores: List[float] = []
    locality_ground_truth_case_exact: List[float] = []
    locality_ground_truth_token_hits: List[bool] = []

    for position, item in enumerate(items):
        request_index = int(item.get("request_index", position))
        if request_index < generation_count and item["efficacy"]:
            efficacy_case_scores.append(mean([float(entry["correct"]) for entry in item["efficacy"]]))
        if request_index < generation_count and item["generalization"]:
            generalization_case_scores.append(
                mean([float(entry["correct"]) for entry in item["generalization"]])
            )
        if request_index < generation_count and item.get("wild_efficacy"):
            wild_efficacy_case_scores.append(
                mean([float(entry["correct"]) for entry in item["wild_efficacy"]])
            )
            wild_em_efficacy_case_scores.append(
                mean([float(entry["em_correct"]) for entry in item["wild_efficacy"]])
            )
        if request_index < generation_count and item.get("wild_generalization"):
            wild_generalization_case_scores.append(
                mean([float(entry["correct"]) for entry in item["wild_generalization"]])
            )
            wild_em_generalization_case_scores.append(
                mean([float(entry["em_correct"]) for entry in item["wild_generalization"]])
            )
        if request_index >= locality_count:
            continue
        case_locality_hits = [
            bool(hit)
            for entry in item["locality"]
            for hit in entry["token_correct"]
        ]
        if case_locality_hits:
            locality_token_hits.extend(case_locality_hits)
            locality_pair_scores = [
                mean([float(hit) for hit in entry["token_correct"]])
                for entry in item["locality"]
                if entry["token_correct"]
            ]
            locality_case_scores.append(mean(locality_pair_scores))
            locality_case_exact.append(float(all(case_locality_hits)))
        case_ground_truth_hits = [
            bool(hit)
            for entry in item["locality"]
            for hit in entry.get("ground_truth_token_correct", [])
        ]
        if case_ground_truth_hits:
            locality_ground_truth_token_hits.extend(case_ground_truth_hits)
            ground_truth_pair_scores = [
                mean([float(hit) for hit in entry.get("ground_truth_token_correct", [])])
                for entry in item["locality"]
                if entry.get("ground_truth_token_correct")
            ]
            locality_ground_truth_case_scores.append(mean(ground_truth_pair_scores))
            locality_ground_truth_case_exact.append(float(all(case_ground_truth_hits)))

    efficacy = mean(efficacy_case_scores)
    generalization = mean(generalization_case_scores)
    specificity = mean(locality_case_scores)
    summary = {
        "n_requests": generation_count,
        "generation_panel_size": generation_count,
        "locality_panel_size": locality_count,
        "efficacy": efficacy,
        "generalization": generalization,
        "specificity": specificity,
        "derived_harmonic_mean": harmonic_mean([efficacy, generalization, specificity]),
        "efficacy_std": population_std(efficacy_case_scores),
        "generalization_std": population_std(generalization_case_scores),
        "specificity_std": population_std(locality_case_scores),
        "efficacy_case_count": len(efficacy_case_scores),
        "generalization_case_count": len(generalization_case_scores),
        "specificity_case_count": len(locality_case_scores),
        "specificity_token_count": len(locality_token_hits),
        "specificity_token_micro_accuracy": mean([float(hit) for hit in locality_token_hits]),
        "specificity_case_exact_rate": mean(locality_case_exact),
        "locality_ground_truth_case_macro_accuracy": mean(locality_ground_truth_case_scores),
        "locality_ground_truth_token_micro_accuracy": mean(
            [float(hit) for hit in locality_ground_truth_token_hits]
        ),
        "locality_ground_truth_case_exact_rate": mean(locality_ground_truth_case_exact),
    }
    if any("wild_efficacy" in item or "wild_generalization" in item for item in items):
        wild_efficacy = mean(wild_efficacy_case_scores)
        wild_generalization = mean(wild_generalization_case_scores)
        wild_em_efficacy = mean(wild_em_efficacy_case_scores)
        wild_em_generalization = mean(wild_em_generalization_case_scores)
        summary.update(
            {
                "wild_efficacy": wild_efficacy,
                "wild_generalization": wild_generalization,
                "wild_em_efficacy": wild_em_efficacy,
                "wild_em_generalization": wild_em_generalization,
                "wild_efficacy_std": population_std(wild_efficacy_case_scores),
                "wild_generalization_std": population_std(
                    wild_generalization_case_scores
                ),
                "wild_em_efficacy_std": population_std(
                    wild_em_efficacy_case_scores
                ),
                "wild_em_generalization_std": population_std(
                    wild_em_generalization_case_scores
                ),
                "wild_efficacy_case_count": len(wild_efficacy_case_scores),
                "wild_generalization_case_count": len(
                    wild_generalization_case_scores
                ),
                "wild_eff_gen_base_preservation_harmonic_mean": harmonic_mean(
                    [wild_efficacy, wild_generalization, specificity]
                ),
                "wild_em_eff_gen_base_preservation_harmonic_mean": harmonic_mean(
                    [wild_em_efficacy, wild_em_generalization, specificity]
                ),
            }
        )
    return summary


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_steps(run_dir: Path, requested: Optional[Sequence[int]]) -> List[int]:
    found = []
    for child in run_dir.iterdir():
        match = STEP_RE.match(child.name)
        if match and (child / "edited_parameter_deltas.pt").is_file():
            found.append(int(match.group(1)))
    found.sort()
    if requested is not None:
        wanted = set(requested)
        found = [step for step in found if step in wanted]
    return found


def checkpoint_identity(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "storage_mode": manifest.get("storage_mode"),
        "edit_count": manifest.get("edit_count"),
        "request_fingerprint": manifest.get("request_fingerprint"),
        "parameter_names": manifest.get("parameter_names"),
    }


def cached_payload_is_valid(
    payload: Dict[str, Any],
    *,
    fingerprint: str,
    checkpoint: Optional[Dict[str, Any]],
) -> bool:
    if payload.get("protocol_fingerprint") != fingerprint:
        return False
    if checkpoint is None:
        return payload.get("checkpoint", {}).get("type") == "base"
    cached = payload.get("checkpoint") or {}
    return all(cached.get(key) == checkpoint.get(key) for key in ("path", "size", "mtime_ns"))


def load_or_evaluate_base(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: List[Dict[str, Any]],
    panel_sizes: Sequence[int],
    args: argparse.Namespace,
    fingerprint: str,
    raw_requests_sha256: str,
    wild_judge: Optional[WildJudge] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    locality_count = min(args.locality_panel_size, len(requests))
    if locality_count <= 0:
        raise ValueError("--locality-panel-size must be positive")
    output_path = args.output_root / "base" / "items.json"
    if output_path.is_file() and not args.force:
        payload = json.loads(output_path.read_text())
        if cached_payload_is_valid(payload, fingerprint=fingerprint, checkpoint=None):
            prefix_summaries = payload.setdefault("prefix_summaries", {})
            added_prefix = False
            for size in sorted(set(panel_sizes)):
                if 0 < size <= len(requests) and str(size) not in prefix_summaries:
                    prefix_summaries[str(size)] = summarize_items(
                        payload["cases"],
                        generation_count=size,
                        locality_count=locality_count,
                    )
                    added_prefix = True
            if added_prefix:
                atomic_json(output_path, payload, compact=True)
            write_measurement_exports(output_path.parent, payload)
            print(f"[resume] Base items: {output_path}", flush=True)
            return payload["cases"], payload
        raise ValueError(f"stale Base result exists; inspect or rerun with --force: {output_path}")

    started = time.time()
    print(f"[base] evaluating {len(requests)} requests", flush=True)
    items, full_summary = evaluate_state(
        model,
        tokenizer,
        requests,
        generation_count=len(requests),
        locality_count=locality_count,
        max_new_tokens=args.max_new_tokens,
        generation_batch_size=args.generation_batch_size,
        locality_batch_size=args.locality_batch_size,
        easyedit_max_length=args.easyedit_max_length,
        wild_eff_gen=args.wild_eff_gen,
        wild_context_type=args.wild_context_type,
        wild_max_new_tokens=args.wild_max_new_tokens,
        wild_generation_batch_size=args.wild_generation_batch_size,
        wild_score_mode=args.wild_score_mode,
        wild_judge=wild_judge,
    )
    prefix_summaries = {
        str(size): summarize_items(
            items,
            generation_count=size,
            locality_count=locality_count,
        )
        for size in sorted(set(panel_sizes))
        if 0 < size <= len(requests)
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol_metadata(args),
        "protocol_fingerprint": fingerprint,
        "request_provenance": request_provenance_metadata(args),
        "base_model": args.base_model,
        "requests_path": str(args.requests_path),
        "raw_requests_sha256": raw_requests_sha256,
        "checkpoint": {"type": "base", "edit_count": 0},
        "panel": {
            "generation": {"type": "full_requests", "num_cases": len(requests)},
            "locality": {
                "type": "fixed_prefix",
                "start": 0,
                "end": locality_count,
                "num_cases": locality_count,
            },
        },
        "summary": full_summary,
        "prefix_summaries": prefix_summaries,
        "seconds": round(time.time() - started, 1),
        "cases": items,
    }
    atomic_json(output_path, payload, compact=True)
    write_measurement_exports(output_path.parent, payload)
    print(f"[base] wrote {output_path} ({payload['seconds']:.1f}s)", flush=True)
    return items, payload


def validate_checkpoint_manifest(
    checkpoint: Dict[str, Any],
    *,
    step: int,
    expected_names: Sequence[str],
) -> None:
    if checkpoint.get("storage_mode") != STORAGE_DELTA:
        raise ValueError(f"checkpoint is not a cumulative parameter delta: {checkpoint['path']}")
    manifest_step = checkpoint.get("edit_count")
    if manifest_step is not None and int(manifest_step) != step:
        raise ValueError(f"checkpoint edit_count mismatch for {checkpoint['path']}")
    names = checkpoint.get("parameter_names")
    if names is not None and list(names) != list(expected_names):
        raise ValueError(f"checkpoint parameter_names mismatch for {checkpoint['path']}")


def evaluate_run(
    run: RunSpec,
    model: torch.nn.Module,
    tokenizer: Any,
    requests: List[Dict[str, Any]],
    pristine: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    fingerprint: str,
    base_payload: Dict[str, Any],
    wild_judge: Optional[WildJudge] = None,
) -> Dict[str, Any]:
    steps = discover_steps(run.source_dir, args.steps)
    if not steps:
        print(f"[warn] no requested delta checkpoints: {run.source_dir}", flush=True)
        return {
            "label": run.label,
            "editor": run.editor,
            "variant": run.variant,
            "source_run_dir": str(run.source_dir),
            "results": [],
        }

    output_dir = args.output_root / "runs" / run.slug
    named_parameters = dict(model.named_parameters())
    locality_count = min(args.locality_panel_size, len(requests))
    base_references = locality_reference_predictions(base_payload["cases"], locality_count)
    trajectory_rows = []
    for step in steps:
        checkpoint_path = run.source_dir / f"step_{step:03d}" / "edited_parameter_deltas.pt"
        checkpoint = checkpoint_identity(checkpoint_path)
        validate_checkpoint_manifest(checkpoint, step=step, expected_names=list(pristine))
        item_path = output_dir / f"step_{step:03d}" / "items.json"

        summary: Dict[str, Any]
        seconds: float
        if item_path.is_file() and not args.force:
            payload = json.loads(item_path.read_text())
            if not cached_payload_is_valid(payload, fingerprint=fingerprint, checkpoint=checkpoint):
                raise ValueError(f"stale step result exists; inspect or rerun with --force: {item_path}")
            write_measurement_exports(item_path.parent, payload)
            summary = payload["summary"]
            seconds = float(payload.get("seconds", 0.0))
            print(f"[resume] {run.label} step={step}: {item_path}", flush=True)
        else:
            with torch.no_grad():
                for name, base_value in pristine.items():
                    named_parameters[name].copy_(base_value)
            loaded_metadata = load_edited_parameter_checkpoint(model, checkpoint_path)
            if loaded_metadata.get("storage_mode") != STORAGE_DELTA:
                raise ValueError(f"loader returned non-delta checkpoint: {checkpoint_path}")

            started = time.time()
            items, summary = evaluate_state(
                model,
                tokenizer,
                requests,
                generation_count=step,
                locality_count=locality_count,
                max_new_tokens=args.max_new_tokens,
                generation_batch_size=args.generation_batch_size,
                locality_batch_size=args.locality_batch_size,
                easyedit_max_length=args.easyedit_max_length,
                locality_reference_predictions=base_references,
                wild_eff_gen=args.wild_eff_gen,
                wild_context_type=args.wild_context_type,
                wild_max_new_tokens=args.wild_max_new_tokens,
                wild_generation_batch_size=args.wild_generation_batch_size,
                wild_score_mode=args.wild_score_mode,
                wild_judge=wild_judge,
            )
            seconds = round(time.time() - started, 1)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "protocol": protocol_metadata(args),
                "protocol_fingerprint": fingerprint,
                "request_provenance": request_provenance_metadata(args),
                "base_model": args.base_model,
                "run": {
                    "label": run.label,
                    "editor": run.editor,
                    "variant": run.variant,
                    "source_run_dir": str(run.source_dir),
                },
                "checkpoint": checkpoint,
                "panel": {
                    "efficacy_generalization": {
                        "type": "edited_prefix",
                        "start": 0,
                        "end": step,
                        "num_cases": step,
                    },
                    "locality": {
                        "type": "fixed_prefix",
                        "start": 0,
                        "end": locality_count,
                        "num_cases": locality_count,
                    },
                },
                "summary": summary,
                "seconds": seconds,
                "cases": items,
            }
            atomic_json(item_path, payload, compact=True)
            write_measurement_exports(item_path.parent, payload)

        pre_summary = base_payload["prefix_summaries"].get(str(step))
        row = {
            "edit_count": step,
            "panel_size": summary["n_requests"],
            "pre": pre_summary,
            "post": summary,
            "seconds": seconds,
            "items_path": str(item_path),
            "checkpoint_path": str(checkpoint_path),
        }
        trajectory_rows.append(row)
        wild_status = ""
        if args.wild_eff_gen:
            wild_status = (
                f" wild_eff={summary['wild_efficacy']:.4f}"
                f" wild_gen={summary['wild_generalization']:.4f}"
            )
        print(
            f"[{run.label} step={step:4d}] "
            f"eff={summary['efficacy']:.4f} gen={summary['generalization']:.4f} "
            f"loc={summary['specificity']:.4f} H={summary['derived_harmonic_mean']:.4f} "
            f"{wild_status} "
            f"({seconds:.1f}s)",
            flush=True,
        )

    trajectory = {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol_metadata(args),
        "protocol_fingerprint": fingerprint,
        "request_provenance": request_provenance_metadata(args),
        "label": run.label,
        "editor": run.editor,
        "variant": run.variant,
        "source_run_dir": str(run.source_dir),
        "base_items_path": str(args.output_root / "base" / "items.json"),
        "results": trajectory_rows,
    }
    atomic_json(output_dir / "trajectory.json", trajectory)
    return trajectory


def write_comparison(
    output_root: Path,
    base_payload: Dict[str, Any],
    trajectories: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    fingerprint: str,
) -> None:
    rows: List[Dict[str, Any]] = []
    for trajectory in trajectories:
        for result in trajectory["results"]:
            row = {
                "label": trajectory["label"],
                "editor": trajectory["editor"],
                "variant": trajectory["variant"],
                "edit_count": result["edit_count"],
                "panel_size": result["panel_size"],
                "pre": result["pre"],
                "post": result["post"],
                "items_path": result["items_path"],
                "checkpoint_path": result["checkpoint_path"],
            }
            rows.append(row)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol_metadata(args),
        "protocol_fingerprint": fingerprint,
        "request_provenance": request_provenance_metadata(args),
        "base": {
            "summary": base_payload["summary"],
            "prefix_summaries": base_payload["prefix_summaries"],
            "items_path": str(output_root / "base" / "items.json"),
        },
        "rows": rows,
    }
    atomic_json(output_root / "comparison_summary.json", payload)

    csv_path = output_root / "comparison_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    fields = [
        "label",
        "editor",
        "variant",
        "edit_count",
        "panel_size",
        "locality_panel_size",
        "pre_efficacy",
        "pre_generalization",
        "pre_specificity",
        "pre_harmonic",
        "efficacy",
        "generalization",
        "specificity",
        "derived_harmonic_mean",
        "specificity_token_micro_accuracy",
        "specificity_case_exact_rate",
        "locality_ground_truth_case_macro_accuracy",
        "items_path",
        "checkpoint_path",
    ]
    if args.wild_eff_gen:
        insertion = fields.index("specificity")
        fields[insertion:insertion] = [
            "pre_wild_efficacy",
            "pre_wild_generalization",
            "pre_wild_em_efficacy",
            "pre_wild_em_generalization",
            "wild_efficacy",
            "wild_generalization",
            "wild_em_efficacy",
            "wild_em_generalization",
            "wild_eff_gen_base_preservation_harmonic_mean",
        ]
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                pre = row["pre"] or {}
                post = row["post"]
                output_row = {
                        "label": row["label"],
                        "editor": row["editor"],
                        "variant": row["variant"],
                        "edit_count": row["edit_count"],
                        "panel_size": row["panel_size"],
                        "locality_panel_size": post.get("locality_panel_size"),
                        "pre_efficacy": pre.get("efficacy"),
                        "pre_generalization": pre.get("generalization"),
                        "pre_specificity": pre.get("specificity"),
                        "pre_harmonic": pre.get("derived_harmonic_mean"),
                        "efficacy": post.get("efficacy"),
                        "generalization": post.get("generalization"),
                        "specificity": post.get("specificity"),
                        "derived_harmonic_mean": post.get("derived_harmonic_mean"),
                        "specificity_token_micro_accuracy": post.get(
                            "specificity_token_micro_accuracy"
                        ),
                        "specificity_case_exact_rate": post.get("specificity_case_exact_rate"),
                        "locality_ground_truth_case_macro_accuracy": post.get(
                            "locality_ground_truth_case_macro_accuracy"
                        ),
                        "items_path": row["items_path"],
                        "checkpoint_path": row["checkpoint_path"],
                    }
                if args.wild_eff_gen:
                    output_row.update(
                        {
                            "pre_wild_efficacy": pre.get("wild_efficacy"),
                            "pre_wild_generalization": pre.get("wild_generalization"),
                            "pre_wild_em_efficacy": pre.get("wild_em_efficacy"),
                            "pre_wild_em_generalization": pre.get(
                                "wild_em_generalization"
                            ),
                            "wild_efficacy": post.get("wild_efficacy"),
                            "wild_generalization": post.get("wild_generalization"),
                            "wild_em_efficacy": post.get("wild_em_efficacy"),
                            "wild_em_generalization": post.get(
                                "wild_em_generalization"
                            ),
                            "wild_eff_gen_base_preservation_harmonic_mean": post.get(
                                "wild_eff_gen_base_preservation_harmonic_mean"
                            ),
                        }
                    )
                writer.writerow(output_row)
        os.replace(temporary, csv_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def select_parameter_names(runs: Sequence[RunSpec], steps_by_run: Dict[str, List[int]]) -> List[str]:
    for run in runs:
        steps = steps_by_run[run.label]
        if not steps:
            continue
        manifest_path = run.source_dir / f"step_{steps[-1]:03d}" / "edited_parameter_deltas.json"
        manifest = json.loads(manifest_path.read_text())
        names = manifest.get("parameter_names")
        if not isinstance(names, list) or not names:
            raise ValueError(f"checkpoint manifest has no parameter_names: {manifest_path}")
        return [str(name) for name in names]
    raise ValueError("no usable checkpoints in requested runs")


def validate_run_requests(
    runs: Sequence[RunSpec],
    canonical_sha256: str,
) -> None:
    for run in runs:
        request_path = run.source_dir / "requests.json"
        if not request_path.is_file():
            raise FileNotFoundError(f"run has no requests.json: {run.source_dir}")
        digest = file_sha256(request_path)
        if digest != canonical_sha256:
            raise ValueError(
                f"requests.json differs for {run.label}: {digest} != {canonical_sha256}"
            )


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[Any, Any]:
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    print(f"[model] loading {args.base_model} ({args.dtype}) on {args.device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    runs = [parse_run_spec(value) for value in args.run]
    if not runs and not args.base_only:
        raise SystemExit("pass at least one --run or use --base-only")
    if len({run.slug for run in runs}) != len(runs):
        raise ValueError("run labels collapse to duplicate output slugs")

    if args.requests_path is None:
        if not runs:
            raise ValueError("--requests-path is required for --base-only without a run")
        args.requests_path = runs[0].source_dir / "requests.json"
    else:
        args.requests_path = args.requests_path.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.wild_eff_gen and args.wild_score_mode != "em":
        raise ValueError("--wild-score-mode is meaningful only with --wild-eff-gen")
    wild_judge: Optional[WildJudge] = None
    if args.wild_eff_gen and args.wild_score_mode == "judge":
        wild_judge = WildJudge(
            model=args.wild_judge_model,
            cache_path=args.output_root / "wild_judge_cache.sqlite3",
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )

    (
        requests,
        raw_requests_sha256,
        normalized_requests_sha256,
        request_provenance,
    ) = load_requests_with_provenance(
        args.requests_path,
        materialize_subject_placeholders=args.materialize_subject_placeholders,
    )
    args.request_provenance = request_provenance
    validate_run_requests(runs, raw_requests_sha256)
    fingerprint = protocol_fingerprint(
        args,
        normalized_requests_sha256,
        raw_requests_sha256,
    )
    steps_by_run = {run.label: discover_steps(run.source_dir, args.steps) for run in runs}
    panel_sizes = sorted(
        {
            step
            for steps in steps_by_run.values()
            for step in steps
            if 0 < step <= len(requests)
        }
    )
    if not panel_sizes:
        panel_sizes = [len(requests)]

    print(
        f"[data] requests={len(requests)} raw_sha256={raw_requests_sha256} "
        "effective_normalized_sha256="
        f"{normalized_requests_sha256} "
        "prompt_materialization="
        f"{request_provenance['prompt_materialization']['mode']} "
        "materialized_efficacy="
        f"{request_provenance['prompt_materialization']['materialized_efficacy_prompts']} "
        f"runs={len(runs)} states={sum(map(len, steps_by_run.values()))} "
        f"wild_eff_gen={args.wild_eff_gen} wild_score={args.wild_score_mode}",
        flush=True,
    )
    model, tokenizer = load_model_and_tokenizer(args)

    base_items, base_payload = load_or_evaluate_base(
        model,
        tokenizer,
        requests,
        panel_sizes,
        args,
        fingerprint,
        raw_requests_sha256,
        wild_judge,
    )
    del base_items

    trajectories: List[Dict[str, Any]] = []
    if not args.base_only:
        parameter_names = select_parameter_names(runs, steps_by_run)
        named_parameters = dict(model.named_parameters())
        missing = [name for name in parameter_names if name not in named_parameters]
        if missing:
            raise KeyError(f"Base model lacks edited parameters: {missing}")
        pristine = {
            name: named_parameters[name].detach().clone()
            for name in parameter_names
        }
        print(f"[model] cached {len(pristine)} pristine edited parameters", flush=True)
        for run in runs:
            trajectories.append(
                evaluate_run(
                    run,
                    model,
                    tokenizer,
                    requests,
                    pristine,
                    args,
                    fingerprint,
                    base_payload,
                    wild_judge,
                )
            )

    write_comparison(args.output_root, base_payload, trajectories, args, fingerprint)
    if wild_judge is not None:
        wild_judge.close()
    print(f"[done] wrote {args.output_root / 'comparison_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
