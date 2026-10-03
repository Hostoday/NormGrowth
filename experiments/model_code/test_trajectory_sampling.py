"""Check trajectory cohorts and checkpoint scoring without loading a model.

The runner imports GPU/model diagnostics at module scope. Extract its pure
selection and orchestration functions so these tests need only Python's stdlib.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "diagnostics/analyze_edit_count_gain_trajectory.py"


def runner_functions():
    names = {
        "parse_csv_strings", "parse_steps", "build_parser", "add_oedit_arguments",
        "canonical_json", "analysis_request_fingerprint", "locality_request_count",
        "select_trajectory_requests", "validate_analysis_panel", "evaluate_step",
    }
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes,
        type_ignores=[],
    )
    namespace = {
        "argparse": argparse, "hashlib": hashlib, "json": json, "random": random,
        "DATA_ROOT": ROOT.parent / "data",
        "parse_int_list": lambda text: [int(part) for part in text.split(",")],
        "parse_post_update_nodes": lambda text: text.split(","),
        "parse_fixed_probe_contexts": lambda text: text.split(","),
        "FIXED_PROBE_DEFAULT_CONTEXTS": (),
        "add_early_attention_arguments": lambda parser: None,
        "add_o0_axis_arguments": lambda parser: None,
        "apply_manifest_order": lambda requests, order: [requests[index] for index in order],
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace


class TrajectorySamplingTests(unittest.TestCase):
    def setUp(self):
        self.runner = runner_functions()
        self.source = [
            {
                "case_id": index, "prompt": f"edit {index}", "target_new": "target",
                "locality": {"neighbors": {
                    "prompt": [f"locality {index}a", f"locality {index}b"],
                    "ground_truth": ["a", "b"],
                }},
            }
            for index in range(1200)
        ]

    def select(self, **overrides):
        args = dict(source_requests=self.source, sample_size=1000,
                    locality_eval_prompts=None, edit_order="prefix", seed=42)
        args.update(overrides)
        return self.runner["select_trajectory_requests"](**args)

    def test_cli_defaults_are_1000_edits_batch_one_and_all_locality(self):
        args = self.runner["build_parser"]().parse_args(["--output-dir", "build/toy"])
        self.assertEqual((args.sample_size, args.batch_size), (1000, 1))
        self.assertIsNone(args.locality_eval_prompts)
        self.assertEqual(self.runner["locality_request_count"](
            args.sample_size, args.locality_eval_prompts), 1000)
        self.assertEqual(args.steps, [0, 50, 100, 150, 200, 250, 300, 500, 750, 1000])
        self.assertEqual(Path(args.data_path), ROOT.parent / "data/zsre/zsre_3k.json")

    def test_all_locality_belongs_to_selected_prefix_and_keeps_every_prompt(self):
        edits, locality = self.select()
        self.assertEqual([item["case_id"] for item in edits], list(range(1000)))
        self.assertEqual(locality, self.source[:1000])
        self.assertEqual(sum(len(item["locality"]["neighbors"]["prompt"]) for item in locality), 2000)

    def test_shuffle_and_manifest_change_edit_order_but_not_locality_panel(self):
        edits, locality = self.select(edit_order="shuffle")
        self.assertNotEqual(edits, self.source[:1000])
        self.assertEqual(sorted(item["case_id"] for item in edits), list(range(1000)))
        self.assertEqual(locality, self.source[:1000])
        edits, locality = self.select(edit_order_permutation=list(reversed(range(1000))))
        self.assertEqual([item["case_id"] for item in edits], list(reversed(range(1000))))
        self.assertEqual(locality, self.source[:1000])

    def test_explicit_legacy_limit_stays_within_selected_cohort(self):
        _, locality = self.select(edit_order="shuffle", locality_eval_prompts=50)
        self.assertEqual(locality, self.source[:50])
        for invalid in (0, -1, 1001):
            with self.assertRaisesRegex(ValueError, "sample-size"):
                self.select(locality_eval_prompts=invalid)

    def test_short_source_cannot_silently_reduce_edit_or_locality_counts(self):
        with self.assertRaisesRegex(ValueError, "only 999 valid requests"):
            self.select(source_requests=self.source[:999])

    def test_resume_rejects_old_50_case_panel_or_changed_membership(self):
        validate = self.runner["validate_analysis_panel"]
        path = Path("build/toy/analysis_requests.json")
        expected = self.source[:1000]
        with self.assertRaisesRegex(ValueError, "50 saved requests; 1000 required"):
            validate(expected[:50], expected, path)
        with self.assertRaisesRegex(ValueError, "requested locality panel"):
            validate(self.source[1:1001], expected, path)
        saved_with_eos = [dict(item, target_new="target<eos>") for item in expected]
        validate(saved_with_eos, expected, path)

    def test_checkpoint_scores_all_edited_cases_and_the_same_full_locality_panel(self):
        selected, locality = self.select()
        seen = {}

        def scorer(group):
            def score(model, name, hparams, tokenizer, requests, *args):
                seen[group] = [item["case_id"] for item in requests]
                return [{"case_id": item["case_id"]} for item in requests], 1.0
            return score

        for group in ("locality", "rewrite", "rephrase"):
            self.runner[f"compute_{group}_evaluation"] = scorer(group)
        for step in (0, 250, 1000):
            with self.subTest(step=step):
                result = self.runner["evaluate_step"](
                    step, None, "toy", None, None, selected, locality, {}, 8, False
                )
                self.assertEqual(seen["rewrite"], list(range(step)))
                self.assertEqual(seen["rephrase"], list(range(step)))
                self.assertEqual(seen["locality"], list(range(1000)))
                self.assertEqual(result["summary"]["rewrite_count"], step)
                self.assertEqual(result["summary"]["rephrase_count"], step)
                self.assertEqual(result["summary"]["locality_count"], 1000)


if __name__ == "__main__":
    unittest.main()
