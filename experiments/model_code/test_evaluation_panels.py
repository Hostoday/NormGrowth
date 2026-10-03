"""CPU checks for editing-request selection and fixed locality coverage.

Run directly in the normgrowth environment. No weights or dataset are needed.
"""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_evaluator(filename):
    name = "_panel_test_" + filename
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "evaluate" / (filename + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HF = load_evaluator("eval_hf_easyedit")
CUMULATIVE = load_evaluator("eval_cumulative_generation_locality")


class PanelSelectionTests(unittest.TestCase):
    def hf_args(self, *options):
        return HF.build_parser().parse_args([
            "--model_path", "unused-model", "--data_path", "unused.json", *options,
        ])

    def test_default_uses_first_1000_requests_in_editing_order(self):
        args = self.hf_args()
        requests = [{"case_id": index} for index in range(1100)]
        selected = HF.select_evaluation_data(requests, args)
        self.assertEqual(args.batch_size, 1)
        self.assertEqual([row["case_id"] for row in selected], list(range(1000)))

    def test_saved_requests_are_neither_truncated_nor_resampled(self):
        args = self.hf_args(
            "--requests_path", "requests.json", "--num_samples", "1",
            "--selection", "random",
        )
        requests = [{"case_id": index} for index in (7, 1, 4)]
        self.assertIs(HF.select_evaluation_data(requests, args), requests)

    def test_random_selection_is_explicit_repeatable_and_has_separate_cache(self):
        prefix_args = self.hf_args("--pre_cache_dir", "cache", "--num_samples", "10")
        random_args = self.hf_args(
            "--pre_cache_dir", "cache", "--num_samples", "10", "--selection", "random",
        )
        requests = [{"case_id": index} for index in range(100)]
        selected = HF.select_evaluation_data(requests, random_args)
        self.assertEqual(selected, HF.select_evaluation_data(requests, random_args))
        self.assertNotEqual(selected, HF.select_evaluation_data(requests, prefix_args))
        self.assertNotEqual(
            HF.resolve_pre_cache_path(prefix_args, "base"),
            HF.resolve_pre_cache_path(random_args, "base"),
        )

    def test_every_locality_group_and_prompt_survives_normalization(self):
        record = {
            "prompt": "rewrite", "target_new": "answer",
            "locality": {
                "neighborhood": {"prompt": ["loc-a", "loc-b"], "ground_truth": "old"},
                "other": {"prompt": "loc-c", "ground_truth": "old-c"},
            },
        }
        parsed = HF.parse_record(record, 0)
        pairs = HF.collect_group_items([HF.build_easyedit_request(parsed)], "locality")
        self.assertEqual([pair[2] for pair in pairs], ["loc-a", "loc-b", "loc-c"])
        normalized = CUMULATIVE.normalize_request(record, 0)
        self.assertEqual(
            [pair["prompt"] for pair in normalized["locality_pairs"]],
            ["loc-a", "loc-b", "loc-c"],
        )

    def test_locality_defaults_to_all_selected_requests_with_explicit_override(self):
        args = CUMULATIVE.parse_args(["--output-root", "unused"])
        self.assertEqual(args.generation_batch_size, 1)
        self.assertEqual(args.locality_batch_size, 1)
        self.assertEqual(CUMULATIVE.locality_panel_count(args, 1000), 1000)
        self.assertEqual(CUMULATIVE.locality_panel_count(args, 1010), 1010)
        args.locality_panel_size = 20
        self.assertEqual(CUMULATIVE.locality_panel_count(args, 1000), 20)

    def test_checkpoint_prefix_changes_eff_gen_but_keeps_all_locality(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=True):
                return [1, 2]

        requests = [
            CUMULATIVE.normalize_request({
                "case_id": index,
                "prompt": f"rewrite-{index}", "target_new": "answer",
                "rephrase_prompt": [f"rephrase-a-{index}", f"rephrase-b-{index}"],
                "locality": {"neighborhood": {
                    "prompt": [f"locality-a-{index}", f"locality-b-{index}"],
                    "ground_truth": "old",
                }},
            }, index)
            for index in range(4)
        ]
        args = CUMULATIVE.parse_args(["--output-root", "unused"])
        observed_generation_prompts = []
        observed_locality_prompts = []

        def generate(model, tokenizer, prompts, **kwargs):
            observed_generation_prompts[:] = prompts
            return ["answer"] * len(prompts)

        def score_locality(model, tokenizer, examples, **kwargs):
            observed_locality_prompts[:] = [example.prompt for example in examples]
            return [{"token_correct": [True]} for example in examples]

        with patch.object(CUMULATIVE, "greedy_continuations", generate), patch.object(
            CUMULATIVE, "score_teacher_forced_examples", score_locality
        ):
            for step in (1, 2, 4):
                _, summary = CUMULATIVE.evaluate_state(
                    None, Tokenizer(), requests,
                    generation_count=step,
                    locality_count=CUMULATIVE.locality_panel_count(args, len(requests)),
                    max_new_tokens=12, generation_batch_size=1,
                    locality_batch_size=1, easyedit_max_length=256,
                )
                self.assertEqual(len(observed_generation_prompts), 3 * step)
                self.assertEqual(observed_generation_prompts[-1], f"rephrase-b-{step-1}")
                self.assertEqual(len(observed_locality_prompts), 8)
                self.assertEqual(observed_locality_prompts[-1], "locality-b-3")
                self.assertEqual(summary["efficacy_case_count"], step)
                self.assertEqual(summary["generalization_case_count"], step)
                self.assertEqual(summary["specificity_case_count"], 4)


if __name__ == "__main__":
    unittest.main()
