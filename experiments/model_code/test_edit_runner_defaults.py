"""CPU checks of the public runner's defaults and cumulative evaluation wiring.

The integration test executes the real CLI/main loop with synthetic requests
and mocked model inference. It never loads model weights or accesses a GPU.
"""
from collections import defaultdict
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_rgr_batch as runner


def toy_requests(count):
    return [
        {
            "case_id": index, "subject": f"subject{index}",
            "prompt": "{} has which property?", "target_new": "new target",
            "rephrase_prompt": "What is the property of {}?",
            "locality": {"neighbors": {
                "prompt": [f"neighbor {index}a?", f"neighbor {index}b?"],
                "ground_truth": ["answer a", "answer b"],
            }},
        }
        for index in range(count)
    ]


class EditRunnerDefaultsTests(unittest.TestCase):
    def test_cli_defaults_select_1000_in_order_with_batch_one(self):
        with mock.patch.object(sys, "argv", [
            "run_rgr_batch", "--hparams_path", "toy.yaml", "--data_path", "toy.json"
        ]):
            args = runner.parse_args()
        self.assertEqual((args.sample_size, args.batch_size, args.eval_batch_size), (1000, 1, 1))
        self.assertEqual((args.selection, args.seed), ("prefix", 42))

    def test_prefix_excludes_unselected_requests_and_rejects_short_source(self):
        source = toy_requests(1100)
        selected = runner.maybe_sample_requests(source, 1000, 42)
        self.assertEqual([item["case_id"] for item in selected], list(range(1000)))
        self.assertEqual(len(source), 1100)
        with self.assertRaisesRegex(ValueError, "only 999 usable requests"):
            runner.maybe_sample_requests(source[:999], 1000, 42)

    def test_normalization_preserves_all_nested_locality_pairs(self):
        source = toy_requests(1)[0]
        normalized = runner.build_request(source, 0)
        self.assertEqual(normalized["locality"], source["locality"])
        counterfact = {
            "case_id": 7,
            "requested_rewrite": {
                "prompt": "{} lives in", "subject": "Person",
                "target_new": {"str": "City A"}, "target_true": {"str": "City B"},
            },
            "neighborhood_prompts": ["Neighbor one lives in", "Neighbor two lives in"],
        }
        normalized = runner.build_request(counterfact, 0)
        self.assertEqual(normalized["locality"]["neighborhood"], {
            "prompt": counterfact["neighborhood_prompts"],
            "ground_truth": ["City B", "City B"],
        })

    def test_main_captures_base_once_and_evaluates_growing_prefix_with_fixed_locality(self):
        from diagnostics import analyze_edit_count_gain_trajectory as trajectory

        expected_steps = [50, 100, 150, 200, 250, 300, 500, 750, 1000]
        locality_calls = []
        rewrite_calls = []
        edit_calls = []

        class ToyModel:
            edit_count = 0

            def eval(self):
                return self

        model = ToyModel()
        hparams = SimpleNamespace(alg_name="MEMIT", model_name="toy", device=0,
                                  batch_size=1, layers=[0], append_eos_to_target=False)

        def apply_algo(current_model, tokenizer, requests, current_hparams, **kwargs):
            edit_calls.append([request["case_id"] for request in requests])
            current_model.edit_count += len(requests)
            return current_model, {}

        editor = SimpleNamespace(model=model, tok=object(), hparams=hparams,
                                 model_name="toy", apply_algo=apply_algo)

        def locality_outputs(**kwargs):
            items = kwargs["locality_items"]
            locality_calls.append((model.edit_count, list(items)))
            outputs = defaultdict(lambda: defaultdict(list))
            # Every edited state differs from Base, but equals other edited states.
            # Thus agreement must remain zero if Base is the actual reference.
            token = int(model.edit_count > 0)
            for index, key, prompt, target in items:
                outputs[index][key].append([token])
            return outputs

        def rewrite_scores(**kwargs):
            rewrite_calls.append((model.edit_count, list(kwargs["prompts"])))
            return [1.0] * len(kwargs["prompts"])

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "requests.json"
            source.write_text(json.dumps(toy_requests(1100)), encoding="utf-8")
            hparams_path = directory / "toy.yaml"
            hparams_path.write_text("alg_name: MEMIT\n", encoding="utf-8")
            output = directory / "run"
            argv = ["run_rgr_batch", "--hparams_path", str(hparams_path),
                    "--data_path", str(source), "--output_dir", str(output),
                    "--do_eval", "--save_model", "0", "--append_eos_to_target", "0"]
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(sys, "argv", argv))
                stack.enter_context(mock.patch.object(runner, "init_easyedit_imports"))
                stack.enter_context(mock.patch.object(runner, "fix_seed"))
                stack.enter_context(mock.patch.object(runner, "HPARAMS_REGISTRY", {
                    "MEMIT": SimpleNamespace(from_hparams=lambda path: hparams),
                }))
                stack.enter_context(mock.patch.object(runner, "BaseEditor", SimpleNamespace(
                    from_hparams=lambda params: editor)))
                stack.enter_context(mock.patch.object(runner, "BatchEditor", SimpleNamespace(
                    is_batchable_method=lambda method: True)))
                stack.enter_context(mock.patch.object(trajectory, "compute_locality_outputs", locality_outputs))
                stack.enter_context(mock.patch.object(trajectory, "compute_rewrite_scores", rewrite_scores))
                stack.enter_context(redirect_stdout(io.StringIO()))
                runner.main()

            self.assertEqual(edit_calls, [[index] for index in range(1000)])
            self.assertEqual([step for step, _ in locality_calls], [0] + expected_steps)
            for _, items in locality_calls:
                self.assertEqual(len(items), 2000)
                self.assertEqual({index for index, _, _, _ in items}, set(range(1000)))
            self.assertEqual(len(rewrite_calls), 2 * len(expected_steps))
            for position, step in enumerate(expected_steps):
                efficacy = rewrite_calls[2 * position]
                generalization = rewrite_calls[2 * position + 1]
                self.assertEqual(efficacy, (step, [
                    f"subject{index} has which property?" for index in range(step)
                ]))
                self.assertEqual(generalization, (step, [
                    f"What is the property of subject{index}?" for index in range(step)
                ]))
                result = json.loads((output / "evaluations" / f"step_{step:04d}.json").read_text())
                self.assertEqual(result["summary"]["rewrite_count"], step)
                self.assertEqual(result["summary"]["rephrase_count"], step)
                self.assertEqual(result["summary"]["locality_count"], 1000)
                self.assertEqual(result["summary"]["locality_acc"], 0.0)
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["num_requests_used"], 1000)
            self.assertEqual(manifest["locality_panel_requests"], 1000)
            self.assertEqual(manifest["eval_steps"], expected_steps)


if __name__ == "__main__":
    unittest.main()
