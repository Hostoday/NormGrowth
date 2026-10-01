"""CPU regression for SPHERE's semantic axis across Linear/Conv1D storage.

Run this file directly; no model weights, dataset, or GPU are needed.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from transformers.pytorch_utils import Conv1D


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "EasyEdit/easyeditor/util/official_editing_baselines.py"
SPEC = importlib.util.spec_from_file_location("_sphere_orientation_under_test", SOURCE)
BASELINES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASELINES
SPEC.loader.exec_module(BASELINES)


class SphereOrientationTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(20260927)

    def pair(self, outputs, inputs):
        weight = torch.randn(outputs, inputs, generator=self.generator)
        update = torch.randn(outputs, inputs, generator=self.generator)
        linear = torch.nn.Linear(inputs, outputs, bias=False)
        conv = Conv1D(outputs, inputs)
        with torch.no_grad():
            linear.weight.copy_(weight)
            conv.weight.copy_(weight.T)
        return linear, conv, weight, update

    @staticmethod
    def svd_reference(weight, update, beta, alpha):
        # Independent primal SVD reference, versus the production dual eigensolve.
        row_unit = weight.double() / weight.double().norm(dim=1, keepdim=True)
        _, singular, vh = torch.linalg.svd(row_unit, full_matrices=False)
        energy = singular.square()
        rank = int(torch.searchsorted(energy.cumsum(0)/energy.sum(), beta)) + 1
        basis = vh[:rank].T
        return update.double() - alpha*(update.double() @ basis) @ basis.T

    def test_conv1d_matches_input_axis_svd_and_transposed_linear(self):
        for outputs, inputs in [(4, 7), (5, 5)]:
            linear, conv, weight, update = self.pair(outputs, inputs)
            expected = self.svd_reference(weight, update, .5, .8)
            actual, stats = BASELINES.sphere_project_update_for_module(
                conv, conv.weight, update.T.contiguous(), beta=.5, alpha=.8)
            linear_result, _ = BASELINES.sphere_project_update_for_module(
                linear, linear.weight, update, beta=.5, alpha=.8)
            torch.testing.assert_close(actual.T.double(), expected, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(actual.T, linear_result, rtol=2e-5, atol=2e-5)
            self.assertTrue(stats["orientation_transposed"])
            self.assertEqual(stats["projection_dimension"], inputs)
            self.assertEqual(stats["projection_weight_shape"], [outputs, inputs])
            # The historical raw-storage right projection is a different operation.
            old_result, _ = BASELINES.sphere_project_update(
                conv.weight, update.T.contiguous(), beta=.5, alpha=.8)
            self.assertGreater(float((actual-old_result).norm()), .01)

    def test_linear_preserves_legacy_results_bitwise(self):
        linear, _, weight, update = self.pair(4, 7)
        for dtype in [torch.float32, torch.bfloat16]:
            for alpha in [0., .5, 1.]:
                reference, reference_stats = BASELINES.sphere_project_update(
                    weight.to(dtype), update.to(dtype), beta=.5, alpha=alpha)
                actual, stats = BASELINES.sphere_project_update_for_module(
                    linear, weight.to(dtype), update.to(dtype), beta=.5, alpha=alpha)
                self.assertTrue(torch.equal(actual, reference))
                self.assertEqual(actual.dtype, dtype)
                self.assertFalse(stats["orientation_transposed"])
                for key, value in reference_stats.items():
                    self.assertEqual(stats[key], value)

    def test_dispatch_records_axis_and_keeps_model_weights_unchanged(self):
        linear, conv, weight, update = self.pair(4, 7)
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([linear, conv])
        updates = {"layers.0.weight": update, "layers.1.weight": update.T.contiguous()}
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "sphere.jsonl"
            hp = SimpleNamespace(sphere_enabled=True, sphere_beta=.5, sphere_alpha=.5,
                                 official_baseline_log_path=str(log))
            result = BASELINES.project_updates_with_sphere(
                model, hp, updates, method="MEMIT",
                parameter_getter=lambda model, name: model.get_parameter(name))
            records = [json.loads(line) for line in log.read_text().splitlines()]
        torch.testing.assert_close(result["layers.0.weight"], result["layers.1.weight"].T)
        self.assertTrue(torch.equal(linear.weight, weight))
        self.assertTrue(torch.equal(conv.weight, weight.T))
        self.assertEqual(len(records), 2)
        for record, transposed in zip(records, [False, True]):
            self.assertEqual(record["projection_axis"], "mlp_input")
            self.assertEqual(record["projection_dimension"], 7)
            self.assertEqual(record["orientation_transposed"], transposed)
            self.assertEqual(record["sphere_orientation_version"], "semantic_output_input_v1")

    def test_unknown_module_fails_instead_of_guessing_from_shape(self):
        with self.assertRaisesRegex(TypeError, "explicit weight-layout adapter"):
            BASELINES.sphere_project_update_for_module(
                torch.nn.Identity(), torch.eye(4), torch.eye(4), beta=.5, alpha=.5)

    def test_disabled_method_does_not_inspect_module_or_parameters(self):
        update = {"arbitrary.weight": torch.eye(2)}
        result = BASELINES.project_updates_with_sphere(
            object(), SimpleNamespace(sphere_enabled=False), update, method="MEMIT",
            parameter_getter=lambda *args: self.fail("Disabled SPHERE accessed a weight"))
        self.assertIs(result, update)


if __name__ == "__main__":
    unittest.main()
