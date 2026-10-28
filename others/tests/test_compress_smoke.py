"""CPU end-to-end checks using each default model route, without real assets.

Run: python3 -m pytest tests/test_compress_smoke.py -q -s
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required")
class CompressSmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(25)

    def run_flow(self, model, data, ranks, route, options, forward_args):
        from safe_svd.compress import compress

        model.eval()
        before_state = {name: value.clone() for name, value in model.state_dict().items()}
        before_parameters = sum(parameter.numel() for parameter in model.parameters())
        expected_shape = model(*forward_args).shape
        student = compress(
            model, data, ranks, model_type=route, device="cpu",
            route_options=options, alpha=0.7, trace_balance=True,
        )  # No capture_fn override: exercise the actual default router.
        with torch.no_grad():
            output = student(*forward_args)
        after_parameters = sum(parameter.numel() for parameter in student.parameters())
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertLess(after_parameters, before_parameters)
        for name in ranks:
            self.assertIsInstance(student.get_submodule(name), nn.Sequential)
            torch.testing.assert_close(student.get_submodule(name)[1].bias, before_state[f"{name}.bias"])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before_state[name])
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in student.parameters()))
        self.assertTrue(all(not module._forward_pre_hooks for module in model.modules()))
        print(json.dumps({"route": route, "parameters_before": before_parameters,
                          "parameters_after": after_parameters, "output_shape": list(output.shape),
                          "finite": True}, sort_keys=True))
        return student

    def test_mpp_default_route(self):
        class FileDataset(torch.utils.data.Dataset):
            values = torch.randn(6, 2, 4, 2, 2)
            def get_name(self):
                return "incompNS"
            def __len__(self):
                return len(self.values)
            def __getitem__(self, index):
                return self.values[index], torch.zeros(1), torch.zeros_like(self.values[index])
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.first, self.second = nn.Linear(4, 4), nn.Linear(4, 4)
            def forward(self, inp, labels, bcs):
                return self.second(torch.relu(self.first(inp.permute(0, 1, 3, 4, 2))))
        subset = FileDataset()
        dataset = SimpleNamespace(sub_dsets=[SimpleNamespace(get_per_file_dsets=lambda: [subset])],
                                  subset_dict={"incompNS": [0, 1]})
        args = (subset.values[:2].permute(1, 0, 2, 3, 4), torch.tensor([[0, 1], [0, 1]]), torch.zeros(2, 1))
        self.run_flow(Model(), dataset, {"first": 1, "second": 1}, "mpp", {
            "params": SimpleNamespace(batch_size=2, num_data_workers=0),
            "batches_per_dset": 2, "calibration_seed": 7,
        }, args)

    def test_vicon_default_route(self):
        model = nn.Sequential(nn.Linear(8, 12), nn.GELU(), nn.Linear(12, 8))
        class Trainer:
            def __init__(self, model):
                self.model = model
            def _move_to_device(self, pairs):
                return pairs.to("cpu")
            def _data_preprocess(self, pairs):
                return pairs * 0.5, None, None
            def _model_forward(self, data):
                return self.model(data)
        def loader():
            batches = [(index, torch.randn(8), 0., 1., 1.) for index in range(8)]
            return torch.utils.data.DataLoader(batches, batch_size=4, shuffle=False)
        trainer = Trainer(model)
        self.run_flow(model, {"NS": loader(), "SWE": loader()}, {"0": 2, "2": 2}, "vicon", {
            "trainer": trainer, "max_calib_batches": 2,
        }, (torch.randn(3, 8),))
        self.assertIs(trainer.model, model)

    def test_poseidon_default_route_with_mlp_cache(self):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.pwconv1, self.pwconv2, self.act = nn.Linear(8, 12), nn.Linear(12, 8), nn.GELU()
            def forward(self, x):
                return self.pwconv2(self.act(self.pwconv1(x)))
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(name_or_path="smoke")
                self.residual_blocks = nn.ModuleList([nn.ModuleList([Block()])])
            def forward(self, x):
                return self.residual_blocks[0][0](x)
        class Trainer:
            def __init__(self, model):
                self.model, self.ar_steps, self.output_all_steps = model, 3, True
                self.compute_metrics = object()
                self.rollouts = 0
            def set_ar_steps(self, steps, output_all_steps=False):
                self.ar_steps, self.output_all_steps = steps, output_all_steps
            def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
                return model(inputs[0])
        def rollout(trainer, dataset, model_path=""):
            trainer.rollouts += 1
            for inputs in torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False):
                trainer.prediction_step(trainer.model, inputs, False)
        model = Model()
        trainer = Trainer(model)
        values = torch.randn(12, 8)
        self.run_flow(model, torch.utils.data.TensorDataset(values), {
            "residual_blocks.0.0.pwconv1": 2, "residual_blocks.0.0.pwconv2": 2,
        }, "poseidon", {"trainer": trainer, "rollout_fn": rollout,
                        "capture_batch_size": 4, "offload_cache": True}, (values,))
        self.assertEqual(trainer.rollouts, 6)  # First layer only; second uses its cache.
        self.assertIs(trainer.model, model)
        self.assertEqual(trainer.ar_steps, 3)
        self.assertTrue(trainer.output_all_steps)


if __name__ == "__main__":
    unittest.main()
