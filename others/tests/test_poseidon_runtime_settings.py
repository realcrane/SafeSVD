from __future__ import annotations

import types
import unittest

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for Poseidon runtime tests")
class PoseidonRuntimeSettingsTest(unittest.TestCase):
    def test_official_factory_is_forced_to_origin_safe_loader_settings(self) -> None:
        from safesvd.adapters.poseidon_runtime import _build_memory_safe_trainer

        trainer = types.SimpleNamespace(args=types.SimpleNamespace())
        calls: list[dict[str, object]] = []

        def official_factory(**kwargs):
            calls.append(kwargs)
            return trainer

        dataset = object()
        result = _build_memory_safe_trainer(official_factory, model_path="/checkpoint", batch_size=256, dataset=dataset)

        self.assertIs(result, trainer)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model_path"], "/checkpoint")
        self.assertEqual(calls[0]["batch_size"], 256)
        self.assertIs(calls[0]["dataset"], dataset)
        self.assertEqual(calls[0]["workers"], 1)
        self.assertEqual(trainer.args.dataloader_num_workers, 0)
        self.assertFalse(trainer.args.remove_unused_columns)
