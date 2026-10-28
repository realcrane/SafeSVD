"""Exercise the bundled research architecture through the unified pipeline."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from VICON.src.models import ICON_CROPPED


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def model_config():
    return Config(transformer=Config(dim_channel=7, dim_token=8, nhead=2,
                                    dim_feedforward=16, num_layers=1,
                                    dropout=0., norm_first=False),
                  demo_num=2, patch_num_in=2, patch_num_out=2, patch_resolution=2,
                  use_patch_pos_encoding=True, use_func_pos_encoding=True)


class TinyTrainer:
    def __init__(self, model):
        self.model = model
        self.loss_cfg = SimpleNamespace(min_ex=1)

    def _move_to_device(self, pairs):
        return [value.to(next(self.model.parameters()).device) for value in pairs]

    def _data_preprocess(self, pairs):
        init, end, mask = pairs
        return (init * 0.5, end * 0.5, mask[:, None, :, None, None]), 0., 2.

    def _model_forward(self, data):
        return self.model(data)


class VendoredViconTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)

    def test_default_architecture_matches_historical_checkpoint_shapes(self):
        root = Path(__file__).resolve().parents[1]
        values = yaml.safe_load((root / "VICON/configs/model/default.yaml").read_text())
        self.assertEqual(values["type"], "crop")
        values["transformer"] = Config(values["transformer"])
        with torch.device("meta"):
            model = ICON_CROPPED(Config(values))
        shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()
                  if "pos_encoding" in name}
        self.assertEqual(shapes, {
            "patch_pos_encoding_init": (64, 1024), "patch_pos_encoding_end": (64, 1024),
            "func_pos_encoding_init": (10, 1024), "func_pos_encoding_end": (10, 1024),
        })
        self.assertEqual(sum(p.numel() for p in model.parameters()), 87823104)

    def test_research_forward_matches_non_inplace_fisher_forward(self):
        from safe_svd.helper.collector.vicon_fisher_forward import _patched_forward
        for patch_num_out in (2, 4):
            config = model_config()
            config.patch_num_in = 4
            config.patch_num_out = patch_num_out
            model = ICON_CROPPED(config).eval()
            size = patch_num_out * config.patch_resolution
            data = (torch.randn(1, 2, 7, 8, 8), torch.randn(1, 2, 7, size, size), torch.ones(1, 7))
            with torch.no_grad():
                expected = model(data)
                with _patched_forward(model):
                    actual = model(data)
            self.assertEqual(tuple(actual.shape), (1, 2, 7, size, size))
            torch.testing.assert_close(actual, expected)

    def test_real_cropped_model_input_fisher_rank_compression_and_reload(self):
        from safe_svd.main import main
        model = ICON_CROPPED(model_config()).eval()
        trainer = TinyTrainer(model)
        records = []
        for index in range(4):
            init, end = torch.randn(2, 7, 4, 4), torch.randn(2, 7, 4, 4)
            init[:, -1] = 0
            end[:, -1] = 0
            records.append((index, (init, end, torch.tensor([1.] * 6 + [0.])), 0., 1., 1.))
        loader = torch.utils.data.DataLoader(records, batch_size=2, shuffle=False)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            student = main(model, {"NS2D": loader}, model_type="vicon", rank_ratio=0.5,
                           rank_selector="legacy", device="cpu", trainer=trainer,
                           max_calib_batches=1, train_seed=42, verbose=False,
                           profile_cache=directory / "input.pt", fisher_cache=directory / "fisher.pt",
                           rank_cache=directory / "ranks.json", save_path=directory / "compressed")
            self.assertTrue((directory / "fisher.pt").is_file())
            state = torch.load(directory / "compressed/compressed_model.pth", weights_only=True)
            reloaded = copy.deepcopy(student)
            reloaded.load_state_dict(state, strict=True)
            data, _, _ = trainer._data_preprocess(next(iter(loader))[1])
            with torch.no_grad():
                actual, expected = reloaded(data), student(data)
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, expected)
            self.assertTrue(any(isinstance(layer, torch.nn.Sequential) for layer in student.modules()))
            self.assertFalse((directory / "compressed/compressing_model.pt").exists())
