"""Regression checks for the research MPP/VICON evaluation protocols."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from safesvd.adapters.mpp import MPPAdapter
from safesvd.adapters.mpp_runner import parser as mpp_parser
from safesvd.adapters.mpp_runtime import MPPEvaluationOptions, _mpp_batch_factory
from safesvd.adapters.vicon_runtime import (
    ViconBundle, ViconEvaluationOptions, _apply_data_root, _vicon_batch_factory, evaluate_vicon,
)

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:
    OmegaConf = None


class FileDataset(Dataset):
    title = "validation-file"

    def __len__(self):
        return 80

    def get_name(self):
        return "incompNS"

    def __getitem__(self, index):
        return torch.tensor(index), torch.tensor(0), torch.tensor(index)


class TimePairDataset(IterableDataset):
    """Expose which traversal provided each time pair, like VICON's RNG stream."""

    def __init__(self, category, events):
        self.category, self.events = category, events
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        self.events.append((self.category, self.iterations))
        for index in range(2):
            value = float(self.iterations + index)
            initial = torch.full((2, 7, 4, 4), value)
            label = torch.full_like(initial, value + 1)
            yield index, (initial, label, torch.ones(7)), 0, 1, 1


class TinyTrainer:
    loss_cfg = SimpleNamespace(min_ex=1)

    def __init__(self):
        self.model = torch.nn.Identity()

    def _move_to_device(self, pairs):
        return pairs

    def _data_preprocess(self, pairs):
        initial, label, mask = pairs
        shape = (initial.shape[0], 1, 7, 1, 1)
        return (initial, label, mask[:, None, :, None, None]), torch.zeros(shape), torch.ones(shape)

    def _get_label(self, data):
        return data[1]

    def _model_forward(self, data):
        return data[0]

    def get_loss(self, pairs):
        data, _, _ = self._data_preprocess(pairs)
        return float((self._model_forward(data)[:, 1:, :-1] - data[1][:, 1:, :-1]).square().mean())


class EvaluationAlignmentTest(unittest.TestCase):
    def test_mpp_runner_and_dataclass_use_research_defaults(self):
        options = MPPEvaluationOptions(Path("mpp"), Path("model"), Path("data"), None, Path("metrics"), "cpu")
        self.assertEqual((options.batch_size, options.max_batches), (1, 64))
        required = ["evaluate", "--mpp-root", "mpp", "--model-path", "model",
                    "--data-path", "data", "--output-path", "metrics"]
        args = mpp_parser().parse_args(required)
        self.assertEqual((args.batch_size, args.max_batches), (1, 64))
        args = mpp_parser().parse_args(required + ["--batch-size", "3", "--max-batches", "2"])
        self.assertEqual((args.batch_size, args.max_batches), (3, 2))

    def test_public_mpp_route_exposes_effective_defaults_and_overrides(self):
        request = SimpleNamespace(
            repo_root=Path("/repo"), device="cpu", compressed_model=None, output_path=Path("metrics"),
            assets=SimpleNamespace(resolved_model_path=Path("model"), resolved_data_root=Path("data")),
            config=SimpleNamespace(source_root=Path("/mpp"), batch_size=None, max_batches=None, extra=()),
        )
        for batch_size, max_batches in ((None, None), (4, 2)):
            request.config.batch_size, request.config.max_batches = batch_size, max_batches
            command = MPPAdapter().evaluation_command(request)
            self.assertEqual(command[command.index("--batch-size") + 1], str(batch_size or 1))
            self.assertEqual(command[command.index("--max-batches") + 1], str(max_batches or 64))

    def test_mpp_per_file_cap_and_shuffle_match_reference_dataloader(self):
        subset = FileDataset()
        group = SimpleNamespace(get_per_file_dsets=lambda: [subset])
        dataset = SimpleNamespace(sub_dsets=[group], subset_dict={"incompNS": [0, 1]})
        bundle = SimpleNamespace(params=SimpleNamespace(batch_size=1, num_data_workers=0))
        batches = _mpp_batch_factory(bundle, dataset, labels_from=dataset,
                                     calibration_size=None, seed=0, max_batches=64)
        expected = [int(target) for _, _, target in DataLoader(
            subset, batch_size=1, num_workers=0, shuffle=True,
            generator=torch.Generator().manual_seed(0), drop_last=True,
        )][:64]
        for _ in range(2):
            actual = [int(tagged.payload.target) for tagged in batches()]
            self.assertEqual(actual, expected)
            self.assertEqual(len(actual), 64)

    def test_vicon_four_pass_order_and_json_output(self):
        events = []
        categories = ("NS2D", "EULER2D")
        datasets = {name: TimePairDataset(name, events) for name in categories}
        config = SimpleNamespace(datasets={}, dataset_workers=0, test_seed=84)
        bundle = ViconBundle(Path("VICON"), config, TinyTrainer(), lambda *_: datasets, Path("data"), torch.device("cpu"))
        # Use the actual factory, trainer route, metric functions and JSON writer.
        with tempfile.TemporaryDirectory() as directory:
            options = ViconEvaluationOptions(Path("VICON"), Path("model"), Path("data"),
                                             Path("compressed"), Path(directory) / "metrics.json", "cpu", batch_size=1)
            with patch("safesvd.adapters.vicon_runtime.load_vicon_bundle", return_value=bundle), \
                 patch("safesvd.adapters.vicon_runtime.load_compressed_state", side_effect=lambda model, _: model):
                evaluate_vicon(options)
            payload = json.loads(options.output_path.read_text())
        self.assertEqual(events, [(category, traversal) for traversal in range(1, 5) for category in categories])
        for section, energies in (("origin", (4., 9.)), ("compressed", (16., 25.))):
            metrics = payload[section]["metrics"]
            for category in categories:
                self.assertEqual(metrics["losses"][category], 1.)
                self.assertAlmostEqual(metrics["relative_losses"][category], 1 / (sum(energies) / 2))
                self.assertAlmostEqual(metrics["physics"][category]["sobolev"], 1.)

    def test_vicon_max_batches_applies_to_each_dataset_on_both_passes(self):
        events = []
        datasets = {name: TimePairDataset(name, events) for name in ("NS2D", "EULER2D")}
        bundle = ViconBundle(Path("VICON"), SimpleNamespace(datasets={}, dataset_workers=0, test_seed=84),
                             TinyTrainer(), lambda *_: datasets, Path("data"), torch.device("cpu"))
        batches = _vicon_batch_factory(bundle, split="test", batch_size=1, calibration_size=None,
                                       seed=84, random_subset=False, max_batches=1)
        for traversal in (1, 2):
            values = list(batches())
            self.assertEqual([value.category for value in values], ["NS2D", "EULER2D"])
            self.assertEqual(float(values[0].payload[1][0][0, 0, 0, 0, 0]), traversal)

    @unittest.skipIf(OmegaConf is None, "VICON path resolution requires omegaconf")
    def test_vicon_folder_rebasing_matches_main(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = OmegaConf.create({"datasets": {"types": {
                "NS2D": {"folder": "NavierStokes-2D-conditoned/"},
                "EULER2D": {"folder": "../icon-data/2D_Train_Turb/"},
                "COMPRESSIBLE2D": {"folder": "/old/machine/2D_Train_Rand/"},
                "nested": {"path": "sub/data.pt"},
            }}})
            _apply_data_root(config, root)
            for name, relative in (("NS2D", "NavierStokes-2D-conditoned"), ("EULER2D", "2D_Train_Turb"),
                                   ("COMPRESSIBLE2D", "2D_Train_Rand")):
                self.assertEqual(config.datasets.types[name].folder, str(root / relative))
            self.assertEqual(config.datasets.types.nested.path, str(root / "sub/data.pt"))

    @unittest.skipIf(OmegaConf is None, "VICON path resolution requires omegaconf")
    def test_vicon_explicit_path_overrides_take_precedence(self):
        config = OmegaConf.create({"datasets": {"types": {
            "NS2D": {"folder": "/chosen/ns"}, "EULER2D": {"folder": "chosen/euler"},
        }}})
        _apply_data_root(config, Path("/data"), config_overrides=(
            "datasets.types.NS2D.folder=/chosen/ns", "+datasets.types.EULER2D.folder=chosen/euler",
        ))
        self.assertEqual(config.datasets.types.NS2D.folder, "/chosen/ns")
        self.assertEqual(config.datasets.types.EULER2D.folder, str((Path.cwd() / "chosen/euler").resolve()))

    @unittest.skipIf(OmegaConf is None, "VICON path resolution requires omegaconf")
    def test_actual_vicon_yaml_folders_resolve_under_data_root(self):
        source = Path(__file__).resolve().parents[1] / "VICON/configs/datasets/default.yaml"
        config = OmegaConf.create({"datasets": OmegaConf.to_container(OmegaConf.load(source))})
        _apply_data_root(config, Path("/test/converted"))
        for dataset in config.datasets.types.values():
            self.assertEqual(Path(dataset.folder).parent, Path("/test/converted"))
