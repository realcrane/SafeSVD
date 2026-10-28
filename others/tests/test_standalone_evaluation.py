"""Verify evaluator numerics, checkpoint loading and independence from compression."""
from __future__ import annotations

import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


class ViconTrainer:
    loss_cfg = SimpleNamespace(min_ex=1)

    def __init__(self):
        self.model = torch.nn.Identity()

    def _move_to_device(self, pairs):
        return pairs

    def _data_preprocess(self, pairs):
        x, y, mask = pairs
        shape = (x.shape[0], 1, 7, 1, 1)
        return (x, y, mask[:, None, :, None, None]), torch.zeros(shape), torch.ones(shape)

    def _get_label(self, data):
        return data[1]

    def _model_forward(self, data):
        return self.model(data[0])

    def get_loss(self, pairs):
        data, _, _ = self._data_preprocess(pairs)
        return (self._model_forward(data)[:, 1:, :-1] - data[1][:, 1:, :-1]).square().mean()


class ViconDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        yield 0, (torch.ones(2, 7, 4, 4), torch.full((2, 7, 4, 4), 2.), torch.ones(7)), 0, 1, 1


class MppDataset(torch.utils.data.Dataset):
    title = 'tiny-ns'

    def __len__(self):
        return 3

    def get_name(self):
        return 'incompNS'

    def __getitem__(self, index):
        return torch.ones(2, 2, 4, 4), torch.zeros(1), torch.full((2, 4, 4), 2.)


class MppModel(torch.nn.Module):
    def forward(self, history, labels, boundary):
        return history[-1] * .5


def blocked_import_smoke():
    class BlockCompression(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'safesvd', 'safe_svd'}:
                raise AssertionError(f'Compression dependency: {fullname}')
    sys.meta_path.insert(0, BlockCompression())
    from evaluation.vicon import evaluate as vicon_entry, runtime as vicon
    from evaluation.mpp import evaluate as mpp_entry, runtime as mpp
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        v_checkpoint = directory / 'vicon.pth'
        torch.save({'state_dict': {}}, v_checkpoint)
        bundle = vicon.ViconBundle(ROOT/'VICON', SimpleNamespace(datasets={}, dataset_workers=0, test_seed=84),
                                   ViconTrainer(), lambda *_: {'NS2D': ViconDataset()}, directory, torch.device('cpu'))
        with patch.object(vicon, 'load_vicon_bundle', return_value=bundle):
            vicon_entry.main(['--checkpoint', str(v_checkpoint), '--model', str(v_checkpoint),
                              '--data-root', str(directory), '--device', 'cpu', '--batch-size', '1',
                              '--output', str(directory/'vicon.json')])
        d = SimpleNamespace(sub_dsets=[SimpleNamespace(get_per_file_dsets=lambda: [MppDataset()])],
                            subset_dict={'incompNS': [0, 1]})
        bundle = mpp.MPPBundle(ROOT, SimpleNamespace(batch_size=1, num_data_workers=0), MppModel(), d, d, torch.device('cpu'))
        m_checkpoint = directory / 'mpp.pt'
        torch.save(MppModel(), m_checkpoint)
        with patch.object(mpp, 'load_mpp_bundle', return_value=bundle):
            mpp_entry.main(['--checkpoint', str(m_checkpoint), '--model', str(m_checkpoint),
                            '--data-root', str(directory), '--device', 'cpu', '--max-batches', '2',
                            '--output', str(directory/'mpp.json')])
        assert json.loads((directory/'vicon.json').read_text())['compressed']['metrics']['losses']['NS2D'] == 1.
        assert json.loads((directory/'mpp.json').read_text())['compressed']['metrics']['datasets']['tiny-ns']['mse'] == 2.25
    assert not any(k.split('.')[0] in {'safesvd', 'safe_svd'} for k in sys.modules)


class StandaloneEvaluationTests(unittest.TestCase):
    def test_both_evaluators_run_with_compression_imports_blocked(self):
        code = f"import sys;sys.path[:0]=[{str(ROOT)!r},{str(ROOT/'tests')!r}];from test_standalone_evaluation import blocked_import_smoke;blocked_import_smoke()"
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=tempfile.gettempdir(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_physical_metrics_match_previous_implementation_all_categories(self):
        from evaluation.vicon import physics as vicon
        from evaluation.mpp import physics as mpp
        from safesvd.adapters import vicon_physics, mpp_physics
        torch.manual_seed(7)
        for current, previous, channels in ((vicon, vicon_physics, 6), (mpp, mpp_physics, 4)):
            prediction, target, initial = [torch.randn(2, channels, 8, 8) for _ in range(3)]
            for category in current.SOBOLEV_ORDERS:
                args = (category, prediction, target, initial) if current is mpp else (category, prediction, target)
                self.assertEqual(current.physics_metrics(*args), previous.physics_metrics(*args))

    def test_low_rank_checkpoint_rebuild_is_exact(self):
        from evaluation.mpp import checkpoints as mpp
        from evaluation.vicon import checkpoints as vicon
        torch.manual_seed(8)
        compressed = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(5, 2, bias=False), torch.nn.Linear(2, 3)))
        x = torch.randn(4, 5)
        with tempfile.TemporaryDirectory() as directory:
            for index, payload in enumerate((compressed, {'state_dict': compressed.state_dict()}, compressed.state_dict())):
                path = Path(directory)/f'model{index}.pt'
                torch.save(payload, path)
                for module in (mpp, vicon):
                    rebuilt = module.load_compressed_state(torch.nn.Sequential(torch.nn.Linear(5, 3)), path)
                    self.assertTrue(torch.equal(rebuilt(x), compressed(x)))


if __name__ == '__main__':
    unittest.main()
