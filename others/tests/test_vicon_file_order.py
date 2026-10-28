"""A data copy must replay research file order without source-tree access."""

import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from safe_svd.helper.collector.vicon_file_order import apply_vicon_file_order


def dataset(folder, names):
    for name in names:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return SimpleNamespace(
        cfg=SimpleNamespace(types={"NS2D": SimpleNamespace(folder=str(folder))}),
        dp=[str(folder / name) for name in names], rng=None, tc_rng=None,
    )


def manifest(root, orders):
    path = root / "order.json"
    path.write_text(json.dumps({"version": 1, "splits": {"train": orders}}))
    return path


class ViconFileOrderTests(unittest.TestCase):
    def test_independent_copy_replays_file_and_rng_stream(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            research_order = ["train/10.h5", "train/2.h5", "train/1.h5"]
            copied = dataset(root / "local", list(reversed(research_order)))
            order_path = manifest(root, {"NS2D": research_order})
            actual_hash = apply_vicon_file_order({"NS2D": copied}, order_path)
            self.assertEqual(actual_hash, hashlib.sha256(order_path.read_bytes()).hexdigest())
            self.assertEqual(copied.dp, [str(root / "local" / name) for name in research_order])
            # In-place shuffles over successive passes make initial order
            # observable even when the seed and file set match exactly.
            expected = list(research_order)
            wanted_rng, actual_rng = random.Random(42), random.Random(42)
            for _ in range(3):
                wanted_rng.shuffle(expected)
                actual_rng.shuffle(copied.dp)
                wanted = [(name, wanted_rng.randrange(5)) for name in expected]
                actual = [(Path(name).relative_to(root / "local").as_posix(), actual_rng.randrange(5))
                          for name in copied.dp]
                self.assertEqual(actual, wanted)

    def test_missing_or_extra_files_do_not_change_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copied = dataset(root / "local", ["b.h5", "a.h5"])
            before = copied.dp[:]
            for names in (["a.h5"], ["a.h5", "b.h5", "missing.h5"]):
                with self.subTest(names=names), self.assertRaisesRegex(ValueError, "set mismatch"):
                    apply_vicon_file_order({"NS2D": copied}, manifest(root, {"NS2D": names}))
                self.assertEqual(copied.dp, before)

    def test_rejects_absolute_traversal_noncanonical_and_duplicate_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copied = dataset(root / "local", ["a.h5"])
            for names in (["/source/a.h5"], ["../a.h5"], ["./a.h5"], ["x//a.h5"],
                          ["x\\a.h5"], [""], [None], ["a.h5", "a.h5"]):
                with self.subTest(names=names), self.assertRaises(ValueError):
                    apply_vicon_file_order({"NS2D": copied}, manifest(root, {"NS2D": names}))

    def test_validates_all_categories_before_changing_any_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = dataset(root / "first", ["b.h5", "a.h5"])
            second = dataset(root / "second", ["a.h5"])
            second.cfg.types = {"EULER2D": SimpleNamespace(folder=str(root / "second"))}
            before = first.dp[:]
            order_path = manifest(root, {"NS2D": ["a.h5", "b.h5"], "EULER2D": ["missing.h5"]})
            with self.assertRaisesRegex(ValueError, "set mismatch"):
                apply_vicon_file_order({"NS2D": first, "EULER2D": second}, order_path)
            self.assertEqual(first.dp, before)

    def test_rejects_mismatched_category_order_or_already_started_stream(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copied = dataset(root / "local", ["a.h5"])
            with self.assertRaisesRegex(ValueError, "categories/order"):
                apply_vicon_file_order({"NS2D": copied}, manifest(root, {"EULER2D": ["a.h5"]}))
            order_path = manifest(root, {"NS2D": ["a.h5"]})
            copied.rng = random.Random(1)
            with self.assertRaisesRegex(ValueError, "before dataset iteration"):
                apply_vicon_file_order({"NS2D": copied}, order_path)

    def test_rejects_paths_outside_the_local_folder_or_missing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copied = dataset(root / "local", ["a.h5"])
            order_path = manifest(root, {"NS2D": ["a.h5"]})
            copied.dp = [str(root / "source/a.h5")]
            with self.assertRaisesRegex(ValueError, "outside"):
                apply_vicon_file_order({"NS2D": copied}, order_path)
            copied.dp = [str(root / "local/a.h5")]
            Path(copied.dp[0]).unlink()
            with self.assertRaises(FileNotFoundError):
                apply_vicon_file_order({"NS2D": copied}, order_path)

    def test_rejects_unsupported_missing_split_and_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copied = dataset(root / "local", ["a.h5"])
            path = root / "order.json"
            for content in ('{"version": 2}', '{"version": 1, "splits": {}}',
                            '{"version": 1, "version": 1}'):
                with self.subTest(content=content), self.assertRaises(ValueError):
                    path.write_text(content)
                    apply_vicon_file_order({"NS2D": copied}, path)


if __name__ == "__main__":
    unittest.main()
