from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.swc import parse_swc
from pointneuron.data.vaa3d_raw import decode_pbd8


class SwcParsingTests(unittest.TestCase):
    def test_parse_valid_swc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cell.swc"
            path.write_text("# id type x y z r parent\n1 1 0 0 0 1 -1\n2 3 1 0 0 0.5 1\n", encoding="utf-8")

            tree = parse_swc(path)

            self.assertEqual(len(tree.nodes), 2)
            self.assertEqual(tree.root_count, 1)
            self.assertEqual(tree.edge_count, 1)
            self.assertEqual(tree.validate(), [])

    def test_validate_missing_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cell.swc"
            path.write_text("1 1 0 0 0 1 -1\n2 3 1 0 0 0.5 99\n", encoding="utf-8")

            tree = parse_swc(path)

            self.assertIn("missing parent ids", tree.validate()[0])


class Gold166ScanTests(unittest.TestCase):
    def test_prefers_sorted_then_stamped_then_base_swc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "group" / "sample"
            sample.mkdir(parents=True)
            (sample / "raw.v3dpbd").write_bytes(b"fake")
            (sample / "raw.v3dpbd.swc").write_text("1 1 0 0 0 1 -1\n", encoding="utf-8")
            (sample / "raw.v3dpbd.ano_stamp_1.swc").write_text("1 1 0 0 0 1 -1\n", encoding="utf-8")
            (sample / "raw.v3dpbd.ano_stamp_1.swc_sorted.swc").write_text("1 1 0 0 0 1 -1\n", encoding="utf-8")

            samples = scan_gold166(root)

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].swc_selection, "sorted")
            self.assertTrue(samples[0].swc_path.name.endswith("swc_sorted.swc"))

    def test_falls_back_when_preferred_swc_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "group" / "sample"
            sample.mkdir(parents=True)
            (sample / "raw.v3dpbd").write_bytes(b"fake")
            (sample / "raw.v3dpbd.swc").write_text("1 1 0 0 0 1 -1\n", encoding="utf-8")
            (sample / "raw.v3dpbd.ano_stamp_1.swc_sorted.swc").write_text(
                "1 1 0 0 0 1 -1\n1 1 1 0 0 1 -1\n",
                encoding="utf-8",
            )

            samples = scan_gold166(root)

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].swc_selection, "base")


class Vaa3dRawTests(unittest.TestCase):
    def test_decode_pbd8_repeat_literal_and_difference_runs(self) -> None:
        encoded = bytes(
            [
                130,
                0,
                2,
                5,
                6,
                7,
                35,
                0b00011011,
            ]
        )

        decoded = decode_pbd8(encoded, expected_voxels=9)

        self.assertEqual(decoded, bytes([0, 0, 0, 5, 6, 7, 7, 8, 10]))


if __name__ == "__main__":
    unittest.main()
