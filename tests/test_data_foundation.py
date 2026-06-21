from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.point_cloud import volume_to_point_cloud
from pointneuron.data.swc import parse_swc
from pointneuron.data.training_cache import skeleton_edge_index, skeleton_to_array
from pointneuron.data.point_cloud import SkeletonRecord
from pointneuron.data.splits import SplitRatios, split_records
from pointneuron.data.vaa3d_raw import Vaa3dHeader, Vaa3dVolume
from pointneuron.data.vaa3d_raw import decode_pbd8, decode_pbd16


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

    def test_decode_pbd16_literal_difference_and_repeat_runs(self) -> None:
        encoded = bytes(
            [
                0,
                0xE8,
                0x03,
                33,
                0b00111100,
                224,
                0x05,
                0x00,
            ]
        )

        decoded = decode_pbd16(encoded, expected_voxels=5)
        values = [
            int.from_bytes(decoded[index : index + 2], "little")
            for index in range(0, len(decoded), 2)
        ]

        self.assertEqual(values, [1000, 1001, 1000, 5, 5])


class PointCloudTests(unittest.TestCase):
    def test_volume_to_point_cloud_uses_xyz_order(self) -> None:
        header = Vaa3dHeader(
            key="raw_image_stack_by_hpeng",
            endian="L",
            datatype=1,
            dimensions=(2, 2, 2, 1),
        )
        volume = Vaa3dVolume(
            path=Path("fake.v3draw"),
            header=header,
            data=bytes([0, 1, 2, 0, 3, 0, 4, 5]),
        )

        point_cloud = volume_to_point_cloud(volume, threshold=1)

        self.assertEqual(
            [(point.x, point.y, point.z, point.intensity) for point in point_cloud.points],
            [(0, 1, 0, 2), (0, 0, 1, 3), (0, 1, 1, 4), (1, 1, 1, 5)],
        )

    def test_volume_to_point_cloud_samples_deterministically(self) -> None:
        header = Vaa3dHeader(
            key="raw_image_stack_by_hpeng",
            endian="L",
            datatype=1,
            dimensions=(4, 1, 1, 1),
        )
        volume = Vaa3dVolume(path=Path("fake.v3draw"), header=header, data=bytes([1, 2, 3, 4]))

        first = volume_to_point_cloud(volume, max_points=2, seed=7)
        second = volume_to_point_cloud(volume, max_points=2, seed=7)

        self.assertEqual(first.points, second.points)

    def test_volume_to_point_cloud_reads_uint16_data(self) -> None:
        header = Vaa3dHeader(
            key="raw_image_stack_by_hpeng",
            endian="L",
            datatype=2,
            dimensions=(3, 1, 1, 1),
        )
        volume = Vaa3dVolume(
            path=Path("fake.v3draw"),
            header=header,
            data=b"\x00\x00\x2c\x01\x10\x27",
        )

        point_cloud = volume_to_point_cloud(volume, threshold=300)

        self.assertEqual(
            [(point.x, point.y, point.z, point.intensity) for point in point_cloud.points],
            [(2, 0, 0, 10000)],
        )


class TrainingCacheTests(unittest.TestCase):
    def test_skeleton_arrays_and_edges_use_node_indices(self) -> None:
        skeleton = (
            SkeletonRecord(node_id=10, x=0, y=0, z=0, radius=1, parent_id=-1),
            SkeletonRecord(node_id=20, x=1, y=0, z=0, radius=1, parent_id=10),
            SkeletonRecord(node_id=30, x=2, y=0, z=0, radius=1, parent_id=20),
        )

        nodes = skeleton_to_array(skeleton)
        edges = skeleton_edge_index(skeleton)

        self.assertEqual(nodes.shape, (3, 6))
        self.assertEqual(edges.tolist(), [[0, 1], [1, 2]])


class SplitTests(unittest.TestCase):
    def test_split_records_is_deterministic(self) -> None:
        records = [f"sample_{index}.npz" for index in range(10)]

        first = split_records(records, seed=11)
        second = split_records(records, seed=11)

        self.assertEqual(first, second)
        self.assertEqual({key: len(value) for key, value in first.items()}, {"train": 7, "val": 1, "test": 2})

    def test_split_ratios_must_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            split_records(["a", "b", "c"], ratios=SplitRatios(train=0.5, val=0.3, test=0.3))


if __name__ == "__main__":
    unittest.main()
