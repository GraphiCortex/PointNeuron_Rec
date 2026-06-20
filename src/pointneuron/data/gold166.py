from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

from pointneuron.data.swc import parse_swc


VOLUME_EXTENSIONS = (".v3dpbd", ".v3draw")


@dataclass(frozen=True)
class Gold166Sample:
    sample_id: str
    sample_dir: Path
    volume_path: Path | None
    swc_path: Path
    swc_selection: str
    alternate_swc_paths: tuple[Path, ...]
    ano_paths: tuple[Path, ...]
    pixel_size_path: Path | None

    def to_json_dict(self, root: Path) -> dict[str, object]:
        def rel(path: Path | None) -> str | None:
            if path is None:
                return None
            return path.relative_to(root).as_posix()

        data = asdict(self)
        data["sample_dir"] = rel(self.sample_dir)
        data["volume_path"] = rel(self.volume_path)
        data["swc_path"] = rel(self.swc_path)
        data["alternate_swc_paths"] = [rel(path) for path in self.alternate_swc_paths]
        data["ano_paths"] = [rel(path) for path in self.ano_paths]
        data["pixel_size_path"] = rel(self.pixel_size_path)
        return data


def scan_gold166(
    root: str | Path,
    include_without_volume: bool = False,
    include_invalid_swc: bool = False,
) -> list[Gold166Sample]:
    gold_root = Path(root)
    if not gold_root.exists():
        raise FileNotFoundError(f"Gold166 root does not exist: {gold_root}")

    samples: list[Gold166Sample] = []
    for sample_dir in sorted(path for path in gold_root.rglob("*") if path.is_dir()):
        volumes = _find_volumes(sample_dir)
        swcs = sorted(sample_dir.glob("*.swc"))
        if not volumes and not swcs:
            continue
        if not swcs:
            continue
        if not volumes and not include_without_volume:
            continue

        selected = select_ground_truth_swc(swcs, require_valid=not include_invalid_swc)
        if selected is None:
            continue
        selected_swc, selection = selected
        alternates = tuple(path for path in swcs if path != selected_swc)
        anos = tuple(sorted(sample_dir.glob("*.ano")))
        pixel_sizes = sorted(sample_dir.glob("*.pixelsize.txt"))

        samples.append(
            Gold166Sample(
                sample_id=_sample_id(gold_root, sample_dir),
                sample_dir=sample_dir,
                volume_path=volumes[0] if volumes else None,
                swc_path=selected_swc,
                swc_selection=selection,
                alternate_swc_paths=alternates,
                ano_paths=anos,
                pixel_size_path=pixel_sizes[0] if pixel_sizes else None,
            )
        )

    return samples


def select_ground_truth_swc(swcs: list[Path], require_valid: bool = True) -> tuple[Path, str] | None:
    candidates = sorted(swcs, key=_swc_preference_key)
    if not require_valid:
        path = candidates[0]
        return path, _swc_selection_name(path)

    for path in candidates:
        try:
            if not parse_swc(path).validate():
                return path, _swc_selection_name(path)
        except ValueError:
            continue

    return None


def manifest_summary(samples: list[Gold166Sample]) -> dict[str, int]:
    return {
        "samples": len(samples),
        "with_volume": sum(1 for sample in samples if sample.volume_path is not None),
        "without_volume": sum(1 for sample in samples if sample.volume_path is None),
        "selected_sorted": sum(1 for sample in samples if sample.swc_selection == "sorted"),
        "selected_stamped": sum(1 for sample in samples if sample.swc_selection == "stamped"),
        "selected_base": sum(1 for sample in samples if sample.swc_selection == "base"),
        "with_pixel_size": sum(1 for sample in samples if sample.pixel_size_path is not None),
    }


def write_manifest(samples: list[Gold166Sample], root: str | Path, output_path: str | Path) -> None:
    root_path = Path(root)
    payload = {
        "dataset": "gold166",
        "root": root_path.as_posix(),
        "summary": manifest_summary(samples),
        "samples": [sample.to_json_dict(root_path) for sample in samples],
    }
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _find_volumes(sample_dir: Path) -> list[Path]:
    volumes: list[Path] = []
    for extension in VOLUME_EXTENSIONS:
        volumes.extend(sample_dir.glob(f"*{extension}"))
    return sorted(volumes)


def _sample_id(root: Path, sample_dir: Path) -> str:
    return sample_dir.relative_to(root).as_posix()


def _swc_preference_key(path: Path) -> tuple[int, str]:
    selection = _swc_selection_name(path)
    rank = {"sorted": 0, "stamped": 1, "base": 2}[selection]
    return rank, path.name


def _swc_selection_name(path: Path) -> str:
    if path.name.endswith("swc_sorted.swc"):
        return "sorted"
    if "stamp" in path.name.lower():
        return "stamped"
    return "base"
