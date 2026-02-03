import base64
import importlib.util
from pathlib import Path


def _load_crater_detector_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "code" / "crater_detector_final.py"
    spec = importlib.util.spec_from_file_location("crater_detector_final", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"Failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_crater_detector_module()


def test_validate_data_folder_missing(tmp_path):
    detector = MODULE.CraterDetector()
    ok, msg, image_files = detector.validate_data_folder(str(tmp_path / "missing"))
    assert ok is False
    assert "does not exist" in msg
    assert image_files == []


def test_validate_data_folder_wrong_structure_with_pngs(tmp_path):
    # Create a PNG but not under altitude*/longitude* structure.
    minimal_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
    )
    (tmp_path / "image.png").write_bytes(minimal_png)

    detector = MODULE.CraterDetector()
    ok, msg, image_files = detector.validate_data_folder(str(tmp_path))
    assert ok is False
    assert "Expected:" in msg
    assert image_files == []


def test_generate_sample_data_creates_expected_files(tmp_path):
    out_dir = tmp_path / "sample_data"
    assert MODULE.generate_sample_data(str(out_dir)) is True

    images_dir = out_dir / "altitude01" / "longitude01"
    images = sorted(images_dir.glob("orientation*_light01.png"))
    assert len(images) == 3
    assert all(p.stat().st_size > 0 for p in images)


def test_validate_data_folder_collects_pngs_and_excludes_masks_truth(tmp_path):
    minimal_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
    )
    images_dir = tmp_path / "altitude01" / "longitude01"
    images_dir.mkdir(parents=True)

    (images_dir / "a.png").write_bytes(minimal_png)
    (images_dir / "b_mask.png").write_bytes(minimal_png)
    (images_dir / "c_truth.png").write_bytes(minimal_png)

    detector = MODULE.CraterDetector()
    ok, msg, image_files = detector.validate_data_folder(str(tmp_path))
    assert ok is True
    assert msg == ""

    names = sorted(p.name for p in image_files)
    assert names == ["a.png"]
