#!/usr/bin/env python3
"""
Create a Topcoder submission ZIP for the NASA Lunar Crater Detection Challenge.

Expected ZIP structure:
  /solution/solution.csv
  /code/crater_detector.py
  /code/train.sh
  /code/test.sh
  /code/README.md
"""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path


def _train_sh() -> str:
    return """#!/bin/bash
# Training script for crater detection
# This is a rule-based computer vision approach, so no training is required.

echo "No training required - using algorithmic crater detection"
echo "Method: Edge detection + Ellipse fitting + Filtering"
exit 0
"""


def _test_sh() -> str:
    return """#!/bin/bash
# Testing script for crater detection
# Usage: ./test.sh <data_folder> <output_folder>

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <data_folder> <output_folder>"
  exit 1
fi

DATA_FOLDER="$1"
OUTPUT_FOLDER="$2"

echo "Running crater detection..."
echo "Input: $DATA_FOLDER"
echo "Output: $OUTPUT_FOLDER"

mkdir -p "$OUTPUT_FOLDER"
python3 crater_detector.py --data_folder "$DATA_FOLDER" --output "$OUTPUT_FOLDER/solution.csv"

echo "Detection complete!"
"""


def _submission_readme() -> str:
    return """# NASA Crater Detection Challenge Solution

## Approach

This solution uses a classical computer vision pipeline:

1. Preprocessing (Gaussian blur + CLAHE)
2. Edge detection (Canny + morphological closing)
3. Contour extraction
4. Ellipse fitting (OpenCV)
5. Filtering (per contest rules)

## Usage

### Training (not required)

```bash
./train.sh /path/to/training/data
```

### Testing

```bash
./test.sh /path/to/test/data /path/to/output
```

### Direct Python usage

```bash
python crater_detector.py --data_folder /path/to/data --output solution.csv
```
"""


def _ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")


def create_submission_zip(
    *, solution_csv: Path, crater_detector_py: Path, output_zip: Path
) -> None:
    _ensure_file(solution_csv, "solution.csv")
    _ensure_file(crater_detector_py, "crater_detector.py")

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="submission_") as tmp_dir:
        tmp = Path(tmp_dir)

        solution_dir = tmp / "solution"
        code_dir = tmp / "code"
        solution_dir.mkdir(parents=True, exist_ok=True)
        code_dir.mkdir(parents=True, exist_ok=True)

        (solution_dir / "solution.csv").write_bytes(solution_csv.read_bytes())
        (code_dir / "crater_detector.py").write_bytes(crater_detector_py.read_bytes())
        (code_dir / "train.sh").write_text(_train_sh(), encoding="utf-8")
        (code_dir / "test.sh").write_text(_test_sh(), encoding="utf-8")
        (code_dir / "README.md").write_text(_submission_readme(), encoding="utf-8")

        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in tmp.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(tmp))


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a Topcoder submission ZIP.")
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path("."),
        help="Repo root (default: current directory)",
    )
    parser.add_argument(
        "--solution",
        type=Path,
        default=None,
        help='Path to solution CSV (default: "<base_dir>/solution.csv")',
    )
    parser.add_argument(
        "--crater_detector",
        type=Path,
        default=None,
        help=(
            'Path to detector script (default: "<base_dir>/code/crater_detector_final.py" if present, '
            'otherwise "<base_dir>/code/crater_detector.py")'
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.zip"),
        help='Output zip (default: "<base_dir>/submission.zip")',
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    solution_csv = (args.solution or (base_dir / "solution.csv")).resolve()
    if args.crater_detector is not None:
        crater_detector_py = args.crater_detector.resolve()
    else:
        preferred = base_dir / "code" / "crater_detector_final.py"
        fallback = base_dir / "code" / "crater_detector.py"
        crater_detector_py = (preferred if preferred.exists() else fallback).resolve()
    output_zip = (
        (base_dir / args.output).resolve()
        if not args.output.is_absolute()
        else args.output.resolve()
    )

    create_submission_zip(
        solution_csv=solution_csv,
        crater_detector_py=crater_detector_py,
        output_zip=output_zip,
    )
    print(f"Created: {output_zip} ({output_zip.stat().st_size / (1024 * 1024):.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
