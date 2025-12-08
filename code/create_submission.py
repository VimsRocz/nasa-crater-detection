#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 23:30:08 2025

@author: vimalchawda
"""

#!/usr/bin/env python3
"""
Create NASA Crater Detection Challenge Submission Package

Creates a ZIP file with the required structure:
/solution
  /solution.csv
/code
  /crater_detector.py
  /train.sh
  /test.sh
  /README.md
"""

import zipfile
import os
from pathlib import Path
import shutil

def create_train_sh():
    """Create train.sh script"""
    return """#!/bin/bash
# Training script for crater detection
# This is a rule-based computer vision approach, no training required

echo "No training required - using algorithmic crater detection"
echo "Method: Edge detection + Ellipse fitting + Filtering"
exit 0
"""

def create_test_sh():
    """Create test.sh script"""
    return """#!/bin/bash
# Testing script for crater detection
# Usage: ./test.sh <data_folder> <output_folder>

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <data_folder> <output_folder>"
    exit 1
fi

DATA_FOLDER=$1
OUTPUT_FOLDER=$2

echo "Running crater detection..."
echo "Input: $DATA_FOLDER"
echo "Output: $OUTPUT_FOLDER"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Run detection
python3 crater_detector.py --data_folder "$DATA_FOLDER" --output "$OUTPUT_FOLDER/solution.csv"

echo "Detection complete!"
"""

def create_readme():
    """Create README.md"""
    return """# NASA Crater Detection Challenge Solution

## Author
VimsRocz

## Approach
This solution uses a computer vision approach based on:
1. **Preprocessing**: CLAHE enhancement + Gaussian blur
2. **Edge Detection**: Canny edge detection with morphological operations
3. **Ellipse Fitting**: Fit ellipses to detected contours
4. **Filtering**: Apply contest rules (size, visibility, bounding box)

## Method
- **No ML training required** - Pure algorithmic approach
- **Edge-based detection** using OpenCV's ellipse fitting
- **Filtering** based on contest specifications

## Dependencies
- Python 3.12+
- OpenCV (cv2)
- NumPy
- Pandas

## Usage

### Training (No training needed)
```bash
./train.sh /path/to/training/data
```

### Testing
```bash
./test.sh /path/to/test/data /path/to/output
```

### Direct Python Usage
```bash
python crater_detector.py --data_folder /path/to/data --output solution.csv
```

## Performance
- Detection based on edge detection and ellipse fitting
- Filters applied: min semi-minor axis (40px), max size ratio (0.6), full visibility
- Classification: Not implemented (-1 for all craters)

## Notes
- Single-threaded processing as per contest requirements
- No GPU required
- Processes images sequentially
- Memory efficient (< 6GB RAM)
"""

def create_submission_package(base_dir: str, output_zip: str = "submission.zip"):
    """
    Create submission package
    
    Args:
        base_dir: Base directory (nasa-crater-detection)
        output_zip: Output ZIP filename
    """
    base_path = Path(base_dir)
    
    # Check required files
    solution_csv = base_path / "solution.csv"
    crater_detector = base_path / "code" / "crater_detector.py"
    
    if not solution_csv.exists():
        print(f"Error: solution.csv not found at {solution_csv}")
        return False
    
    if not crater_detector.exists():
        print(f"Error: crater_detector.py not found at {crater_detector}")
        return False
    
    print("Creating submission package...")
    print("="*70)
    
    # Create temporary directory structure
    temp_dir = base_path / "submission_temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    solution_dir = temp_dir / "solution"
    code_dir = temp_dir / "code"
    solution_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    
    # Copy solution.csv
    print("✓ Copying solution.csv")
    shutil.copy2(solution_csv, solution_dir / "solution.csv")
    
    # Copy crater_detector.py
    print("✓ Copying crater_detector.py")
    shutil.copy2(crater_detector, code_dir / "crater_detector.py")
    
    # Create train.sh
    print("✓ Creating train.sh")
    train_sh = code_dir / "train.sh"
    train_sh.write_text(create_train_sh())
    train_sh.chmod(0o755)  # Make executable
    
    # Create test.sh
    print("✓ Creating test.sh")
    test_sh = code_dir / "test.sh"
    test_sh.write_text(create_test_sh())
    test_sh.chmod(0o755)  # Make executable
    
    # Create README.md
    print("✓ Creating README.md")
    readme = code_dir / "README.md"
    readme.write_text(create_readme())
    
    # Create ZIP file
    zip_path = base_path / output_zip
    print(f"\n✓ Creating ZIP package: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all files maintaining directory structure
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(temp_dir)
                zipf.write(file_path, arcname)
                print(f"  Added: {arcname}")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    # Get file size
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    
    print("\n" + "="*70)
    print("✅ SUBMISSION PACKAGE CREATED SUCCESSFULLY!")
    print("="*70)
    print(f"📦 Package: {zip_path}")
    print(f"📊 Size: {size_mb:.2f} MB")
    print(f"📁 Contents:")
    print(f"   /solution/solution.csv")
    print(f"   /code/crater_detector.py")
    print(f"   /code/train.sh")
    print(f"   /code/test.sh")
    print(f"   /code/README.md")
    print("\n✓ Ready for submission to Topcoder!")
    print("="*70)
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create NASA Crater Detection submission package'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default='.',
        help='Base directory (default: current directory)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='submission.zip',
        help='Output ZIP filename (default: submission.zip)'
    )
    
    args = parser.parse_args()
    
    success = create_submission_package(args.base_dir, args.output)
    exit(0 if success else 1)

if __name__ == '__main__':
    main()