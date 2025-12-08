# Crater Detector - Usage Guide

## 🚀 Quick Start

### Method 1: Interactive Mode (Recommended for beginners)

Simply run the script without any arguments:

```powershell
python code\crater_detector_final.py
```

You'll see an interactive menu:

```
🌙 NASA LUNAR CRATER DETECTION CHALLENGE
   Computer Vision Solution by VimsRocz

======================================================================
DATA FOLDER SELECTION
======================================================================

Available data folders:
  1. data\train-sample         - Small training set (50 images, ~234 MB)
     Status: ✓ Available
  2. data\train                - Full training set (19 GB)
     Status: ✓ Available
  3. data\test                 - Test set for submission (3.3 GB)
     Status: ✓ Available

  0. Cancel / Enter path manually
======================================================================

Select option (0-3):
```

### Method 2: Command-Line Mode (For automation)

**Training Sample with Evaluation:**
```powershell
python code\crater_detector_final.py --data_folder data\train-sample --output results\train_sample_detections.csv --evaluate
```

**Full Training Set:**
```powershell
python code\crater_detector_final.py --data_folder data\train --output results\train_full_detections.csv --evaluate
```

**Test Set (for submission):**
```powershell
python code\crater_detector_final.py --data_folder data\test --output results\solution.csv
```

## 🎛️ Parameter Tuning

### Default Parameters
- Canny threshold 1: 50
- Canny threshold 2: 150
- Circularity threshold: 0.6

### Custom Parameters Example
```powershell
python code\crater_detector_final.py --data_folder data\train-sample --output results\tuned_detections.csv --canny1 100 --canny2 200 --circularity 0.7 --evaluate
```

### Parameter Effects

**`--canny1` and `--canny2`** (Edge Detection)
- Lower values: More edges detected (may include noise)
- Higher values: Fewer edges detected (may miss faint craters)
- Recommended range: 50-150 (canny1), 100-250 (canny2)

**`--circularity`** (Shape Validation)
- Range: 0.0 to 1.0
- Higher values: Only near-circular craters accepted
- Lower values: More elliptical shapes accepted
- Recommended range: 0.5-0.8

## 📁 Data Folder Structure

**✅ CORRECT:**
```
data\train-sample\
├── altitude01\
│   ├── longitude01\
│   │   ├── orientation01_light01.png
│   │   └── ...
│   └── longitude02\
└── altitude02\
```

**❌ INCORRECT:**
```
data\                          # Too high level
Data\                          # Wrong case
data\train-sample.tar         # Not extracted
```

## 📊 Understanding Output

### Console Output
```
Starting crater detection on: data\train-sample
======================================================================
Found 50 images to process.
Processing...
  Progress: 10/50
  Progress: 20/50
  ...
✓ Results saved to results\train-sample_detections.csv
✓ Total detections: 1000
```

### Evaluation Results (with --evaluate flag)
```
======================================================================
EVALUATION MODE
======================================================================
  Loaded 1388 ground truth craters

Comparison Statistics:
  Ground Truth Craters: 1388
  Detected Craters: 1000
  Ground Truth Images: 50
  Detected Images: 50
  Common Images: 50
  Missing Images: 0

Overall Score: 1.22
```

### CSV Output Format
```csv
ellipseCenterX(px),ellipseCenterY(px),ellipseSemimajor(px),ellipseSemiminor(px),ellipseRotation(deg),inputImage,crater_classification
512.5,768.3,125.7,98.2,45.3,altitude01/longitude01/orientation01_light01,-1
650.2,450.8,87.4,72.1,12.8,altitude01/longitude01/orientation01_light01,-1
```

## 🔧 Troubleshooting

### Error: "No altitude folders found"
**Problem:** Wrong data folder specified
**Solution:** Use `data\train-sample`, NOT `data` or `Data`

### Error: "Could not read image"
**Problem:** TAR archive not extracted or corrupted
**Solution:** Extract `train.tar`, `train-sample.tar`, or `test.tar` properly

### Low Score (< 5.0)
**Problem:** Detection parameters may need tuning
**Solutions:**
1. Try higher Canny thresholds (reduce noise)
2. Adjust circularity threshold
3. Check if images are properly preprocessed

### Very High Detection Count
**Problem:** Too many false positives
**Solutions:**
1. Increase `--canny1` and `--canny2` values
2. Increase `--circularity` threshold
3. Review confidence threshold in code

## 💡 Tips & Best Practices

1. **Always test on train-sample first** before running on full dataset
2. **Use --evaluate flag** when working with training data to get immediate feedback
3. **Save different parameter configurations** with descriptive output filenames:
   ```powershell
   --output results\canny100_200_circ07.csv
   ```
4. **Keep a log** of parameter combinations and their scores
5. **Extract TAR archives** before running (don't point to .tar files)

## 📈 Workflow Recommendations

### Beginner Workflow
1. Run interactive mode on train-sample
2. Review output and score
3. Try different parameters
4. Run on full training set with best parameters
5. Generate submission from test set

### Advanced Workflow
1. Batch test multiple parameter combinations
2. Analyze statistical outputs
3. Identify problematic images
4. Fine-tune parameters per altitude/lighting
5. Validate on full training set
6. Generate final submission

## 🆘 Getting Help

Run with `--help` to see all options:
```powershell
python code\crater_detector_final.py --help
```

For detailed algorithm information, see comments in `crater_detector_final.py`
