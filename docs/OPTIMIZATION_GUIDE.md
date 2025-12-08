# Crater Detection Optimization Guide

## Current Status
- **Current Score**: 1.22 (train-sample)
- **Target Score**: 61.37
- **Ground Truth**: 1000 craters in train-sample
- **Currently Detecting**: ~1000 craters, but low quality matches

## Key Issues Identified

### 1. **Parameter Tuning Strategy**
The current parameters are detecting approximately the right NUMBER of craters, but the QUALITY of detections is poor. This means:
- Detection locations are inaccurate
- Ellipse sizes/orientations don't match ground truth well
- Many false positives, many false negatives

### 2. **What the Score Means**
The dGA (Geodesic Angle) scoring function:
- Compares detected ellipse parameters with ground truth
- Penalizes position errors
- Penalizes size/shape errors  
- Penalizes rotation errors
- A score of 1.22 means very few accurate matches

### 3. **Optimization Strategy**

#### Phase 1: Increase Detection Sensitivity (Get MORE craters)
```python
CANNY_THRESHOLD_1 = 20      # Lower from 50
CANNY_THRESHOLD_2 = 60      # Lower from 150  
CIRCULARITY_THRESHOLD = 0.3  # Lower from 0.6
MIN_SEMI_MINOR_AXIS = 35     # Lower from 40
CONFIDENCE_THRESHOLD = 0.15  # Lower from 0.3
```

####Phase 2: Improve Preprocessing
- Use bilateral filter to preserve edges
- Increase CLAHE clip limit
- Add sharpening kernel
- Try morphological gradient

#### Phase 3: Multi-Scale Detection
- Run detection at multiple Canny threshold pairs
- Merge results using Non-Maximum Suppression
- This catches both faint and strong crater rims

#### Phase 4: Better Ellipse Fitting
- Use OpenCV's fitEllipse with different methods
- Try RANSAC for robust fitting
- Filter based on ellipse fitting error

## Recommended Next Steps

### STEP 1: Test with extreme sensitivity
```bash
cd c:\Users\vimsr\Desktop\nasa-crater-detection
python code\crater_detector_final.py --canny1 15 --canny2 50 --circularity 0.25
```

### STEP 2: Check detection count
Target: Should detect 1500-2000 craters (over-detection is OK, false positives will have low dGA overlap)

### STEP 3: Analyze results
```python
import pandas as pd
gt = pd.read_csv('data/output_train-sample.csv')
det = pd.read_csv('results/train-sample_detections.csv')

print(f"Ground truth: {len(gt)} craters")
print(f"Detected: {len(det)} craters")  
print(f"\nGT size range: {gt['ellipseSemiminor(px)'].min():.1f} - {gt['ellipseSemiminor(px)'].max():.1f}")
print(f"Det size range: {det['ellipseSemiminor(px)'].min():.1f} - {det['ellipseSemiminor(px)'].max():.1f}")
```

### STEP 4: Implement multi-pass detection
Create a function that runs detection with 3 different threshold combinations and merges results.

##Ground Truth Analysis
From `data/output_train-sample.csv`:
- Total craters: 1000
- Semi-minor axis range: 40-270 pixels
- Mean semi-minor: 64 pixels
- Many elongated ellipses (aspect ratio > 2)

## Critical Insight
**You need to match not just the COUNT, but the LOCATION and SHAPE accurately.**
Score of 1.22 suggests only ~2% of detections have good overlap with ground truth.

To reach 61.37, you need ~95% of detections to closely match ground truth ellipses.
