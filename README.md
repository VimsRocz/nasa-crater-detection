# NASA Lunar Crater Detection Challenge

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains a computer vision solution for the **NASA Lunar Crater Detection Challenge**. The solution implements ellipse fitting algorithms to automatically detect and classify crater rims in orbital lunar imagery.

### Challenge Description

Crater rims are vital landmarks for planetary science and navigation. However, detecting them in real imagery is challenging due to:
- Shadows and lighting shifts
- Broken edges obscuring crater shapes
- Variable crater sizes and degradation states

This project addresses these challenges by developing robust methods that can reliably fit ellipses to crater rims visible in synthetic lunar orbital images.

## Features

- **Ellipse Fitting**: Advanced ellipse fitting using OpenCV's contour detection and ellipse approximation
- **Image Preprocessing**: CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhanced feature visibility
- **Edge Detection**: Canny edge detection with morphological operations to handle broken edges
- **Crater Filtering**: Implements challenge-specific filtering criteria:
  - Minimum semi-minor axis threshold (40 pixels)
  - Maximum crater size relative to image dimensions
  - Full visibility requirement (no partial craters at image edges)
- **Crater Classification**: Framework for rim classification based on morphology (A, AB, B, BC, C)
- **Batch Processing**: Processes entire datasets with hierarchical folder structure
- **CSV Output**: Generates submission-ready CSV files following challenge format

## Repository Structure

```
nasa-crater-detection/
├── code/
│   └── crater_detector.py    # Main detection algorithm
├── solution/
│   └── solution.csv          # Example output CSV file
├── README.md                  # This file
└── .gitignore                # Python gitignore
```

## Installation

### Prerequisites

- Python 3.12 or higher
- pip package manager

### Required Python Libraries

Install dependencies using pip:

```bash
pip install opencv-python numpy pandas
```

Or create a requirements.txt:

```
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
```

Then install:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the crater detector on a test dataset:

```bash
python code/crater_detector.py --data_folder /path/to/test/data --output solution.csv
```

### Command Line Arguments

- `--data_folder`: **(Required)** Path to the folder containing test images
- `--output`: **(Optional)** Output CSV file path (default: `solution.csv`)

### Input Data Structure

The input data should follow this hierarchical structure:

```
test_data/
├── altitude01/
│   ├── longitude01/
│   │   ├── orientation01_light01.png
│   │   ├── orientation01_light02.png
│   │   └── ...
│   └── longitude02/
│       └── ...
└── altitude02/
    └── ...
```

### Output Format

The solution generates a CSV file with the following columns:

- `ellipseCenterX(px)`: X-coordinate of ellipse center in pixels
- `ellipseCenterY(px)`: Y-coordinate of ellipse center in pixels
- `ellipseSemimajor(px)`: Semi-major axis length in pixels
- `ellipseSemiminor(px)`: Semi-minor axis length in pixels
- `ellipseRotation(deg)`: Rotation angle in degrees (clockwise from x-axis)
- `inputImage`: Unique image identifier (altitude/longitude/filename)
- `crater_classification`: Classification value (0-4 or -1 for unclassified)

### Example Output

```csv
ellipseCenterX(px),ellipseCenterY(px),ellipseSemimajor(px),ellipseSemiminor(px),ellipseRotation(deg),inputImage,crater_classification
1024.5,768.3,125.7,98.2,45.3,altitude01/longitude01/orientation01_light01,-1
650.2,450.8,87.4,72.1,12.8,altitude01/longitude01/orientation01_light01,-1
```

## Algorithm Details

### Detection Pipeline

1. **Image Preprocessing**
   - Gaussian blur to reduce noise
   - CLAHE for contrast enhancement

2. **Edge Detection**
   - Canny edge detection with adaptive thresholds
   - Morphological closing to connect broken edges

3. **Contour Extraction**
   - Find external contours in edge map
   - Filter contours by minimum point count

4. **Ellipse Fitting**
   - Fit ellipse to each valid contour
   - Extract parameters (center, axes, rotation)

5. **Crater Filtering**
   - Apply size constraints
   - Check visibility (full crater within image bounds)
   - Remove outliers

6. **Classification** (Optional)
   - Analyze rim characteristics
   - Classify based on rim crispness

### Key Parameters

- **Minimum semi-minor axis**: 40 pixels (challenge requirement)
- **Maximum crater ratio**: 0.6 × min(image_width, image_height)
- **Canny thresholds**: 50 (low), 150 (high)
- **Gaussian kernel**: 5×5
- **CLAHE clip limit**: 2.0

## Challenge Compliance

### Filtering Criteria

The implementation strictly follows challenge filtering rules:

✓ **Too small craters**: Filtered if semi-minor axis < 40 pixels  
✓ **Too large craters**: Filtered if (width + height) ≥ 0.6 × S  
✓ **Partially visible craters**: Filtered if bounding box extends beyond image

### Output Format

✓ CSV format matching `train-gt.csv` structure  
✓ Special handling for images with no detected craters (`-1` values)  
✓ Proper image ID construction (altitude/longitude/filename)

### Submission Requirements

✓ Single CSV file with all detections  
✓ Maximum 500,000 rows  
✓ Algorithmically generated (no manual annotations)  
✓ Independent image processing (no cross-image state)

## Performance Considerations

### Optimization Tips

- Processing speed: ~1-5 seconds per image (CPU-dependent)
- Memory usage: ~100-500 MB for typical datasets
- Scalability: Designed for batch processing of large datasets

### Hardware Requirements

**Training Phase**:
- Not applicable (pure computer vision, no ML training)

**Testing/Inference Phase**:
- CPU: Any modern processor (GPU not required)
- RAM: Minimum 2 GB, recommended 8 GB for large datasets
- Storage: Depends on dataset size

## Topcoder Challenge Information

- **Challenge**: NASA Crater Detection Challenge
- **Platform**: Topcoder Marathon Match
- **Prize Pool**: $45,000 (1st: $12,000, 2nd: $10,000, 3rd: $8,000, etc.)
- **Submission Deadline**: Check Topcoder platform
- **Challenge Page**: [Topcoder Challenge](https://www.topcoder.com/challenges/e53d30e9-c4b1-40bc-b834-f92483a73223)

### Special Prizes

- **Midway Leaderboard**: 3 × $1,000
- **Scientific Innovation Award**: 2 × $2,000
- **Best Crater Classification**: $3,000

## References

This implementation is inspired by academic research on crater detection:

1. J. A. Christian, H. Derksen, R. Watkins. "Lunar Crater Identification in Digital Images." *Journal of the Astronautical Sciences*, vol. 68, no. 4, pp. 1056-1144, December 2021.

2. P. Mahanti et al. "Small lunar craters at the Apollo 16 and 17 landing sites - morphology and degradation." 2018.

3. M. Krause, J. Price, J. Christian. "Analytical Methods In Crater Rim Fitting And Pattern Recognition." 2023.

## Future Improvements

- [ ] Implement machine learning-based crater classification
- [ ] Add Hough Circle Transform for circular crater detection
- [ ] Integrate deep learning models (YOLO, U-Net)
- [ ] Implement multi-scale detection for varied crater sizes
- [ ] Add lighting condition normalization
- [ ] Optimize for GPU acceleration
- [ ] Implement parallel processing for faster batch operations

## Contributing

Contributions are welcome! Areas for improvement:

- Enhanced ellipse fitting algorithms
- Better crater classification methods
- Performance optimizations
- Additional preprocessing techniques
- Visualization tools

## License

MIT License - Feel free to use this code for research and competition purposes.

## Author

**VimsRocz**
- GitHub: [@VimsRocz](https://github.com/VimsRocz)
- Topcoder: VimsRocz

## Acknowledgments

- NASA for providing the challenge and dataset
- Topcoder for hosting the competition
- OpenCV community for excellent computer vision tools
- Research community for crater detection algorithms

---

**Note**: This solution represents a baseline computer vision approach. For competitive performance in the challenge, consider incorporating advanced techniques such as machine learning, ensemble methods, and sophisticated feature engineering.
