# Code Directory README

This directory contains the Python scripts essential for the NASA Lunar Crater Detection Challenge.

## Scripts Overview:

### 1. `crater_detector_final.py`
This is the **main application script** for detecting craters and performing preliminary analysis and scoring. It integrates the core computer vision pipeline with evaluation functionalities:
-   **Image Preprocessing**: Enhances raw lunar images.
-   **Edge Detection**: Identifies potential crater boundaries.
-   **Contour Detection**: Extracts shapes from the edges.
-   **Ellipse Fitting**: Fits ellipses to detected contours, representing craters.
-   **Crater Filtering**: Applies contest-specific rules to filter out invalid detections (e.g., too small, too large, or partially visible craters).
-   **Confidence Scoring**: Assesses how likely a detected ellipse is an actual crater.
-   **Dataset Processing**: Capable of processing entire image datasets and outputting results in a structured CSV format.
-   **Integrated Analysis**: Now includes functionalities for post-processing analysis of detection results, such as descriptive statistics and outlier detection (previously in `analyze_results.py`).
-   **Integrated Scoring**: Incorporates the official scoring logic to compare detections against ground truth and calculate a score (previously in `scorer.py`).

### 2. `create_submission.py`
This script facilitates the **packaging of the solution** for official submission to the challenge. It automates the process of organizing all necessary files into a specific ZIP file structure required by the contest:
-   Collects the generated `solution.csv` (typically produced by `crater_detector_final.py`).
-   Includes the `crater_detector_final.py` script itself.
-   Generates essential shell scripts (`train.sh` and `test.sh`) that define how the solution should be run for training (or rather, "no training" in this rule-based approach) and testing.
-   Creates a `README.md` specific to the submission, detailing the approach and usage.

This script ensures that the submission adheres to all formatting and content requirements, making it ready for evaluation.

### 3. `run_full_workflow.py`
This is the **orchestration script** that streamlines the entire workflow by processing all predefined datasets automatically. It acts as the central entry point for:
-   **Automated Dataset Processing**: Iterates through the `train-sample`, `train`, and `test` datasets (paths are hardcoded within the script).
-   **Running Crater Detection**: Executes `crater_detector_final.py` for each dataset to process images and generate detection results (e.g., `train-sample_detections.csv`, `train_detections.csv`, `test_detections.csv`).
-   **Scoring Detections**: If a ground truth file is available and specified internally for a dataset, it triggers the integrated scoring logic within `crater_detector_final.py` to evaluate the generated detections and saves the score to a uniquely named file (e.g., `result_train-sample.txt`).
-   **Submission Package Creation (Optional)**: While currently set up for processing individual datasets, this script can be extended to utilize `create_submission.py` to bundle necessary files into a final submission ZIP archive for a specific dataset (e.g., the `test` set).

This script simplifies the execution of the full detection, evaluation, and organized output process across all major datasets.
