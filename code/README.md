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
This is the **orchestration script** that streamlines the entire workflow. It acts as the central entry point for:
-   **Running Crater Detection**: Executes `crater_detector_final.py` to process images and generate detection results.
-   **Scoring Detections**: If a ground truth file is provided, it triggers the integrated scoring logic within `crater_detector_final.py` to evaluate the generated detections.
-   **Creating Submission Package**: Utilizes `create_submission.py` to bundle all necessary files into the final submission ZIP archive.

This script simplifies the execution of the full detection, evaluation, and submission process.
