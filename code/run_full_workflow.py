import sys
from pathlib import Path
import shutil
import pandas as pd
from typing import Optional

# Import the main detector script
import crater_detector_final as detector_script

# Import the submission creation script
import create_submission as submission_script

def run_workflow_for_dataset(data_folder: str, output_filename: str, truth_file: Optional[str] = None, perform_scoring: bool = True):
    """
    Runs the crater detection workflow for a single dataset.
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING DATASET: {data_folder}")
    print(f"{'='*70}")

    output_detections_path = Path(output_filename)

    # --- Step 1: Run Crater Detection ---
    print("\n" + "="*70)
    print("STEP 1: RUNNING CRATER DETECTION")
    print("="*70)
    
    detector = detector_script.CraterDetector() # Use default parameters for now
    detection_success = detector.process_dataset(data_folder, str(output_detections_path), verbose=True)
    
    if not detection_success:
        print(f"Crater detection failed for {data_folder}. Skipping remaining steps for this dataset.")
        return

    # --- Step 2: Score Detections (if truth_file is provided and scoring is enabled) ---
    if perform_scoring and truth_file:
        print("\n" + "="*70)
        print("STEP 2: SCORING DETECTIONS")
        print("="*70)
        
        try:
            truth_path = Path(truth_file)
            if truth_path.is_file():
                ground_truth_df = pd.read_csv(truth_path)
            else:
                ground_truth_df = detector_script.load_ground_truth(data_folder) 
            
            if ground_truth_df is None or ground_truth_df.empty:
                print(f"Error: Could not load ground truth from {truth_file} or {data_folder}. Skipping scoring.")
            else:
                print(f'✓ Loaded ground truth with {len(ground_truth_df)} craters')

                truth_for_scoring = (
                    ground_truth_df.set_index('inputImage').groupby(level='inputImage')
                    .apply(lambda g: g.to_dict(orient='records'))
                    .to_dict()
                )

                detections_for_scoring_df = pd.read_csv(output_detections_path)
                detections_for_scoring = (
                    detections_for_scoring_df.set_index('inputImage').groupby(level='inputImage')
                    .apply(lambda g: g.to_dict(orient='records'))
                    .to_dict()
                )

                image_ids = list(truth_for_scoring.keys())
                total_score = 0
                for img_id in image_ids:
                    truth_craters = truth_for_scoring.get(img_id, [])
                    detection_craters = detections_for_scoring.get(img_id, [])
                    
                    for tc in truth_craters:
                        tc['matched'] = False
                    for dc in detection_craters:
                        dc['matched'] = False

                    score_for_image = detector_script.score1(truth_craters, detection_craters)
                    total_score += score_for_image

                if len(image_ids) > 0:
                    average_score = total_score / len(image_ids)
                    final_score = 100 * average_score
                else:
                    final_score = 0.0

                print(f'Overall Score: {final_score}')
                score_output_dir = Path("results/scorer-out")
                score_output_dir.mkdir(parents=True, exist_ok=True)
                # Ensure the score file name is unique for each dataset
                score_file_name = f"result_{Path(data_folder).name}.txt"
                detector_script.writeScore(final_score, str(score_output_dir / score_file_name))
                print(f"Score saved to {score_output_dir / score_file_name}")

        except Exception as e:
            print(f"Error during scoring process with truth file {truth_file}: {e}. Skipping scoring for this dataset.")


def main():
    # Define datasets to process
    datasets = [
        {
            "data_folder": "../data/train-sample",
            "output_filename": "train-sample_detections.csv",
            "truth_file": "../data/train-sample-gt.csv" # Assuming this file exists
        },
        {
            "data_folder": "../data/train",
            "output_filename": "train_detections.csv",
            "truth_file": "../data/train-gt.csv" # Assuming this file exists
        },
        {
            "data_folder": "../data/test",
            "output_filename": "test_detections.csv",
            "truth_file": None # No ground truth for test data
        }
    ]

    for dataset_info in datasets:
        run_workflow_for_dataset(
            data_folder=dataset_info["data_folder"],
            output_filename=dataset_info["output_filename"],
            truth_file=dataset_info["truth_file"],
            perform_scoring=True # Always try to score if truth is provided
        )

    # --- Final Step: Create a combined submission package if desired ---
    # This part can be uncommented and adjusted if a single submission package
    # is needed after processing all datasets.
    # For now, focusing on individual dataset processing.
    
    # print(f"\n{'='*70}")
    # print("CREATING FINAL SUBMISSION PACKAGE (OPTIONAL)")
    # print(f"{'='*70}")
    # try:
    #     # Assuming 'test_detections.csv' is the one for final submission
    #     final_submission_output = "test_detections.csv" 
    #     solution_csv_path_for_submission = Path("solution.csv")
    #     shutil.copy(final_submission_output, solution_csv_path_for_submission)
    #     print(f"Copied {final_submission_output} to {solution_csv_path_for_submission} for submission packaging.")

    #     submission_success = submission_script.create_submission_package(
    #         base_dir=".", 
    #         output_zip="submission_all_datasets.zip"
    #     )
        
    #     solution_csv_path_for_submission.unlink()
    #     print(f"Cleaned up temporary file: {solution_csv_path_for_submission}")

    #     if not submission_success:
    #         print("Final submission package creation failed.")
    # except Exception as e:
    #     print(f"Error creating final submission package: {e}")

    print(f"\n{'='*70}")
    print("ALL DATASET WORKFLOWS COMPLETE!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
