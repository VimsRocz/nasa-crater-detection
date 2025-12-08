import argparse
import sys
from pathlib import Path
import shutil

# Import the main detector script
import crater_detector_final as detector_script

# Import the submission creation script
import create_submission as submission_script

def main():
    parser = argparse.ArgumentParser(
        description='Orchestrates the full crater detection workflow: detect, score, and create submission.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full workflow on training sample data (detect, score, and package)
  python run_full_workflow.py --data_folder ../data/train-sample --output_detections detections.csv --truth_file ../data/train-gt.csv --create_submission

  # Run full workflow on test data for final submission (detect and package)
  python run_full_workflow.py --data_folder ../data/test --output_detections solution.csv --create_submission
        """
    )
    parser.add_argument('--data_folder', type=str, required=True,
                        help='Path to data folder containing images (e.g., ../data/train-sample or ../data/test)')
    parser.add_argument('--output_detections', type=str, default='solution.csv',
                        help='Output CSV file path for detections (default: solution.csv)')
    parser.add_argument('--truth_file', type=str,
                        help='Path to the ground truth CSV file for scoring (e.g., ../data/train-gt.csv). Optional.')
    parser.add_argument('--create_submission', action='store_true',
                        help='If set, create the submission ZIP package.')
    parser.add_argument('--output_submission_zip', type=str, default='submission.zip',
                        help='Name of the output submission ZIP file (default: submission.zip)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print detailed progress information')
    
    # Add detector-specific arguments if needed, or pass them through
    parser.add_argument('--canny1', type=int, default=50, help='First threshold for the Canny edge detector.')
    parser.add_argument('--canny2', type=int, default=150, help='Second threshold for the Canny edge detector.')
    parser.add_argument('--circularity', type=float, default=0.6, help='Circularity threshold for crater validation.')

    args = parser.parse_args()

    # --- Step 1: Run Crater Detection ---
    print("\n" + "="*70)
    print("STEP 1: RUNNING CRATER DETECTION")
    print("="*70)
    
    # Create detector instance using arguments
    detector = detector_script.CraterDetector(
        canny_th1=args.canny1,
        canny_th2=args.canny2,
        circularity_threshold=args.circularity
    )
    
    detection_success = detector.process_dataset(args.data_folder, args.output_detections, verbose=args.verbose)
    
    if not detection_success:
        print("Crater detection failed. Exiting.")
        sys.exit(1)

    # --- Step 2: Score Detections (if truth_file is provided) ---
    if args.truth_file:
        print("\n" + "="*70)
        print("STEP 2: SCORING DETECTIONS")
        print("="*70)
        
        try:
            # Need to handle load_ground_truth based on whether args.truth_file is a single CSV
            # or a folder containing 'truth/detections.csv'
            truth_path = Path(args.truth_file)
            if truth_path.is_file():
                # Assume it's a direct path to a train-gt.csv type file
                ground_truth_df = pd.read_csv(truth_path)
            else:
                # Assume it's a data folder containing truth/detections.csv structure
                ground_truth_df = detector_script.load_ground_truth(args.data_folder) 
            
            if ground_truth_df is None or ground_truth_df.empty:
                print(f"Error: Could not load ground truth from {args.truth_file} or {args.data_folder}. Skipping scoring.")
            else:
                print(f'✓ Loaded ground truth with {len(ground_truth_df)} craters')

                # Prepare truth data for scoring
                truth_for_scoring = (
                    ground_truth_df.set_index('inputImage').groupby(level='inputImage')
                    .apply(lambda g: g.to_dict(orient='records'))
                    .to_dict()
                )

                detections_for_scoring_df = pd.read_csv(args.output_detections)
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
                    
                    # Reset 'matched' status for each image's detections and truth
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
                detector_script.writeScore(final_score, str(score_output_dir))
                print(f"Score saved to {score_output_dir / 'result.txt'}")

        except Exception as e:
            print(f"Error during scoring process with truth file {args.truth_file}: {e}")

    # --- Step 3: Create Submission Package (if requested) ---
    if args.create_submission:
        print("\n" + "="*70)
        print("STEP 3: CREATING SUBMISSION PACKAGE")
        print("="*70)
        
        # The submission expects solution.csv to be in the base directory of the submission temp structure
        # We need to copy args.output_detections to a file named 'solution.csv' in the current working directory
        # so that create_submission_package can find it.
        
        # Ensure solution.csv exists in the current directory for submission_script
        solution_csv_path_for_submission = Path("solution.csv")
        shutil.copy(args.output_detections, solution_csv_path_for_submission)
        print(f"Copied {args.output_detections} to {solution_csv_path_for_submission} for submission packaging.")

        submission_success = submission_script.create_submission_package(
            base_dir=".", # Now solution.csv is in the current directory
            output_zip=args.output_submission_zip
        )
        
        # Clean up the temporary solution.csv file
        solution_csv_path_for_submission.unlink()
        print(f"Cleaned up temporary file: {solution_csv_path_for_submission}")

        if not submission_success:
            print("Submission package creation failed. Exiting.")
            sys.exit(1)

    print("\n" + "="*70)
    print("WORKFLOW COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
