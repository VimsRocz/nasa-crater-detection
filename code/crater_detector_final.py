#!/usr/bin/env python3
"""
================================================================================
NASA LUNAR CRATER DETECTION CHALLENGE SOLUTION - MAIN VERSION
================================================================================
Author: VimsRocz
Description: Computer vision solution for detecting and classifying crater rims
             in orbital lunar imagery using ellipse fitting algorithms.

Version: 1.0 (Production)
Last Updated: 2024-12-05
================================================================================
"""

import cv2
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

DEFAULT_SAMPLE_FOLDER = "sample_data"
DEFAULT_MIN_SEMI_MINOR_AXIS = 40
DEFAULT_MAX_CRATER_RATIO = 0.6
DEFAULT_CONFIDENCE_THRESHOLD = 0.3


# ============================================================================
# MAIN CRATER DETECTOR CLASS
# ============================================================================

class CraterDetector:
    """Main crater detection class implementing ellipse fitting algorithms."""
    
    def __init__(self, min_semi_minor_axis=DEFAULT_MIN_SEMI_MINOR_AXIS,
                 max_crater_ratio=DEFAULT_MAX_CRATER_RATIO,
                 confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                 canny_th1=50, canny_th2=150):
        self.min_semi_minor_axis = min_semi_minor_axis
        self.max_crater_ratio = max_crater_ratio
        self.confidence_threshold = confidence_threshold
        self.canny_th1 = canny_th1
        self.canny_th2 = canny_th2
    
    # ========================================================================
    # SECTION 1: IMAGE PREPROCESSING
    # ========================================================================
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the grayscale lunar image for crater detection."""
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        return enhanced
    
    # ========================================================================
    # SECTION 2: EDGE DETECTION
    # ========================================================================
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Detect edges using Canny edge detection."""
        edges = cv2.Canny(image, self.canny_th1, self.canny_th2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        return edges
    
    # ========================================================================
    # SECTION 3: CONTOUR DETECTION
    # ========================================================================
    
    def find_contours(self, edges: np.ndarray) -> List:
        """Find contours in the edge map."""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    # ========================================================================
    # SECTION 4: ELLIPSE FITTING
    # ========================================================================
    
    def fit_ellipse_to_contour(self, contour: np.ndarray) -> Optional[Tuple]:
        """Fit an ellipse to a contour if it has sufficient points."""
        if len(contour) < 5:
            return None
        
        try:
            ellipse = cv2.fitEllipse(contour)
            center_x, center_y = ellipse[0]
            width, height = ellipse[1]
            rotation = ellipse[2]
            
            semi_major = max(width, height) / 2.0
            semi_minor = min(width, height) / 2.0
            
            if width < height:
                rotation = (rotation + 90) % 180
            
            return (center_x, center_y, semi_major, semi_minor, rotation)
        except cv2.error:
            return None
    
    # ========================================================================
    # SECTION 5: CRATER FILTERING (CONTEST RULES)
    # ========================================================================
    
    def filter_crater(self, ellipse_params: Optional[Tuple], 
                     image_shape: Tuple) -> bool:
        """Apply filtering criteria per contest rules."""
        if ellipse_params is None:
            return False
        
        center_x, center_y, semi_major, semi_minor, rotation = ellipse_params
        height, width = image_shape[:2]
        
        # Filter 1: Too small craters
        if semi_minor < self.min_semi_minor_axis:
            return False
        
        # Filter 2: Too large craters
        S = min(width, height)
        w = semi_major * 2
        h = semi_minor * 2
        if (w + h) >= (self.max_crater_ratio * S):
            return False
        
        # Filter 3: Not fully visible
        cos_angle = np.cos(np.radians(rotation))
        sin_angle = np.sin(np.radians(rotation))
        
        a = semi_major
        b = semi_minor
        bbox_width = 2 * np.sqrt((a * cos_angle)**2 + (b * sin_angle)**2)
        bbox_height = 2 * np.sqrt((a * sin_angle)**2 + (b * cos_angle)**2)
        
        left = center_x - bbox_width / 2
        right = center_x + bbox_width / 2
        top = center_y - bbox_height / 2
        bottom = center_y + bbox_height / 2
        
        if left < 0 or right >= width or top < 0 or bottom >= height:
            return False
        
        return True
    
    # ========================================================================
    # SECTION 6: CRATER VALIDATION & CONFIDENCE SCORING
    # ========================================================================
    
    def validate_crater_appearance(self, image: np.ndarray, 
                                   ellipse_params: Tuple) -> float:
        """Validate if a detected region looks like a crater."""
        center_x, center_y, semi_major, semi_minor, rotation = ellipse_params
        
        if (np.isnan(center_x) or np.isnan(center_y) or 
            np.isnan(semi_major) or np.isnan(semi_minor)):
            return 0.0
        
        margin = 20
        x1 = max(0, int(center_x - semi_major - margin))
        x2 = min(image.shape[1], int(center_x + semi_major + margin))
        y1 = max(0, int(center_y - semi_minor - margin))
        y2 = min(image.shape[0], int(center_y + semi_minor + margin))
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return 0.0
        
        mask = np.zeros(region.shape, dtype=np.uint8)
        local_center = (int(center_x - x1), int(center_y - y1))
        
        cv2.ellipse(mask, local_center, (int(semi_major), int(semi_minor)),
                    rotation, 0, 360, 255, -1)
        
        crater_pixels = region[mask > 0]
        surrounding_pixels = region[mask == 0]
        
        if len(crater_pixels) == 0 or len(surrounding_pixels) == 0:
            return 0.0
        
        crater_mean = np.mean(crater_pixels)
        surround_mean = np.mean(surrounding_pixels)
        contrast = (surround_mean - crater_mean) / (surround_mean + 1)
        
        edges = cv2.Canny(region, 50, 150)
        edge_pixels = edges[mask > 0]
        edge_strength = np.sum(edge_pixels) / (len(edge_pixels) + 1)
        
        circularity = min(semi_minor, semi_major) / max(semi_minor, semi_major)
        
        confidence = 0.0
        if contrast > 0.05:
            confidence += 0.4
        if edge_strength > 0.1:
            confidence += 0.3
        if circularity > 0.6:
            confidence += 0.3
        
        return confidence
    
    # ========================================================================
    # SECTION 7: CRATER CLASSIFICATION
    # ========================================================================
    
    def classify_crater(self, ellipse_params: Tuple, image: np.ndarray,
                       confidence: float = None) -> int:
        """Classify crater based on rim crispness."""
        return -1  # Placeholder: No classification
    
    # ========================================================================
    # SECTION 8: SINGLE IMAGE DETECTION PIPELINE
    # ========================================================================
    
    def detect_craters_in_image(self, image_path: str, 
                                verbose: bool = False) -> List[Dict]:
        """Detect all craters in a single image."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            if verbose:
                print(f"Error: Could not read image {image_path}")
            return []
        
        preprocessed = self.preprocess_image(image)
        edges = self.detect_edges(preprocessed)
        contours = self.find_contours(edges)
        
        detected_craters = []
        
        for contour in contours:
            ellipse_params = self.fit_ellipse_to_contour(contour)
            
            if not self.filter_crater(ellipse_params, image.shape):
                continue
            
            confidence = self.validate_crater_appearance(image, ellipse_params)
            
            if confidence < self.confidence_threshold:
                continue
            
            center_x, center_y, semi_major, semi_minor, rotation = ellipse_params
            classification = self.classify_crater(ellipse_params, image, confidence)
            
            crater_dict = {
                'ellipseCenterX(px)': center_x,
                'ellipseCenterY(px)': center_y,
                'ellipseSemimajor(px)': semi_major,
                'ellipseSemiminor(px)': semi_minor,
                'ellipseRotation(deg)': rotation,
                'crater_classification': classification
            }
            
            detected_craters.append(crater_dict)
        
        if verbose:
            print(f"  Detected {len(detected_craters)} craters")
        
        return detected_craters
    
    # ========================================================================
    # SECTION 9: DATA FOLDER VALIDATION
    # ========================================================================
    
    def validate_data_folder(self, data_folder: str) -> Tuple[bool, str, List[Path]]:
        """Validate that the data folder exists and contains expected structure."""
        data_path = Path(data_folder)
        
        if not data_path.exists():
            return False, f"Error: Data folder '{data_folder}' does not exist.", []
        
        if not data_path.is_dir():
            return False, f"Error: '{data_folder}' is not a directory.", []
        
        altitude_folders = list(data_path.glob('altitude*'))
        
        if not altitude_folders:
            return False, (
                f"Error: No altitude folders found in '{data_folder}'.\n"
                f"Expected: {data_folder}/altitude*/longitude*/*.png"
            ), []
        
        image_files = []
        for altitude_folder in sorted(altitude_folders):
            for longitude_folder in sorted(altitude_folder.glob('longitude*')):
                for image_file in sorted(longitude_folder.glob('*.png')):
                    if '_mask' not in image_file.name and '_truth' not in image_file.name:
                        image_files.append(image_file)
        
        if not image_files:
            return False, "Error: No valid PNG images found.", []
        
        return True, "", image_files
    
    # ========================================================================
    # SECTION 10: BATCH DATASET PROCESSING
    # ========================================================================
    
    def process_dataset(self, data_folder: str, output_file: str, 
                       verbose: bool = True) -> bool:
        """Process entire dataset and generate CSV output."""
        is_valid, error_msg, image_files = self.validate_data_folder(data_folder)
        
        if not is_valid:
            print(error_msg)
            return False
        
        if verbose:
            print(f"Found {len(image_files)} images to process.")
            print(f"Processing...")
        
        results = []
        data_path = Path(data_folder)
        
        for idx, image_file in enumerate(image_files, 1):
            if verbose and idx % 10 == 0:
                print(f"  Progress: {idx}/{len(image_files)}")
            
            rel_path = image_file.relative_to(data_path)
            parts = rel_path.parts
            
            if len(parts) >= 3:
                image_id = f"{parts[0]}/{parts[1]}/{image_file.stem}"
            else:
                image_id = image_file.stem
            
            craters = self.detect_craters_in_image(str(image_file), verbose=False)
            
            if len(craters) == 0:
                results.append({
                    'ellipseCenterX(px)': -1,
                    'ellipseCenterY(px)': -1,
                    'ellipseSemimajor(px)': -1,
                    'ellipseSemiminor(px)': -1,
                    'ellipseRotation(deg)': -1,
                    'inputImage': image_id,
                    'crater_classification': -1
                })
            else:
                for crater in craters:
                    crater['inputImage'] = image_id
                    results.append(crater)
        
        if len(results) > 0:
            df = pd.DataFrame(results)
            column_order = [
                'ellipseCenterX(px)', 'ellipseCenterY(px)',
                'ellipseSemimajor(px)', 'ellipseSemiminor(px)',
                'ellipseRotation(deg)', 'inputImage', 'crater_classification'
            ]
            df = df[column_order]
            df.to_csv(output_file, index=False)
            
            if verbose:
                print(f"\n✓ Results saved to {output_file}")
                print(f"✓ Total detections: {len(results)}")
            
            return True
        else:
            print("No detections found.")
            return True


# ============================================================================
# HELPER FUNCTIONS SECTION
# ============================================================================

def load_ground_truth(data_folder: str) -> Optional[pd.DataFrame]:
    """Load ground truth from truth folders in the dataset."""
    data_path = Path(data_folder)
    all_truth_data = []
    
    truth_files = list(data_path.rglob('truth/detections.csv'))
    
    if not truth_files:
        return None
    
    for truth_file in truth_files:
        try:
            df = pd.read_csv(truth_file)
            
            longitude_folder = truth_file.parent.parent.name
            altitude_folder = truth_file.parent.parent.parent.name
            
            if 'inputImage' in df.columns:
                if not df['inputImage'].iloc[0].startswith('altitude'):
                    df['inputImage'] = df['inputImage'].apply(
                        lambda x: f"{altitude_folder}/{longitude_folder}/{Path(x).stem}"
                    )
            elif 'inputFile' in df.columns:
                df['inputImage'] = df['inputFile'].apply(
                    lambda x: f"{altitude_folder}/{longitude_folder}/{Path(x).stem}"
                )
            else:
                continue
            
            required_cols = ['ellipseCenterX(px)', 'ellipseCenterY(px)', 
                           'ellipseSemimajor(px)', 'ellipseSemiminor(px)', 
                           'ellipseRotation(deg)', 'inputImage']
            
            if 'crater_classification' in df.columns:
                required_cols.append('crater_classification')
            
            available_cols = [col for col in required_cols if col in df.columns]
            df = df[available_cols]
            
            all_truth_data.append(df)
        except Exception as e:
            print(f"Warning: Could not read {truth_file}: {e}")
    
    if all_truth_data:
        combined_df = pd.concat(all_truth_data, ignore_index=True)
        print(f"  Loaded {len(combined_df)} ground truth craters")
        return combined_df
    return None


def compare_with_ground_truth(detections_file: str, 
                              ground_truth: pd.DataFrame) -> Dict:
    """Compare detections with ground truth."""
    detections = pd.read_csv(detections_file)
    
    gt_images = set(ground_truth['inputImage'].unique())
    det_images = set(detections['inputImage'].unique())
    
    stats = {
        'ground_truth_craters': len(ground_truth),
        'detected_craters': len(detections),
        'ground_truth_images': len(gt_images),
        'detected_images': len(det_images),
        'common_images': len(gt_images & det_images),
        'missing_images': len(gt_images - det_images)
    }
    
    return stats


def generate_sample_data(output_folder: str) -> bool:
    """Generate sample test images with synthetic crater features."""
    output_path = Path(output_folder)
    
    try:
        sample_path = output_path / "altitude01" / "longitude01"
        sample_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating sample data in: {output_path}")
        
        for i in range(3):
            img_size = 512
            image = np.random.randint(80, 120, (img_size, img_size), dtype=np.uint8)
            
            noise = np.random.normal(0, 10, (img_size, img_size))
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
            
            num_craters = np.random.randint(2, 5)
            
            for j in range(num_craters):
                margin = 100
                center_x = np.random.randint(margin, img_size - margin)
                center_y = np.random.randint(margin, img_size - margin)
                
                semi_major = np.random.randint(50, 80)
                semi_minor = np.random.randint(45, semi_major + 1)
                rotation = np.random.randint(0, 180)
                
                cv2.ellipse(image, (center_x, center_y), (semi_major, semi_minor),
                           rotation, 0, 360, 50, 3)
                
                shadow_offset = 5
                cv2.ellipse(image, (center_x + shadow_offset, center_y + shadow_offset),
                           (max(semi_major - 5, 10), max(semi_minor - 5, 10)),
                           rotation, 180, 270, 40, 2)
                
                cv2.ellipse(image, (center_x, center_y),
                           (max(semi_major - 10, 5), max(semi_minor - 10, 5)),
                           rotation, 0, 360, 130, -1)
            
            image = cv2.GaussianBlur(image, (3, 3), 0)
            
            image_path = sample_path / f"orientation{i+1:02d}_light01.png"
            cv2.imwrite(str(image_path), image)
            print(f"  Created: {image_path}")
        
        print(f"\n✓ Sample data created successfully!")
        return True
        
    except Exception as e:
        print(f"Error creating sample data: {e}")
        return False


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run crater detection."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='NASA Lunar Crater Detection Challenge Solution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on training sample data and evaluate
  python crater_detector.py --data_folder ../train-sample --output detections.csv --evaluate

  # Run on test data for submission
  python crater_detector.py --data_folder ../test --output solution.csv

  # Auto-generate sample data and test
  python crater_detector.py --auto_generate

Expected data structure:
  data_folder/
  ├── altitude01/
  │   ├── longitude01/
  │   │   ├── orientation01_light01.png
  │   │   └── ...
  │   └── longitude02/
  └── altitude02/
        """
    )
    parser.add_argument('--data_folder', type=str, 
                       help='Path to data folder containing images')
    parser.add_argument('--output', type=str, default='solution.csv',
                       help='Output CSV file path (default: solution.csv)')
    parser.add_argument('--generate_sample', type=str, metavar='PATH',
                       help='Generate sample test data in the specified folder')
    parser.add_argument('--auto_generate', action='store_true',
                       help=f'Auto-generate sample data in "{DEFAULT_SAMPLE_FOLDER}/"')
    parser.add_argument('--evaluate', action='store_true',
                       help='Compare detections with ground truth (training data)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed progress information')
    parser.add_argument('--canny1', type=int, default=50, help='First threshold for the Canny edge detector.')
    parser.add_argument('--canny2', type=int, default=150, help='Second threshold for the Canny edge detector.')
    
    args = parser.parse_args()
    
    # Handle sample data generation
    if args.generate_sample:
        print("Generating sample test data...")
        success = generate_sample_data(args.generate_sample)
        sys.exit(0 if success else 1)
    
    # Handle auto_generate mode
    if args.auto_generate:
        if not args.data_folder:
            args.data_folder = DEFAULT_SAMPLE_FOLDER
        
        data_path = Path(args.data_folder)
        if not data_path.exists():
            print(f"Auto-generating sample data in '{args.data_folder}/'...")
            success = generate_sample_data(args.data_folder)
            if not success:
                print("Failed to generate sample data.")
                sys.exit(1)
            print()
    
    # Validate data_folder is provided
    if not args.data_folder:
        parser.print_help()
        print("\n" + "="*70)
        print("ERROR: No data folder specified!")
        print("="*70)
        print("\nQuick start:")
        print("  python crater_detector.py --data_folder ../train-sample --evaluate")
        sys.exit(1)
    
    # Create detector
    detector = CraterDetector(
        canny_th1=args.canny1,
        canny_th2=args.canny2
    )
    
    # Process dataset
    print(f"Starting crater detection on: {args.data_folder}")
    print("="*70)
    success = detector.process_dataset(args.data_folder, args.output, 
                                       verbose=args.verbose)
    
    if not success:
        print("\nDetection failed. Please check the error messages above.")
        sys.exit(1)
    
    # Evaluate against ground truth if requested
    if args.evaluate:
        print("\n" + "="*70)
        print("EVALUATION MODE")
        print("="*70)
        
        ground_truth = load_ground_truth(args.data_folder)
        
        if ground_truth is not None:
            print(f"✓ Loaded ground truth from truth folders")
            stats = compare_with_ground_truth(args.output, ground_truth)
            
            print("\nComparison Statistics:")
            print(f"  Ground Truth Craters: {stats['ground_truth_craters']}")
            print(f"  Detected Craters: {stats['detected_craters']}")
            print(f"  Ground Truth Images: {stats['ground_truth_images']}")
            print(f"  Detected Images: {stats['detected_images']}")
            print(f"  Common Images: {stats['common_images']}")
            print(f"  Missing Images: {stats['missing_images']}")
            
            print("\nNOTE: For precise scoring, use the official scorer.py:")
            print(f"  python ../scorer.py --pred {args.output} --truth <ground_truth.csv>")
        else:
            print("⚠ No ground truth found in truth/ folders")
            print("  This appears to be test data (no truth/ folders)")
    
    print("\n" + "="*70)
    print("✓ Detection complete!")
    print("="*70)


if __name__ == '__main__':
    main()