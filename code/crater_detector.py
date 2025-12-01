#!/usr/bin/env python3
"""
NASA Lunar Crater Detection Challenge Solution
Author: VimsRocz
Description: Computer vision solution for detecting and classifying crater rims
             in orbital lunar imagery using ellipse fitting algorithms.
"""

import cv2
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import List, Tuple, Dict


class CraterDetector:
    """
    Main crater detection class implementing ellipse fitting algorithms
    for lunar crater rim detection in orbital imagery.
    """
    
    def __init__(self, min_semi_minor_axis=40, max_crater_ratio=0.6):
        """
        Initialize crater detector with filtering parameters.
        
        Args:
            min_semi_minor_axis: Minimum semi-minor axis length (pixels)
            max_crater_ratio: Maximum crater bounding box ratio
        """
        self.min_semi_minor_axis = min_semi_minor_axis
        self.max_crater_ratio = max_crater_ratio
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the grayscale lunar image for crater detection.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Preprocessed image ready for edge detection
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges in the preprocessed image using Canny edge detection.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Binary edge map
        """
        # Apply Canny edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Apply morphological operations to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def find_contours(self, edges: np.ndarray) -> List:
        """
        Find contours in the edge map.
        
        Args:
            edges: Binary edge map
            
        Returns:
            List of contours
        """
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def fit_ellipse_to_contour(self, contour: np.ndarray) -> Tuple:
        """
        Fit an ellipse to a contour if it has sufficient points.
        
        Args:
            contour: Contour points
            
        Returns:
            Ellipse parameters (center_x, center_y, semi_major, semi_minor, rotation)
            or None if ellipse cannot be fitted
        """
        # Need at least 5 points to fit an ellipse
        if len(contour) < 5:
            return None
        
        try:
            # Fit ellipse
            ellipse = cv2.fitEllipse(contour)
            
            # Extract ellipse parameters
            center_x, center_y = ellipse[0]
            width, height = ellipse[1]  # These are diameters
            rotation = ellipse[2]
            
            # Convert to semi-axes
            semi_major = max(width, height) / 2.0
            semi_minor = min(width, height) / 2.0
            
            # Adjust rotation to be from x-axis (clockwise)
            if width < height:
                rotation = (rotation + 90) % 180
            
            return (center_x, center_y, semi_major, semi_minor, rotation)
            
        except cv2.error:
            return None
    
    def filter_crater(self, ellipse_params: Tuple, image_shape: Tuple) -> bool:
        """
        Apply filtering criteria to determine if detected ellipse is a valid crater.
        
        Args:
            ellipse_params: Ellipse parameters
            image_shape: Shape of the input image (height, width)
            
        Returns:
            True if crater passes all filters, False otherwise
        """
        if ellipse_params is None:
            return False
        
        center_x, center_y, semi_major, semi_minor, rotation = ellipse_params
        height, width = image_shape[:2]
        
        # Filter 1: Too small craters (semi-minor axis < 40 pixels)
        if semi_minor < self.min_semi_minor_axis:
            return False
        
        # Filter 2: Too large craters
        S = min(width, height)
        w = semi_major * 2
        h = semi_minor * 2
        if (w + h) >= (self.max_crater_ratio * S):
            return False
        
        # Filter 3: Not fully visible (bounding rectangle extends beyond image)
        # Calculate bounding rectangle for the ellipse
        cos_angle = np.cos(np.radians(rotation))
        sin_angle = np.sin(np.radians(rotation))
        
        # Bounding box calculation
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
    
    def classify_crater(self, ellipse_params: Tuple, image: np.ndarray) -> int:
        """
        Classify crater based on rim crispness (placeholder implementation).
        
        Args:
            ellipse_params: Ellipse parameters
            image: Input image
            
        Returns:
            Classification value (0-4 for A, AB, B, BC, C, or -1 for no classification)
        """
        # Placeholder: Return -1 (no classification)
        # In a full implementation, this would analyze rim pixel characteristics
        # to estimate rim steepness and classify accordingly
        return -1
    
    def detect_craters_in_image(self, image_path: str) -> List[Dict]:
        """
        Detect all craters in a single image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            List of detected crater dictionaries
        """
        # Read image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return []
        
        # Preprocess
        preprocessed = self.preprocess_image(image)
        
        # Detect edges
        edges = self.detect_edges(preprocessed)
        
        # Find contours
        contours = self.find_contours(edges)
        
        # Process each contour
        detected_craters = []
        
        for contour in contours:
            # Fit ellipse
            ellipse_params = self.fit_ellipse_to_contour(contour)
            
            # Apply filters
            if self.filter_crater(ellipse_params, image.shape):
                center_x, center_y, semi_major, semi_minor, rotation = ellipse_params
                
                # Classify crater
                classification = self.classify_crater(ellipse_params, image)
                
                crater_dict = {
                    'ellipseCenterX(px)': center_x,
                    'ellipseCenterY(px)': center_y,
                    'ellipseSemimajor(px)': semi_major,
                    'ellipseSemiminor(px)': semi_minor,
                    'ellipseRotation(deg)': rotation,
                    'crater_classification': classification
                }
                
                detected_craters.append(crater_dict)
        
        return detected_craters
    
    def validate_data_folder(self, data_folder: str) -> Tuple[bool, str, List[Path]]:
        """
        Validate that the data folder exists and contains the expected structure.
        
        Args:
            data_folder: Path to folder containing test images
            
        Returns:
            Tuple of (is_valid, error_message, list_of_image_files)
        """
        data_path = Path(data_folder)
        
        # Check if path exists
        if not data_path.exists():
            return False, f"Error: Data folder '{data_folder}' does not exist.", []
        
        # Check if it's a directory
        if not data_path.is_dir():
            return False, f"Error: '{data_folder}' is not a directory.", []
        
        # Look for altitude folders
        altitude_folders = list(data_path.glob('altitude*'))
        
        if not altitude_folders:
            # Check if there are any PNG images directly or in subdirectories
            all_pngs = list(data_path.rglob('*.png'))
            if all_pngs:
                return False, (
                    f"Error: Found {len(all_pngs)} PNG files, but the folder structure is incorrect.\n"
                    f"Expected structure: {data_folder}/altitude*/longitude*/*.png\n"
                    f"Found images at: {all_pngs[0].parent if all_pngs else 'N/A'}"
                ), []
            else:
                return False, (
                    f"Error: No data found in '{data_folder}'.\n"
                    f"Expected folder structure: {data_folder}/altitude*/longitude*/*.png\n"
                    f"Please ensure your data follows this hierarchical structure."
                ), []
        
        # Collect all valid image files
        image_files = []
        for altitude_folder in sorted(altitude_folders):
            for longitude_folder in sorted(altitude_folder.glob('longitude*')):
                for image_file in sorted(longitude_folder.glob('*.png')):
                    if '_mask' not in image_file.name and '_truth' not in image_file.name:
                        image_files.append(image_file)
        
        if not image_files:
            return False, (
                f"Error: Found altitude folders but no valid PNG images.\n"
                f"Looking for: {data_folder}/altitude*/longitude*/*.png\n"
                f"Excluding files with '_mask' or '_truth' in the name."
            ), []
        
        return True, "", image_files

    def process_dataset(self, data_folder: str, output_file: str) -> bool:
        """
        Process entire dataset and generate CSV output.
        
        Args:
            data_folder: Path to folder containing test images
            output_file: Path to output CSV file
            
        Returns:
            True if processing was successful, False otherwise
        """
        # Validate data folder first
        is_valid, error_msg, image_files = self.validate_data_folder(data_folder)
        
        if not is_valid:
            print(error_msg)
            print("\nTo get started, you can:")
            print("1. Download the official NASA crater dataset from Topcoder")
            print("2. Use --generate_sample to create sample test data")
            print("3. Use --auto_generate to automatically generate sample data and run detection")
            print("4. Ensure your data follows the required structure:")
            print("   data_folder/")
            print("   ├── altitude01/")
            print("   │   ├── longitude01/")
            print("   │   │   ├── image1.png")
            print("   │   │   └── image2.png")
            print("   │   └── longitude02/")
            print("   │       └── ...")
            print("   └── altitude02/")
            print("       └── ...")
            return False
        
        print(f"Found {len(image_files)} images to process.")
        
        results = []
        data_path = Path(data_folder)
        
        for image_file in image_files:
            print(f"Processing: {image_file}")
            
            # Construct image ID from folder structure
            rel_path = image_file.relative_to(data_path)
            parts = rel_path.parts
            if len(parts) >= 3:
                altitude = parts[0]
                longitude = parts[1]
                filename = image_file.stem  # filename without extension
                image_id = f"{altitude}/{longitude}/{filename}"
            else:
                # Fallback for unexpected structure
                image_id = image_file.stem
            
            # Detect craters
            craters = self.detect_craters_in_image(str(image_file))
            
            if len(craters) == 0:
                # No craters detected - add special case entry
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
                # Add image ID to each crater detection
                for crater in craters:
                    crater['inputImage'] = image_id
                    results.append(crater)
        
        # Write results to CSV
        if len(results) > 0:
            df = pd.DataFrame(results)
            # Reorder columns
            column_order = [
                'ellipseCenterX(px)',
                'ellipseCenterY(px)',
                'ellipseSemimajor(px)',
                'ellipseSemiminor(px)',
                'ellipseRotation(deg)',
                'inputImage',
                'crater_classification'
            ]
            df = df[column_order]
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")
            print(f"Total detections: {len(results)}")
            return True
        else:
            print("No detections found in the processed images.")
            return True


def generate_sample_data(output_folder: str) -> bool:
    """
    Generate sample test images with synthetic crater-like features
    for testing the crater detection pipeline.
    
    Args:
        output_folder: Path to folder where sample data will be created
        
    Returns:
        True if generation was successful, False otherwise
    """
    output_path = Path(output_folder)
    
    try:
        # Create directory structure
        sample_path = output_path / "altitude01" / "longitude01"
        sample_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating sample data in: {output_path}")
        
        # Generate sample images with synthetic craters
        for i in range(3):
            # Create a grayscale image simulating lunar surface
            img_size = 512
            image = np.random.randint(80, 120, (img_size, img_size), dtype=np.uint8)
            
            # Add some noise/texture
            noise = np.random.normal(0, 10, (img_size, img_size))
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
            
            # Add 2-4 synthetic craters per image
            num_craters = np.random.randint(2, 5)
            
            for j in range(num_craters):
                # Random crater position (ensuring it's fully visible)
                margin = 100
                center_x = np.random.randint(margin, img_size - margin)
                center_y = np.random.randint(margin, img_size - margin)
                
                # Random crater size (semi-axes) - ensure minimum values for interior drawing
                semi_major = np.random.randint(50, 80)
                semi_minor = np.random.randint(45, semi_major + 1)
                
                # Random rotation
                rotation = np.random.randint(0, 180)
                
                # Draw crater rim (ellipse)
                cv2.ellipse(
                    image,
                    (center_x, center_y),
                    (semi_major, semi_minor),
                    rotation,
                    0, 360,
                    color=50,  # Darker rim
                    thickness=3
                )
                
                # Add shadow effect (darker on one side)
                shadow_offset = 5
                cv2.ellipse(
                    image,
                    (center_x + shadow_offset, center_y + shadow_offset),
                    (max(semi_major - 5, 10), max(semi_minor - 5, 10)),
                    rotation,
                    180, 270,
                    color=40,
                    thickness=2
                )
                
                # Add lighter interior
                cv2.ellipse(
                    image,
                    (center_x, center_y),
                    (max(semi_major - 10, 5), max(semi_minor - 10, 5)),
                    rotation,
                    0, 360,
                    color=130,
                    thickness=-1
                )
            
            # Apply slight blur to make it more realistic
            image = cv2.GaussianBlur(image, (3, 3), 0)
            
            # Save the image
            image_path = sample_path / f"orientation{i+1:02d}_light01.png"
            cv2.imwrite(str(image_path), image)
            print(f"  Created: {image_path}")
        
        print(f"\nSample data created successfully!")
        print(f"You can now run: python crater_detector.py --data_folder {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating sample data: {e}")
        return False


def main():
    """
    Main function to run crater detection.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='NASA Lunar Crater Detection Challenge Solution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate sample data for testing
  python crater_detector.py --generate_sample ./sample_data

  # Run detection on sample data
  python crater_detector.py --data_folder ./sample_data --output solution.csv

  # Run detection on real data
  python crater_detector.py --data_folder /path/to/nasa/data --output solution.csv

  # Auto-generate sample data if data folder doesn't exist, then run detection
  python crater_detector.py --data_folder /path/to/test/data --auto_generate --output solution.csv

Expected data structure:
  data_folder/
  ├── altitude01/
  │   ├── longitude01/
  │   │   ├── orientation01_light01.png
  │   │   └── orientation01_light02.png
  │   └── longitude02/
  │       └── ...
  └── altitude02/
      └── ...
        """
    )
    parser.add_argument(
        '--data_folder',
        type=str,
        help='Path to test data folder containing images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='solution.csv',
        help='Output CSV file path (default: solution.csv)'
    )
    parser.add_argument(
        '--generate_sample',
        type=str,
        metavar='PATH',
        help='Generate sample test data in the specified folder for testing'
    )
    parser.add_argument(
        '--auto_generate',
        action='store_true',
        help='Automatically generate sample data in "sample_data/" folder if data_folder does not exist, then run detection'
    )
    
    args = parser.parse_args()
    
    # Handle sample data generation
    if args.generate_sample:
        print("Generating sample test data...")
        success = generate_sample_data(args.generate_sample)
        if not success:
            sys.exit(1)
        return
    
    # Validate that data_folder is provided for detection
    if not args.data_folder:
        parser.error("--data_folder is required for crater detection. "
                    "Use --generate_sample to create test data first.")
    
    # Handle auto_generate: if data folder doesn't exist, generate sample data
    data_path = Path(args.data_folder)
    if not data_path.exists() and args.auto_generate:
        sample_folder = "sample_data"
        print(f"Data folder '{args.data_folder}' does not exist.")
        print(f"Auto-generating sample data in '{sample_folder}/'...")
        success = generate_sample_data(sample_folder)
        if not success:
            print("Failed to generate sample data.")
            sys.exit(1)
        # Use the generated sample_data folder instead
        args.data_folder = sample_folder
        print()
    
    # Create detector
    detector = CraterDetector()
    
    # Process dataset
    print("Starting crater detection...")
    success = detector.process_dataset(args.data_folder, args.output)
    if success:
        print("Detection complete!")
    else:
        print("\nDetection could not be completed. Please check the error messages above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
