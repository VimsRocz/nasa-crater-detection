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
import os
from pathlib import Path
from typing import List, Tuple, Dict
import csv


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
    
    def process_dataset(self, data_folder: str, output_file: str):
        """
        Process entire dataset and generate CSV output.
        
        Args:
            data_folder: Path to folder containing test images
            output_file: Path to output CSV file
        """
        results = []
        
        # Walk through directory structure
        data_path = Path(data_folder)
        
        for altitude_folder in sorted(data_path.glob('altitude*')):
            for longitude_folder in sorted(altitude_folder.glob('longitude*')):
                # Process all PNG images in this folder
                for image_file in sorted(longitude_folder.glob('*.png')):
                    # Skip mask and truth files
                    if '_mask' in image_file.name or '_truth' in image_file.name:
                        continue
                    
                    print(f"Processing: {image_file}")
                    
                    # Construct image ID
                    altitude = altitude_folder.name
                    longitude = longitude_folder.name
                    filename = image_file.stem  # filename without extension
                    image_id = f"{altitude}/{longitude}/{filename}"
                    
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
        else:
            print("No results to save.")


def main():
    """
    Main function to run crater detection.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='NASA Lunar Crater Detection Challenge Solution'
    )
    parser.add_argument(
        '--data_folder',
        type=str,
        required=True,
        help='Path to test data folder'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='solution.csv',
        help='Output CSV file path (default: solution.csv)'
    )
    
    args = parser.parse_args()
    
    # Create detector
    detector = CraterDetector()
    
    # Process dataset
    print("Starting crater detection...")
    detector.process_dataset(args.data_folder, args.output)
    print("Detection complete!")


if __name__ == '__main__':
    main()
