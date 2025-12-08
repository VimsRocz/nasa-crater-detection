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
# DATA PATHS AND LOADING SECTION
# ============================================================================ 

# Base data directory - contains all datasets
DATA_ROOT = Path("data")

# Dataset paths - three main datasets available in the project
DATASET_PATHS = {
    'sample-submission': DATA_ROOT / "sample-submission",
    'test': DATA_ROOT / "test",
    'train': DATA_ROOT / "train",
    'train-sample': DATA_ROOT / "train-sample"  # Small training subset
}

# Dataset descriptions for user guidance
DATASET_INFO = {
    'train-sample': {
        'name': 'Training Sample',
        'description': 'Small training set (50 images, ~234 MB)',
        'has_ground_truth': True,
        'purpose': 'Quick testing and parameter tuning',
        'suggested_output': 'results/train-sample_detections.csv'
    },
    'train': {
        'name': 'Full Training Set',
        'description': 'Complete training data (requires extracting train.tar, ~19 GB)',
        'has_ground_truth': True,
        'purpose': 'Full model evaluation and validation',
        'suggested_output': 'results/train_full_detections.csv'
    },
    'test': {
        'name': 'Test Set',
        'description': 'Test data for submission (requires extracting test.tar, ~3.3 GB)',
        'has_ground_truth': False,
        'purpose': 'Generate final submission file',
        'suggested_output': 'results/solution.csv'
    },
    'sample-submission': {
        'name': 'Sample Submission',
        'description': 'Example submission format and reference code',
        'has_ground_truth': False,
        'purpose': 'Reference for submission format',
        'suggested_output': 'results/sample_solution.csv'
    }
}


class DatasetLoader:
    """
    Handles loading and validation of NASA crater detection datasets.
    Manages paths for train, test, and sample-submission datasets.
    """
    
    def __init__(self, data_root: str = "data"):
        """
        Initialize the dataset loader. 
        
        Args:
            data_root: Root directory containing all datasets (default: "data")
        """
        self.data_root = Path(data_root)
        self.available_datasets = self._discover_datasets()
    
    def _discover_datasets(self) -> Dict[str, Dict]:
        """
        Discover available datasets in the data directory. 
        
        Returns:
            Dictionary mapping dataset names to their info and status
        """
        datasets = {}
        
        for dataset_key, dataset_path in DATASET_PATHS.items():
            info = DATASET_INFO.get(dataset_key, {}).copy()
            info['path'] = dataset_path
            info['exists'] = dataset_path.exists()
            info['key'] = dataset_key
            
            # Check for altitude folders (indicates extracted/valid dataset)
            if info['exists']:
                altitude_folders = list(dataset_path.glob('altitude*'))
                info['is_valid'] = len(altitude_folders) > 0
                info['altitude_count'] = len(altitude_folders)
            else:
                info['is_valid'] = False
                info['altitude_count'] = 0
            
            datasets[dataset_key] = info
        
        return datasets
    
    def get_dataset_path(self, dataset_name: str) -> Optional[Path]:
        """
        Get the full path for a dataset. 
        
        Args:
            dataset_name: Name of dataset ('train', 'test', 'train-sample', 'sample-submission')
        
        Returns:
            Path object if dataset exists, None otherwise
        """
        if dataset_name in self.available_datasets:
            dataset_info = self.available_datasets[dataset_name]
            if dataset_info['exists']:
                return dataset_info['path']
        return None
    
    def validate_dataset(self, dataset_name: str) -> Tuple[bool, str]:
        """
        Validate that a dataset exists and has proper structure. 
        
        Args:
            dataset_name: Name of dataset to validate
        
        Returns:
            Tuple of (is_valid, message)
        """
        if dataset_name not in self.available_datasets:
            available = ', '.join(self.available_datasets.keys())
            return False, f"Unknown dataset '{dataset_name}'. Available: {available}"
        
        dataset_info = self.available_datasets[dataset_name]
        
        if not dataset_info['exists']:
            return False, f"Dataset '{dataset_name}' not found at {dataset_info['path']}"
        
        if not dataset_info['is_valid']:
            return False, (
                f"Dataset '{dataset_name}' exists but has no altitude folders. \n"
                f"Expected structure: {dataset_info['path']}/altitude*/longitude*/*.png\n"
                f"You may need to extract the TAR archive."
            )
        
        return True, f"Dataset '{dataset_name}' is valid ({dataset_info['altitude_count']} altitude folders found)"
    
    def list_available_datasets(self) -> List[Dict]:
        """
        Get list of all available datasets with their status. 
        
        Returns:
            List of dataset information dictionaries
        """
        return [
            {
                'key': key,
                'name': info.get('name', key),
                'description': info.get('description', ''),
                'exists': info['exists'],
                'is_valid': info['is_valid'],
                'has_ground_truth': info.get('has_ground_truth', False),
                'path': str(info['path']),
                'suggested_output': info.get('suggested_output', 'results/detections.csv')
            }
            for key, info in self.available_datasets.items()
        ]
    
    def get_dataset_images(self, dataset_name: str) -> List[Path]:
        """
        Get all image files from a dataset. 
        
        Args:
            dataset_name: Name of dataset
        
        Returns:
            List of Path objects for all PNG images (excludes masks and truth overlays)
        """
        dataset_path = self.get_dataset_path(dataset_name)
        if not dataset_path:
            return []
        
        image_files = []
        for altitude_folder in sorted(dataset_path.glob('altitude*')):
            for longitude_folder in sorted(altitude_folder.glob('longitude*')):
                for image_file in sorted(longitude_folder.glob('*.png')):
                    # Exclude mask and truth overlay images
                    if '_mask' not in image_file.name and '_truth' not in image_file.name:
                        image_files.append(image_file)
        
        return image_files


# ============================================================================ 
# ALGORITHM CONFIGURATION SECTION
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
                 canny_th1=50, canny_th2=150,
                 circularity_threshold=0.6):
        self.min_semi_minor_axis = min_semi_minor_axis
        self.max_crater_ratio = max_crater_ratio
        self.confidence_threshold = confidence_threshold
        self.canny_th1 = canny_th1
        self.canny_th2 = canny_th2
        self.circularity_threshold = circularity_threshold
    
    # ======================================================================== 
    # SECTION 1: IMAGE PREPROCESSING
    # ======================================================================== 
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the grayscale lunar image for crater detection. 
        
        Steps:
        1. Gaussian blur to reduce high-frequency noise
        2. CLAHE to enhance local contrast (especially important for shadows) 
        
        Args:
            image: Input grayscale image (numpy array)
        
        Returns:
            Enhanced image ready for edge detection
        """
        # Apply Gaussian blur to reduce noise while preserving edges
        # Kernel size (5,5) is a balance between noise reduction and detail preservation
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Enhances local contrast, critical for detecting craters in varying lighting
        # clipLimit=2.0: prevents over-amplification of noise
        # tileGridSize=(8,8): processes image in 8x8 regions for local enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        return enhanced
    
    # ======================================================================== 
    # SECTION 2: EDGE DETECTION
    # ======================================================================== 
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges using Canny edge detection algorithm. 
        
        Challenge: Crater rims often have broken/incomplete edges due to:
        - Shadows and lighting variations
        - Surface degradation and erosion
        - Image noise and artifacts
        
        Solution: Use Canny edge detection + morphological closing to connect gaps
        
        Args:
            image: Preprocessed grayscale image
        
        Returns:
            Binary edge map where white pixels indicate edges
        """
        # Canny edge detection with configurable thresholds
        # Lower threshold (canny_th1): minimum gradient for edge consideration
        # Upper threshold (canny_th2): strong edge confirmation
        # Pixels between thresholds are edges only if connected to strong edges
        edges = cv2.Canny(image, self.canny_th1, self.canny_th2)
        
        # Morphological closing: connects nearby edge pixels
        # Critical for handling broken crater rims
        # Ellipse kernel better matches circular/elliptical crater shapes
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
        """
        Apply filtering criteria per NASA challenge rules. 
        
        NASA Challenge Filtering Requirements:
        1. Too small craters: semi_minor < 40 pixels
        2. Too large craters: (width + height) >= 0.6 * min(image_width, image_height)
        3. Partially visible: any part of bounding box extends beyond image boundaries
        
        These filters ensure we only detect craters that:
        - Are large enough to be reliably measured
        - Are not so large they dominate the scene
        - Are fully visible (complete ellipse within frame)
        
        Args:
            ellipse_params: Tuple of (center_x, center_y, semi_major, semi_minor, rotation)
            image_shape: Shape of the image (height, width)
        
        Returns:
            bool: True if crater passes all filters, False if rejected
        """
        if ellipse_params is None:
            return False
        
        center_x, center_y, semi_major, semi_minor, rotation = ellipse_params
        height, width = image_shape[:2]
        
        # FILTER 1: Reject craters that are too small to measure accurately
        # Challenge requirement: minimum semi-minor axis = 40 pixels
        if semi_minor < self.min_semi_minor_axis:
            return False
        
        # FILTER 2: Reject craters that are too large relative to image size
        # Rationale: Very large craters may not fit entirely, or dominate the scene
        # S = minimum image dimension (handles non-square images)
        S = min(width, height)
        w = semi_major * 2  # Full width of ellipse
        h = semi_minor * 2  # Full height of ellipse
        # Challenge rule: (width + height) must be < 60% of smallest dimension
        if (w + h) >= (self.max_crater_ratio * S):
            return False
        
        # FILTER 3: Reject partially visible craters (must be fully in frame)
        # Calculate axis-aligned bounding box considering rotation
        cos_angle = np.cos(np.radians(rotation))
        sin_angle = np.sin(np.radians(rotation))
        
        a = semi_major
        b = semi_minor
        # Bounding box dimensions accounting for rotation
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
        if circularity > self.circularity_threshold:
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
            print("Processing...")
        
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

def display_data_configuration():
    """
    Automatically discover and display all data paths and results configuration.
    Shows dataset availability and paths at script startup.
    """
    print("\n" + "="*70)
    print("DATA CONFIGURATION - AUTOMATIC DISCOVERY")
    print("="*70)
    
    # Initialize dataset loader
    loader = DatasetLoader()
    
    print(f"\n📁 DATA ROOT: {DATA_ROOT.absolute()}")
    print("\n📂 AVAILABLE DATASETS:")
    print("-" * 70)
    
    for dataset_key in ['train-sample', 'train', 'test', 'sample-submission']:
        if dataset_key in loader.available_datasets:
            info = loader.available_datasets[dataset_key]
            status_icon = "✓" if info['exists'] and info['is_valid'] else "✗"
            
            print(f"\n  {status_icon} {info.get('name', dataset_key).upper()}")
            print(f"     Path: {info['path']}")
            
            if info['exists'] and info['is_valid']:
                print(f"     Status: Ready ({info['altitude_count']} altitude folders)")
                image_count = len(loader.get_dataset_images(dataset_key))
                print(f"     Images: {image_count} files")
                if info.get('has_ground_truth'):
                    print(f"     Ground Truth: Available (use --evaluate flag)")
            elif info['exists'] and not info['is_valid']:
                print(f"     Status: ⚠ Needs extraction (folder exists but no altitude/* structure)")
            else:
                print(f"     Status: Not found (may need to extract TAR archive)")
            
            print(f"     Suggested Output: {info.get('suggested_output', 'N/A')}")
    
    print("\n" + "-" * 70)
    
    print("\n📊 RESULTS OUTPUT DIRECTORY:")
    results_path = Path("results")
    if results_path.exists():
        print(f"     Path: {results_path.absolute()}")
        csv_files = list(results_path.glob('*.csv'))
        print(f"     Existing results: {len(csv_files)} CSV files")
    else:
        print(f"     Path: {results_path.absolute()} (will be created)")
    
    print("\n" + "="*70)
    return loader


def select_data_folder() -> Optional[str]:
    """
    Interactively prompt user to select a data folder from available options.
    Uses DatasetLoader to discover and validate datasets. 
    
    Returns:
        str: Path to the selected data folder, or None if cancelled
    """
    print("\n" + "="*70)
    print("DATA FOLDER SELECTION")
    print("="*70)
    
    # Initialize dataset loader to discover available datasets
    loader = DatasetLoader()
    datasets = loader.list_available_datasets()
    
    # Filter to show main datasets (train-sample, train, test)
    main_datasets = [d for d in datasets if d['key'] in ['train-sample', 'train', 'test']]
    
    print("\nAvailable datasets in 'data/' folder:")
    print("\nStructure: data/")
    print("           ├── sample-submission/  (reference format)")
    print("           ├── test/               (for final submission)")
    print("           ├── train/              (full training set)")
    print("           └── train-sample/       (small training subset)")
    print("\n" + "-" * 70)
    
    for idx, dataset in enumerate(main_datasets, 1):
        # Determine status with validation
        if dataset['exists'] and dataset['is_valid']:
            status = "✓ Ready"
            status_color = ""
        elif dataset['exists'] and not dataset['is_valid']:
            status = "⚠ Needs extraction"
            status_color = ""
        else:
            status = "✗ Not found"
            status_color = ""
        
        print(f"  {idx}. {dataset['key']:<20} - {dataset['description']}")
        print(f"     Path: {dataset['path']}")
        print(f"     Status: {status}")
        print(f"     Purpose: {DATASET_INFO[dataset['key']]['purpose']}")
        if dataset['has_ground_truth']:
            print(f"     Ground Truth: Available (use --evaluate flag)")
        print()
    
    print("  0. Cancel / Enter custom path")
    print("="*70)
    
    while True:
        try:
            choice = input(f"\nSelect dataset (0-{len(main_datasets)}): ").strip()
            
            if choice == '0':
                manual_path = input("Enter data folder path (or press Enter to cancel): ").strip()
                if not manual_path:
                    return None
                return manual_path
            
            choice_idx = int(choice)
            if 1 <= choice_idx <= len(main_datasets):
                selected = main_datasets[choice_idx - 1]
                selected_path = selected['path']
                
                # Validate the selected dataset
                is_valid, message = loader.validate_dataset(selected['key'])
                
                if not is_valid:
                    print(f"\n⚠ Warning: {message}")
                    retry = input("Use this path anyway? (y/n): ").strip().lower()
                    if retry != 'y':
                        continue
                else:
                    print(f"\n✓ {message}")
                
                return selected_path
            else:
                print(f"Invalid option. Please select 0-{len(main_datasets)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nSelection cancelled.")
            return None


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


def select_output_file(data_folder: str = None, default_name: str = "solution.csv") -> str:
    """
    Prompt user to specify output CSV file name with path.
    Suggests appropriate filenames based on the selected dataset.
    
    Args:
        data_folder: Path to the data folder being processed (for smart suggestions)
        default_name: Default filename if user presses Enter
    
    Returns:
        str: Path to output CSV file
    """
    print("\n" + "="*70)
    print("OUTPUT FILE SELECTION")
    print("="*70)
    
    # Smart suggestion based on data folder
    suggested_default = default_name
    if data_folder:
        data_path = Path(data_folder)
        if 'train-sample' in str(data_path):
            suggested_default = "results\train-sample_detections.csv"
        elif 'train' in str(data_path) and 'sample' not in str(data_path):
            suggested_default = "results\train_full_detections.csv"
        elif 'test' in str(data_path):
            suggested_default = "results\solution.csv"
    
    suggested_outputs = [
        ("results\train-sample_detections.csv", "For data\train-sample"),
        ("results\train_full_detections.csv", "For data\train (full set)"),
        ("results\solution.csv", "For data\test (submission)"),
        ("results\custom_detections.csv", "Custom output name")
    ]
    
    print(f"\nData folder: {data_folder if data_folder else 'Not specified'}")
    print(f"Recommended: {suggested_default}\n")
    
    print("\nSuggested output files:")
    for idx, (filepath, description) in enumerate(suggested_outputs, 1):
        print(f"  {idx}. {filepath:<45} - {description}")
    
    print(f"\n  0. Use default: {default_name}")
    print("="*70)
    
    while True:
        try:
            choice = input(f"\nSelect option (0-{len(suggested_outputs)}) or enter custom path: ").strip()
            
            if choice == '0' or choice == '':
                return default_name
            
            # Try to parse as integer selection
            try:
                choice_idx = int(choice)
                if 1 <= choice_idx <= len(suggested_outputs):
                    return suggested_outputs[choice_idx - 1][0]
                else:
                    print(f"Invalid option. Please select 0-{len(suggested_outputs)}.")
            except ValueError:
                # User entered a custom path
                if choice.endswith('.csv'):
                    return choice
                else:
                    print("Output file must end with .csv")
        except KeyboardInterrupt:
            print(f"\n\nUsing default: {default_name}")
            return default_name


def generate_sample_data(output_folder: str) -> bool:
    """
    Generate synthetic sample test images with crater features.
    Useful for quick testing without downloading the full dataset. 
    
    Args:
        output_folder: Directory where sample images will be created
    
    Returns:
        bool: True if generation successful, False otherwise
    """
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


def analyze_detections(file_path):
    """
    Analyzes the crater detection results for potential issues.
    """
    print(f"Analyzing detection results from: {file_path}\n")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return

    # Filter out placeholder rows for images with no detections
    df_valid = df[df['ellipseCenterX(px)'] != -1].copy()

    if df_valid.empty:
        print("No valid detections found in the file.")
        return

    print("--- Descriptive Statistics for Ellipse Axes (in pixels) ---\\n")
    
    # Calculate aspect ratio
    df_valid['aspect_ratio'] = df_valid['ellipseSemimajor(px)'] / df_valid['ellipseSemiminor(px)']
    
    stats = df_valid[['ellipseSemimajor(px)', 'ellipseSemiminor(px)', 'aspect_ratio']].describe()
    print(stats)
    print("\n" + "="*50 + "\n")

    # --- Outlier Detection ---
    # According to the challenge rules, craters are filtered if:
    # (width + height) >= (0.6 * S), where S is min(image_width, image_height)
    # The images are 2048x2048, so S = 2048.
    # Max (w+h) = 0.6 * 2048 = 1228.8
    # w = 2 * semi_major, h = 2 * semi_minor
    # 2 * (semi_major + semi_minor) >= 1228.8
    # semi_major + semi_minor >= 614.4
    
    image_dim = 2048
    max_size_threshold = 0.6 * image_dim / 2 
    
    df_valid['size_sum'] = df_valid['ellipseSemimajor(px)'] + df_valid['ellipseSemiminor(px)']
    
    large_craters = df_valid[df_valid['size_sum'] > max_size_threshold]
    
    print(f"--- Potential Outliers (Ellipses larger than challenge filter) ---\\n")
    print(f"The detection script should already filter craters where (semi_major + semi_minor) >= {max_size_threshold:.1f} pixels.")
    
    if not large_craters.empty:
        print(f"Found {len(large_craters)} detections that might be too large:")
        print(large_craters[['inputImage', 'ellipseSemimajor(px)', 'ellipseSemiminor(px)', 'size_sum']].to_string())
    else:
        print("No detections appear to violate the maximum size filter.")
        
    print("\n" + "="*50 + "\n")
    
    # Check for unusually high aspect ratios (very elongated ellipses)
    aspect_threshold = 10
    elongated_craters = df_valid[df_valid['aspect_ratio'] > aspect_threshold]
    
    print(f"--- Unusually Elongated Ellipses (Aspect Ratio > {aspect_threshold}) ---\\n")
    
    if not elongated_craters.empty:
        print(f"Found {len(elongated_craters)} detections that are very elongated:")
        print(elongated_craters[['inputImage', 'ellipseSemimajor(px)', 'ellipseSemiminor(px)', 'aspect_ratio']].to_string())
    else:
        print("No unusually elongated ellipses found.")


# Scorer functions and constants
import os
import math

outDir = ''

XI_2_THRESH = 13.277
NN_PIX_ERR_RATIO = 0.07
    
def calcYmat(a, b, phi):
    unit_1 = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    unit_2 = np.array([[1 / (a ** 2), 0], [0, 1 / (b ** 2)]])
    unit_3 = np.array([[math.cos(phi), math.sin(phi)], [-math.sin(phi), math.cos(phi)]])
    return unit_1 @ unit_2 @ unit_3

def calc_dGA(Yi, Yj, yi, yj):
    multiplicand = 4 * np.sqrt(np.linalg.det(Yi) * np.linalg.det(Yj)) / np.linalg.det(Yi + Yj)
    exponent = (-0.5 * (yi - yj).T @ Yi @ np.linalg.inv(Yi + Yj) @ Yj @ (yi - yj))
    e = exponent[0, 0]
    cos = multiplicand * np.exp(e)
    cos = min(1, cos)
    return np.arccos(cos)

def dGA(crater_A, crater_B):
    
    A_a = crater_A['ellipseSemimajor(px)']
    A_b = crater_A['ellipseSemiminor(px)']
    A_xc = crater_A['ellipseCenterX(px)']
    A_yc = crater_A['ellipseCenterY(px)']
    A_phi = crater_A['ellipseRotation(deg)'] / 180 * math.pi

    B_a = crater_B['ellipseSemimajor(px)']
    B_b = crater_B['ellipseSemiminor(px)']
    B_xc = crater_B['ellipseCenterX(px)']
    B_yc = crater_B['ellipseCenterY(px)']
    B_phi = crater_B['ellipseRotation(deg)'] / 180 * math.pi

    A_Y = calcYmat(A_a, A_b, A_phi)
    B_Y = calcYmat(B_a, B_b, B_phi)
    
    A_y = np.array([[A_xc], [A_yc]])
    B_y = np.array([[B_xc], [B_yc]])

    dGA = calc_dGA(A_Y, B_Y, A_y, B_y)

    ab_min = np.min([A_a, A_b])
    comparison_sig = NN_PIX_ERR_RATIO * ab_min
    ref_sig = 0.85 / np.sqrt(A_a * A_b) * comparison_sig
    xi_2 = dGA * dGA / (ref_sig * ref_sig)
    
    return dGA, xi_2


def writeScore(s, out_dir):
    # This function's outDir is globally defined in the scorer.py context.
    # We will need to decide if we want to retain this global behavior or
    # pass out_dir as an argument. For now, assuming global.
    path = os.path.join(out_dir, 'result.txt')
    os.makedirs(os.path.dirname(path), exist_ok=True) # Ensure directory exists
    out = open(path, 'w')
    out.write(str(s))
    out.close()

def score1(ts, ps):
    if len(ps) == 0:
        return 0.0
    t_empty = False
    p_empty = False
    if len(ts) == 1 and ts[0].get('ellipseSemimajor(px)') == -1:
        t_empty = True
    if len(ps) == 1 and ps[0].get('ellipseSemimajor(px)') == -1:
        p_empty = True
    if t_empty and p_empty:
        return 1.0
    if t_empty != p_empty:
        return 0.0
    
    dgas = []

    for t in ts:
        # find best matching prediction
        best_p = None
        best_dGA = math.pi / 2
        best_xi_2 = float('inf')

        for p in ps:
            if p['matched']:
                continue
            # short-circuit checks
            rA = min(t['ellipseSemimajor(px)'], t['ellipseSemiminor(px)'])
            rB = min(p['ellipseSemimajor(px)'], p['ellipseSemiminor(px)'])
            if rA > 1.5 * rB or rB > 1.5 * rA:
                continue
            r = min(rA, rB )
            if abs(t['ellipseCenterX(px)'] - p['ellipseCenterX(px)']) > r:
                continue
            if abs(t['ellipseCenterY(px)'] - p['ellipseCenterY(px)']) > r:
                continue
            d, xi_2 = dGA(t, p)
            if d < best_dGA:
                best_dGA = d
                best_p = p
                best_xi_2 = xi_2
        if best_xi_2 < XI_2_THRESH: # matched
            t['matched'] = True
            best_p['matched'] = True
            dgas.append(1 - best_dGA / math.pi)

    if len(dgas) == 0:
        return 0.0        
    avg_dga = sum(dgas) / len(ps)
    tp_count = len(dgas)
    ret = avg_dga * min(1.0, tp_count / min(10, len(ts)))
    return ret


# ============================================================================ 
# MAIN FUNCTION
# ============================================================================ 

def main():
    """Main function to run crater detection."""
    import argparse
    
    # ======================================================================== 
    # STEP 0: Automatic Data Discovery and Display
    # ======================================================================== 
    
    # Display data configuration automatically when script starts
    # This shows all available datasets, their paths, and status
    print("\n*** NASA LUNAR CRATER DETECTION CHALLENGE ***")
    print("    Computer Vision Solution by VimsRocz\n")
    
    # Automatically discover and display all data paths
    dataset_loader = display_data_configuration()
    
    # Build the CLI so users can point the detector at any structured dataset folder.
    # The supplied location must follow the altitude/longitude/png hierarchy used by the challenge.
    parser = argparse.ArgumentParser(
        description='NASA Lunar Crater Detection Challenge Solution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from repo root: C:\\Users\\vimsr\\Desktop\\nasa-crater-detection):
  # Run on training sample data and evaluate
  python code/crater_detector_final.py --data_folder data\train-sample --output results/detections.csv --evaluate

  # Run on full training data (if available)
  python code/crater_detector_final.py --data_folder data\train --output results/train_detections.csv --evaluate

  # Run on test data for submission
  python code/crater_detector_final.py --data_folder data\test --output results/solution.csv

  # Auto-generate sample data and test
  python code\\crater_detector_final.py --auto_generate

Data Folder Options (relative to project root):
  - Small training set: data/train-sample
  - Full training set:  data/train (requires extracting train.tar)
  - Test set:           data/test (requires extracting test.tar)

IMPORTANT: The data_folder must contain altitude*/longitude*/*.png hierarchy.
           Do NOT use 'data' or 'Data' directly - use data\\train-sample, data\\train, or data\\test

Expected data structure within the specified data_folder:
  data_folder/
  ├── altitude01/
  │   ├── longitude01/
  │   │   ├── orientation01_light01.png
  │   │   └── ...
  │   └── longitude02/
  └── altitude02/
        """)
    parser.add_argument('--data_folder', type=str, 
                       help='Path to data folder containing altitude/longitude hierarchy (e.g., data/train-sample, data/train, data/test)')
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
    parser.add_argument('--circularity', type=float, default=0.6, help='Circularity threshold for crater validation.')
    
    args = parser.parse_args()
    
    # ======================================================================== 
    # STEP 1: Handle special modes (sample generation, auto-generate)
    # ======================================================================== 
    
    # Handle sample data generation (useful for quick smoke tests without downloading the archive)
    if args.generate_sample:
        print("Generating sample test data...")
        success = generate_sample_data(args.generate_sample)
        sys.exit(0 if success else 1)
    
    # Handle auto_generate mode - creates synthetic data for testing
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
    
    # ======================================================================== 
    # STEP 2: Interactive data folder selection if not provided via CLI
    # ======================================================================== 
    
    # If no data_folder provided, offer interactive selection
    if not args.data_folder:
        # Check if running in interactive mode (not piped/redirected)
        if sys.stdin.isatty():
            print("\n" + "="*70)
            print("QUICK START COMMANDS")
            print("="*70)
            
            # Show ready-to-use commands based on available datasets
            available = dataset_loader.list_available_datasets()
            
            print("\nBased on your available datasets, try:")
            print()
            
            for dataset in available:
                if dataset['key'] in ['train-sample', 'train', 'test'] and dataset['is_valid']:
                    cmd = f"python code\\crater_detector_final.py --data_folder {dataset['path']} --output {dataset['suggested_output']}"
                    if dataset['has_ground_truth']:
                        cmd += " --evaluate"
                    print(f"  # {dataset['name']}")
                    print(f"  {cmd}")
                    print()
            
            print("="*70)
            print("\n⚠ No data folder specified. Please run one of the commands above.")
            sys.exit(1)
        else:
            # Non-interactive mode - show help and exit
            parser.print_help()
            print("\n" + "="*70)
            print("ERROR: No data folder specified!")
            print("="*70)
            print("\nQuick start (from repo root):")
            print("  python code/crater_detector_final.py --data_folder data\train-sample --output results/detections.csv --evaluate")
            sys.exit(1)
    
    # ======================================================================== 
    # STEP 3: Initialize detector with specified parameters
    # ======================================================================== 
    
    # Create detector instance with user-specified or default parameters
    # Canny thresholds control edge detection sensitivity
    # Circularity thresholds filters out non-circular/non-elliptical shapes
    detector = CraterDetector(
        canny_th1=args.canny1,
        canny_th2=args.canny2,
        circularity_threshold=args.circularity
    )
    
    # ======================================================================== 
    # STEP 4: Process the dataset and generate detections
    # ======================================================================== 
    
    # Display processing configuration
    print("\n" + "="*70)
    print("CRATER DETECTION CONFIGURATION")
    print("="*70)
    print(f"  Data Folder:      {args.data_folder}")
    print(f"  Output File:      {args.output}")
    print(f"  Canny Threshold:  {args.canny1}, {args.canny2}")
    print(f"  Circularity:      {args.circularity}")
    print(f"  Evaluation Mode:  {'Enabled' if args.evaluate else 'Disabled'}")
    print("="*70)
    
    print(f"\nStarting crater detection on: {args.data_folder}")
    print("="*70)
    success = detector.process_dataset(args.data_folder, args.output, 
                                       verbose=args.verbose)
    
    if not success:
        print("\nDetection failed. Please check the error messages above.")
        sys.exit(1)
    
    # Evaluate against ground truth if training data is supplied; requires truth folders to exist
    if args.evaluate:
        print("\n" + "="*70)
        print("EVALUATION MODE")
        print("="*70)
        
        ground_truth_df = load_ground_truth(args.data_folder)
        
        if ground_truth_df is not None:
            print(f"✓ Loaded ground truth from truth folders")
            stats = compare_with_ground_truth(args.output, ground_truth_df)
            
            print("\nComparison Statistics:")
            print(f"  Ground Truth Craters: {stats['ground_truth_craters']}")
            print(f"  Detected Craters: {stats['detected_craters']}")
            print(f"  Ground Truth Images: {stats['ground_truth_images']}")
            print(f"  Detected Images: {stats['detected_images']}")
            print(f"  Common Images: {stats['common_images']}")
            print(f"  Missing Images: {stats['missing_images']}")

            print("\nNOTE: Performing additional analysis:")
            analyze_detections(args.output)
            
            print("\n" + "="*70)
            print("SCORING RESULTS")
            print("="*70)

            try:
                print('Reading truth data for scoring...')
                # Need to use the loaded ground_truth_df, not re-read from file
                truth_for_scoring = (
                    ground_truth_df.set_index('inputImage').groupby(level='inputImage')
                    .apply(lambda g: g.to_dict(orient='records'))
                    .to_dict()
                )
                print('Reading detections for scoring...')
                detections_for_scoring_df = pd.read_csv(args.output)
                detections_for_scoring = (
                    detections_for_scoring_df.set_index('inputImage').groupby(level='inputImage')
                    .apply(lambda g: g.to_dict(orient='records'))
                    .to_dict()
                )

                image_ids = list(truth_for_scoring.keys())
                total_score = 0
                for img_id in image_ids:
                    truth_craters = truth_for_scoring[img_id]
                    # Ensure detections exist for this image
                    if img_id not in detections_for_scoring:
                        detection_craters = [] # No detections for this image
                    else:
                        detection_craters = detections_for_scoring[img_id]
                    
                    # Reset 'matched' status for each image's detections and truth
                    for tc in truth_craters:
                        tc['matched'] = False
                    for dc in detection_craters:
                        dc['matched'] = False

                    try:
                        score_for_image = score1(truth_craters, detection_craters)
                    except Exception as e:
                        print(f'Error scoring image {img_id}: {str(e)}')
                        score_for_image = 0 # Assign 0 if scoring fails for an image
                    total_score += score_for_image

                # Calculate average score
                if len(image_ids) > 0:
                    average_score = total_score / len(image_ids)
                    final_score = 100 * average_score
                else:
                    final_score = 0.0

                print(f'Overall Score: {final_score}')
                # Assuming 'results' folder for score output if args.out_dir is not provided
                score_output_dir = Path("results/score") # Default path
                score_output_dir.mkdir(parents=True, exist_ok=True)
                writeScore(final_score, str(score_output_dir)) # Pass path to writeScore

            except Exception as e:
                print(f'Error during scoring process: {str(e)}')
            
        else:
            print("⚠ No ground truth found in truth/ folders for scoring")
            print("  This appears to be test data (no truth/ folders)")
    
    print("\n" + "="*70)
    print("✓ Detection complete!")
    print("="*70)


if __name__ == '__main__':
    main()
