import pandas as pd

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

    print("--- Descriptive Statistics for Ellipse Axes (in pixels) ---\n")
    
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
    
    print(f"--- Potential Outliers (Ellipses larger than challenge filter) ---\n")
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
    
    print(f"--- Unusually Elongated Ellipses (Aspect Ratio > {aspect_threshold}) ---\n")
    
    if not elongated_craters.empty:
        print(f"Found {len(elongated_craters)} detections that are very elongated:")
        print(elongated_craters[['inputImage', 'ellipseSemimajor(px)', 'ellipseSemiminor(px)', 'aspect_ratio']].to_string())
    else:
        print("No unusually elongated ellipses found.")


if __name__ == '__main__':
    analyze_detections('results/train-sample_results.csv')
