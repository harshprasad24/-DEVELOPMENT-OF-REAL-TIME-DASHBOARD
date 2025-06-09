import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def detect_marbles(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhanced preprocessing
    # 1. Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    
    # 2. Denoising
    denoised = cv2.fastNlMeansDenoising(equalized, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # 3. Adaptive thresholding as fallback
    adaptive_thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(denoised, (7, 7), 2)

    # Dual detection approach: Hough Circles + Contour detection
    all_circles = []

    # Approach 1: Hough Circle Detection
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=60
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        all_circles.extend([tuple(circle) for circle in circles])

    # Approach 2: Contour-based detection (as fallback)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_radius = 10
    max_radius = 60
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100 or area > 10000:  # Filter by area
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter**2)
        if circularity > 0.5:  # Check for circularity
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            radius = int(radius)
            if min_radius <= radius <= max_radius:
                all_circles.append((int(x), int(y), radius))

    # Non-maximum suppression to remove overlapping circles
    def nms_circles(circles, overlap_threshold=0.5):
        if len(circles) == 0:
            return []
            
        # Sort by radius (largest first)
        circles = sorted(circles, key=lambda x: x[2], reverse=True)
        keep = []
        
        for current in circles:
            x1, y1, r1 = current
            keep_current = True
            
            for kept in keep:
                x2, y2, r2 = kept
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                # If circles overlap more than the threshold
                if dist < (r1 + r2) * overlap_threshold:
                    keep_current = False
                    break
                    
            if keep_current:
                keep.append(current)
                
        return keep

    filtered_circles = nms_circles(all_circles, overlap_threshold=0.7)

    # Image physical size (mm)
    physical_width_mm = 100
    physical_height_mm = 80
    height_px, width_px = image.shape[:2]
    px_per_mm = (width_px / physical_width_mm + height_px / physical_height_mm) / 2

    # Analyze circles
    output = image.copy()
    sizes_mm = []

    print(f"Accurate marble count: {len(filtered_circles)}")

    for (x, y, r) in filtered_circles:
        diameter_mm = (2 * r) / px_per_mm
        sizes_mm.append(round(diameter_mm, 1))

        # Draw the circle and center
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), -1)
        cv2.putText(output, f"{diameter_mm:.1f}mm", (x-r, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Group sizes into bins
    size_bins = defaultdict(int)
    for size in sizes_mm:
        low = int(size)
        high = low + 1
        bin_label = f"{low}-{high}"
        size_bins[bin_label] += 1

    # Sort bin labels
    sorted_bins = sorted(size_bins.items(), key=lambda x: int(x[0].split('-')[0]))
    labels = [k for k, _ in sorted_bins]
    counts = [v for _, v in sorted_bins]

    print("\nGrouped Marble Sizes (in mm):")
    for label, count in sorted_bins:
        print(f"{label} mm: {count}")

    # Plot results
    plt.figure(figsize=(15, 5))

    # Bar graph for size distribution
    plt.subplot(1, 3, 1)
    plt.title("Size Distribution (mm)")
    plt.bar(labels, counts, color='skyblue', edgecolor='black')
    plt.xlabel("Diameter Range (mm)")
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    # Preprocessing visualization
    plt.subplot(1, 3, 2)
    plt.title("Preprocessed Image")
    plt.imshow(blurred, cmap='gray')
    plt.axis("off")

    # Final detected marbles
    plt.subplot(1, 3, 3)
    plt.title(f"Detected Marbles: {len(filtered_circles)}")
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Usage
image_path = r"C:\Users\win11\Documents\Tata_steel_object\Marbel.png"
detect_marbles(image_path)

#"C:\Users\win11\Documents\Tata_steel_object\Marble.png"