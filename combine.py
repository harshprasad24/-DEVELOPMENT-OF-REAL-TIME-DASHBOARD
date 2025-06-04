import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_circular_objects(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # Read and preprocess the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=30, minRadius=10, maxRadius=100)

    marble_count = 0
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        marble_count = len(circles)
        for idx, (x, y, r) in enumerate(circles, 1):
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            cv2.putText(image, f"marble {idx}", (x - 20, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Total Marbles Detected: {marble_count}")
    plt.axis('off')
    plt.show()

example_path = "C:/Users/win11/Documents/Tata_steel_object/test.png"  
detect_circular_objects(example_path)


