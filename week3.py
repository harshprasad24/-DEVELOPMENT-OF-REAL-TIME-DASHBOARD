import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "test.png"  # Make sure the image is in the same folder or use full path
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Use Hough Circle Transform to detect circles
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=20,
    param1=100,
    param2=30,
    minRadius=15,
    maxRadius=30
)

# Draw detected circles
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        cv2.rectangle(image, (x - 2, y - 2), (x + 2, y + 2), (0, 0, 255), -1)

    print(f"✅ Detected {len(circles)} marbles.")
else:
    print("❌ No marbles detected.")

# Show result
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Detected Marbles")
plt.show()
