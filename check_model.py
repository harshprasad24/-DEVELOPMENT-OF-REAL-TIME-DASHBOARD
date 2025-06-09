import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image_path = r"C:\Users\win11\Documents\Tata_steel_object\Marble.png"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Enhance contrast using CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized = clahe.apply(gray)

# Blur to reduce noise
blurred = cv2.GaussianBlur(equalized, (5, 5), 1.5)

# Hough Circle Detection
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=20,
    param1=50,
    param2=30,
    minRadius=10,
    maxRadius=60  # Increase maxRadius if bigger marbles exist
)

# Overlap removal function for variable sized circles
def nms_circles_variable_size(circles, dist_factor=0.8):
    if len(circles) == 0:
        return []

    circles = sorted(circles, key=lambda x: x[2], reverse=True)  # Sort by radius desc
    keep = []

    for c in circles:
        x1, y1, r1 = c
        overlap = False
        for k in keep:
            x2, y2, r2 = k
            dist = np.hypot(x1 - x2, y1 - y2)
            if dist < dist_factor * (r1 + r2):
                overlap = True
                break
        if not overlap:
            keep.append(c)
    return keep


# Draw output
output = image.copy()
filtered_circles = []

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    filtered_circles = nms_circles_variable_size(circles, dist_factor=0.8)

    print(f"Accurate marble count: {len(filtered_circles)}")

    for (x, y, r) in filtered_circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), -1)
else:
    print("No marbles detected.")

# Show results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("CLAHE Enhanced")
plt.imshow(equalized, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Edges (Canny)")
plt.imshow(cv2.Canny(blurred, 50, 150), cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title(f"Detected Marbles: {len(filtered_circles)}")
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()

