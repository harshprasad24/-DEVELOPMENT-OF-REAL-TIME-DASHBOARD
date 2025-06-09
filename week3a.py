import cv2
from ultralytics.solutions import ObjectCounter

image_path = "test.png"
im0 = cv2.imread(image_path)
assert im0 is not None, "❌ Error: Image not found."

h, w = im0.shape[:2]
region_point = [(0, 0), (w, 0), (w, h), (0, h)]

counter = ObjectCounter(model="marble_detector.pt")
counter.set_args(region=region_point)

result = counter(im0)  # Run detection + counting

print(f"Detected objects: {result.count}")

# Access annotated image:
if hasattr(result, 'orig_img'):
    im0_annotated = result.orig_img
elif hasattr(result, 'img'):
    im0_annotated = result.img
else:
    im0_annotated = im0  # fallback: original image

cv2.imwrite("counted_marbles_output.png", im0_annotated)
print("✅ Output saved as counted_marbles_output.png")

cv2.imshow("Marble Detection", im0_annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
