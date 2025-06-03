from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms import functional as F

def detect_objects_ssd(image_path):
    model = ssdlite320_mobilenet_v3_large(pretrained=True)
    model.eval()
    image = Image.open(image_path).convert("RGB")  # Ensure RGB format
    image_tensor = F.to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)
 
    return predictions, image  # Return both predictions and original image

# Example usage
predictions, image = detect_objects_ssd('test.jpg')

# Convert PIL image to NumPy array
image_np = np.array(image)
pred = predictions[0]

# Draw bounding boxes for detected objects
for box, score in zip(pred['boxes'], pred['scores']):
    if score > 0.5:
        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Convert BGR to RGB for matplotlib
plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
