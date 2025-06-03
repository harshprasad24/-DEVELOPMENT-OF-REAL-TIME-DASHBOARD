from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

def detect_objects_faster_rcnn(image_path):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    image = Image.open(image_path).convert("RGB")  # Ensure it's in RGB
    image_tensor = F.to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor) 

    return predictions, image  # Return both prediction and original image

# Example usage
predictions, image = detect_objects_faster_rcnn('test.jpg')

# Convert image to numpy and draw predictions
image_np = np.array(image)
pred = predictions[0]

for box, score in zip(pred['boxes'], pred['scores']):
    if score > 0.5:
        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Convert BGR (used by cv2) to RGB for matplotlib
plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()