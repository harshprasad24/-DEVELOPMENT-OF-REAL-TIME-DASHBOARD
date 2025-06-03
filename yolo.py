import cv2
import numpy as np
from ultralytics import YOLO

def detect_objects_yolo(image_path):
    model = YOLO('yolov8n.pt')
    image = cv2.imread(image_path)
    results = model(image)
    return results

# Example usage
results = detect_objects_yolo('test.jpg')
