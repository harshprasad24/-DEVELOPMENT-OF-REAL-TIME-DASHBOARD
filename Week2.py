import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssdlite320_mobilenet_v3_large
from torchvision.transforms import functional as F


COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'marble', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def detect_objects_yolo(image_path, target_class=None):
    # Convert to raw string (handle backslashes)
    image_path = os.path.normpath(image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
    model = YOLO('yolov8n.pt')
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Invalid image format or corrupted file: {image_path}")
    
    results = model(image)
    
    filtered_results = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        class_names = [result.names[class_id] for class_id in class_ids]
        
        if target_class:
            indices = [i for i, name in enumerate(class_names) if name.lower() == target_class.lower()]
            boxes = boxes[indices]
            confidences = confidences[indices]
            class_names = [class_names[i] for i in indices]
        
        filtered_results.append((boxes, confidences, class_names))
    
    return filtered_results, image

def visualize_detections(image, detections, method='yolo'):
    image_np = np.array(image)
    if method == 'yolo':
        boxes, confidences, class_names = detections[0]
        for box, confidence, class_name in zip(boxes, confidences, class_names):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image_np, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    print("=== Object Detection (YOLOv8) ===")
    
    # Get valid image path
    while True:
        image_path = input("Enter image path: ").strip('"').strip()  # Remove quotes/spaces
        image_path = os.path.normpath(image_path)  # Fix backslashes for Windows
        
        if not os.path.exists(image_path):
            print(f"❌ Error: File not found at: {image_path}")
            continue
        break
    
    # Optional: Filter for a specific object class
    target_class = input("Enter target object (e.g., 'car', 'person', or leave empty for all): ").strip()

    try:
        print("\n🔍 Detecting objects...")
        detections, image = detect_objects_yolo(image_path, target_class if target_class else None)
        print("✅ Detection complete!")
        visualize_detections(image, detections)
    except Exception as e:
        print(f"❌ Error: {e}")