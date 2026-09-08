"""
Model 1: The Finder (YOLOv8 Nano Eye Detector)
Locates sheep/goat inner eye/eyelids from live face photos
and extracts padded crops for Model 2 (The Judge).
"""
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image


class EyeFinder:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initializes YOLOv8 Nano eye detector.
        If model_path is None or doesn't exist, loads base yolov8n.pt.
        """
        from ultralytics import YOLO
        
        if model_path and Path(model_path).exists():
            print(f"Loading trained Finder model: {model_path}")
            self.model = YOLO(model_path)
        else:
            print("Loading base YOLOv8n model...")
            self.model = YOLO("yolov8n.pt")

    def detect_eyes(self, image: Image.Image, conf: float = 0.25) -> List[dict]:
        """
        Runs object detection on the input image.
        Returns list of detections with bbox [x1, y1, x2, y2] and confidence.
        """
        results = self.model(image, conf=conf, verbose=False)
        detections = []

        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            coords = box.xyxy[0].cpu().numpy().tolist()
            confidence = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            detections.append({
                "bbox": coords,  # [x1, y1, x2, y2]
                "confidence": confidence,
                "class_id": cls_id
            })

        # Sort detections by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    @staticmethod
    def crop_eye(image: Image.Image, bbox: List[float], padding: float = 0.15) -> Image.Image:
        """
        Crops the eye region with an extra padding margin (e.g. 15%)
        to ensure full mucosal margin and inner eyelid are captured.
        """
        w, h = image.size
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1

        pad_x = box_w * padding
        pad_y = box_h * padding

        # Apply padding and clamp to image boundaries
        crop_x1 = max(0, int(x1 - pad_x))
        crop_y1 = max(0, int(y1 - pad_y))
        crop_x2 = min(w, int(x2 + pad_x))
        crop_y2 = min(h, int(y2 + pad_y))

        return image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    def get_best_eye(self, image: Image.Image, conf: float = 0.25, padding: float = 0.15) -> Optional[Tuple[Image.Image, dict]]:
        """
        Finds the most confident eye detection and returns (eye_crop, detection_info).
        Returns None if no eye is detected.
        """
        detections = self.detect_eyes(image, conf=conf)
        if not detections:
            return None

        best_det = detections[0]
        crop = self.crop_eye(image, best_det["bbox"], padding=padding)
        return crop, best_det
