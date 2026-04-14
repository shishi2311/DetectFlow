"""
Object detection module using YOLOv8.

Wraps the Ultralytics YOLOv8 model to provide a clean interface for
detecting objects (primarily people) in video frames.
"""

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from config import DetectorConfig


@dataclass
class Detection:
    """A single object detection."""

    bbox: np.ndarray  # [x1, y1, x2, y2] in pixel coordinates
    confidence: float
    class_id: int

    @property
    def center(self) -> tuple[float, float]:
        """Return the center point of the bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height


class Detector:
    """YOLOv8-based object detector."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.model = YOLO(config.model_name)
        print(f"[Detector] Loaded model: {config.model_name}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run detection on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of Detection objects passing the confidence threshold
            and matching target classes.
        """
        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            device=self.config.device,
            verbose=False,
            classes=self.config.target_classes,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(np.float32)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                detections.append(Detection(bbox=bbox, confidence=conf, class_id=cls_id))

        return detections
