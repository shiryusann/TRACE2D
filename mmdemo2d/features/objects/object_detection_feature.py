from typing import final, Dict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

from mmdemo2d.base_feature import BaseFeature
from mmdemo2d.interfaces import ColorImageInterface, DetectedObjectsInterface, DetectedObjectInterface

@final
class ObjectiveDetection(BaseFeature):
    def __init__(
        self,
        path: str,
        threshold: float,
        color: BaseFeature[ColorImageInterface]
    ) -> None:
        super().__init__(color)
        assert Path(path).exists, "the model path does not exist!"
        self.model = YOLO(path)

        self.threshold = threshold

        self.used_objects = {"laptop", "keyboard"}

    def get_output(
        self,
        color: ColorImageInterface
    ):
        if not color.is_new:
            return None

        #copy the color image
        copy = color.frame.copy()
        copy = cv2.cvtColor(copy, cv2.COLOR_RGB2BGR)

        result = self.model(copy, verbose = False)
        boxes = result[0].boxes
        names = result[0].names
        mapped_result = self.map_result(boxes, names)

        objects = []
        for i in mapped_result:
            if i[0] in self.used_objects and i[1] >= self.threshold:
                objects.append(DetectedObjectInterface(name = i[0], position = i[2]))
    
        return DetectedObjectsInterface(objects = objects)
    
    @staticmethod
    def map_result(
        boxes: Boxes,
        names: Dict[int, str]
    ) -> dict[str, torch.Tensor]:
        mapped_result = []
        n = len(boxes.cls)
        for i in range(n):
            cls = int(boxes.cls[i])
            confidence = float(boxes.conf[i])
            points = boxes.xyxy[i]
            mapped_result.append((names[cls], confidence, points.type(torch.int32).tolist()))
        return mapped_result