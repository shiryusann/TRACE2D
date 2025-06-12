from typing import final, List, Tuple
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Keypoints

from mmdemo2d.base_feature import BaseFeature
from mmdemo2d.interfaces import ColorImageInterface, BodyPointsInterface
from mmdemo2d.interfaces.data import JointPointInterface
from mmdemo2d.utils.joints import Joint

@final
class Bodypoints(BaseFeature):
    def __init__(
        self,
        path: str,
        color: BaseFeature[ColorImageInterface]
    ) -> None:
        super().__init__(color)
        assert Path(path).exists, "the model path does not exist!"
        self.model = YOLO(path)
    
    def initialize(self):
        pass
    
    def get_output(
        self,
        color: ColorImageInterface
    ) -> BodyPointsInterface | None:
        if not color.is_new():
            return None
        
        # copy the color image
        copy = color.frame.copy()
        copy = cv2.cvtColor(copy, cv2.COLOR_RGB2BGR)

        result = self.model(copy, verbose = False)[0].keypoints
        mapped_result = self.map_result(result)

        bodies = []
        for i in mapped_result:
            bodies.append(JointPointInterface(name = i[0], points = i[1]))

        return BodyPointsInterface(bodies = bodies)

    @staticmethod
    def map_result(result: Keypoints) -> List[Tuple[str, np.ndarray]]:
        mapped_result = []
        # sort the result tensor by the point of noises
        data = result.data.type(torch.int32)
        nose = data[:, Joint.NOSE.value, 0]
        indices = torch.sort(nose).indices.tolist()

        # name participants from left to right by p*
        for i in range(len(indices)):
            mapped_result.append(("P" + str(indices[i]), data[indices[i], :, 0:2].cpu().numpy()))
        
        return mapped_result