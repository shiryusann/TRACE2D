import random
from typing import final

import cv2 as cv
import torch

from mmdemo2d.base_feature import BaseFeature
from mmdemo2d.interfaces import ColorImageInterface, DetectedObjectsInterface, EmptyInterface

GREEN = (0, 255, 0)

@final
class DisplayFrame(BaseFeature[EmptyInterface]):
    """
    Show a color frame with opencv. The demo will exit once
    the window is closed.

    Input interface is `ColorImageInterface`

    Output interface is `EmptyInterface`
    """

    def __init__(
        self,
        color: BaseFeature[ColorImageInterface],
        objects: BaseFeature[DetectedObjectsInterface]
    ) -> None:
        super().__init__(color, objects)

    def initialize(self):
        self.window_name = str(random.random())
        self.window_should_be_up = False

    def get_output(
        self,
        color: ColorImageInterface,
        objects: DetectedObjectsInterface
    ) -> EmptyInterface | None:
        if not color.is_new():
            self.window_should_be_up = False
            return None
        
        frame = color.frame
        
        for i in objects.objects:
            top_left = i.position[0:2]
            bottom_right = i.position[2:4]
            cv.rectangle(
                img = frame,
                pt1 = top_left,
                pt2 = bottom_right,
                color = GREEN,
                thickness = 3
            )

        frame = cv.resize(frame, (800, 600))
        cv.imshow(self.window_name, frame)
        cv.waitKey(1)
        self.window_should_be_up = True

        return EmptyInterface()

    def is_done(self):
        return (
            self.window_should_be_up
            and cv.getWindowProperty(self.window_name, cv.WND_PROP_VISIBLE) < 1
        )
