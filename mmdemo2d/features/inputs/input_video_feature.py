from typing import final
from pathlib import Path

import cv2

import time

from mmdemo2d.base_feature import BaseFeature
from mmdemo2d.interfaces import VideoInputInterface, ColorImageInterface, TimeInterface

@final
class InputVideo(BaseFeature):
    def __init__(
        self,
        path: str,
        frame_rate: int,
        play_back: bool
    ) -> None:
        super().__init__()
        self.path = path
        self.frame_rate = frame_rate
        self.done = False

        #get the video frame rate
        if play_back:
            assert Path(path).exists, "the video path does not exist!"
            video = cv2.VideoCapture(path)
            assert video.isOpened(), "the video is not opened!"
            video_frame_rate = int(video.get(cv2.CAP_PROP_FPS))
            self.video_frame_interval = int(1 / video_frame_rate * 1000)
            video.release()
            
            assert video_frame_rate % self.frame_rate == 0, "video frame rate must be a multiple of user set frame rate!"
            self.frame_interval = video_frame_rate / self.frame_rate
    
    def initialize(self):
        self.video = cv2.VideoCapture(self.path)
        self.start_time = int(time.time() * 1000)
        self.last_time = self.start_time
        self.frame_count = 0
    
    def finalize(self):
        self.video.release()

    def get_output(self) -> VideoInputInterface | None:
        new = False
        ret, frame = self.video.read()
        if not ret:
            self.done = True
            return None
        
        current_time = int(time.time() * 1000)

        if self.frame_count % self.frame_interval == 0:
            new = True

        # decide whether we wait for the next frame
        if current_time - self.last_time < self.video_frame_interval:
            # wait for a while to make sure that the video is played in the original speed
            cv2.waitKey(self.video_frame_interval - current_time + self.last_time)

        self.frame_count += 1
        self.last_time = current_time
    
        if new:
            return VideoInputInterface(
                colorimage = ColorImageInterface(frame = frame),
                time = TimeInterface(frametime = current_time - self.start_time, count = self.frame_count)
            )
        else:
            return None

    def is_done(self):
        return self.done

@final
class ColorImage(BaseFeature):
    def get_output(
        self,
        videoinput: VideoInputInterface       
    ) -> ColorImageInterface | None:
        if not videoinput.is_new():
            return None
        else:
            return ColorImageInterface(frame = videoinput.colorimage.frame)

@final
class FrameTime(BaseFeature):
    def get_output(
        self,
        videoinput: VideoInputInterface       
    ) -> TimeInterface | None:
        if not videoinput.is_new():
            return None
        else:
            return TimeInterface(frametime = videoinput.time.frametime, count = videoinput.time.count)

def create_input_features(
    path: str,
    frame_rate: int,
    play_back: bool
):
    inputvideo = InputVideo(
        path = path,
        frame_rate = frame_rate,
        play_back = play_back
    )

    colorimage = ColorImage(inputvideo)
    frametime = FrameTime(inputvideo)
    return colorimage, frametime