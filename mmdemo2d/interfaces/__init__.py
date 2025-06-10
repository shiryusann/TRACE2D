"""
Premade output interfaces
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import numpy as np

from mmdemo2d.base_interface import BaseInterface

@dataclass
class EmptyInterface(BaseInterface):
    """
    Output interface when the feature does not have any output.
    """

@dataclass
class ColorImageInterface(BaseInterface):
    """
    frame -- image data with shape (h, w, 3) in RGB format.
        The values should be integers between 0 and 255.
    """

    frame: np.ndarray

@dataclass
class DetectedObjectInterface(BaseInterface):
    """
    name -- the name of detected object.

    position -- the point of top-left and right-bottom in the original frame.
    """

    name: str
    position: np.ndarray

@dataclass
class DetectedObjectsInterface(BaseInterface):
    """
    objects -- a list of detected objects in a frame
    """

    objects: List[DetectedObjectInterface]

@dataclass
class TimeInterface(BaseInterface):
    """
    frametime -- time of the currently processed frame (unit:millisecond).

    count -- number of the currently processed frame.
    """

    frametime: float
    count: int

@dataclass
class VideoInputInterface(BaseInterface):
    """
    colorimage -- colorimage interface

    time -- time interface
    """
    colorimage: ColorImageInterface
    time: TimeInterface