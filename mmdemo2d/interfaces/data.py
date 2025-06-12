"""
Helper dataclasses for interface definitions
"""

from dataclasses import dataclass

from mmdemo2d.base_interface import BaseInterface

import numpy as np

@dataclass
class ObjectInterface(BaseInterface):
    """
    name -- the name of detected object.

    position -- the point of top-left and right-bottom in the original frame.
    """

    name: str
    position: np.ndarray

@dataclass
class JointPointInterface(BaseInterface):
    """
    name -- the name of participant.

    position -- the joint points of detected participant.
    """

    name: str
    points: np.ndarray