import cv2
import numpy as np
import numpy.typing as npt


def solve(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.int32]:
    return np.zeros((9, 9), dtype="int32")
