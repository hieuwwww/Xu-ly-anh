import cv2
from typing import List, Optional, Tuple


class Stitcher:
    PANORAMA = 0
    SCANS = 1

    class Status:
        OK = 0
        ERR_NEED_MORE_IMGS = 1
        ERR_HOMOGRAPHY_EST_FAIL = 2
        ERR_CAMERA_PARAMS_ADJUST_FAIL = 3

    def __init__(self, mode=PANORAMA):
        # Delegate to OpenCV's high-level Stitcher for robustness
        self._stitcher = cv2.Stitcher_create(mode)

    @staticmethod
    def create(mode=PANORAMA):
        return Stitcher(mode)

    def stitch(self, images: List, masks: Optional[List] = None) -> Tuple[int, Optional[any]]:
        if masks:
            return self._stitcher.stitch(images, masks)
        return self._stitcher.stitch(images)