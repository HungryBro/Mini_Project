"""Fixed camera regions for the Krung Thon Bridge pair.

The coordinates were clicked on the native 800x450 camera frames.  The
helpers scale them if a caller supplies a resized frame, so the same config
can be used for previews and video writers without changing the source data.
"""

from __future__ import annotations

import cv2
import numpy as np


REFERENCE_WIDTH = 800
REFERENCE_HEIGHT = 450


def _polygon(points: list[list[int]]) -> np.ndarray:
    return np.asarray(points, dtype=np.int32)


# Camera 112: whole bridge roadway and four lane polygons, numbered from left
# to right in the camera image.  The four top caps divide the outer ROI's
# [331, 41] -> [428, 41] edge continuously, so their visible top line is level.
CAMERA_112_ROI = _polygon(
    [
        [331, 41], [428, 41], [453, 76], [494, 137], [544, 212],
        [598, 302], [638, 372], [675, 447], [5, 443], [3, 427],
        [52, 345], [100, 286], [141, 236], [183, 188], [235, 135],
        [277, 94], [315, 57], [328, 41],
    ]
)

CAMERA_112_LANES = {
    "lane_1": _polygon(
        [[4, 429], [79, 313], [159, 216], [225, 144], [270, 101],
         [311, 60], [331, 41], [355, 41], [322, 85], [295, 129],
         [265, 173], [229, 231], [178, 317], [137, 400], [126, 444],
         [8, 443], [3, 431]]
    ),
    "lane_2": _polygon(
        [[116, 445], [166, 341], [210, 264], [249, 197], [287, 140],
         [316, 92], [347, 52], [355, 41], [379, 41], [357, 136],
         [291, 441], [118, 444]]
    ),
    "lane_3": _polygon(
        [[289, 445], [319, 304], [346, 178], [362, 107], [379, 41],
         [403, 41], [421, 111], [437, 179], [462, 294], [493, 445],
         [292, 445]]
    ),
    "lane_4": _polygon(
        [[497, 447], [472, 325], [446, 218], [427, 134], [405, 55],
         [403, 41], [428, 41], [484, 120], [532, 194], [580, 271],
         [629, 357], [675, 446], [498, 444]]
    ),
}


# Camera 147: the four overhead lane-signal displays corresponding to the
# four lane polygons above.  These regions intentionally include the whole
# display face so the color detector can tolerate a small camera shift.
CAMERA_147_SIGNAL_ROIS = {
    "lane_1": _polygon([[136, 1], [141, 24], [252, 22], [249, 1], [135, 0]]),
    "lane_2": _polygon([[395, 4], [398, 11], [477, 11], [481, 1], [397, 0]]),
    "lane_3": _polygon([[588, 1], [586, 12], [619, 15], [624, 2], [591, 0]]),
    "lane_4": _polygon([[668, 0], [669, 28], [695, 31], [703, 1], [671, 0]]),
}


# Camera 156 is on the opposite side of the bridge.  Its left-to-right LEDs
# correspond to camera 112's lanes 4, 3, 2, 1, respectively.  The dictionary
# is deliberately keyed by the *camera 112 lane* so fusion never has to guess
# which physical lane a light belongs to.
CAMERA_156_SIGNAL_ROIS = {
    "lane_1": _polygon([[413, 2], [415, 14], [424, 13], [425, 3], [416, 3]]),
    "lane_2": _polygon([[396, 1], [396, 11], [407, 14], [408, 2], [399, 1]]),
    "lane_3": _polygon([[378, 0], [378, 12], [392, 14], [392, 3], [379, 1]]),
    "lane_4": _polygon([[362, 0], [362, 11], [372, 12], [375, 2], [363, 1]]),
}


def scale_polygon(points: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Scale native 800x450 points to ``frame`` resolution."""
    height, width = frame.shape[:2]
    scale_x = width / REFERENCE_WIDTH
    scale_y = height / REFERENCE_HEIGHT
    scaled = points.astype(np.float32).copy()
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    return np.rint(scaled).astype(np.int32)


def camera_112_roi(frame: np.ndarray) -> np.ndarray:
    return scale_polygon(CAMERA_112_ROI, frame)


def camera_112_lane_rois(frame: np.ndarray) -> dict[str, np.ndarray]:
    return {name: scale_polygon(points, frame) for name, points in CAMERA_112_LANES.items()}


def camera_147_signal_rois(frame: np.ndarray) -> dict[str, np.ndarray]:
    return {name: scale_polygon(points, frame) for name, points in CAMERA_147_SIGNAL_ROIS.items()}


def camera_156_signal_rois(frame: np.ndarray) -> dict[str, np.ndarray]:
    return {name: scale_polygon(points, frame) for name, points in CAMERA_156_SIGNAL_ROIS.items()}


def point_lane(point: tuple[float, float], lane_rois: dict[str, np.ndarray]) -> str | None:
    """Return the lane containing a bottom-centre point, if any."""
    containing: list[tuple[float, str]] = []
    for name, polygon in lane_rois.items():
        distance = cv2.pointPolygonTest(polygon.astype(np.float32), point, True)
        if distance >= 0:
            containing.append((distance, name))
    if not containing:
        return None
    # At a shared polygon edge, choose the lane with the most interior point.
    return max(containing)[1]
