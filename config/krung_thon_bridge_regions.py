"""Fixed camera-112 regions for the Krung Thon Bridge V2 tracker.

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


# Camera 112: the full scan area sent to YOLO, followed by four smaller lane
# polygons used only to assign each detected vehicle to a lane.  Keeping these
# two jobs separate means a vehicle is never hidden from YOLO merely because
# its bottom-centre falls close to a lane boundary.
CAMERA_112_ROI = _polygon(
    [
        [6, 254], [98, 178], [184, 114], [233, 82], [282, 47],
        [303, 40], [315, 32], [361, 29], [421, 28], [446, 25],
        [475, 41], [498, 69], [544, 102], [598, 150], [633, 198],
        [696, 259], [748, 308], [773, 354], [782, 398], [793, 444],
        [5, 443], [4, 252],
    ]
)

CAMERA_112_LANES = {
    "lane_1": _polygon(
        [[4, 429], [79, 313], [159, 216], [225, 144], [270, 101],
         [311, 60], [331, 41], [355, 41], [347, 52], [316, 92],
         [287, 140], [249, 197], [210, 264], [166, 341], [116, 445],
         [8, 445], [3, 431]]
    ),
    "lane_2": _polygon(
        # The left edge is also lane 1's right edge. Keeping the two polygons
        # separate while sharing the original left-hand seam removes the
        # triangular gap without shifting the lane-2 boundary to the right.
        [[116, 445], [166, 341], [210, 264], [249, 197], [287, 140],
         [316, 92], [347, 52], [355, 41], [379, 41], [357, 136],
         [291, 445], [116, 445]]
    ),
    "lane_3": _polygon(
        [[289, 445], [319, 304], [346, 178], [362, 107], [379, 41],
         [403, 41], [421, 111], [437, 179], [462, 294], [493, 445],
         [292, 445]]
    ),
    "lane_4": _polygon(
        [[497, 447], [472, 325], [446, 218], [427, 134], [405, 55],
         [403, 41], [428, 41], [484, 120], [532, 194], [580, 271],
         [629, 357], [675, 445], [498, 445]]
    ),
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
