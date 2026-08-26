"""YOLO vehicle detection/tracking for the bridge camera.

COCO classes used by this project:
    2 car, 3 motorcycle, 5 bus, 7 truck

Examples:
    python3 vehicle_tracking.py --source locations/taksin/image/image.png
    python3 vehicle_tracking.py --source locations/taksin/video/taksin_bridge_sathorn_1min.mp4
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

from config.krung_thon_bridge_regions import (
    camera_112_lane_rois,
    camera_112_roi,
    camera_147_signal_rois,
    camera_156_signal_rois,
    point_lane,
)


VEHICLE_NAMES = {
    "car",
    "motorcycle",  # COCO name
    "motorbike",  # Thai-Cars name
    "bus",
    "truck",
    "pickup",
    "taxi",
    "van",
    "truck trailer",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# OpenCV uses BGR tuples.  Keep the car/taxi color explicitly shared so that
# taxi detections look identical after ``--merge-taxi`` (and even without it).
CLASS_COLORS_BGR = {
    "car": (255, 180, 0),  # sky blue
    "taxi": (255, 180, 0),
    "bus": (0, 165, 255),
    "motorbike": (255, 0, 255),
    "motorcycle": (255, 0, 255),
    "moto": (255, 0, 255),
    "pickup": (0, 180, 0),
    "truck": (0, 0, 255),
    "truck trailer": (0, 120, 255),
    "van": (0, 200, 200),
}


# Clockwise points expressed as fractions of this fixed bridge-camera frame.
# They form the green trapezoid in the project reference: road on the bridge
# only, without the side road, buildings, train track, or foreground corner.
BRIDGE_ROI_NORMALIZED = (
    (0.170, 0.954),  # 1: bottom-left
    (0.580, 0.160),  # 2: top-left (adjusted at mark 1)
    (0.661, 0.142),  # 3: top-right
    (0.508, 0.953),  # 4: bottom-right
)

# Reference midpoints of the two cross-road lines.  The code extends each
# line to the matching left/right edge of the bridge polygon.
GATE_A_NORMALIZED = (0.576, 0.297)
GATE_B_NORMALIZED = (0.394, 0.771)

PROFILE_AUTO = "auto"
PROFILE_TAKSIN = "taksin"
PROFILE_KRUNG_THON = "krung_thon_bridge"


def vehicle_class_map(model: YOLO, selected_names: set[str] | None = None) -> dict[int, str]:
    """Match either COCO weights or the project's four-class checkpoint.

    COCO uses class IDs 2, 3, 5 and 7, whereas the fine-tuned bridge model
    uses IDs 0..3.  Looking up class names keeps the inference pipeline
    compatible with both and avoids silently filtering out custom-model cars.
    """
    selected_names = selected_names or VEHICLE_NAMES
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    return {int(class_id): str(name).lower() for class_id, name in items if str(name).lower() in selected_names}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and track bridge vehicles with YOLO + ByteTrack")
    parser.add_argument("--source", type=Path, required=True, help="Input image or video")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vehicles"))
    parser.add_argument("--model", default="model/coco/yolo11m.pt", help="Ultralytics model or local .pt path")
    parser.add_argument(
        "--profile",
        choices=(PROFILE_AUTO, PROFILE_TAKSIN, PROFILE_KRUNG_THON),
        default=PROFILE_AUTO,
        help="Camera geometry profile; auto selects Krung Thon for its source folder",
    )
    parser.add_argument(
        "--signal-source",
        type=Path,
        default=None,
        help="Camera 147 video used to read lane signal colors for a camera 112 run",
    )
    parser.add_argument(
        "--signal-offset",
        type=float,
        default=0.0,
        help="Seconds added to camera 112 time when sampling camera 147 (default: 0)",
    )
    parser.add_argument(
        "--signal156-source",
        type=Path,
        default=None,
        help="Camera 156 video used as the opposite-side signal confirmation",
    )
    parser.add_argument(
        "--signal156-offset",
        type=float,
        default=0.0,
        help="Seconds added to camera 112 time when sampling camera 156 (default: 0)",
    )
    parser.add_argument("--conf", type=float, default=0.16)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Run raw per-frame detection only; do not create or retain ByteTrack IDs",
    )
    parser.add_argument(
        "--wrong-way-alerts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "For tracked Krung Thon video, draw normal vehicle boxes green and "
            "wrong-way vehicles red (default: enabled for the Krung Thon profile)"
        ),
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.30,
        help="Overlap threshold for duplicate-box suppression (default: 0.30)",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tracker", default="config/vehicle_bytetrack.yaml")
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated model class names; by default all supported vehicle names in the model",
    )
    parser.add_argument(
        "--merge-taxi",
        action="store_true",
        help="Report detections named taxi as car without retraining",
    )
    parser.add_argument(
        "--merge-car-like",
        action="store_true",
        help="Report taxi, van, and pickup detections as car without retraining",
    )
    parser.add_argument(
        "--bridge-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep and draw only detections whose bottom-centre is inside the bridge ROI (default: enabled)",
    )
    parser.add_argument(
        "--show-gates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw Gate A/B guide lines (default: enabled)",
    )
    parser.add_argument(
        "--show-lanes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Draw the four Krung Thon lane polygons (default: enabled for that profile)",
    )
    parser.add_argument(
        "--save-signal-views",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write annotated camera 147 and 156 light videos (default: enabled for Krung Thon)",
    )
    parser.add_argument(
        "--agnostic-nms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress overlapping boxes even when the model assigns different classes (default: enabled)",
    )
    parser.add_argument(
        "--occlusion-hold",
        type=int,
        default=8,
        help="Keep a short predicted box on screen when a tracked vehicle is hidden (frames; default: 8)",
    )
    parser.add_argument(
        "--max-held-tracks",
        type=int,
        default=3,
        help="Maximum reliable occluded-track boxes shown in one frame (default: 3)",
    )
    return parser.parse_args()


def resolve_profile(source: Path, requested: str) -> str:
    """Resolve the camera geometry without changing existing Taksin commands."""
    if requested != PROFILE_AUTO:
        return requested
    return PROFILE_KRUNG_THON if "krung_thon_bridge" in str(source) else PROFILE_TAKSIN


def default_signal_source(source: Path, profile: str) -> Path | None:
    """Find the matching camera 147 file for a camera 112 source when available."""
    if profile != PROFILE_KRUNG_THON or "cam112" not in source.name:
        return None
    candidate = source.with_name(source.name.replace("cam112", "cam147"))
    return candidate if candidate.exists() else None


def default_signal156_source(source: Path, profile: str) -> Path | None:
    """Find camera 156 alongside a camera 112 v2 video when it is available."""
    if profile != PROFILE_KRUNG_THON or "cam112" not in source.name:
        return None
    candidate = source.with_name(source.name.replace("cam112", "cam156"))
    return candidate if candidate.exists() else None


def bridge_roi(frame, profile: str = PROFILE_TAKSIN) -> np.ndarray:
    """Return the selected bridge-road ROI at the current frame resolution."""
    if profile == PROFILE_KRUNG_THON:
        return camera_112_roi(frame)
    height, width = frame.shape[:2]
    return np.array(
        [(round(x * width), round(y * height)) for x, y in BRIDGE_ROI_NORMALIZED],
        dtype=np.int32,
    )


def lane_rois(frame, profile: str) -> dict[str, np.ndarray] | None:
    if profile != PROFILE_KRUNG_THON:
        return None
    return camera_112_lane_rois(frame)


def direction_from_signal(camera: str, signal_state: str) -> str:
    """Translate a signal LED colour to the travel direction in camera 112.

    Camera 147 faces the same direction as camera 112: green means ``up`` and
    red means ``down``.  Camera 156 sees the opposite end of the reversible
    lane, so its physical LED meaning is inverted: red means ``up`` and green
    means ``down``.  Its regions are already keyed to the matching camera 112
    lane in ``krung_thon_bridge_regions.py``.
    """
    if signal_state not in {"green", "red"}:
        return "unknown"
    if camera == "156":
        return "down" if signal_state == "green" else "up"
    return "up" if signal_state == "green" else "down"


def allowed_direction(signal_state: str) -> str:
    """Backward-compatible camera 147 direction mapping."""
    return direction_from_signal("147", signal_state)


def fuse_lane_signals(
    states_147: dict[str, str] | None,
    states_156: dict[str, str] | None,
) -> dict[str, dict[str, object]]:
    """Fuse the two signal cameras into one complete direction per lane.

    A readable state from either camera is enough.  When both provide the same
    direction the result is marked ``both``.  A true disagreement stays
    visible as ``conflict_147_priority`` but still yields a deterministic
    direction for camera 112; camera 147 is the priority because its signal
    boards occupy more pixels in this dataset.
    """
    states_147 = states_147 or {}
    states_156 = states_156 or {}
    fused: dict[str, dict[str, object]] = {}
    for index in range(1, 5):
        lane = f"lane_{index}"
        light_147 = states_147.get(lane, "unknown")
        light_156 = states_156.get(lane, "unknown")
        direction_147 = direction_from_signal("147", light_147)
        direction_156 = direction_from_signal("156", light_156)
        known_147 = direction_147 != "unknown"
        known_156 = direction_156 != "unknown"
        if known_147 and known_156 and direction_147 == direction_156:
            final_direction, source, agrees = direction_147, "both", True
        elif known_147 and known_156:
            final_direction, source, agrees = direction_147, "conflict_147_priority", False
        elif known_147:
            final_direction, source, agrees = direction_147, "147", None
        elif known_156:
            final_direction, source, agrees = direction_156, "156", None
        else:
            final_direction, source, agrees = "unknown", "none", None
        fused[lane] = {
            "light_147": light_147,
            "light_156": light_156,
            "direction_147": direction_147,
            "direction_156": direction_156,
            "direction": final_direction,
            "source": source,
            "agrees": agrees,
        }
    return fused


def classify_signal_state(frame: np.ndarray, polygon: np.ndarray) -> str:
    """Classify a small overhead signal as green, red, or unknown.

    The signal ROIs are only a few pixels high.  We therefore require a small
    aggregate colored region instead of trusting the average color, which
    would mistake the bridge structure and sky for a signal.
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    # The top two rows are frequently a black-camera border or compression
    # edge, not part of the LED display.
    mask[:2] = 0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    green = (
        (hue >= 70)
        & (hue <= 130)
        & (saturation >= 40)
        & (value >= 30)
        & (mask > 0)
    ).astype(np.uint8)
    red = (
        ((hue <= 20) | (hue >= 150))
        & (saturation >= 40)
        & (value >= 30)
        & (mask > 0)
    ).astype(np.uint8)

    area = max(1, int((mask > 0).sum()))
    # LED arrows are intentionally low-resolution and often appear as several
    # disconnected pixels after H.264 compression.  Aggregate colored pixels
    # rather than requiring one connected blob.
    green_area = int(green.sum())
    red_area = int(red.sum())
    green_score = green_area / area
    red_score = red_area / area
    # Green boards occupy most of the display face.  Red X/arrow LEDs are
    # dimmer and fragmented, so they use a lower but still conservative ratio.
    if green_area >= 20 and green_score >= 0.50 and green_score > red_score * 1.25:
        return "green"
    if red_area >= 8 and red_score >= 0.03:
        return "red"
    return "unknown"


class SignalStateSmoother:
    """Suppress one-frame color glitches in the tiny camera 147 ROIs."""

    def __init__(self, window: int = 7):
        self.history: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=window))

    def update(self, raw_states: dict[str, str]) -> dict[str, str]:
        stable: dict[str, str] = {}
        for lane, state in raw_states.items():
            if state != "unknown":
                self.history[lane].append(state)
            values = list(self.history[lane])
            stable[lane] = Counter(values).most_common(1)[0][0] if values else "unknown"
        return stable


class SignalVideoReader:
    """Read one signal-camera stream by elapsed camera 112 time."""

    def __init__(self, source: Path, camera: str, offset_seconds: float = 0.0):
        self.source = source
        self.camera = camera
        self.offset_seconds = offset_seconds
        self.capture = cv2.VideoCapture(str(source))
        if not self.capture.isOpened():
            raise SystemExit(f"Could not open signal video: {source}")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 25.0
        self.next_frame = 0
        self.last_frame: np.ndarray | None = None

    def states_at(self, elapsed_seconds: float) -> dict[str, dict[str, str]]:
        target = max(0, round((elapsed_seconds + self.offset_seconds) * self.fps))
        if target < self.next_frame:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.next_frame = target
        while self.next_frame <= target:
            ok, frame = self.capture.read()
            if not ok:
                break
            self.last_frame = frame
            self.next_frame += 1
        if self.last_frame is None:
            return {
                f"lane_{index}": {"primary": "unknown"}
                for index in range(1, 5)
            }
        return classify_signal_roi_states(self.last_frame, self.camera)

    def close(self) -> None:
        self.capture.release()


def signal_rois_for_camera(frame: np.ndarray, camera: str) -> dict[str, dict[str, np.ndarray]]:
    if camera == "156":
        return camera_156_signal_rois(frame)
    return camera_147_signal_rois(frame)


def classify_signal_roi_states(frame: np.ndarray, camera: str) -> dict[str, dict[str, str]]:
    """Read every primary/backup LED ROI for one signal camera."""
    return {
        lane: {name: classify_signal_state(frame, polygon) for name, polygon in roi_set.items()}
        for lane, roi_set in signal_rois_for_camera(frame, camera).items()
    }


def resolve_signal_roi_states(
    roi_states: dict[str, dict[str, str]]
) -> tuple[dict[str, str], dict[str, str]]:
    """Select one reliable LED colour per lane from its primary/backup ROIs.

    The primary remains authoritative when it is readable.  A backup takes
    over only when the primary is ``unknown``.  If two readable backup ROIs
    disagree, the lane is kept unknown rather than guessing.
    """
    resolved: dict[str, str] = {}
    source: dict[str, str] = {}
    for index in range(1, 5):
        lane = f"lane_{index}"
        values = roi_states.get(lane, {})
        primary = values.get("primary", "unknown")
        backup_values = [
            state for name, state in values.items()
            if name != "primary" and state in {"green", "red"}
        ]
        if primary in {"green", "red"}:
            resolved[lane] = primary
            source[lane] = "primary"
        elif len(set(backup_values)) == 1 and backup_values:
            resolved[lane] = backup_values[0]
            source[lane] = "backup"
        else:
            resolved[lane] = "unknown"
            source[lane] = "unknown"
    return resolved, source


def box_is_inside_roi(box, roi_points: np.ndarray | None) -> bool:
    """Use a vehicle's tyre-side centre, rather than its whole box, for ROI membership."""
    if roi_points is None:
        return True
    x1, _y1, x2, y2 = box.xyxy[0].cpu().tolist()
    return point_is_inside_roi(((x1 + x2) / 2, y2), roi_points)


def box_is_inside_lanes(box, lane_points: dict[str, np.ndarray] | None) -> bool:
    """Keep a vehicle only when its tyre-side centre falls in one of 4 lanes."""
    if not lane_points:
        return True
    x1, _y1, x2, y2 = box.xyxy[0].cpu().tolist()
    return point_lane(((x1 + x2) / 2, y2), lane_points) is not None


def mask_outside_roi(frame: np.ndarray, roi_points: np.ndarray | None) -> np.ndarray:
    """Hide pixels outside the selected scan area before raw YOLO inference."""
    if roi_points is None:
        return frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points.astype(np.int32)], 255)
    return cv2.bitwise_and(frame, frame, mask=mask)


def point_is_inside_roi(bottom_center: tuple[float, float], roi_points: np.ndarray | None) -> bool:
    if roi_points is None:
        return True
    return cv2.pointPolygonTest(roi_points.astype(np.float32), bottom_center, False) >= 0


@dataclass
class TrackState:
    """Small display cache used while ByteTrack temporarily loses a detection."""

    label: str
    bbox_xyxy: list[float]
    center: tuple[float, float]
    velocity: tuple[float, float]
    last_seen_frame: int
    observations: int
    confidence: float
    vertical_motion: deque[float] = field(default_factory=lambda: deque(maxlen=10))


def track_directions(track_memory: dict[int, TrackState]) -> dict[int, str]:
    """Classify the recent vertical motion of confirmed ByteTrack objects.

    Image ``y`` gets smaller when a vehicle travels up the bridge, and grows
    when it travels down toward the camera.  Summing a short motion window
    avoids marking a stationary vehicle wrong-way when its detection box
    jitters by a pixel or two.
    """
    directions: dict[int, str] = {}
    for track_id, state in track_memory.items():
        if state.observations < 4 or len(state.vertical_motion) < 3:
            continue
        movement_y = sum(state.vertical_motion)
        if movement_y <= -3.0:
            directions[track_id] = "up"
        elif movement_y >= 3.0:
            directions[track_id] = "down"
        else:
            directions[track_id] = "unknown"
    return directions


def wrong_way_track_ids(
    track_memory: dict[int, TrackState],
    directions: dict[int, str],
    lane_points: dict[str, np.ndarray] | None,
    allowed_by_lane: dict[str, str],
) -> set[int]:
    """Return only tracks moving opposite to a readable lane direction."""
    violations: set[int] = set()
    if not lane_points:
        return violations
    for track_id, direction in directions.items():
        state = track_memory.get(track_id)
        if state is None:
            continue
        x1, _y1, x2, y2 = state.bbox_xyxy
        lane_id = point_lane(((x1 + x2) / 2, y2), lane_points)
        expected = allowed_by_lane.get(lane_id, "unknown") if lane_id else "unknown"
        if direction in {"up", "down"} and expected in {"up", "down"} and direction != expected:
            violations.add(track_id)
    return violations


def stabilized_tracks(
    result,
    class_map: dict[int, str],
    class_aliases: dict[str, str],
    roi_points: np.ndarray | None,
    lane_points: dict[str, np.ndarray] | None,
    memory: dict[int, TrackState],
    frame_number: int,
    hold_frames: int,
    max_held_tracks: int,
) -> tuple[dict[int, str], list[dict[str, object]]]:
    """Lock a track's first label and briefly predict its box during occlusion."""
    label_overrides: dict[int, str] = {}
    seen_ids: set[int] = set()
    boxes = result.boxes
    if boxes is not None and boxes.id is not None:
        ids = boxes.id.int().cpu().tolist()
        for box, track_id in zip(boxes, ids):
            class_id = int(box.cls.item())
            if (
                class_id not in class_map
                or not box_is_inside_roi(box, roi_points)
                or not box_is_inside_lanes(box, lane_points)
            ):
                continue
            xyxy = [float(value) for value in box.xyxy[0].cpu().tolist()]
            center = ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)
            raw_label = class_aliases.get(class_map[class_id], class_map[class_id])
            confidence = float(box.conf.item())
            state = memory.get(track_id)
            if state is None:
                state = TrackState(raw_label, xyxy, center, (0.0, 0.0), frame_number, 1, confidence)
                memory[track_id] = state
            else:
                state.velocity = (center[0] - state.center[0], center[1] - state.center[1])
                state.vertical_motion.append(state.velocity[1])
                state.bbox_xyxy = xyxy
                state.center = center
                state.last_seen_frame = frame_number
                state.observations += 1
                state.confidence = confidence
            # Keep the first stable track label so it does not flicker between
            # bus/truck or motorcycle/car as detections vary frame to frame.
            label_overrides[track_id] = state.label
            seen_ids.add(track_id)

    retained_candidates: list[tuple[int, int, float, dict[str, object]]] = []
    frame_height, frame_width = result.orig_img.shape[:2]
    for track_id, state in list(memory.items()):
        missed_frames = frame_number - state.last_seen_frame
        if track_id in seen_ids or missed_frames <= 0:
            continue
        # A one-frame false positive must disappear immediately rather than
        # creating a long-lived ghost box.  Confirm real tracks first.
        if missed_frames <= hold_frames and state.observations >= 5 and state.confidence >= 0.50:
            dx, dy = state.velocity[0] * missed_frames, state.velocity[1] * missed_frames
            x1, y1, x2, y2 = state.bbox_xyxy
            predicted = [
                max(0.0, min(frame_width - 1.0, x1 + dx)),
                max(0.0, min(frame_height - 1.0, y1 + dy)),
                max(0.0, min(frame_width - 1.0, x2 + dx)),
                max(0.0, min(frame_height - 1.0, y2 + dy)),
            ]
            bottom_center = ((predicted[0] + predicted[2]) / 2, predicted[3])
            if point_is_inside_roi(bottom_center, roi_points):
                retained_candidates.append(
                    (
                        state.observations,
                        -missed_frames,
                        state.confidence,
                        {
                            "track_id": track_id,
                            "label": state.label,
                            "bbox_xyxy": predicted,
                            "bottom_center": bottom_center,
                        },
                    )
                )
        # Let ByteTrack keep its own longer-lived state, but do not keep stale
        # display data forever after an object has left the scene.
        if missed_frames > max(hold_frames, 90):
            del memory[track_id]
    # Prefer confirmed, recently observed, high-confidence tracks.  This avoids
    # filling a busy scene with stale boxes when several vehicles overlap.
    retained_candidates.sort(key=lambda item: item[:3], reverse=True)
    retained_tracks = [item[3] for item in retained_candidates[:max_held_tracks]]
    return label_overrides, retained_tracks


def gate_endpoints(roi_points: np.ndarray, y_normalized: float, frame_height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Intersect a horizontal gate at ``y_normalized`` with the bridge polygon."""
    target_y = round(y_normalized * frame_height)
    intersections: list[tuple[int, int]] = []
    for start, end in zip(roi_points, np.roll(roi_points, -1, axis=0)):
        x1, y1 = map(int, start)
        x2, y2 = map(int, end)
        if y1 == y2 or not min(y1, y2) <= target_y <= max(y1, y2):
            continue
        ratio = (target_y - y1) / (y2 - y1)
        intersections.append((round(x1 + ratio * (x2 - x1)), target_y))
    if len(intersections) != 2:
        raise ValueError("Gate does not cross the bridge ROI exactly twice")
    return tuple(sorted(intersections))  # left, right


def horizontal_polygon_span(polygon: np.ndarray, target_y: int) -> tuple[int, int] | None:
    """Return the left/right intersections of a lane polygon at one y level."""
    intersections: list[float] = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        x1, y1 = map(float, start)
        x2, y2 = map(float, end)
        if y1 == y2 or not min(y1, y2) <= target_y <= max(y1, y2):
            continue
        ratio = (target_y - y1) / (y2 - y1)
        intersections.append(x1 + ratio * (x2 - x1))
    if len(intersections) < 2:
        return None
    return round(min(intersections)), round(max(intersections))


def draw_text_panel(
    frame: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int]]],
    corner: str,
    font_scale: float = 0.43,
    padding: int = 5,
    margin: int = 6,
) -> None:
    """Draw a compact panel sized to its longest text line.

    ``corner`` is deliberately limited to the two locations used by the three
    camera outputs.  Text extents are measured with OpenCV before drawing, so
    the dark panel ends just after the last letter rather than reserving a
    fixed, scene-blocking rectangle.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    metrics = [cv2.getTextSize(text, font, font_scale, 1) for text, _color in lines]
    max_width = max(width for (width, _height), _baseline in metrics)
    line_height = max(height + baseline for (_width, height), baseline in metrics)
    panel_width = max_width + padding * 2
    panel_height = line_height * len(lines) + padding * 2
    frame_height, frame_width = frame.shape[:2]
    x = frame_width - panel_width - margin if corner == "top_right" else margin
    y = margin if corner == "top_right" else frame_height - panel_height - margin
    x = max(0, x)
    y = max(0, y)
    cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
    baseline_y = y + padding + metrics[0][0][1]
    for (text, color), ((_, text_height), _baseline) in zip(lines, metrics):
        cv2.putText(frame, text, (x + padding, baseline_y), font, font_scale, color, 1, cv2.LINE_AA)
        baseline_y += line_height


def draw_bridge_guides(
    frame,
    roi_points: np.ndarray,
    show_gates: bool,
    lane_points: dict[str, np.ndarray] | None = None,
    lane_signals: dict[str, dict[str, object]] | None = None,
) -> None:
    """Draw the selected ROI, fused lane directions, and optional Taksin gates."""
    # Keep the outer polygon as the detection filter, but show only the
    # individual lane guides so the camera-112 view stays uncluttered.
    lane_label_baseline = round(frame.shape[0] * 0.644)
    lane_arrow_center_y = round(frame.shape[0] * 0.735)
    lane_label_font = cv2.FONT_HERSHEY_SIMPLEX
    lane_label_scale = 0.38
    lane_guide_color = (255, 255, 255)
    for lane_name, polygon in (lane_points or {}).items():
        summary = (lane_signals or {}).get(lane_name, {})
        direction = str(summary.get("direction", "unknown"))
        # The road guides deliberately stay neutral.  Traffic direction is
        # conveyed by the white arrow, while green/red remain reserved for
        # the vehicle boxes and the signal-comparison panel.
        cv2.polylines(
            frame,
            [polygon],
            isClosed=True,
            color=lane_guide_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        span = horizontal_polygon_span(polygon, lane_label_baseline)
        if span:
            label = f"{lane_name.replace('_', ' ')} {direction}"
            (label_width, _label_height), _baseline = cv2.getTextSize(
                label, lane_label_font, lane_label_scale, 1
            )
            label_x = round((span[0] + span[1] - label_width) / 2)
            cv2.putText(
                frame,
                label,
                (max(2, label_x), lane_label_baseline),
                lane_label_font,
                lane_label_scale,
                lane_guide_color,
                1,
                cv2.LINE_AA,
            )
        arrow_span = horizontal_polygon_span(polygon, lane_arrow_center_y)
        if arrow_span and direction in {"up", "down"}:
            arrow_x = round((arrow_span[0] + arrow_span[1]) / 2)
            arrow_length = max(28, round(frame.shape[0] * 0.095))
            half_length = round(arrow_length / 2)
            if direction == "up":
                start, end = (arrow_x, lane_arrow_center_y + half_length), (arrow_x, lane_arrow_center_y - half_length)
            else:
                start, end = (arrow_x, lane_arrow_center_y - half_length), (arrow_x, lane_arrow_center_y + half_length)
            cv2.arrowedLine(
                frame,
                start,
                end,
                lane_guide_color,
                2,
                cv2.LINE_AA,
                tipLength=0.32,
            )
    if lane_signals:
        # A compact comparison panel keeps all raw light states readable even
        # when the four lane polygons converge at the far end of the bridge.
        panel_lines: list[tuple[str, tuple[int, int, int]]] = []
        for index in range(1, 5):
            lane = f"lane_{index}"
            summary = lane_signals.get(lane, {})
            direction = str(summary.get("direction", "unknown"))
            light_147 = str(summary.get("light_147", "unknown"))[0].upper()
            light_156 = str(summary.get("light_156", "unknown"))[0].upper()
            source = str(summary.get("source", "none"))
            color = {"up": (0, 220, 0), "down": (0, 0, 255)}.get(direction, (0, 220, 220))
            panel_lines.append((f"L{index}: {direction} | 147:{light_147} 156:{light_156} | {source}", color))
        draw_text_panel(frame, panel_lines, "top_right")
    if not show_gates:
        return
    gate_specs = (("GATE A", GATE_A_NORMALIZED, (255, 0, 0)), ("GATE B", GATE_B_NORMALIZED, (0, 0, 255)))
    for label, (_x_normalized, y_normalized), color in gate_specs:
        left, right = gate_endpoints(roi_points, y_normalized, frame.shape[0])
        cv2.line(frame, left, right, color, 4, cv2.LINE_AA)
        label_anchor = (left[0] + 8, max(26, left[1] - 10))
        cv2.putText(frame, label, label_anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_signal_guides(
    frame: np.ndarray,
    camera: str,
    signal_states: dict[str, str],
    roi_states: dict[str, dict[str, str]] | None = None,
    selected_sources: dict[str, str] | None = None,
) -> None:
    """Draw all primary/backup LED ROIs and their selected lane states."""
    rois = signal_rois_for_camera(frame, camera)
    roi_states = roi_states or {}
    selected_sources = selected_sources or {}
    panel_lines: list[tuple[str, tuple[int, int, int]]] = [(f"CAMERA {camera} LIGHTS", (255, 255, 255))]
    for index in range(1, 5):
        lane = f"lane_{index}"
        light = signal_states.get(lane, "unknown")
        direction = direction_from_signal(camera, light)
        # green/red are the actual LED colours.  Yellow is only for unknown.
        color = {"green": (0, 220, 0), "red": (0, 0, 255)}.get(light, (0, 220, 220))
        labels: list[str] = []
        for roi_name, polygon in rois[lane].items():
            roi_light = roi_states.get(lane, {}).get(roi_name, "unknown")
            roi_color = {"green": (0, 220, 0), "red": (0, 0, 255)}.get(roi_light, (0, 220, 220))
            cv2.polylines(frame, [polygon], True, roi_color, 2, cv2.LINE_AA)
            labels.append(f"{roi_name[0].upper()}={roi_light}")
        source = selected_sources.get(lane, "unknown")
        panel_lines.append((
            f"112 L{index}: {' '.join(labels)} -> {light} ({source}) -> {direction}",
            color,
        ))
    draw_text_panel(frame, panel_lines, "bottom_left")


def grouped_detection(
    result,
    class_map: dict[int, str],
    include_track_id: bool = False,
    class_aliases: dict[str, str] | None = None,
    roi_points: np.ndarray | None = None,
    label_overrides: dict[int, str] | None = None,
    retained_tracks: list[dict[str, object]] | None = None,
    lane_points: dict[str, np.ndarray] | None = None,
    require_lane_membership: bool = True,
    direction_by_track: dict[int, str] | None = None,
    allowed_by_lane: dict[str, str] | None = None,
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    boxes = result.boxes
    class_aliases = class_aliases or {}
    label_overrides = label_overrides or {}
    direction_by_track = direction_by_track or {}
    allowed_by_lane = allowed_by_lane or {}

    def movement_fields(bottom_center: list[float], track_id: int | None) -> dict[str, object]:
        lane_id = point_lane((bottom_center[0], bottom_center[1]), lane_points) if lane_points else None
        direction = direction_by_track.get(track_id) if track_id is not None else None
        expected = allowed_by_lane.get(lane_id, "unknown") if lane_id else "unknown"
        return {
            "lane_id": lane_id,
            "direction": direction or "unknown",
            "expected_direction": expected,
            "wrong_way": bool(direction and direction != "unknown" and expected != "unknown" and direction != expected),
        }

    if boxes is not None:
        ids = boxes.id.int().cpu().tolist() if include_track_id and boxes.id is not None else [None] * len(boxes)
        for box, track_id in zip(boxes, ids):
            class_id = int(box.cls.item())
            if (
                class_id not in class_map
                or not box_is_inside_roi(box, roi_points)
                or (require_lane_membership and not box_is_inside_lanes(box, lane_points))
            ):
                continue
            output_name = label_overrides.get(track_id, class_aliases.get(class_map[class_id], class_map[class_id]))
            xyxy = [round(float(value), 2) for value in box.xyxy[0].cpu().tolist()]
            bottom_center = [round((xyxy[0] + xyxy[2]) / 2, 2), round(xyxy[3], 2)]
            detection = {
                "class_id": class_id,
                "confidence": round(float(box.conf.item()), 4),
                "bbox_xyxy": xyxy,
                "bottom_center": bottom_center,
                "occluded_prediction": False,
                **movement_fields(bottom_center, track_id),
            }
            if include_track_id:
                detection["track_id"] = track_id
            grouped[output_name].append(detection)
    for track in retained_tracks or []:
        xyxy = [round(float(value), 2) for value in track["bbox_xyxy"]]
        bottom_center = [round((xyxy[0] + xyxy[2]) / 2, 2), round(xyxy[3], 2)]
        if (
            require_lane_membership
            and lane_points
            and point_lane((bottom_center[0], bottom_center[1]), lane_points) is None
        ):
            continue
        track_id = track["track_id"]
        grouped[str(track["label"])].append(
            {
                "class_id": None,
                "track_id": track_id,
                "confidence": None,
                "bbox_xyxy": xyxy,
                "bottom_center": bottom_center,
                "occluded_prediction": True,
                **movement_fields(bottom_center, track_id),
            }
        )
    return grouped


def detections_grouped_by_lane(
    detections_by_class: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Present raw detections by their lane after the large-ROI scan.

    ``outside_lane`` is intentional: it preserves a valid detection in the
    scan area whose bottom-centre is on a divider or outside the four lane
    polygons, instead of silently deleting it.
    """
    by_lane: dict[str, list[dict]] = {
        **{f"lane_{index}": [] for index in range(1, 5)},
        "outside_lane": [],
    }
    for class_name, detections in detections_by_class.items():
        for detection in detections:
            item = {"class_name": class_name, **detection}
            lane_id = str(detection.get("lane_id") or "outside_lane")
            by_lane.setdefault(lane_id, []).append(item)
    return by_lane


def apply_class_aliases(model: YOLO, class_aliases: dict[str, str]) -> None:
    """Change only the display names in-place; model class IDs stay unchanged."""
    if not class_aliases:
        return
    # ``YOLO.names`` returns a copy; mutate the underlying model names so
    # Ultralytics' plotted annotations also use the aliased label.
    names = model.model.names
    if isinstance(names, dict):
        for class_id, name in list(names.items()):
            names[class_id] = class_aliases.get(str(name).lower(), str(name))
    else:
        for class_id, name in enumerate(names):
            names[class_id] = class_aliases.get(str(name).lower(), str(name))


def annotated_frame(
    result,
    class_map: dict[int, str],
    class_aliases: dict[str, str] | None = None,
    roi_points: np.ndarray | None = None,
    show_gates: bool = True,
    label_overrides: dict[int, str] | None = None,
    retained_tracks: list[dict[str, object]] | None = None,
    lane_points: dict[str, np.ndarray] | None = None,
    lane_signals: dict[str, dict[str, object]] | None = None,
    lane_filter_points: dict[str, np.ndarray] | None = None,
    require_lane_membership: bool = True,
    alert_wrong_way: bool = False,
    wrong_way_ids: set[int] | None = None,
    source_frame: np.ndarray | None = None,
) -> object:
    """Render boxes and optionally highlight tracked wrong-way vehicles."""
    frame = (source_frame if source_frame is not None else result.orig_img).copy()
    if roi_points is not None:
        draw_bridge_guides(frame, roi_points, show_gates, lane_points, lane_signals)
    boxes = result.boxes
    class_aliases = class_aliases or {}
    label_overrides = label_overrides or {}
    wrong_way_ids = wrong_way_ids or set()

    def draw_box(label: str, xyxy: list[float], occluded: bool = False, wrong_way: bool = False) -> None:
        if alert_wrong_way:
            color = (0, 0, 255) if wrong_way else (0, 220, 0)
            display_label = label
            text_color = (0, 0, 0)
        else:
            color_name = "car" if label in {"car", "taxi"} else label
            color = CLASS_COLORS_BGR.get(color_name, (255, 255, 255))
            display_label = f"{label} (hold)" if occluded else label
            text_color = color
        x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1 if occluded else 2)
        cv2.putText(
            frame,
            display_label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 if occluded else 0.65,
            text_color,
            1 if occluded else 2,
            cv2.LINE_AA,
        )

    if boxes is not None:
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
        for box, track_id in zip(boxes, ids):
            class_id = int(box.cls.item())
            if (
                class_id not in class_map
                or not box_is_inside_roi(box, roi_points)
                or (require_lane_membership and not box_is_inside_lanes(box, lane_filter_points))
            ):
                continue
            original_name = class_map[class_id]
            output_name = label_overrides.get(track_id, class_aliases.get(original_name, original_name))
            draw_box(
                output_name,
                [float(value) for value in box.xyxy[0].cpu().tolist()],
                wrong_way=track_id in wrong_way_ids,
            )
    for track in retained_tracks or []:
        draw_box(
            str(track["label"]),
            list(track["bbox_xyxy"]),
            occluded=True,
            wrong_way=track["track_id"] in wrong_way_ids,
        )
    return frame


def render_signal_view(source: Path, output_dir: Path, camera: str) -> tuple[Path, Path]:
    """Write an annotated signal-camera view and matching state JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {camera} signal video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    video_path = output_dir / f"{source.stem}_lights_annotated.mp4"
    jsonl_path = output_dir / f"{source.stem}_lights.jsonl"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Could not create signal output: {video_path}")
    smoother = SignalStateSmoother()
    try:
        with jsonl_path.open("w", encoding="utf-8") as handle:
            frame_number = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                roi_states = classify_signal_roi_states(frame, camera)
                raw_states, selected_sources = resolve_signal_roi_states(roi_states)
                states = smoother.update(raw_states)
                annotated = frame.copy()
                draw_signal_guides(annotated, camera, states, roi_states, selected_sources)
                writer.write(annotated)
                record = {
                    "frame": frame_number,
                    "time_seconds": round(frame_number / fps, 3),
                    "camera": camera,
                    "light_roi_states": roi_states,
                    "light_states": states,
                    "selected_light_source": selected_sources,
                    "directions_for_camera_112": {
                        lane: direction_from_signal(camera, light)
                        for lane, light in states.items()
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                frame_number += 1
    finally:
        capture.release()
        writer.release()
    return video_path, jsonl_path


def run_image(
    model: YOLO,
    class_map: dict[int, str],
    source: Path,
    output_dir: Path,
    conf: float,
    imgsz: int,
    class_aliases: dict[str, str],
    bridge_only: bool,
    show_gates: bool,
    agnostic_nms: bool,
    iou: float,
    profile: str,
    show_lanes: bool,
    signal147_source: Path | None,
    signal147_offset: float,
    signal156_source: Path | None,
    signal156_offset: float,
) -> None:
    apply_class_aliases(model, class_aliases)
    results = model.predict(
        source=str(source),
        classes=list(class_map),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        agnostic_nms=agnostic_nms,
        verbose=False,
    )
    result = results[0]
    roi_points = bridge_roi(result.orig_img, profile) if bridge_only else None
    lane_filter_points = lane_rois(result.orig_img, profile)
    lane_points = lane_filter_points if show_lanes else None
    states_147: dict[str, str] = {}
    states_156: dict[str, str] = {}
    roi_states_147: dict[str, dict[str, str]] = {}
    roi_states_156: dict[str, dict[str, str]] = {}
    selected_147: dict[str, str] = {}
    selected_156: dict[str, str] = {}
    reader_147 = SignalVideoReader(signal147_source, "147", signal147_offset) if signal147_source else None
    reader_156 = SignalVideoReader(signal156_source, "156", signal156_offset) if signal156_source else None
    if reader_147:
        roi_states_147 = reader_147.states_at(0.0)
        raw_states_147, selected_147 = resolve_signal_roi_states(roi_states_147)
        states_147 = SignalStateSmoother().update(raw_states_147)
        reader_147.close()
    if reader_156:
        roi_states_156 = reader_156.states_at(0.0)
        raw_states_156, selected_156 = resolve_signal_roi_states(roi_states_156)
        states_156 = SignalStateSmoother().update(raw_states_156)
        reader_156.close()
    lane_signals = fuse_lane_signals(states_147, states_156)
    allowed_by_lane = {lane: str(summary["direction"]) for lane, summary in lane_signals.items()}
    grouped = grouped_detection(
        result,
        class_map,
        class_aliases=class_aliases,
        roi_points=roi_points,
        lane_points=lane_filter_points,
        allowed_by_lane=allowed_by_lane,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / f"{source.stem}_annotated.jpg"
    cv2.imwrite(
        str(annotated_path),
        annotated_frame(
            result,
            class_map,
            class_aliases,
            roi_points,
            show_gates,
            lane_points=lane_points,
            lane_signals=lane_signals,
            lane_filter_points=lane_filter_points,
        ),
    )
    payload = {
        "source": str(source),
        "mode": "image_detection",
        "camera_profile": profile,
        "classes": class_map,
        "class_aliases": class_aliases,
        "bridge_roi_normalized": BRIDGE_ROI_NORMALIZED if bridge_only else None,
        "roi_filter": "bottom_center_inside_bridge_polygon" if bridge_only else None,
        "gates_normalized": {"gate_a": GATE_A_NORMALIZED, "gate_b": GATE_B_NORMALIZED}
        if bridge_only and show_gates
        else None,
        "detections_by_class": grouped,
        "signal_147_roi_states": roi_states_147,
        "signal_147_states": states_147,
        "signal_147_selected_light_source": selected_147,
        "signal_156_roi_states": roi_states_156,
        "signal_156_states": states_156,
        "signal_156_selected_light_source": selected_156,
        "lane_signal_fusion": lane_signals,
        "lane_direction_rule": "147: green=up/red=down; 156: green=down/red=up; unknown uses the other camera",
        "note": "A single image has detections but no temporal track IDs; use video mode for ByteTrack IDs.",
    }
    (output_dir / f"{source.stem}_detections.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({name: len(items) for name, items in grouped.items()}, ensure_ascii=False))
    print(f"Annotated image: {annotated_path}")


def run_video(
    model: YOLO,
    class_map: dict[int, str],
    source: Path,
    output_dir: Path,
    conf: float,
    imgsz: int,
    tracker: str,
    class_aliases: dict[str, str],
    bridge_only: bool,
    show_gates: bool,
    agnostic_nms: bool,
    occlusion_hold: int,
    iou: float,
    max_held_tracks: int,
    profile: str,
    show_lanes: bool,
    signal147_source: Path | None,
    signal147_offset: float,
    signal156_source: Path | None,
    signal156_offset: float,
    save_signal_views: bool,
    alert_wrong_way: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_class_aliases(model, class_aliases)
    video_path = output_dir / f"{source.stem}_tracked.mp4"
    jsonl_path = output_dir / f"{source.stem}_tracks.jsonl"
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise SystemExit(f"Could not create video output: {video_path}")
    reader_147 = SignalVideoReader(signal147_source, "147", signal147_offset) if signal147_source else None
    reader_156 = SignalVideoReader(signal156_source, "156", signal156_offset) if signal156_source else None
    smoother_147 = SignalStateSmoother()
    smoother_156 = SignalStateSmoother()
    results = model.track(
        source=str(source),
        stream=True,
        persist=True,
        tracker=tracker,
        classes=list(class_map),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        agnostic_nms=agnostic_nms,
        save=False,
        verbose=False,
    )
    track_memory: dict[int, TrackState] = {}
    try:
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for frame_number, result in enumerate(results):
                elapsed_seconds = frame_number / fps
                roi_points = bridge_roi(result.orig_img, profile) if bridge_only else None
                lane_filter_points = lane_rois(result.orig_img, profile)
                lane_points = lane_filter_points if show_lanes else None
                roi_states_147 = reader_147.states_at(elapsed_seconds) if reader_147 else {}
                roi_states_156 = reader_156.states_at(elapsed_seconds) if reader_156 else {}
                raw_147, selected_147 = resolve_signal_roi_states(roi_states_147) if roi_states_147 else ({}, {})
                raw_156, selected_156 = resolve_signal_roi_states(roi_states_156) if roi_states_156 else ({}, {})
                states_147 = smoother_147.update(raw_147) if raw_147 else {}
                states_156 = smoother_156.update(raw_156) if raw_156 else {}
                lane_signals = fuse_lane_signals(states_147, states_156)
                allowed_by_lane = {lane: str(summary["direction"]) for lane, summary in lane_signals.items()}
                label_overrides, retained_tracks = stabilized_tracks(
                    result,
                    class_map,
                    class_aliases,
                    roi_points,
                    lane_filter_points,
                    track_memory,
                    frame_number,
                    occlusion_hold,
                    max_held_tracks,
                )
                direction_by_track = track_directions(track_memory)
                violations = wrong_way_track_ids(
                    track_memory,
                    direction_by_track,
                    lane_filter_points,
                    allowed_by_lane,
                )
                writer.write(
                    annotated_frame(
                        result,
                        class_map,
                        class_aliases,
                        roi_points,
                        show_gates,
                        label_overrides,
                        retained_tracks,
                        lane_points,
                        lane_signals,
                        lane_filter_points=lane_filter_points,
                        alert_wrong_way=alert_wrong_way,
                        wrong_way_ids=violations,
                    )
                )
                grouped = grouped_detection(
                    result,
                    class_map,
                    include_track_id=True,
                    class_aliases=class_aliases,
                    roi_points=roi_points,
                    label_overrides=label_overrides,
                    retained_tracks=retained_tracks,
                    lane_points=lane_filter_points,
                    direction_by_track=direction_by_track,
                    allowed_by_lane=allowed_by_lane,
                )
                record = {
                    "frame": frame_number,
                    "time_seconds": round(elapsed_seconds, 3),
                    "camera_profile": profile,
                    "signal_147_roi_states": roi_states_147,
                    "signal_147_states": states_147,
                    "signal_147_selected_light_source": selected_147,
                    "signal_156_roi_states": roi_states_156,
                    "signal_156_states": states_156,
                    "signal_156_selected_light_source": selected_156,
                    "lane_signal_fusion": lane_signals,
                    "lane_directions": allowed_by_lane,
                    "wrong_way_track_ids": sorted(violations),
                    "tracks_by_class": grouped,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if (frame_number + 1) % 100 == 0:
                    print(f"Processed {frame_number + 1}/{total_frames} frames", flush=True)
    finally:
        writer.release()
        if reader_147:
            reader_147.close()
        if reader_156:
            reader_156.close()
    print(f"Track records: {jsonl_path}")
    print(f"Annotated video: {video_path}")
    if save_signal_views:
        if signal147_source:
            signal_video, signal_jsonl = render_signal_view(signal147_source, output_dir, "147")
            print(f"Camera 147 light view: {signal_video}")
            print(f"Camera 147 light records: {signal_jsonl}")
        if signal156_source:
            signal_video, signal_jsonl = render_signal_view(signal156_source, output_dir, "156")
            print(f"Camera 156 light view: {signal_video}")
            print(f"Camera 156 light records: {signal_jsonl}")


def run_raw_video(
    model: YOLO,
    class_map: dict[int, str],
    source: Path,
    output_dir: Path,
    conf: float,
    imgsz: int,
    class_aliases: dict[str, str],
    bridge_only: bool,
    show_gates: bool,
    agnostic_nms: bool,
    iou: float,
    profile: str,
    show_lanes: bool,
    signal147_source: Path | None,
    signal147_offset: float,
    signal156_source: Path | None,
    signal156_offset: float,
    save_signal_views: bool,
) -> None:
    """Run raw YOLO on the full camera-112 scan ROI, then assign lane IDs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_class_aliases(model, class_aliases)
    video_path = output_dir / f"{source.stem}_raw.mp4"
    jsonl_path = output_dir / f"{source.stem}_raw_detections.jsonl"
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Could not create video output: {video_path}")
    reader_147 = SignalVideoReader(signal147_source, "147", signal147_offset) if signal147_source else None
    reader_156 = SignalVideoReader(signal156_source, "156", signal156_offset) if signal156_source else None
    smoother_147 = SignalStateSmoother()
    smoother_156 = SignalStateSmoother()
    try:
        with jsonl_path.open("w", encoding="utf-8") as handle:
            frame_number = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                elapsed_seconds = frame_number / fps
                lane_filter_points = lane_rois(frame, profile)
                lane_points = lane_filter_points if show_lanes else None
                roi_points = bridge_roi(frame, profile) if bridge_only else None

                # YOLO sees the complete user-defined scan area.  Lane polygons
                # are not used as a detection mask; they assign ``lane_id`` only
                # after a box has been found.  The original frame is restored for
                # the visual output.
                result = model.predict(
                    source=mask_outside_roi(frame, roi_points),
                    classes=list(class_map),
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    agnostic_nms=agnostic_nms,
                    verbose=False,
                )[0]

                roi_states_147 = reader_147.states_at(elapsed_seconds) if reader_147 else {}
                roi_states_156 = reader_156.states_at(elapsed_seconds) if reader_156 else {}
                raw_147, selected_147 = resolve_signal_roi_states(roi_states_147) if roi_states_147 else ({}, {})
                raw_156, selected_156 = resolve_signal_roi_states(roi_states_156) if roi_states_156 else ({}, {})
                states_147 = smoother_147.update(raw_147) if raw_147 else {}
                states_156 = smoother_156.update(raw_156) if raw_156 else {}
                lane_signals = fuse_lane_signals(states_147, states_156)
                allowed_by_lane = {lane: str(summary["direction"]) for lane, summary in lane_signals.items()}

                writer.write(
                    annotated_frame(
                        result,
                        class_map,
                        class_aliases,
                        roi_points,
                        show_gates,
                        lane_points=lane_points,
                        lane_signals=lane_signals,
                        lane_filter_points=lane_filter_points,
                        require_lane_membership=False,
                        source_frame=frame,
                    )
                )
                detections = grouped_detection(
                    result,
                    class_map,
                    class_aliases=class_aliases,
                    roi_points=roi_points,
                    lane_points=lane_filter_points,
                    require_lane_membership=False,
                    allowed_by_lane=allowed_by_lane,
                )
                detections_by_lane = detections_grouped_by_lane(detections)
                record = {
                    "frame": frame_number,
                    "time_seconds": round(elapsed_seconds, 3),
                    "mode": "raw_yolo_scan_roi_then_lane_assignment",
                    "camera_profile": profile,
                    "signal_147_roi_states": roi_states_147,
                    "signal_147_states": states_147,
                    "signal_147_selected_light_source": selected_147,
                    "signal_156_roi_states": roi_states_156,
                    "signal_156_states": states_156,
                    "signal_156_selected_light_source": selected_156,
                    "lane_signal_fusion": lane_signals,
                    "lane_directions": allowed_by_lane,
                    "detections_by_class": detections,
                    "detections_by_lane": detections_by_lane,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                frame_number += 1
                if frame_number % 100 == 0:
                    print(f"Processed {frame_number}/{total_frames} frames", flush=True)
    finally:
        capture.release()
        writer.release()
        if reader_147:
            reader_147.close()
        if reader_156:
            reader_156.close()
    print(f"Raw detection records: {jsonl_path}")
    print(f"Annotated video: {video_path}")
    if save_signal_views:
        if signal147_source:
            signal_video, signal_jsonl = render_signal_view(signal147_source, output_dir, "147")
            print(f"Camera 147 light view: {signal_video}")
            print(f"Camera 147 light records: {signal_jsonl}")
        if signal156_source:
            signal_video, signal_jsonl = render_signal_view(signal156_source, output_dir, "156")
            print(f"Camera 156 light view: {signal_video}")
            print(f"Camera 156 light records: {signal_jsonl}")


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise SystemExit(f"Source not found: {args.source}")
    profile = resolve_profile(args.source, args.profile)
    signal147_source = args.signal_source or default_signal_source(args.source, profile)
    signal156_source = args.signal156_source or default_signal156_source(args.source, profile)
    if signal147_source and not signal147_source.exists():
        raise SystemExit(f"Camera 147 signal source not found: {signal147_source}")
    if signal156_source and not signal156_source.exists():
        raise SystemExit(f"Camera 156 signal source not found: {signal156_source}")
    show_lanes = args.show_lanes if args.show_lanes is not None else profile == PROFILE_KRUNG_THON
    wrong_way_alerts = args.wrong_way_alerts if args.wrong_way_alerts is not None else (
        profile == PROFILE_KRUNG_THON
    )
    save_signal_views = args.save_signal_views if args.save_signal_views is not None else (
        profile == PROFILE_KRUNG_THON and bool(signal147_source or signal156_source)
    )
    # Taksin's A/B gates are unrelated to the Krung Thon lane profile.  Keep
    # the old default for Taksin, while Krung Thon starts with lane polygons.
    show_gates = args.show_gates and profile != PROFILE_KRUNG_THON
    model = YOLO(args.model)
    requested_names = {name.strip().lower() for name in args.classes.split(",") if name.strip()} if args.classes else None
    class_map = vehicle_class_map(model, requested_names)
    if not class_map:
        requested = ", ".join(sorted(requested_names or VEHICLE_NAMES))
        raise SystemExit(f"Model has none of the requested classes: {requested}")
    class_aliases: dict[str, str] = {}
    if args.merge_taxi:
        class_aliases["taxi"] = "car"
    if args.merge_car_like:
        class_aliases.update({"taxi": "car", "van": "car", "pickup": "car"})
    if profile == PROFILE_KRUNG_THON:
        # Keep the model's COCO class ID/name for inference, but use the
        # shorter project label in the rendered video and JSON records.
        class_aliases.setdefault("motorcycle", "moto")
    if args.source.is_dir():
        images = sorted(path for path in args.source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise SystemExit(f"No supported images found in: {args.source}")
        for index, image in enumerate(images, start=1):
            print(f"[{index}/{len(images)}] {image.name}")
            run_image(
                model,
                class_map,
                image,
                args.output_dir,
                args.conf,
                args.imgsz,
                class_aliases,
                args.bridge_only,
                show_gates,
                args.agnostic_nms,
                args.iou,
                profile,
                show_lanes,
                signal147_source,
                args.signal_offset,
                signal156_source,
                args.signal156_offset,
            )
    elif args.source.suffix.lower() in IMAGE_SUFFIXES:
        run_image(
            model,
            class_map,
            args.source,
            args.output_dir,
            args.conf,
            args.imgsz,
            class_aliases,
            args.bridge_only,
            show_gates,
            args.agnostic_nms,
            args.iou,
            profile,
            show_lanes,
            signal147_source,
            args.signal_offset,
            signal156_source,
            args.signal156_offset,
        )
    else:
        if args.raw:
            run_raw_video(
                model,
                class_map,
                args.source,
                args.output_dir,
                args.conf,
                args.imgsz,
                class_aliases,
                args.bridge_only,
                show_gates,
                args.agnostic_nms,
                args.iou,
                profile,
                show_lanes,
                signal147_source,
                args.signal_offset,
                signal156_source,
                args.signal156_offset,
                save_signal_views,
            )
        else:
            run_video(
                model,
                class_map,
                args.source,
                args.output_dir,
                args.conf,
                args.imgsz,
                args.tracker,
                class_aliases,
                args.bridge_only,
                show_gates,
                args.agnostic_nms,
                args.occlusion_hold,
                args.iou,
                args.max_held_tracks,
                profile,
                show_lanes,
                signal147_source,
                args.signal_offset,
                signal156_source,
                args.signal156_offset,
                save_signal_views,
                wrong_way_alerts,
            )


if __name__ == "__main__":
    main()
