"""V2 — YOLO vehicle tracking for Krung Thon camera 112 only.

Choose the source in settings.py:
    SOURCE_MODE = "video_files"  # local video file
    SOURCE_MODE = "live_stream"  # live camera stream

V2 never opens, reads, or uses other cameras. Lane direction comes only from
the timetable and the time obtained from camera 112 (or the configured fallback).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import socket
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
from ultralytics.utils import LOGGER

LOGGER.setLevel(logging.ERROR)

# v2 lives in tracking/v2; project-wide regions and payload code live two levels up.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
payload_dir = PROJECT_ROOT / "payload"
if str(payload_dir) not in sys.path:
    sys.path.insert(0, str(payload_dir))

# pyrefly: ignore [missing-import]
from common.traffic_payload import TrafficWindowAggregator
from mqtt.settings import (
    GATEWAY_INPUT_TOPIC,
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_QOS,
    TRACKER_CLIENT_ID,
)
from config.krung_thon_bridge_regions import (
    REFERENCE_HEIGHT,
    REFERENCE_WIDTH,
    camera_112_lane_rois,
    camera_112_roi,
    point_lane,
)

VEHICLE_NAMES = {
    "car", "motorcycle", "motorbike", "bus", "truck", "pickup", "taxi", "van", "truck trailer",
}

CLASS_COLORS_BGR = {
    "car": (255, 180, 0),
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

BRIDGE_ROI_NORMALIZED = (
    (0.170, 0.954), (0.580, 0.160), (0.661, 0.142), (0.508, 0.953),
)
GATE_A_NORMALIZED = (0.576, 0.297)
GATE_B_NORMALIZED = (0.394, 0.771)
PROFILE_TAKSIN = "taksin"
PROFILE_KRUNG_THON = "krung_thon_bridge"
KRUNG_THON_TIMETABLE = (
    ("morning", ((time(5, 30), time(7, 30)), (time(8, 20), time(8, 45))), ("up", "down", "down", "down")),
    ("evening", ((time(15, 30), time(16, 30)), (time(17, 0), time(17, 45)), (time(18, 15), time(20, 30))), ("up", "up", "up", "down")),
)
KRUNG_THON_DEFAULT_DIRECTIONS = ("up", "up", "down", "down")

def parse_clock_timestamp(value: str) -> datetime:
    """Parse an operator override for a camera overlay timestamp."""
    normalized = value.strip().replace("T", " ").replace("/", "-")
    try:
        return datetime.strptime(normalized, "%Y-%m-%d %H:%M:%S")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use YYYY-MM-DD HH:MM:SS, for example 2026-08-26 11:06:44"
        ) from exc

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

def timetable_directions(timestamp: datetime | None) -> tuple[dict[str, str] | None, str | None]:
    """Return the daily timetable direction map at one camera timestamp."""
    if timestamp is None:
        timestamp = datetime.now().astimezone()
    now = timestamp.time()
    for period, windows, directions in KRUNG_THON_TIMETABLE:
        if any(start <= now < end for start, end in windows):
            return {f"lane_{index}": direction for index, direction in enumerate(directions, start=1)}, period
    return {f"lane_{index}": direction for index, direction in enumerate(KRUNG_THON_DEFAULT_DIRECTIONS, start=1)}, "default"

def timestamp_at(start_timestamp: datetime | None, elapsed_seconds: float) -> datetime | None:
    return start_timestamp + timedelta(seconds=elapsed_seconds) if start_timestamp else None

def _clock_crop(frame: np.ndarray, camera: str) -> np.ndarray:
    """Crop the known on-screen timestamp location for the three cameras."""
    height, width = frame.shape[:2]
    if camera == "112":
        return frame[: min(height, 62), : min(width, 310)]
    return frame[max(0, height - 78): height, max(0, width - 320): width]

def read_timestamp_from_frame(frame: np.ndarray, camera: str, scratch_dir: Path) -> datetime | None:
    """OCR one clock overlay once; operator-supplied timestamps remain available.

    Tesseract is only invoked for the first frame of each input video, never
    per frame. If it is unavailable or cannot read the overlay, the caller
    simply omits that camera from the schedule vote.
    """
    crop = _clock_crop(frame, camera)
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    prepared = enlarged
    scratch_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"clock_{camera}_", suffix=".png", dir=scratch_dir, delete=False
    ) as temporary:
        image_path = Path(temporary.name)
    try:
        if not cv2.imwrite(str(image_path), prepared):
            return None
        result = subprocess.run(
            ["tesseract", str(image_path), "stdout", "--psm", "6"],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    finally:
        image_path.unlink(missing_ok=True)
    match = re.search(r"20\d{2}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2}", result.stdout)
    if not match:
        return None
    try:
        return parse_clock_timestamp(match.group())
    except argparse.ArgumentTypeError:
        return None

def read_video_start_timestamp(source: Path, camera: str, scratch_dir: Path) -> datetime | None:
    """Read a camera timestamp from only the first frame of its input video."""
    capture = cv2.VideoCapture(str(source))
    try:
        ok, frame = capture.read()
        return read_timestamp_from_frame(frame, camera, scratch_dir) if ok else None
    finally:
        capture.release()

def resolve_start_timestamp(
    source: Path | None, camera: str, override: datetime | None, scratch_dir: Path
) -> tuple[datetime | None, str]:
    """Use the explicit value first, then one OSD read from that camera."""
    if override is not None:
        return override, "argument"
    if source is None:
        return None, "unavailable"
    detected = read_video_start_timestamp(source, camera, scratch_dir)
    return (detected, "osd_ocr") if detected else (None, "unavailable")

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

def point_is_inside_roi(bottom_center: tuple[float, float], roi_points: np.ndarray | None) -> bool:
    if roi_points is None:
        return True
    return cv2.pointPolygonTest(roi_points.astype(np.float32), bottom_center, False) >= 0

@dataclass
class TrackState:
    """State retained across short ByteTrack ID changes and occlusions."""

    vehicle_id: str
    home_lane_id: str | None
    bbox_xyxy: list[float]
    center: tuple[float, float]
    velocity: tuple[float, float]
    last_seen_frame: int
    observations: int
    confidence: float
    vertical_motion: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    wrong_way_consecutive: int = 0
    gate_sides: dict[str, str] = field(default_factory=dict)
    gate_crossing_directions: dict[str, str] = field(default_factory=dict)
    gate_crossing_frames: dict[str, int] = field(default_factory=dict)
    gate_wrong_way_consecutive: dict[str, int] = field(default_factory=dict)
    wrong_way_gate_id: str | None = None
    wrong_way_event_id: str | None = None


def _bbox_iou(first: list[float], second: list[float]) -> float:
    """Return overlap between two boxes for conservative ID re-association."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0:
        return 0.0
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _predicted_box(state: TrackState, missed_frames: int) -> list[float]:
    dx = state.velocity[0] * missed_frames
    dy = state.velocity[1] * missed_frames
    x1, y1, x2, y2 = state.bbox_xyxy
    return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]


def _next_vehicle_id(lane_id: str | None, lane_counters: dict[str, int]) -> str:
    """Create a human-readable lane-local ID, for example ``L2-V4``."""
    lane_key = lane_id or "lane_unknown"
    lane_counters[lane_key] += 1
    lane_number = re.search(r"(\d+)$", lane_key)
    prefix = f"L{lane_number.group(1)}" if lane_number else "LX"
    return f"{prefix}-V{lane_counters[lane_key]}"


def scaled_wrong_way_gates(
    frame: np.ndarray,
    reference_gates: object,
) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    """Scale the configured 800x450 Gate 1–3 endpoints to this frame."""
    if not isinstance(reference_gates, dict):
        return {}
    height, width = frame.shape[:2]
    scaled: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
    for raw_gate_id, reference_gate in reference_gates.items():
        if not isinstance(reference_gate, (tuple, list)) or len(reference_gate) != 2:
            continue
        try:
            start, end = reference_gate
            x1, y1 = float(start[0]), float(start[1])
            x2, y2 = float(end[0]), float(end[1])
        except (TypeError, IndexError, ValueError):
            continue
        scaled[str(raw_gate_id)] = (
            (round(x1 * width / REFERENCE_WIDTH), round(y1 * height / REFERENCE_HEIGHT)),
            (round(x2 * width / REFERENCE_WIDTH), round(y2 * height / REFERENCE_HEIGHT)),
        )
    return scaled


def wrong_way_gate_side(
    point: tuple[float, float],
    gate_points: tuple[tuple[int, int], tuple[int, int]],
    margin_pixels: float,
) -> str | None:
    """Return a stable side of the gate, ignoring the margin around its line."""
    (x1, y1), (x2, y2) = gate_points
    dx, dy = float(x2 - x1), float(y2 - y1)
    length_squared = dx * dx + dy * dy
    if length_squared <= 0:
        return None
    projection = ((point[0] - x1) * dx + (point[1] - y1) * dy) / length_squared
    if projection < 0.0 or projection > 1.0:
        return None
    signed_distance = ((point[0] - x1) * dy - (point[1] - y1) * dx) / math.sqrt(length_squared)
    if signed_distance >= margin_pixels:
        return "above"
    if signed_distance <= -margin_pixels:
        return "below"
    return None


def update_wrong_way_gate_crossings(
    track_memory: dict[int, TrackState],
    gate_points_by_id: dict[str, tuple[tuple[int, int], tuple[int, int]]],
    frame_number: int,
    margin_pixels: float,
) -> None:
    """Record clear crossings for every configured gate, per stable vehicle ID."""
    if not gate_points_by_id:
        return
    for state in track_memory.values():
        if state.last_seen_frame != frame_number:
            continue
        x1, _y1, x2, y2 = state.bbox_xyxy
        bottom_center = ((x1 + x2) / 2, y2)
        for gate_id, gate_points in gate_points_by_id.items():
            side = wrong_way_gate_side(bottom_center, gate_points, margin_pixels)
            if side is None:
                continue
            previous_side = state.gate_sides.get(gate_id)
            if previous_side is None:
                state.gate_sides[gate_id] = side
                continue
            if side == previous_side:
                continue
            state.gate_crossing_directions[gate_id] = (
                "up" if previous_side == "below" and side == "above" else "down"
            )
            state.gate_crossing_frames[gate_id] = frame_number
            state.gate_sides[gate_id] = side
            state.gate_wrong_way_consecutive[gate_id] = 0


def _reassociate_previous_track(
    memory: dict[int, TrackState],
    seen_ids: set[int],
    bbox_xyxy: list[float],
    center: tuple[float, float],
    lane_id: str | None,
    frame_number: int,
    max_gap_frames: int,
    max_center_distance: float,
    min_iou: float,
) -> int | None:
    """Find a recently lost same-lane track that most likely is this vehicle.

    This needs temporal proximity and geometric agreement.  Lane membership on
    its own is never enough, which prevents merging two following vehicles.
    """
    candidates: list[tuple[float, int]] = []
    for prior_track_id, state in memory.items():
        if prior_track_id in seen_ids:
            continue
        missed_frames = frame_number - state.last_seen_frame
        if missed_frames <= 0 or missed_frames > max_gap_frames:
            continue
        if lane_id and state.home_lane_id and lane_id != state.home_lane_id:
            continue
        predicted = _predicted_box(state, missed_frames)
        predicted_center = (
            (predicted[0] + predicted[2]) / 2,
            (predicted[1] + predicted[3]) / 2,
        )
        distance = math.dist(center, predicted_center)
        speed = math.hypot(*state.velocity)
        allowed_distance = max_center_distance + min(max_center_distance, speed * missed_frames)
        overlap = _bbox_iou(bbox_xyxy, predicted)
        if distance > allowed_distance or (overlap < min_iou and distance > allowed_distance * 0.55):
            continue
        score = (overlap * 2.0) + max(0.0, 1.0 - distance / max(allowed_distance, 1.0))
        candidates.append((score, prior_track_id))
    return max(candidates)[1] if candidates else None


def track_vehicle_ids(track_memory: dict[int, TrackState]) -> dict[int, str]:
    """Map temporary ByteTrack IDs to V2's stable lane-local vehicle IDs."""
    return {track_id: state.vehicle_id for track_id, state in track_memory.items()}


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


def confirmed_wrong_way_track_ids(
    track_memory: dict[int, TrackState],
    directions: dict[int, str],
    lane_points: dict[str, np.ndarray] | None,
    allowed_by_lane: dict[str, str],
    frame_number: int,
    confirm_frames: int,
    min_displacement_pixels: float,
    gate_points_by_id: dict[str, tuple[tuple[int, int], tuple[int, int]]],
    gate_confirm_frames: int,
) -> set[int]:
    """Confirm one wrong-way event after any one clear wrong-direction gate crossing.

    Gate 1–3 are independent observation points.  A vehicle only needs to
    cross one of them in the wrong direction, then continue moving, so a
    vehicle first tracked after another gate can still be detected.  The event
    remains keyed by its stable vehicle ID and is therefore emitted once.
    """
    violations: set[int] = set()
    if not lane_points:
        return violations
    gate_is_required = bool(gate_points_by_id)
    for track_id, direction in directions.items():
        state = track_memory.get(track_id)
        if state is None:
            continue
        x1, _y1, x2, y2 = state.bbox_xyxy
        lane_id = point_lane(((x1 + x2) / 2, y2), lane_points)
        expected = allowed_by_lane.get(lane_id, "unknown") if lane_id else "unknown"
        moving_wrong_way = (
            direction in {"up", "down"}
            and expected in {"up", "down"}
            and direction != expected
        )
        matching_gates = {
            gate_id
            for gate_id, crossing_direction in state.gate_crossing_directions.items()
            if crossing_direction == direction
        }
        # A predicted/held box cannot add confirmation frames by itself.
        if state.last_seen_frame == frame_number:
            state.wrong_way_consecutive = (
                state.wrong_way_consecutive + 1 if moving_wrong_way else 0
            )
            for gate_id in list(state.gate_wrong_way_consecutive):
                if moving_wrong_way and gate_id in matching_gates:
                    state.gate_wrong_way_consecutive[gate_id] += 1
                else:
                    state.gate_wrong_way_consecutive[gate_id] = 0
        vertical_displacement = abs(sum(state.vertical_motion))
        confirmed_gate_id = next(
            (
                gate_id
                for gate_id in matching_gates
                if state.gate_crossing_frames.get(gate_id) is not None
                and state.gate_wrong_way_consecutive.get(gate_id, 0) >= gate_confirm_frames
            ),
            None,
        )
        gate_confirmed = not gate_is_required or confirmed_gate_id is not None
        if (
            moving_wrong_way
            and gate_confirmed
            and state.wrong_way_consecutive >= confirm_frames
            and vertical_displacement >= min_displacement_pixels
        ):
            if state.wrong_way_event_id is None:
                state.wrong_way_event_id = f"WW-{state.vehicle_id}"
                state.wrong_way_gate_id = confirmed_gate_id
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
    lane_counters: dict[str, int],
    reassociate_gap_frames: int,
    reassociate_max_distance: float,
    reassociate_min_iou: float,
) -> tuple[dict[int, str], list[dict[str, object]], dict[int, str]]:
    """Label every detection as vehicle and retain that ID across short losses."""
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
            lane_id = point_lane((center[0], xyxy[3]), lane_points) if lane_points else None
            confidence = float(box.conf.item())
            state = memory.get(track_id)
            if state is None:
                prior_track_id = _reassociate_previous_track(
                    memory,
                    seen_ids,
                    xyxy,
                    center,
                    lane_id,
                    frame_number,
                    reassociate_gap_frames,
                    reassociate_max_distance,
                    reassociate_min_iou,
                )
                if prior_track_id is not None:
                    state = memory.pop(prior_track_id)
                    memory[track_id] = state
                    state.velocity = (center[0] - state.center[0], center[1] - state.center[1])
                    state.vertical_motion.append(state.velocity[1])
                    state.bbox_xyxy = xyxy
                    state.center = center
                    state.last_seen_frame = frame_number
                    state.observations += 1
                    state.confidence = confidence
                else:
                    state = TrackState(
                        vehicle_id=_next_vehicle_id(lane_id, lane_counters),
                        home_lane_id=lane_id,
                        bbox_xyxy=xyxy,
                        center=center,
                        velocity=(0.0, 0.0),
                        last_seen_frame=frame_number,
                        observations=1,
                        confidence=confidence,
                    )
                    memory[track_id] = state
            else:
                state.velocity = (center[0] - state.center[0], center[1] - state.center[1])
                state.vertical_motion.append(state.velocity[1])
                state.bbox_xyxy = xyxy
                state.center = center
                state.last_seen_frame = frame_number
                state.observations += 1
                state.confidence = confidence
            if state.home_lane_id is None and lane_id:
                state.home_lane_id = lane_id
            label_overrides[track_id] = state.vehicle_id
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
            x1, y1, x2, y2 = _predicted_box(state, missed_frames)
            predicted = [
                max(0.0, min(frame_width - 1.0, x1)),
                max(0.0, min(frame_height - 1.0, y1)),
                max(0.0, min(frame_width - 1.0, x2)),
                max(0.0, min(frame_height - 1.0, y2)),
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
                            "vehicle_id": state.vehicle_id,
                            "wrong_way_event_id": state.wrong_way_event_id,
                            "label": state.vehicle_id,
                            "bbox_xyxy": predicted,
                            "bottom_center": bottom_center,
                        },
                    )
                )
        # Let ByteTrack keep its own longer-lived state, but do not keep stale
        # display data forever after an object has left the scene.
        if missed_frames > max(hold_frames, 90):
            del memory[track_id]
    retained_candidates.sort(key=lambda item: item[:3], reverse=True)
    retained_tracks = [item[3] for item in retained_candidates[:max_held_tracks]]
    return label_overrides, retained_tracks, track_vehicle_ids(memory)

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
    return tuple(sorted(intersections))

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
    camera_timestamp: datetime | None = None,
    wrong_way_gates: dict[str, tuple[tuple[int, int], tuple[int, int]]] | None = None,
) -> None:
    """Draw camera-112 lane regions and timetable directions only."""
    h_frame, w_frame = frame.shape[:2]
    cv2.putText(frame, "<- Bang Phlat ->", (round(w_frame * 0.39), 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "<- Samsen ->", (round(w_frame * 0.41), h_frame - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

    label_y = round(h_frame * 0.644)
    arrow_y = round(h_frame * 0.735)
    white = (255, 255, 255)
    for lane_name, polygon in (lane_points or {}).items():
        direction = str((lane_signals or {}).get(lane_name, {}).get("direction", "unknown"))
        cv2.polylines(frame, [polygon], True, white, 2, cv2.LINE_AA)
        span = horizontal_polygon_span(polygon, label_y)
        if span:
            label = f"{lane_name.replace('_', ' ')} {direction}"
            (label_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(frame, label, (max(2, round((span[0] + span[1] - label_width) / 2)), label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, white, 1, cv2.LINE_AA)
        arrow_span = horizontal_polygon_span(polygon, arrow_y)
        if arrow_span and direction in {"up", "down"}:
            arrow_x = round((arrow_span[0] + arrow_span[1]) / 2)
            half_length = max(14, round(h_frame * 0.0475))
            start, end = ((arrow_x, arrow_y + half_length), (arrow_x, arrow_y - half_length)) if direction == "up" else ((arrow_x, arrow_y - half_length), (arrow_x, arrow_y + half_length))
            cv2.arrowedLine(frame, start, end, white, 2, cv2.LINE_AA, tipLength=0.32)

    gate_colors = ((0, 220, 220),)
    for index, (gate_id, (gate_start, gate_end)) in enumerate((wrong_way_gates or {}).items()):
        gate_color = gate_colors[index % len(gate_colors)]
        cv2.line(frame, gate_start, gate_end, gate_color, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"WRONG-WAY {gate_id}",
            (gate_start[0] + 4, min(h_frame - 8, gate_start[1] + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            gate_color,
            1,
            cv2.LINE_AA,
        )

    if lane_signals:
        timestamp = camera_timestamp or datetime.now().astimezone()
        lines: list[tuple[str, tuple[int, int, int]]] = [(f"TIME: {timestamp:%Y-%m-%d %H:%M:%S}", white)]
        for index in range(1, 5):
            summary = lane_signals.get(f"lane_{index}", {})
            direction = str(summary.get("direction", "unknown"))
            period = str(summary.get("schedule_period", "default"))
            color = {"up": (0, 220, 0), "down": (0, 0, 255)}.get(direction, (0, 220, 220))
            lines.append((f"L{index} {direction.upper()} | 112 schedule ({period})", color))
        draw_text_panel(frame, lines, "top_right", font_scale=0.36, padding=6, margin=8)

    if not show_gates:
        return
    for label, (_x_normalized, y_normalized), color in (("GATE A", GATE_A_NORMALIZED, (255, 0, 0)), ("GATE B", GATE_B_NORMALIZED, (0, 0, 255))):
        left, right = gate_endpoints(roi_points, y_normalized, h_frame)
        cv2.line(frame, left, right, color, 4, cv2.LINE_AA)
        cv2.putText(frame, label, (left[0] + 8, max(26, left[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

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
    vehicle_ids: dict[int, str] | None = None,
    wrong_way_ids: set[int] | None = None,
    wrong_way_event_ids: dict[int, str] | None = None,
) -> dict[str, list[dict]]:
    """Write a single ``vehicle`` class while retaining lane-local IDs."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    boxes = result.boxes
    direction_by_track = direction_by_track or {}
    allowed_by_lane = allowed_by_lane or {}
    vehicle_ids = vehicle_ids or {}
    wrong_way_ids = wrong_way_ids or set()
    wrong_way_event_ids = wrong_way_event_ids or {}

    def movement_fields(bottom_center: list[float], track_id: int | None) -> dict[str, object]:
        lane_id = point_lane((bottom_center[0], bottom_center[1]), lane_points) if lane_points else None
        direction = direction_by_track.get(track_id) if track_id is not None else None
        expected = allowed_by_lane.get(lane_id, "unknown") if lane_id else "unknown"
        wrong_way = track_id in wrong_way_ids if track_id is not None else False
        return {
            "lane_id": lane_id,
            "direction": direction or "unknown",
            "expected_direction": expected,
            "wrong_way": wrong_way,
            "wrong_way_event_id": wrong_way_event_ids.get(track_id) if wrong_way and track_id is not None else None,
        }

    def add_vehicle(
        xyxy: list[float],
        track_id: int | None,
        confidence: float | None,
        occluded_prediction: bool,
        retained_vehicle_id: object | None = None,
        retained_event_id: object | None = None,
    ) -> None:
        rounded_box = [round(float(value), 2) for value in xyxy]
        bottom_center = [
            round((rounded_box[0] + rounded_box[2]) / 2, 2),
            round(rounded_box[3], 2),
        ]
        vehicle_id = str(retained_vehicle_id or vehicle_ids.get(track_id) or "vehicle-untracked")
        fields = movement_fields(bottom_center, track_id)
        if fields["wrong_way"] and retained_event_id:
            fields["wrong_way_event_id"] = str(retained_event_id)
        detection: dict[str, object] = {
            "class": "vehicle",
            "vehicle_id": vehicle_id,
            "confidence": round(float(confidence), 4) if confidence is not None else None,
            "bbox_xyxy": rounded_box,
            "bottom_center": bottom_center,
            "occluded_prediction": occluded_prediction,
            **fields,
        }
        if include_track_id:
            # ``track_id`` is stable for the V2 consumer; the ByteTrack number
            # remains available only as diagnostic information.
            detection["track_id"] = vehicle_id
            detection["byte_track_id"] = track_id
        grouped["vehicle"].append(detection)

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
            add_vehicle(
                [float(value) for value in box.xyxy[0].cpu().tolist()],
                track_id,
                float(box.conf.item()),
                False,
            )
    for track in retained_tracks or []:
        xyxy = [float(value) for value in track["bbox_xyxy"]]
        bottom_center = ((xyxy[0] + xyxy[2]) / 2, xyxy[3])
        if (
            require_lane_membership
            and lane_points
            and point_lane(bottom_center, lane_points) is None
        ):
            continue
        add_vehicle(
            xyxy,
            int(track["track_id"]),
            None,
            True,
            retained_vehicle_id=track.get("vehicle_id"),
            retained_event_id=track.get("wrong_way_event_id"),
        )
    return grouped

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
    camera_timestamp: datetime | None = None,
    wrong_way_gates: dict[str, tuple[tuple[int, int], tuple[int, int]]] | None = None,
) -> object:
    """Render boxes and optionally highlight tracked wrong-way vehicles."""
    frame = (source_frame if source_frame is not None else result.orig_img).copy()
    if roi_points is not None:
        draw_bridge_guides(
            frame,
            roi_points,
            show_gates,
            lane_points,
            lane_signals,
            camera_timestamp=camera_timestamp,
            wrong_way_gates=wrong_way_gates,
        )
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
            output_name = label_overrides.get(track_id, "vehicle")
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

def append_jsonl_record(handle, record: dict[str, object]) -> None:
    """Append one complete JSONL record so logs remain readable while running."""
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    # Flush every frame: pressing q or an unexpected interruption still leaves
    # all earlier records visible in the JSONL file immediately.
    handle.flush()

def _configured_timestamp(value: object) -> datetime | None:
    """Allow None or the readable timestamp text used in settings.py."""
    if value is None or isinstance(value, datetime):
        return value
    return parse_clock_timestamp(str(value))

def send_udp_payload(sock: socket.socket | None, payload: dict[str, Any], host: str, port: int) -> None:
    if sock is None:
        return
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    summary = payload.get("traffic", {})
    wrong_way = payload.get("wrong_way", {})
    print(
        f"\n[UDP LIVE OUT] {payload.get('location', {}).get('site_id', 'site')} "
        f"{payload.get('window', {}).get('start')} -> {payload.get('window', {}).get('end')} "
        f"| vehicles={summary.get('unique_vehicle_count', 0)} wrong_way={wrong_way.get('count', 0)} bytes={len(encoded)}",
        flush=True,
    )
    try:
        sock.sendto(encoded, (host, port))
    except Exception as err:
        print(f"[UDP ERROR] Could not send payload to {host}:{port}: {err}", flush=True)

def send_mqtt_payload(client: Any, payload: dict[str, Any], topic: str, qos: int = 1) -> None:
    if client is None:
        return
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    summary = payload.get("traffic", {})
    wrong_way = payload.get("wrong_way", {})
    try:
        client.publish(topic, encoded, qos=qos, retain=False)
    except Exception as err:
        print(f"[MQTT ERROR] Could not publish payload to topic {topic}: {err}", flush=True)

def _is_url(val: object) -> bool:
    s = str(val).strip().lower()
    return s.startswith(("http://", "https://", "rtsp://", "rtmp://"))

def live_recording_fps(source_fps: object, fallback_fps: object = 25.0) -> float:
    """Pick the supported 25/30 FPS rate closest to the live camera rate."""
    try:
        candidate = float(source_fps)
    except (TypeError, ValueError):
        candidate = 0.0
    if candidate <= 0.0 or candidate > 120.0:
        candidate = float(fallback_fps)
    return 30.0 if candidate >= 27.5 else 25.0

BANGKOK_TIMEZONE = ZoneInfo("Asia/Bangkok")

def schedule_lane_signals_from_112(
    timestamp: datetime | None, timestamp_source: str
) -> dict[str, dict[str, object]]:
    """Set all four lane directions from the Krung Thon timetable and camera 112 time."""
    directions, period = timetable_directions(timestamp)
    result: dict[str, dict[str, object]] = {}
    for index in range(1, 5):
        lane = f"lane_{index}"
        direction = directions.get(lane, "unknown") if directions else "unknown"
        result[lane] = {
            "direction": direction,
            "enforcement_direction": direction,
            "source": "schedule_112",
            "schedule_direction": direction,
            "schedule_period": period or "unknown",
            "schedule_timestamp": timestamp.isoformat(sep=" ") if timestamp else None,
            "schedule_timestamp_source": timestamp_source,
            "schedule_confidence": 1.0 if timestamp else 0.0,
        }
    return result

def run_v2_single_camera_from_settings() -> None:
    """Run v2: camera 112 detects vehicles and its timetable sets lane direction."""
    import settings

    source_value = getattr(settings, "CAMERA_112_SOURCE", "")
    source: str | Path = str(source_value) if _is_url(source_value) else Path(source_value).expanduser()
    model_path = Path(settings.MODEL_PATH).expanduser()
    tracker_path = Path(settings.TRACKER_PATH).expanduser()
    if not model_path.exists():
        raise SystemExit(f"YOLO model not found: {model_path}\nEdit tracking/v2/settings.py and try again.")
    if not tracker_path.exists():
        raise SystemExit(f"ByteTrack configuration not found: {tracker_path}\nEdit tracking/v2/settings.py and try again.")
    if not _is_url(source) and isinstance(source, Path) and not source.exists():
        raise SystemExit(f"Camera 112 source not found: {source}\nEdit tracking/v2/settings.py and try again.")

    profile = str(settings.PROFILE)
    if profile not in {PROFILE_TAKSIN, PROFILE_KRUNG_THON}:
        raise SystemExit("settings.PROFILE must be 'taksin' or 'krung_thon_bridge'.")

    model = YOLO(str(model_path))
    class_map = vehicle_class_map(model)
    if not class_map:
        raise SystemExit("The selected model has no supported vehicle classes.")
    class_aliases: dict[str, str] = {}
    if profile == PROFILE_KRUNG_THON:
        class_aliases["motorcycle"] = "moto"
    apply_class_aliases(model, class_aliases)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera 112 source: {source}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    fps = source_fps or 25.0
    if fps <= 0 or fps > 120:
        fps = 25.0
    source_is_live = _is_url(source)
    record_live_mp4 = source_is_live and bool(getattr(settings, "RECORD_LIVE_MP4", True))
    recording_fps = live_recording_fps(
        source_fps, getattr(settings, "LIVE_RECORDING_FPS_FALLBACK", 25.0)
    )
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    configured_time = _configured_timestamp(getattr(settings, "TIMESTAMP_112", None))
    if _is_url(source):
        clock_112 = datetime.now(BANGKOK_TIMEZONE).replace(tzinfo=None)
        clock_source_112 = "system_clock_live"
    else:
        clock_112, clock_source_112 = resolve_start_timestamp(
            source, "112", configured_time, Path(settings.LOG_DIRECTORY)
        )
    if clock_112 is None and bool(
        getattr(settings, "USE_SYSTEM_CLOCK_IF_112_TIME_UNAVAILABLE", True)
    ):
        clock_112 = datetime.now(BANGKOK_TIMEZONE).replace(tzinfo=None)
        clock_source_112 = "system_clock_fallback"

    log_root = Path(settings.LOG_DIRECTORY).expanduser()
    session_started = datetime.now(BANGKOK_TIMEZONE)
    session_stamp = session_started.strftime("%Y%m%d_%H%M%S")
    session_dir = log_root / f"{'live' if source_is_live else 'v2'}_{session_stamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    stem_112 = "live_stream_112" if source_is_live else source.stem
    log_112 = session_dir / f"{session_stamp}_{stem_112}_camera112.jsonl"
    recording_path = session_dir / f"{session_stamp}_{stem_112}_camera112_{int(recording_fps)}fps.mp4" if record_live_mp4 else None

    enable_mqtt_gateway = bool(getattr(settings, "ENABLE_MQTT_GATEWAY", True))
    gateway_window_seconds = float(getattr(settings, "GATEWAY_WINDOW_SECONDS", 60.0))
    use_wall_clock = bool(getattr(settings, "USE_WALL_CLOCK_TIME", _is_url(source)))

    mqtt_client = None
    if enable_mqtt_gateway:
        try:
            import paho.mqtt.client as mqtt

            try:
                mqtt_client = mqtt.Client(
                    client_id=TRACKER_CLIENT_ID,
                    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                )
            except AttributeError:
                mqtt_client = mqtt.Client(client_id=TRACKER_CLIENT_ID)
            mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            mqtt_client.loop_start()
        except Exception as err:
            print(f"[MQTT WARNING] Tracker could not connect to MQTT Broker: {err}", flush=True)
            mqtt_client = None
    else:
        print("[MQTT] Gateway publishing is disabled in tracking/v2/settings.py.", flush=True)

    gateway_input_aggregator = (
        TrafficWindowAggregator(
            window_seconds=gateway_window_seconds,
            anchor_time=clock_112 or datetime.now(BANGKOK_TIMEZONE),
            use_wall_clock=use_wall_clock,
        )
        if enable_mqtt_gateway
        else None
    )

    def forward_gateway_payload(gateway_payload: dict[str, Any]) -> None:
        """Publish one local one-minute payload to the separate MQTT Gateway."""
        window = gateway_payload.get("window", {})
        print(
            f"[MQTT GATEWAY OUT] topic={GATEWAY_INPUT_TOPIC} "
            f"{window.get('start')} -> {window.get('end')} "
            f"| vehicles={gateway_payload.get('traffic', {}).get('vehicle_count', 0)} "
            f"wrong_way={gateway_payload.get('wrong_way', {}).get('count', 0)}",
            flush=True,
        )
        send_mqtt_payload(mqtt_client, gateway_payload, GATEWAY_INPUT_TOPIC, MQTT_QOS)

    track_kwargs: dict[str, Any] = {
        "source": str(source),
        "stream": True,
        "persist": True,
        "tracker": str(tracker_path),
        "classes": list(class_map),
        "conf": float(settings.CONFIDENCE),
        "iou": float(settings.IOU),
        "imgsz": int(settings.IMAGE_SIZE),
        "device": str(getattr(settings, "DEVICE", "mps")),
        "agnostic_nms": bool(settings.AGNOSTIC_NMS),
        "save": False,
        "verbose": False,
    }
    if _is_url(source):
        track_kwargs["stream_buffer"] = True

    results = model.track(**track_kwargs)
    track_memory: dict[int, TrackState] = {}
    lane_counters: dict[str, int] = defaultdict(int)
    reassociate_gap_frames = max(
        1,
        round(fps * float(getattr(settings, "VEHICLE_ID_REASSOCIATE_SECONDS", 2.0))),
    )
    window_112 = "Camera 112 - Vehicle tracking (v2 schedule only)"
    cv2.namedWindow(window_112, cv2.WINDOW_NORMAL)
    video_writer: cv2.VideoWriter | None = None

    print("V2 started: camera 112 only. Lane direction comes from the timetable.")
    print("Press q in the video window to stop. JSONL is saved after every frame.")
    print(f"JSONL log: {log_112}")
    if record_live_mp4:
        print(f"MP4 recording: {recording_path} ({int(recording_fps)} FPS)")
    else:
        print("MP4 recording is disabled by settings or the selected source is not live_stream.")
    try:
        with log_112.open("w", encoding="utf-8") as handle_112:
            for frame_number, result in enumerate(results):
                elapsed_seconds = frame_number / fps
                if _is_url(source):
                    lane_timestamp = datetime.now(BANGKOK_TIMEZONE).replace(tzinfo=None)
                else:
                    lane_timestamp = timestamp_at(clock_112, elapsed_seconds)
                lane_signals = schedule_lane_signals_from_112(
                    lane_timestamp, clock_source_112
                )
                allowed_by_lane = {
                    lane: str(summary["enforcement_direction"])
                    for lane, summary in lane_signals.items()
                }
                roi_points = bridge_roi(result.orig_img, profile) if bool(settings.BRIDGE_ONLY) else None
                lane_filter_points = lane_rois(result.orig_img, profile)
                lane_points = lane_filter_points if bool(settings.SHOW_LANES) else None
                wrong_way_gates = scaled_wrong_way_gates(
                    result.orig_img,
                    getattr(settings, "WRONG_WAY_GATES_REFERENCE", {}),
                )
                gate_margin_pixels = float(
                    getattr(settings, "WRONG_WAY_GATE_MARGIN_PIXELS", 20.0)
                ) * result.orig_img.shape[0] / REFERENCE_HEIGHT

                label_overrides, retained_tracks, vehicle_ids = stabilized_tracks(
                    result,
                    class_map,
                    class_aliases,
                    roi_points,
                    lane_filter_points,
                    track_memory,
                    frame_number,
                    int(settings.OCCLUSION_HOLD),
                    int(settings.MAX_HELD_TRACKS),
                    lane_counters,
                    reassociate_gap_frames,
                    float(getattr(settings, "VEHICLE_ID_MAX_CENTER_DISTANCE", 70.0)),
                    float(getattr(settings, "VEHICLE_ID_MIN_IOU", 0.05)),
                )
                update_wrong_way_gate_crossings(
                    track_memory,
                    wrong_way_gates,
                    frame_number,
                    gate_margin_pixels,
                )
                direction_by_track = track_directions(track_memory)
                violations = confirmed_wrong_way_track_ids(
                    track_memory,
                    direction_by_track,
                    lane_filter_points,
                    allowed_by_lane,
                    frame_number,
                    int(getattr(settings, "WRONG_WAY_CONFIRM_FRAMES", 12)),
                    float(getattr(settings, "WRONG_WAY_MIN_DISPLACEMENT_PIXELS", 24.0)),
                    wrong_way_gates,
                    int(getattr(settings, "WRONG_WAY_GATE_CONFIRM_FRAMES", 12)),
                )
                wrong_way_event_ids_by_track = {
                    track_id: state.wrong_way_event_id
                    for track_id, state in track_memory.items()
                    if track_id in violations and state.wrong_way_event_id
                }
                wrong_way_vehicle_ids = sorted(
                    {vehicle_ids.get(track_id, f"byte-{track_id}") for track_id in violations}
                )
                vehicle_view = annotated_frame(
                    result,
                    class_map,
                    class_aliases,
                    roi_points,
                    bool(settings.SHOW_GATES),
                    label_overrides,
                    retained_tracks,
                    lane_points,
                    lane_signals,
                    lane_filter_points=lane_filter_points,
                    alert_wrong_way=bool(settings.WRONG_WAY_ALERTS),
                    wrong_way_ids=violations,
                    camera_timestamp=lane_timestamp,
                    wrong_way_gates=(
                        wrong_way_gates
                        if bool(getattr(settings, "SHOW_WRONG_WAY_GATES", True))
                        else None
                    ),
                )
                cv2.imshow(window_112, vehicle_view)

                if record_live_mp4:
                    if video_writer is None:
                        height, width = vehicle_view.shape[:2]
                        video_writer = cv2.VideoWriter(
                            str(recording_path), cv2.VideoWriter_fourcc(*"mp4v"), recording_fps, (width, height)
                        )
                        if not video_writer.isOpened():
                            video_writer.release()
                            video_writer = None
                            raise SystemExit(f"Could not create MP4 recording: {recording_path}")
                    video_writer.write(vehicle_view)

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
                    vehicle_ids=vehicle_ids,
                    wrong_way_ids=violations,
                    wrong_way_event_ids=wrong_way_event_ids_by_track,
                )
                record_112 = {
                    "frame": frame_number,
                    "time_seconds": round(elapsed_seconds, 3),
                    "recording": {
                        "enabled": record_live_mp4,
                        "mp4_file": recording_path.name if recording_path else None,
                        "fps": int(recording_fps) if record_live_mp4 else None,
                    },
                    "mode": "v2_camera112_schedule_only",
                    "camera_profile": profile,
                    "camera_timestamp": (
                        lane_timestamp.isoformat(sep=" ") if lane_timestamp else None
                    ),
                    "camera_timestamp_source": clock_source_112,
                    "timetable_by_lane": {
                        lane: {
                            "direction": summary["direction"],
                            "period": summary["schedule_period"],
                            "timestamp_source": summary["schedule_timestamp_source"],
                        }
                        for lane, summary in lane_signals.items()
                    },
                    "lane_signal_fusion": lane_signals,
                    "lane_directions": {
                        lane: str(summary["direction"])
                        for lane, summary in lane_signals.items()
                    },
                    "lane_enforcement_directions": allowed_by_lane,
                    "wrong_way_tracker_ids": sorted(violations),
                    "wrong_way_vehicle_ids": wrong_way_vehicle_ids,
                    "wrong_way_event_ids": sorted(set(wrong_way_event_ids_by_track.values())),
                    "tracks_by_class": grouped,
                }
                append_jsonl_record(handle_112, record_112)

                if gateway_input_aggregator:
                    for gateway_payload in gateway_input_aggregator.add_frame(record_112):
                        forward_gateway_payload(gateway_payload)

                if frame_number % 100 == 0:
                    progress = f"{frame_number + 1}/{total_frames}" if total_frames > 0 else str(frame_number + 1)
                    print(f"Displayed {progress} frames", flush=True)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    print("Stopped by q. Earlier MP4 frames and JSONL records have already been saved.")
                    break
    finally:
        if gateway_input_aggregator:
            final_gateway_payload = gateway_input_aggregator.flush(complete_window=False)
            if final_gateway_payload:
                forward_gateway_payload(final_gateway_payload)
        if mqtt_client:
            try:
                mqtt_client.loop_stop()
                mqtt_client.disconnect()
            except Exception:
                pass
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

    print(f"Camera 112 JSONL: {log_112}")
    if recording_path:
        print(f"Camera 112 MP4: {recording_path}")

def main() -> None:
    if len(sys.argv) > 1:
        raise SystemExit("V2 selects video_files or live_stream in tracking/v2/settings.py. Do not pass command-line sources.")
    run_v2_single_camera_from_settings()


if __name__ == "__main__":
    main()
