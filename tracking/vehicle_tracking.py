"""YOLO vehicle detection/tracking for the bridge camera.

COCO classes used by this project:
    2 car, 3 motorcycle, 5 bus, 7 truck

Examples:
    python3 vehicle_tracking.py
    python3 vehicle_tracking.py --source locations/taksin/image/image.png
    python3 vehicle_tracking.py --source locations/taksin/video/taksin_bridge_sathorn_1min.mp4

Traffic-light rules for Krung Thon: use camera 147/156 LEDs first, with a
colour-confidence score and weighted temporal smoothing to reject flicker.
The timetable is a fallback: each of cameras 112, 147, and 156 votes from
its own timestamp and elapsed video time. A clear visual/schedule conflict is
kept visible but disables wrong-way alerts for safety; weak visual evidence
uses the timetable instead.

Daily timetable: 05:30-07:30 and 08:20-08:45 = up/up/up/down;
15:30-16:30, 17:00-17:45, and 18:15-20:30 = up/down/down/down;
all other times = up/up/down/down. Three matching clocks score 0.92, two
score 0.85, and one scores 0.75.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import subprocess
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from ultralytics import YOLO


# ``tracking`` is intentionally runnable as its own working folder.  Keep the
# project root importable so its existing region definitions remain available.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
payload_dir = PROJECT_ROOT / "payload"
if str(payload_dir) not in sys.path:
    sys.path.insert(0, str(payload_dir))

# pyrefly: ignore [missing-import]
from traffic_payload import TrafficWindowAggregator

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

# These periods repeat every day. The values are the expected camera-112 lane
# directions and intentionally live next to the fusion rules documented in
# this file so an operator can audit or change them together.
KRUNG_THON_TIMETABLE = (
    ("morning", ((time(5, 30), time(7, 30)), (time(8, 20), time(8, 45))), ("up", "up", "up", "down")),
    ("evening", ((time(15, 30), time(16, 30)), (time(17, 0), time(17, 45)), (time(18, 15), time(20, 30))), ("up", "down", "down", "down")),
)
KRUNG_THON_DEFAULT_DIRECTIONS = ("up", "up", "down", "down")
KNOWN_SIGNAL_STATES = {"green", "red"}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and track bridge vehicles with YOLO + ByteTrack")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Input image or video. Omit it to start the live three-camera view from settings.py",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vehicles"))
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "model/coco/yolo11m.pt"),
        help="Ultralytics model or local .pt path",
    )
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
    parser.add_argument(
        "--timestamp-112",
        type=parse_clock_timestamp,
        default=None,
        help="Camera 112 first-frame timestamp; otherwise read its on-screen clock",
    )
    parser.add_argument(
        "--timestamp-147",
        type=parse_clock_timestamp,
        default=None,
        help="Camera 147 first-frame timestamp; otherwise read its on-screen clock",
    )
    parser.add_argument(
        "--timestamp-156",
        type=parse_clock_timestamp,
        default=None,
        help="Camera 156 first-frame timestamp; otherwise read its on-screen clock",
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


def timetable_directions(timestamp: datetime | None) -> tuple[dict[str, str] | None, str | None]:
    """Return the daily timetable direction map at one camera timestamp."""
    if timestamp is None:
        return None, None
    now = timestamp.time()
    for period, windows, directions in KRUNG_THON_TIMETABLE:
        if any(start <= now < end for start, end in windows):
            return {f"lane_{index}": direction for index, direction in enumerate(directions, start=1)}, period
    return {f"lane_{index}": direction for index, direction in enumerate(KRUNG_THON_DEFAULT_DIRECTIONS, start=1)}, "default"


def fuse_schedule_clocks(camera_times: dict[str, datetime | None]) -> dict[str, dict[str, object]]:
    """Vote timetable directions using the independently stamped three cameras."""
    camera_votes: dict[str, dict[str, str]] = {}
    periods: dict[str, str] = {}
    for camera, timestamp in camera_times.items():
        directions, period = timetable_directions(timestamp)
        if directions is not None and period is not None:
            camera_votes[camera] = directions
            periods[camera] = period

    fused: dict[str, dict[str, object]] = {}
    confidence_by_count = {1: 0.75, 2: 0.85, 3: 0.92}
    for index in range(1, 5):
        lane = f"lane_{index}"
        votes = {camera: directions[lane] for camera, directions in camera_votes.items()}
        if not votes:
            fused[lane] = {
                "direction": "unknown",
                "confidence": 0.0,
                "sources": [],
                "periods": {},
                "timestamps": {},
            }
            continue
        counts = Counter(votes.values())
        direction, count = counts.most_common(1)[0]
        agreeing_sources = [camera for camera, vote in votes.items() if vote == direction]
        fused[lane] = {
            "direction": direction,
            "confidence": confidence_by_count[len(agreeing_sources)],
            "sources": agreeing_sources,
            "periods": {camera: periods[camera] for camera in agreeing_sources},
            "timestamps": {
                camera: camera_times[camera].isoformat(sep=" ")
                for camera in agreeing_sources
                if camera_times[camera] is not None
            },
        }
    return fused


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


def direction_from_signal(camera: str, signal_state: str) -> str:
    """Translate a signal LED colour to the travel direction in camera 112."""
    if signal_state not in KNOWN_SIGNAL_STATES:
        return "unknown"
    if camera == "156":
        return "down" if signal_state == "green" else "up"
    return "up" if signal_state == "green" else "down"


def allowed_direction(signal_state: str) -> str:
    """Backward-compatible camera 147 direction mapping."""
    return direction_from_signal("147", signal_state)


def _visual_signal_decision(
    direction_147: str, confidence_147: float, direction_156: str, confidence_156: float
) -> tuple[str, float, str, bool | None]:
    """Choose a visual lane direction only when its light evidence is clear."""
    known_147 = direction_147 in {"up", "down"}
    known_156 = direction_156 in {"up", "down"}
    if known_147 and known_156 and direction_147 == direction_156:
        confidence = min(0.99, 0.55 + 0.45 * ((confidence_147 + confidence_156) / 2))
        return direction_147, confidence, "visual_both", True
    if known_147 and known_156:
        if confidence_147 >= 0.80 and confidence_147 >= confidence_156 + 0.18:
            return direction_147, confidence_147, "visual_147", False
        if confidence_156 >= 0.80 and confidence_156 >= confidence_147 + 0.18:
            return direction_156, confidence_156, "visual_156", False
        return "unknown", max(confidence_147, confidence_156), "visual_conflict", False
    if known_147:
        return direction_147, confidence_147, "visual_147", None
    if known_156:
        return direction_156, confidence_156, "visual_156", None
    return "unknown", 0.0, "visual_unknown", None


def fuse_lane_signals(
    states_147: dict[str, str] | None,
    states_156: dict[str, str] | None,
    confidence_147: dict[str, float] | None = None,
    confidence_156: dict[str, float] | None = None,
    schedule_by_lane: dict[str, dict[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    """Fuse visual LEDs with a schedule fallback and a safe conflict state."""
    states_147 = states_147 or {}
    states_156 = states_156 or {}
    confidence_147 = confidence_147 or {}
    confidence_156 = confidence_156 or {}
    schedule_by_lane = schedule_by_lane or {}
    fused: dict[str, dict[str, object]] = {}
    for index in range(1, 5):
        lane = f"lane_{index}"
        light_147 = states_147.get(lane, "unknown")
        light_156 = states_156.get(lane, "unknown")
        light_confidence_147 = float(confidence_147.get(lane, 0.0))
        light_confidence_156 = float(confidence_156.get(lane, 0.0))
        direction_147 = direction_from_signal("147", light_147)
        direction_156 = direction_from_signal("156", light_156)
        visual_direction, visual_confidence, visual_source, agrees = _visual_signal_decision(
            direction_147, light_confidence_147, direction_156, light_confidence_156
        )
        schedule = schedule_by_lane.get(lane, {})
        schedule_direction = str(schedule.get("direction", "unknown"))
        schedule_confidence = float(schedule.get("confidence", 0.0))
        visual_reliable = (
            visual_source == "visual_both" and visual_confidence >= 0.72
        ) or (visual_source in {"visual_147", "visual_156"} and visual_confidence >= 0.78)
        if visual_reliable and schedule_direction in {"up", "down"} and visual_direction != schedule_direction:
            direction, enforcement_direction, source = visual_direction, "unknown", "schedule_conflict_safe"
        elif visual_reliable:
            direction, enforcement_direction, source = visual_direction, visual_direction, visual_source
        elif schedule_direction in {"up", "down"}:
            direction, enforcement_direction, source = schedule_direction, schedule_direction, "schedule_fallback"
        else:
            direction, enforcement_direction, source = "unknown", "unknown", "unknown"
        fused[lane] = {
            "light_147": light_147,
            "light_156": light_156,
            "light_confidence_147": round(light_confidence_147, 3),
            "light_confidence_156": round(light_confidence_156, 3),
            "direction_147": direction_147,
            "direction_156": direction_156,
            "visual_direction": visual_direction,
            "visual_confidence": round(visual_confidence, 3),
            "schedule_direction": schedule_direction,
            "schedule_confidence": round(schedule_confidence, 3),
            "schedule_sources": list(schedule.get("sources", [])),
            "schedule_periods": dict(schedule.get("periods", {})),
            "schedule_timestamps": dict(schedule.get("timestamps", {})),
            "direction": direction,
            "enforcement_direction": enforcement_direction,
            "source": source,
            "agrees": agrees,
        }
    return fused


def classify_signal_measurement(frame: np.ndarray, polygon: np.ndarray) -> dict[str, object]:
    """Classify one LED ROI and quantify how clearly its colour is visible."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    mask[:2] = 0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    green = (
        (hue >= 70) & (hue <= 130) & (saturation >= 40) & (value >= 30) & (mask > 0)
    ).astype(np.uint8)
    red = (
        ((hue <= 20) | (hue >= 150)) & (saturation >= 40) & (value >= 30) & (mask > 0)
    ).astype(np.uint8)
    area = max(1, int((mask > 0).sum()))
    green_area, red_area = int(green.sum()), int(red.sum())
    green_score, red_score = green_area / area, red_area / area

    state = "unknown"
    confidence = 0.0
    if green_area >= 20 and green_score >= 0.50 and green_score > red_score * 1.25:
        separation = min(1.0, (green_score - red_score) / max(green_score, 0.01))
        coverage = min(1.0, green_score / 0.75)
        state = "green"
        confidence = 0.45 + 0.55 * coverage * separation
    elif red_area >= 8 and red_score >= 0.03 and red_score > green_score * 1.10:
        separation = min(1.0, (red_score - green_score) / max(red_score, 0.01))
        coverage = min(1.0, red_score / 0.12)
        pixel_strength = min(1.0, red_area / 28)
        state = "red"
        confidence = 0.40 + 0.60 * coverage * pixel_strength * separation
    return {
        "state": state,
        "confidence": round(float(confidence), 3),
        "green_score": round(float(green_score), 3),
        "red_score": round(float(red_score), 3),
    }


def classify_signal_state(frame: np.ndarray, polygon: np.ndarray) -> str:
    """Backward-compatible state-only signal classifier."""
    return str(classify_signal_measurement(frame, polygon)["state"])


class SignalStateSmoother:
    """Use a weighted half-second history to suppress tiny-ROI colour flicker."""

    def __init__(self, window: int = 15):
        self.history: dict[str, deque[tuple[str, float]]] = defaultdict(lambda: deque(maxlen=window))

    def update(
        self, raw_states: dict[str, str], raw_confidence: dict[str, float] | None = None
    ) -> tuple[dict[str, str], dict[str, float]]:
        raw_confidence = raw_confidence or {}
        stable: dict[str, str] = {}
        confidence: dict[str, float] = {}
        for index in range(1, 5):
            lane = f"lane_{index}"
            state = raw_states.get(lane, "unknown")
            value = float(raw_confidence.get(lane, 0.0)) if state in KNOWN_SIGNAL_STATES else 0.0
            self.history[lane].append((state, value))
            known = [(name, weight) for name, weight in self.history[lane] if name in KNOWN_SIGNAL_STATES]
            if not known:
                stable[lane], confidence[lane] = "unknown", 0.0
                continue
            weights: dict[str, float] = defaultdict(float)
            for name, weight in known:
                weights[name] += weight
            best_state, best_weight = max(weights.items(), key=lambda item: item[1])
            total_weight = sum(weights.values())
            winning = [weight for name, weight in known if name == best_state]
            coverage = min(1.0, len(known) / 4)
            stable[lane] = best_state
            confidence[lane] = round((sum(winning) / len(winning)) * (best_weight / max(total_weight, 0.001)) * coverage, 3)
        return stable, confidence


class SignalVideoReader:
    """Read one signal-camera stream by elapsed camera-112 time."""

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

    def measurements_at(self, elapsed_seconds: float) -> dict[str, dict[str, dict[str, object]]]:
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
            return {f"lane_{index}": {"primary": {"state": "unknown", "confidence": 0.0}} for index in range(1, 5)}
        return classify_signal_roi_measurements(self.last_frame, self.camera)

    def states_at(self, elapsed_seconds: float) -> dict[str, dict[str, str]]:
        measurements = self.measurements_at(elapsed_seconds)
        return {lane: {name: str(value["state"]) for name, value in roi_set.items()} for lane, roi_set in measurements.items()}

    def close(self) -> None:
        self.capture.release()


def signal_rois_for_camera(frame: np.ndarray, camera: str) -> dict[str, dict[str, np.ndarray]]:
    if camera == "156":
        return camera_156_signal_rois(frame)
    return camera_147_signal_rois(frame)


def classify_signal_roi_measurements(
    frame: np.ndarray, camera: str
) -> dict[str, dict[str, dict[str, object]]]:
    """Read every primary/backup LED ROI with confidence values."""
    return {
        lane: {name: classify_signal_measurement(frame, polygon) for name, polygon in roi_set.items()}
        for lane, roi_set in signal_rois_for_camera(frame, camera).items()
    }


def classify_signal_roi_states(frame: np.ndarray, camera: str) -> dict[str, dict[str, str]]:
    """Backward-compatible state-only view of all LED ROIs."""
    measurements = classify_signal_roi_measurements(frame, camera)
    return {lane: {name: str(value["state"]) for name, value in roi_set.items()} for lane, roi_set in measurements.items()}


def resolve_signal_roi_measurements(
    measurements: dict[str, dict[str, dict[str, object]]]
) -> tuple[dict[str, str], dict[str, float], dict[str, str]]:
    """Use a backup ROI when it is clearer than a weak primary LED ROI."""
    resolved: dict[str, str] = {}
    confidence: dict[str, float] = {}
    source: dict[str, str] = {}
    for index in range(1, 5):
        lane = f"lane_{index}"
        values = measurements.get(lane, {})
        candidates = [
            (name, str(value.get("state", "unknown")), float(value.get("confidence", 0.0)))
            for name, value in values.items()
            if str(value.get("state", "unknown")) in KNOWN_SIGNAL_STATES
        ]
        if not candidates:
            resolved[lane], confidence[lane], source[lane] = "unknown", 0.0, "unknown"
            continue
        candidates.sort(key=lambda item: item[2], reverse=True)
        best_name, best_state, best_confidence = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        if second and second[1] != best_state and second[2] >= best_confidence - 0.12:
            resolved[lane], confidence[lane], source[lane] = "unknown", 0.0, "roi_conflict"
            continue
        same_colour = [item for item in candidates if item[1] == best_state]
        if len(same_colour) > 1:
            best_confidence = min(0.99, max(item[2] for item in same_colour) + 0.06)
            best_name = "+".join(item[0] for item in same_colour)
        resolved[lane] = best_state
        confidence[lane] = round(best_confidence, 3)
        source[lane] = best_name
    return resolved, confidence, source


def resolve_signal_roi_states(
    roi_states: dict[str, dict[str, str]]
) -> tuple[dict[str, str], dict[str, str]]:
    """Retain the older state-only resolver for callers outside video mode."""
    measurements = {
        lane: {name: {"state": state, "confidence": 1.0 if state in KNOWN_SIGNAL_STATES else 0.0} for name, state in values.items()}
        for lane, values in roi_states.items()
    }
    states, _confidence, sources = resolve_signal_roi_measurements(measurements)
    return states, sources


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
            visual_confidence = float(summary.get("visual_confidence", 0.0))
            schedule_direction = str(summary.get("schedule_direction", "unknown"))[0].upper()
            schedule_confidence = float(summary.get("schedule_confidence", 0.0))
            source = str(summary.get("source", "unknown"))
            source_label = {
                "visual_both": "visual",
                "visual_147": "147",
                "visual_156": "156",
                "schedule_fallback": "schedule",
                "schedule_conflict_safe": "safe conflict",
            }.get(source, "unknown")
            color = {"up": (0, 220, 0), "down": (0, 0, 255)}.get(direction, (0, 220, 220))
            # Two short rows per lane keep the panel entirely in the right-side
            # margin instead of spanning into the road and vehicle tracks.
            panel_lines.extend((
                (f"L{index} {direction.upper()} | {source_label}", color),
                (f"V:{visual_confidence:.2f} 147:{light_147} 156:{light_156} S:{schedule_direction} {schedule_confidence:.2f}", color),
            ))
        draw_text_panel(frame, panel_lines, "top_right", font_scale=0.36, padding=6, margin=8)
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
    signal_confidence: dict[str, float] | None = None,
    roi_measurements: dict[str, dict[str, dict[str, object]]] | None = None,
    selected_sources: dict[str, str] | None = None,
    clock_timestamp: datetime | None = None,
) -> None:
    """Draw light state and confidence for every primary/backup signal ROI."""
    rois = signal_rois_for_camera(frame, camera)
    signal_confidence = signal_confidence or {}
    roi_measurements = roi_measurements or {}
    selected_sources = selected_sources or {}
    heading = f"CAMERA {camera} LIGHTS"
    if clock_timestamp is not None:
        heading = f"{heading} | {clock_timestamp:%Y-%m-%d %H:%M:%S}"
    panel_lines: list[tuple[str, tuple[int, int, int]]] = [(heading, (255, 255, 255))]
    schedule_directions, _period = timetable_directions(clock_timestamp)
    for index in range(1, 5):
        lane = f"lane_{index}"
        light = signal_states.get(lane, "unknown")
        light_confidence = float(signal_confidence.get(lane, 0.0))
        direction = direction_from_signal(camera, light)
        color = {"green": (0, 220, 0), "red": (0, 0, 255)}.get(light, (0, 220, 220))
        labels: list[str] = []
        for roi_name, polygon in rois[lane].items():
            measurement = roi_measurements.get(lane, {}).get(roi_name, {})
            roi_light = str(measurement.get("state", "unknown"))
            roi_confidence = float(measurement.get("confidence", 0.0))
            roi_color = {"green": (0, 220, 0), "red": (0, 0, 255)}.get(roi_light, (0, 220, 220))
            cv2.polylines(frame, [polygon], True, roi_color, 2, cv2.LINE_AA)
            labels.append(f"{roi_name[0].upper()}={roi_light} {roi_confidence:.2f}")
        source = selected_sources.get(lane, "unknown")
        schedule = schedule_directions.get(lane, "-") if schedule_directions else "-"
        labels_text = " ".join(labels)
        panel_lines.append((
            f"112 L{index}: {labels_text} -> {light} {light_confidence:.2f} ({source}) -> {direction} | sch:{schedule}",
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


def append_jsonl_record(handle, record: dict[str, object]) -> None:
    """Append one complete JSONL record so logs remain readable while running."""
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    # Flush every frame: pressing q or an unexpected interruption still leaves
    # all earlier records visible in the JSONL file immediately.
    handle.flush()


def render_signal_view(
    source: Path, output_dir: Path, camera: str, start_timestamp: datetime | None = None
) -> tuple[Path, Path]:
    """Write an annotated signal-camera view with colour confidence records."""
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
                measurements = classify_signal_roi_measurements(frame, camera)
                raw_states, raw_confidence, selected_sources = resolve_signal_roi_measurements(measurements)
                states, confidence = smoother.update(raw_states, raw_confidence)
                clock_timestamp = timestamp_at(start_timestamp, frame_number / fps)
                annotated = frame.copy()
                draw_signal_guides(
                    annotated, camera, states, confidence, measurements, selected_sources, clock_timestamp
                )
                writer.write(annotated)
                roi_states = {
                    lane: {name: str(value["state"]) for name, value in roi_set.items()}
                    for lane, roi_set in measurements.items()
                }
                record = {
                    "frame": frame_number,
                    "time_seconds": round(frame_number / fps, 3),
                    "camera": camera,
                    "clock_timestamp": clock_timestamp.isoformat(sep=" ") if clock_timestamp else None,
                    "light_roi_measurements": measurements,
                    "light_roi_states": roi_states,
                    "light_states": states,
                    "light_confidence": confidence,
                    "selected_light_source": selected_sources,
                    "directions_for_camera_112": {
                        lane: direction_from_signal(camera, light)
                        for lane, light in states.items()
                    },
                }
                append_jsonl_record(handle, record)
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
    confidence_147: dict[str, float] = {}
    confidence_156: dict[str, float] = {}
    roi_measurements_147: dict[str, dict[str, dict[str, object]]] = {}
    roi_measurements_156: dict[str, dict[str, dict[str, object]]] = {}
    selected_147: dict[str, str] = {}
    selected_156: dict[str, str] = {}
    reader_147 = SignalVideoReader(signal147_source, "147", signal147_offset) if signal147_source else None
    reader_156 = SignalVideoReader(signal156_source, "156", signal156_offset) if signal156_source else None
    if reader_147:
        roi_measurements_147 = reader_147.measurements_at(0.0)
        raw_states_147, raw_confidence_147, selected_147 = resolve_signal_roi_measurements(roi_measurements_147)
        states_147, confidence_147 = raw_states_147, raw_confidence_147
        reader_147.close()
    if reader_156:
        roi_measurements_156 = reader_156.measurements_at(0.0)
        raw_states_156, raw_confidence_156, selected_156 = resolve_signal_roi_measurements(roi_measurements_156)
        states_156, confidence_156 = raw_states_156, raw_confidence_156
        reader_156.close()
    roi_states_147 = {lane: {name: str(value["state"]) for name, value in values.items()} for lane, values in roi_measurements_147.items()}
    roi_states_156 = {lane: {name: str(value["state"]) for name, value in values.items()} for lane, values in roi_measurements_156.items()}
    lane_signals = fuse_lane_signals(states_147, states_156, confidence_147, confidence_156)
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
        "signal_147_roi_measurements": roi_measurements_147,
        "signal_147_roi_states": roi_states_147,
        "signal_147_states": states_147,
        "signal_147_confidence": confidence_147,
        "signal_147_selected_light_source": selected_147,
        "signal_156_roi_measurements": roi_measurements_156,
        "signal_156_roi_states": roi_states_156,
        "signal_156_states": states_156,
        "signal_156_confidence": confidence_156,
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
    timestamp_112: datetime | None,
    timestamp_147: datetime | None,
    timestamp_156: datetime | None,
    save_signal_views: bool,
    alert_wrong_way: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if profile == PROFILE_KRUNG_THON:
        clock_112, clock_source_112 = resolve_start_timestamp(source, "112", timestamp_112, output_dir)
        clock_147, clock_source_147 = resolve_start_timestamp(signal147_source, "147", timestamp_147, output_dir)
        clock_156, clock_source_156 = resolve_start_timestamp(signal156_source, "156", timestamp_156, output_dir)
    else:
        clock_112 = clock_147 = clock_156 = None
        clock_source_112 = clock_source_147 = clock_source_156 = "disabled"
    clock_start_sources = {"112": clock_source_112, "147": clock_source_147, "156": clock_source_156}
    clock_start_times = {
        "112": clock_112.isoformat(sep=" ") if clock_112 else None,
        "147": clock_147.isoformat(sep=" ") if clock_147 else None,
        "156": clock_156.isoformat(sep=" ") if clock_156 else None,
    }
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
                measurements_147 = reader_147.measurements_at(elapsed_seconds) if reader_147 else {}
                measurements_156 = reader_156.measurements_at(elapsed_seconds) if reader_156 else {}
                raw_147, raw_confidence_147, selected_147 = resolve_signal_roi_measurements(measurements_147) if measurements_147 else ({}, {}, {})
                raw_156, raw_confidence_156, selected_156 = resolve_signal_roi_measurements(measurements_156) if measurements_156 else ({}, {}, {})
                states_147, confidence_147 = smoother_147.update(raw_147, raw_confidence_147) if raw_147 else ({}, {})
                states_156, confidence_156 = smoother_156.update(raw_156, raw_confidence_156) if raw_156 else ({}, {})
                roi_states_147 = {lane: {name: str(value["state"]) for name, value in values.items()} for lane, values in measurements_147.items()}
                roi_states_156 = {lane: {name: str(value["state"]) for name, value in values.items()} for lane, values in measurements_156.items()}
                camera_times = {
                    "112": timestamp_at(clock_112, elapsed_seconds),
                    "147": timestamp_at(clock_147, elapsed_seconds + signal147_offset),
                    "156": timestamp_at(clock_156, elapsed_seconds + signal156_offset),
                }
                schedule_by_lane = fuse_schedule_clocks(camera_times)
                lane_signals = fuse_lane_signals(
                    states_147, states_156, confidence_147, confidence_156, schedule_by_lane
                )
                allowed_by_lane = {
                    lane: str(summary["enforcement_direction"]) for lane, summary in lane_signals.items()
                }
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
                    "clock_start_sources": clock_start_sources,
                    "clock_start_times": clock_start_times,
                    "camera_timestamps": {camera: value.isoformat(sep=" ") if value else None for camera, value in camera_times.items()},
                    "timetable_by_lane": schedule_by_lane,
                    "signal_147_roi_measurements": measurements_147,
                    "signal_147_roi_states": roi_states_147,
                    "signal_147_states": states_147,
                    "signal_147_confidence": confidence_147,
                    "signal_147_selected_light_source": selected_147,
                    "signal_156_roi_measurements": measurements_156,
                    "signal_156_roi_states": roi_states_156,
                    "signal_156_states": states_156,
                    "signal_156_confidence": confidence_156,
                    "signal_156_selected_light_source": selected_156,
                    "lane_signal_fusion": lane_signals,
                    "lane_directions": {lane: str(summary["direction"]) for lane, summary in lane_signals.items()},
                    "lane_enforcement_directions": allowed_by_lane,
                    "wrong_way_track_ids": sorted(violations),
                    "tracks_by_class": grouped,
                }
                append_jsonl_record(handle, record)
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
            signal_video, signal_jsonl = render_signal_view(signal147_source, output_dir, "147", clock_147)
            print(f"Camera 147 light view: {signal_video}")
            print(f"Camera 147 light records: {signal_jsonl}")
        if signal156_source:
            signal_video, signal_jsonl = render_signal_view(signal156_source, output_dir, "156", clock_156)
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
    timestamp_112: datetime | None,
    timestamp_147: datetime | None,
    timestamp_156: datetime | None,
    save_signal_views: bool,
) -> None:
    """Run raw YOLO on the full camera-112 scan ROI, then assign lane IDs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if profile == PROFILE_KRUNG_THON:
        clock_112, clock_source_112 = resolve_start_timestamp(source, "112", timestamp_112, output_dir)
        clock_147, clock_source_147 = resolve_start_timestamp(signal147_source, "147", timestamp_147, output_dir)
        clock_156, clock_source_156 = resolve_start_timestamp(signal156_source, "156", timestamp_156, output_dir)
    else:
        clock_112 = clock_147 = clock_156 = None
        clock_source_112 = clock_source_147 = clock_source_156 = "disabled"
    clock_start_sources = {"112": clock_source_112, "147": clock_source_147, "156": clock_source_156}
    clock_start_times = {
        "112": clock_112.isoformat(sep=" ") if clock_112 else None,
        "147": clock_147.isoformat(sep=" ") if clock_147 else None,
        "156": clock_156.isoformat(sep=" ") if clock_156 else None,
    }
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

                measurements_147 = reader_147.measurements_at(elapsed_seconds) if reader_147 else {}
                measurements_156 = reader_156.measurements_at(elapsed_seconds) if reader_156 else {}
                raw_147, raw_confidence_147, selected_147 = resolve_signal_roi_measurements(measurements_147) if measurements_147 else ({}, {}, {})
                raw_156, raw_confidence_156, selected_156 = resolve_signal_roi_measurements(measurements_156) if measurements_156 else ({}, {}, {})
                states_147, confidence_147 = smoother_147.update(raw_147, raw_confidence_147) if raw_147 else ({}, {})
                states_156, confidence_156 = smoother_156.update(raw_156, raw_confidence_156) if raw_156 else ({}, {})
                roi_states_147 = {lane: {name: str(value["state"]) for name, value in values.items()} for lane, values in measurements_147.items()}
                roi_states_156 = {lane: {name: str(value["state"]) for name, value in values.items()} for lane, values in measurements_156.items()}
                camera_times = {
                    "112": timestamp_at(clock_112, elapsed_seconds),
                    "147": timestamp_at(clock_147, elapsed_seconds + signal147_offset),
                    "156": timestamp_at(clock_156, elapsed_seconds + signal156_offset),
                }
                schedule_by_lane = fuse_schedule_clocks(camera_times)
                lane_signals = fuse_lane_signals(
                    states_147, states_156, confidence_147, confidence_156, schedule_by_lane
                )
                allowed_by_lane = {
                    lane: str(summary["enforcement_direction"]) for lane, summary in lane_signals.items()
                }

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
                    "clock_start_sources": clock_start_sources,
                    "clock_start_times": clock_start_times,
                    "camera_timestamps": {camera: value.isoformat(sep=" ") if value else None for camera, value in camera_times.items()},
                    "timetable_by_lane": schedule_by_lane,
                    "signal_147_roi_measurements": measurements_147,
                    "signal_147_roi_states": roi_states_147,
                    "signal_147_states": states_147,
                    "signal_147_confidence": confidence_147,
                    "signal_147_selected_light_source": selected_147,
                    "signal_156_roi_measurements": measurements_156,
                    "signal_156_roi_states": roi_states_156,
                    "signal_156_states": states_156,
                    "signal_156_confidence": confidence_156,
                    "signal_156_selected_light_source": selected_156,
                    "lane_signal_fusion": lane_signals,
                    "lane_directions": {lane: str(summary["direction"]) for lane, summary in lane_signals.items()},
                    "lane_enforcement_directions": allowed_by_lane,
                    "detections_by_class": detections,
                    "detections_by_lane": detections_by_lane,
                }
                append_jsonl_record(handle, record)
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
            signal_video, signal_jsonl = render_signal_view(signal147_source, output_dir, "147", clock_147)
            print(f"Camera 147 light view: {signal_video}")
            print(f"Camera 147 light records: {signal_jsonl}")
        if signal156_source:
            signal_video, signal_jsonl = render_signal_view(signal156_source, output_dir, "156", clock_156)
            print(f"Camera 156 light view: {signal_video}")
            print(f"Camera 156 light records: {signal_jsonl}")


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


def run_live_tracking_from_settings() -> None:
    """Show the three configured cameras live and persist JSONL as frames arrive."""
    import settings

    source = Path(settings.CAMERA_112_SOURCE).expanduser()
    signal147_source = Path(settings.CAMERA_147_SOURCE).expanduser()
    signal156_source = Path(settings.CAMERA_156_SOURCE).expanduser()
    model_path = Path(settings.MODEL_PATH).expanduser()
    required_files = {
        "Camera 112 video": source,
        "Camera 147 video": signal147_source,
        "Camera 156 video": signal156_source,
        "YOLO model": model_path,
        "ByteTrack configuration": Path(settings.TRACKER_PATH).expanduser(),
    }
    for label, path in required_files.items():
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}\nEdit tracking/settings.py and try again.")

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

    log_root = Path(settings.LOG_DIRECTORY).expanduser()
    session_dir = log_root / datetime.now().strftime("live_%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    log_112 = session_dir / f"{source.stem}_camera112.jsonl"
    log_147 = session_dir / f"{signal147_source.stem}_camera147.jsonl"
    log_156 = session_dir / f"{signal156_source.stem}_camera156.jsonl"

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera 112 video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    clock_112, clock_source_112 = resolve_start_timestamp(
        source, "112", _configured_timestamp(settings.TIMESTAMP_112), session_dir
    )
    clock_147, clock_source_147 = resolve_start_timestamp(
        signal147_source, "147", _configured_timestamp(settings.TIMESTAMP_147), session_dir
    )
    clock_156, clock_source_156 = resolve_start_timestamp(
        signal156_source, "156", _configured_timestamp(settings.TIMESTAMP_156), session_dir
    )
    clock_start_sources = {"112": clock_source_112, "147": clock_source_147, "156": clock_source_156}
    clock_start_times = {
        "112": clock_112.isoformat(sep=" ") if clock_112 else None,
        "147": clock_147.isoformat(sep=" ") if clock_147 else None,
        "156": clock_156.isoformat(sep=" ") if clock_156 else None,
    }

    enable_udp = bool(getattr(settings, "ENABLE_UDP_PAYLOAD", True))
    udp_host = str(getattr(settings, "UDP_HOST", "127.0.0.1"))
    udp_port = int(getattr(settings, "UDP_PORT", 5005))
    window_sec = float(getattr(settings, "WINDOW_SECONDS", 30.0))
    use_wall_clock = bool(getattr(settings, "USE_WALL_CLOCK_TIME", True))

    aggregator = (
        TrafficWindowAggregator(
            window_seconds=window_sec,
            anchor_time=clock_112 or datetime.now(timezone.utc),
            use_wall_clock=use_wall_clock,
        )
        if enable_udp
        else None
    )
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if enable_udp else None

    reader_147 = SignalVideoReader(
        signal147_source, "147", float(settings.SIGNAL_147_OFFSET_SECONDS)
    )
    reader_156 = SignalVideoReader(
        signal156_source, "156", float(settings.SIGNAL_156_OFFSET_SECONDS)
    )
    smoother_147 = SignalStateSmoother()
    smoother_156 = SignalStateSmoother()
    track_memory: dict[int, TrackState] = {}
    results = model.track(
        source=str(source),
        stream=True,
        persist=True,
        tracker=str(settings.TRACKER_PATH),
        classes=list(class_map),
        conf=float(settings.CONFIDENCE),
        iou=float(settings.IOU),
        imgsz=int(settings.IMAGE_SIZE),
        device=str(getattr(settings, "DEVICE", "mps")),
        agnostic_nms=bool(settings.AGNOSTIC_NMS),
        save=False,
        verbose=False,
    )

    window_112 = "Camera 112 - Vehicle tracking"
    window_147 = "Camera 147 - Traffic lights"
    window_156 = "Camera 156 - Traffic lights"
    for name in (window_112, window_147, window_156):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    print("Live view started. Press q in any video window to stop immediately.")
    print(f"JSONL logs are being written to: {session_dir}")
    if enable_udp:
        print(f"UDP Payloads are streaming live to udp://{udp_host}:{udp_port} every {window_sec}s")

    try:
        with (
            log_112.open("w", encoding="utf-8") as handle_112,
            log_147.open("w", encoding="utf-8") as handle_147,
            log_156.open("w", encoding="utf-8") as handle_156,
        ):
            for frame_number, result in enumerate(results):
                elapsed_seconds = frame_number / fps
                roi_points = bridge_roi(result.orig_img, profile) if bool(settings.BRIDGE_ONLY) else None
                lane_filter_points = lane_rois(result.orig_img, profile)
                lane_points = lane_filter_points if bool(settings.SHOW_LANES) else None

                measurements_147 = reader_147.measurements_at(elapsed_seconds)
                measurements_156 = reader_156.measurements_at(elapsed_seconds)
                raw_147, raw_confidence_147, selected_147 = resolve_signal_roi_measurements(measurements_147)
                raw_156, raw_confidence_156, selected_156 = resolve_signal_roi_measurements(measurements_156)
                states_147, confidence_147 = smoother_147.update(raw_147, raw_confidence_147)
                states_156, confidence_156 = smoother_156.update(raw_156, raw_confidence_156)
                roi_states_147 = {
                    lane: {name: str(value["state"]) for name, value in values.items()}
                    for lane, values in measurements_147.items()
                }
                roi_states_156 = {
                    lane: {name: str(value["state"]) for name, value in values.items()}
                    for lane, values in measurements_156.items()
                }
                camera_times = {
                    "112": timestamp_at(clock_112, elapsed_seconds),
                    "147": timestamp_at(
                        clock_147, elapsed_seconds + float(settings.SIGNAL_147_OFFSET_SECONDS)
                    ),
                    "156": timestamp_at(
                        clock_156, elapsed_seconds + float(settings.SIGNAL_156_OFFSET_SECONDS)
                    ),
                }
                schedule_by_lane = fuse_schedule_clocks(camera_times)
                lane_signals = fuse_lane_signals(
                    states_147, states_156, confidence_147, confidence_156, schedule_by_lane
                )
                allowed_by_lane = {
                    lane: str(summary["enforcement_direction"])
                    for lane, summary in lane_signals.items()
                }
                label_overrides, retained_tracks = stabilized_tracks(
                    result,
                    class_map,
                    class_aliases,
                    roi_points,
                    lane_filter_points,
                    track_memory,
                    frame_number,
                    int(settings.OCCLUSION_HOLD),
                    int(settings.MAX_HELD_TRACKS),
                )
                direction_by_track = track_directions(track_memory)
                violations = wrong_way_track_ids(
                    track_memory,
                    direction_by_track,
                    lane_filter_points,
                    allowed_by_lane,
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
                )
                camera147_view = reader_147.last_frame.copy()
                camera156_view = reader_156.last_frame.copy()
                draw_signal_guides(
                    camera147_view,
                    "147",
                    states_147,
                    confidence_147,
                    measurements_147,
                    selected_147,
                    camera_times["147"],
                )
                draw_signal_guides(
                    camera156_view,
                    "156",
                    states_156,
                    confidence_156,
                    measurements_156,
                    selected_156,
                    camera_times["156"],
                )
                cv2.imshow(window_112, vehicle_view)
                cv2.imshow(window_147, camera147_view)
                cv2.imshow(window_156, camera156_view)

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
                record_112 = {
                    "frame": frame_number,
                    "time_seconds": round(elapsed_seconds, 3),
                    "mode": "live_tracking",
                    "camera_profile": profile,
                    "clock_start_sources": clock_start_sources,
                    "clock_start_times": clock_start_times,
                    "camera_timestamps": {
                        camera: value.isoformat(sep=" ") if value else None
                        for camera, value in camera_times.items()
                    },
                    "timetable_by_lane": schedule_by_lane,
                    "signal_147_roi_measurements": measurements_147,
                    "signal_147_states": states_147,
                    "signal_147_confidence": confidence_147,
                    "signal_156_roi_measurements": measurements_156,
                    "signal_156_states": states_156,
                    "signal_156_confidence": confidence_156,
                    "lane_signal_fusion": lane_signals,
                    "lane_directions": {
                        lane: str(summary["direction"])
                        for lane, summary in lane_signals.items()
                    },
                    "lane_enforcement_directions": allowed_by_lane,
                    "wrong_way_track_ids": sorted(violations),
                    "tracks_by_class": grouped,
                }
                append_jsonl_record(handle_112, record_112)
                if aggregator and udp_sock:
                    for payload in aggregator.add_frame(record_112):
                        send_udp_payload(udp_sock, payload, udp_host, udp_port)

                append_jsonl_record(
                    handle_147,
                    {
                        "sampled_frame": max(reader_147.next_frame - 1, 0),
                        "time_seconds": round(elapsed_seconds, 3),
                        "camera": "147",
                        "clock_timestamp": (
                            camera_times["147"].isoformat(sep=" ") if camera_times["147"] else None
                        ),
                        "light_roi_measurements": measurements_147,
                        "light_roi_states": roi_states_147,
                        "light_states": states_147,
                        "light_confidence": confidence_147,
                        "selected_light_source": selected_147,
                        "directions_for_camera_112": {
                            lane: direction_from_signal("147", light)
                            for lane, light in states_147.items()
                        },
                    },
                )
                append_jsonl_record(
                    handle_156,
                    {
                        "sampled_frame": max(reader_156.next_frame - 1, 0),
                        "time_seconds": round(elapsed_seconds, 3),
                        "camera": "156",
                        "clock_timestamp": (
                            camera_times["156"].isoformat(sep=" ") if camera_times["156"] else None
                        ),
                        "light_roi_measurements": measurements_156,
                        "light_roi_states": roi_states_156,
                        "light_states": states_156,
                        "light_confidence": confidence_156,
                        "selected_light_source": selected_156,
                        "directions_for_camera_112": {
                            lane: direction_from_signal("156", light)
                            for lane, light in states_156.items()
                        },
                    },
                )

                if frame_number % 100 == 0:
                    print(f"Displayed {frame_number + 1}/{total_frames} frames", flush=True)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    print("Stopped by q. Earlier JSONL records have already been saved.")
                    break
    finally:
        if aggregator and udp_sock:
            final_payload = aggregator.flush()
            if final_payload:
                send_udp_payload(udp_sock, final_payload, udp_host, udp_port)
            udp_sock.close()
        reader_147.close()
        reader_156.close()
        cv2.destroyAllWindows()

    print(f"Camera 112 JSONL: {log_112}")
    print(f"Camera 147 JSONL: {log_147}")
    print(f"Camera 156 JSONL: {log_156}")

def main() -> None:
    args = parse_args()
    if args.source is None:
        run_live_tracking_from_settings()
        return

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
                args.timestamp_112,
                args.timestamp_147,
                args.timestamp_156,
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
                args.timestamp_112,
                args.timestamp_147,
                args.timestamp_156,
                save_signal_views,
                wrong_way_alerts,
            )


if __name__ == "__main__":
    main()
