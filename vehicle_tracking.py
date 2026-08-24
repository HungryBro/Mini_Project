"""YOLO vehicle detection/tracking for the bridge camera.

COCO classes used by this project:
    2 car, 3 motorcycle, 5 bus, 7 truck

Examples:
    python vehicle_tracking.py --source locations/taksin/image/image.png
    python vehicle_tracking.py --source locations/taksin/video/taksin_bridge_sathorn_1min.mp4
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from ultralytics import YOLO


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
    parser.add_argument("--conf", type=float, default=0.16)
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


def bridge_roi(frame) -> np.ndarray:
    """Return the bridge-road trapezoid at the current frame resolution."""
    height, width = frame.shape[:2]
    return np.array(
        [(round(x * width), round(y * height)) for x, y in BRIDGE_ROI_NORMALIZED],
        dtype=np.int32,
    )


def box_is_inside_roi(box, roi_points: np.ndarray | None) -> bool:
    """Use a vehicle's tyre-side centre, rather than its whole box, for ROI membership."""
    if roi_points is None:
        return True
    x1, _y1, x2, y2 = box.xyxy[0].cpu().tolist()
    return point_is_inside_roi(((x1 + x2) / 2, y2), roi_points)


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


def stabilized_tracks(
    result,
    class_map: dict[int, str],
    class_aliases: dict[str, str],
    roi_points: np.ndarray | None,
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
            if class_id not in class_map or not box_is_inside_roi(box, roi_points):
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


def draw_bridge_guides(frame, roi_points: np.ndarray, show_gates: bool) -> None:
    """Draw the bridge boundary and optionally the guide-only Gate A/B lines."""
    cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    if not show_gates:
        return
    gate_specs = (("GATE A", GATE_A_NORMALIZED, (255, 0, 0)), ("GATE B", GATE_B_NORMALIZED, (0, 0, 255)))
    for label, (_x_normalized, y_normalized), color in gate_specs:
        left, right = gate_endpoints(roi_points, y_normalized, frame.shape[0])
        cv2.line(frame, left, right, color, 4, cv2.LINE_AA)
        label_anchor = (left[0] + 8, max(26, left[1] - 10))
        cv2.putText(frame, label, label_anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def grouped_detection(
    result,
    class_map: dict[int, str],
    include_track_id: bool = False,
    class_aliases: dict[str, str] | None = None,
    roi_points: np.ndarray | None = None,
    label_overrides: dict[int, str] | None = None,
    retained_tracks: list[dict[str, object]] | None = None,
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    boxes = result.boxes
    class_aliases = class_aliases or {}
    label_overrides = label_overrides or {}
    if boxes is not None:
        ids = boxes.id.int().cpu().tolist() if include_track_id and boxes.id is not None else [None] * len(boxes)
        for box, track_id in zip(boxes, ids):
            class_id = int(box.cls.item())
            if class_id not in class_map or not box_is_inside_roi(box, roi_points):
                continue
            output_name = label_overrides.get(track_id, class_aliases.get(class_map[class_id], class_map[class_id]))
            xyxy = [round(float(value), 2) for value in box.xyxy[0].cpu().tolist()]
            grouped[output_name].append(
                {
                    "class_id": class_id,
                    "track_id": track_id,
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox_xyxy": xyxy,
                    "bottom_center": [round((xyxy[0] + xyxy[2]) / 2, 2), round(xyxy[3], 2)],
                    "occluded_prediction": False,
                }
            )
    for track in retained_tracks or []:
        xyxy = [round(float(value), 2) for value in track["bbox_xyxy"]]
        grouped[str(track["label"])].append(
            {
                "class_id": None,
                "track_id": track["track_id"],
                "confidence": None,
                "bbox_xyxy": xyxy,
                "bottom_center": [round((xyxy[0] + xyxy[2]) / 2, 2), round(xyxy[3], 2)],
                "occluded_prediction": True,
            }
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
) -> object:
    """Render boxes with stable class colors and labels without confidence text."""
    frame = result.orig_img.copy()
    if roi_points is not None:
        draw_bridge_guides(frame, roi_points, show_gates)
    boxes = result.boxes
    class_aliases = class_aliases or {}
    label_overrides = label_overrides or {}

    def draw_box(label: str, xyxy: list[float], occluded: bool = False) -> None:
        color_name = "car" if label in {"car", "taxi"} else label
        color = CLASS_COLORS_BGR.get(color_name, (255, 255, 255))
        x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1 if occluded else 2)
        cv2.putText(
            frame,
            f"{label} (hold)" if occluded else label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 if occluded else 0.65,
            color,
            1 if occluded else 2,
            cv2.LINE_AA,
        )

    if boxes is not None:
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
        for box, track_id in zip(boxes, ids):
            class_id = int(box.cls.item())
            if class_id not in class_map or not box_is_inside_roi(box, roi_points):
                continue
            original_name = class_map[class_id]
            output_name = label_overrides.get(track_id, class_aliases.get(original_name, original_name))
            draw_box(output_name, [float(value) for value in box.xyxy[0].cpu().tolist()])
    for track in retained_tracks or []:
        draw_box(str(track["label"]), list(track["bbox_xyxy"]), occluded=True)
    return frame


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
    roi_points = bridge_roi(result.orig_img) if bridge_only else None
    grouped = grouped_detection(result, class_map, class_aliases=class_aliases, roi_points=roi_points)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / f"{source.stem}_annotated.jpg"
    cv2.imwrite(str(annotated_path), annotated_frame(result, class_map, class_aliases, roi_points, show_gates))
    payload = {
        "source": str(source),
        "mode": "image_detection",
        "classes": class_map,
        "class_aliases": class_aliases,
        "bridge_roi_normalized": BRIDGE_ROI_NORMALIZED if bridge_only else None,
        "roi_filter": "bottom_center_inside_bridge_polygon" if bridge_only else None,
        "gates_normalized": {"gate_a": GATE_A_NORMALIZED, "gate_b": GATE_B_NORMALIZED}
        if bridge_only and show_gates
        else None,
        "detections_by_class": grouped,
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
                roi_points = bridge_roi(result.orig_img) if bridge_only else None
                label_overrides, retained_tracks = stabilized_tracks(
                    result,
                    class_map,
                    class_aliases,
                    roi_points,
                    track_memory,
                    frame_number,
                    occlusion_hold,
                    max_held_tracks,
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
                )
                record = {"frame": frame_number, "tracks_by_class": grouped}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if (frame_number + 1) % 100 == 0:
                    print(f"Processed {frame_number + 1}/{total_frames} frames", flush=True)
    finally:
        writer.release()
    print(f"Track records: {jsonl_path}")
    print(f"Annotated video: {video_path}")


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise SystemExit(f"Source not found: {args.source}")
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
                args.show_gates,
                args.agnostic_nms,
                args.iou,
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
            args.show_gates,
            args.agnostic_nms,
            args.iou,
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
            args.show_gates,
            args.agnostic_nms,
            args.occlusion_hold,
            args.iou,
            args.max_held_tracks,
        )


if __name__ == "__main__":
    main()
