"""YOLO vehicle detection/tracking for the bridge camera.

COCO classes used by this project:
    2 car, 3 motorcycle, 5 bus, 7 truck

Examples:
    python vehicle_tracking.py --source taksin_bridge_sathorn/image.png
    python vehicle_tracking.py --source taksin_bridge_sathorn/taksin_bridge_sathorn_1min.mp4
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO


VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck"}


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
    parser.add_argument("--model", default="model/yolo11m.pt", help="Ultralytics model or local .pt path")
    parser.add_argument("--conf", type=float, default=0.16)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated model class names; by default car,motorcycle,bus,truck",
    )
    return parser.parse_args()


def grouped_detection(result, class_map: dict[int, str], include_track_id: bool = False) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    boxes = result.boxes
    if boxes is None:
        return grouped

    ids = boxes.id.int().cpu().tolist() if include_track_id and boxes.id is not None else [None] * len(boxes)
    for box, track_id in zip(boxes, ids):
        class_id = int(box.cls.item())
        if class_id not in class_map:
            continue
        xyxy = [round(float(value), 2) for value in box.xyxy[0].cpu().tolist()]
        grouped[class_map[class_id]].append(
            {
                "class_id": class_id,
                "track_id": track_id,
                "confidence": round(float(box.conf.item()), 4),
                "bbox_xyxy": xyxy,
                "bottom_center": [round((xyxy[0] + xyxy[2]) / 2, 2), round(xyxy[3], 2)],
            }
        )
    return grouped


def run_image(model: YOLO, class_map: dict[int, str], source: Path, output_dir: Path, conf: float, imgsz: int) -> None:
    results = model.predict(source=str(source), classes=list(class_map), conf=conf, imgsz=imgsz, verbose=False)
    result = results[0]
    grouped = grouped_detection(result, class_map)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / f"{source.stem}_annotated.jpg"
    result.save(filename=str(annotated_path))
    payload = {
        "source": str(source),
        "mode": "image_detection",
        "classes": class_map,
        "detections_by_class": grouped,
        "note": "A single image has detections but no temporal track IDs; use video mode for ByteTrack IDs.",
    }
    (output_dir / f"{source.stem}_detections.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({name: len(items) for name, items in grouped.items()}, ensure_ascii=False))
    print(f"Annotated image: {annotated_path}")


def run_video(model: YOLO, class_map: dict[int, str], source: Path, output_dir: Path, conf: float, imgsz: int, tracker: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{source.stem}_tracked.mp4"
    jsonl_path = output_dir / f"{source.stem}_tracks.jsonl"
    results = model.track(
        source=str(source),
        stream=True,
        persist=True,
        tracker=tracker,
        classes=list(class_map),
        conf=conf,
        imgsz=imgsz,
        save=True,
        project=str(output_dir),
        name="ultralytics_preview",
        exist_ok=True,
        verbose=False,
    )
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for frame_number, result in enumerate(results):
            grouped = grouped_detection(result, class_map, include_track_id=True)
            record = {"frame": frame_number, "tracks_by_class": grouped}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Track records: {jsonl_path}")
    print("Ultralytics preview is under:", output_dir / "ultralytics_preview")
    print(f"Requested output name: {video_path} (use the preview if the backend selected a different codec)")


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
    if args.source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        run_image(model, class_map, args.source, args.output_dir, args.conf, args.imgsz)
    else:
        run_video(model, class_map, args.source, args.output_dir, args.conf, args.imgsz, args.tracker)


if __name__ == "__main__":
    main()
