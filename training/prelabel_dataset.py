"""Create YOLO-format draft labels using a COCO-pretrained YOLO model.

The labels are deliberately placed under ``review``: they are a starting point
for annotation, not ground truth.  A person must correct boxes and classes
before ``split_reviewed_dataset.py`` and training.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


# Source COCO ID -> target training ID.
COCO_TO_TARGET = {2: 0, 3: 1, 5: 2, 7: 3}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-label bridge frames for human review")
    parser.add_argument("--source", type=Path, default=Path("data/taksin_vehicles/raw/images"))
    parser.add_argument("--review-root", type=Path, default=Path("data/taksin_vehicles/review"))
    parser.add_argument("--model", default="model/yolo11m.pt", help="COCO model or another .pt checkpoint")
    parser.add_argument("--conf", type=float, default=0.10, help="Low threshold keeps candidates for review")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default=None, help="For example mps, cpu, or 0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def yolo_line(class_id: int, xyxy: list[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = xyxy
    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    values = [max(0.0, min(1.0, value)) for value in (x_center, y_center, box_width, box_height)]
    return f"{class_id} " + " ".join(f"{value:.6f}" for value in values)


def main() -> None:
    args = parse_args()
    images = sorted(path for path in args.source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES) if args.source.exists() else []
    if not images:
        raise SystemExit(f"No images found in {args.source}")
    image_dir = args.review_root / "images"
    label_dir = args.review_root / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    total_detections = 0

    for index, source in enumerate(images, start=1):
        destination = image_dir / source.name
        label_path = label_dir / f"{source.stem}.txt"
        if (destination.exists() or label_path.exists()) and not args.overwrite:
            continue
        result = model.predict(
            source=str(source),
            classes=list(COCO_TO_TARGET),
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        height, width = result.orig_shape
        lines: list[str] = []
        for box in result.boxes or []:
            coco_class = int(box.cls.item())
            target_class = COCO_TO_TARGET.get(coco_class)
            if target_class is None:
                continue
            lines.append(yolo_line(target_class, box.xyxy[0].cpu().tolist(), width, height))
        shutil.copy2(source, destination)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        total_detections += len(lines)
        if index % 25 == 0 or index == len(images):
            print(f"Pre-labelled {index}/{len(images)} images")
    print(f"Draft labels written to {args.review_root}; {total_detections} vehicle candidates require review.")


if __name__ == "__main__":
    main()
