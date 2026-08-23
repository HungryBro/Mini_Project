"""Render YOLO draft labels as visible review images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


CLASS_NAMES = ("car", "motorcycle", "bus", "truck")
COLORS = ((255, 180, 0), (170, 60, 255), (0, 220, 80), (0, 145, 255))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw YOLO review labels onto their images")
    parser.add_argument("--review-root", type=Path, default=Path("data/taksin_vehicles/review"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def draw_label(image, class_id: int, x_center: float, y_center: float, box_width: float, box_height: float) -> None:
    height, width = image.shape[:2]
    x1 = max(0, round((x_center - box_width / 2) * width))
    y1 = max(0, round((y_center - box_height / 2) * height))
    x2 = min(width - 1, round((x_center + box_width / 2) * width))
    y2 = min(height - 1, round((y_center + box_height / 2) * height))
    color = COLORS[class_id]
    label = CLASS_NAMES[class_id]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    text_top = max(0, y1 - text_height - baseline - 4)
    cv2.rectangle(image, (x1, text_top), (min(width - 1, x1 + text_width + 6), y1), color, -1)
    cv2.putText(image, label, (x1 + 3, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    image_dir = args.review_root / "images"
    label_dir = args.review_root / "labels"
    output = args.output or args.review_root / "previews"
    output.mkdir(parents=True, exist_ok=True)
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES) if image_dir.exists() else []
    if not images:
        raise SystemExit(f"No images found in {image_dir}")
    rendered = 0
    for image_path in images:
        destination = output / f"{image_path.stem}_preview.jpg"
        if destination.exists() and not args.overwrite:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Cannot read {image_path}")
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"{label_path}:{line_number}: expected 5 fields")
                class_id = int(fields[0])
                if class_id not in range(len(CLASS_NAMES)):
                    raise ValueError(f"{label_path}:{line_number}: class must be 0..3")
                draw_label(image, class_id, *map(float, fields[1:]))
        if not cv2.imwrite(str(destination), image):
            raise RuntimeError(f"Cannot write {destination}")
        rendered += 1
    print(f"Rendered {rendered} labelled previews in {output}")


if __name__ == "__main__":
    main()
