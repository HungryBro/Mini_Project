"""Validate reviewed labels and make leakage-resistant train/val/test splits."""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path


CLASS_NAMES = ("car", "motorcycle", "bus", "truck")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split reviewed 4-class YOLO labels")
    parser.add_argument("--review-root", type=Path, default=Path("data/taksin_vehicles/review"))
    parser.add_argument("--output", type=Path, default=Path("data/taksin_vehicles/yolo"))
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.20)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_label(path: Path) -> None:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields")
        class_id = int(fields[0])
        if class_id not in range(len(CLASS_NAMES)):
            raise ValueError(f"{path}:{line_number}: class must be 0..3")
        coordinates = [float(value) for value in fields[1:]]
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"{path}:{line_number}: YOLO coordinates must be within 0..1")
        if coordinates[2] <= 0 or coordinates[3] <= 0:
            raise ValueError(f"{path}:{line_number}: box width and height must be positive")


def source_key(image: Path) -> str:
    # Frames made by prepare_dataset end in _f0000000; this keeps each video
    # chronologically separated instead of randomizing near-identical frames.
    return image.stem.rsplit("_f", 1)[0]


def copy_pair(image: Path, label: Path, output: Path, split: str) -> None:
    image_destination = output / "images" / split / image.name
    label_destination = output / "labels" / split / label.name
    image_destination.parent.mkdir(parents=True, exist_ok=True)
    label_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, image_destination)
    shutil.copy2(label, label_destination)


def main() -> None:
    args = parse_args()
    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise SystemExit("--train, --val and --test must add up to 1.0")
    images_dir, labels_dir = args.review_root / "images", args.review_root / "labels"
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES) if images_dir.exists() else []
    if not images:
        raise SystemExit(f"No reviewed images in {images_dir}")
    pairs: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        if not label.exists():
            raise SystemExit(f"Missing reviewed label: {label}")
        validate_label(label)
        pairs[source_key(image)].append((image, label))

    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.output} already contains files; use --overwrite after checking it")
    if args.overwrite and args.output.exists():
        shutil.rmtree(args.output)
    counts = defaultdict(int)
    for group in pairs.values():
        group.sort(key=lambda pair: pair[0].name)
        train_end = round(len(group) * args.train)
        val_end = train_end + round(len(group) * args.val)
        for index, (image, label) in enumerate(group):
            split = "train" if index < train_end else "val" if index < val_end else "test"
            copy_pair(image, label, args.output, split)
            counts[split] += 1
    print("Validated and split reviewed images:", dict(counts))
    print("Temporal blocks are kept apart within each source video to reduce frame leakage.")


if __name__ == "__main__":
    main()
