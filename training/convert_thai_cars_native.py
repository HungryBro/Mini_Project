"""Convert the downloaded Thai-Cars segmentation export to native YOLO boxes.

The class IDs and names are kept exactly as they appear in ``data.yaml``.
Only the polygon coordinates are converted to an enclosing bounding box,
which is the format required by the current detection/tracking pipeline.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import yaml


DEFAULT_SOURCE = Path("data/Thai-Cars.v2i.yolov11")
DEFAULT_OUTPUT = Path("data/taksin_vehicles/external/thai_cars_native")
SPLITS = {"train": "train", "valid": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preserve Thai-Cars classes and convert polygons to YOLO boxes")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def polygon_to_box(parts: list[str]) -> str:
    """Return one YOLO ``class cx cy w h`` row from a polygon or box row."""
    if len(parts) < 5:
        raise ValueError("label row has fewer than five fields")
    class_id = parts[0]
    coords = [float(value) for value in parts[1:]]
    if len(coords) == 4:
        cx, cy, width, height = coords
    else:
        if len(coords) % 2:
            raise ValueError("polygon has an odd number of coordinates")
        xs = coords[0::2]
        ys = coords[1::2]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        cx, cy = (left + right) / 2, (top + bottom) / 2
        width, height = right - left, bottom - top
    values = (cx, cy, width, height)
    if not all(0.0 <= value <= 1.0 for value in values[:2]) or width < 0 or height < 0:
        raise ValueError("box coordinates are outside the normalized YOLO range")
    return "{} {:.8f} {:.8f} {:.8f} {:.8f}".format(class_id, cx, cy, width, height)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    config_path = source / "data.yaml"
    if not config_path.exists():
        raise SystemExit(f"Dataset configuration is missing: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    names = config.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda value: int(value))]
    if not names:
        raise SystemExit("data.yaml does not contain class names")

    counts: Counter[str] = Counter()
    for source_split, output_split in SPLITS.items():
        source_images = source / source_split / "images"
        source_labels = source / source_split / "labels"
        if not source_images.exists():
            raise SystemExit(f"Missing split directory: {source_images}")
        output_images = output / "images" / output_split
        output_labels = output / "labels" / output_split
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(source_images.iterdir()):
            if not image_path.is_file() or image_path.name.startswith("."):
                continue
            shutil.copy2(image_path, output_images / image_path.name)
            label_path = source_labels / f"{image_path.stem}.txt"
            output_label = output_labels / label_path.name
            rows: list[str] = []
            if label_path.exists():
                for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        row = polygon_to_box(line.split())
                    except ValueError as error:
                        raise SystemExit(f"{label_path}:{line_number}: {error}") from error
                    rows.append(row)
                    counts[names[int(row.split()[0])]] += 1
            output_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    output_config = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "data.yaml").write_text(yaml.safe_dump(output_config, sort_keys=False), encoding="utf-8")
    print(f"Wrote native detection dataset: {output}")
    print(f"Classes ({len(names)}): {', '.join(str(name) for name in names)}")
    print("Objects by class:")
    for class_name in names:
        print(f"  {class_name}: {counts[class_name]}")


if __name__ == "__main__":
    main()
