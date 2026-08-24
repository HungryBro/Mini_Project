"""Fine-tune a YOLO detector on the selected bridge-vehicle dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a detector using a YOLO dataset configuration")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/taksin_vehicles/external/thai_cars_native/data.yaml"),
    )
    parser.add_argument(
        "--weights",
        default="model/coco/yolo11m.pt",
        help="COCO weights or a compatible Thailand-vehicle checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=-1, help="-1 automatically fits available memory")
    parser.add_argument("--device", default=None, help="For example mps, cpu, or 0")
    parser.add_argument("--project", type=Path, default=Path("model/taksin"))
    parser.add_argument("--name", default="yolo11m_native")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise SystemExit(f"Dataset configuration is missing: {args.data}")
    data_config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    data_root = Path(data_config.get("path", "."))
    if not data_root.is_absolute():
        data_root = (args.data.parent / data_root).resolve()
    data_config["path"] = str(data_root)
    resolved_data = args.data.with_name(f"{args.data.stem}.resolved.yaml")
    resolved_data.write_text(yaml.safe_dump(data_config, sort_keys=False), encoding="utf-8")
    model = YOLO(args.weights)
    project_dir = args.project if args.project.is_absolute() else Path.cwd() / args.project
    model.train(
        data=str(resolved_data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
        patience=20,
        degrees=0.0,
        fliplr=0.5,
        mosaic=0.5,
    )


if __name__ == "__main__":
    main()
