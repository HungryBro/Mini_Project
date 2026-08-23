"""Sample representative frames from the Taksin Bridge camera videos.

This creates *unlabelled* images only.  Do not train until every sampled image
has a corresponding reviewed YOLO label file (an empty label file is valid for
frames containing no vehicles in the bridge ROI).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


DEFAULT_VIDEOS = sorted(Path("taksin_bridge_sathorn/video").glob("*.mp4"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract evenly spaced bridge-camera frames")
    parser.add_argument("--video", type=Path, action="append", help="Video to sample; repeat for more videos")
    parser.add_argument("--output", type=Path, default=Path("data/taksin_vehicles/raw/images"))
    parser.add_argument("--manifest", type=Path, default=Path("data/taksin_vehicles/raw/manifest.csv"))
    parser.add_argument("--sample-every", type=float, default=3.0, help="Seconds between frames")
    parser.add_argument("--max-per-video", type=int, default=200)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sample_video(video: Path, output: Path, interval: float, maximum: int, quality: int, overwrite: bool) -> list[dict[str, object]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps * interval))
    records: list[dict[str, object]] = []
    output.mkdir(parents=True, exist_ok=True)

    for frame_index in range(0, frame_count, step):
        if len(records) >= maximum:
            break
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        filename = f"{video.stem}_f{frame_index:07d}.jpg"
        destination = output / filename
        if overwrite or not destination.exists():
            if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                raise RuntimeError(f"Cannot write {destination}")
        records.append(
            {
                "image": destination.as_posix(),
                "source_video": video.as_posix(),
                "frame": frame_index,
                "timestamp_seconds": round(frame_index / fps, 3),
                "fps": round(fps, 3),
            }
        )
    capture.release()
    return records


def main() -> None:
    args = parse_args()
    if args.sample_every <= 0 or args.max_per_video < 1:
        raise SystemExit("--sample-every must be positive and --max-per-video must be at least 1")
    videos = args.video or DEFAULT_VIDEOS
    if not videos:
        raise SystemExit("No videos found. Pass one or more --video paths.")
    missing = [str(video) for video in videos if not video.exists()]
    if missing:
        raise SystemExit("Missing video(s): " + ", ".join(missing))

    records: list[dict[str, object]] = []
    for video in videos:
        records.extend(sample_video(video, args.output, args.sample_every, args.max_per_video, args.jpeg_quality, args.overwrite))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "source_video", "frame", "timestamp_seconds", "fps"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Prepared {len(records)} frames in {args.output}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
