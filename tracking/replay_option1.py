"""Option 1 Replay script — Replay tracked JSONL log records directly to MQTT Gateway without needing a live camera."""

from __future__ import annotations

import sys
from pathlib import Path

TRACKING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACKING_DIR.parent
V2_DIR = TRACKING_DIR / "v2"

if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

import settings
from vehicle_tracking import run_jsonl_replay_mode

if __name__ == "__main__":
    print("Executing Option 1: JSONL Replay...")
    run_jsonl_replay_mode(settings)
