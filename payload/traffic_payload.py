"""Build one-minute traffic summaries from vehicle_tracking JSONL records.

The tracker writes one raw record per video frame.  This module keeps the
inference output untouched and turns those records into compact payloads that
are practical to send over UDP or MQTT.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
from typing import Any


VEHICLE_TYPE_ALIASES = {
    "moto": "motorcycle",
    "motorbike": "motorcycle",
    "truck trailer": "truck",
}


def parse_start_time(value: str | None) -> datetime:
    """Return a timezone-aware anchor time for a replay session."""
    if not value:
        return datetime.now(timezone.utc).replace(microsecond=0)
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iso_time(anchor: datetime, source_seconds: float) -> str:
    return (anchor + timedelta(seconds=source_seconds)).isoformat().replace("+00:00", "Z")


def normalize_vehicle_type(value: object) -> str:
    label = str(value).strip().lower()
    return VEHICLE_TYPE_ALIASES.get(label, label or "unknown")


@dataclass
class TrafficWindowAggregator:
    """Accumulate tracker records and emit one normalized payload per window."""

    window_seconds: float = 60.0
    camera_id: str = "CAM_112"
    site_id: str = "krung_thon_bridge"
    site_name: str = "Krung Thon Bridge"
    camera_name: str = "Main traffic camera"
    anchor_time: datetime = field(default_factory=lambda: parse_start_time(None))

    _window_start: float | None = field(default=None, init=False)
    _last_time: float | None = field(default=None, init=False)
    _camera_profile: str = field(default="krung_thon_bridge", init=False)
    _samples: int = field(default=0, init=False)
    _vehicles: dict[str, str] = field(default_factory=dict, init=False)
    _wrong_way_events: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _signals_147: dict[str, str] = field(default_factory=dict, init=False)
    _signals_156: dict[str, str] = field(default_factory=dict, init=False)
    _lanes: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def add_frame(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        """Add one raw tracker record and return any completed payloads."""
        source_time = float(record.get("time_seconds", 0.0))
        bucket_start = math.floor(source_time / self.window_seconds) * self.window_seconds
        completed: list[dict[str, Any]] = []

        if self._window_start is None:
            self._window_start = bucket_start
        elif bucket_start != self._window_start:
            payload = self.flush()
            if payload:
                completed.append(payload)
            self._window_start = bucket_start

        self._last_time = source_time
        self._samples += 1
        self._camera_profile = str(record.get("camera_profile", self._camera_profile))
        self._signals_147 = dict(record.get("signal_147_states") or {})
        self._signals_156 = dict(record.get("signal_156_states") or {})
        self._lanes = self._normalized_lanes(record.get("lane_signal_fusion") or {})

        for raw_type, vehicles in (record.get("tracks_by_class") or {}).items():
            vehicle_type = normalize_vehicle_type(raw_type)
            for vehicle in vehicles or []:
                track_id = vehicle.get("track_id")
                if track_id is None:
                    continue
                track_key = str(track_id)
                self._vehicles[track_key] = vehicle_type
                if bool(vehicle.get("wrong_way")):
                    self._record_wrong_way(track_id, vehicle_type, vehicle, source_time)

        return completed

    def flush(self) -> dict[str, Any] | None:
        """Emit the current window, then reset only its accumulated state."""
        if self._window_start is None or self._last_time is None or self._samples == 0:
            return None

        window_end = self._last_time
        counts = Counter(self._vehicles.values())
        payload = {
            "schema_version": "traffic.v1",
            "message_type": "traffic_summary",
            "timestamp": iso_time(self.anchor_time, window_end),
            "timestamp_unix": int((self.anchor_time + timedelta(seconds=window_end)).timestamp()),
            "window": {
                "start": iso_time(self.anchor_time, self._window_start),
                "end": iso_time(self.anchor_time, window_end),
                "seconds": round(window_end - self._window_start, 3),
                "configured_seconds": self.window_seconds,
            },
            "location": {
                "site_id": self.site_id,
                "site_name": self.site_name,
                "camera_id": self.camera_id,
                "camera_name": self.camera_name,
            },
            "camera_profile": self._camera_profile,
            "signals": {
                "camera_147": self._signals_147,
                "camera_156": self._signals_156,
            },
            "lanes": self._lanes,
            "traffic": {
                "unique_vehicle_count": len(self._vehicles),
                "vehicles_by_type": dict(sorted(counts.items())),
                "sample_count": self._samples,
            },
            "wrong_way": {
                "count": len(self._wrong_way_events),
                "events": list(self._wrong_way_events.values()),
            },
        }
        self._reset_window()
        return payload

    def _record_wrong_way(
        self,
        track_id: object,
        vehicle_type: str,
        vehicle: dict[str, Any],
        source_time: float,
    ) -> None:
        key = str(track_id)
        seen_at = iso_time(self.anchor_time, source_time)
        existing = self._wrong_way_events.get(key)
        if existing is None:
            self._wrong_way_events[key] = {
                "event_type": "WRONG_WAY",
                "track_id": track_id,
                "vehicle_type": vehicle_type,
                "lane_id": vehicle.get("lane_id"),
                "direction": vehicle.get("direction", "unknown"),
                "expected_direction": vehicle.get("expected_direction", "unknown"),
                "first_seen": seen_at,
                "last_seen": seen_at,
                "observations": 1,
            }
            return

        existing["last_seen"] = seen_at
        existing["observations"] += 1
        existing["lane_id"] = vehicle.get("lane_id")
        existing["direction"] = vehicle.get("direction", "unknown")
        existing["expected_direction"] = vehicle.get("expected_direction", "unknown")

    @staticmethod
    def _normalized_lanes(raw_lanes: dict[str, Any]) -> dict[str, dict[str, Any]]:
        lanes: dict[str, dict[str, Any]] = {}
        for lane, state in raw_lanes.items():
            if not isinstance(state, dict):
                continue
            lanes[str(lane)] = {
                "last_direction": state.get("direction", "unknown"),
                "source": state.get("source", "none"),
                "agrees": state.get("agrees"),
            }
        return lanes

    def _reset_window(self) -> None:
        self._last_time = None
        self._samples = 0
        self._vehicles.clear()
        self._wrong_way_events.clear()
        self._signals_147 = {}
        self._signals_156 = {}
        self._lanes = {}
