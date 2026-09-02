"""Build local gateway summaries from ``vehicle_tracking`` JSONL records.

The tracker feeds 15-second windows into the Mac gateway. The gateway then
combines those windows into one one-minute MQTT summary for the cloud broker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
from typing import Any
from zoneinfo import ZoneInfo


BANGKOK_TIMEZONE = ZoneInfo("Asia/Bangkok")


def _as_bangkok(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=BANGKOK_TIMEZONE)
    return value.astimezone(BANGKOK_TIMEZONE)


def parse_start_time(value: str | None) -> datetime:
    """Return a Bangkok-time anchor for a replay session."""
    if not value:
        return datetime.now(BANGKOK_TIMEZONE).replace(microsecond=0)
    return _as_bangkok(datetime.fromisoformat(value.replace("Z", "+00:00")))


def iso_time(anchor: datetime, source_seconds: float) -> str:
    """Format timestamps in Thailand time, including the +07:00 offset."""
    return (_as_bangkok(anchor) + timedelta(seconds=source_seconds)).isoformat(timespec="seconds")


def _parse_iso_time(value: object) -> datetime:
    return _as_bangkok(datetime.fromisoformat(str(value).replace("Z", "+00:00")))


def _rate_per_100(numerator: int, denominator: int) -> float:
    return round((numerator / denominator) * 100, 2) if denominator else 0.0


@dataclass
class TrafficWindowAggregator:
    """Turn per-frame tracker records into 15-second gateway input payloads."""

    window_seconds: float = 15.0
    camera_id: str = "CAM_112"
    site_id: str = "krung_thon_bridge"
    site_name: str = "Krung Thon Bridge"
    camera_name: str = "Main traffic camera"
    anchor_time: datetime = field(default_factory=lambda: parse_start_time(None))
    use_wall_clock: bool = False

    _window_start: float | None = field(default=None, init=False)
    _last_time: float | None = field(default=None, init=False)
    _camera_profile: str = field(default="krung_thon_bridge", init=False)
    _samples: int = field(default=0, init=False)
    _vehicles: dict[str, str] = field(default_factory=dict, init=False)
    _wrong_way_events: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _lanes: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.anchor_time = _as_bangkok(self.anchor_time)

    def add_frame(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        """Add one tracker record and return any complete 15-second windows."""
        completed: list[dict[str, Any]] = []
        incoming_lanes = self._normalized_lanes(record.get("lane_signal_fusion") or {})

        if self.use_wall_clock:
            import time

            current_time = time.time()
            if self._window_start is None:
                self._window_start = current_time
            elif current_time - self._window_start >= self.window_seconds:
                self._last_time = current_time
                payload = self.flush(complete_window=True)
                if payload:
                    completed.append(payload)
                self._window_start = current_time
            self._last_time = current_time
        else:
            source_time = float(record.get("time_seconds", 0.0))
            bucket_start = math.floor(source_time / self.window_seconds) * self.window_seconds
            if self._window_start is None:
                self._window_start = bucket_start
            elif bucket_start != self._window_start:
                payload = self.flush(complete_window=True)
                if payload:
                    completed.append(payload)
                self._window_start = bucket_start
            self._last_time = source_time
            current_time = source_time

        # Do not mix traffic before and after a timetable direction change in
        # the same one-minute payload. The transition starts a new partial one.
        if self._samples and self._signature(incoming_lanes) != self._signature(self._lanes):
            payload = self.flush(complete_window=False)
            if payload:
                completed.append(payload)
            self._window_start = current_time
            self._last_time = current_time

        self._samples += 1
        self._camera_profile = str(record.get("camera_profile", self._camera_profile))
        self._lanes = incoming_lanes

        source_time_for_event = float(record.get("time_seconds", 0.0))
        for vehicles in (record.get("tracks_by_class") or {}).values():
            for vehicle in vehicles or []:
                vehicle_id = vehicle.get("vehicle_id") or vehicle.get("track_id")
                if vehicle_id is None:
                    continue
                vehicle_key = str(vehicle_id)
                self._vehicles[vehicle_key] = str(vehicle.get("lane_id") or "unknown")
                if bool(vehicle.get("wrong_way")):
                    event_id = vehicle.get("wrong_way_event_id") or f"WW-{vehicle_key}"
                    self._record_wrong_way(event_id, vehicle, source_time_for_event)

        return completed

    def flush(self, *, complete_window: bool = False) -> dict[str, Any] | None:
        """Emit the active gateway-input window, including a partial final window."""
        if self._window_start is None or self._last_time is None or self._samples == 0:
            return None

        window_end = self._last_time
        if self.use_wall_clock:
            start_dt = datetime.fromtimestamp(self._window_start, tz=BANGKOK_TIMEZONE)
            end_dt = datetime.fromtimestamp(window_end, tz=BANGKOK_TIMEZONE)
        else:
            start_dt = _as_bangkok(self.anchor_time) + timedelta(seconds=self._window_start)
            end_dt = _as_bangkok(self.anchor_time) + timedelta(seconds=window_end)
        elapsed_sec = round(window_end - self._window_start, 3)

        payload = {
            "timestamp": end_dt.isoformat(timespec="seconds"),
            "timestamp_unix": int(end_dt.timestamp()),
            "window": {
                "start": start_dt.isoformat(timespec="seconds"),
                "end": end_dt.isoformat(timespec="seconds"),
                "seconds": elapsed_sec,
                "configured_seconds": self.window_seconds,
                "complete_window": complete_window,
            },
            "location": {
                "site_id": self.site_id,
                "site_name": self.site_name,
                "camera_id": self.camera_id,
            },
            "camera_profile": self._camera_profile,
            "lanes": self._lanes,
            "traffic": {
                "vehicle_count": len(self._vehicles),
                "lane_vehicle_counts": self._lane_vehicle_counts(),
                # Local-only details let the one-minute gateway count a car
                # that remains in view across minute boundaries only once.
                # They are stripped before the cloud MQTT publish.
                "vehicle_ids": sorted(self._vehicles),
                "vehicle_lanes": dict(self._vehicles),
            },
            "wrong_way": self._wrong_way_summary(),
        }
        self._reset_window()
        return payload

    def _lane_vehicle_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for lane_id in self._vehicles.values():
            counts[lane_id] = counts.get(lane_id, 0) + 1
        return dict(sorted(counts.items()))

    def _wrong_way_summary(self) -> dict[str, Any]:
        by_lane: dict[str, dict[str, Any]] = {}
        lane_vehicle_counts = self._lane_vehicle_counts()
        for event in self._wrong_way_events.values():
            lane_id = str(event.get("lane_id") or "unknown")
            if lane_id not in by_lane:
                by_lane[lane_id] = {
                    "count": 0,
                    "direction": event.get("direction", "unknown"),
                    "expected_direction": event.get("expected_direction", "unknown"),
                }
            by_lane[lane_id]["count"] += 1

        for lane_id, summary in by_lane.items():
            vehicle_count = lane_vehicle_counts.get(lane_id, 0)
            summary["vehicle_count"] = vehicle_count
            summary["rate_per_100_vehicles"] = _rate_per_100(
                int(summary["count"]), vehicle_count
            )

        return {
            "count": len(self._wrong_way_events),
            "rate_per_100_vehicles": _rate_per_100(
                len(self._wrong_way_events), len(self._vehicles)
            ),
            "by_lane": dict(sorted(by_lane.items())),
            "events": list(self._wrong_way_events.values()),
        }

    def _record_wrong_way(
        self, event_id: object, vehicle: dict[str, Any], source_time: float
    ) -> None:
        key = str(event_id)
        seen_at = iso_time(self.anchor_time, source_time)
        existing = self._wrong_way_events.get(key)
        if existing is None:
            self._wrong_way_events[key] = {
                "event_id": key,
                "lane_id": vehicle.get("lane_id"),
                "direction": vehicle.get("direction", "unknown"),
                "expected_direction": vehicle.get("expected_direction", "unknown"),
                "first_seen": seen_at,
            }
            return
        existing["lane_id"] = vehicle.get("lane_id")
        existing["direction"] = vehicle.get("direction", "unknown")
        existing["expected_direction"] = vehicle.get("expected_direction", "unknown")

    @staticmethod
    def _normalized_lanes(raw_lanes: dict[str, Any]) -> dict[str, dict[str, Any]]:
        lanes: dict[str, dict[str, Any]] = {}
        for lane, state in raw_lanes.items():
            if isinstance(state, dict):
                lanes[str(lane)] = {"direction": state.get("direction", "unknown")}
        return lanes

    @staticmethod
    def _signature(lanes: dict[str, dict[str, Any]]) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (str(lane_id), str((state or {}).get("direction", "unknown")))
                for lane_id, state in lanes.items()
            )
        )

    def _reset_window(self) -> None:
        self._last_time = None
        self._samples = 0
        self._vehicles.clear()
        self._wrong_way_events.clear()
        self._lanes = {}


@dataclass
class TrafficGatewayAggregator:
    """Combine one-minute local gateway inputs into one-minute cloud payloads."""

    window_seconds: float = 60.0

    _window_start: datetime | None = field(default=None, init=False)
    _window_end: datetime | None = field(default=None, init=False)
    _covered_seconds: float = field(default=0.0, init=False)
    _location: dict[str, Any] = field(default_factory=dict, init=False)
    _camera_profile: str = field(default="krung_thon_bridge", init=False)
    _lanes: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _lane_signature: tuple[tuple[str, str], ...] = field(default=(), init=False)
    _vehicles: dict[str, str] = field(default_factory=dict, init=False)
    _fallback_vehicle_count: int = field(default=0, init=False)
    _lane_vehicle_counts: dict[str, int] = field(default_factory=dict, init=False)
    _wrong_way_events: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def add_payload(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Add one gateway-input payload and return any completed cloud windows."""
        completed: list[dict[str, Any]] = []
        incoming_lanes = dict(payload.get("lanes") or {})
        incoming_signature = self._signature(incoming_lanes)

        if self._window_start is not None and incoming_signature != self._lane_signature:
            partial = self.flush(complete_window=False)
            if partial:
                completed.append(partial)

        if self._window_start is None:
            self._begin_window(payload, incoming_lanes, incoming_signature)

        self._window_end = _parse_iso_time(payload["window"]["end"])
        source_window = payload.get("window") or {}
        source_seconds = float(source_window.get("seconds", 0.0))
        if bool(source_window.get("complete_window")):
            source_seconds = float(source_window.get("configured_seconds", source_seconds))
        self._covered_seconds += source_seconds

        traffic = payload.get("traffic") or {}
        vehicle_lanes = traffic.get("vehicle_lanes") or {}
        if isinstance(vehicle_lanes, dict) and vehicle_lanes:
            for vehicle_id, lane_id in vehicle_lanes.items():
                self._vehicles[str(vehicle_id)] = str(lane_id or "unknown")
        else:
            self._fallback_vehicle_count += int(traffic.get("vehicle_count", 0))
            for lane_id, count in (traffic.get("lane_vehicle_counts") or {}).items():
                lane_key = str(lane_id)
                self._lane_vehicle_counts[lane_key] = (
                    self._lane_vehicle_counts.get(lane_key, 0) + int(count)
                )

        for event in (payload.get("wrong_way") or {}).get("events") or []:
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("event_id") or "")
            if event_id:
                self._wrong_way_events[event_id] = dict(event)

        if self._covered_seconds >= self.window_seconds:
            cloud_payload = self.flush(complete_window=True)
            if cloud_payload:
                completed.append(cloud_payload)
        return completed

    def flush(self, *, complete_window: bool = False) -> dict[str, Any] | None:
        """Emit the active cloud summary. Final or schedule-change windows are partial."""
        if self._window_start is None or self._window_end is None:
            return None

        by_lane: dict[str, dict[str, Any]] = {}
        vehicle_count = len(self._vehicles) if self._vehicles else self._fallback_vehicle_count
        lane_vehicle_counts = (
            self._vehicle_lane_counts() if self._vehicles else self._lane_vehicle_counts
        )
        for event in self._wrong_way_events.values():
            lane_id = str(event.get("lane_id") or "unknown")
            if lane_id not in by_lane:
                by_lane[lane_id] = {
                    "count": 0,
                    "vehicle_count": lane_vehicle_counts.get(lane_id, 0),
                    "direction": event.get("direction", "unknown"),
                    "expected_direction": event.get("expected_direction", "unknown"),
                }
            by_lane[lane_id]["count"] += 1

        for summary in by_lane.values():
            summary["rate_per_100_vehicles"] = _rate_per_100(
                int(summary["count"]), int(summary["vehicle_count"])
            )

        payload = {
            "timestamp": self._window_end.isoformat(timespec="seconds"),
            "timestamp_unix": int(self._window_end.timestamp()),
            "window": {
                "start": self._window_start.isoformat(timespec="seconds"),
                "end": self._window_end.isoformat(timespec="seconds"),
                "seconds": round(self._covered_seconds, 3),
                "configured_seconds": self.window_seconds,
                "complete_window": complete_window,
            },
            "location": self._location,
            "camera_profile": self._camera_profile,
            "lanes": self._lanes,
            "traffic": {"vehicle_count": vehicle_count},
            "wrong_way": {
                "count": len(self._wrong_way_events),
                "rate_per_100_vehicles": _rate_per_100(
                    len(self._wrong_way_events), vehicle_count
                ),
                "by_lane": dict(sorted(by_lane.items())),
            },
        }
        self._reset_window()
        return payload

    def _begin_window(
        self,
        payload: dict[str, Any],
        lanes: dict[str, dict[str, Any]],
        signature: tuple[tuple[str, str], ...],
    ) -> None:
        self._window_start = _parse_iso_time(payload["window"]["start"])
        self._location = dict(payload.get("location") or {})
        self._camera_profile = str(payload.get("camera_profile", self._camera_profile))
        self._lanes = lanes
        self._lane_signature = signature

    @staticmethod
    def _signature(lanes: dict[str, dict[str, Any]]) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (
                    str(lane_id),
                    str((state or {}).get("direction", (state or {}).get("last_direction", "unknown"))),
                )
                for lane_id, state in lanes.items()
            )
        )

    def _reset_window(self) -> None:
        self._window_start = None
        self._window_end = None
        self._covered_seconds = 0.0
        self._location = {}
        self._lanes = {}
        self._lane_signature = ()
        self._vehicles.clear()
        self._fallback_vehicle_count = 0
        self._lane_vehicle_counts.clear()
        self._wrong_way_events.clear()

    def _vehicle_lane_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for lane_id in self._vehicles.values():
            counts[lane_id] = counts.get(lane_id, 0) + 1
        return counts
