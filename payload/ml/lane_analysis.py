"""Worker A: descriptive rates and exploratory lane-3 binomial association.

No causal claim: lane allocation is scheduled and can be confounded with time.
"""
from __future__ import annotations

import json
import math
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import statsmodels.api as sm

TZ = ZoneInfo("Asia/Bangkok")
MODEL_VERSION = "lane3_binomial_v1"
MIN_DAYS = 14
MIN_WINDOWS_PER_MODE = 60


def parse_summary(raw, student_id="6610301004"):
    body = json.loads(raw)
    if not isinstance(body, dict) or str(body.get("field_id")) != student_id:
        raise ValueError("Unexpected field_id")
    ts = body.get("timestamp")
    if type(ts) is not int or ts <= 0:
        raise ValueError("timestamp must be positive Unix seconds")
    date = datetime.fromtimestamp(ts, TZ)
    # Catch milliseconds and incorrect human-readable timestamp mappings.
    if abs(datetime.fromisoformat(body["timestamp_th"]).timestamp() - ts) > 1:
        raise ValueError("timestamp and timestamp_th disagree")
    if not isinstance(body.get("place_id"), str) or not body["place_id"]:
        raise ValueError("Missing place_id")
    payload = body["payload"]
    directions, counts, wrong = [], [], []
    for lane in range(1, 5):
        prefix = f"lane_{lane}_"
        direction = payload[prefix + "direction_value"]
        if type(direction) is not int or direction not in (0, 1):
            raise ValueError("Lane direction must be integer 0/1")
        if payload.get(prefix + "direction", "up" if direction else "down") != ("up" if direction else "down"):
            raise ValueError("Lane direction text and value disagree")
        n, y = payload[prefix + "vehicle_count"], payload[prefix + "wrong_way_count"]
        if type(n) is not int or type(y) is not int or not 0 <= y <= n:
            raise ValueError("Counts require 0 <= wrong_way_count <= vehicle_count")
        directions.append(direction)
        counts.append(n)
        wrong.append(y)
    if directions not in ([1, 1, 0, 0], [1, 1, 1, 0]):
        raise ValueError("Unsupported lane arrangement; expected 2/2 or 3/1")
    for key in ("vehicle_count", "wrong_way_count"):
        if type(payload[key]) is not int or payload[key] < 0:
            raise ValueError("Total counts must be nonnegative integers")
    if payload["wrong_way_count"] > payload["vehicle_count"]:
        raise ValueError("Total wrong-way count exceeds vehicle count")
    return dict(timestamp=ts, place_id=body["place_id"], mode=directions[2],
                n=counts[2], y=wrong[2], total_n=payload["vehicle_count"],
                total_y=payload["wrong_way_count"], day=date.date().isoformat(),
                hour=date.hour, weekday=date.weekday(),
                hour_fraction=date.hour + date.minute / 60 + date.second / 3600)


def observed_fields(row):
    fields = dict(vehicle_count=row["total_n"], wrong_way_count=row["total_y"],
                  lane_3_vehicle_count=row["n"], lane_3_wrong_way_count=row["y"],
                  lane_mode_31=row["mode"], lane_3_rate_valid=int(row["n"] > 0))
    if row["n"]:
        fields["lane_3_wrong_way_rate_pct"] = 100.0 * row["y"] / row["n"]
    if row["total_n"]:
        fields["wrong_way_rate_pct"] = 100.0 * row["total_y"] / row["total_n"]
    return fields


def analyze(rows):
    """Fit a rolling explanatory model; zero traffic is missing exposure, not 0%."""
    rows = [r for r in rows if r["n"] > 0]
    days = {r["day"] for r in rows}
    by_mode = [[r for r in rows if r["mode"] == mode] for mode in (0, 1)]
    fields = dict(model_ready=0, training_windows=len(rows), training_days=len(days))
    for mode, subset in zip(("22", "31"), by_mode):
        n, y = sum(r["n"] for r in subset), sum(r["y"] for r in subset)
        fields[f"windows_{mode}"] = len(subset)
        fields[f"vehicles_{mode}"] = n
        fields[f"wrong_way_{mode}"] = y
        if n:
            fields[f"observed_rate_{mode}_pct"] = 100.0 * y / n
    if all(by_mode):
        fields["observed_difference_pp"] = fields["observed_rate_31_pct"] - fields["observed_rate_22_pct"]
    if len(days) < MIN_DAYS or min(map(len, by_mode)) < MIN_WINDOWS_PER_MODE:
        return "collecting", fields
    if any(sum(r["y"] for r in subset) < 10 or sum(r["n"] - r["y"] for r in subset) < 10 for subset in by_mode):
        return "insufficient_events", fields

    # Same weekday/hour seen under BOTH modes offers at least coarse time overlap.
    slots = [{(r["weekday"], r["hour"]) for r in subset} for subset in by_mode]
    fields["time_overlap_slots"] = len(slots[0] & slots[1])
    # Treat nearby windows as correlated: standard errors clustered by local day.
    x = np.array([[1, r["mode"], math.sin(2*math.pi*r["hour_fraction"]/24),
                   math.cos(2*math.pi*r["hour_fraction"]/24),
                   *[int(r["weekday"] == d) for d in range(1, 7)]] for r in rows], dtype=float)
    if np.linalg.matrix_rank(x) < x.shape[1] or np.linalg.cond(x) > 1e6:
        return "confounded_design", fields
    y = np.array([[r["y"], r["n"] - r["y"]] for r in rows])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": np.array([r["day"] for r in rows])})
        if not fit.converged:
            return "fit_failed", fields
        lower, upper = fit.conf_int()[1]
        estimates = np.exp([fit.params[1], lower, upper])
        if not np.all(np.isfinite(estimates)) or not lower <= fit.params[1] <= upper:
            return "unstable_fit", fields
        fields.update(model_ready=1, adjusted_odds_ratio=float(estimates[0]),
                      odds_ratio_ci_low=float(estimates[1]), odds_ratio_ci_high=float(estimates[2]))
        # These are fitted, standardized probabilities, NOT held-out forecasts.
        for mode in (0, 1):
            scenario = x.copy()
            scenario[:, 1] = mode
            rate = float(np.average(fit.predict(scenario), weights=[r["n"] for r in rows])) * 100
            fields[f"fitted_rate_{'31' if mode else '22'}_pct"] = rate
        fields["fitted_difference_pp"] = fields["fitted_rate_31_pct"] - fields["fitted_rate_22_pct"]
        fields["time_overlap_available"] = int(fields["time_overlap_slots"] > 0)
        return ("exploratory" if fields["time_overlap_slots"] else "time_confounded"), fields
    except (Warning, ValueError, np.linalg.LinAlgError, OverflowError):
        return "fit_failed", fields
