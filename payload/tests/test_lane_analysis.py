import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.lane_analysis import TZ, analyze, observed_fields, parse_summary
from ml.lane_effect_worker import flush, open_db, stage


def sample(mode=0, timestamp=1788942855, n=100, y=5):
    p = dict(vehicle_count=n*4, wrong_way_count=y*4)
    for lane, direction in enumerate([1, 1, mode, 0], 1):
        p.update({f"lane_{lane}_direction_value": direction,
                  f"lane_{lane}_direction": "up" if direction else "down",
                  f"lane_{lane}_vehicle_count": n, f"lane_{lane}_wrong_way_count": y})
    return json.dumps(dict(field_id="6610301004", place_id="krung_thon_bridge",
                           timestamp=timestamp, timestamp_th=datetime.fromtimestamp(timestamp, TZ).isoformat(), payload=p))


def synthetic_rows():
    """Known association with balanced mode/time exposure, never sent to cloud."""
    rng = np.random.default_rng(301004)
    start = datetime(2026, 8, 1, tzinfo=TZ)
    rows = []
    for day in range(21):
        for hour in range(24):
            for mode in (0, 1):
                ts = int((start + timedelta(days=day, hours=hour, minutes=mode)).timestamp())
                y = int(rng.binomial(100, 0.05 if mode == 0 else 0.15))
                rows.append(parse_summary(sample(mode, ts, y=y)))
    return rows


class AnalysisTests(unittest.TestCase):
    def test_schema_and_zero_exposure(self):
        r = parse_summary(sample(n=0, y=0))
        self.assertNotIn("lane_3_wrong_way_rate_pct", observed_fields(r))
        self.assertEqual(observed_fields(r)["lane_3_rate_valid"], 0)

    def test_reject_bad_counts_and_timestamp(self):
        with self.assertRaises(ValueError):
            parse_summary(sample(n=2, y=3))
        body = json.loads(sample())
        body["timestamp"] += 60
        with self.assertRaises(ValueError):
            parse_summary(json.dumps(body))

    def test_unsupported_mode(self):
        body = json.loads(sample())
        body['payload']['lane_1_direction_value'] = 0
        body['payload']['lane_1_direction'] = 'down'
        with self.assertRaises(ValueError):
            parse_summary(json.dumps(body))

    def test_weighted_rate_not_mean_of_percentages(self):
        rows = [parse_summary(sample(n=10, y=1)), parse_summary(sample(n=100, y=1, timestamp=1788942915))]
        status, f = analyze(rows)
        self.assertEqual(status, "collecting")
        self.assertAlmostEqual(f["observed_rate_22_pct"], 200 / 110)
        self.assertNotIn("adjusted_odds_ratio", f)

    def test_model_recovers_known_positive_association(self):
        status, f = analyze(synthetic_rows())
        self.assertEqual(status, "exploratory")
        self.assertEqual(f["model_ready"], 1)
        self.assertGreater(f["odds_ratio_ci_low"], 1)
        self.assertAlmostEqual(f["adjusted_odds_ratio"], (0.15/0.85)/(0.05/0.95), delta=0.4)
        self.assertGreater(f["fitted_difference_pp"], 8)

    def test_one_mode_never_fits(self):
        status, f = analyze([r for r in synthetic_rows() if r['mode'] == 0])
        self.assertEqual(status, "collecting")
        self.assertEqual(f['model_ready'], 0)

    def test_no_events_never_fits(self):
        rows = synthetic_rows()
        for r in rows:
            r['y'] = 0
        status, _ = analyze(rows)
        self.assertEqual(status, "insufficient_events")

    def test_persistent_dedup_and_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'history.sqlite3'
            db = open_db(path)
            row = parse_summary(sample())
            stage(db, row)
            self.assertEqual(stage(db, row), 'duplicate')
            conflict = dict(row, y=9)
            with self.assertRaises(ValueError):
                stage(db, conflict)
            with patch('ml.lane_effect_worker.write_to_influx', side_effect=RuntimeError('offline')):
                with self.assertRaises(RuntimeError):
                    flush(db, {})
            db.close()
            db = open_db(path)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM outbox').fetchone()[0], 2)
            with patch('ml.lane_effect_worker.write_to_influx') as writer:
                self.assertEqual(flush(db, {'INFLUX_BUCKET':'mini_project'}), 2)
                self.assertIn('traffic_lane_observed_6610301004', writer.call_args.args[0])
            self.assertEqual(db.execute('SELECT COUNT(*) FROM outbox').fetchone()[0], 0)
            self.assertEqual(stage(db, row), 'duplicate')
            db.close()

    def test_late_record_does_not_rewrite_later_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = open_db(Path(tmp) / 'history.sqlite3')
            stage(db, parse_summary(sample(timestamp=1788943455)))
            stage(db, parse_summary(sample(timestamp=1788942855)))
            self.assertEqual(db.execute('SELECT COUNT(*) FROM outbox').fetchone()[0], 3)
            self.assertEqual(db.execute('SELECT ts FROM fits').fetchone()[0], 1788943455)
            db.close()


if __name__ == '__main__':
    unittest.main()
