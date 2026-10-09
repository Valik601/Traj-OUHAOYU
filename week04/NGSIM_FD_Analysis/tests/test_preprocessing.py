from __future__ import annotations

import unittest

import pandas as pd

from src.preprocessing import preprocess_lane


class TestPreprocessing(unittest.TestCase):
    def test_lane_change_splits_target_lane_runs(self) -> None:
        frame = pd.DataFrame(
            {
                "vehicle_id": [1, 1, 1, 1, 1], "frame_id": [1, 2, 3, 4, 5],
                "time_s": [0.0, 0.1, 0.2, 0.3, 0.4],
                "global_time_ms": [0, 100, 200, 300, 400],
                "x_m": [1.0] * 5, "s_m": [0.0, 1.0, 2.0, 3.0, 4.0],
                "length_m": [4.0] * 5, "vehicle_class": [2] * 5,
                "speed_m_s": [10.0] * 5, "accel_m_s2": [0.0] * 5,
                "lane_id": [3, 3, 2, 3, 3],
            }
        )
        config = {
            "preprocessing": {
                "max_time_gap_s": 0.15, "min_speed_m_s": 0.0, "max_speed_m_s": 50.0,
                "max_abs_accel_m_s2": 10.0, "max_backward_jump_m": 0.5,
                "max_implied_speed_m_s": 60.0,
                "smoothing": {"enabled": False, "window_length": 5, "polyorder": 2},
            }
        }
        cleaned, summary = preprocess_lane(frame, 3, config)
        self.assertEqual(summary["continuous_segments"], 2)
        self.assertEqual(len(cleaned), 4)


if __name__ == "__main__":
    unittest.main()
