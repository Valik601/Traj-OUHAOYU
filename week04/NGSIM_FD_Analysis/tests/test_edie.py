from __future__ import annotations

import unittest

import numpy as np

from src.edie_calculation import edie_state, parallelogram_polygon, rectangle_polygon
from src.trajectory_processing import TrajectorySegments


def one_segment(t0: float, t1: float, s0: float, s1: float) -> TrajectorySegments:
    return TrajectorySegments(
        t0=np.array([t0]), t1=np.array([t1]), s0=np.array([s0]), s1=np.array([s1]),
        speed0=np.array([(s1 - s0) / (t1 - t0)]),
        speed1=np.array([(s1 - s0) / (t1 - t0)]),
        accel=np.array([0.0]), vehicle_id=np.array([1]), segment_id=np.array([1]),
    )


class TestEdieCalculation(unittest.TestCase):
    def state(self, segment: TrajectorySegments, polygon: np.ndarray, **window: float) -> dict:
        return edie_state(
            segment, polygon, min_vehicles=1, minimum_occupied_time_fraction=0.0, **window
        )

    def test_full_rectangular_intersection(self) -> None:
        segment = one_segment(0.0, 10.0, 0.0, 100.0)
        polygon = rectangle_polygon(0.0, 0.0, 10.0, 100.0)
        result = self.state(
            segment, polygon, t_start=0.0, s_start=0.0, duration_s=10.0,
            length_m=100.0, wave_speed_m_s=0.0,
        )
        self.assertAlmostEqual(result["distance_sum_m"], 100.0)
        self.assertAlmostEqual(result["time_sum_s"], 10.0)
        self.assertAlmostEqual(result["flow_veh_h"], 360.0)
        self.assertAlmostEqual(result["density_veh_km"], 10.0)
        self.assertAlmostEqual(result["speed_km_h"], 36.0)
        self.assertLess(result["closure_relative_error"], 1e-12)

    def test_boundary_interpolation(self) -> None:
        segment = one_segment(0.0, 10.0, 0.0, 100.0)
        polygon = rectangle_polygon(0.0, 25.0, 10.0, 50.0)
        result = self.state(
            segment, polygon, t_start=0.0, s_start=25.0, duration_s=10.0,
            length_m=50.0, wave_speed_m_s=0.0,
        )
        self.assertAlmostEqual(result["distance_sum_m"], 50.0)
        self.assertAlmostEqual(result["time_sum_s"], 5.0)
        self.assertLess(result["closure_relative_error"], 1e-12)

    def test_parallelogram_physical_closure(self) -> None:
        segment = one_segment(0.0, 10.0, 50.0, 100.0)
        polygon = parallelogram_polygon(0.0, 0.0, 10.0, 120.0, -2.0)
        result = self.state(
            segment, polygon, t_start=0.0, s_start=0.0, duration_s=10.0,
            length_m=120.0, wave_speed_m_s=-2.0,
        )
        self.assertTrue(result["valid"])
        self.assertLess(result["closure_relative_error"], 1e-12)


if __name__ == "__main__":
    unittest.main()

