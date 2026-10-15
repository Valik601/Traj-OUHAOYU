"""Focused tests for direct-adjacency reconstruction and continuity."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import (  # noqa: E402
    REQUIRED_COLUMNS,
    add_continuous_segment_ids,
    reconstruct_adjacent_pairs,
)


def _row(
    vehicle_id: int,
    frame_id: int,
    lane_id: int,
    y: float,
    *,
    vehicle_class: int = 2,
    length: float = 4.0,
    speed: float = 10.0,
) -> dict[str, float | int]:
    time_s = (frame_id - 1) * 0.1
    return {
        "vehicle_id": vehicle_id,
        "frame_id": frame_id,
        "time_s": time_s,
        "global_time_ms": 1_113_433_200_000 + int(round(time_s * 1000)),
        "local_x_m": float(lane_id * 3.5),
        "local_y_m": y,
        "vehicle_length_m": length,
        "vehicle_class": vehicle_class,
        "speed_m_s": speed,
        "acceleration_m_s2": 0.0,
        "lane_id": lane_id,
    }


def _frame(rows: list[dict[str, float | int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def test_adjacency_is_built_before_passenger_class_filtering() -> None:
    observations = _frame(
        [
            _row(10, 1, 1, 0.0, vehicle_class=2, length=4.0),
            _row(20, 1, 1, 10.0, vehicle_class=3, length=6.0),
            _row(30, 1, 1, 30.0, vehicle_class=2, length=5.0),
        ]
    )

    pairs, _ = reconstruct_adjacent_pairs(
        observations, minimum_gap_m=0.0, maximum_gap_m=100.0
    )

    assert list(zip(pairs["leader_id"], pairs["follower_id"])) == [(20, 10), (30, 20)]
    assert not ((pairs["leader_id"] == 30) & (pairs["follower_id"] == 10)).any()
    passenger_pairs = pairs.loc[
        pairs["leader_class"].eq(2) & pairs["follower_class"].eq(2)
    ]
    assert passenger_pairs.empty


def test_net_gap_subtracts_leader_length_not_follower_length() -> None:
    observations = _frame(
        [
            _row(1, 1, 1, 100.0, length=9.0),
            _row(2, 1, 1, 125.0, length=4.5),
        ]
    )

    pairs, _ = reconstruct_adjacent_pairs(
        observations, minimum_gap_m=0.0, maximum_gap_m=100.0
    )

    assert len(pairs) == 1
    assert pairs.iloc[0]["leader_id"] == 2
    assert pairs.iloc[0]["follower_id"] == 1
    assert np.isclose(pairs.iloc[0]["gap_m"], 125.0 - 100.0 - 4.5)


def test_pairing_never_crosses_lane_or_frame_boundaries() -> None:
    observations = _frame(
        [
            _row(1, 1, 1, 0.0),
            _row(2, 1, 2, 10.0),
            _row(3, 2, 1, 20.0),
            _row(4, 2, 1, 40.0),
        ]
    )

    pairs, _ = reconstruct_adjacent_pairs(
        observations, minimum_gap_m=0.0, maximum_gap_m=100.0
    )

    assert len(pairs) == 1
    assert (pairs.iloc[0]["frame_id"], pairs.iloc[0]["lane_id"]) == (2, 1)
    assert (pairs.iloc[0]["leader_id"], pairs.iloc[0]["follower_id"]) == (4, 3)


def test_tied_positions_exclude_the_whole_ambiguous_frame_lane_group() -> None:
    observations = _frame(
        [
            _row(1, 1, 1, 0.0),
            _row(2, 1, 1, 10.0),
            _row(3, 1, 1, 10.0),
            _row(4, 1, 1, 30.0),
            _row(5, 1, 2, 0.0),
            _row(6, 1, 2, 20.0),
        ]
    )

    pairs, statistics = reconstruct_adjacent_pairs(
        observations, minimum_gap_m=0.0, maximum_gap_m=100.0
    )

    assert set(pairs["lane_id"]) == {2}
    assert statistics["ambiguous_frame_lane_groups_excluded"] == 1
    assert statistics["rows_in_ambiguous_groups_excluded"] == 4


def test_continuity_label_splits_missing_frames_and_lane_changes() -> None:
    observations = _frame(
        [
            _row(1, 1, 1, 0.0),
            _row(2, 1, 1, 20.0),
            _row(1, 2, 1, 1.0),
            _row(2, 2, 1, 21.0),
            _row(1, 4, 1, 3.0),
            _row(2, 4, 1, 23.0),
            _row(1, 5, 2, 4.0),
            _row(2, 5, 2, 24.0),
        ]
    )
    # Correct the synthetic frame-based timestamps so frame 4 follows the source grid.
    observations["time_s"] = (observations["frame_id"] - 1) * 0.1
    observations["global_time_ms"] = (
        1_113_433_200_000 + np.rint(observations["time_s"] * 1000).astype(int)
    )
    pairs, _ = reconstruct_adjacent_pairs(
        observations, minimum_gap_m=0.0, maximum_gap_m=100.0
    )

    segmented = add_continuous_segment_ids(pairs, sampling_interval_s=0.1)
    segment_sizes = sorted(segmented.groupby("segment_id").size().tolist())

    assert segment_sizes == [1, 1, 2]

