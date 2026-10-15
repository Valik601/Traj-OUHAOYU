"""STEP 1: reconstruct direct same-lane car-following relationships."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import (
    configure_logging,
    default_config_path,
    ensure_output_dir,
    load_config,
    read_ngsim_csv,
    reconstruct_adjacent_pairs,
    require_config_section,
    write_csv_atomic,
    write_json_atomic,
)


LOGGER = configure_logging("find_following_pairs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct direct leaders/followers from all vehicle classes before "
            "any passenger-car filtering."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(__file__),
        help="YAML configuration path (default: project config.yaml)",
    )
    return parser.parse_args()


def find_pairs(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["_paths"]
    pairing_config = require_config_section(config, "pairing")
    output_dir = ensure_output_dir(config, "pairs_dir")

    observations = read_ngsim_csv(paths["si_csv"])
    frame_mapping = observations.groupby("frame_id", observed=True).agg(
        time_values=("time_s", "nunique"),
        global_time_values=("global_time_ms", "nunique"),
    )
    ambiguous_frame_times = frame_mapping[
        frame_mapping["time_values"].gt(1)
        | frame_mapping["global_time_values"].gt(1)
    ]
    if not ambiguous_frame_times.empty:
        raise ValueError(
            f"{len(ambiguous_frame_times)} frame IDs map to multiple timestamps; "
            "cannot reconstruct same-time adjacency safely"
        )

    minimum_gap = float(pairing_config.get("minimum_gap_m", 0.0))
    maximum_gap = float(pairing_config.get("maximum_gap_m", 150.0))
    LOGGER.info(
        "Reconstructing adjacency from all %d records and all vehicle classes",
        len(observations),
    )
    pairs, statistics = reconstruct_adjacent_pairs(
        observations,
        minimum_gap_m=minimum_gap,
        maximum_gap_m=maximum_gap,
    )

    # This filter is deliberately computed only after adjacency has been fixed.
    passenger_class = int(
        config.get("selection", {}).get("passenger_vehicle_class", 2)
    )
    both_passenger = pairs["leader_class"].eq(passenger_class) & pairs[
        "follower_class"
    ].eq(passenger_class)
    statistics.update(
        {
            "passenger_vehicle_class": passenger_class,
            "both_passenger_pairs_after_adjacency": int(both_passenger.sum()),
            "non_passenger_or_mixed_pairs_retained_for_audit": int(
                (~both_passenger).sum()
            ),
        }
    )
    output_path = output_dir / "frame_pairs.csv"
    write_csv_atomic(pairs, output_path)

    summary = {
        "method": {
            "grouping": ["frame_id", "lane_id"],
            "ordering": "local_y_m ascending",
            "follower_definition": "lower local_y_m member of each adjacent pair",
            "leader_definition": "immediately next higher local_y_m vehicle",
            "gap_formula": "leader_y - follower_y - leader_length",
            "vehicle_class_filter_timing": (
                "No class filter was applied before adjacency reconstruction. "
                "frame_pairs.csv retains all valid classes; passenger-car filtering "
                "is performed during case selection."
            ),
        },
        "thresholds": {
            "minimum_gap_m_exclusive": minimum_gap,
            "maximum_gap_m_inclusive": maximum_gap,
        },
        "statistics": statistics,
        "output": {
            "filename": output_path.name,
            "rows": int(len(pairs)),
            "columns": list(pairs.columns),
        },
    }
    write_json_atomic(summary, output_dir / "pair_reconstruction_summary.json")
    LOGGER.info(
        "Wrote %d valid direct pairs (%d both-passenger observations)",
        len(pairs),
        both_passenger.sum(),
    )
    return summary


def main() -> None:
    args = parse_args()
    find_pairs(load_config(args.config))


if __name__ == "__main__":
    main()
