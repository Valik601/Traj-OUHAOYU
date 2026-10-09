# NGSIM I-80 Empirical Fundamental Diagrams

This project builds lane-specific empirical fundamental diagrams from continuous NGSIM trajectories. It compares a fixed rectangular Edie baseline, a wave-aligned parallelogram baseline, and an exploratory adaptive window method. Every reported traffic state is computed from exact piecewise-linear trajectory/window intersections.

## Environment

- Python 3.11
- Windows, Linux, or macOS
- No paid API or network service is required to run the analysis

Install the recorded environment:

```bash
python -m pip install -r requirements.txt
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## Input data

The supplied classroom CSV is already in SI units and uses these fields:

| Canonical field | Supplied field | Unit |
|---|---|---|
| vehicle_id | vehicle_id | identifier |
| frame_id | frame_id | frame |
| time_s | time_s | s |
| global_time_ms | global_time_ms | ms |
| x_m | local_x_m | m |
| s_m | local_y_m | m |
| length_m | vehicle_length_m | m |
| vehicle_class | vehicle_class | code |
| speed_m_s | speed_m_s | m/s |
| accel_m_s2 | acceleration_m_s2 | m/s² |
| lane_id | lane_id | code |

Edit `input.field_map` in `config.yaml` for a different header. Set `input.source_units: ngsim_imperial` for a raw NGSIM-style file in feet, ft/s, and ft/s². The longitudinal coordinate is never reset per vehicle.

## Reproduction

From this directory:

```bash
python main.py --input ../datasets/NGSIM-I80/NGSIM_I80_20050413_1600-1615_SI.csv --lane 3
```

Optional overrides:

```bash
python main.py --input PATH_TO_CSV --config config.yaml --lane 3 --output-root results
```

The program prints progress for all 12 execution stages. All window sizes, wave-speed candidates, adaptive-score weights, thresholds, split fractions, and output resolution are centralized in `config.yaml`.

## Processing rules

1. Validate the mapped schema and convert it to a common metre-second representation.
2. Remove only exact duplicate primary keys. Conflicting duplicate keys stop the run.
3. Report and remove rows missing required fields.
4. Flag configured speed, acceleration, backward-motion, time-step, and longitudinal-jump anomalies.
5. Split trajectories at time gaps and lane changes. A vehicle that changes lane is not discarded wholesale.
6. Preserve raw and cleaned position, speed, and acceleration columns. Savitzky-Golay smoothing is available but disabled by default.
7. Keep only continuous target-lane runs with at least two points.

Every removal or correction count is saved in `results/processed/preprocessing_summary.json`.

## Traffic-state methods

### Rectangular baseline

For each fixed rectangle, the code clips every piecewise-linear trajectory segment at all four boundaries. It sums the exact distance and time inside the polygon and applies Edie's generalized definitions.

### Wave-aligned parallelogram baseline

The code estimates an upstream wave-speed candidate by lagged correlation of a gridded congested speed field. A manual value can be supplied in `config.yaml`. The implemented polygon follows the task-specified boundaries

```text
s0 + w(t-t0) <= s <= s0 + L + w(t-t0).
```

This is a simplified implementation inspired by He and Wu (2025), not a full reproduction. Their paper also uses a target traffic speed, CV/NAE scoring, extensive candidate generation, and non-overlap selection.

### Exploratory adaptive method

At each common anchor, the algorithm searches the configured spatial lengths, temporal lengths, and rectangular/wave-aligned orientations. It minimizes an explicitly configurable score combining duration-weighted speed CV, first/second-half speed change, mean acceleration, and spatial speed heterogeneity. Minimum vehicle coverage and a scale penalty prevent selecting arbitrarily small low-variance windows.

The score is a research hypothesis, not an accepted equilibrium-state definition. All selected samples remain labeled as quasi-stationary, acceleration, deceleration, transient, or insufficient coverage; difficult states are not silently deleted.

## Evaluation

- `q = k v` is checked for every valid state.
- Greenshields is fitted only as a classical reference curve.
- The earliest 70% of common time blocks form the training set; the final 30% form the test set.
- Test MAE and RMSE assess held-out flow reconstruction.
- Capacity stability is the coefficient of variation across contiguous time-block fits.
- Parameter sensitivity covers all 100/150/200 m and 10/20/30 s combinations.
- Sampling sensitivity reruns all methods at 0.1, 0.2, and 0.5 s trajectory intervals.
- Sample coverage and free/congested branch counts remain visible.

Lower scatter or better fit does not by itself prove more accurate state extraction.

## Outputs

```text
results/
├── processed/
│   ├── lane_3_cleaned_trajectories.csv
│   ├── common_window_anchors.csv
│   └── preprocessing_summary.json
├── states/
│   ├── rectangular_states.csv
│   ├── parallelogram_states.csv
│   ├── adaptive_states.csv
│   ├── adaptive_candidate_windows.csv
│   └── *_parameter_sweep.csv
├── metrics/
│   ├── evaluation_metrics.csv
│   ├── evaluation_details.json
│   ├── adaptive_state_class_summary.csv
│   ├── parameter_sensitivity.csv
│   ├── sampling_sensitivity.csv
│   └── wave_speed_*.csv/json
├── figures/
│   ├── NGSIM_I80_Fundamental_Diagram_Report.pdf
│   ├── NGSIM_I80_Fundamental_Diagram_Report.png
│   ├── NGSIM_I80_Fundamental_Diagram_Report.manifest.json
│   └── {Rectangular,Parallelogram,Adaptive}_Fundamental_Diagrams.png
└── run_summary.json
```

The PDF is a single A3 landscape page. The PNG is exported at 300 DPI. Raw empirical states remain available as CSV files behind every plotted point.

## Limitations

- The empirical result represents one I-80 lane and one 15-minute observation period.
- The wave-speed estimator is a lagged gridded-field diagnostic, not a unique physical-wave estimator.
- Adaptive selection is exploratory and can introduce selection bias even when all state classes are retained.
- NGSIM measurement error and potential smoothing sensitivity remain relevant. Smoothing is disabled in the reported default run.
- Greenshields model error is not a direct measure of traffic-state extraction accuracy.

## References

- Edie, L. C. (1965). Discussion of traffic stream measurements and definitions. *Proceedings of the Second International Symposium on the Theory of Traffic Flow*, 139-154.
- He, Z., & Wu, C. (2025). Constructing the fundamental diagrams of traffic flow from large-scale vehicle trajectory data. arXiv:2507.09648v2. https://doi.org/10.48550/arXiv.2507.09648
- Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
