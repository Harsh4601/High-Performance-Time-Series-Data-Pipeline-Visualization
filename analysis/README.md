# Analysis add-on (Options B + C)

This folder contains a **low-effort analysis script** that expands the project from “visualization” to a simple, reportable **study**:

- **Option B (primary)**: *Behavior/state segmentation* using k-means on interpretable features (speed, turn rate, vertical rate, etc.)
- **Option C**: *Trajectory geometry metrics* (distance, tortuosity, turn-angle distribution, per-state summaries)

## How to run

From the project root:

```bash
python3 analysis/flight_states.py --input detailed.csv --output-dir analysis_outputs --k auto
```

If you want a fixed number of states:

```bash
python3 analysis/flight_states.py --input detailed.csv --output-dir analysis_outputs --k 4
```

## Outputs

The script writes to `analysis_outputs/`:

- `detailed_labeled.csv`: original rows plus derived features and a `state` column
- `state_summary.csv`: per-state time/distance fractions + feature means
- `trajectory_metrics.json`: Option C metrics (distance, tortuosity, etc.)
- `plots/`:
  - `state_timeline.png`
  - `state_feature_scatter.png`
  - `turn_angle_hist.png`
  - `map_state_scatter.png`

## Notes

- `detailed.csv` longitudes are in **0–360** degrees; the script wraps them into **−180..180** before computing distances/headings.
- The dataset is treated as roughly **1 Hz**; smoothing uses a rolling window of samples (`--smooth-window`, default 15).

