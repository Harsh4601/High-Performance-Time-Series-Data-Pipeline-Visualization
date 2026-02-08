"""
Flight behavior segmentation + trajectory metrics (Options B + C).

This script is intentionally "one-command" to minimize effort:
1) Reads a single-flight CSV (e.g., detailed.csv)
2) Computes GPS-derived kinematic features (speed, turn rate, vertical rate, etc.)
3) Clusters points into a small number of behavioral states (k-means)
4) Exports:
   - labeled per-row CSV with `state` and derived features
   - per-state summary table
   - trajectory geometry metrics (distance, tortuosity, turn stats)
   - a few plots you can paste into the report

Non-trivial math (bearing, turn wrapping, haversine) is commented inline.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrajectoryMetrics:
    n_rows_raw: int
    n_rows_used: int
    start_time: str
    end_time: str
    duration_s: float
    total_distance_m: float
    net_displacement_m: float
    tortuosity: float
    mean_speed_mps: float
    p95_speed_mps: float
    max_speed_mps: float
    mean_vertical_rate_mps: float
    max_abs_vertical_rate_mps: float
    mean_turn_rate_radps: float
    p95_turn_rate_radps: float


def lon_360_to_180(lon_deg: np.ndarray) -> np.ndarray:
    """
    Convert longitudes from [0, 360) into (-180, 180].

    Your detailed.csv uses ~320 degrees, which corresponds to -40 degrees.
    """
    # ((lon + 180) % 360) - 180 wraps into [-180, 180)
    return ((lon_deg + 180.0) % 360.0) - 180.0


def haversine_m(lat1_deg: np.ndarray, lon1_deg: np.ndarray, lat2_deg: np.ndarray, lon2_deg: np.ndarray) -> np.ndarray:
    """
    Vectorized haversine distance in meters.

    Haversine computes the great-circle distance on a sphere:
      a = sin²(dlat/2) + cos(lat1)cos(lat2)sin²(dlon/2)
      c = 2 atan2(sqrt(a), sqrt(1-a))
      d = R * c
    """
    R = 6371000.0  # meters
    lat1 = np.deg2rad(lat1_deg)
    lon1 = np.deg2rad(lon1_deg)
    lat2 = np.deg2rad(lat2_deg)
    lon2 = np.deg2rad(lon2_deg)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def bearing_rad(lat1_deg: np.ndarray, lon1_deg: np.ndarray, lat2_deg: np.ndarray, lon2_deg: np.ndarray) -> np.ndarray:
    """
    Initial bearing (forward azimuth) from point1 -> point2 in radians.

    Formula:
      y = sin(dlon) * cos(lat2)
      x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
      bearing = atan2(y, x)
    """
    lat1 = np.deg2rad(lat1_deg)
    lon1 = np.deg2rad(lon1_deg)
    lat2 = np.deg2rad(lat2_deg)
    lon2 = np.deg2rad(lon2_deg)

    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)


def wrap_angle_pi(angle_rad: np.ndarray) -> np.ndarray:
    """
    Wrap angle into [-pi, pi].

    This is important for turn calculations: e.g., +179° to -179° is a 2° turn, not 358°.
    """
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def safe_div(numer: np.ndarray, denom: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return numer / np.maximum(denom, eps)


def rolling_features(series: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
    """
    Rolling mean + rolling std using a fixed sample window.
    We keep this simple because your data is roughly 1 Hz.
    """
    r = series.rolling(window=window, min_periods=max(3, window // 3))
    return r.mean(), r.std(ddof=0)


def build_features(df: pd.DataFrame, smooth_window: int) -> pd.DataFrame:
    """
    Build per-row features for clustering (Option B).
    Also returns raw kinematics useful for Option C.
    """
    out = df.copy()

    # Longitude wrapping: your CSV uses 0..360 degrees.
    out["lon_wrapped"] = lon_360_to_180(out["lon"].to_numpy(dtype=float))

    # Time deltas (seconds). Non-positive dt is invalid for kinematic rates.
    t = pd.to_datetime(out["datetime"], errors="coerce", utc=False)
    out["datetime_parsed"] = t
    out = out.sort_values("datetime_parsed").reset_index(drop=True)
    dt_s = out["datetime_parsed"].diff().dt.total_seconds().to_numpy(dtype=float)
    out["dt_s"] = dt_s

    lat = out["lat"].to_numpy(dtype=float)
    lon = out["lon_wrapped"].to_numpy(dtype=float)
    alt = out["altitude"].to_numpy(dtype=float)

    # Distances between consecutive GPS points
    dist_m = np.full(len(out), np.nan, dtype=float)
    dist_m[1:] = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    out["dist_m"] = dist_m

    # Ground speed (m/s)
    speed_mps = np.full(len(out), np.nan, dtype=float)
    speed_mps[1:] = safe_div(dist_m[1:], dt_s[1:])
    out["speed_mps"] = speed_mps

    # Vertical rate (m/s)
    dalt = np.full(len(out), np.nan, dtype=float)
    dalt[1:] = alt[1:] - alt[:-1]
    out["vertical_rate_mps"] = np.where(np.isfinite(dt_s) & (dt_s > 0), safe_div(dalt, dt_s), np.nan)

    # GPS-derived bearing and turn rate
    brng = np.full(len(out), np.nan, dtype=float)
    brng[1:] = bearing_rad(lat[:-1], lon[:-1], lat[1:], lon[1:])
    out["bearing_rad"] = brng

    dbrng = np.full(len(out), np.nan, dtype=float)
    dbrng[2:] = wrap_angle_pi(brng[2:] - brng[1:-1])
    out["turn_rate_radps"] = np.where(np.isfinite(dt_s) & (dt_s > 0), np.abs(safe_div(dbrng, dt_s)), np.nan)

    # Optional IMU proxy: acceleration magnitude (in "g" units per your CSV)
    ax = out["Ax"].to_numpy(dtype=float)
    ay = out["Ay"].to_numpy(dtype=float)
    az = out["Az"].to_numpy(dtype=float)
    out["accel_mag_g"] = np.sqrt(ax * ax + ay * ay + az * az)

    # Smoothing / aggregation for clustering stability
    speed_mean, _ = rolling_features(out["speed_mps"], smooth_window)
    turn_mean, _ = rolling_features(out["turn_rate_radps"], smooth_window)
    vmean, _ = rolling_features(out["vertical_rate_mps"], smooth_window)
    amag_mean, amag_std = rolling_features(out["accel_mag_g"], smooth_window)

    out["speed_mps_smooth"] = speed_mean
    out["turn_rate_radps_smooth"] = turn_mean
    out["vertical_rate_mps_smooth"] = vmean
    out["accel_mag_g_smooth"] = amag_mean
    out["accel_mag_g_std"] = amag_std

    # Features used for clustering (keep interpretable + robust)
    features = out[
        [
            "speed_mps_smooth",
            "turn_rate_radps_smooth",
            "vertical_rate_mps_smooth",
            "accel_mag_g_smooth",
            "accel_mag_g_std",
            "altitude",
        ]
    ].copy()

    return out.join(features.add_prefix("feat_"))


def choose_k_auto(X: np.ndarray, k_min: int, k_max: int, sample_n: int, random_state: int) -> int:
    """
    Pick k by silhouette score on a subsample to keep runtime low.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = X.shape[0]
    if n <= k_max:
        sample_idx = np.arange(n)
    else:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(n, size=min(sample_n, n), replace=False)

    Xs = X[sample_idx]
    best_k = k_min
    best_score = -1.0

    for k in range(k_min, k_max + 1):
        # n_init='auto' is supported in recent sklearn; if older, it may require an int.
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(Xs)

        # Silhouette is undefined if a cluster gets only 1 point; handle gracefully.
        try:
            score = float(silhouette_score(Xs, labels))
        except Exception:
            score = -1.0

        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def clip_outliers_by_quantile(X: np.ndarray, q_low: float = 0.01, q_high: float = 0.99) -> np.ndarray:
    """
    Simple robustification: clip features to [q1%, q99%] per column.
    This prevents rare GPS glitches from dominating k-means centroids.
    """
    lo = np.nanquantile(X, q_low, axis=0)
    hi = np.nanquantile(X, q_high, axis=0)
    return np.clip(X, lo, hi)


def compute_metrics(df_feat: pd.DataFrame, n_rows_raw: int) -> TrajectoryMetrics:
    used = df_feat.dropna(subset=["datetime_parsed", "dt_s", "dist_m"]).copy()
    used = used[used["dt_s"] > 0].copy()

    duration_s = float((used["datetime_parsed"].iloc[-1] - used["datetime_parsed"].iloc[0]).total_seconds())
    total_distance_m = float(used["dist_m"].sum(skipna=True))

    # Net displacement: start->end great-circle distance
    lat = used["lat"].to_numpy(dtype=float)
    lon = used["lon_wrapped"].to_numpy(dtype=float)
    net_disp = float(haversine_m(lat[:1], lon[:1], lat[-1:], lon[-1:])[0])
    tort = float(total_distance_m / max(net_disp, 1e-9))

    speed = used["speed_mps"].to_numpy(dtype=float)
    v = used["vertical_rate_mps"].to_numpy(dtype=float)
    turn = used["turn_rate_radps"].to_numpy(dtype=float)

    return TrajectoryMetrics(
        n_rows_raw=int(n_rows_raw),
        n_rows_used=int(len(used)),
        start_time=str(used["datetime_parsed"].iloc[0]),
        end_time=str(used["datetime_parsed"].iloc[-1]),
        duration_s=duration_s,
        total_distance_m=total_distance_m,
        net_displacement_m=net_disp,
        tortuosity=tort,
        mean_speed_mps=float(np.nanmean(speed)),
        p95_speed_mps=float(np.nanquantile(speed, 0.95)),
        max_speed_mps=float(np.nanmax(speed)),
        mean_vertical_rate_mps=float(np.nanmean(v)),
        max_abs_vertical_rate_mps=float(np.nanmax(np.abs(v))),
        mean_turn_rate_radps=float(np.nanmean(turn)),
        p95_turn_rate_radps=float(np.nanquantile(turn, 0.95)),
    )


def make_plots(df: pd.DataFrame, out_dir: Path, max_points_scatter: int = 20000, random_state: int = 42) -> None:
    """
    Generate a small set of report-ready plots.
    """
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # Downsample to keep plotting fast
    rng = np.random.default_rng(random_state)
    n = len(df)
    if n > max_points_scatter:
        idx = np.sort(rng.choice(n, size=max_points_scatter, replace=False))
        dplot = df.iloc[idx].copy()
    else:
        dplot = df

    # 1) State timeline
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df["datetime_parsed"], df["state"], lw=0.8)
    ax.set_title("Behavior state over time (k-means)")
    ax.set_xlabel("Time")
    ax.set_ylabel("State (integer id)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "state_timeline.png", dpi=160)
    plt.close(fig)

    # 2) Speed vs turn-rate scatter (colored by state)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        dplot["speed_mps_smooth"],
        dplot["turn_rate_radps_smooth"],
        c=dplot["state"],
        s=6,
        alpha=0.6,
        cmap="tab10",
    )
    ax.set_title("State clusters in feature space")
    ax.set_xlabel("Speed (m/s) [smoothed]")
    ax.set_ylabel("Turn rate (rad/s) [smoothed]")
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label="state")
    fig.tight_layout()
    fig.savefig(out_dir / "state_feature_scatter.png", dpi=160)
    plt.close(fig)

    # 3) Turn angle histogram (absolute delta bearing in degrees)
    # d_bearing uses wrapped differences, so it reflects true small turns.
    d_bearing = np.abs(wrap_angle_pi(df["bearing_rad"].diff().to_numpy(dtype=float)))
    d_deg = np.rad2deg(d_bearing[np.isfinite(d_bearing)])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(d_deg, bins=60, color="#2563eb", alpha=0.85)
    ax.set_title("Turn angle distribution (degrees per sample)")
    ax.set_xlabel("Absolute turn angle (deg)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "turn_angle_hist.png", dpi=160)
    plt.close(fig)

    # 4) Map-view scatter colored by state (quick sanity check)
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(dplot["lon_wrapped"], dplot["lat"], c=dplot["state"], s=5, alpha=0.65, cmap="tab10")
    ax.set_title("Geographic track colored by state (downsampled)")
    ax.set_xlabel("Longitude (deg, wrapped)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="state")
    fig.tight_layout()
    fig.savefig(out_dir / "map_state_scatter.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Flight segmentation (Option B) + trajectory metrics (Option C).")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "detailed.csv"),
        help="Path to input CSV (default: project_root/detailed.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "analysis_outputs"),
        help="Directory to write outputs (default: project_root/analysis_outputs).",
    )
    parser.add_argument("--k", type=str, default="auto", help='Number of states (e.g., "4") or "auto".')
    parser.add_argument("--k-min", type=int, default=3, help="Min k if --k auto.")
    parser.add_argument("--k-max", type=int, default=6, help="Max k if --k auto.")
    parser.add_argument("--smooth-window", type=int, default=15, help="Rolling window (samples) for smoothed features.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    n_rows_raw = len(df)

    df_feat = build_features(df, smooth_window=args.smooth_window)

    # Prepare feature matrix for clustering
    feat_cols = [c for c in df_feat.columns if c.startswith("feat_")]
    X = df_feat[feat_cols].to_numpy(dtype=float)
    X = clip_outliers_by_quantile(X, 0.01, 0.99)

    # Fill NaNs with column medians (simple + effective)
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(col_medians, inds[1])

    # Standardize: k-means is sensitive to feature scaling
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Choose k (auto or fixed)
    if args.k.strip().lower() == "auto":
        k = choose_k_auto(
            Xs,
            k_min=max(2, args.k_min),
            k_max=max(args.k_min, args.k_max),
            sample_n=50000,
            random_state=args.random_state,
        )
    else:
        k = int(args.k)

    # Cluster
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=args.random_state, n_init="auto")
    labels = km.fit_predict(Xs)
    df_feat["state"] = labels.astype(int)

    # Make state IDs more interpretable: reorder by mean speed (slow -> fast)
    state_order = (
        df_feat.groupby("state", as_index=True)["speed_mps_smooth"]
        .mean()
        .sort_values()
        .index.to_list()
    )
    remap = {old: new for new, old in enumerate(state_order)}
    df_feat["state"] = df_feat["state"].map(remap).astype(int)

    # Metrics + per-state summaries (Option C)
    metrics = compute_metrics(df_feat, n_rows_raw=n_rows_raw)

    per_state = (
        df_feat.dropna(subset=["dt_s", "dist_m"])
        .assign(dt_s=lambda d: d["dt_s"].where(d["dt_s"] > 0))
        .groupby("state", as_index=False)
        .agg(
            n_points=("state", "size"),
            time_s=("dt_s", "sum"),
            distance_m=("dist_m", "sum"),
            mean_speed_mps=("speed_mps", "mean"),
            p95_speed_mps=("speed_mps", lambda s: float(np.nanquantile(s.to_numpy(dtype=float), 0.95))),
            mean_turn_rate_radps=("turn_rate_radps", "mean"),
            mean_vertical_rate_mps=("vertical_rate_mps", "mean"),
            mean_altitude=("altitude", "mean"),
        )
    )
    total_time = float(per_state["time_s"].sum())
    total_dist = float(per_state["distance_m"].sum())
    per_state["time_frac"] = per_state["time_s"] / max(total_time, 1e-9)
    per_state["distance_frac"] = per_state["distance_m"] / max(total_dist, 1e-9)

    # Write outputs
    labeled_csv = out_dir / "detailed_labeled.csv"
    # Keep original columns plus derived ones; drop intermediate parsed datetime duplicate if desired
    df_feat.to_csv(labeled_csv, index=False)

    per_state_csv = out_dir / "state_summary.csv"
    per_state.to_csv(per_state_csv, index=False)

    metrics_json = out_dir / "trajectory_metrics.json"
    metrics_json.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")

    run_info = {
        "input": str(in_path),
        "output_dir": str(out_dir),
        "k": int(k),
        "smooth_window": int(args.smooth_window),
        "random_state": int(args.random_state),
        "feature_columns": feat_cols,
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    # Plots
    make_plots(df_feat.dropna(subset=["datetime_parsed"]), out_dir / "plots", random_state=args.random_state)

    print("Wrote:")
    print(f"- {labeled_csv}")
    print(f"- {per_state_csv}")
    print(f"- {metrics_json}")
    print(f"- {out_dir / 'plots'} (png figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

