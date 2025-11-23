"""Validate alignment between event-derived pass coordinates and tracking ball positions.

Goal
-----
Use tracking (frame, ball x/y) to assess how well our current event-based
interpretation of pass origin (x_end,y_end) and reception (player_targeted_x_reception,
player_targeted_y_reception) matches the actual ball location.

Scope (Phase 1)
---------------
1. Load a small sample of matches (configurable) from event data.
2. Extract pass events (successful + unsuccessful) using existing pipeline helpers.
3. For each pass:
   - Pass release frame = frame_end  (event semantics: possession ends when ball is released)
   - Pass origin (event) = (x_end, y_end)
   - Compare with tracking ball position at frame_end after mirroring & rescaling.
4. For successful passes only:
   - Event reception = player_targeted_x_reception, player_targeted_y_reception
   - Search tracking ball positions in window [frame_end+1, frame_end+search_window]
     and find frame with minimal Euclidean distance to event reception.
   - Record distance and frame lag (# frames after release).
5. For unsuccessful passes:
   - Use intended target coordinates player_targeted_x_pass / y_pass if available
     and perform same search (diagnostic only; may be noisier).
6. Output summary statistics (origin diffs, reception diffs, lag distribution, thresholds).

Assumptions & Alignment Logic
-----------------------------
Event data coordinates are already MIRRORED so possessing team always attacks left→right (0→105m).
Tracking coordinates are raw SkillCorner pitch coordinates (approx -52..52 / -34..34) and NOT mirrored.
Therefore, for passes with attacking_side == 'right_to_left' we mirror tracking coordinates (x,y)->(-x,-y)
before rescaling so they match the event's left→right orientation.

Rescaling uses identical linear mapping as in data_processing.rescale_coordinates:
   x_rescaled = (x - X_MIN) / (X_MAX - X_MIN) * 105
   y_rescaled = (y - Y_MIN) / (Y_MAX - Y_MIN) * 68
Values outside nominal bounds are clipped.

Future Extensions (Phase 2)
---------------------------
- More robust reception frame identification (e.g., detect velocity sign change or first touch frame).
- Integrate freeze-frame data where available for validation of player positions.
- Evaluate direction & distance classification vs true tracking trajectory.
- Incorporate ball detection confidence; fallback to nearest detected frame if missing.

Usage
-----
Run directly for a quick sample:
    python validate_tracking_pass_alignment.py --matches 3 --search-window 40 --sample-size 500

Outputs printed to stdout and simple CSV (optional) for deeper analysis.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict
import sys

import numpy as np
import pandas as pd

# Ensure local src/ is importable whether run from project root or repo root
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Reuse constants from data_processing
from data_processing import (
    load_premier_league_events,
    extract_pass_events,
    rescale_coordinates,
    normalize_attack_direction,
    X_MIN,
    X_MAX,
    Y_MIN,
    Y_MAX,
)

TRACKING_DIR = Path.cwd().parents[0] / "PremierLeague_data" / "2024" / "tracking_parquets"


@dataclass
class PassAlignmentResult:
    # Required (non-default) fields
    match_id: int
    event_id: int
    period: int
    frame_release: int
    origin_event_x: float
    origin_event_y: float
    success: int
    attacking_side: Optional[str]
    pass_outcome: Optional[str]
    # Optional / computed fields with defaults after required ones
    frame_reception_found: Optional[int] = None
    frame_lag: Optional[int] = None
    origin_track_x: Optional[float] = None
    origin_track_y: Optional[float] = None
    origin_diff: Optional[float] = None
    reception_event_x: Optional[float] = None
    reception_event_y: Optional[float] = None
    reception_track_x: Optional[float] = None
    reception_track_y: Optional[float] = None
    reception_min_diff: Optional[float] = None
    next_possession_frame: Optional[int] = None
    next_possession_start_x: Optional[float] = None
    next_possession_start_y: Optional[float] = None
    next_possession_event_diff: Optional[float] = None
    next_possession_tracking_x: Optional[float] = None
    next_possession_tracking_y: Optional[float] = None
    next_possession_tracking_diff: Optional[float] = None
    # Tracking-based receiver touch detection
    player_touch_frame: Optional[int] = None
    player_touch_ball_x: Optional[float] = None
    player_touch_ball_y: Optional[float] = None
    player_touch_player_x: Optional[float] = None
    player_touch_player_y: Optional[float] = None
    player_touch_ball_player_dist: Optional[float] = None
    player_touch_event_reception_dist: Optional[float] = None


def load_tracking_match(match_id: int) -> pd.DataFrame:
    """Load tracking parquet for a match; return only ball rows with indexing by frame."""
    fpath = TRACKING_DIR / f"{match_id}.parquet"
    if not fpath.exists():
        return pd.DataFrame(columns=["frame", "x", "y", "period", "is_detected"]).set_index("frame")
    df = pd.read_parquet(fpath)
    # Keep ball only
    df_ball = df[df["is_ball"] == True].copy()
    if df_ball.empty:
        return df_ball.set_index("frame")
    # Index by frame for fast lookup
    df_ball.set_index("frame", inplace=True)
    return df_ball


def mirror_if_needed(x: float, y: float, attacking_side: str) -> tuple[float, float]:
    """Mirror raw tracking coordinates if attacking_side is right_to_left (event already mirrored)."""
    if attacking_side == "right_to_left":
        return -x, -y
    return x, y


def rescale_tracking_coords(x: float, y: float) -> tuple[float, float]:
    """Rescale tracking coords (SkillCorner raw) to 0-105 / 0-68, clipping to bounds."""
    x_clipped = max(X_MIN, min(X_MAX, x))
    y_clipped = max(Y_MIN, min(Y_MAX, y))
    x_res = (x_clipped - X_MIN) / (X_MAX - X_MIN) * 105.0
    y_res = (y_clipped - Y_MIN) / (Y_MAX - Y_MIN) * 68.0
    return x_res, y_res


def find_reception_ball_position(
    tracking_ball: pd.DataFrame,
    frame_release: int,
    target_x: float,
    target_y: float,
    attacking_side: str,
    search_window: int,
) -> tuple[Optional[int], Optional[float], Optional[float], Optional[float]]:
    """Search frames after release for closest ball position to reception target.

    Returns (frame_found, track_x_rescaled, track_y_rescaled, min_distance)
    If not found (no frames in window) returns (None, None, None, None).
    """
    if tracking_ball.empty:
        return None, None, None, None
    # Candidate frames
    candidate = tracking_ball.loc[(tracking_ball.index > frame_release) & (tracking_ball.index <= frame_release + search_window)]
    if candidate.empty:
        return None, None, None, None
    # Mirror + rescale each row and compute distance
    xs = []
    ys = []
    dists = []
    frames = []
    for f, row in candidate.iterrows():
        mx, my = mirror_if_needed(row["x"], row["y"], attacking_side)
        rx, ry = rescale_tracking_coords(mx, my)
        dist = ((rx - target_x) ** 2 + (ry - target_y) ** 2) ** 0.5
        xs.append(rx)
        ys.append(ry)
        dists.append(dist)
        frames.append(f)
    if not dists:
        return None, None, None, None
    idx_min = int(np.argmin(dists))
    return frames[idx_min], xs[idx_min], ys[idx_min], dists[idx_min]


def compute_alignment(
    passes: pd.DataFrame,
    tracking_cache: Dict[int, pd.DataFrame],
    search_window: int = 40,
    sample_size: Optional[int] = None,
    events_all: Optional[pd.DataFrame] = None,
    next_possession_max_frame_gap: int = 40,
    next_possession_distance_tolerance: float = 5.0,
    receiver_touch_radius: float = 1.5,
) -> List[PassAlignmentResult]:
    """Compute alignment metrics for passes using cached tracking data."""
    results: List[PassAlignmentResult] = []
    if sample_size is not None and sample_size < len(passes):
        passes = passes.sample(sample_size, random_state=42)

    # Precompute candidate next possession events if full events provided
    if events_all is not None:
        # Filter to player_possession events (potential possession starts)
        poss_events = events_all[events_all['event_type'] == 'player_possession'].copy()
        poss_events = poss_events[['match_id','event_id','team_id','player_id','frame_start','frame_end','x_start','y_start','period']]
        poss_events.sort_values(['match_id','period','frame_start'], inplace=True)
    else:
        poss_events = None

    for _, row in passes.iterrows():
        match_id = int(row["match_id"])
        tracking_full = tracking_cache.get(match_id)
        if tracking_full is None:
            track_path = TRACKING_DIR / f"{match_id}.parquet"
            if track_path.exists():
                tracking_full = pd.read_parquet(track_path)
            else:
                tracking_full = pd.DataFrame(columns=["frame","x","y","player_id","is_ball"])
            tracking_cache[match_id] = tracking_full
        tracking_ball = tracking_full[tracking_full.is_ball == True].set_index("frame") if not tracking_full.empty else pd.DataFrame().set_index("frame")

        attacking_side = row.get("attacking_side", None)
        frame_release = int(row.get("frame_end", -1)) if not pd.isna(row.get("frame_end")) else -1

        # Event origin coordinates (release point already rescaled & normalized after pipeline)
        origin_event_x = row.get("x_end_norm", np.nan)
        origin_event_y = row.get("y_end_norm", np.nan)

        origin_track_x = None
        origin_track_y = None
        origin_diff = None

        if frame_release != -1 and not tracking_ball.empty and frame_release in tracking_ball.index:
            tx_raw = tracking_ball.loc[frame_release, "x"]
            ty_raw = tracking_ball.loc[frame_release, "y"]
            mx, my = mirror_if_needed(tx_raw, ty_raw, attacking_side)
            origin_track_x, origin_track_y = rescale_tracking_coords(mx, my)
            if not np.isnan(origin_event_x) and not np.isnan(origin_event_y):
                origin_diff = ((origin_track_x - origin_event_x) ** 2 + (origin_track_y - origin_event_y) ** 2) ** 0.5

        # Reception handling (successful passes only)
        success = int(row.get("success", 0))
        reception_event_x = None
        reception_event_y = None
        reception_track_x = None
        reception_track_y = None
        reception_min_diff = None
        frame_found = None
        frame_lag = None

        if success == 1:
            reception_event_x = row.get("player_targeted_x_reception_norm", np.nan)
            reception_event_y = row.get("player_targeted_y_reception_norm", np.nan)
            if not np.isnan(reception_event_x) and not np.isnan(reception_event_y) and frame_release != -1:
                frame_found, reception_track_x, reception_track_y, reception_min_diff = find_reception_ball_position(
                    tracking_ball,
                    frame_release,
                    reception_event_x,
                    reception_event_y,
                    attacking_side or "left_to_right",
                    search_window,
                )
                if frame_found is not None:
                    frame_lag = frame_found - frame_release

        # Next possession comparison (successful passes only)
        next_possession_frame = None
        next_possession_start_x = None
        next_possession_start_y = None
        next_possession_event_diff = None
        next_possession_tracking_x = None
        next_possession_tracking_y = None
        next_possession_tracking_diff = None

        if success == 1 and poss_events is not None and frame_release != -1:
            player_targeted_id = row.get('player_targeted_id')
            # Candidate possession events for same match & period after release within frame gap
            cand_base = poss_events[(poss_events.match_id == match_id) &
                                    (poss_events.period == row.get('period')) &
                                    (poss_events.frame_start > frame_release) &
                                    (poss_events.frame_start <= frame_release + next_possession_max_frame_gap)]
            if not cand_base.empty and not np.isnan(reception_event_x) and not np.isnan(reception_event_y):
                # First try receiver-specific events
                cand_receiver = cand_base[cand_base.player_id == player_targeted_id]
                if not cand_receiver.empty:
                    cand_receiver = cand_receiver.assign(_dist=lambda d: np.sqrt((d.x_start - reception_event_x)**2 + (d.y_start - reception_event_y)**2))
                    cand_receiver = cand_receiver.sort_values('_dist')
                    best = cand_receiver.iloc[0]
                    if best._dist <= next_possession_distance_tolerance:
                        next_possession_frame = int(best.frame_start)
                        next_possession_start_x = float(best.x_start)
                        next_possession_start_y = float(best.y_start)
                        next_possession_event_diff = float(best._dist)
                # Fallback: nearest possession by coordinate within tolerance (any player)
                if next_possession_frame is None:
                    cand_any = cand_base.assign(_dist=lambda d: np.sqrt((d.x_start - reception_event_x)**2 + (d.y_start - reception_event_y)**2))
                    cand_any = cand_any.sort_values('_dist')
                    best_any = cand_any.iloc[0]
                    if best_any._dist <= next_possession_distance_tolerance:
                        next_possession_frame = int(best_any.frame_start)
                        next_possession_start_x = float(best_any.x_start)
                        next_possession_start_y = float(best_any.y_start)
                        next_possession_event_diff = float(best_any._dist)
                # Tracking comparison if we found a candidate
                if next_possession_frame is not None and next_possession_frame in tracking_ball.index:
                    tx_raw = tracking_ball.loc[next_possession_frame, 'x']
                    ty_raw = tracking_ball.loc[next_possession_frame, 'y']
                    mx, my = mirror_if_needed(tx_raw, ty_raw, attacking_side or 'left_to_right')
                    t_res_x, t_res_y = rescale_tracking_coords(mx, my)
                    next_possession_tracking_x = t_res_x
                    next_possession_tracking_y = t_res_y
                    next_possession_tracking_diff = np.sqrt((t_res_x - reception_event_x)**2 + (t_res_y - reception_event_y)**2)

        # Receiver touch detection
        player_touch_frame = None
        player_touch_ball_x = None
        player_touch_ball_y = None
        player_touch_player_x = None
        player_touch_player_y = None
        player_touch_ball_player_dist = None
        player_touch_event_reception_dist = None
        if success == 1 and frame_release != -1 and not np.isnan(reception_event_x) and not np.isnan(reception_event_y):
            targeted_id = row.get('player_targeted_id')
            if targeted_id and not tracking_full.empty:
                # Search frames forward for proximity
                candidate_frames = tracking_full[(tracking_full.frame > frame_release) & (tracking_full.frame <= frame_release + search_window)].frame.unique()
                for f in sorted(candidate_frames):
                    ball_row = tracking_full[(tracking_full.frame == f) & (tracking_full.is_ball == True)]
                    player_row = tracking_full[(tracking_full.frame == f) & (tracking_full.player_id == targeted_id)]
                    if ball_row.empty or player_row.empty:
                        continue
                    bx_raw, by_raw = float(ball_row.iloc[0].x), float(ball_row.iloc[0].y)
                    px_raw, py_raw = float(player_row.iloc[0].x), float(player_row.iloc[0].y)
                    mbx, mby = mirror_if_needed(bx_raw, by_raw, attacking_side or 'left_to_right')
                    mpx, mpy = mirror_if_needed(px_raw, py_raw, attacking_side or 'left_to_right')
                    bx_res, by_res = rescale_tracking_coords(mbx, mby)
                    px_res, py_res = rescale_tracking_coords(mpx, mpy)
                    dist_bp = np.sqrt((bx_res - px_res)**2 + (by_res - py_res)**2)
                    if dist_bp <= receiver_touch_radius:
                        player_touch_frame = int(f)
                        player_touch_ball_x = bx_res
                        player_touch_ball_y = by_res
                        player_touch_player_x = px_res
                        player_touch_player_y = py_res
                        player_touch_ball_player_dist = dist_bp
                        player_touch_event_reception_dist = np.sqrt((px_res - reception_event_x)**2 + (py_res - reception_event_y)**2)
                        break

        results.append(PassAlignmentResult(
            match_id=match_id,
            event_id=int(row.get("event_id", -1)),
            period=int(row.get("period", -1)),
            frame_release=frame_release,
            origin_event_x=origin_event_x,
            origin_event_y=origin_event_y,
            success=success,
            attacking_side=attacking_side,
            pass_outcome=row.get("pass_outcome"),
            frame_reception_found=frame_found,
            frame_lag=frame_lag,
            origin_track_x=origin_track_x,
            origin_track_y=origin_track_y,
            origin_diff=origin_diff,
            reception_event_x=reception_event_x,
            reception_event_y=reception_event_y,
            reception_track_x=reception_track_x,
            reception_track_y=reception_track_y,
            reception_min_diff=reception_min_diff,
            next_possession_frame=next_possession_frame,
            next_possession_start_x=next_possession_start_x,
            next_possession_start_y=next_possession_start_y,
            next_possession_event_diff=next_possession_event_diff,
            next_possession_tracking_x=next_possession_tracking_x,
            next_possession_tracking_y=next_possession_tracking_y,
            next_possession_tracking_diff=next_possession_tracking_diff,
            player_touch_frame=player_touch_frame,
            player_touch_ball_x=player_touch_ball_x,
            player_touch_ball_y=player_touch_ball_y,
            player_touch_player_x=player_touch_player_x,
            player_touch_player_y=player_touch_player_y,
            player_touch_ball_player_dist=player_touch_ball_player_dist,
            player_touch_event_reception_dist=player_touch_event_reception_dist,
        ))
    return results


def summarize(results: List[PassAlignmentResult]) -> None:
    """Print summary statistics for alignment results."""
    df = pd.DataFrame([r.__dict__ for r in results])
    print("\n=== PASS ORIGIN (Release Frame) ALIGNMENT ===")
    origin_valid = df["origin_diff"].dropna()
    if not origin_valid.empty:
        print(f"Samples: {len(origin_valid)}")
        print(origin_valid.describe(percentiles=[0.25, 0.5, 0.75]).to_string())
        for thr in [0.25, 0.5, 1.0, 2.0, 3.0]:
            pct = (origin_valid <= thr).mean() * 100
            print(f"  ≤ {thr:.2f} m: {pct:5.1f}%")
    else:
        print("No origin diffs computed.")

    print("\n=== SUCCESSFUL PASS RECEPTION ALIGNMENT ===")
    succ = df[df["success"] == 1]
    rec_valid = succ["reception_min_diff"].dropna()
    if not rec_valid.empty:
        print(f"Samples: {len(rec_valid)}")
        print(rec_valid.describe(percentiles=[0.25, 0.5, 0.75]).to_string())
        for thr in [0.25, 0.5, 1.0, 2.0, 3.0]:
            pct = (rec_valid <= thr).mean() * 100
            print(f"  ≤ {thr:.2f} m: {pct:5.1f}%")
        lag = succ["frame_lag"].dropna()
        if not lag.empty:
            print("\nFrame lag (release→closest reception frame):")
            print(lag.describe(percentiles=[0.25, 0.5, 0.75]).to_string())
    else:
        print("No reception diffs computed.")

    # Basic correlation (origin)
    if not df.empty and df["origin_track_x"].notna().any():
        corr_x = df[["origin_event_x", "origin_track_x"]].dropna().corr().iloc[0, 1]
        corr_y = df[["origin_event_y", "origin_track_y"]].dropna().corr().iloc[0, 1]
        print(f"\nOrigin X correlation (event vs tracking): {corr_x:.3f}")
        print(f"Origin Y correlation (event vs tracking): {corr_y:.3f}")

    # Next possession stats
    succ_np = df[(df.success == 1) & df.next_possession_event_diff.notna()]
    if not succ_np.empty:
        print("\n=== NEXT POSSESSION START ALIGNMENT (Successful Passes) ===")
        print("Event reception vs next possession start:")
        print(succ_np.next_possession_event_diff.describe(percentiles=[0.25,0.5,0.75]).to_string())
        for thr in [0.5,1.0,2.0,3.0]:
            pct = (succ_np.next_possession_event_diff <= thr).mean()*100
            print(f"  ≤ {thr:.1f} m: {pct:5.1f}%")
        track_valid = succ_np.next_possession_tracking_diff.dropna()
        if not track_valid.empty:
            print("\nEvent reception vs tracking ball at next possession frame:")
            print(track_valid.describe(percentiles=[0.25,0.5,0.75]).to_string())
            for thr in [0.5,1.0,2.0,3.0]:
                pct = (track_valid <= thr).mean()*100
                print(f"  ≤ {thr:.1f} m: {pct:5.1f}%")
            frame_gap = (succ_np.next_possession_frame - succ_np.frame_release).dropna()
            if not frame_gap.empty:
                print("\nFrame gap release→next possession start:")
                print(frame_gap.describe(percentiles=[0.25,0.5,0.75]).to_string())

    # Receiver touch summary
    touch_df = df[(df.success == 1) & df.player_touch_frame.notna()]
    if not touch_df.empty:
        print("\n=== RECEIVER TOUCH DETECTION (Tracking-based) ===")
        print("Ball–player distance at detected touch (≤ radius):")
        print(touch_df.player_touch_ball_player_dist.describe(percentiles=[0.25,0.5,0.75]).to_string())
        print("Event reception vs player position at touch:")
        print(touch_df.player_touch_event_reception_dist.describe(percentiles=[0.25,0.5,0.75]).to_string())
        gap_touch = (touch_df.player_touch_frame - touch_df.frame_release).dropna()
        print("Frame gap release→touch frame:")
        print(gap_touch.describe(percentiles=[0.25,0.5,0.75]).to_string())
        coverage_pct = len(touch_df)/len(df[df.success==1])*100
        print(f"Coverage of successful passes with detected touch: {coverage_pct:.1f}%")
    else:
        print("\nNo receiver touches detected within radius for sample.")


def prepare_passes(limit_matches: Optional[int] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load events and return (events, passes) with normalized coordinates ready for alignment."""
    events = load_premier_league_events(limit_matches=limit_matches)
    passes = extract_pass_events(events)
    passes = rescale_coordinates(passes)
    passes = normalize_attack_direction(passes)
    return events, passes


def main():
    parser = argparse.ArgumentParser(description="Validate tracking vs event pass coordinates")
    parser.add_argument("--matches", type=int, default=3, help="Number of matches to load (for speed)")
    parser.add_argument("--search-window", type=int, default=40, help="Frames to search after release for reception")
    parser.add_argument("--sample-size", type=int, default=500, help="Sample passes (set None for all)")
    parser.add_argument("--output-csv", type=str, default="", help="Optional path to write detailed results CSV")
    args = parser.parse_args()

    print("Loading passes...")
    events, passes = prepare_passes(limit_matches=args.matches)
    print(f"Loaded {len(passes):,} passes from {args.matches} matches")

    results = compute_alignment(
        passes=passes,
        tracking_cache={},
        search_window=args.search_window,
        sample_size=None if args.sample_size <= 0 else args.sample_size,
        events_all=events,
    )

    summarize(results)

    if args.output_csv:
        out_path = Path(args.output_csv)
        pd.DataFrame([r.__dict__ for r in results]).to_csv(out_path, index=False)
        print(f"\nDetailed results written to: {out_path}")


if __name__ == "__main__":
    main()
