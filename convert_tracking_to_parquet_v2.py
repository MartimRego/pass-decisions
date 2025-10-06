# -*- coding: utf-8 -*-
"""
Aggregate all tracking JSONs into one parquet file.

Author: Pegah & Gustimorth
"""

import os
import json
import pandas as pd
from pathlib import Path

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
BASE_DIR = Path.cwd()
REALMADRID_DIR = BASE_DIR / "RealMadrid"
TRACKING_DIR = REALMADRID_DIR / "tracking"
OUTPUT_FILE = BASE_DIR / "tracking_raw.parquet"

# --------------------------------------------------------------------------
# LOADER FUNCTION
# --------------------------------------------------------------------------
def load_tracking_full(match_id: int, sort_rows: bool = False, require_ball_detected: bool = True) -> pd.DataFrame:
    fpath = TRACKING_DIR / f"{match_id}.json"
    if not fpath.exists():
        raise FileNotFoundError(f"No tracking for match {match_id}")
    with open(fpath, "r") as f:
        raw = json.load(f)

    rows = []
    for d in raw:
        frame = int(d.get("frame"))
        ts = d.get("timestamp")
        period = d.get("period")

        # Players
        for p in d.get("player_data") or []:
            rows.append({
                "match_id": match_id,
                "time": ts,
                "frame": frame,
                "period": period,
                "player_id": p.get("player_id"),
                "is_detected": bool(p.get("is_detected", False)),
                "is_ball": False,
                "x": p.get("x"),
                "y": p.get("y"),
            })

        # Ball
        ball = d.get("ball_data")
        if ball is not None:
            if (not require_ball_detected) or (ball.get("is_detected") is not None):
                rows.append({
                    "match_id": match_id,
                    "time": ts,
                    "frame": frame,
                    "period": period,
                    "player_id": -1,
                    "is_detected": bool(ball.get("is_detected", False)),
                    "is_ball": True,
                    "x": ball.get("x"),
                    "y": ball.get("y"),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if sort_rows:
        df = df.sort_values(["match_id", "frame", "player_id"]).reset_index(drop=True)
    return df

# --------------------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------------------
if __name__ == "__main__":
    matches_path = REALMADRID_DIR / "matches.parquet"
    if not matches_path.exists():
        raise FileNotFoundError(f"Missing matches file: {matches_path}")

    df_matches = pd.read_parquet(matches_path)
    print(f"Found {len(df_matches)} matches")

    all_dfs = []
    for match_id in df_matches["id"].values:
        try:
            df_tracking = load_tracking_full(match_id)
            if df_tracking is None or df_tracking.empty:
                print(f"⚠️ Empty tracking for {match_id}, skipping")
                continue
            all_dfs.append(df_tracking)
            print(f"✅ Loaded match {match_id} ({len(df_tracking)} rows)")
        except FileNotFoundError:
            print(f"⚠️ No tracking for match {match_id}")
        except Exception as e:
            print(f"❌ Error with match {match_id}: {e}")

    # Concatenate all and save
    if len(all_dfs) == 0:
        print("⚠️ No tracking data found — nothing to save.")
    else:
        df_all = pd.concat(all_dfs, ignore_index=True)
        df_all.to_parquet(OUTPUT_FILE)
        print(f"✅ Saved combined tracking data to {OUTPUT_FILE}")
        print(f"   Total rows: {len(df_all):,}")
