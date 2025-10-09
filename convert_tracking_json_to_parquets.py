# -*- coding: utf-8 -*-
"""
Convert each tracking JSON into its own Parquet file.

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
PARQUET_DIR = REALMADRID_DIR / "tracking_parquets"

# Create output folder if it doesn’t exist
PARQUET_DIR.mkdir(exist_ok=True)

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

    for match_id in df_matches["id"].values:
        try:
            df_tracking = load_tracking_full(match_id)
            if df_tracking is None or df_tracking.empty:
                print(f"⚠️ Empty tracking for {match_id}, skipping")
                continue

            # Save each match as its own parquet
            output_path = PARQUET_DIR / f"{match_id}.parquet"
            df_tracking.to_parquet(output_path)
            print(f"✅ Saved {output_path} ({len(df_tracking)} rows)")

        except FileNotFoundError:
            print(f"⚠️ No tracking for match {match_id}")
        except Exception as e:
            print(f"❌ Error with match {match_id}: {e}")

    print("✅ Finished converting all JSON files to Parquet.")
