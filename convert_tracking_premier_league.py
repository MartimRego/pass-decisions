# -*- coding: utf-8 -*-
"""
Convert Premier League tracking JSON files into Parquet format.

This script processes tracking data for the Premier League 2024 season.
"""

import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
BASE_DIR = Path.cwd()
PL_DIR = BASE_DIR / "PremierLeague_data" / "2024"
TRACKING_DIR = PL_DIR / "tracking"
PARQUET_DIR = PL_DIR / "tracking_parquets"

# Create output folder if it doesn't exist
PARQUET_DIR.mkdir(exist_ok=True, parents=True)

print(f"📁 Input directory:  {TRACKING_DIR}")
print(f"📁 Output directory: {PARQUET_DIR}")

# --------------------------------------------------------------------------
# LOADER FUNCTION
# --------------------------------------------------------------------------
def load_tracking_full(match_id: int, sort_rows: bool = False, require_ball_detected: bool = True) -> pd.DataFrame:
    """Load tracking data from JSON and convert to DataFrame."""
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
    # Get list of available JSON files
    json_files = list(TRACKING_DIR.glob("*.json"))
    
    if not json_files:
        print(f"❌ No JSON files found in {TRACKING_DIR}")
        print(f"   Please check that the tracking directory exists and contains .json files")
        exit(1)
    
    print(f"\n✅ Found {len(json_files)} tracking JSON files")
    print(f"🔄 Converting to Parquet format...\n")
    
    success_count = 0
    error_count = 0
    
    for json_file in tqdm(json_files, desc="Converting"):
        try:
            # Extract match_id from filename (e.g., "3869151.json" -> 3869151)
            match_id = int(json_file.stem)
            
            # Load and convert
            df_tracking = load_tracking_full(match_id)
            
            if df_tracking is None or df_tracking.empty:
                print(f"⚠️  Empty tracking for match {match_id}, skipping")
                continue

            # Save as parquet
            output_path = PARQUET_DIR / f"{match_id}.parquet"
            df_tracking.to_parquet(output_path, index=False)
            success_count += 1

        except FileNotFoundError:
            print(f"⚠️  No tracking for match {match_id}")
            error_count += 1
        except Exception as e:
            print(f"❌ Error with match {match_id}: {e}")
            error_count += 1

    print(f"\n{'='*60}")
    print(f"✅ Conversion complete!")
    print(f"   Successfully converted: {success_count} files")
    if error_count > 0:
        print(f"   Errors: {error_count} files")
    print(f"   Output location: {PARQUET_DIR}")
    print(f"{'='*60}\n")
