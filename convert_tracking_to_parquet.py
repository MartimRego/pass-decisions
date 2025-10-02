# -*- coding: utf-8 -*-
"""
Created on Fri May  9 09:39:47 2025

Code to generate freeze frames from the start and end frames 
of the events in the dynamic events

@author: gustimorth
"""
import json
import pandas as pd
import os
from settings import DATA_DIR

#Set the data path to the JSONL files. 
data_path = f"{DATA_DIR}/RealMadrid" #Change this to the path where your data is stored
df_matches = pd.read_parquet(f"{data_path}/matches.parquet")

if not os.path.exists(f"{data_path}/tracking_parquet"):
    os.makedirs(f"{data_path}/tracking_parquet")

#Save each player and ball position per frame into a parquet file
for match_id in df_matches['id'].values[:5]:  # First 5 matches - remove [0:5] to process all
    fpath = f"{data_path}/tracking/{match_id}.json"
    if not os.path.exists(fpath):
        print(f"⚠️ No tracking file for match {match_id}")
        continue

    #If the parquet file already exists, skip
    if os.path.exists(f"{data_path}/tracking_parquet/{match_id}.parquet"):
        print(f"⏭️ Parquet file already exists for match {match_id}, skipping")
        continue

    with open(fpath, "r") as f:
        tracking_data = json.load(f)
    all_data = []
    for d in tracking_data:
        frame = d.get("frame")
        period = d.get("period")
        timestamp = d.get("timestamp")

        # Players
        for p in d["player_data"]:
            all_data.append({
                "match_id": match_id,
                "time": timestamp,
                "frame": frame,
                "period": period,
                "player_id": p.get("player_id"),
                "is_detected": p.get("is_detected"),
                "is_ball": False,
                "x": p.get("x"),
                "y": p.get("y"),
            })

        # Ball
        ball = d.get("ball_data")
        if ball and ball.get("is_detected") is not None:
            all_data.append({
                "match_id": match_id,
                "time": timestamp,
                "frame": frame,
                "period": period,
                "player_id": -1,   # Ball
                "is_detected": ball.get("is_detected"),
                "is_ball": True,
                "x": ball.get("x"),
                "y": ball.get("y"),
            })
    
     # Transform to dataframe
    df_tracking = pd.DataFrame(all_data)

    # Save as freeze frames
    df_tracking.to_parquet(f"{data_path}/tracking_parquet/{match_id}.parquet")
    print(f"Saved tracking parquet for match {match_id}")
