"""
Diagnose pass direction classification to understand why backward passes
have lower success rates than forward passes.
"""
import sys
sys.path.insert(0, 'src')

from data_processing import (
    load_premier_league_events,
    extract_pass_events,
    rescale_coordinates,
    normalize_attack_direction,
    classify_pass_type
)
import pandas as pd
import numpy as np

# Load data
print("Loading data...")
events = load_premier_league_events()
passes = extract_pass_events(events)
passes = rescale_coordinates(passes)

print(f"\n{'='*60}")
print("BEFORE NORMALIZATION")
print(f"{'='*60}")

# Check attacking_side distribution
print(f"\nAttacking side distribution:")
print(passes['attacking_side'].value_counts())

# Sample passes for each attacking side
print(f"\n--- Sample passes attacking LEFT TO RIGHT ---")
ltr = passes[passes['attacking_side'] == 'left_to_right'].head(3)
for idx, row in ltr.iterrows():
    dx_raw = row['x_end_rescaled'] - row['x_start_rescaled']
    print(f"  x: {row['x_start_rescaled']:.1f} → {row['x_end_rescaled']:.1f}, dx = {dx_raw:+.1f}")

print(f"\n--- Sample passes attacking RIGHT TO LEFT ---")
rtl = passes[passes['attacking_side'] == 'right_to_left'].head(3)
for idx, row in rtl.iterrows():
    dx_raw = row['x_end_rescaled'] - row['x_start_rescaled']
    print(f"  x: {row['x_start_rescaled']:.1f} → {row['x_end_rescaled']:.1f}, dx = {dx_raw:+.1f}")

# Now normalize
passes = normalize_attack_direction(passes)

print(f"\n{'='*60}")
print("AFTER NORMALIZATION")
print(f"{'='*60}")

print(f"\n--- Sample normalized passes (originally LEFT TO RIGHT) ---")
ltr_norm = passes[passes['attacking_side'] == 'left_to_right'].head(3)
for idx, row in ltr_norm.iterrows():
    dx_norm = row['x_end_norm'] - row['x_start_norm']
    print(f"  x: {row['x_start_norm']:.1f} → {row['x_end_norm']:.1f}, dx = {dx_norm:+.1f}")

print(f"\n--- Sample normalized passes (originally RIGHT TO LEFT) ---")
rtl_norm = passes[passes['attacking_side'] == 'right_to_left'].head(3)
for idx, row in rtl_norm.iterrows():
    dx_norm = row['x_end_norm'] - row['x_start_norm']
    print(f"  x: {row['x_start_norm']:.1f} → {row['x_end_norm']:.1f}, dx = {dx_norm:+.1f}")

# Classify
passes = classify_pass_type(passes)

print(f"\n{'='*60}")
print("PASS DIRECTION ANALYSIS")
print(f"{'='*60}")

# Check dx distribution
print(f"\ndx statistics:")
print(passes['dx'].describe())

# Direction distribution
print(f"\nDirection distribution:")
print(passes['pass_direction'].value_counts())

# Success rates by direction
print(f"\nSuccess rates by direction:")
for direction in ['forward', 'lateral', 'backward']:
    mask = passes['pass_direction'] == direction
    if mask.sum() > 0:
        success_rate = passes[mask]['success'].mean()
        count = mask.sum()
        avg_dx = passes[mask]['dx'].mean()
        print(f"  {direction:10s}: {success_rate*100:5.2f}% ({count:6,} passes, avg dx = {avg_dx:+6.2f}m)")

# Check for potential issues
print(f"\n{'='*60}")
print("DIAGNOSTIC CHECKS")
print(f"{'='*60}")

# 1. Are backward passes actually going backward?
backward_passes = passes[passes['pass_direction'] == 'backward']
print(f"\nBackward passes (should have dx < -5):")
print(f"  Count: {len(backward_passes):,}")
print(f"  dx range: [{backward_passes['dx'].min():.2f}, {backward_passes['dx'].max():.2f}]")
print(f"  All have dx < -5? {(backward_passes['dx'] < -5).all()}")

# 2. Are forward passes actually going forward?
forward_passes = passes[passes['pass_direction'] == 'forward']
print(f"\nForward passes (should have dx > 5):")
print(f"  Count: {len(forward_passes):,}")
print(f"  dx range: [{forward_passes['dx'].min():.2f}, {forward_passes['dx'].max():.2f}]")
print(f"  All have dx > 5? {(forward_passes['dx'] > 5).all()}")

# 3. Check if there's a relationship between attacking_side and success
print(f"\nSuccess rate by original attacking side:")
for side in ['left_to_right', 'right_to_left']:
    mask = passes['attacking_side'] == side
    if mask.sum() > 0:
        success_rate = passes[mask]['success'].mean()
        count = mask.sum()
        print(f"  {side:15s}: {success_rate*100:5.2f}% ({count:6,} passes)")

# 4. Sample some backward passes to understand them
print(f"\n{'='*60}")
print("SAMPLE BACKWARD PASSES")
print(f"{'='*60}")
backward_sample = backward_passes.sample(min(5, len(backward_passes)))
for idx, row in backward_sample.iterrows():
    print(f"\nPass {idx}:")
    print(f"  Position: ({row['x_start_norm']:.1f}, {row['y_start_norm']:.1f}) → ({row['x_end_norm']:.1f}, {row['y_end_norm']:.1f})")
    print(f"  dx = {row['dx']:+.1f}m, dy = {row['dy']:+.1f}m, distance = {row['pass_distance']:.1f}m")
    print(f"  Success: {row['success']}, Outcome: {row.get('pass_outcome', 'N/A')}")
    print(f"  Attacking side: {row['attacking_side']}")
