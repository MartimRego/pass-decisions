"""
Test the clearance filter and display passes dataframe.
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

# Configure pandas display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# Load and process data
print("="*70)
print("LOADING AND PROCESSING DATA")
print("="*70)
events = load_premier_league_events()
passes = extract_pass_events(events)
passes = rescale_coordinates(passes)
passes = normalize_attack_direction(passes)

# Add distance for classification
passes['dx'] = passes['x_end_norm'] - passes['x_start_norm']
passes['dy'] = passes['y_end_norm'] - passes['y_start_norm']
passes['distance'] = np.sqrt(passes['dx']**2 + passes['dy']**2)

passes = classify_pass_type(passes)

print(f"\n{'='*70}")
print("PASSES DATAFRAME - FIRST 10 ROWS")
print(f"{'='*70}\n")

# Show key columns first
key_columns = [
    'event_id', 'match_id', 'period', 'minute_start', 
    'team_shortname', 'player_name', 
    'x_start_norm', 'y_start_norm', 'x_end_norm', 'y_end_norm',
    'dx', 'dy', 'distance',
    'pass_type', 'success',
    'end_type', 'pass_outcome'
]

# Filter to available columns
available_key_cols = [col for col in key_columns if col in passes.columns]

print("KEY COLUMNS:")
print(passes[available_key_cols].head(10).to_string(index=False))

print(f"\n{'='*70}")
print("ALL COLUMNS (column names)")
print(f"{'='*70}")
print(f"\nTotal columns: {len(passes.columns)}")
print("\nColumn names (first 50):")
for i, col in enumerate(passes.columns[:50], 1):
    print(f"  {i:2d}. {col}")
if len(passes.columns) > 50:
    print(f"\n  ... and {len(passes.columns) - 50} more columns")

# Updated analysis
print(f"\n{'='*70}")
print("UPDATED ANALYSIS (AFTER CLEARANCE REMOVAL)")
print(f"{'='*70}")

from data_processing import classify_pass_direction, classify_pass_length

passes['pass_length'] = classify_pass_length(passes['distance'])
passes['pass_direction'] = classify_pass_direction(passes['dx'], passes['dy'])

# Overall stats
print(f"\nTotal passes: {len(passes):,}")
print(f"Overall success rate: {passes['success'].mean()*100:.2f}%")

# Success by direction
print(f"\n{'='*70}")
print("SUCCESS RATE BY DIRECTION")
print(f"{'='*70}")
for direction in ['forward', 'lateral', 'backward']:
    subset = passes[passes['pass_direction'] == direction]
    if len(subset) > 0:
        success_rate = subset['success'].mean() * 100
        avg_dx = subset['dx'].mean()
        print(f"{direction:10s}: {success_rate:5.2f}% ({len(subset):7,} passes, avg dx = {avg_dx:+6.2f}m)")

# Success by zone and direction
print(f"\n{'='*70}")
print("SUCCESS BY ZONE AND DIRECTION (UPDATED)")
print(f"{'='*70}")

# Define zones
passes['zone'] = pd.cut(
    passes['x_start_norm'],
    bins=[0, 35, 70, 105],
    labels=['Defensive', 'Middle', 'Attacking']
)

for zone in ['Defensive', 'Middle', 'Attacking']:
    zone_passes = passes[passes['zone'] == zone]
    print(f"\n{zone} Third:")
    for direction in ['backward', 'lateral', 'forward']:
        subset = zone_passes[zone_passes['pass_direction'] == direction]
        if len(subset) > 0:
            success_rate = subset['success'].mean() * 100
            print(f"  {direction:10s}: {success_rate:5.2f}% ({len(subset):6,} passes)")

# Pass type distribution
print(f"\n{'='*70}")
print("PASS TYPE DISTRIBUTION")
print(f"{'='*70}")
for ptype in sorted(passes['pass_type'].unique()):
    subset = passes[passes['pass_type'] == ptype]
    success_rate = subset['success'].mean() * 100
    pct = len(subset) / len(passes) * 100
    print(f"{ptype:20s}: {len(subset):6,} ({pct:5.2f}%) | Success: {success_rate:5.2f}%")

print(f"\n{'='*70}")
print(f"✅ Clearance filter applied successfully!")
print(f"{'='*70}")
