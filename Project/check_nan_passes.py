"""Quick check of NaN pass distances"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import (
    load_premier_league_events,
    extract_pass_events,
    rescale_coordinates,
    normalize_attack_direction
)
import numpy as np

events = load_premier_league_events()
passes = extract_pass_events(events)
passes = rescale_coordinates(passes)
passes = normalize_attack_direction(passes)

# Calculate distance manually
passes['dx'] = passes['x_end_norm'] - passes['x_start_norm']
passes['dy'] = passes['y_end_norm'] - passes['y_start_norm']
passes['pass_distance'] = np.sqrt(passes['dx']**2 + passes['dy']**2)

print(f"Total passes: {len(passes)}")
print(f"Passes with NaN distance: {passes['pass_distance'].isna().sum()}")
print(f"\nDistance statistics:")
print(passes['pass_distance'].describe())
print(f"\nMin distance: {passes['pass_distance'].min()}")
print(f"Max distance: {passes['pass_distance'].max()}")

# Check for zero distances
zero_dist = (passes['pass_distance'] == 0).sum()
print(f"\nPasses with zero distance: {zero_dist}")

# Check NaN in coordinates
print(f"\nNaN in x_start_norm: {passes['x_start_norm'].isna().sum()}")
print(f"NaN in x_end_norm: {passes['x_end_norm'].isna().sum()}")
print(f"NaN in y_start_norm: {passes['y_start_norm'].isna().sum()}")
print(f"NaN in y_end_norm: {passes['y_end_norm'].isna().sum()}")
