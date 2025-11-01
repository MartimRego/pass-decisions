"""
Test script to filter out zero-distance passes and test different thresholds.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import (
    load_premier_league_events,
    extract_pass_events,
    rescale_coordinates,
    normalize_attack_direction,
    classify_pass_direction
)
import numpy as np
import pandas as pd

def classify_pass_length_10m(distance: pd.Series) -> pd.Series:
    """Classify with 10m threshold for medium passes."""
    return pd.cut(
        distance,
        bins=[-0.001, 10, 25, np.inf],
        labels=['short', 'medium', 'long']
    )

def main():
    print("Loading Premier League 2024 events...")
    events = load_premier_league_events()
    
    print("\nExtracting pass events...")
    passes = extract_pass_events(events)
    print(f"Total passes extracted: {len(passes):,}")
    
    print("\nRescaling coordinates...")
    passes = rescale_coordinates(passes)
    
    print("\nNormalizing attack direction...")
    passes = normalize_attack_direction(passes)
    
    # Calculate distances
    print("\nCalculating pass distances...")
    passes['dx'] = passes['x_end_norm'] - passes['x_start_norm']
    passes['dy'] = passes['y_end_norm'] - passes['y_start_norm']
    passes['pass_distance'] = np.sqrt(passes['dx']**2 + passes['dy']**2)
    
    # Show zero-distance stats
    zero_dist = passes['pass_distance'] == 0
    print(f"\nZero-distance passes: {zero_dist.sum():,} ({zero_dist.sum()/len(passes)*100:.2f}%)")
    
    # Original success rate
    original_success_rate = passes['success'].mean() * 100
    print(f"\n📊 ORIGINAL SUCCESS RATE (with zero-distance passes):")
    print(f"   {original_success_rate:.2f}%")
    
    # Filter out zero-distance passes
    print(f"\n🔧 Filtering out zero-distance passes...")
    passes_filtered = passes[passes['pass_distance'] > 0].copy()
    print(f"   Remaining passes: {len(passes_filtered):,}")
    
    # New success rate
    new_success_rate = passes_filtered['success'].mean() * 100
    print(f"\n📊 NEW SUCCESS RATE (zero-distance passes removed):")
    print(f"   {new_success_rate:.2f}%")
    print(f"   Change: {new_success_rate - original_success_rate:+.2f} percentage points")
    
    # Classify with 10m threshold
    print(f"\n{'='*60}")
    print(f"PASS CLASSIFICATION WITH 10m MEDIUM THRESHOLD")
    print(f"{'='*60}")
    
    passes_filtered['pass_length'] = classify_pass_length_10m(passes_filtered['pass_distance'])
    passes_filtered['pass_direction'] = classify_pass_direction(passes_filtered['dx'])
    passes_filtered['pass_type'] = passes_filtered['pass_length'].astype(str) + '_' + passes_filtered['pass_direction'].astype(str)
    
    print(f"\nTotal passes (filtered): {len(passes_filtered):,}")
    
    print("\n--- Length Distribution ---")
    length_dist = passes_filtered['pass_length'].value_counts().sort_index()
    print(length_dist)
    print(f"\nPercentages:")
    for label in ['short', 'medium', 'long']:
        if label in length_dist.index:
            count = length_dist[label]
            pct = (count / len(passes_filtered)) * 100
            print(f"  {label:8s}: {count:7,} ({pct:5.2f}%)")
    
    print("\n--- Direction Distribution ---")
    dir_dist = passes_filtered['pass_direction'].value_counts().sort_index()
    print(dir_dist)
    print(f"\nPercentages:")
    for label in ['backward', 'lateral', 'forward']:
        if label in dir_dist.index:
            count = dir_dist[label]
            pct = (count / len(passes_filtered)) * 100
            print(f"  {label:8s}: {count:7,} ({pct:5.2f}%)")
    
    print("\n--- Combined Type Distribution (sorted by count) ---")
    type_dist = passes_filtered['pass_type'].value_counts()
    print(type_dist)
    
    print(f"\nPercentages:")
    for pass_type, count in type_dist.items():
        pct = (count / len(passes_filtered)) * 100
        status = "✅" if pct >= 1.0 else "⚠️" if pct >= 0.5 else "❌"
        print(f"  {status} {pass_type:20s}: {count:7,} ({pct:5.2f}%)")
    
    # Success rates by type
    print("\n--- Success Rate by Type ---")
    success_by_type = passes_filtered.groupby('pass_type')['success'].agg(['mean', 'count'])
    success_by_type.columns = ['success_rate', 'count']
    success_by_type['percentage'] = (success_by_type['count'] / len(passes_filtered)) * 100
    success_by_type = success_by_type.sort_values('success_rate', ascending=False)
    print(success_by_type)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Passes after filtering: {len(passes_filtered):,}")
    print(f"Overall success rate: {new_success_rate:.2f}%")
    print(f"Number of pass types: {passes_filtered['pass_type'].nunique()}")
    print(f"\nPass types with ≥1% frequency:")
    frequent_types = type_dist[type_dist / len(passes_filtered) >= 0.01]
    print(f"  {len(frequent_types)} types: {', '.join(frequent_types.index.tolist())}")
    print(f"\nPass types with <1% frequency (candidates for removal/merging):")
    rare_types = type_dist[type_dist / len(passes_filtered) < 0.01]
    print(f"  {len(rare_types)} types: {', '.join(rare_types.index.tolist())}")
    print(f"{'='*60}")
    
    return passes_filtered

if __name__ == "__main__":
    passes = main()
