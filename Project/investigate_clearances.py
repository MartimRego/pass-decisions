"""
Investigate what fields we can use to identify and filter clearances.
"""
import sys
sys.path.insert(0, 'src')

from data_processing import load_premier_league_events, extract_pass_events
import pandas as pd

# Load data
print("Loading data...")
events = load_premier_league_events()
passes = extract_pass_events(events)

print(f"\n{'='*70}")
print("AVAILABLE FIELDS IN PASS DATA")
print(f"{'='*70}")
print(f"\nAll columns: {len(passes.columns)}")
print(passes.columns.tolist())

print(f"\n{'='*70}")
print("FIELD ANALYSIS FOR CLEARANCE DETECTION")
print(f"{'='*70}")

# Check end_type field
if 'end_type' in passes.columns:
    print(f"\n1. end_type values:")
    print(passes['end_type'].value_counts())
    print(f"\nTypes that might indicate clearances:")
    for end_type in passes['end_type'].unique():
        if pd.notna(end_type):
            count = (passes['end_type'] == end_type).sum()
            pct = count / len(passes) * 100
            print(f"  {end_type:20s}: {count:7,} ({pct:5.2f}%)")

# Check pass_outcome
if 'pass_outcome' in passes.columns:
    print(f"\n2. pass_outcome values:")
    print(passes['pass_outcome'].value_counts())

# Check if there's a 'type' or 'subtype' field
for col in ['type', 'subtype', 'pass_type_detail', 'technique']:
    if col in passes.columns:
        print(f"\n3. {col} values:")
        print(passes[col].value_counts().head(20))

# Check distance and direction for potential clearances
print(f"\n{'='*70}")
print("CLEARANCE HEURISTICS")
print(f"{'='*70}")

from data_processing import rescale_coordinates, normalize_attack_direction
passes_proc = rescale_coordinates(passes)
passes_proc = normalize_attack_direction(passes_proc)

# Calculate distance
import numpy as np
passes_proc['dx'] = passes_proc['x_end_norm'] - passes_proc['x_start_norm']
passes_proc['dy'] = passes_proc['y_end_norm'] - passes_proc['y_start_norm']
passes_proc['distance'] = np.sqrt(passes_proc['dx']**2 + passes_proc['dy']**2)

# Defensive third
defensive = passes_proc[passes_proc['x_start_norm'] < 35].copy()

# Long backward passes in defensive third (likely clearances)
potential_clearances = defensive[
    (defensive['dx'] < -10) & 
    (defensive['distance'] > 20)
].copy()

print(f"\nPotential clearances (defensive third, dx < -10m, distance > 20m):")
print(f"  Count: {len(potential_clearances):,}")
print(f"  Success rate: {potential_clearances['success'].mean()*100:.2f}%")

if 'end_type' in potential_clearances.columns:
    print(f"\n  end_type distribution for potential clearances:")
    print(potential_clearances['end_type'].value_counts())

if 'pass_outcome' in potential_clearances.columns:
    print(f"\n  pass_outcome distribution for potential clearances:")
    print(potential_clearances['pass_outcome'].value_counts())

# Sample some
print(f"\n{'='*70}")
print("SAMPLE POTENTIAL CLEARANCES")
print(f"{'='*70}")

sample = potential_clearances.sample(min(5, len(potential_clearances)))
for idx, row in sample.iterrows():
    print(f"\nPotential clearance {idx}:")
    print(f"  Position: ({row['x_start_norm']:.1f}, {row['y_start_norm']:.1f}) → ({row['x_end_norm']:.1f}, {row['y_end_norm']:.1f})")
    print(f"  dx = {row['dx']:+.1f}m, distance = {row['distance']:.1f}m")
    print(f"  Success: {row['success']}, Outcome: {row.get('pass_outcome', 'N/A')}, End type: {row.get('end_type', 'N/A')}")

# Check for other potential indicators
print(f"\n{'='*70}")
print("OTHER POTENTIAL INDICATORS")
print(f"{'='*70}")

# Check if body_part might help
if 'body_part' in passes.columns:
    print(f"\nbody_part distribution:")
    print(passes['body_part'].value_counts())
    
# Check aerial duels
if 'aerial' in passes.columns or 'is_aerial' in passes.columns:
    aerial_col = 'aerial' if 'aerial' in passes.columns else 'is_aerial'
    print(f"\n{aerial_col} distribution:")
    print(passes[aerial_col].value_counts())
