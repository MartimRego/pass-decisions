"""
Test script to validate pass classification and count long_backward passes.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import (
    load_premier_league_events,
    load_match_metadata,
    extract_pass_events,
    rescale_coordinates,
    normalize_attack_direction,
    classify_pass_type,
    validate_pass_classifications
)

def main():
    print("Loading Premier League 2024 events...")
    events = load_premier_league_events()
    print(f"Loaded {len(events):,} events")
    
    print("\nExtracting pass events...")
    passes = extract_pass_events(events)
    print(f"Found {len(passes):,} passes")
    
    print("\nRescaling coordinates...")
    passes = rescale_coordinates(passes)
    
    print("\nNormalizing attack direction...")
    passes = normalize_attack_direction(passes)
    
    print("\nClassifying pass types...")
    passes = classify_pass_type(passes)
    
    print("\n")
    validate_pass_classifications(passes)
    
    # Extract just the long_backward count for user decision
    if 'pass_type' in passes.columns:
        type_counts = passes['pass_type'].value_counts()
        if 'long_backward' in type_counts.index:
            long_backward = type_counts['long_backward']
            pct = (long_backward / len(passes)) * 100
            print(f"\n{'='*60}")
            print(f"📊 DECISION POINT: LONG_BACKWARD ACTION")
            print(f"{'='*60}")
            print(f"Long backward passes: {long_backward:,} ({pct:.2f}%)")
            print(f"Total passes: {len(passes):,}")
            print(f"\nRecommendation:")
            if pct < 1.0:
                print(f"  ❌ REMOVE - Too rare (<1%) for reliable MDP modeling")
            elif pct < 2.0:
                print(f"  ⚠️ BORDERLINE - Consider merging with long_lateral")
            else:
                print(f"  ✅ KEEP - Sufficient data for MDP action space")
            print(f"{'='*60}")
        else:
            print(f"\n⚠️ No long_backward passes found in dataset")
    
    return passes

if __name__ == "__main__":
    passes = main()
