"""
Verify final configuration with updated thresholds and action space.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import (
    load_premier_league_events,
    extract_pass_events,
    rescale_coordinates,
    normalize_attack_direction,
    classify_pass_type,
    validate_pass_classifications
)
from state_action import ACTION_NAMES, FieldGrid

def main():
    print("="*60)
    print("FINAL CONFIGURATION TEST")
    print("="*60)
    
    # Test action space
    print("\n1. ACTION SPACE")
    print(f"   Total actions: {len(ACTION_NAMES)}")
    for aid, aname in ACTION_NAMES.items():
        print(f"   {aid}: {aname}")
    
    # Test grid
    print("\n2. FIELD GRID")
    grid = FieldGrid(n_rows=22, n_cols=34)
    print(f"   Dimensions: {grid.n_rows} rows × {grid.n_cols} cols")
    print(f"   Total states: {grid.n_states}")
    print(f"   Cell size: {grid.cell_width:.2f}m × {grid.cell_height:.2f}m")
    
    # Load and process data
    print("\n3. DATA PROCESSING")
    print("   Loading events...")
    events = load_premier_league_events()
    
    print("   Extracting passes...")
    passes = extract_pass_events(events)
    print(f"   Raw passes: {len(passes):,}")
    
    print("   Rescaling coordinates...")
    passes = rescale_coordinates(passes)
    
    print("   Normalizing attack direction...")
    passes = normalize_attack_direction(passes)
    
    print("   Classifying pass types...")
    passes = classify_pass_type(passes)
    print(f"   Filtered passes: {len(passes):,}")
    
    # Validate
    print("\n4. VALIDATION")
    validate_pass_classifications(passes)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"CONFIGURATION SUMMARY")
    print(f"{'='*60}")
    print(f"Grid: {grid.n_rows}×{grid.n_cols} = {grid.n_states} states")
    print(f"Actions: {len(ACTION_NAMES)} (6 pass types + shoot + carry)")
    print(f"Passes: {len(passes):,}")
    print(f"Success rate: {passes['success'].mean()*100:.2f}%")
    print(f"State-action pairs: {grid.n_states} × {len(ACTION_NAMES)} = {grid.n_states * len(ACTION_NAMES)}")
    print(f"Avg observations per pair: {len(passes) / (grid.n_states * len(ACTION_NAMES)):.1f}")
    print(f"{'='*60}")
    
    return passes, grid

if __name__ == "__main__":
    passes, grid = main()
