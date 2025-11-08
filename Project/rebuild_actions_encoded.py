"""
Rebuild actions_encoded.parquet with correct action mapping and grid configuration.

This script:
1. Loads raw action data
2. Applies correct coordinate normalization
3. Encodes states/actions with correct mapping (shoot=6, carry=7)
4. Saves to data/actions_encoded.parquet

Run this after fixing normalize_attack_direction() to rebuild the MDP input data.
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import data_processing as dp
import state_action as sa

def main():
    print("=" * 80)
    print("REBUILDING actions_encoded.parquet")
    print("=" * 80)
    
    # 1. Load and process raw data
    print("\n1. Loading Premier League 2024 data...")
    data_path = Path(__file__).parent.parent / 'PremierLeague_data' / '2024'
    
    # Load actions (this loads actions_complete.parquet which has all data)
    actions_path = data_path / 'processed' / 'actions_complete.parquet'
    
    if not actions_path.exists():
        print(f"❌ Error: {actions_path} not found!")
        print("   Please run the notebook Section 2 to create this file first.")
        return 1
    
    actions = pd.read_parquet(actions_path)
    print(f"✅ Loaded {len(actions):,} actions")
    print(f"   Matches: {actions['match_id'].nunique()}")
    print(f"   Teams: {actions['team_id'].nunique()}")
    
    # Verify action types
    action_counts = actions['action_type'].value_counts()
    print(f"\n   Action types:")
    for action_type, count in action_counts.items():
        print(f"     {action_type}: {count:,}")
    
    # 2. Apply coordinate normalization (already done in actions_complete.parquet
    # but let's verify)
    print("\n2. Verifying coordinate normalization...")
    required_cols = ['x_start_norm', 'y_start_norm', 'x_end_norm', 'y_end_norm']
    if all(col in actions.columns for col in required_cols):
        print(f"✅ Normalized coordinates present")
    else:
        print(f"⚠️  Some normalized coordinates missing, applying normalization...")
        # This should not happen if actions_complete.parquet is up to date
        actions = dp.normalize_attack_direction(actions)
    
    # Also check for player_targeted coordinates for passes
    if 'player_targeted_x_reception' in actions.columns:
        if 'player_targeted_x_reception_norm' not in actions.columns:
            print(f"   Adding normalized reception coordinates...")
            # Apply same logic for reception coordinates
            actions['player_targeted_x_reception_norm'] = actions['player_targeted_x_reception_rescaled']
            actions['player_targeted_y_reception_norm'] = actions['player_targeted_y_reception_rescaled']
    
    # 3. Create field grid (7×11)
    print("\n3. Creating field grid...")
    grid = sa.FieldGrid(n_rows=7, n_cols=11)
    print(f"✅ Grid: {grid.n_rows}×{grid.n_cols} = {grid.n_states} states")
    print(f"   Cell size: {grid.cell_width:.2f}m × {grid.cell_height:.2f}m")
    
    # 4. Encode states and actions
    print("\n4. Encoding states and actions...")
    actions = sa.add_state_action_encoding(actions, grid)
    
    # Verify encoding
    print(f"\n✅ Encoding complete!")
    print(f"   Unique states: {actions['state_from'].nunique()}")
    print(f"   Unique actions: {actions['action'].nunique()}")
    
    # Show action distribution
    print(f"\n   Action distribution:")
    for action_id in sorted(actions['action'].unique()):
        count = (actions['action'] == action_id).sum()
        pct = count / len(actions) * 100
        action_name = sa.ACTION_NAMES.get(action_id, 'unknown')
        print(f"     {action_id}: {action_name:18s} - {count:7,} ({pct:5.2f}%)")
    
    # Check for encoding errors
    errors = actions[actions['action'] == -1]
    if len(errors) > 0:
        print(f"\n⚠️  WARNING: {len(errors)} actions failed to encode!")
        return 1
    
    # 5. Save encoded actions
    output_path = Path(__file__).parent / 'data' / 'actions_encoded.parquet'
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"\n5. Saving to {output_path}...")
    actions.to_parquet(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"SUCCESS!")
    print(f"{'='*80}")
    print(f"💾 Saved: {output_path}")
    print(f"📊 Size: {len(actions):,} actions")
    print(f"🎯 Grid: {grid.n_rows}×{grid.n_cols} = {grid.n_states} states")
    print(f"🎬 Actions: {actions['action'].nunique()} (shoot=6, carry=7)")
    print(f"\n✅ Ready for MDP construction!")
    print(f"{'='*80}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
