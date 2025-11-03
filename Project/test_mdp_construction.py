"""
Test script for MDP construction
Tests building team-specific MDPs with the new implementation
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src import mdp
from src.state_action import create_action_availability_mask, FieldGrid

def main():
    print("="*70)
    print("Testing MDP Construction")
    print("="*70)
    
    # Load encoded actions
    print("\n1. Loading encoded actions...")
    actions_path = Path(__file__).parent / 'data' / 'actions_encoded.parquet'
    if not actions_path.exists():
        print(f"❌ Error: {actions_path} not found!")
        return
    
    actions = pd.read_parquet(actions_path)
    print(f"✅ Loaded {len(actions):,} actions")
    print(f"   Teams: {actions['team_id'].nunique()}")
    print(f"   Columns: {list(actions.columns[:10])}...")
    
    # Create action availability mask
    print("\n2. Creating action availability mask...")
    grid = FieldGrid(n_rows=22, n_cols=34)
    action_mask = create_action_availability_mask(grid, shoot_distance_threshold=30.0)
    print(f"✅ Mask shape: {action_mask.shape}")
    print(f"   Masked pairs: {(~action_mask).sum()} / {action_mask.size}")
    print(f"   Masked percentage: {(~action_mask).sum() / action_mask.size * 100:.1f}%")
    
    # Test building MDP for a single team (Manchester City - team_id=40)
    print("\n3. Testing single team MDP (Manchester City)...")
    print("-"*70)
    
    test_team_id = 40
    
    # Build transition matrix
    P = mdp.build_transition_matrix(
        actions=actions,
        n_states=748,
        n_actions=10,
        alpha=2.0,
        team_id=test_team_id,
        action_mask=action_mask
    )
    
    print(f"\nTransition matrix P shape: {P.shape}")
    print(f"Expected: (751, 10, 751)")
    
    # Build policy matrix
    pi = mdp.build_policy_matrix(
        actions=actions,
        n_states=748,
        n_actions=10,
        team_id=test_team_id,
        action_mask=action_mask
    )
    
    print(f"Policy matrix π shape: {pi.shape}")
    print(f"Expected: (748, 10)")
    
    # Build reward function
    R = mdp.build_reward_function(n_states=748)
    print(f"Reward vector R shape: {R.shape}")
    print(f"Expected: (751,)")
    print(f"Goal state reward (R[748]): {R[748]}")
    
    # Validate MDP
    print("\n4. Validating MDP...")
    print("-"*70)
    try:
        mdp.validate_mdp(P[:748, :, :], pi)
        print("✅ MDP validation passed!")
    except AssertionError as e:
        print(f"❌ Validation failed: {e}")
        return
    
    # Get statistics
    print("\n5. MDP Statistics...")
    print("-"*70)
    team_actions = actions[actions['team_id'] == test_team_id]
    stats = mdp.get_mdp_statistics(P, pi, team_actions)
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    # Test building MDPs for top 3 teams (for speed)
    print("\n6. Testing multi-team MDP construction (3 teams)...")
    print("="*70)
    
    top_teams = actions['team_id'].value_counts().head(3).index.tolist()
    print(f"Building MDPs for teams: {top_teams}")
    
    mdps = mdp.build_team_mdps(
        actions=actions,
        n_states=748,
        n_actions=10,
        alpha=2.0,
        action_mask=action_mask,
        team_ids=top_teams
    )
    
    print("\n7. Summary of built MDPs...")
    print("="*70)
    for team_id, mdp_data in mdps.items():
        print(f"\n{mdp_data['team_name']} (ID: {team_id}):")
        print(f"  Actions: {mdp_data['n_actions']:,}")
        print(f"  P shape: {mdp_data['P'].shape}")
        print(f"  π shape: {mdp_data['pi'].shape}")
        print(f"  R shape: {mdp_data['R'].shape}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)

if __name__ == "__main__":
    main()
