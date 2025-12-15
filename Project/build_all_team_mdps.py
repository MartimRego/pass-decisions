"""
Build and save MDPs for all 20 Premier League teams
Run this to generate the MDP models needed for analysis
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src import mdp
from src.state_action import create_action_availability_mask, FieldGrid

def main():
    start_time = time.time()
    
    print("="*70)
    print("Building MDPs for All Premier League 2024 Teams")
    print("="*70)
    
    # Load encoded actions
    print("\n1. Loading encoded actions...")
    actions_path = Path(__file__).parent / 'data' / 'actions_encoded.parquet'
    if not actions_path.exists():
        print(f"❌ Error: {actions_path} not found!")
        print("   Please run the main notebook first to generate encoded actions.")
        return
    
    actions = pd.read_parquet(actions_path)
    print(f"✅ Loaded {len(actions):,} actions")
    print(f"   Teams: {actions['team_id'].nunique()}")
    
    # Create action availability mask
    print("\n2. Creating action availability mask...")
    grid = FieldGrid(n_rows=22, n_cols=34)
    action_mask = create_action_availability_mask(grid, shoot_distance_threshold=30.0)
    print(f"✅ Mask shape: {action_mask.shape}")
    print(f"   Masked (invalid) pairs: {(~action_mask).sum()} / {action_mask.size}")
    
    # Build all team MDPs
    print("\n3. Building MDPs for all 20 teams...")
    print(f"   Laplace smoothing for passes/carries: α = 2.0")
    print(f"   Bayesian shrinkage for shots: α = 10.0")
    print("="*70)
    
    mdps = mdp.build_team_mdps(
        actions=actions,
        n_states=grid.n_states,
        n_actions=8,
        alpha=2.0,
        action_mask=action_mask,
        team_ids=None,  # Build for all teams
        grid=grid,
        bayesian_alpha=10.0,
        shoot_distance_threshold=30.0
    )
    
    # Save MDPs to disk
    print("\n4. Saving MDPs to disk...")
    save_dir = Path(__file__).parent / 'data' / 'team_mdps'
    mdp.save_team_mdps(mdps, save_dir)
    
    # Summary statistics
    print("\n5. Summary Statistics")
    print("="*70)
    
    # Aggregate statistics across all teams
    all_stats = []
    for team_id, mdp_data in mdps.items():
        team_actions = actions[actions['team_id'] == team_id]
        stats = mdp.get_mdp_statistics(mdp_data['P'], mdp_data['pi'], team_actions)
        stats['team_id'] = team_id
        stats['team_name'] = mdp_data['team_name']
        all_stats.append(stats)
    
    # Create summary DataFrame
    stats_df = pd.DataFrame(all_stats)
    stats_df = stats_df.sort_values('n_observations', ascending=False)
    
    print("\nTop 5 teams by data volume:")
    print(stats_df[['team_name', 'n_observations', 'state_action_coverage', 'avg_policy_entropy']].head())
    
    print("\nBottom 5 teams by data volume:")
    print(stats_df[['team_name', 'n_observations', 'state_action_coverage', 'avg_policy_entropy']].tail())
    
    print("\n\nOverall Statistics:")
    print(f"  Mean observations per team: {stats_df['n_observations'].mean():.0f}")
    print(f"  Mean state-action coverage: {stats_df['state_action_coverage'].mean():.1%}")
    print(f"  Mean policy entropy: {stats_df['avg_policy_entropy'].mean():.3f}")
    
    # Save statistics
    stats_path = save_dir / 'team_mdp_statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"\n✅ Statistics saved to: {stats_path}")
    
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print(f"✅ All 20 team MDPs built and saved successfully!")
    print(f"   Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"   Saved to: {save_dir}")
    print("="*70)
    
    # Test loading
    print("\n6. Testing MDP loading...")
    test_team_id = list(mdps.keys())[0]
    loaded_mdp = mdp.load_team_mdp(test_team_id, save_dir)
    print(f"✅ Successfully loaded {loaded_mdp['team_name']} MDP")
    print(f"   P shape: {loaded_mdp['P'].shape}")
    print(f"   π shape: {loaded_mdp['pi'].shape}")
    print(f"   R shape: {loaded_mdp['R'].shape}")
    
    print("\n" + "="*70)
    print("🎉 MDP construction complete! Ready for analysis.")
    print("="*70)

if __name__ == "__main__":
    main()
