"""
Train xG Model
==============

Train a logistic regression xG model on Premier League 2024 shot data.
This model is used as the Bayesian prior for shot success probability estimation.

Run this script once before building team MDPs.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src import data_processing
from src import xg_model


def main():
    print("="*70)
    print("Training xG Model on Premier League 2024 Data")
    print("="*70)
    
    # 1. Load events
    print("\n1. Loading Premier League events...")
    data_dir = Path(__file__).parent.parent / "PremierLeague_data" / "2024" / "dynamic"
    
    events = data_processing.load_premier_league_events(
        data_dir=data_dir,
        limit_matches=None  # Use all matches
    )
    
    # 2. Extract and process shots
    print("\n2. Extracting shot events...")
    shots = data_processing.extract_shot_events(events)
    
    # 3. Rescale and normalize coordinates
    print("\n3. Processing coordinates...")
    shots = data_processing.rescale_coordinates(shots)
    shots = data_processing.normalize_attack_direction(shots)
    
    print(f"✅ Processed {len(shots):,} shots with normalized coordinates")
    
    # 4. Train model
    print("\n4. Training logistic regression xG model...")
    print("   Feature: angle to goal (degrees)")
    print("   Target: is_goal (0/1)")
    
    model, metrics = xg_model.train_logistic_xg_model(
        shots_df=shots,
        pitch_length=105,
        pitch_width=68,
        random_state=42
    )
    
    # 5. Save model
    print("\n5. Saving model...")
    save_path = Path(__file__).parent / 'data' / 'xg_model.pkl'
    xg_model.save_xg_model(model, metrics, save_path)
    
    # 6. Test model
    print("\n6. Testing model predictions...")
    print("\nSample xG predictions:")
    test_angles = [5, 10, 15, 20, 25, 30, 35, 40]
    for angle in test_angles:
        xg = xg_model.predict_xg(angle, model_path=save_path)
        print(f"   Angle {angle:2d}° → xG = {xg:.3f}")
    
    # Compare with old geometric model
    print("\nComparison with old geometric model:")
    print(f"{'Angle':>6} | {'Trained':>8} | {'Geometric':>10} | {'Diff':>8}")
    print("-" * 40)
    for angle in test_angles:
        trained_xg = xg_model.predict_xg(angle, model_path=save_path)
        geom_xg = xg_model.geometric_xg_model(angle)
        diff = trained_xg - geom_xg
        print(f"{angle:5d}° | {trained_xg:8.3f} | {geom_xg:10.3f} | {diff:+8.3f}")
    
    print("\n" + "="*70)
    print("✅ xG model training complete!")
    print(f"   Model saved to: {save_path}")
    print(f"   Ready to use in MDP construction")
    print("="*70)
    
    print("\nNext steps:")
    print("  1. Run build_all_team_mdps.py to rebuild MDPs with new xG prior")
    print("  2. Re-run analysis notebooks to see impact on results")


if __name__ == "__main__":
    main()
