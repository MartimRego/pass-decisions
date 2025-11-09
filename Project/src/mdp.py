"""
MDP Construction Module
=======================

Functions for building the Markov Decision Process components:
transition matrix, policy, and reward function.

Author: Your Name
Date: November 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from scipy.sparse import csr_matrix
from . import xg_model


def build_transition_matrix(
    actions: pd.DataFrame,
    n_states: int = 77,
    n_actions: int = 8,
    alpha: float = 2.0,
    team_id: Optional[int] = None,
    action_mask: Optional[np.ndarray] = None,
    grid = None,
    bayesian_alpha: float = 10.0,
    shoot_distance_threshold: float = 30.0
) -> np.ndarray:
    """
    Build transition probability matrix P(s, a, s') with smoothing.
    
    Uses Laplace smoothing for passes/carries and Bayesian shrinkage for shots.
    
    Parameters
    ----------
    actions : pd.DataFrame
        Action events with state_from, action, state_to, success columns.
        Must include: state_from, action, state_to, success, action_type
        If team_id provided, must also have team_id column.
    n_states : int, default=77
        Number of field states (not including absorbing states)
    n_actions : int, default=8
        Number of actions (6 pass types + shoot + carry)
    alpha : float, default=2.0
        Laplace smoothing parameter for passes/carries. Higher = more smoothing.
        Recommended: 2-3 for team-specific MDPs, 1-2 for league-wide.
    team_id : int, optional
        If provided, only use actions from this team
    action_mask : np.ndarray, optional
        Boolean mask of shape (n_states, n_actions) indicating valid actions.
        If provided, invalid actions get zero probability.
    grid : FieldGrid, optional
        Grid object for Bayesian shrinkage calculation. Required for shooting priors.
    bayesian_alpha : float, default=10.0
        Prior strength for Bayesian shrinkage on shooting (equivalent sample size)
    shoot_distance_threshold : float, default=30.0
        Maximum distance from goal for shooting (meters)
        
    Returns
    -------
    np.ndarray, shape (n_states + 3, n_actions, n_states + 3)
        Transition probabilities. State indices:
        - 0 to n_states-1: field states (77 states for 7×11 grid)
        - n_states (77): goal (absorbing)
        - n_states+1 (78): no_goal (absorbing)
        - n_states+2 (79): loss_possession (absorbing)
        
    Notes
    -----
    For move actions (passes & carries):
        - P(s, a, s') = prob of successfully moving to s'
        - P(s, a, loss) = prob of losing possession
        - Uses Laplace smoothing on success counts
    
    For shoot action:
        - P(s, shoot, goal) = empirical goal rate from state s
        - P(s, shoot, no_goal) = 1 - goal rate
        
    Absorbing states have self-loops with probability 1.
    """
    # Filter by team if specified
    if team_id is not None:
        if 'team_id' not in actions.columns:
            raise ValueError("team_id column required when team_id parameter is provided")
        actions = actions[actions['team_id'] == team_id].copy()
        print(f"Building MDP for team {team_id}: {len(actions):,} actions")
    else:
        print(f"Building league-wide MDP: {len(actions):,} actions")
    
    # Total number of states including absorbing
    n_total = n_states + 3
    goal_state = n_states      # 77
    no_goal_state = n_states + 1  # 78
    loss_state = n_states + 2     # 79
    
    # Initialize transition matrix
    P = np.zeros((n_total, n_actions, n_total))
    
    # Set absorbing states (self-loops with probability 1)
    for absorbing_state in [goal_state, no_goal_state, loss_state]:
        P[absorbing_state, :, absorbing_state] = 1.0
    
    print("Processing transitions...")
    
    # Separate shoots from moves (passes & carries)
    shoots = actions[actions['action_type'] == 'shoot'].copy()
    moves = actions[actions['action_type'].isin(['pass', 'carry'])].copy()
    
    # 1. Handle shooting transitions with Bayesian shrinkage
    print(f"  Processing {len(shoots):,} shots...")
    print(f"  Applying Bayesian shrinkage (α={bayesian_alpha})...")
    
    if grid is not None:
        # Use Bayesian shrinkage with geometric xG prior
        shoot_probs = xg_model.apply_bayesian_shrinkage(
            actions_df=shoots,
            grid=grid,
            shoot_action_id=6,
            alpha=bayesian_alpha,
            shoot_distance_threshold=shoot_distance_threshold
        )
        
        for state in range(n_states):
            goal_prob = shoot_probs.get(state, 0.0)
            if goal_prob > 0:
                # Action 6 is shoot
                P[state, 6, goal_state] = goal_prob
                P[state, 6, no_goal_state] = 1.0 - goal_prob
    else:
        # Fallback to simple Laplace smoothing if grid not provided
        print("  WARNING: No grid provided, using Laplace smoothing for shots")
        for state in range(n_states):
            state_shots = shoots[shoots['state_from'] == state]
            if len(state_shots) > 0:
                n_goals = state_shots['success'].sum()
                n_total_shots = len(state_shots)
                goal_prob = (n_goals + alpha) / (n_total_shots + 2 * alpha)
                P[state, 6, goal_state] = goal_prob
                P[state, 6, no_goal_state] = 1.0 - goal_prob
    
    # 2. Handle move actions (passes & carries)
    print(f"  Processing {len(moves):,} moves (passes + carries)...")
    
    # Group by (state_from, action) to count transitions
    for (state, action_id), group in moves.groupby(['state_from', 'action']):
        if state >= n_states:  # Skip if somehow in absorbing state
            continue
        
        # Count successful moves to each destination
        successful = group[group['success'] == True]
        failed = group[group['success'] == False]
        
        # Count transitions to each next state
        destination_counts = successful['state_to'].value_counts()
        
        # Total attempts for this (state, action)
        n_attempts = len(group)
        n_successful = len(successful)
        n_failed = len(failed)
        
        # Build transition probabilities with Laplace smoothing
        # We add alpha to the count of each observed transition
        # And add alpha * (number of possible destinations + 1) to denominator
        # The +1 is for the loss_possession state
        n_destinations = len(destination_counts) + 1  # +1 for loss state
        
        for dest_state, count in destination_counts.items():
            if dest_state < n_states:  # Valid field state
                # Smoothed probability of reaching this destination
                P[state, action_id, dest_state] = (count + alpha) / (n_attempts + n_destinations * alpha)
        
        # Probability of losing possession (failed move)
        # Use actual failed count + smoothing
        P[state, action_id, loss_state] = (n_failed + alpha) / (n_attempts + n_destinations * alpha)
    
    # 3. Apply action masking if provided
    if action_mask is not None:
        print("  Applying action availability mask...")
        for state in range(n_states):
            for action_id in range(n_actions):
                if not action_mask[state, action_id]:
                    # Zero out this (state, action) - make it invalid
                    P[state, action_id, :] = 0.0
    
    # 4. Normalize each (state, action) to sum to 1
    print("  Normalizing probabilities...")
    for state in range(n_states):
        for action_id in range(n_actions):
            row_sum = P[state, action_id, :].sum()
            if row_sum > 0:
                P[state, action_id, :] /= row_sum
            elif action_mask is None or action_mask[state, action_id]:
                # No data for this (state, action) but it's valid
                # Assign small uniform probability (mostly to loss state)
                P[state, action_id, loss_state] = 1.0
    
    print("✅ Transition matrix built successfully!")
    return P


def build_policy_matrix(
    actions: pd.DataFrame,
    n_states: int = 77,
    n_actions: int = 8,
    team_id: Optional[int] = None,
    action_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Build policy matrix π(a | s) from observed action frequencies.
    
    Parameters
    ----------
    actions : pd.DataFrame
        Action events with state_from and action columns.
        If team_id provided, must also have team_id column.
    n_states : int, default=77
        Number of field states
    n_actions : int, default=8
        Number of actions
    team_id : int, optional
        If provided, only use actions from this team
    action_mask : np.ndarray, optional
        Boolean mask of shape (n_states, n_actions) indicating valid actions.
        Invalid actions get zero probability.
        
    Returns
    -------
    np.ndarray, shape (n_states, n_actions)
        Policy probabilities. Each row sums to 1 (over valid actions).
        
    Notes
    -----
    π(a | s) = count(s, a) / sum_over_valid_actions(count(s, *))
    
    For states with no observations, uses uniform distribution over
    valid actions (according to action_mask if provided).
    """
    # Filter by team if specified
    if team_id is not None:
        if 'team_id' not in actions.columns:
            raise ValueError("team_id column required when team_id parameter is provided")
        actions = actions[actions['team_id'] == team_id].copy()
    
    # Initialize policy matrix
    pi = np.zeros((n_states, n_actions))
    
    print("Building policy matrix...")
    
    # Count action frequencies for each state
    for state in range(n_states):
        state_actions = actions[actions['state_from'] == state]
        
        if len(state_actions) > 0:
            # Count each action type
            action_counts = state_actions['action'].value_counts()
            
            for action_id, count in action_counts.items():
                if action_id < n_actions:  # Valid action
                    pi[state, action_id] = count
            
            # Normalize to probabilities (only over valid actions if mask provided)
            if action_mask is not None:
                # Zero out invalid actions
                pi[state, :] *= action_mask[state, :]
            
            # Normalize row to sum to 1
            row_sum = pi[state, :].sum()
            if row_sum > 0:
                pi[state, :] /= row_sum
            else:
                # No valid actions observed - uniform over valid actions
                if action_mask is not None:
                    valid_actions = action_mask[state, :]
                    n_valid = valid_actions.sum()
                    if n_valid > 0:
                        pi[state, :] = valid_actions / n_valid
                    else:
                        # No valid actions at all (shouldn't happen)
                        pi[state, :] = 1.0 / n_actions
                else:
                    pi[state, :] = 1.0 / n_actions
        else:
            # No observations for this state - uniform distribution
            if action_mask is not None:
                valid_actions = action_mask[state, :]
                n_valid = valid_actions.sum()
                if n_valid > 0:
                    pi[state, :] = valid_actions / n_valid
                else:
                    pi[state, :] = 1.0 / n_actions
            else:
                pi[state, :] = 1.0 / n_actions
    
    print("✅ Policy matrix built successfully!")
    return pi


def build_reward_function(
    n_states: int = 77
) -> np.ndarray:
    """
    Build reward function R(s).
    
    Parameters
    ----------
    n_states : int, default=77
        Number of field states
        
    Returns
    -------
    np.ndarray, shape (n_states + 3,)
        Rewards: 1.0 for goal state, 0.0 elsewhere
        
    Notes
    -----
    Simplified reward aligned with Van Roy et al. methodology:
    - R(goal_state) = 1.0  (state 77)
    - R(no_goal_state) = 0.0  (state 78)
    - R(loss_possession_state) = 0.0  (state 79)
    - R(field_states) = 0.0  (states 0-76)
    
    This creates a binary outcome: only scoring a goal receives reward.
    """
    # Total states: field states + 3 absorbing states
    rewards = np.zeros(n_states + 3)
    rewards[n_states] = 1.0  # Goal state (index 77)
    
    return rewards


def build_team_mdps(
    actions: pd.DataFrame,
    n_states: int = 77,
    n_actions: int = 8,
    alpha: float = 2.0,
    action_mask: Optional[np.ndarray] = None,
    team_ids: Optional[list] = None,
    grid = None,
    bayesian_alpha: float = 10.0,
    shoot_distance_threshold: float = 30.0
) -> dict:
    """
    Build separate MDP for each team in the dataset.
    
    Parameters
    ----------
    actions : pd.DataFrame
        All actions with team_id column
    n_states : int, default=77
        Number of field states
    n_actions : int, default=8
        Number of actions
    alpha : float, default=2.0
        Laplace smoothing parameter for passes/carries
    action_mask : np.ndarray, optional
        Action availability mask to apply to all teams
    team_ids : list, optional
        List of team IDs to build MDPs for. If None, builds for all teams.
    grid : FieldGrid, optional
        Grid object for Bayesian shrinkage on shooting
    bayesian_alpha : float, default=10.0
        Prior strength for Bayesian shrinkage on shooting
    shoot_distance_threshold : float, default=30.0
        Distance threshold for shooting
        
    Returns
    -------
    dict
        Dictionary mapping team_id to MDP components:
        {
            team_id: {
                'P': transition matrix,
                'pi': policy matrix,
                'R': reward vector,
                'team_name': team shortname,
                'n_actions': number of actions for this team
            }
        }
    """
    if 'team_id' not in actions.columns:
        raise ValueError("actions DataFrame must have team_id column")
    
    # Get all unique teams if not specified
    if team_ids is None:
        team_ids = sorted(actions['team_id'].unique())
    
    # Get team names if available
    team_names = {}
    if 'team_shortname' in actions.columns:
        team_mapping = actions[['team_id', 'team_shortname']].drop_duplicates()
        team_names = dict(zip(team_mapping['team_id'], team_mapping['team_shortname']))
    
    mdps = {}
    
    print(f"\n{'='*70}")
    print(f"Building MDPs for {len(team_ids)} teams")
    print(f"{'='*70}\n")
    
    for i, team_id in enumerate(team_ids, 1):
        team_name = team_names.get(team_id, f"Team {team_id}")
        print(f"\n[{i}/{len(team_ids)}] {team_name} (ID: {team_id})")
        print("-" * 70)
        
        # Build transition matrix for this team
        P = build_transition_matrix(
            actions, n_states, n_actions, alpha, team_id, action_mask,
            grid, bayesian_alpha, shoot_distance_threshold
        )
        
        # Build policy matrix for this team
        pi = build_policy_matrix(
            actions, n_states, n_actions, team_id, action_mask
        )
        
        # Reward function is same for all teams
        R = build_reward_function(n_states)
        
        # Validate the MDP
        try:
            validate_mdp(P[:n_states, :, :], pi)
        except AssertionError as e:
            print(f"⚠️  Validation warning for {team_name}: {e}")
        
        # Store MDP components
        mdps[team_id] = {
            'P': P,
            'pi': pi,
            'R': R,
            'team_name': team_name,
            'n_actions': (actions['team_id'] == team_id).sum()
        }
        
        print(f"✅ MDP complete for {team_name}\n")
    
    print(f"\n{'='*70}")
    print(f"✅ All {len(team_ids)} team MDPs built successfully!")
    print(f"{'='*70}\n")
    
    return mdps


def validate_mdp(
    P: np.ndarray,
    pi: np.ndarray,
    tolerance: float = 1e-5
) -> bool:
    """
    Validate MDP construction (probability constraints).
    
    Parameters
    ----------
    P : np.ndarray
        Transition matrix
    pi : np.ndarray
        Policy matrix
    tolerance : float
        Numerical tolerance for probability sums
        
    Returns
    -------
    bool
        True if valid, raises AssertionError otherwise
        
    Checks
    ------
    1. All probabilities in [0, 1]
    2. Transition probabilities sum to 1 for each (s, a)
    3. Policy probabilities sum to 1 for each s
    4. No NaN or Inf values
    """
    print("Validating MDP...")
    
    # Check bounds
    assert np.all(P >= 0) and np.all(P <= 1), "Transition probs not in [0,1]"
    assert np.all(pi >= 0) and np.all(pi <= 1), "Policy probs not in [0,1]"
    
    # Check normalization
    P_sums = P.sum(axis=2)  # Sum over next states
    # Only check non-zero rows (masked actions will be all zeros)
    non_zero_mask = P_sums > 0
    if non_zero_mask.any():
        non_zero_sums = P_sums[non_zero_mask]
        assert np.allclose(non_zero_sums, 1.0, atol=tolerance), \
            f"Transitions don't sum to 1: min={non_zero_sums.min()}, max={non_zero_sums.max()}"
    
    pi_sums = pi.sum(axis=1)  # Sum over actions
    assert np.allclose(pi_sums, 1.0, atol=tolerance), \
        f"Policy doesn't sum to 1: min={pi_sums.min()}, max={pi_sums.max()}"
    
    # Check for invalid values
    assert not np.any(np.isnan(P)), "NaN in transition matrix"
    assert not np.any(np.isnan(pi)), "NaN in policy matrix"
    assert not np.any(np.isinf(P)), "Inf in transition matrix"
    assert not np.any(np.isinf(pi)), "Inf in policy matrix"
    
    print("✓ MDP validation passed!")
    return True


def get_mdp_statistics(P: np.ndarray, pi: np.ndarray, actions: pd.DataFrame = None) -> dict:
    """
    Compute summary statistics of the learned MDP.
    
    Parameters
    ----------
    P : np.ndarray
        Transition matrix (n_states + 3, n_actions, n_states + 3)
    pi : np.ndarray
        Policy matrix (n_states, n_actions)
    actions : pd.DataFrame, optional
        Original actions data for additional statistics
        
    Returns
    -------
    dict
        Statistics including sparsity, entropy, coverage, etc.
    """
    n_states = pi.shape[0]
    n_actions = pi.shape[1]
    
    # Basic structure stats
    stats = {
        'n_states': n_states,
        'n_actions': n_actions,
        'n_total_states': P.shape[0],
    }
    
    # Transition matrix statistics (only field states)
    P_field = P[:n_states, :, :]
    stats['transition_sparsity'] = (P_field == 0).sum() / P_field.size
    stats['transition_nonzero'] = (P_field > 0).sum()
    
    # Policy matrix statistics
    stats['policy_sparsity'] = (pi == 0).sum() / pi.size
    stats['policy_nonzero'] = (pi > 0).sum()
    
    # Policy entropy (measure of randomness)
    # Higher entropy = more uniform policy, lower = more deterministic
    epsilon = 1e-10
    entropy_per_state = -np.sum(pi * np.log(pi + epsilon), axis=1)
    stats['avg_policy_entropy'] = entropy_per_state.mean()
    stats['max_policy_entropy'] = entropy_per_state.max()
    stats['min_policy_entropy'] = entropy_per_state.min()
    
    # Action usage statistics
    total_action_prob = pi.sum(axis=0)
    stats['most_used_action'] = int(np.argmax(total_action_prob))
    stats['least_used_action'] = int(np.argmin(total_action_prob))
    
    # State coverage
    states_with_actions = (pi.sum(axis=1) > 0).sum()
    stats['state_coverage'] = states_with_actions / n_states
    stats['states_with_data'] = int(states_with_actions)
    
    # If original data provided, compute additional stats
    if actions is not None:
        stats['n_observations'] = len(actions)
        stats['obs_per_state'] = len(actions) / n_states
        
        # State-action pair coverage
        if 'state_from' in actions.columns and 'action' in actions.columns:
            unique_pairs = actions[['state_from', 'action']].drop_duplicates()
            stats['state_action_pairs_observed'] = len(unique_pairs)
            stats['state_action_coverage'] = len(unique_pairs) / (n_states * n_actions)
    
    return stats


def save_team_mdps(mdps: dict, save_dir: Path) -> None:
    """
    Save team MDPs to disk.
    
    Parameters
    ----------
    mdps : dict
        Dictionary of team MDPs from build_team_mdps()
    save_dir : Path
        Directory to save MDP files
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving {len(mdps)} team MDPs to {save_dir}...")
    
    for team_id, mdp_data in mdps.items():
        team_name = mdp_data['team_name'].replace(' ', '_')
        
        # Save each component
        np.save(save_dir / f"P_{team_id}_{team_name}.npy", mdp_data['P'])
        np.save(save_dir / f"pi_{team_id}_{team_name}.npy", mdp_data['pi'])
        np.save(save_dir / f"R_{team_id}_{team_name}.npy", mdp_data['R'])
    
    # Save metadata
    metadata = {
        int(team_id): {  # Convert numpy int64 to Python int
            'team_name': mdp_data['team_name'],
            'n_actions': int(mdp_data['n_actions'])  # Convert to Python int
        }
        for team_id, mdp_data in mdps.items()
    }
    
    import json
    with open(save_dir / 'team_mdp_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved {len(mdps)} team MDPs!")


def load_team_mdp(team_id: int, load_dir: Path) -> dict:
    """
    Load a single team's MDP from disk.
    
    Parameters
    ----------
    team_id : int
        Team ID to load
    load_dir : Path
        Directory containing saved MDPs
        
    Returns
    -------
    dict
        MDP components for the team
    """
    load_dir = Path(load_dir)
    
    # Load metadata to get team name
    import json
    with open(load_dir / 'team_mdp_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    team_info = metadata[str(team_id)]
    team_name = team_info['team_name'].replace(' ', '_')
    
    # Load MDP components
    P = np.load(load_dir / f"P_{team_id}_{team_name}.npy")
    pi = np.load(load_dir / f"pi_{team_id}_{team_name}.npy")
    R = np.load(load_dir / f"R_{team_id}_{team_name}.npy")
    
    return {
        'P': P,
        'pi': pi,
        'R': R,
        'team_name': team_info['team_name'],
        'n_actions': team_info['n_actions']
    }


if __name__ == "__main__":
    print("MDP Construction Module - Skeleton Created")
    print("Ready for implementation!")
