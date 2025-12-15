"""
Analysis Module
===============

Functions for MDP analysis: fundamental matrix, expected goals,
optimal actions, and counterfactual policy evaluation.

Author: Your Name
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.linalg import inv


def compute_fundamental_matrix(
    P: np.ndarray,
    pi: np.ndarray
) -> np.ndarray:
    """
    Compute the fundamental matrix N = (I - Q)^(-1).
    
    Parameters
    ----------
    P : np.ndarray, shape (n_states, n_actions, n_states + 3)
        Transition probability matrix
    pi : np.ndarray, shape (n_states, n_actions)
        Policy matrix
        
    Returns
    -------
    np.ndarray, shape (n_states, n_states)
        Fundamental matrix N, where N[i,j] is expected number of visits
        to state j before absorption, starting from state i
        
    Notes
    -----
    Q[i,j] = sum_a P(i, a, j) * π(a | i)
    N = (I - Q)^(-1)
    
    Each entry N[i,j] represents the expected number of times the system
    will be in state j given that it started in state i.
    """
    # Number of transient (field) states
    n_transient = pi.shape[0]

    # Efficient vectorized construction of Q:
    # P has shape (n_states_total, n_actions, n_states_total).
    # We only need transient->transient block: P[:n_transient, :, :n_transient]
    # Q[s, s'] = sum_a P[s, a, s'] * pi[s, a]
    P_block = P[:n_transient, :, :n_transient]  # shape (n_transient, n_actions, n_transient)

    # Multiply by policy probabilities and sum over actions (axis=1)
    # Broadcasting: pi[:, :, None] has shape (n_transient, n_actions, 1)
    Q = (P_block * pi[:, :, None]).sum(axis=1)

    # Compute fundamental matrix N = (I - Q)^{-1}
    I = np.eye(n_transient)
    N = inv(I - Q)

    return N


def expected_goals_from_state(state, N, pi, P, R):
    """
    Calculate the expected number of goals starting from a given state.
    
    Uses the fundamental matrix N to account for all possible future paths
    from the current state, weighted by the policy π.
    
    Args:
        state: Starting state index (0-76 for field states)
        N: Fundamental matrix (n_transient x n_transient), where N[i,j] gives
           the expected number of times state j is visited starting from state i
        pi: Policy matrix (n_transient x n_actions) giving action probabilities
        P: Transition probability tensor (n_states x n_actions x n_states)
        R: Reward vector (n_states,) where R[goal_state] = 1
        
    Returns:
        Expected number of goals starting from the given state
    """
    # Vectorized computation for expected goals from a given start state.
    # Precompute per-state probability of scoring from that state under policy:
    # score_prob[s'] = sum_a pi[s', a] * P[s', a, goal_state]
    n_transient = N.shape[0]
    goal_state = np.where(R == 1)[0][0]

    # P_block for goal probabilities: shape (n_transient, n_actions)
    goal_probs = P[:n_transient, :, goal_state]

    # Per-state scoring probability under policy
    score_prob = (pi * goal_probs).sum(axis=1)  # shape (n_transient,)

    # Expected goals from 'state' is dot product of N[state, :] and score_prob
    expected_goals = float(np.dot(N[state, :], score_prob))

    return expected_goals
def expected_goals_total(
    N: np.ndarray,
    pi: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    possession_starts: np.ndarray
) -> float:
    """
    Calculate expected goals per possession start.
    
    Weights the expected goals from each state by how often possessions
    start in that state. This correctly accounts for possession sequences:
    each possession contains multiple sequential actions, so we weight by
    possession starts (first action in possession), not all actions.
    
    Args:
        N: Fundamental matrix (n_transient x n_transient)
        pi: Policy matrix (n_transient x n_actions)
        P: Transition probability tensor (n_states x n_actions x n_states)
        R: Reward vector (n_states,)
        possession_starts: Distribution over possession starting states (n_transient,)
                          Should sum to 1.0. Compute from actions where
                          first_player_possession_in_team_possession == True
        
    Returns:
        Expected goals per possession start
        
    Notes:
        Multiply result by number of possessions (not total actions) to get
        total expected goals for a season.
    """
    n_transient = N.shape[0]

    # Defensive check: possession_starts must have length equal to number of
    # transient (field) states. A common error is creating possession_starts
    # using the old grid constant (e.g. 77) instead of deriving it from the
    # MDP / fundamental matrix. Provide a helpful error message if mismatch.
    if possession_starts.shape[0] != n_transient:
        raise ValueError(
            f"possession_starts length ({possession_starts.shape[0]}) != "
            f"expected n_transient ({n_transient}).\n"
            "This usually means `possession_starts` was created with the old "
            "grid size (77) instead of using the current MDP shape.\n"
            "Fix: recreate possession_starts to match the MDP transient state "
            "count, e.g.:\n"
            "  n_total = P.shape[0]\n"
            "  n_transient = n_total - 3\n"
            "  possession_starts = np.ones(n_transient) / n_transient\n"
            "Or compute an empirical distribution from actions (bincount over "
            "state indices)."
        )

    # Vectorized: compute per-state expected goals once, then weight by possession_starts
    goal_state = np.where(R == 1)[0][0]

    # Per-state scoring probability under policy (shape: n_transient,)
    goal_probs = P[:n_transient, :, goal_state]
    score_prob = (pi * goal_probs).sum(axis=1)

    # Expected goals from each starting state: eg_state = N @ score_prob
    eg_per_state = N.dot(score_prob)

    # Weighted by possession start distribution
    expected_goals_per_possession = float(np.dot(possession_starts, eg_per_state))

    return expected_goals_per_possession


def optimal_action_per_state(
    N: np.ndarray,
    pi: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    n_actions: int,
    action_mask: np.ndarray = None
) -> np.ndarray:
    """
    Determine optimal action for each state.
    
    Parameters
    ----------
    N : np.ndarray
        Fundamental matrix
    pi : np.ndarray
        Policy matrix
    P : np.ndarray
        Transition matrix
    R : np.ndarray
        Reward vector
    n_actions : int
        Number of actions
    action_mask : np.ndarray, optional
        Boolean mask of shape (n_states, n_actions) indicating valid actions.
        If provided, only considers valid actions when finding optimal.
        
    Returns
    -------
    np.ndarray, shape (n_states,)
        Optimal action ID for each state
        
    Notes
    -----
    For each state, computes E[goals | s, a] for all actions a,
    and selects the action with maximum expected goals.
    Only considers valid actions according to action_mask if provided.
    """
    n_states = N.shape[0]
    optimal_actions = np.zeros(n_states, dtype=int)

    # Precompute quantities reused across (state,action) computations
    # Transient block of P: shape (n_states, n_actions, n_states)
    P_block = P[:n_states, :, :n_states]

    # Original Q and score probability under original policy
    Q = (P_block * pi[:, :, None]).sum(axis=1)  # shape (n_states, n_states)
    goal_state = np.where(R == 1)[0][0]
    goal_probs = P[:n_states, :, goal_state]  # shape (n_states, n_actions)
    score_prob_orig = (pi * goal_probs).sum(axis=1)  # shape (n_states,)

    # Precompute N @ score_prob_orig for use in Sherman-Morrison formulas
    w_orig = N.dot(score_prob_orig)  # shape (n_states,)

    best_eg_arr = np.full(n_states, -np.inf)

    for s in range(n_states):
        Ns_s = N[s, s]
        Ns_col = N[:, s]  # column s of N

        # Base expected goals from state s under original policy
        base_eg = float(np.dot(N[s, :], score_prob_orig))

        best_action = -1
        best_eg = -np.inf

        for a in range(n_actions):
            if action_mask is not None and not action_mask[s, a]:
                continue

            # New row for Q when forcing action a at state s
            new_row = P_block[s, a, :]
            v = new_row - Q[s, :]

            # Sherman-Morrison components
            vTNs = float(v.dot(Ns_col))
            denom = 1.0 - vTNs
            if abs(denom) < 1e-12:
                denom = np.sign(denom) * 1e-12 if denom != 0 else 1e-12

            vTworig = float(v.dot(w_orig))

            # Change in per-state scoring probability at s due to forcing action a
            delta_score_s = float(goal_probs[s, a] - score_prob_orig[s])

            # Expected goals with rank-1 update (derived from Sherman-Morrison)
            eg = base_eg + (Ns_s / denom) * vTworig + (Ns_s / denom) * delta_score_s

            if eg > best_eg:
                best_eg = eg
                best_action = a

        optimal_actions[s] = best_action if best_action is not None else -1
        best_eg_arr[s] = best_eg

    return optimal_actions, best_eg_arr


def eg_per_state_action_matrix(
    N: np.ndarray,
    pi: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    n_actions: int,
    action_mask: np.ndarray = None
) -> np.ndarray:
    """
    Compute E[goals | state, action] for every (state, action) pair efficiently.

    Uses the same Sherman-Morrison based algebra as `optimal_action_per_state`
    but returns the full matrix of expected goals for all valid actions.

    Parameters
    ----------
    N : np.ndarray
        Fundamental matrix (n_states x n_states)
    pi : np.ndarray
        Original policy (n_states x n_actions)
    P : np.ndarray
        Transition tensor (n_states_total x n_actions x n_states_total)
    R : np.ndarray
        Reward vector
    n_actions : int
        Number of actions
    action_mask : np.ndarray, optional
        Boolean mask of valid actions (n_states x n_actions)

    Returns
    -------
    np.ndarray
        Matrix of shape (n_states, n_actions) with expected goals for
        each state-action. Invalid actions are filled with -1.0.
    """
    n_states = N.shape[0]

    # Prepare transient block and quantities reused
    P_block = P[:n_states, :, :n_states]
    Q = (P_block * pi[:, :, None]).sum(axis=1)
    goal_state = np.where(R == 1)[0][0]
    goal_probs = P[:n_states, :, goal_state]  # shape (n_states, n_actions)
    score_prob_orig = (pi * goal_probs).sum(axis=1)
    w_orig = N.dot(score_prob_orig)

    eg_mat = np.full((n_states, n_actions), -1.0, dtype=float)

    for s in range(n_states):
        Ns_s = N[s, s]
        Ns_col = N[:, s]
        base_eg = float(np.dot(N[s, :], score_prob_orig))

        for a in range(n_actions):
            if action_mask is not None and not action_mask[s, a]:
                continue

            new_row = P_block[s, a, :]
            v = new_row - Q[s, :]

            vTNs = float(v.dot(Ns_col))
            denom = 1.0 - vTNs
            if abs(denom) < 1e-12:
                denom = 1e-12

            vTworig = float(v.dot(w_orig))
            delta_score_s = float(goal_probs[s, a] - score_prob_orig[s])

            eg = base_eg + (Ns_s / denom) * vTworig + (Ns_s / denom) * delta_score_s
            eg_mat[s, a] = eg

    return eg_mat


def create_uniform_adjustment(
    action_ids: List[int],
    change_pct: float,
    grid,
    states: Optional[List[int]] = None,
    col_range: Optional[Tuple[int, int]] = None,
    action_mask: Optional[np.ndarray] = None
) -> Dict[Tuple[int, int], float]:
    """
    Create uniform policy adjustment for specific action(s) across states.
    
    Parameters
    ----------
    action_ids : list of int
        Action IDs to increase/decrease (e.g., [3, 5] for both long passes)
    change_pct : float
        Percentage change as decimal (e.g., 0.20 for +20%)
    grid : FieldGrid
        Grid object for spatial constraints
    states : list of int, optional
        Specific states to apply change to. If None, uses col_range.
    col_range : tuple of int, optional
                # Vectorized per-state impact computation
                n_transient = N_original.shape[0]

                # Score probability per state under original and modified policies
                goal_state = np.where(R == 1)[0][0]
                goal_probs_orig = P[:n_transient, :, goal_state]
                score_prob_orig = (pi_original * goal_probs_orig).sum(axis=1)

                goal_probs_mod = P[:n_transient, :, goal_state]
                score_prob_mod = (pi_modified * goal_probs_mod).sum(axis=1)

                # Expected goals per state under each policy
                eg_per_state_orig = N_original.dot(score_prob_orig)
                eg_per_state_mod = N_modified.dot(score_prob_mod)

                # Difference weighted by possession starts
                goal_diff_per_state = (eg_per_state_mod - eg_per_state_orig) * possession_starts[:n_transient]
    --------
    >>> # Increase both long passes by 20% in midfield (columns 6-9)
    >>> mods = create_uniform_adjustment(
    ...     action_ids=[3, 5],  # long_backward, long_forward
    ...     change_pct=0.20,
    ...     grid=grid,
    ...     col_range=(5, 8)  # columns 6-9 in 1-indexed
    ... )
    """
    modifications = {}
    
    # Determine which states to modify
    if states is not None:
        target_states = states
    elif col_range is not None:
        min_col, max_col = col_range
        target_states = []
        for state in range(grid.n_states):
            col = state % grid.n_cols
            if min_col <= col <= max_col:
                target_states.append(state)
    else:
        # Apply to all states
        target_states = list(range(grid.n_states))
    
    # Create modifications for each (state, action) pair
    for state in target_states:
        for action_id in action_ids:
            # Skip if action is masked (invalid) for this state
            if action_mask is not None and not action_mask[state, action_id]:
                continue
            
            modifications[(state, action_id)] = change_pct
    
    return modifications


def modify_policy(
    pi: np.ndarray,
    modifications: Dict[Tuple[int, int], float]
) -> np.ndarray:
    """
    Create modified policy π' with specified changes.
    
    Handles multiple actions being modified in the same state by:
    1. Grouping modifications by state
    2. Computing total probability increase for that state
    3. Redistributing remaining probability proportionally to unchanged actions
    
    Parameters
    ----------
    pi : np.ndarray
        Original policy matrix
    modifications : dict
        Dictionary of {(state, action): change_fraction}
        e.g., {(100, 6): 0.20} means increase action 6 by 20% in state 100
        Can have multiple actions per state
        
    Returns
    -------
    np.ndarray
        Modified policy π'
        
    Notes
    -----
    When increasing multiple actions in same state:
        1. Calculate new probabilities for modified actions: π'(a|s) = π(a|s) * (1 + x)
        2. Sum total increase: Δ = sum(π'(a|s) - π(a|s)) for modified actions
        3. Redistribute to unchanged actions: π'(a'|s) = π(a'|s) * (1 - Δ) / sum(π(a'|s))
        4. Normalize to ensure sum = 1
    """
    pi_modified = pi.copy()
    
    # Group modifications by state
    state_modifications = {}
    for (state, action), change in modifications.items():
        if state not in state_modifications:
            state_modifications[state] = []
        state_modifications[state].append((action, change))
    
    # Process each state that has modifications
    for state, action_changes in state_modifications.items():
        # Track which actions are being modified
        modified_actions = {action for action, _ in action_changes}
        
        # Calculate new probabilities for modified actions
        total_increase = 0.0
        for action, change in action_changes:
            original_prob = pi[state, action]
            new_prob = original_prob * (1 + change)
            increase = new_prob - original_prob
            total_increase += increase
            pi_modified[state, action] = new_prob
        
        # Redistribute probability to unchanged actions
        unchanged_actions = [a for a in range(pi.shape[1]) if a not in modified_actions]
        total_unchanged = sum(pi[state, a] for a in unchanged_actions)
        
        if total_unchanged > 0 and len(unchanged_actions) > 0:
            # Scale down unchanged actions proportionally
            remaining_prob = 1.0 - sum(pi_modified[state, a] for a in modified_actions)
            scale_factor = remaining_prob / total_unchanged
            
            for a in unchanged_actions:
                pi_modified[state, a] = pi[state, a] * scale_factor
        
        # Normalize to ensure sum = 1 (handles numerical errors)
        row_sum = pi_modified[state, :].sum()
        if row_sum > 0:
            pi_modified[state, :] /= row_sum
    
    return pi_modified


def compute_spatial_impact(
    pi_original: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    possession_starts: np.ndarray,
    action_ids: List[int],
    change_pct: float,
    grid,
    action_mask: Optional[np.ndarray] = None,
    col_range: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, float, float]:
    """
    Compute expected goals difference for each grid cell from policy change.
    
    Parameters
    ----------
    pi_original : np.ndarray
        Original policy
    P : np.ndarray
        Transition matrix
    R : np.ndarray
        Reward vector
    possession_starts : np.ndarray
        Possession start distribution
    action_ids : list of int
        Actions to modify
    change_pct : float
        Percentage change (0.20 for +20%)
    grid : FieldGrid
        Grid object
    action_mask : np.ndarray, optional
        Action availability mask
    col_range : tuple of int, optional
        Column range to restrict modifications
        
    Returns
    -------
    goal_diff_grid : np.ndarray, shape (n_rows, n_cols)
        Goal difference for each grid cell
    total_diff : float
        Total expected goals difference
    baseline_goals : float
        Baseline expected goals
    """
    # Baseline expected goals
    N_original = compute_fundamental_matrix(P, pi_original)
    eg_baseline = expected_goals_total(N_original, pi_original, P, R, possession_starts)
    
    # Create modifications
    modifications = create_uniform_adjustment(
        action_ids=action_ids,
        change_pct=change_pct,
        grid=grid,
        col_range=col_range,
        action_mask=action_mask
    )
    
    # Modified policy
    pi_modified = modify_policy(pi_original, modifications)
    
    # Modified expected goals
    N_modified = compute_fundamental_matrix(P, pi_modified)
    eg_modified = expected_goals_total(N_modified, pi_modified, P, R, possession_starts)
    
    total_diff = eg_modified - eg_baseline
    
    # Compute per-state impact (how much does each state contribute to difference?)
    # This requires computing expected goals starting from each state
    goal_diff_per_state = np.zeros(grid.n_states)
    
    for state in range(grid.n_states):
        # Expected goals from this state under original policy
        eg_from_state_orig = expected_goals_from_state(state, N_original, pi_original, P, R)
        
        # Expected goals from this state under modified policy
        eg_from_state_mod = expected_goals_from_state(state, N_modified, pi_modified, P, R)
        
        # Difference, weighted by how often possessions start in this state
        goal_diff_per_state[state] = (eg_from_state_mod - eg_from_state_orig) * possession_starts[state]
    
    # Reshape to grid
    goal_diff_grid = goal_diff_per_state.reshape(grid.n_rows, grid.n_cols)
    
    return goal_diff_grid, total_diff, eg_baseline


def counterfactual_analysis(
    pi_original: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    possession_starts: np.ndarray,
    scenarios: Dict[str, Dict[Tuple[int, int], float]],
    quality_distributions: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Run counterfactual policy experiments.
    
    Parameters
    ----------
    pi_original : np.ndarray
        Original observed policy
    P : np.ndarray
        Transition matrix (will be modified if quality_distributions provided)
    R : np.ndarray
        Reward vector
    possession_starts : np.ndarray
        Possession start counts per state
    scenarios : dict
        Dictionary of {scenario_name: {(state, action): change}}
    quality_distributions : dict, optional
        Quality statistics for success rate adjustment
        
    Returns
    -------
    pd.DataFrame
        Results with columns: scenario, expected_goals, difference, pct_change
        
    Examples
    --------
    >>> scenarios = {
    ...     'baseline': {},
    ...     'more_long_forward_mid': {(s, 6): 0.20 for s in range(150, 200)}
    ... }
    >>> results = counterfactual_analysis(pi, P, R, poss_starts, scenarios)
    """
    results = []
    
    # Baseline
    N_original = compute_fundamental_matrix(P, pi_original)
    eg_baseline = expected_goals_total(N_original, pi_original, P, R, possession_starts)
    
    for scenario_name, modifications in scenarios.items():
        if not modifications:  # Baseline scenario
            results.append({
                'scenario': scenario_name,
                'expected_goals': eg_baseline,
                'difference': 0.0,
                'pct_change': 0.0
            })
            continue
        
        # Modify policy
        pi_modified = modify_policy(pi_original, modifications)
        
        # Modify transitions if quality distributions provided
        if quality_distributions:
            from .success_modeling import modify_transition_matrix
            P_modified = modify_transition_matrix(P, quality_distributions, modifications)
        else:
            P_modified = P
        
        # Compute expected goals
        N_modified = compute_fundamental_matrix(P_modified, pi_modified)
        eg_modified = expected_goals_total(N_modified, pi_modified, P_modified, R, possession_starts)
        
        results.append({
            'scenario': scenario_name,
            'expected_goals': eg_modified,
            'difference': eg_modified - eg_baseline,
            'pct_change': (eg_modified - eg_baseline) / eg_baseline * 100
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Analysis Module - Skeleton Created")
    print("Implements fundamental matrix and counterfactual analysis")
