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
    n_states = P.shape[0]
    
    # Build Q matrix: transitions between field states only
    Q = np.zeros((n_states, n_states))
    for s in range(n_states):
        for a in range(P.shape[1]):
            for s_prime in range(n_states):  # Only field states
                Q[s, s_prime] += P[s, a, s_prime] * pi[s, a]
    
    # Compute fundamental matrix
    I = np.eye(n_states)
    N = inv(I - Q)
    
    return N


def expected_goals_from_state(
    state: int,
    N: np.ndarray,
    pi: np.ndarray,
    P: np.ndarray,
    R: np.ndarray
) -> float:
    """
    Calculate expected goals starting from a given state.
    
    Parameters
    ----------
    state : int
        Starting state
    N : np.ndarray
        Fundamental matrix
    pi : np.ndarray
        Policy matrix
    P : np.ndarray
        Transition matrix
    R : np.ndarray
        Reward vector
        
    Returns
    -------
    float
        Expected number of goals
    """
    # TODO: Implement
    # E[goals | s] = sum over all states s' of:
    #   N[s, s'] * π(shoot | s') * P(s', shoot, goal) * R(goal)
    
    raise NotImplementedError("To be implemented")


def expected_goals_total(
    N: np.ndarray,
    pi: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    possession_starts: np.ndarray
) -> float:
    """
    Calculate total expected goals over a season.
    
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
    possession_starts : np.ndarray, shape (n_states,)
        Number of possessions starting in each state
        
    Returns
    -------
    float
        Total expected goals for the season
    """
    # TODO: Implement
    # E[goals] = sum_s possession_starts[s] * expected_goals_from_state(s, ...)
    
    raise NotImplementedError("To be implemented")


def optimal_action_per_state(
    N: np.ndarray,
    pi: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    n_actions: int
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
        
    Returns
    -------
    np.ndarray, shape (n_states,)
        Optimal action ID for each state
        
    Notes
    -----
    For each state, computes E[goals | s, a] for all actions a,
    and selects the action with maximum expected goals.
    """
    n_states = N.shape[0]
    optimal_actions = np.zeros(n_states, dtype=int)
    
    for s in range(n_states):
        expected_goals_per_action = np.zeros(n_actions)
        
        for a in range(n_actions):
            # TODO: Compute E[goals | s, a]
            # This requires modifying pi temporarily to force action a in state s
            # Then computing expected goals
            pass
        
        optimal_actions[s] = np.argmax(expected_goals_per_action)
    
    return optimal_actions


def modify_policy(
    pi: np.ndarray,
    modifications: Dict[Tuple[int, int], float]
) -> np.ndarray:
    """
    Create modified policy π' with specified changes.
    
    Parameters
    ----------
    pi : np.ndarray
        Original policy matrix
    modifications : dict
        Dictionary of {(state, action): change_fraction}
        e.g., {(100, 6): 0.20} means increase action 6 by 20% in state 100
        
    Returns
    -------
    np.ndarray
        Modified policy π'
        
    Notes
    -----
    When increasing π(a|s) by x:
        π'(a|s) = π(a|s) * (1 + x)
        π'(a'|s) = π(a'|s) * (1 - total_increase) / (1 - π(a|s))
        for all a' ≠ a
    """
    pi_modified = pi.copy()
    
    for (state, action), change in modifications.items():
        original_prob = pi[state, action]
        
        if change > 0:  # Increase
            new_prob = original_prob * (1 + change)
            increase = new_prob - original_prob
            
            # Decrease other actions proportionally
            other_actions = [a for a in range(pi.shape[1]) if a != action]
            total_other = pi[state, other_actions].sum()
            
            if total_other > 0:
                for a in other_actions:
                    pi_modified[state, a] = pi[state, a] * (1 - increase) / total_other
            
            pi_modified[state, action] = new_prob
            
        else:  # Decrease
            new_prob = original_prob * (1 + change)  # change is negative
            decrease = original_prob - new_prob
            
            # Increase other actions proportionally
            other_actions = [a for a in range(pi.shape[1]) if a != action]
            total_other = pi[state, other_actions].sum()
            
            if total_other > 0:
                for a in other_actions:
                    pi_modified[state, a] = pi[state, a] * (1 + decrease) / total_other
            
            pi_modified[state, action] = new_prob
        
        # Normalize to ensure sum = 1
        pi_modified[state, :] /= pi_modified[state, :].sum()
    
    return pi_modified


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
