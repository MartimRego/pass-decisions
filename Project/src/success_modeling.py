"""
Success Rate Modeling Module
=============================

Quality-quantity trade-off modeling for pass success rates.
Implements Method 3 from Van Roy et al. (2020).

Author: Your Name
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def compute_quality_distributions(
    passes: pd.DataFrame,
    quality_metric: str = 'outcome'
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Compute quality distributions for each (state, action) pair.
    
    Parameters
    ----------
    passes : pd.DataFrame
        Pass events with state_from, action, success/quality columns
    quality_metric : str, default='outcome'
        Metric to use for ranking quality:
        - 'outcome': binary success/failure
        - 'xT': expected threat value added
        - 'goal_outcome': whether possession led to goal
        
    Returns
    -------
    dict
        Nested dictionary: {(state, action): {'mu': X, 'mu_below': Y, 'mu_top90': Z}}
        where:
        - mu: average success rate
        - mu_below: average success rate of below-median passes
        - mu_top90: average success rate of top 90% passes
        
    Examples
    --------
    >>> quality_dist = compute_quality_distributions(passes)
    >>> s, a = 100, 0  # state 100, action 0 (short_forward)
    >>> print(quality_dist[(s, a)])
    {'mu': 0.85, 'mu_below': 0.78, 'mu_top90': 0.91}
    """
    quality_dist = {}
    
    # Group by (state, action)
    for (state, action), group in passes.groupby(['state_from', 'action']):
        if len(group) < 5:  # Skip sparse pairs
            continue
        
        # Get quality measure
        if quality_metric == 'outcome':
            quality = group['success'].values
        elif quality_metric == 'xT':
            quality = group['xT_added'].values  # Assumes this column exists
        else:
            quality = group['success'].values  # Default to binary
        
        # Compute statistics
        mu = quality.mean()
        
        # Below-median quality
        median_quality = np.median(quality)
        below_median = quality[quality <= median_quality]
        mu_below = below_median.mean() if len(below_median) > 0 else mu
        
        # Top 90% quality
        percentile_10 = np.percentile(quality, 10)
        top_90 = quality[quality >= percentile_10]
        mu_top90 = top_90.mean() if len(top_90) > 0 else mu
        
        quality_dist[(state, action)] = {
            'mu': mu,
            'mu_below': mu_below,
            'mu_top90': mu_top90,
            'n_samples': len(group)
        }
    
    return quality_dist


def adjust_success_rate(
    original_rate: float,
    quality_dist: Dict[str, float],
    frequency_change: float,
    direction: str = 'increase'
) -> float:
    """
    Adjust success rate based on frequency change (quality-quantity trade-off).
    
    Parameters
    ----------
    original_rate : float
        Original success rate P(s, a, s')
    quality_dist : dict
        Quality statistics for this (state, action) pair
    frequency_change : float
        Percentage change in frequency (e.g., 0.10 for +10%)
    direction : str
        'increase' or 'decrease'
        
    Returns
    -------
    float
        Adjusted success rate
        
    Notes
    -----
    From Van Roy et al. Section 3.2.2:
    
    If increasing frequency:
        New passes are below-average quality
        P'(s,a,s') = P(s,a,s') - (μ - μ_below) * freq_change
    
    If decreasing frequency:
        Drop lowest-quality passes
        P'(s,a,s') = P(s,a,s') + (μ_top - μ) * freq_change
    """
    if direction == 'increase':
        # New passes are below-average
        penalty = (quality_dist['mu'] - quality_dist['mu_below']) * frequency_change
        adjusted = original_rate - penalty
    else:  # decrease
        # Remove lowest-quality passes
        bonus = (quality_dist['mu_top90'] - quality_dist['mu']) * frequency_change
        adjusted = original_rate + bonus
    
    # Clip to valid probability range
    return np.clip(adjusted, 0.0, 1.0)


def modify_transition_matrix(
    P: np.ndarray,
    quality_distributions: Dict[Tuple[int, int], Dict[str, float]],
    policy_changes: Dict[Tuple[int, int], float]
) -> np.ndarray:
    """
    Create modified transition matrix with adjusted success rates.
    
    Parameters
    ----------
    P : np.ndarray
        Original transition matrix
    quality_distributions : dict
        Quality statistics per (state, action)
    policy_changes : dict
        Dictionary of {(state, action): freq_change}
        
    Returns
    -------
    np.ndarray
        Modified transition matrix P'
        
    Examples
    --------
    >>> # Increase long_forward passes by 20% in states 100-110
    >>> changes = {(s, 6): 0.20 for s in range(100, 111)}
    >>> P_modified = modify_transition_matrix(P, quality_dist, changes)
    """
    P_modified = P.copy()
    
    for (state, action), freq_change in policy_changes.items():
        if (state, action) not in quality_distributions:
            continue  # Skip if no quality data
        
        quality_dist = quality_distributions[(state, action)]
        direction = 'increase' if freq_change > 0 else 'decrease'
        
        # Adjust success rates for all successful transitions
        for next_state in range(P.shape[2] - 3):  # Exclude absorbing states
            if P[state, action, next_state] > 0:
                P_modified[state, action, next_state] = adjust_success_rate(
                    P[state, action, next_state],
                    quality_dist,
                    abs(freq_change),
                    direction
                )
        
        # Renormalize to ensure probabilities sum to 1
        total_success = P_modified[state, action, :-3].sum()
        if total_success > 1.0:
            P_modified[state, action, :-3] /= total_success
            P_modified[state, action, -1] = 0  # Loss state
        else:
            P_modified[state, action, -1] = 1.0 - total_success  # Loss state
    
    return P_modified


def plot_quality_distribution(
    passes: pd.DataFrame,
    state: int,
    action: int,
    quality_metric: str = 'outcome'
) -> None:
    """
    Visualize quality distribution for a specific (state, action) pair.
    
    Parameters
    ----------
    passes : pd.DataFrame
        Pass events
    state : int
        State index
    action : int
        Action index
    quality_metric : str
        Quality metric to plot
    """
    # TODO: Implement visualization
    # - Histogram of quality values
    # - Mark μ, μ_below, μ_top90
    # - Show sample size
    
    raise NotImplementedError("To be implemented in visualization module")


if __name__ == "__main__":
    print("Success Modeling Module - Skeleton Created")
    print("Implements quality-quantity trade-off (Van Roy Method 3)")
