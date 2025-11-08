"""
Expected Goals (xG) Model
==========================

Position-based xG model using viewing angle to goal.
Used as a Bayesian prior for smoothing sparse shooting probabilities.
"""

import numpy as np


def calculate_goal_angle(x, y, pitch_length=105, pitch_width=68):
    """
    Calculate the viewing angle to goal from position (x, y).
    
    This is the angle subtended by the goal as seen from the ball's position.
    - Larger angle = goal appears wider (closer and more centered)
    - Smaller angle = goal appears narrower (far or wide position)
    
    Parameters
    ----------
    x : float
        X-coordinate (0-105m, length of pitch)
    y : float  
        Y-coordinate (0-68m, width of pitch)
    pitch_length : float
        Length of pitch in meters
    pitch_width : float
        Width of pitch in meters
        
    Returns
    -------
    float
        Viewing angle to goal in degrees (0-180)
    """
    # Goal is at x=pitch_length (105m), centered at y=pitch_width/2 (34m)
    # Attacking direction: toward x=105m
    goal_x = pitch_length  # 105m (far end)
    goal_y = pitch_width / 2.0  # 34m (centered)
    
    # Goal posts (standard goal width is 7.32m)
    goal_width = 7.32
    post_left_y = goal_y - goal_width / 2.0   # 30.34m
    post_right_y = goal_y + goal_width / 2.0  # 37.66m
    post_x = goal_x
    
    # Calculate angles from ball to each post (using atan2 for proper quadrant)
    angle_to_left = np.arctan2(post_left_y - y, post_x - x)
    angle_to_right = np.arctan2(post_right_y - y, post_x - x)
    
    # The viewing angle is the absolute difference between these angles
    # This gives us how wide the goal appears from this position
    viewing_angle_rad = abs(angle_to_right - angle_to_left)
    viewing_angle_deg = np.degrees(viewing_angle_rad)
    
    return viewing_angle_deg


def geometric_xg_model(angle, penalty_xg=0.76, penalty_angle_deg=36.8):
    """
    Geometric xG model based on angle alone.
    
    Assumes xG scales with viewing angle, calibrated to penalty spot.
    Penalty spot is at x=94m, y=34m (11m from goal), with angle ≈ 36.8°.
    
    Parameters
    ----------
    angle : float
        Viewing angle to goal in degrees
    penalty_xg : float, default=0.76
        Expected xG from penalty spot
    penalty_angle_deg : float, default=36.8
        Viewing angle from penalty spot in degrees
        
    Returns
    -------
    float
        Expected goal probability (0-0.95)
    """
    if angle <= 0.1:  # Very small angles
        return 0.001
    
    # Power law: larger angles = higher xG
    xg = penalty_xg * (angle / penalty_angle_deg) ** 1.5
    
    return min(xg, 0.95)  # Cap at 95%


def apply_bayesian_shrinkage(
    actions_df,
    grid,
    shoot_action_id=6,
    alpha=10,
    shoot_distance_threshold=30
):
    """
    Apply Bayesian shrinkage to shooting probabilities using geometric xG prior.
    
    Parameters
    ----------
    actions_df : pd.DataFrame
        Actions dataframe with columns: state_from, action, success
    grid : FieldGrid
        Field grid for coordinate conversion
    shoot_action_id : int, default=6
        Action ID for shooting
    alpha : int, default=10
        Prior strength (equivalent sample size for Bayesian shrinkage)
    shoot_distance_threshold : int, default=30
        Maximum distance from goal (in meters) where shooting is allowed
        
    Returns
    -------
    dict
        Dictionary mapping state -> smoothed xG probability
    """
    shoot_probs_bayesian = {}
    
    for state in range(grid.n_states):
        x, y = grid.state_to_xy(state)
        
        # Hard constraint: shooting only allowed within 30m of goal (x > 75m)
        if x < (grid.pitch_length - shoot_distance_threshold):
            # Too far from goal - shooting not allowed
            shoot_probs_bayesian[state] = 0.0
        else:
            # Within shooting range - apply Bayesian smoothing
            # Calculate geometric prior
            angle = calculate_goal_angle(x, y)
            prior_xg = geometric_xg_model(angle)
            
            # Get empirical data
            state_shots = actions_df[
                (actions_df['state_from'] == state) & 
                (actions_df['action'] == shoot_action_id)
            ]
            n_shots = len(state_shots)
            
            if n_shots > 0:
                n_goals = state_shots['success'].sum()
                empirical_xg = n_goals / n_shots
                
                # Bayesian shrinkage: P = (α * prior + n * empirical) / (α + n)
                smoothed_xg = (alpha * prior_xg + n_shots * empirical_xg) / (alpha + n_shots)
            else:
                # No data - use pure prior
                smoothed_xg = prior_xg
            
            shoot_probs_bayesian[state] = smoothed_xg
    
    return shoot_probs_bayesian
