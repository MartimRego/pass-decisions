"""
Expected Goals (xG) Model
==========================

Position-based xG model using viewing angle to goal.
Uses a trained logistic regression model instead of hand-crafted formulas.
Used as a Bayesian prior for smoothing sparse shooting probabilities.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import joblib
from sklearn.linear_model import LogisticRegression


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


def build_shot_dataset(
    shots_df: pd.DataFrame,
    pitch_length: float = 105,
    pitch_width: float = 68
) -> pd.DataFrame:
    """
    Build a shot dataset with angle feature for xG model training.
    
    Parameters
    ----------
    shots_df : pd.DataFrame
        Shot events with x_norm, y_norm (normalized coordinates) and success columns
    pitch_length : float
        Length of pitch in meters
    pitch_width : float
        Width of pitch in meters
        
    Returns
    -------
    pd.DataFrame
        Dataset with columns: angle_to_goal, is_goal
    """
    # Calculate angle for each shot
    angles = []
    for _, shot in shots_df.iterrows():
        # Use x_end and y_end (shot location) in normalized coordinates
        x = shot['x_end_norm']
        y = shot['y_end_norm']
        angle = calculate_goal_angle(x, y, pitch_length, pitch_width)
        angles.append(angle)
    
    # Create training dataset
    training_data = pd.DataFrame({
        'angle_to_goal': angles,
        'is_goal': shots_df['success'].values
    })
    
    return training_data


def train_logistic_xg_model(
    shots_df: pd.DataFrame,
    pitch_length: float = 105,
    pitch_width: float = 68,
    random_state: int = 42
) -> Tuple[LogisticRegression, dict]:
    """
    Train a logistic regression xG model using angle to goal.
    
    Parameters
    ----------
    shots_df : pd.DataFrame
        Shot events with x_norm, y_norm and success columns
    pitch_length : float
        Length of pitch in meters
    pitch_width : float
        Width of pitch in meters
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    model : LogisticRegression
        Trained scikit-learn model
    metrics : dict
        Training metrics (accuracy, log-loss, etc.)
    """
    print(f"Training xG model on {len(shots_df):,} shots...")
    
    # Build dataset
    training_data = build_shot_dataset(shots_df, pitch_length, pitch_width)
    
    # Prepare features and target
    X = training_data[['angle_to_goal']].values
    y = training_data['is_goal'].values
    
    # Train logistic regression
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        solver='lbfgs'
    )
    model.fit(X, y)
    
    # Calculate metrics
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    
    metrics = {
        'n_shots': len(shots_df),
        'n_goals': y.sum(),
        'goal_rate': y.mean(),
        'accuracy': accuracy_score(y, y_pred),
        'log_loss': log_loss(y, y_pred_proba),
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'coef': model.coef_[0][0],
        'intercept': model.intercept_[0]
    }
    
    print(f"✅ Model trained successfully!")
    print(f"   Goals: {metrics['n_goals']:,} / {metrics['n_shots']:,} ({metrics['goal_rate']:.1%})")
    print(f"   Accuracy: {metrics['accuracy']:.3f}")
    print(f"   Log-loss: {metrics['log_loss']:.3f}")
    print(f"   ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"   Coefficient (angle): {metrics['coef']:.4f}")
    print(f"   Intercept: {metrics['intercept']:.4f}")
    
    return model, metrics


def save_xg_model(model: LogisticRegression, metrics: dict, save_path: Path):
    """
    Save trained xG model to disk.
    
    Parameters
    ----------
    model : LogisticRegression
        Trained model
    metrics : dict
        Training metrics
    save_path : Path
        Path to save model (.pkl file)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save both model and metrics
    model_data = {
        'model': model,
        'metrics': metrics
    }
    
    joblib.dump(model_data, save_path)
    print(f"✅ xG model saved to: {save_path}")


def load_xg_model(load_path: Path) -> Tuple[LogisticRegression, dict]:
    """
    Load trained xG model from disk.
    
    Parameters
    ----------
    load_path : Path
        Path to model file (.pkl)
        
    Returns
    -------
    model : LogisticRegression
        Trained model
    metrics : dict
        Training metrics
    """
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Model file not found: {load_path}")
    
    model_data = joblib.load(load_path)
    return model_data['model'], model_data['metrics']


# Global model cache
_XG_MODEL = None
_XG_METRICS = None


def get_xg_model(model_path: Optional[Path] = None) -> Tuple[LogisticRegression, dict]:
    """
    Get the trained xG model (loads once and caches).
    
    Parameters
    ----------
    model_path : Path, optional
        Path to model file. If None, uses default location.
        
    Returns
    -------
    model : LogisticRegression
        Trained model
    metrics : dict
        Training metrics
    """
    global _XG_MODEL, _XG_METRICS
    
    # Return cached model if available
    if _XG_MODEL is not None:
        return _XG_MODEL, _XG_METRICS
    
    # Default path
    if model_path is None:
        model_path = Path(__file__).parent.parent / 'data' / 'xg_model.pkl'
    
    # Load model
    _XG_MODEL, _XG_METRICS = load_xg_model(model_path)
    
    return _XG_MODEL, _XG_METRICS


def predict_xg(angle: float, model_path: Optional[Path] = None) -> float:
    """
    Predict xG for a given angle using the trained model.
    
    Parameters
    ----------
    angle : float
        Viewing angle to goal in degrees
    model_path : Path, optional
        Path to model file. If None, uses default location.
        
    Returns
    -------
    float
        Predicted goal probability (0-1)
    """
    model, _ = get_xg_model(model_path)
    
    # Predict
    X = np.array([[angle]])
    xg = model.predict_proba(X)[0, 1]
    
    return xg


def geometric_xg_model(angle, penalty_xg=0.76, penalty_angle_deg=36.8):
    """
    DEPRECATED: Geometric xG model based on angle alone.
    
    This function is kept for backwards compatibility but should not be used.
    Use predict_xg() instead, which uses the trained logistic regression model.
    
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
    shoot_distance_threshold=30,
    model_path: Optional[Path] = None
):
    """
    Apply Bayesian shrinkage to shooting probabilities using trained xG model.
    
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
    model_path : Path, optional
        Path to xG model file. If None, uses default location.
        
    Returns
    -------
    dict
        Dictionary mapping state -> smoothed xG probability
    """
    # Load trained model
    try:
        model, metrics = get_xg_model(model_path)
        print(f"  Using trained xG model (ROC-AUC: {metrics['roc_auc']:.3f})")
    except FileNotFoundError:
        print(f"  ⚠️  WARNING: Trained xG model not found, falling back to geometric model")
        print(f"     Run train_xg_model.py to train the model first")
        model = None
    shoot_probs_bayesian = {}
    
    for state in range(grid.n_states):
        x, y = grid.state_to_xy(state)
        
        # Hard constraint: shooting only allowed within 30m of goal (x > 75m)
        if x < (grid.pitch_length - shoot_distance_threshold):
            # Too far from goal - shooting not allowed
            shoot_probs_bayesian[state] = 0.0
        else:
            # Within shooting range - apply Bayesian smoothing
            # Calculate angle
            angle = calculate_goal_angle(x, y)
            
            # Get prior xG from trained model (or fallback to geometric)
            if model is not None:
                X = np.array([[angle]])
                prior_xg = model.predict_proba(X)[0, 1]
            else:
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
