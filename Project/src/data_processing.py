"""
Data Processing Module
======================

Functions for loading, filtering, and classifying pass events from
Premier League event data.

Author: Your Name
Date: November 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


# Pitch dimensions (SkillCorner coordinates)
X_MIN, X_MAX = -52, 52
Y_MIN, Y_MAX = -34, 34
PITCH_LENGTH, PITCH_WIDTH = 105, 68


def load_premier_league_events(
    data_dir: Optional[Path] = None,
    season: str = "2024"
) -> pd.DataFrame:
    """
    Load all Premier League event data for a given season.
    
    Parameters
    ----------
    data_dir : Path, optional
        Path to data directory. If None, uses default location.
    season : str, default="2024"
        Season identifier
        
    Returns
    -------
    pd.DataFrame
        Combined event data from all matches
        
    Examples
    --------
    >>> events = load_premier_league_events()
    >>> print(f"Loaded {len(events):,} events")
    """
    if data_dir is None:
        data_dir = Path.cwd().parents[0] / "PremierLeague_data" / season / "dynamic"
    
    # TODO: Implement loading logic
    # - Read all parquet files from data_dir
    # - Combine into single DataFrame
    # - Add match_id column
    # - Return combined data
    
    raise NotImplementedError("To be implemented in data exploration phase")


def rescale_coordinates(
    df: pd.DataFrame,
    x_cols: Tuple[str, str] = ("x_start", "x_end"),
    y_cols: Tuple[str, str] = ("y_start", "y_end")
) -> pd.DataFrame:
    """
    Rescale SkillCorner coordinates to FIFA pitch scale (0-105m x 0-68m).
    
    Parameters
    ----------
    df : pd.DataFrame
        Event data with SkillCorner coordinates
    x_cols : tuple of str
        Names of x coordinate columns (start, end)
    y_cols : tuple of str
        Names of y coordinate columns (start, end)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added *_rescaled columns
    """
    df = df.copy()
    
    # Rescale x coordinates
    for col in x_cols:
        if col in df.columns:
            df[f"{col}_rescaled"] = (df[col] - X_MIN) / (X_MAX - X_MIN) * PITCH_LENGTH
    
    # Rescale y coordinates
    for col in y_cols:
        if col in df.columns:
            df[f"{col}_rescaled"] = (df[col] - Y_MIN) / (Y_MAX - Y_MIN) * PITCH_WIDTH
    
    return df


def extract_pass_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    Filter events to only pass actions with required fields.
    
    Parameters
    ----------
    events : pd.DataFrame
        All event data
        
    Returns
    -------
    pd.DataFrame
        Only pass events with complete coordinate information
    """
    # TODO: Implement
    # - Filter to event_type == 'pass' or similar
    # - Remove passes with missing coordinates
    # - Remove set pieces (optional)
    # - Add pass distance and angle calculations
    
    raise NotImplementedError("To be implemented")


def classify_pass_length(distance: pd.Series) -> pd.Series:
    """
    Classify pass distance into short/medium/long categories.
    
    Parameters
    ----------
    distance : pd.Series
        Pass distances in meters
        
    Returns
    -------
    pd.Series
        Category labels: 'short', 'medium', 'long'
        
    Notes
    -----
    Thresholds:
    - short: < 15m
    - medium: 15-25m
    - long: > 25m
    """
    return pd.cut(
        distance,
        bins=[0, 15, 25, np.inf],
        labels=['short', 'medium', 'long']
    )


def classify_pass_direction(
    dx: pd.Series,
    dy: pd.Series,
    forward_threshold: float = 2.0,
    lateral_threshold: float = 2.0
) -> pd.Series:
    """
    Classify pass direction into forward/lateral/backward.
    
    Parameters
    ----------
    dx : pd.Series
        Change in x coordinate (horizontal)
    dy : pd.Series
        Change in y coordinate (vertical)
    forward_threshold : float
        Minimum dx to be considered forward (meters)
    lateral_threshold : float
        Maximum abs(dx) to be considered lateral (meters)
        
    Returns
    -------
    pd.Series
        Direction labels: 'forward', 'lateral', 'backward'
        
    Notes
    -----
    Classification logic:
    - forward: dx > forward_threshold
    - backward: dx < -forward_threshold
    - lateral: abs(dx) <= lateral_threshold
    """
    direction = pd.Series('lateral', index=dx.index)
    direction[dx > forward_threshold] = 'forward'
    direction[dx < -forward_threshold] = 'backward'
    return direction


def classify_pass_type(passes: pd.DataFrame) -> pd.DataFrame:
    """
    Add pass type classification (9 categories) to pass events.
    
    Parameters
    ----------
    passes : pd.DataFrame
        Pass events with rescaled coordinates
        
    Returns
    -------
    pd.DataFrame
        Passes with added columns:
        - pass_length: 'short', 'medium', 'long'
        - pass_direction: 'forward', 'lateral', 'backward'
        - pass_type: combined (e.g., 'short_forward')
        - action_id: integer 0-8 for MDP indexing
    """
    passes = passes.copy()
    
    # Calculate pass metrics
    passes['dx'] = passes['x_end_rescaled'] - passes['x_start_rescaled']
    passes['dy'] = passes['y_end_rescaled'] - passes['y_start_rescaled']
    passes['pass_distance'] = np.sqrt(passes['dx']**2 + passes['dy']**2)
    
    # Classify
    passes['pass_length'] = classify_pass_length(passes['pass_distance'])
    passes['pass_direction'] = classify_pass_direction(passes['dx'], passes['dy'])
    passes['pass_type'] = passes['pass_length'] + '_' + passes['pass_direction']
    
    # TODO: Add action_id mapping (0-8)
    # 0: short_forward, 1: short_lateral, 2: short_backward
    # 3: medium_forward, 4: medium_lateral, 5: medium_backward
    # 6: long_forward, 7: long_lateral, 8: shoot (handled separately)
    
    return passes


def validate_pass_classifications(passes: pd.DataFrame) -> None:
    """
    Print validation statistics and sanity checks for pass classifications.
    
    Parameters
    ----------
    passes : pd.DataFrame
        Classified pass events
    """
    print("=" * 60)
    print("PASS CLASSIFICATION VALIDATION")
    print("=" * 60)
    
    print(f"\nTotal passes: {len(passes):,}")
    
    print("\n--- Length Distribution ---")
    print(passes['pass_length'].value_counts().sort_index())
    
    print("\n--- Direction Distribution ---")
    print(passes['pass_direction'].value_counts().sort_index())
    
    print("\n--- Combined Type Distribution ---")
    print(passes['pass_type'].value_counts())
    
    print("\n--- Distance Statistics ---")
    print(passes.groupby('pass_length')['pass_distance'].describe())
    
    # TODO: Add more validation
    # - Check success rates by type
    # - Visualize sample passes
    # - Check for edge cases
    
    print("=" * 60)


if __name__ == "__main__":
    # Example usage and testing
    print("Data Processing Module - Skeleton Created")
    print("Ready for implementation!")
