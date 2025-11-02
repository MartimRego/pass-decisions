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
    limit_matches: Optional[int] = None
) -> pd.DataFrame:
    """
    Load all Premier League 2024 event data.
    
    Parameters
    ----------
    data_dir : Path, optional
        Path to data directory. If None, uses default location
        (PremierLeague_data/2024/dynamic/)
    limit_matches : int, optional
        If provided, only load first N matches (for testing)
        
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
        # Default: assume we're in Project/ directory, go up to twelve-deep-learning/
        data_dir = Path.cwd().parents[0] / "PremierLeague_data" / "2024" / "dynamic"
    
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Get all parquet files
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if len(parquet_files) == 0:
        raise ValueError(f"No parquet files found in {data_dir}")
    
    print(f"Found {len(parquet_files)} matches in {data_dir}")
    
    # Limit matches if specified (useful for testing)
    if limit_matches is not None:
        parquet_files = parquet_files[:limit_matches]
        print(f"Loading only first {limit_matches} matches for testing")
    
    # Load and combine all matches
    dfs = []
    for i, fpath in enumerate(parquet_files):
        match_id = int(fpath.stem)
        df_match = pd.read_parquet(fpath)
        
        # Add match_id if not present
        if 'match_id' not in df_match.columns:
            df_match['match_id'] = match_id
        
        dfs.append(df_match)
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(parquet_files)} matches...")
    
    # Combine all matches
    events_all = pd.concat(dfs, ignore_index=True)
    
    print(f"✅ Loaded {len(events_all):,} total events from {len(parquet_files)} matches")
    print(f"   Event types: {events_all['event_type'].nunique()}")
    print(f"   Date range: {events_all['match_id'].min()} to {events_all['match_id'].max()}")
    
    return events_all


def load_match_metadata(
    meta_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load match metadata (teams, dates) for Premier League 2024.
    
    Parameters
    ----------
    meta_dir : Path, optional
        Path to metadata directory. If None, uses default location
        (PremierLeague_data/2024/meta/)
        
    Returns
    -------
    pd.DataFrame
        Metadata with columns: match_id, home_team, away_team, etc.
        
    Examples
    --------
    >>> metadata = load_match_metadata()
    >>> print(metadata.head())
    """
    import json
    
    if meta_dir is None:
        meta_dir = Path.cwd().parents[0] / "PremierLeague_data" / "2024" / "meta"
    
    meta_dir = Path(meta_dir)
    if not meta_dir.exists():
        print(f"⚠️  Metadata directory not found: {meta_dir}")
        return pd.DataFrame()
    
    # Get all JSON metadata files
    meta_files = sorted(meta_dir.glob("*.json"))
    
    if len(meta_files) == 0:
        print(f"⚠️  No metadata files found in {meta_dir}")
        return pd.DataFrame()
    
    print(f"Loading metadata for {len(meta_files)} matches...")
    
    rows = []
    for fpath in meta_files:
        match_id = int(fpath.stem)
        
        try:
            with open(fpath, 'r') as f:
                meta = json.load(f)
            
            row = {
                'match_id': match_id,
                'home_team': meta.get('home_team', {}).get('name'),
                'away_team': meta.get('away_team', {}).get('name'),
                'home_team_id': meta.get('home_team', {}).get('id'),
                'away_team_id': meta.get('away_team', {}).get('id'),
            }
            rows.append(row)
        except Exception as e:
            print(f"  ⚠️  Error loading {fpath.name}: {e}")
    
    metadata = pd.DataFrame(rows)
    print(f"✅ Loaded metadata for {len(metadata)} matches")
    
    return metadata


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


def normalize_attack_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize attack direction so ALL teams attack left-to-right (0 → 105m).
    
    Flips coordinates for teams attacking right-to-left based on the
    'attacking_side' field in SkillCorner data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Pass events with rescaled coordinates and 'attacking_side' column
        
    Returns
    -------
    pd.DataFrame
        Pass events with normalized coordinates (all attacking left→right)
        
    Notes
    -----
    This function flips BOTH x and y coordinates when attacking_side == 'right_to_left':
    - x_norm = PITCH_LENGTH - x_rescaled
    - y_norm = PITCH_WIDTH - y_rescaled
    
    The y-flip ensures tactical formations remain consistent when viewing
    from the attacking team's perspective.
    
    Examples
    --------
    >>> passes = rescale_coordinates(passes)
    >>> passes = normalize_attack_direction(passes)
    >>> # Now all teams attack from x=0 to x=105
    """
    df = df.copy()
    
    if 'attacking_side' not in df.columns:
        print("⚠️  'attacking_side' column not found - skipping normalization")
        print("   Coordinates may not be normalized for attack direction!")
        return df
    
    # Normalize x_start_rescaled (origin position)
    if 'x_start_rescaled' in df.columns:
        df['x_start_norm'] = np.where(
            df['attacking_side'] == 'right_to_left',
            PITCH_LENGTH - df['x_start_rescaled'],
            df['x_start_rescaled']
        )
    
    # Normalize y_start_rescaled
    if 'y_start_rescaled' in df.columns:
        df['y_start_norm'] = np.where(
            df['attacking_side'] == 'right_to_left',
            PITCH_WIDTH - df['y_start_rescaled'],
            df['y_start_rescaled']
        )
    
    # Normalize x_end_rescaled (destination position)
    if 'x_end_rescaled' in df.columns:
        df['x_end_norm'] = np.where(
            df['attacking_side'] == 'right_to_left',
            PITCH_LENGTH - df['x_end_rescaled'],
            df['x_end_rescaled']
        )
    
    # Normalize y_end_rescaled
    if 'y_end_rescaled' in df.columns:
        df['y_end_norm'] = np.where(
            df['attacking_side'] == 'right_to_left',
            PITCH_WIDTH - df['y_end_rescaled'],
            df['y_end_rescaled']
        )
    
    # Report normalization stats
    n_flipped = (df['attacking_side'] == 'right_to_left').sum()
    n_total = len(df)
    print(f"✅ Attack direction normalized:")
    print(f"   Flipped: {n_flipped:,} / {n_total:,} ({n_flipped/n_total:.1%})")
    print(f"   All teams now attack left→right (x: 0→105)")
    
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
        
    Notes
    -----
    SkillCorner uses 'player_possession' as the event type for passes.
    We filter to events with valid start and end coordinates.
    """
    print(f"Extracting pass events from {len(events):,} total events...")
    
    # Filter to player_possession events (SkillCorner's pass events)
    passes = events[events['event_type'] == 'player_possession'].copy()
    print(f"  Found {len(passes):,} player_possession events")
    
    # Filter to events with complete coordinate information
    required_cols = ['x_start', 'y_start', 'x_end', 'y_end']
    missing_coords = passes[required_cols].isna().any(axis=1)
    
    print(f"  Removing {missing_coords.sum():,} events with missing coordinates")
    passes = passes[~missing_coords].copy()
    
    # Filter out clearances (defensive actions, not tactical passes)
    if 'end_type' in passes.columns:
        clearances = passes['end_type'] == 'clearance'
        print(f"  Removing {clearances.sum():,} clearances")
        passes = passes[~clearances].copy()
    
    # Add success indicator - use pass_outcome field (more accurate than end_type)
    if 'pass_outcome' in passes.columns:
        # SkillCorner provides: 'successful', 'unsuccessful', 'offside', etc.
        passes['success'] = (passes['pass_outcome'] == 'successful').astype(int)
        
        # Print breakdown
        outcome_counts = passes['pass_outcome'].value_counts()
        successful_count = outcome_counts.get('successful', 0)
        unsuccessful_count = outcome_counts.get('unsuccessful', 0)
        offside_count = outcome_counts.get('offside', 0)
        
        print(f"  Pass outcomes:")
        print(f"    Successful:   {successful_count:6,} ({successful_count/len(passes):5.1%})")
        print(f"    Unsuccessful: {unsuccessful_count:6,} ({unsuccessful_count/len(passes):5.1%})")
        if offside_count > 0:
            print(f"    Offside:      {offside_count:6,} ({offside_count/len(passes):5.1%})")
        
    elif 'end_type' in passes.columns:
        # Fallback: use end_type field if pass_outcome not available
        # Successful if end_type is 'pass', 'carry', or 'shot'
        success_outcomes = ['pass', 'carry', 'shot']
        passes['success'] = passes['end_type'].isin(success_outcomes).astype(int)
        print(f"  ⚠️  Using end_type for success (pass_outcome not available)")
        print(f"  Success rate: {passes['success'].mean():.1%}")
    else:
        print("  ⚠️  No success indicator found, setting all to successful")
        passes['success'] = 1
    
    print(f"✅ Extracted {len(passes):,} pass events")
    
    return passes


def extract_shot_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    Filter events to only shooting actions.
    
    Parameters
    ----------
    events : pd.DataFrame
        All event data
        
    Returns
    -------
    pd.DataFrame
        Only shot events with outcome information (goal vs no_goal)
        
    Notes
    -----
    Shots are identified by end_type='shot' in player_possession events.
    Success is determined by whether the shot resulted in a goal.
    """
    print(f"Extracting shot events from {len(events):,} total events...")
    
    # Filter to player_possession events with end_type='shot'
    shots = events[
        (events['event_type'] == 'player_possession') & 
        (events['end_type'] == 'shot')
    ].copy()
    print(f"  Found {len(shots):,} shot events")
    
    # Filter to events with valid start coordinates (end coords not needed for shots)
    required_cols = ['x_start', 'y_start']
    missing_coords = shots[required_cols].isna().any(axis=1)
    
    if missing_coords.sum() > 0:
        print(f"  Removing {missing_coords.sum():,} shots with missing coordinates")
        shots = shots[~missing_coords].copy()
    
    # Determine shot success (goal vs no_goal)
    # Check if there's a goal-related field
    if 'lead_to_goal' in shots.columns:
        # SkillCorner data uses lead_to_goal boolean column
        shots['success'] = shots['lead_to_goal'].fillna(False).astype(int)
    elif 'is_goal' in shots.columns:
        shots['success'] = shots['is_goal'].astype(int)
    elif 'pass_outcome' in shots.columns:
        # If pass_outcome exists, check for 'goal' value
        shots['success'] = (shots['pass_outcome'] == 'goal').astype(int)
    else:
        # Default: assume we need to infer from subsequent events or other fields
        # For now, set to 0 (we'll need to cross-reference with goal events)
        print("  ⚠️  No explicit goal indicator found, setting all shots to unsuccessful")
        print("     (This may need refinement based on actual data structure)")
        shots['success'] = 0
    
    # Add dummy end coordinates (shots end at goal location)
    # We'll use the goal center as a proxy
    shots['x_end'] = PITCH_LENGTH  # Goal line
    shots['y_end'] = PITCH_WIDTH / 2  # Center of goal
    
    # Print success breakdown
    goals = shots['success'].sum()
    print(f"  Shot outcomes:")
    print(f"    Goals:     {goals:6,} ({goals/len(shots):5.1%})")
    print(f"    No goals:  {len(shots)-goals:6,} ({(len(shots)-goals)/len(shots):5.1%})")
    
    print(f"✅ Extracted {len(shots):,} shot events")
    
    return shots


def extract_carry_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    Filter events to only carrying/dribbling actions.
    
    Parameters
    ----------
    events : pd.DataFrame
        All event data
        
    Returns
    -------
    pd.DataFrame
        Only carry events with outcome information
        
    Notes
    -----
    Carries are identified by carry=True in player_possession events.
    Success is determined by end_type: successful if ends in 'pass' or 'shot',
    failed if ends in 'possession_loss'.
    """
    print(f"Extracting carry events from {len(events):,} total events...")
    
    # Filter to player_possession events with carry=True
    carries = events[
        (events['event_type'] == 'player_possession') & 
        (events['carry'] == True)
    ].copy()
    print(f"  Found {len(carries):,} carry events")
    
    # Filter to events with valid coordinates
    required_cols = ['x_start', 'y_start', 'x_end', 'y_end']
    missing_coords = carries[required_cols].isna().any(axis=1)
    
    if missing_coords.sum() > 0:
        print(f"  Removing {missing_coords.sum():,} carries with missing coordinates")
        carries = carries[~missing_coords].copy()
    
    # Determine carry success based on end_type
    # Successful if ends in pass or shot, failed if possession_loss
    carries['success'] = carries['end_type'].isin(['pass', 'shot']).astype(int)
    
    # Print success breakdown
    successful = carries['success'].sum()
    failed = len(carries) - successful
    print(f"  Carry outcomes:")
    print(f"    Successful (ended in pass/shot): {successful:6,} ({successful/len(carries):5.1%})")
    print(f"    Failed (possession lost):        {failed:6,} ({failed/len(carries):5.1%})")
    
    print(f"✅ Extracted {len(carries):,} carry events")
    
    return carries


def combine_passes_shots_carries(
    passes: pd.DataFrame,
    shots: pd.DataFrame,
    carries: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine pass, shot, and carry events into a single DataFrame for MDP.
    
    Parameters
    ----------
    passes : pd.DataFrame
        Pass events from extract_pass_events()
    shots : pd.DataFrame
        Shot events from extract_shot_events()
    carries : pd.DataFrame
        Carry events from extract_carry_events()
        
    Returns
    -------
    pd.DataFrame
        Combined events with 'action_type' column ('pass', 'shoot', or 'carry')
        
    Notes
    -----
    Action encoding:
    - Passes: actions 0-7 (based on length/direction)
    - Shots: action 8
    - Carries: action 9
    """
    # Add action type markers
    passes = passes.copy()
    shots = shots.copy()
    carries = carries.copy()
    
    passes['action_type'] = 'pass'
    shots['action_type'] = 'shoot'
    carries['action_type'] = 'carry'
    
    # Keep all columns from all DataFrames
    combined = pd.concat([passes, shots, carries], ignore_index=True)
    
    print(f"✅ Combined {len(passes):,} passes + {len(shots):,} shots + {len(carries):,} carries")
    print(f"   = {len(combined):,} total actions")
    print(f"   Pass success rate:  {passes['success'].mean():.1%}")
    print(f"   Shot success rate:  {shots['success'].mean():.1%}")
    print(f"   Carry success rate: {carries['success'].mean():.1%}")
    
    return combined


# Keep old function for backward compatibility
def combine_passes_and_shots(
    passes: pd.DataFrame,
    shots: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine pass and shot events into a single DataFrame for MDP.
    
    DEPRECATED: Use combine_passes_shots_carries() instead.
    
    Parameters
    ----------
    passes : pd.DataFrame
        Pass events from extract_pass_events()
    shots : pd.DataFrame
        Shot events from extract_shot_events()
        
    Returns
    -------
    pd.DataFrame
        Combined events with 'action_type' column ('pass' or 'shoot')
        
    Notes
    -----
    Shots will be assigned action=8 later during state-action encoding.
    Passes will be assigned actions 0-7 based on length/direction.
    """
    # Add action type marker
    passes = passes.copy()
    shots = shots.copy()
    
    passes['action_type'] = 'pass'
    shots['action_type'] = 'shoot'
    
    # Keep all columns from both DataFrames, filling missing values
    # This preserves important columns like 'attacking_side' needed for normalization
    combined = pd.concat([passes, shots], ignore_index=True)
    
    print(f"✅ Combined {len(passes):,} passes + {len(shots):,} shots = {len(combined):,} total actions")
    print(f"   Pass success rate: {passes['success'].mean():.1%}")
    print(f"   Shot success rate (goals): {shots['success'].mean():.1%}")
    
    return combined


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
    - short: <= 10m
    - medium: 10-25m
    - long: > 25m
    """
    return pd.cut(
        distance,
        bins=[-0.001, 10, 25, np.inf],  # -0.001 to include 0
        labels=['short', 'medium', 'long']
    )


def classify_pass_direction(
    dx: pd.Series,
    forward_threshold: float = 5.0,
    lateral_threshold: float = 5.0
) -> pd.Series:
    """
    Classify pass direction into forward/lateral/backward.
    
    Parameters
    ----------
    dx : pd.Series
        Change in x coordinate (horizontal, progressive direction)
    forward_threshold : float, default=5.0
        Minimum dx to be considered forward (meters)
    lateral_threshold : float, default=5.0
        Maximum abs(dx) to be considered lateral (meters)
        
    Returns
    -------
    pd.Series
        Direction labels: 'forward', 'lateral', 'backward'
        
    Notes
    -----
    Classification logic:
    - forward: dx > forward_threshold (5m)
    - backward: dx < -forward_threshold (-5m)
    - lateral: abs(dx) <= lateral_threshold (within ±5m)
    """
    direction = pd.Series('lateral', index=dx.index)
    direction[dx > forward_threshold] = 'forward'
    direction[dx < -forward_threshold] = 'backward'
    return direction


def classify_pass_type(passes: pd.DataFrame) -> pd.DataFrame:
    """
    Add pass type classification (8 categories) to pass events.
    
    Filters out zero-distance passes (ball controls) and classifies
    remaining passes by length and direction.
    
    Parameters
    ----------
    passes : pd.DataFrame
        Pass events with normalized coordinates (x_start_norm, x_end_norm, etc.)
        
    Returns
    -------
    pd.DataFrame
        Filtered passes with added columns:
        - pass_distance: Euclidean distance in meters
        - dx: Change in x (progressive direction)
        - dy: Change in y (lateral direction)
        - pass_length: 'short', 'medium', 'long'
        - pass_direction: 'forward', 'lateral', 'backward'
        - pass_type: combined (e.g., 'short_forward')
        
    Notes
    -----
    Creates 8 pass types:
    - short/medium × backward/lateral/forward = 6 types
    - long × backward/forward = 2 types (long_lateral merged into long_forward)
    - Plus 'shoot' action (not from passes, added later)
    
    Zero-distance passes are filtered out as they represent ball controls
    rather than actual passes. This improves data quality and success rates.
    
    Examples
    --------
    >>> passes = classify_pass_type(passes)
    >>> passes['pass_type'].value_counts()
    """
    passes = passes.copy()
    
    # Compute pass distance and direction changes
    passes['dx'] = passes['x_end_norm'] - passes['x_start_norm']
    passes['dy'] = passes['y_end_norm'] - passes['y_start_norm']
    passes['pass_distance'] = np.sqrt(passes['dx']**2 + passes['dy']**2)
    
    # Filter out zero-distance passes (ball controls)
    n_before = len(passes)
    passes = passes[passes['pass_distance'] > 0].copy()
    n_after = len(passes)
    n_filtered = n_before - n_after
    
    print(f"🔧 Filtered zero-distance passes:")
    print(f"   Removed: {n_filtered:,} ({n_filtered/n_before*100:.2f}%)")
    print(f"   Remaining: {n_after:,}")
    
    # Classify length and direction
    passes['pass_length'] = classify_pass_length(passes['pass_distance'])
    passes['pass_direction'] = classify_pass_direction(passes['dx'])
    
    # Combine into pass_type
    passes['pass_type'] = passes['pass_length'].astype(str) + '_' + passes['pass_direction'].astype(str)
    
    # Merge long_lateral into medium_lateral
    # Long lateral passes are too rare (0.04%), merge with medium lateral
    long_lateral_mask = passes['pass_type'] == 'long_lateral'
    passes.loc[long_lateral_mask, 'pass_type'] = 'medium_lateral'
    # Also update pass_length so classify_action works correctly
    passes.loc[long_lateral_mask, 'pass_length'] = 'medium'
    
    print(f"✅ Pass types classified:")
    print(f"   Length × Direction = {passes['pass_length'].nunique()} × {passes['pass_direction'].nunique()} = {passes['pass_type'].nunique()} unique types")
    print(f"   (long_lateral merged into medium_lateral by updating pass_length)")
    
    return passes


def validate_pass_classifications(passes: pd.DataFrame) -> None:
    """
    Print validation statistics and sanity checks for pass classifications.
    
    Parameters
    ----------
    passes : pd.DataFrame
        Classified pass events with pass_type column
        
    Notes
    -----
    Reports special attention to long_backward passes to assess if this
    action type is frequent enough to warrant inclusion in MDP action space.
    """
    print("=" * 60)
    print("PASS CLASSIFICATION VALIDATION")
    print("=" * 60)
    
    print(f"\nTotal passes: {len(passes):,}")
    
    print("\n--- Length Distribution ---")
    length_dist = passes['pass_length'].value_counts().sort_index()
    print(length_dist)
    print(f"Percentages: {(length_dist / len(passes) * 100).round(2)}%")
    
    print("\n--- Direction Distribution ---")
    dir_dist = passes['pass_direction'].value_counts().sort_index()
    print(dir_dist)
    print(f"Percentages: {(dir_dist / len(passes) * 100).round(2)}%")
    
    print("\n--- Combined Type Distribution ---")
    type_dist = passes['pass_type'].value_counts()
    print(type_dist)
    print(f"\nPercentages:")
    print((type_dist / len(passes) * 100).round(2))
    
    # Special focus on long_backward for action space decision
    if 'long_backward' in type_dist.index:
        long_backward_count = type_dist['long_backward']
        long_backward_pct = (long_backward_count / len(passes)) * 100
        print(f"\n🔍 LONG_BACKWARD PASSES:")
        print(f"   Count: {long_backward_count:,}")
        print(f"   Percentage: {long_backward_pct:.2f}%")
        print(f"   {'⚠️ RARE - Consider merging/removing' if long_backward_pct < 2.0 else '✅ Sufficient for MDP'}")
    else:
        print(f"\n🔍 LONG_BACKWARD PASSES: None found")
    
    print("\n--- Distance Statistics by Length ---")
    print(passes.groupby('pass_length')['pass_distance'].describe())
    
    print("\n--- Success Rate by Type ---")
    if 'success' in passes.columns:
        success_by_type = passes.groupby('pass_type')['success'].agg(['mean', 'count'])
        success_by_type.columns = ['success_rate', 'count']
        print(success_by_type.sort_values('success_rate', ascending=False))
    
    print("=" * 60)


def get_data_summary(events: pd.DataFrame) -> None:
    """
    Print comprehensive summary of event data structure and contents.
    
    Parameters
    ----------
    events : pd.DataFrame
        Event data to summarize
    """
    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Dataset Size:")
    print(f"   Total events: {len(events):,}")
    print(f"   Total matches: {events['match_id'].nunique()}")
    print(f"   Memory usage: {events.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    print(f"\n⚽ Event Types:")
    event_counts = events['event_type'].value_counts()
    for event_type, count in event_counts.head(10).items():
        pct = count / len(events) * 100
        print(f"   {event_type:30s}: {count:6,} ({pct:5.1f}%)")
    
    print(f"\n🏟️  Coordinate Coverage:")
    coord_cols = ['x_start', 'y_start', 'x_end', 'y_end']
    for col in coord_cols:
        if col in events.columns:
            non_null = events[col].notna().sum()
            pct = non_null / len(events) * 100
            print(f"   {col:15s}: {non_null:7,} / {len(events):7,} ({pct:5.1f}% non-null)")
            if non_null > 0:
                print(f"                   range: [{events[col].min():7.2f}, {events[col].max():7.2f}]")
    
    print(f"\n📅 Match Coverage:")
    if 'period' in events.columns:
        print(f"   Periods: {events['period'].unique()}")
    if 'team_id' in events.columns:
        print(f"   Unique teams: {events['team_id'].nunique()}")
    
    print(f"\n🔑 Key Columns Available:")
    important_cols = [
        'match_id', 'period', 'team_id', 'player_name',
        'x_start', 'y_start', 'x_end', 'y_end',
        'end_type', 'pass_outcome', 'pass_distance', 'pass_direction'
    ]
    for col in important_cols:
        status = "✅" if col in events.columns else "❌"
        print(f"   {status} {col}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Example usage and testing
    print("Data Processing Module - Testing")
    print("=" * 60)
    
    # Test loading a small sample
    try:
        print("\n1. Testing data loading (first 2 matches)...")
        events = load_premier_league_events(limit_matches=2)
        get_data_summary(events)
        
        print("\n2. Testing pass extraction...")
        passes = extract_pass_events(events)
        
        print("\n3. Testing coordinate rescaling...")
        passes = rescale_coordinates(passes)
        print(f"   Added rescaled coordinate columns")
        print(f"   Sample x_start_rescaled: {passes['x_start_rescaled'].head().values}")
        
        print("\n4. Testing attack direction normalization...")
        passes = normalize_attack_direction(passes)
        print(f"   Sample x_start_norm: {passes['x_start_norm'].head().values}")
        
        print("\n5. Verifying pass success rates...")
        print(f"   Overall success rate: {passes['success'].mean():.1%}")
        print(f"   Successful passes: {(passes['success'] == 1).sum():,}")
        print(f"   Failed passes: {(passes['success'] == 0).sum():,}")
        
        print("\n✅ All basic functions working!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

