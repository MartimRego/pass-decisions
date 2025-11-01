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

