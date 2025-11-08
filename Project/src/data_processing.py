"""
Data Processing Module
======================

Functions for loading, filtering, and classifying pass events from
Premier League event data.

Author: Your Name
Date: November 2025

IMPORTANT CHANGES (Nov 6, 2025):
---------------------------------
1. Pass direction classification uses ANGLE-BASED method following SkillCorner
   documentation (page 24) instead of dx-threshold method:

   OLD METHOD (dx-threshold):
   - Forward: dx > 5m
   - Backward: dx < -5m  
   - Lateral: |dx| <= 5m

   NEW METHOD (angle-based):
   - Forward: angle between -45° and +45°
   - Backward: angle < -135° or > 135°
   - Lateral: angle between 45° and 135° OR between -135° and -45°
     (merges SkillCorner's 'sideway_left' and 'sideway_right')

   This captures the actual trajectory of passes, not just horizontal displacement.
   Example: A pass with dx=4m, dy=20m is now correctly classified as 'lateral' 
   (angle=78.7°) instead of 'lateral' by threshold.

2. Pass length classification simplified to SHORT/LONG (threshold at 25m):
   - Short: <= 25m
   - Long: > 25m
   
   This creates 6 pass types (short/long × backward/lateral/forward).

3. Action space: 8 total actions
   - 6 pass types (short/long × backward/lateral/forward)
   - 1 shoot action
   - 1 carry action

See debug.ipynb for validation against SkillCorner's pass_direction field.
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
    x_cols: Tuple[str, str] = ("x_end", "player_targeted_x_reception"),
    y_cols: Tuple[str, str] = ("y_end", "player_targeted_y_reception")
) -> pd.DataFrame:
    """
    Rescale SkillCorner coordinates to FIFA pitch scale (0-105m x 0-68m).
    
    NOTE: x_end/y_end represent where the passer RELEASES the ball (pass origin),
    and player_targeted_x_reception/y_reception is where the receiver gets it.
    This gives us the actual PASS distance, not the passer's movement distance.
    
    Parameters
    ----------
    df : pd.DataFrame
        Event data with SkillCorner coordinates
    x_cols : tuple of str
        Names of x coordinate columns (pass origin, pass destination)
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
    
    Normalizes the following coordinate pairs (if present):
    - x_start_rescaled → x_start_norm
    - y_start_rescaled → y_start_norm
    - x_end_rescaled → x_end_norm (pass origin)
    - y_end_rescaled → y_end_norm
    - player_targeted_x_reception_rescaled → player_targeted_x_reception_norm (pass destination)
    - player_targeted_y_reception_rescaled → player_targeted_y_reception_norm
    
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
    
    # Normalize player_targeted_x_reception_rescaled (pass destination)
    if 'player_targeted_x_reception_rescaled' in df.columns:
        df['player_targeted_x_reception_norm'] = np.where(
            df['attacking_side'] == 'right_to_left',
            PITCH_LENGTH - df['player_targeted_x_reception_rescaled'],
            df['player_targeted_x_reception_rescaled']
        )
    
    # Normalize player_targeted_y_reception_rescaled
    if 'player_targeted_y_reception_rescaled' in df.columns:
        df['player_targeted_y_reception_norm'] = np.where(
            df['attacking_side'] == 'right_to_left',
            PITCH_WIDTH - df['player_targeted_y_reception_rescaled'],
            df['player_targeted_y_reception_rescaled']
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
        Only pass events with complete coordinate information and success indicator
        
    Notes
    -----
    According to SkillCorner documentation:
    - Pass events are identified by: event_type=='player_possession' AND end_type=='pass'
    - pass_outcome indicates success: 'successful', 'unsuccessful', 'offside'
    - Both successful and unsuccessful passes are included (if they have coordinates)
    
    Coordinate requirements:
    - ALL passes need: x_start, y_start, x_end, y_end (pass origin)
    - Successful passes also need: player_targeted_x_reception, player_targeted_y_reception (pass destination)
    - Unsuccessful passes don't have reception coordinates (they're NULL)
    
    We filter out:
    - Passes missing basic coordinates (x_start, y_start, x_end, y_end)
    - Successful passes missing reception coordinates (small % of successful passes)
    - Clearances are already excluded by the end_type=='pass' filter
    """
    print(f"Extracting pass events from {len(events):,} total events...")
    
    # Filter to player_possession events that ended with a pass
    # This correctly identifies ALL pass attempts (successful and unsuccessful)
    if 'end_type' not in events.columns:
        raise ValueError("'end_type' column not found in events data!")
    
    passes = events[
        (events['event_type'] == 'player_possession') & 
        (events['end_type'] == 'pass')
    ].copy()
    print(f"  Found {len(passes):,} pass events (end_type='pass')")
    
    # Filter to events with complete coordinate information
    # x_start, y_start, x_end, y_end are always needed (pass origin)
    # For successful passes, we also need player_targeted_x/y_reception (pass destination)
    required_basic_cols = ['x_start', 'y_start', 'x_end', 'y_end']
    missing_basic = passes[required_basic_cols].isna().any(axis=1)
    
    if missing_basic.sum() > 0:
        print(f"  Removing {missing_basic.sum():,} passes with missing start/end coordinates")
        passes = passes[~missing_basic].copy()
    
    # For successful passes, also filter out those missing reception coordinates
    # (unsuccessful passes won't have these, so we only check successful ones)
    if 'pass_outcome' in passes.columns:
        successful_mask = passes['pass_outcome'] == 'successful'
        reception_cols = ['player_targeted_x_reception', 'player_targeted_y_reception']
        
        if all(col in passes.columns for col in reception_cols):
            # Check which successful passes are missing reception coordinates
            missing_reception = successful_mask & passes[reception_cols].isna().any(axis=1)
            
            if missing_reception.sum() > 0:
                print(f"  Removing {missing_reception.sum():,} successful passes with missing reception coordinates")
                print(f"  ({missing_reception.sum()/successful_mask.sum()*100:.2f}% of successful passes)")
                passes = passes[~missing_reception].copy()
    
    # Add success indicator using pass_outcome field
    if 'pass_outcome' not in passes.columns:
        raise ValueError("'pass_outcome' column not found in pass events!")
    
    # SkillCorner provides: 'successful', 'unsuccessful', 'offside'
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
    According to SkillCorner documentation:
    - Shot events are identified by: event_type=='player_possession' AND end_type=='shot'
    - lead_to_goal (boolean) indicates if the shot resulted in a goal
    - All four coordinates (x_start, y_start, x_end, y_end) are available for shots
    - x_end/y_end represent the shot location (where the shot was taken from)
    - These coordinates are essential for MDP state encoding and xG modeling
    """
    print(f"Extracting shot events from {len(events):,} total events...")
    
    # Filter to player_possession events with end_type='shot'
    if 'end_type' not in events.columns:
        raise ValueError("'end_type' column not found in events data!")
    
    shots = events[
        (events['event_type'] == 'player_possession') & 
        (events['end_type'] == 'shot')
    ].copy()
    print(f"  Found {len(shots):,} shot events (end_type='shot')")
    
    # Filter to events with complete coordinate information
    # x_start, y_start, x_end, y_end are all needed (shot location for MDP)
    required_cols = ['x_start', 'y_start', 'x_end', 'y_end']
    missing_coords = shots[required_cols].isna().any(axis=1)
    
    if missing_coords.sum() > 0:
        print(f"  Removing {missing_coords.sum():,} shots with missing coordinates")
        shots = shots[~missing_coords].copy()
    
    # Add success indicator using lead_to_goal field
    if 'lead_to_goal' not in shots.columns:
        raise ValueError("'lead_to_goal' column not found in shot events!")
    
    # SkillCorner provides lead_to_goal as boolean (True/False)
    shots['success'] = shots['lead_to_goal'].fillna(False).astype(int)
    
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
    According to SkillCorner documentation:
    - Carry events are identified by: event_type=='player_possession' AND carry==True
    - Carries represent dribbling/running with the ball from x_start/y_start to x_end/y_end
    - All four coordinates (x_start, y_start, x_end, y_end) are available for carries
    - Success is determined by end_type:
      * Successful: ends in 'pass' or 'shot' (maintained possession)
      * Unsuccessful: ends in 'possession_loss', 'foul_suffered', 'clearance', etc.
    
    IMPORTANT: Many carries OVERLAP with passes and shots:
    - ~89% of carries end in a pass (carry=True AND end_type='pass')
    - ~2% of carries end in a shot (carry=True AND end_type='shot')
    - These events are BOTH a carry AND a pass/shot
    - When combining actions for MDP, we need to avoid double-counting
    """
    print(f"Extracting carry events from {len(events):,} total events...")
    
    # Filter to player_possession events with carry=True
    if 'carry' not in events.columns:
        raise ValueError("'carry' column not found in events data!")
    
    carries = events[
        (events['event_type'] == 'player_possession') & 
        (events['carry'] == True)
    ].copy()
    print(f"  Found {len(carries):,} carry events (carry=True)")
    
    # Filter to events with complete coordinate information
    # x_start, y_start, x_end, y_end represent the dribble path
    required_cols = ['x_start', 'y_start', 'x_end', 'y_end']
    missing_coords = carries[required_cols].isna().any(axis=1)
    
    if missing_coords.sum() > 0:
        print(f"  Removing {missing_coords.sum():,} carries with missing coordinates")
        carries = carries[~missing_coords].copy()
    
    # Add success indicator based on end_type
    if 'end_type' not in carries.columns:
        raise ValueError("'end_type' column not found in carry events!")
    
    # Successful if player kept possession (ended in pass or shot)
    # Unsuccessful if player lost possession (possession_loss, foul_suffered, etc.)
    carries['success'] = carries['end_type'].isin(['pass', 'shot']).astype(int)
    
    # Print success breakdown and overlap statistics
    successful = carries['success'].sum()
    unsuccessful = len(carries) - successful
    
    # Breakdown by end_type
    end_pass = (carries['end_type'] == 'pass').sum()
    end_shot = (carries['end_type'] == 'shot').sum()
    end_loss = (carries['end_type'] == 'possession_loss').sum()
    
    print(f"  Carry outcomes:")
    print(f"    Successful:   {successful:6,} ({successful/len(carries):5.1%})")
    print(f"      → ended in pass: {end_pass:5,} ({end_pass/len(carries):5.1%})")
    print(f"      → ended in shot: {end_shot:5,} ({end_shot/len(carries):5.1%})")
    print(f"    Unsuccessful: {unsuccessful:6,} ({unsuccessful/len(carries):5.1%})")
    print(f"      → possession_loss: {end_loss:5,} ({end_loss/len(carries):5.1%})")
    
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
    
    # CRITICAL: Preserve unique identifiers before concatenation
    # When we concat with ignore_index=True, the DataFrame index is reset
    # But we need event_id or index as a column for merging later
    for df in [passes, shots, carries]:
        # If event_id exists in columns, we're good
        # If not, but index exists as a column, we're good  
        # Otherwise, preserve the DataFrame index as a column called 'original_index'
        if 'event_id' not in df.columns and 'index' not in df.columns:
            df.reset_index(inplace=True)
            if df.index.name != 'index':
                # Rename whatever the index was to 'index'
                df.rename(columns={df.columns[0]: 'index'}, inplace=True)
    
    # Keep all columns from all DataFrames
    combined = pd.concat([passes, shots, carries], ignore_index=True)
    
    # CRITICAL: Create a unique action_id for each action
    # This is needed for merging with predicted pass types later
    # The index gets reset multiple times throughout the pipeline, so we need a persistent ID
    combined['action_id'] = range(len(combined))
    
    print(f"✅ Combined {len(passes):,} passes + {len(shots):,} shots + {len(carries):,} carries")
    print(f"   = {len(combined):,} total actions")
    print(f"   Pass success rate:  {passes['success'].mean():.1%}")
    print(f"   Shot success rate:  {shots['success'].mean():.1%}")
    print(f"   Carry success rate: {carries['success'].mean():.1%}")
    print(f"   Created unique action_id column for merging")
    
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
    Classify pass distance into short/long categories.
    
    Parameters
    ----------
    distance : pd.Series
        Pass distances in meters
        
    Returns
    -------
    pd.Series
        Category labels: 'short', 'long'
        
    Notes
    -----
    Thresholds:
    - short: <= 25m
    - long: > 25m
    
    This creates 6 pass types when combined with 3 directions:
    - short × (backward, lateral, forward) = 3 types
    - long × (backward, lateral, forward) = 3 types
    - Total: 6 pass types + shoot + carry = 8 action types
    """
    return pd.cut(
        distance,
        bins=[-0.001, 25, np.inf],  # -0.001 to include 0
        labels=['short', 'long']
    )


def classify_pass_direction(
    dx: pd.Series,
    dy: pd.Series
) -> pd.Series:
    """
    Classify pass direction into forward/lateral/backward using angle-based method.
    
    Follows SkillCorner's angle-based classification (see documentation page 24):
    - Forward: angle between -45° and +45°
    - Backward: angle below -135° or above 135°
    - Sideway Left: angle between 45° and 135°
    - Sideway Right: angle between -135° and -45°
    - Lateral: merge of Sideway Left and Sideway Right
    
    Parameters
    ----------
    dx : pd.Series
        Change in x coordinate (horizontal, progressive direction)
    dy : pd.Series
        Change in y coordinate (lateral direction)
        
    Returns
    -------
    pd.Series
        Direction labels: 'forward', 'lateral', 'backward'
        
    Notes
    -----
    The angle is calculated as arctan2(dy, dx) relative to the direction of attack
    (positive x-axis toward goal). This captures the actual trajectory of the pass,
    not just horizontal displacement.
    
    Examples
    --------
    >>> # Pure forward pass (dx=10, dy=0) → 0° → 'forward'
    >>> # Diagonal forward pass (dx=10, dy=10) → 45° → 'forward' (at boundary)
    >>> # Lateral pass (dx=5, dy=20) → 76° → 'lateral'
    >>> # Backward pass (dx=-15, dy=0) → 180° → 'backward'
    """
    # Calculate angle of pass vector relative to direction of attack (positive x-axis)
    pass_angle_rad = np.arctan2(dy, dx)
    pass_angle_deg = np.degrees(pass_angle_rad)  # Range: -180° to +180°
    
    # Classify using SkillCorner's angle thresholds
    direction = pd.Series('lateral', index=dx.index, dtype='object')
    
    # Forward: -45° to +45°
    forward_mask = (pass_angle_deg >= -45) & (pass_angle_deg <= 45)
    direction[forward_mask] = 'forward'
    
    # Backward: < -135° or > 135°
    backward_mask = (pass_angle_deg < -135) | (pass_angle_deg > 135)
    direction[backward_mask] = 'backward'
    
    # Lateral: everything else (45° to 135° and -135° to -45°)
    # Already initialized as 'lateral', so sideway_left and sideway_right are merged
    
    return direction


def classify_pass_type(
    passes: pd.DataFrame,
    use_predicted_types: bool = True,
    predicted_types_path: str = '../PremierLeague_data/2024/processed/passes_with_types_complete.parquet'
) -> pd.DataFrame:
    """
    Add pass type classification (6 categories) to pass events.
    
    For successful passes: calculates pass type from reception coordinates.
    For unsuccessful passes: uses predicted pass types from XGBoost model.
    
    Parameters
    ----------
    passes : pd.DataFrame
        Pass events with normalized coordinates (x_start_norm, x_end_norm, etc.)
    use_predicted_types : bool, default=True
        If True, load predicted types for unsuccessful passes from saved file.
        If False, filter out unsuccessful passes (legacy behavior).
    predicted_types_path : str, default='../PremierLeague_data/2024/processed/passes_with_types_complete.parquet'
        Path to the complete pass dataset with predicted types for unsuccessful passes.
        
    Returns
    -------
    pd.DataFrame
        Passes with added columns:
        - pass_distance: Euclidean distance in meters (for successful passes)
        - dx: Change in x (progressive direction, for successful passes)
        - dy: Change in y (lateral direction, for successful passes)
        - pass_length: 'short', 'long'
        - pass_direction: 'forward', 'lateral', 'backward'
        - pass_type: combined (e.g., 'short_forward')
        
    Notes
    -----
    Creates 6 pass types:
    - short × backward/lateral/forward = 3 types
    - long × backward/lateral/forward = 3 types
    - Total action space: 6 passes + shoot + carry = 8 actions
    
    Pass length threshold:
    - short: <= 25m
    - long: > 25m
    
    Unsuccessful passes get their pass_type from a pre-trained XGBoost model
    that predicts the intended pass type based on game context features.
    
    Examples
    --------
    >>> passes = classify_pass_type(passes)
    >>> passes['pass_type'].value_counts()
    """
    from pathlib import Path
    
    passes = passes.copy()
    n_total = len(passes)
    
    if use_predicted_types and Path(predicted_types_path).exists():
        print(f"📦 Loading complete pass dataset with predicted types...")
        print(f"   Path: {predicted_types_path}")
        
        # Load the complete dataset with all pass types
        passes_complete = pd.read_parquet(predicted_types_path)
        
        # Merge pass_type, pass_length, pass_direction from complete dataset
        # Match on unique identifiers to preserve all columns from input
        merge_cols = ['pass_type', 'pass_length', 'pass_direction', 'pass_distance', 'dx', 'dy']
        
        # Create merge key - prioritize unique IDs over composite keys
        # Strategy: Try single unique IDs first (action_id), then composite keys
        id_cols = []
        
        if 'action_id' in passes.columns and 'action_id' in passes_complete.columns:
            id_cols = ['action_id']
            print(f"   Merging on: action_id")
        elif 'index' in passes.columns and 'index' in passes_complete.columns and \
             'match_id' in passes.columns and 'match_id' in passes_complete.columns and \
             'period' in passes.columns and 'period' in passes_complete.columns:
            # Composite key: index is not unique alone, but index + match_id + period IS unique
            id_cols = ['index', 'match_id', 'period']
            print(f"   Merging on composite key: index + match_id + period")
        elif 'event_id' in passes.columns and 'event_id' in passes_complete.columns:
            id_cols = ['event_id']
            print(f"   ⚠️  Merging on: event_id (may not be unique!)")
        else:
            raise ValueError(
                f"Cannot find suitable merge columns.\n"
                f"Available in passes: {sorted(passes.columns.tolist())}\n"
                f"Available in passes_complete: {sorted(passes_complete.columns.tolist())}\n"
                f"Need either: action_id, or (index + match_id + period), or event_id"
            )
        
        # Select only the columns we need from passes_complete
        cols_to_merge = id_cols + [col for col in merge_cols if col in passes_complete.columns]
        passes_complete_subset = passes_complete[cols_to_merge].copy()
        
        # Merge
        passes = passes.merge(
            passes_complete_subset,
            on=id_cols,
            how='left',
            suffixes=('', '_predicted')
        )
        
        # Count successful vs unsuccessful
        n_successful = passes['success'].sum()
        n_unsuccessful = len(passes) - n_successful
        
        print(f"✅ Pass types loaded and merged:")
        print(f"   Total passes: {len(passes):,}")
        print(f"   Successful (actual types):    {n_successful:,} ({n_successful/len(passes)*100:.1f}%)")
        print(f"   Unsuccessful (predicted):     {n_unsuccessful:,} ({n_unsuccessful/len(passes)*100:.1f}%)")
        
    else:
        # Legacy behavior: calculate for successful passes only, filter unsuccessful
        print(f"⚠️  Predicted types file not found or use_predicted_types=False")
        print(f"   Falling back to legacy behavior (filter unsuccessful passes)")
        
        # Compute pass distance and direction changes
        # NOTE: x_end/y_end = pass origin (where ball is released)
        #       player_targeted_x_reception/y_reception = pass destination (where receiver gets it)
        passes['dx'] = passes['player_targeted_x_reception_norm'] - passes['x_end_norm']
        passes['dy'] = passes['player_targeted_y_reception_norm'] - passes['y_end_norm']
        passes['pass_distance'] = np.sqrt(passes['dx']**2 + passes['dy']**2)
        
        # Filter out passes with NULL reception coordinates (unsuccessful passes)
        n_before_null = len(passes)
        null_reception = passes[['player_targeted_x_reception_norm', 'player_targeted_y_reception_norm']].isna().any(axis=1)
        if null_reception.any():
            print(f"🔧 Filtering passes with NULL reception coordinates:")
            print(f"   (These are unsuccessful passes where target location is unknown)")
            print(f"   Removed: {null_reception.sum():,} ({null_reception.sum()/n_before_null*100:.2f}%)")
            passes = passes[~null_reception].copy()
        
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
        passes['pass_direction'] = classify_pass_direction(passes['dx'], passes['dy'])
        
        # Combine into pass_type
        passes['pass_type'] = passes['pass_length'].astype(str) + '_' + passes['pass_direction'].astype(str)
    
    # Final statistics
    print(f"\n✅ Pass type classification complete:")
    print(f"   Pass types: {passes['pass_type'].nunique()} unique")
    print(f"   Expected: 6 types (short/long × backward/lateral/forward)")
    if 'pass_length' in passes.columns and 'pass_direction' in passes.columns:
        print(f"   Length × Direction = {passes['pass_length'].nunique()} × {passes['pass_direction'].nunique()}")
    print(f"   Total action space: 6 passes + shoot + carry = 8 actions")
    
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
    Validates the 6 pass type classification system:
    - short/long (threshold at 25m) × backward/lateral/forward
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

