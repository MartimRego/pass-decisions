"""
State and Action Space Module
==============================

Functions for defining and encoding the MDP state space (field grid)
and action space (pass types + shooting).

Author: Your Name
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional


# Action space definition (8 total actions: 6 pass types + shoot + carry)
# Pass length thresholds: short ≤25m, long >25m
# Pass direction: forward/lateral/backward (angle-based)
# This matches the pass prediction model output
ACTION_NAMES = {
    0: 'short_backward',
    1: 'short_lateral',
    2: 'short_forward',
    3: 'long_backward',
    4: 'long_lateral',
    5: 'long_forward',
    6: 'shoot',
    7: 'carry'
}

# Reverse mapping
ACTION_IDS = {v: k for k, v in ACTION_NAMES.items()}

# Absorbing state indices (added after field states)
# With 7×11 grid = 77 field states, absorbing states are 77, 78, 79
ABSORBING_STATES = {
    'goal': lambda n_field_states: n_field_states,  # Successful shot
    'no_goal': lambda n_field_states: n_field_states + 1,  # Failed shot
    'loss_possession': lambda n_field_states: n_field_states + 2  # Failed pass/carry
}


class FieldGrid:
    """
    Represents the discretized field as a grid for MDP state space.
    
    Parameters
    ----------
    n_rows : int, default=7
        Number of grid rows (vertical divisions)
    n_cols : int, default=11
        Number of grid columns (horizontal divisions)
    pitch_length : float, default=105
        Pitch length in meters
    pitch_width : float, default=68
        Pitch width in meters
        
    Attributes
    ----------
    n_states : int
        Total number of field states (n_rows * n_cols = 77)
    cell_width : float
        Width of each cell in meters (~15m)
    cell_height : float
        Height of each cell in meters (~6.2m)
    """
    
    def __init__(
        self,
        n_rows: int = 7,
        n_cols: int = 11,
        pitch_length: float = 105,
        pitch_width: float = 68
    ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        
        self.n_states = n_rows * n_cols
        self.cell_width = pitch_length / n_cols
        self.cell_height = pitch_width / n_rows
        
    def xy_to_state(self, x: float, y: float) -> int:
        """
        Convert (x, y) coordinates to state index.
        
        Parameters
        ----------
        x : float
            X coordinate (0 to pitch_length)
        y : float
            Y coordinate (0 to pitch_width)
            
        Returns
        -------
        int
            State index (0 to n_states-1)
            
        Examples
        --------
        >>> grid = FieldGrid()
        >>> state = grid.xy_to_state(52.5, 34.0)  # Center of pitch
        """
        # Clip to bounds
        x = np.clip(x, 0, self.pitch_length - 0.001)
        y = np.clip(y, 0, self.pitch_width - 0.001)
        
        col = int(x / self.cell_width)
        row = int(y / self.cell_height)
        
        return row * self.n_cols + col
    
    def state_to_xy(self, state: int) -> Tuple[float, float]:
        """
        Convert state index to center (x, y) coordinates of cell.
        
        Parameters
        ----------
        state : int
            State index
            
        Returns
        -------
        tuple of float
            (x, y) coordinates of cell center
        """
        row = state // self.n_cols
        col = state % self.n_cols
        
        x = (col + 0.5) * self.cell_width
        y = (row + 0.5) * self.cell_height
        
        return x, y
    
    def get_cell_bounds(self, state: int) -> Dict[str, float]:
        """
        Get the boundary coordinates of a grid cell.
        
        Parameters
        ----------
        state : int
            State index
            
        Returns
        -------
        dict
            Dictionary with keys: x_min, x_max, y_min, y_max
        """
        row = state // self.n_cols
        col = state % self.n_cols
        
        return {
            'x_min': col * self.cell_width,
            'x_max': (col + 1) * self.cell_width,
            'y_min': row * self.cell_height,
            'y_max': (row + 1) * self.cell_height
        }
    
    def get_neighbors(self, state: int, include_diagonals: bool = True) -> list:
        """
        Get neighboring states.
        
        Parameters
        ----------
        state : int
            State index
        include_diagonals : bool, default=True
            Whether to include diagonal neighbors
            
        Returns
        -------
        list of int
            Neighboring state indices
        """
        row = state // self.n_cols
        col = state % self.n_cols
        
        neighbors = []
        
        # Orthogonal neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.n_rows and 0 <= new_col < self.n_cols:
                neighbors.append(new_row * self.n_cols + new_col)
        
        # Diagonal neighbors
        if include_diagonals:
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < self.n_rows and 0 <= new_col < self.n_cols:
                    neighbors.append(new_row * self.n_cols + new_col)
        
        return neighbors


def classify_action(pass_length: str, pass_direction: str) -> int:
    """
    Map pass type to action ID.
    
    Parameters
    ----------
    pass_length : str
        'short' or 'long' (no medium)
    pass_direction : str
        'forward', 'lateral', or 'backward'
        
    Returns
    -------
    int
        Action ID (0-5 for passes, 6=shoot, 7=carry)
        
    Examples
    --------
    >>> classify_action('short', 'forward')
    2
    >>> classify_action('long', 'lateral')
    4
    """
    pass_type = f"{pass_length}_{pass_direction}"
    return ACTION_IDS.get(pass_type, -1)


def add_state_action_encoding(
    df: pd.DataFrame,
    grid: FieldGrid
) -> pd.DataFrame:
    """
    Add state and action encodings to action DataFrame (passes + shots + carries).
    
    Parameters
    ----------
    df : pd.DataFrame
        Action events with rescaled coordinates
        For passes: must have pass_length and pass_direction columns
        For shots: must have action_type='shoot' column
        For carries: must have action_type='carry' column
    grid : FieldGrid
        Grid object for state encoding
        
    Returns
    -------
    pd.DataFrame
        Actions with added columns:
        - state_from: starting state index
        - state_to: ending state index (or absorbing state)
        - action: action ID (0-7 for passes, 8 for shoot, 9 for carry)
    """
    df = df.copy()
    
    # Encode starting states
    df['state_from'] = df.apply(
        lambda row: grid.xy_to_state(row['x_start_rescaled'], row['y_start_rescaled']),
        axis=1
    )
    
    # Encode ending states
    df['state_to'] = df.apply(
        lambda row: grid.xy_to_state(row['x_end_rescaled'], row['y_end_rescaled']),
        axis=1
    )
    
    # Encode actions
    if 'action_type' in df.columns:
        # Handle passes, shots, and carries
        def get_action_id(row):
            if row.get('action_type') == 'shoot':
                return 8
            elif row.get('action_type') == 'carry':
                return 9
            else:  # pass
                return classify_action(row.get('pass_length', ''), row.get('pass_direction', ''))
        
        df['action'] = df.apply(get_action_id, axis=1)
    else:
        # Legacy: only passes
        df['action'] = df.apply(
            lambda row: classify_action(row['pass_length'], row['pass_direction']),
            axis=1
        )
    
    return df


def get_action_name(action_id: int) -> str:
    """Get human-readable name for action ID."""
    return ACTION_NAMES.get(action_id, 'unknown')


def create_action_availability_mask(
    grid: FieldGrid,
    shoot_distance_threshold: float = 30.0
) -> np.ndarray:
    """
    Create binary mask indicating which actions are available in each state.
    
    Parameters
    ----------
    grid : FieldGrid
        Grid object
    shoot_distance_threshold : float, default=30.0
        Maximum distance from goal (meters) where shooting is allowed
        
    Returns
    -------
    np.ndarray
        Boolean array of shape (n_states, 8) where True means action is available
        
    Notes
    -----
    Constraints applied:
    - Shooting (action=6): only allowed within shoot_distance_threshold of goal
    - Backward passes (actions 0, 3): disabled in leftmost column (defensive edge)
    - Forward passes (actions 2, 5): disabled in rightmost column (attacking edge)
    - Carries (action=7): always available
    - Lateral passes (actions 1, 4): always available
    """
    n_actions = len(ACTION_NAMES)  # 8 actions
    mask = np.ones((grid.n_states, n_actions), dtype=bool)
    
    # Get shooting threshold in x-coordinate
    # Goal is at x=105m, so shooting allowed when x > (105 - threshold)
    shoot_x_threshold = grid.pitch_length - shoot_distance_threshold
    
    for state in range(grid.n_states):
        # Get state position
        row = state // grid.n_cols
        col = state % grid.n_cols
        
        # Calculate cell center x-coordinate
        x_center = (col + 0.5) * grid.cell_width
        
        # Shooting constraint: disable if > 30m from goal
        if x_center < shoot_x_threshold:
            mask[state, 6] = False  # Disable shooting
        
        # Backward pass constraints: disable in leftmost column
        if col == 0:
            mask[state, 0] = False  # short_backward
            mask[state, 3] = False  # long_backward
        
        # Forward pass constraints: disable in rightmost column
        if col == grid.n_cols - 1:
            mask[state, 2] = False  # short_forward
            mask[state, 5] = False  # long_forward
    
    return mask


def get_available_actions(state: int, mask: np.ndarray) -> list:
    """Get list of available action IDs for a state."""
    return [action_id for action_id in range(mask.shape[1]) if mask[state, action_id]]


def visualize_grid(grid: FieldGrid) -> None:
    """
    Print ASCII visualization of the grid structure.
    
    Parameters
    ----------
    grid : FieldGrid
        Grid object to visualize
    """
    print(f"\nField Grid: {grid.n_rows} rows × {grid.n_cols} cols = {grid.n_states} states")
    print(f"Cell size: {grid.cell_width:.2f}m × {grid.cell_height:.2f}m")
    print("\nGrid structure (each number is a state):")
    
    for row in range(grid.n_rows):
        row_str = ""
        for col in range(grid.n_cols):
            state = row * grid.n_cols + col
            row_str += f"{state:3d} "
        print(row_str)
    print()


if __name__ == "__main__":
    # Example usage
    print("State-Action Space Module - Skeleton Created")
    
    # Test grid
    grid = FieldGrid(n_rows=17, n_cols=22)
    visualize_grid(grid)
    
    # Test coordinate conversion
    x, y = 52.5, 34.0  # Center of pitch
    state = grid.xy_to_state(x, y)
    x_back, y_back = grid.state_to_xy(state)
    print(f"Test: ({x}, {y}) -> state {state} -> ({x_back:.2f}, {y_back:.2f})")
    
    # Test action classification
    print(f"\nAction IDs:")
    for aid, aname in ACTION_NAMES.items():
        print(f"  {aid}: {aname}")
