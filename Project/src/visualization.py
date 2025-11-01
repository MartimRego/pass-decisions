"""
Visualization Module
====================

Functions for creating plots, heat maps, and pitch visualizations.

Author: Your Name
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mplsoccer import Pitch
from typing import Optional, List, Dict, Tuple

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_pass_distribution(
    passes: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plot distribution of pass types and lengths.
    
    Parameters
    ----------
    passes : pd.DataFrame
        Pass events with classification columns
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Pass length distribution
    passes['pass_length'].value_counts().plot(kind='bar', ax=axes[0])
    axes[0].set_title('Pass Length Distribution')
    axes[0].set_xlabel('Length Category')
    axes[0].set_ylabel('Count')
    
    # Pass direction distribution
    passes['pass_direction'].value_counts().plot(kind='bar', ax=axes[1])
    axes[1].set_title('Pass Direction Distribution')
    axes[1].set_xlabel('Direction')
    axes[1].set_ylabel('Count')
    
    # Combined type distribution
    passes['pass_type'].value_counts().plot(kind='barh', ax=axes[2])
    axes[2].set_title('Pass Type Distribution')
    axes[2].set_xlabel('Count')
    axes[2].set_ylabel('Pass Type')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_grid_on_pitch(
    grid,
    highlight_states: Optional[List[int]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize the field grid discretization.
    
    Parameters
    ----------
    grid : FieldGrid
        Grid object
    highlight_states : list of int, optional
        States to highlight
    save_path : str, optional
        Path to save figure
    """
    pitch = Pitch(
        pitch_type='custom',
        pitch_length=grid.pitch_length,
        pitch_width=grid.pitch_width,
        line_color='white',
        pitch_color='#22ab4d'
    )
    
    fig, ax = pitch.draw(figsize=(12, 8))
    
    # Draw grid lines
    for col in range(grid.n_cols + 1):
        x = col * grid.cell_width
        ax.plot([x, x], [0, grid.pitch_width], color='white', alpha=0.3, linewidth=0.5)
    
    for row in range(grid.n_rows + 1):
        y = row * grid.cell_height
        ax.plot([0, grid.pitch_length], [y, y], color='white', alpha=0.3, linewidth=0.5)
    
    # Highlight specific states
    if highlight_states:
        for state in highlight_states:
            bounds = grid.get_cell_bounds(state)
            rect = plt.Rectangle(
                (bounds['x_min'], bounds['y_min']),
                grid.cell_width,
                grid.cell_height,
                facecolor='yellow',
                alpha=0.5,
                edgecolor='red',
                linewidth=2
            )
            ax.add_patch(rect)
    
    ax.set_title(f'Field Grid: {grid.n_rows}×{grid.n_cols} = {grid.n_states} states', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_optimal_actions_heatmap(
    optimal_actions: np.ndarray,
    grid,
    action_names: Dict[int, str],
    save_path: Optional[str] = None
) -> None:
    """
    Create heat map showing optimal action for each grid cell.
    
    Parameters
    ----------
    optimal_actions : np.ndarray
        Optimal action ID for each state
    grid : FieldGrid
        Grid object
    action_names : dict
        Mapping from action ID to name
    save_path : str, optional
        Path to save figure
    """
    pitch = Pitch(
        pitch_type='custom',
        pitch_length=grid.pitch_length,
        pitch_width=grid.pitch_width,
        line_color='white',
        pitch_color='#22ab4d'
    )
    
    fig, ax = pitch.draw(figsize=(14, 9))
    
    # Create color map for actions
    unique_actions = np.unique(optimal_actions)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_actions)))
    action_to_color = {action: colors[i] for i, action in enumerate(unique_actions)}
    
    # Plot each cell
    for state in range(grid.n_states):
        bounds = grid.get_cell_bounds(state)
        action = optimal_actions[state]
        
        rect = plt.Rectangle(
            (bounds['x_min'], bounds['y_min']),
            grid.cell_width,
            grid.cell_height,
            facecolor=action_to_color[action],
            alpha=0.6,
            edgecolor='white',
            linewidth=0.5
        )
        ax.add_patch(rect)
    
    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=action_to_color[a], alpha=0.6, label=action_names[a])
        for a in unique_actions
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    ax.set_title('Optimal Action per Zone', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_expected_goals_heatmap(
    expected_goals: np.ndarray,
    grid,
    save_path: Optional[str] = None
) -> None:
    """
    Create heat map showing expected goals from each state.
    
    Parameters
    ----------
    expected_goals : np.ndarray
        Expected goals value for each state
    grid : FieldGrid
        Grid object
    save_path : str, optional
        Path to save figure
    """
    pitch = Pitch(
        pitch_type='custom',
        pitch_length=grid.pitch_length,
        pitch_width=grid.pitch_width,
        line_color='white',
        pitch_color='#22ab4d'
    )
    
    fig, ax = pitch.draw(figsize=(14, 9))
    
    # Reshape for heatmap
    eg_grid = expected_goals.reshape(grid.n_rows, grid.n_cols)
    
    # Plot using imshow
    im = ax.imshow(
        eg_grid,
        extent=[0, grid.pitch_length, 0, grid.pitch_width],
        origin='lower',
        aspect='auto',
        cmap='YlOrRd',
        alpha=0.7
    )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Expected Goals', rotation=270, labelpad=20)
    
    ax.set_title('Expected Goals per Zone', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_policy_comparison(
    pi_original: np.ndarray,
    pi_modified: np.ndarray,
    state: int,
    action_names: Dict[int, str],
    save_path: Optional[str] = None
) -> None:
    """
    Compare original vs modified policy for a specific state.
    
    Parameters
    ----------
    pi_original : np.ndarray
        Original policy matrix
    pi_modified : np.ndarray
        Modified policy matrix
    state : int
        State to visualize
    action_names : dict
        Action ID to name mapping
    save_path : str, optional
        Path to save figure
    """
    actions = list(action_names.keys())
    names = [action_names[a] for a in actions]
    
    x = np.arange(len(actions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width/2, pi_original[state, actions], width, label='Original', alpha=0.8)
    ax.bar(x + width/2, pi_modified[state, actions], width, label='Modified', alpha=0.8)
    
    ax.set_xlabel('Action', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f'Policy Comparison - State {state}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_counterfactual_results(
    results: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize results from counterfactual analysis.
    
    Parameters
    ----------
    results : pd.DataFrame
        Counterfactual analysis results
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Expected goals by scenario
    results_sorted = results.sort_values('expected_goals', ascending=False)
    axes[0].barh(results_sorted['scenario'], results_sorted['expected_goals'], color='skyblue')
    axes[0].set_xlabel('Expected Goals')
    axes[0].set_title('Expected Goals by Scenario')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Difference from baseline
    baseline_eg = results[results['scenario'] == 'baseline']['expected_goals'].values[0]
    results_diff = results[results['scenario'] != 'baseline'].copy()
    
    colors = ['green' if x > 0 else 'red' for x in results_diff['difference']]
    axes[1].barh(results_diff['scenario'], results_diff['difference'], color=colors, alpha=0.7)
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Goal Difference from Baseline')
    axes[1].set_title('Impact of Policy Changes')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_quality_quantity_tradeoff(
    frequency_changes: np.ndarray,
    expected_goals: np.ndarray,
    baseline_eg: float,
    title: str = "Quality-Quantity Trade-off",
    save_path: Optional[str] = None
) -> None:
    """
    Plot expected goals vs. frequency change curve.
    
    Parameters
    ----------
    frequency_changes : np.ndarray
        Array of frequency change percentages (e.g., -20 to +50)
    expected_goals : np.ndarray
        Expected goals for each frequency change
    baseline_eg : float
        Baseline expected goals (no change)
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(frequency_changes, expected_goals, marker='o', linewidth=2, markersize=6)
    ax.axhline(y=baseline_eg, color='red', linestyle='--', label='Baseline')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Find optimal
    optimal_idx = np.argmax(expected_goals)
    ax.plot(frequency_changes[optimal_idx], expected_goals[optimal_idx], 
            marker='*', markersize=20, color='gold', label='Optimal')
    
    ax.set_xlabel('Frequency Change (%)', fontsize=12)
    ax.set_ylabel('Expected Goals', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Visualization Module - Skeleton Created")
    print("Ready for creating plots and heat maps!")
