# Pass Decision Analysis – MDP Framework

**Final Project** – Deep Learning & AI in Sport  
**Submission Date**: November 13, 2025  
**Author**: Martim Rêgo

## 📁 Project Structure

```
Project/
├── pass_decision_analysis.ipynb    # Main analysis notebook (final results)
├── src/                            # Source code modules
│   ├── __init__.py
│   ├── data_processing.py          # Data loading & pass classification
│   ├── state_action.py             # State/action space encoding
│   ├── mdp.py                      # MDP construction
│   ├── success_modeling.py         # Quality–quantity trade-offs
│   ├── analysis.py                 # Fundamental matrix & counterfactuals
│   └── visualization.py            # Plotting utilities
├── data/                           # Intermediate, lightweight processed data
├── outputs/                        # Generated outputs (figures, tables)
│   └── figures/                    # Saved visualizations
├── project_plan.txt                # Original project plan
├── reference-paper.pdf             # Van Roy et al. (2020)
└── README.md                       # This file
```

> Large raw datasets and heavy intermediate tensors are **not** stored in this repo
> (they are ignored via `.gitignore` and live in the course environment).

## 🎯 Overview

Evaluates pass decision-making in soccer using Markov Decision Processes (MDPs),
extending Van Roy et al.'s methodology from **shooting** to **passing**.

**Key Features**
- 8‑action space: 6 pass types (short/long × forward/lateral/backward) + shoot + carry
- 7×11 field grid (77 field states + 3 absorbing)
- Quality–quantity trade-off modeling (Van Roy Method 3)
- Counterfactual policy analysis and regret by state

## 📊 Data

- **Source**: Premier League 2024 season event data (Twelve)
- **Scope**: Full Premier League 2024 season (~460k actions: passes, shots, carries)
- **Coverage**: High state–action coverage suitable for MDP estimation

## 🚀 Quick Start

### 1. Environment Setup

```bash
cd Project
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\\Scripts\\activate
pip install -r ../requirements.txt

# Test imports
python -c "from src import data_processing; print('✓ Imports working')"
```

### 2. Train xG Model (Required Before MDP Construction)

```bash
# Train the logistic regression xG model on shot data
python train_xg_model.py
```

This creates `data/xg_model.pkl` which is used as the Bayesian prior for shot success probabilities.

### 3. Run Main Notebook

```bash
jupyter notebook pass_decision_analysis.ipynb
```

### 4. Quick Module Check

```python
from src import state_action as sa
grid = sa.FieldGrid(n_rows=17, n_cols=22)
sa.visualize_grid(grid)
```

## 📦 Module Descriptions

### `data_processing.py`
- Load Premier League event and metadata
- Filter to passes/shots/carries and add success labels
- Rescale and normalize coordinates
- Classify passes into 6 types (short/long × fwd/lat/back)

**Key Functions**
- `load_premier_league_events()` – Load dynamic event data
- `load_match_metadata()` – Load match/team metadata
- `rescale_coordinates()` – Convert SkillCorner coords to meters
- `normalize_attack_direction()` – Create *_norm coords
- `extract_pass_events()` – Filter and clean passes
- `extract_shot_events()` – Filter and clean shots
- `extract_carry_events()` – Filter and clean carries
- `combine_passes_shots_carries()` – Build unified action table
- `classify_pass_type()` – Add pass_type / length / direction
- `validate_pass_classifications()` – Sanity checks for types
- `get_data_summary()` – High‑level dataset summary

### `xg_model.py`
- **NEW**: Trained logistic regression xG model (replaces hand-crafted geometric model)
- Calculate viewing angle to goal
- Train and save xG model using shot data
- Provide Bayesian prior for shot success probabilities

**Key Functions**
- `calculate_goal_angle()` – Compute viewing angle from position
- `build_shot_dataset()` – Prepare shot data for training
- `train_logistic_xg_model()` – Train logistic regression on angle-to-goal
- `save_xg_model()` / `load_xg_model()` – Model persistence
- `predict_xg()` – Get xG prediction for a given angle
- `apply_bayesian_shrinkage()` – Apply Bayesian smoothing using trained prior
- `geometric_xg_model()` – DEPRECATED: Old hand-crafted model (fallback only)

**Model Details**
- Feature: Viewing angle to goal (degrees)
- Model: Logistic regression (scikit-learn)
- Training: ~8,700 shots from Premier League 2024
- Performance: ROC-AUC ~0.70
- Usage: Bayesian prior with α=10 pseudo-observations

### `state_action.py`
- Define field grid discretization
- Encode coordinates to state IDs
- Map pass types to action IDs

**Key Classes/Functions**
- `FieldGrid` – Grid representation
- `classify_action()` – Pass type → action ID
- `add_state_action_encoding()` – Add state_from/state_to/action
- `create_action_availability_mask()` – State–action mask
- `get_available_actions()` – List valid actions per state
- `visualize_grid()` – ASCII grid view
- `ACTION_NAMES` / `ACTION_IDS` – Action label dictionaries

### `mdp.py`
- Build transition matrix P(s, a, s')
- Build policy matrix π(a|s)
- Build reward function R
- Validate MDP structure

**Key Functions**
- `build_transition_matrix()` – With Laplace smoothing
- `build_policy_matrix()` – From observed frequencies
- `build_reward_function()` – Goal/no‑goal reward vector
- `build_team_mdps()` – Per‑team MDPs in batch
- `validate_mdp()` – Probability constraints
- `get_mdp_statistics()` – Sparsity/entropy/coverage summary
- `save_team_mdps()` / `load_team_mdp()` – Disk I/O helpers

### `success_modeling.py`
- Quality distribution analysis
- Quality–quantity trade-off modeling
- Success rate adjustment (Van Roy Method 3)

**Key Functions**
- `compute_quality_distributions()` – μ, μ_below, μ_top90
- `adjust_success_rate()` – Frequency change → success rate
- `modify_transition_matrix()` – Create P' with adjustments

### `analysis.py`
- Fundamental matrix computation
- Expected goals calculation
- Optimal action selection
- Counterfactual policy evaluation

**Key Functions**
- `compute_fundamental_matrix()` – N = (I − Q)⁻¹
- `expected_goals_total()` – E[goals | policy]
- `optimal_action_per_state()` – Best action per zone
- `counterfactual_analysis()` – What‑if scenarios
- `expected_goals_from_state()` – E[goals | single state]
- `create_uniform_adjustment()` / `modify_policy()` – Policy tweaks
- `compute_spatial_impact()` – Grid‑level goal impact

### `visualization.py`
- Pitch plots with `mplsoccer`
- Heat maps for optimal actions and regret
- Policy comparison plots
- Trade‑off curves

**Key Functions**
- `plot_optimal_actions_heatmap()` – Best action per cell
- `plot_counterfactual_results()` – Scenario comparison
- `plot_quality_quantity_tradeoff()` – Frequency vs. E[goals]
- `plot_pass_distribution()` – Pass type/length distributions
- `plot_grid_on_pitch()` – Visualize grid on pitch
- `plot_expected_goals_heatmap()` – xG per zone
- `plot_policy_comparison()` – Original vs modified π(a|s)

## 📝 Implementation Notes

### Design Decisions
1. **Grid size**: 7×11 to balance resolution vs. data sparsity
2. **Pass thresholds**: Short ≤25m, Long >25m
3. **Smoothing**: Laplace α = 1 for transition probabilities

### Validation Checks
- All probability distributions sum to 1
- No NaN or Inf values
- Reasonable success rates (short ≫ long)

## 🎓 References

**Primary Paper**  
Van Roy, M., Robberechts, P., Yang, W. C., De Raedt, L., & Davis, J. (2020).
*Leaving Goals on the Pitch: Evaluating Decision Making in Soccer*. MIT Sloan
Sports Analytics Conference.


## 🤝 Contributing

This is a course project, but feel free to:
1. Report issues in implementation
2. Suggest improvements to methodology
3. Propose additional analyses

## 📧 Contact

**Author**: Martim Rêgo  
**Course**: Deep Learning & AI in Sport  
**Mentor**: Pegah Rahimian

---

**Status**: ✅ Completed for course submission

**Last Updated**: November 13, 2025
