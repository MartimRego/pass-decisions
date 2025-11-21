# Pass Decision Analysis - MDP Framework

**Final Project**: Deep Learning & AI in Sport Course  
**Due Date**: November 13, 2025

## 📁 Project Structure

```
Project/
├── pass_decision_analysis.ipynb    # Main analysis notebook
├── src/                            # Source code modules
│   ├── __init__.py                 # Package initialization
│   ├── data_processing.py          # Data loading & pass classification
│   ├── state_action.py             # State/action space encoding
│   ├── mdp.py                      # MDP construction
│   ├── success_modeling.py         # Quality-quantity trade-offs
│   ├── analysis.py                 # Fundamental matrix & counterfactuals
│   └── visualization.py            # Plotting functions
├── data/                           # Intermediate processed data
├── outputs/                        # Generated outputs
│   └── figures/                    # Saved visualizations
├── project_plan.txt                # Project overview
├── reference-paper.pdf             # Van Roy et al. (2020)
└── README.md                       # This file
```

## 🎯 Overview

Analyzes pass decision-making in soccer using Markov Decision Processes, extending Van Roy et al.'s methodology from shooting to passes.

**Key Features**:
- 9-action space: short/medium/long × forward/lateral/backward + shoot
- Full-field 22×17 grid (374 states)
- Quality-quantity trade-off modeling (Method 3)
- Counterfactual policy analysis

## 📊 Data

- **Source**: Premier League 2024 season event data
- **Scope**: ~380 matches, ~570,000 passes
- **Coverage**: ~170 observations per state-action pair

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Ensure you're in the twelve-deep-learning environment
cd Project

# Test imports
python -c "from src import data_processing; print('✓ Imports working')"
```

### 2. Run Main Notebook

```bash
jupyter notebook pass_decision_analysis.ipynb
```

### 3. Module Testing

Test individual modules:

```python
# Test state-action module
from src import state_action as sa
grid = sa.FieldGrid(n_rows=17, n_cols=22)
sa.visualize_grid(grid)
```

## 📦 Module Descriptions

### `data_processing.py`
- Load Premier League event data
- Filter and rescale coordinates
- Classify passes into 9 categories
- Validation and sanity checks

**Key Functions**:
- `load_premier_league_events()` - Load season data
- `classify_pass_type()` - Categorize passes
- `validate_pass_classifications()` - Quality checks

### `state_action.py`
- Define field grid discretization
- Encode coordinates to states
- Map pass types to action IDs

**Key Classes/Functions**:
- `FieldGrid` - Grid representation
- `classify_action()` - Pass type → action ID
- `ACTION_NAMES` - Action label dictionary

### `mdp.py`
- Build transition matrix P(s,a,s')
- Build policy matrix π(a|s)
- Build reward function R
- MDP validation

**Key Functions**:
- `build_transition_matrix()` - With Laplace smoothing
- `build_policy_matrix()` - From observed frequencies
- `validate_mdp()` - Probability constraints

### `success_modeling.py`
- Quality distribution analysis
- Quality-quantity trade-off modeling
- Success rate adjustment (Van Roy Method 3)

**Key Functions**:
- `compute_quality_distributions()` - μ, μ_below, μ_top90
- `adjust_success_rate()` - Frequency change → success rate
- `modify_transition_matrix()` - Create P' with adjustments

### `analysis.py`
- Fundamental matrix computation
- Expected goals calculation
- Optimal action selection
- Counterfactual policy evaluation

**Key Functions**:
- `compute_fundamental_matrix()` - N = (I - Q)^(-1)
- `expected_goals_total()` - E[goals | policy]
- `optimal_action_per_state()` - Best action per zone
- `counterfactual_analysis()` - What-if scenarios

### `visualization.py`
- Pitch plots with mplsoccer
- Heat maps for optimal actions
- Policy comparison plots
- Trade-off curves

**Key Functions**:
- `plot_optimal_actions_heatmap()` - Best action per cell
- `plot_counterfactual_results()` - Scenario comparison
- `plot_quality_quantity_tradeoff()` - Frequency vs. E[goals]

## 🗓️ Development Timeline

**Weekend 1 (Nov 1-2)**: Data processing, grid definition, pass classification  
**Week 2 (Nov 3-7)**: MDP construction, quality modeling  
**Weekend 2 (Nov 8-9)**: Analysis, counterfactuals, visualizations  
**Final (Nov 10-12)**: Write-up, polish, submission prep

## 📝 Implementation Notes

### TODO Items in Code
Each module has `NotImplementedError` placeholders for functions to be implemented. Search for:
```python
raise NotImplementedError("To be implemented")
```

### Design Decisions
1. **Grid size**: 22×17 chosen to balance resolution vs. data sparsity
2. **Pass thresholds**: Short <15m, Medium 15-25m, Long >25m
3. **Smoothing**: Laplace α=1 for transition probabilities
4. **Quality metric**: Binary success/failure (can upgrade to xT)

### Validation Checks
- All probability distributions sum to 1
- No NaN or Inf values
- Reasonable success rates (short >> long)
- Expected goals ≈ actual goals (within 10-15%)

## 🎓 References

**Primary Paper**:
Van Roy, M., Robberechts, P., Yang, W. C., De Raedt, L., & Davis, J. (2020). *Leaving Goals on the Pitch: Evaluating Decision Making in Soccer*. MIT SLOAN Sports Analytics Conference.

**Course Materials**:
- Module 4: Possession Values (xT, VAEP)
- Module 2: Clustering & Visualization

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

**Status**: 🚧 In Development (Skeleton Complete - Implementation in Progress)

**Last Updated**: November 1, 2025
