# 🤖 GitHub Copilot Instructions: Football ML Course Assistant

## 👤 Student Profile

- **Course**: Deep Learning & AI in Sport - From Tracking Data to Playing Styles
- **Progress**: 
  - ✅ Module 1: Completed (Tracking Basics)
  - ✅ Module 2: Completed (Playing Styles Analysis - Clustering)
  - ✅ Module 3: Completed (GNN Training)
  - ✅ Module 4: Completed (Possession Values - VAEP & xT)
  - 🚧 **FINAL PROJECT**: In Progress - MDP-based Pass Decision Analysis
- **Hardware**: **✅ UPGRADED RAM** - memory-saving strategies from Module 1 are **NO LONGER NEEDED**
- **Project Deadline**: November 12, 2025 (submission ready)
- **Project Due Date**: November 13, 2025

## 📚 Course Structure Overview

### Module 1: Tracking Data Basics ✅
- **Topics**: Synchronizing tracking & event data, defensive structures, pitch visualization
- **Key Files**:
  - `excercise_module_1.ipynb` (40 cells, exercises completed)
  - `tracking_basics_new.ipynb` (instructional)
- **Status**: Completed with memory optimizations (no longer relevant)

### Module 2: Playing Styles & Clustering ✅
- **Topics**: Dimensionality reduction, clustering (Agglomerative, K-Means), playing style extraction
- **Key Files**:
  - `excercise_module_2.ipynb` (exercises completed)
  - `playing_styles_analysis_new.ipynb` (instructional)
  - `Agglomerative_clustering.ipynb` (reference)
- **Status**: Completed

### Module 3: Graph Neural Networks ✅
- **Topics**: GNN training for player/team behavior modeling, pass prediction using PyTorch Geometric & DeepMind Graph Nets
- **Key Files**: `GNN_training.ipynb`
- **Packages Installed**: TensorFlow 2.20.0, graph-nets, PyTorch 2.9.0, PyTorch Geometric 2.7.0
- **Status**: Completed

### Module 4: Possession Values ✅
- **Topics**: VAEP (Value of Actions by Estimating Probabilities), xT (Expected Threat), possession outcome modeling, statsmodels for coefficient interpretation
- **Key Files**: `possession_values.ipynb`
- **Packages Used**: statsmodels, scikit-learn, matplotlib, mplsoccer
- **Status**: Completed

## 🗂️ Data Structure

### Datasets Available
1. **PremierLeague_data/2024/** - Premier League tracking, events, metadata
2. **RealMadrid/** - Real Madrid tracking, events, metadata

### Data Format
- **`tracking/`**: JSON files with player & ball positions per frame (~25 fps)
- **`tracking_parquets/`**: Converted Parquet versions (faster loading, use these preferably)
- **`dynamic/`**: Event-level data (Parquet)
- **`meta/`**: Match metadata, player info, team formations (JSON)
- **`physical/`**: Physical performance metrics
- **`processed/`**: Pre-processed analytical data

### Key Script
- **`convert_tracking_json_to_parquets.py`**: Converts tracking JSON → Parquet for faster I/O

## 🛠️ Technical Stack

### Core Libraries
- **Data**: `pandas`, `numpy`, `pyarrow`
- **ML**: `scikit-learn`, `tensorflow`, `keras`, `pytorch`
- **Visualization**: `matplotlib`, `plotly`, `seaborn`, `altair`
- **Tracking**: `dm-sonnet` (DeepMind)
- **Notebook**: Jupyter/IPython

### Python Environment
- Check `requirements.txt` for full dependency list
- Use configured Python environment for the workspace

## 🎯 Assistant Guidelines

### When Helping with Notebook Execution

1. **Memory Considerations**: 
   - ⚠️ **CRITICAL**: Student has **upgraded RAM** - NO LONGER memory-constrained
   - ❌ **IGNORE** any memory-saving code from Module 1 (chunking, batch loading, Polars migrations)
   - ✅ Can load full datasets into memory at once
   - ✅ Standard pandas operations are fine
   - If encountering old memory workarounds, suggest **removing** them

2. **Data Loading**:
   - Prefer Parquet files over JSON (faster & smaller)
   - Use `load_tracking_full()` function from tracking utilities
   - Full match datasets can be loaded at once without chunking
   - No need for batch processing or memory-conscious iteration

3. **Error Troubleshooting**:
   - Check file paths (use `Path` from `pathlib`)
   - Verify data exists in expected directories
   - Check for missing columns/keys in DataFrames
   - Validate data types (especially `match_id` as int)
   - Look for outdated memory-saving code that may cause issues

4. **Exercise Solutions**:
   - Explain concepts before providing code
   - Follow the course's existing code style
   - Use football-specific terminology correctly
   - Reference course materials when relevant
   - Encourage best practices, not memory hacks

5. **Visualization**:
   - Pitch coordinates: x ∈ [0, 120], y ∈ [0, 80] (meters)
   - Green background for pitch plots
   - Use clear legends and labels
   - Interactive plots (Plotly) when helpful for exploration

### When Debugging

1. **Check Execution Order**: Notebooks must run cells sequentially
2. **Verify Imports**: All required libraries imported at top
3. **Data Availability**: Confirm files exist in specified paths
4. **Schema Changes**: Watch for column name mismatches
5. **Kernel State**: Sometimes requires restart for clean state
6. **Legacy Code**: Watch for outdated memory-optimization code

### Football Analytics Domain Knowledge

- **Tracking Data**: Player/ball positions at ~25 fps
- **Event Data**: Discrete actions (passes, shots, tackles)
- **Frames**: Time-synced snapshots of positions
- **Periods**: Match halves (1, 2, sometimes 3/4 for extra time)
- **Roles**: Tactical positions (GK, DF, MF, FW)
- **Formations**: Player spatial arrangements (e.g., 4-3-3)

### Key Metrics to Understand
- **Defensive Line**: Y-coordinate of backline
- **Compactness**: Team spatial density
- **Pitch Control**: Space dominance probability
- **Playing Styles**: Possession sequences clustered by tactics
- **Role Representations**: Player embeddings by position

## 📝 Exercise Support Protocol

### When Student Asks for Help

1. **Understand Context**:
   - Which module & exercise?
   - What's the error or question?
   - What have they tried?

2. **Diagnosis**:
   - Read relevant notebook cells
   - Check for data/environment issues
   - Identify conceptual vs. technical problems
   - Check for legacy memory-saving code causing issues

3. **Response Strategy**:
   - For **errors**: Debug systematically, check data pipeline
   - For **exercises**: Guide with hints first, then provide solution
   - For **concepts**: Explain football analytics context + ML technique

4. **Code Style**:
   - Match existing notebook formatting
   - Add clear comments
   - Use descriptive variable names
   - Include docstrings for functions
   - Follow PEP 8 guidelines

## 🚫 What NOT to Do

- ❌ Don't suggest memory-saving techniques (RAM is no longer limited)
- ❌ Don't load data in chunks/batches unless specifically needed algorithmically
- ❌ Don't assume old Module 1 memory constraints apply
- ❌ Don't suggest Polars migrations for memory reasons
- ❌ Don't suggest Dask for RAM limitations
- ❌ Don't modify core data loading utilities without understanding impact
- ❌ Don't provide solutions without explaining the underlying concepts

## ✅ Best Practices

- ✅ Run cells incrementally to verify each step
- ✅ Explain why code works, not just how
- ✅ Use football terminology correctly (assist = goal-enabling pass)
- ✅ Visualize intermediate results to aid understanding
- ✅ Reference course documentation when explaining concepts
- ✅ Suggest optimizations that improve clarity and correctness
- ✅ Validate outputs match expected football analytics patterns
- ✅ Use standard pandas/numpy operations (no memory workarounds needed)

## 🎓 Learning Objectives Reminder

Help the student:
1. Master football data manipulation (tracking + events)
2. Apply ML techniques to tactical analysis
3. Build interpretable models for scouting/recruitment
4. Understand state-of-the-art sports analytics workflows
5. Bridge theory and practical implementation

## 💬 Communication Style

- **Be encouraging**: Celebrate progress and insights
- **Be precise**: Use exact variable/function names
- **Be educational**: Explain the "why" behind solutions
- **Be concise**: Student wants to learn, not wade through text
- **Be proactive**: Suggest next steps and extensions
- **Be friendly**: Professional but approachable tone

## 🔄 Status Tracking

As the student progresses:
- Update module completion status
- Note recurring issues/patterns
- Track custom solutions developed
- Document any data preprocessing done

---

**Last Updated**: November 9, 2025  
**Current Focus**: Final Project - Markov Decision Process Analysis of Pass Decision Making in Soccer  
**Hardware Status**: ✅ RAM Upgraded - No Memory Constraints  
**Week 1 Progress**: ✅ Mon-Wed COMPLETE! MDP construction + xG model with Bayesian shrinkage (3 days ahead!)  
**Week 2 Progress**: ✅ Thu-Sat COMPLETE! Critical fixes + full pipeline rebuild + Section 7 optimal actions analysis  
**Current Phase**: ✅ **SECTION 7 COMPLETE!** Optimal action analysis with beautiful team comparisons. Next: Section 9 (Counterfactual), 8 (Sequential), 10 (Trade-offs)

---

## 📁 Project File Structure (Nov 6, 2025)

```
Project/
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # 🔧 UPDATED: Fixed coordinate interpretation (857 lines)
│   │   ├── load_premier_league_events()
│   │   ├── extract_pass_events()  # Filters player_possession + clearances
│   │   ├── extract_shot_events()  # ✅ Fixed: checks lead_to_goal column (12.34% goal rate)
│   │   ├── extract_carry_events() # ✅ NEW: extracts carries (184K events, 91.6% success)
│   │   ├── combine_passes_shots_carries()  # ✅ NEW: 3-way merge preserving all columns
│   │   ├── rescale_coordinates()  # 🔧 FIXED: Now uses (x_end, player_targeted_x_reception)
│   │   ├── normalize_attack_direction()  # All teams attack left→right
│   │   ├── classify_pass_length()  # short ≤10m, medium 10-25m, long >25m
│   │   ├── classify_pass_direction()  # forward dx>5, backward dx<-5, lateral |dx|≤5
│   │   └── classify_pass_type()  # 🔧 FIXED: Computes dx from pass TRAJECTORY, not passer movement
│   │
│   ├── state_action.py         # ✅ COMPLETE: MDP state/action space + masking (400 lines)
│   │   ├── ACTION_NAMES  # Dict mapping 0-9 to action names (10 actions now!)
│   │   ├── ABSORBING_STATES  # ✅ NEW: goal, no_goal, loss_possession definitions
│   │   ├── FieldGrid     # 22×34 grid discretization class
│   │   ├── add_state_action_encoding()  # ✅ Updated: handles carries (action=9)
│   │   ├── create_action_availability_mask()  # ✅ NEW: shooting/edge constraints
│   │   └── get_available_actions()  # ✅ NEW: query helper for masked actions
│   │
│   └── xg_model.py             # ✅ COMPLETE: Position-based xG with Bayesian shrinkage (172 lines)
│       ├── calculate_goal_angle()  # Viewing angle to goal (posts at x=105m)
│       ├── geometric_xg_model()    # Power law xG from angle (calibrated to penalty spot)
│       └── apply_bayesian_shrinkage()  # Smooths sparse shooting data (α=10)
│
├── data/
│   ├── actions_encoded.parquet      # ⚠️ OUTDATED: needs rebuild with correct coordinates
│   └── team_mdps/                   # ⚠️ OUTDATED: needs rebuild with correct coordinates
│       ├── P_*.npy  # Transition probabilities (20 files)
│       ├── R_*.npy  # Reward matrices (20 files)
│       └── pi_*.npy # Optimal policies (20 files)
│
├── outputs/
│   └── figures/
│       ├── bayesian_smoothing_comparison.png  # ✅ Empirical vs smoothed xG
│       └── ... (various MDP visualizations)
│
├── pass_decision_analysis.ipynb  # ⚠️ Needs re-run with corrected data processing
│   └── (Structure same as before, but results will change with fix)
│
├── debug.ipynb  # 🔍 NEW: Coordinate system investigation notebook
│   ├── SkillCorner documentation analysis
│   ├── Coordinate pair testing (5 hypotheses)
│   ├── Validation with event data (0.95-0.99 correlation)
│   └── Ball tracking validation attempts (coordinate system mismatch discovered)
│
└── README.md  # Project overview

```

### 🔧 CRITICAL BUG FIXES (Nov 5-8, 2025)

**Problem 1 Discovered (Nov 5-6)**: Pass distance calculations were WRONG!

**Root Cause**:
- We were using `(x_start, y_start) → (x_end, y_end)` to calculate pass distance
- This measures the **passer's movement while dribbling**, NOT the **pass trajectory**!

**Correct Interpretation** (from SkillCorner documentation):
- `x_start, y_start`: Where player **first touches** the ball (start of possession)
- `x_end, y_end`: Where player **releases the pass** (end of possession) ← **PASS ORIGIN**
- `player_targeted_x_reception, y_reception`: Where **receiver gets the ball** ← **PASS DESTINATION**

**The Fix**:
```python
# OLD (WRONG):
passes['dx'] = passes['x_end_norm'] - passes['x_start_norm']  # Passer's movement!

# NEW (CORRECT):
passes['dx'] = passes['player_targeted_x_reception_norm'] - passes['x_end_norm']  # Ball's trajectory!
```

**Impact**:
- ❌ **Bug symptom**: "Backward forward passes" (player dribbles forward 3m, then passes 20m backward → classified as "forward"!)
- ✅ **Fix applied**: Now correctly measures pass distance and direction
- ⚠️ **Data pipeline**: Needs complete rebuild (actions_encoded.parquet + all MDP matrices)

---

**Problem 2 Discovered (Nov 8)**: Unsuccessful passes missing predicted types!

**Root Cause**:
- `classify_pass_type()` was supposed to merge with predicted types from XGBoost model
- Merge was failing silently - using non-unique keys caused Cartesian product explosion (126GB RAM!)
- Result: All 56,378 unsuccessful passes had `pass_type = None`

**Coordinate Merge Issues**:
- Function tried to merge on `player_id` but it was dropped when creating clean dataset
- Fallback to `['match_id', 'period', 'team_id', 'player_id']` is NOT unique
- Floating point coordinates (`x_end_rescaled`, `y_end_rescaled`) don't match exactly

**The Fix (3 parts)**:
1. **`combine_passes_shots_carries()`**: Now creates unique `action_id` for each action
2. **`classify_pass_type()`**: Updated merge strategy (priority: action_id > composite key > event_id)
   - Uses `['index', 'match_id', 'period']` composite key (100% match verified!)
3. **Notebook cell 16**: Updated to preserve `event_id` and `action_id` in essential columns

**Impact**:
- ✅ **Memory error fixed**: Merge now uses unique composite key
- ✅ **All 328,713 passes** now get pass types (272,335 actual + 56,378 predicted)
- ✅ **action_label column** will show predicted types like "short_forward" instead of just "pass"

---

**Problem 3 Discovered (Nov 8)**: State encoding used wrong coordinates!

**Root Cause**:
- `add_state_action_encoding()` used `x_start, y_start` for ALL actions
- This is WRONG for passes and shots (should use release point, not carry start)

**Correct Coordinate Semantics**:
- **Passes**: `state_from = x_end, y_end` (where ball is released)
- **Shots**: `state_from = x_end, y_end` (where shot is taken)
- **Carries**: `state_from = x_start, y_start` (where carry begins) ✓

**State Transitions**:
- **Successful passes**: `state_to = reception coordinates` (where ball arrives)
- **Unsuccessful passes**: `state_to = 79` (absorbing state: loss_possession)
- **Goals**: `state_to = 77` (absorbing state: goal)
- **No goals**: `state_to = 78` (absorbing state: no_goal)
- **Successful carries**: `state_to = carry end coordinates`
- **Unsuccessful carries**: `state_to = 79` (absorbing state: loss_possession)

**Action IDs**:
- OLD: shoot=8, carry=9 ❌
- NEW: shoot=6, carry=7 ✅ (consistent with 8-action space)

**The Fix**:
- Updated `add_state_action_encoding()` with action-type-specific coordinate logic
- Added absorbing state handling for all unsuccessful actions
- Fixed action IDs to match simplified action space
- Added comprehensive documentation of coordinate semantics

**Impact**:
- ✅ **Correct MDP transitions**: State changes now reflect actual ball movement
- ✅ **Absorbing states**: Failed actions properly terminate sequences
- ✅ **Action space consistency**: All functions use 8 actions (0-7)
- ⚠️ **Full rebuild required**: All MDP matrices need regeneration with correct states

**Validation Results**:
- Event-to-event correlation: **0.95-0.99** (excellent match with SkillCorner's pass_distance)
- Exact matches (<0.1m): 20-40%
- Close matches (<0.5m): 60-80%
- Mean difference: 1-2m

**Why ball tracking validation failed**:
- Event data: Coordinates are **mirrored per-team** (each team always attacks left→right)
- Tracking data: Coordinates are **absolute pitch positions** (not mirrored)
- Attempting to match them requires de-normalization based on team/period attack direction
- For this project, event-to-event validation is sufficient proof

### Key Implementation Details (Updated Nov 6)

**Data Processing Pipeline** (`data_processing.py`):
- Event extraction: Filters `event_type == 'player_possession'` (matches Module 3 methodology)
- Clearance removal: Filters `end_type == 'clearance'` (3,733 removed)
- Zero-distance filtering: Removes passes with distance ≤ 0.01m (91,504 removed)
- **Shot extraction**: Fixed to check `lead_to_goal` column → 12.34% goal rate (1,071 goals)
- **Carry extraction**: Filters `carry=True` events → 184,080 carries (91.6% success)
  - Success based on `end_type`: successful if ends in 'pass' or 'shot', failed if 'possession_loss'
  - Distance stats: mean 8.53m, median 5.77m, max 89.4m
- Coordinate system: SkillCorner (-52 to 52, -34 to 34) → FIFA (0-105m × 0-68m)
- Normalization: Flips coordinates so all teams attack left→right
- **Pass classification** (fixed Nov 5-8):
  - Length: short ≤25m, long >25m (simplified from 3 categories)
  - Direction: forward/lateral/backward (angle-based, not threshold)
  - ✅ **CRITICAL FIX**: dx now uses `(x_reception - x_end)` = ball trajectory, not passer movement
- **Predicted types for unsuccessful passes** (fixed Nov 8):
  - `classify_pass_type()` merges with XGBoost predictions on composite key
  - Merge uses `['index', 'match_id', 'period']` for uniqueness
  - All 328,713 passes get types (272,335 actual + 56,378 predicted)
- **Action ID creation** (fixed Nov 8):
  - `combine_passes_shots_carries()` creates unique `action_id` column
  - Preserved throughout pipeline for reliable merging

**State-Action Space** (`state_action.py`):
- Grid: 7 rows (y-axis, ~9.7m) × 11 columns (x-axis, ~9.5m) = 77 states
- **Actions: 8 total (0-7)** - SIMPLIFIED from 10:
  - 0-5: Pass types (short/long × backward/lateral/forward)
  - 6: shoot
  - 7: carry
- **Action Masking**: Implemented to reduce sparsity
  - Shooting disabled when x < 75m (>30m from goal)
  - Backward passes disabled at col=0 (defensive edge)
  - Forward passes disabled at col=10 (attacking edge)
- **Absorbing states**: Defined and implemented in MDP:
  - State 77: goal (successful shots)
  - State 78: no_goal (failed shots)
  - State 79: loss_possession (failed passes/carries)
- **Coordinate Semantics** (CRITICAL - fixed Nov 8):
  - **Passes/Shots**: `state_from` uses `x_end, y_end` (release/shot location)
  - **Carries**: `state_from` uses `x_start, y_start` (carry start)
  - **Successful passes**: `state_to` uses reception coordinates
  - **Failed actions**: `state_to` = absorbing states (77, 78, or 79)

**MDP Construction** (`pass_decision_analysis.ipynb` Section 5):
- ✅ Built MDPs for all 20 Premier League teams
- ✅ Transition matrices P: (751, 10, 751) - includes 3 absorbing states
- ✅ Reward matrices R: (751, 10) - rewards only for goals
- ✅ Optimal policies π: (748, 10) - computed via value iteration
- ✅ All matrices saved to `data/team_mdps/` (60 .npy files total)

**Position-based xG Model** (`src/xg_model.py`):
- ✅ Geometric model using viewing angle to goal
- ✅ Goal location: x=105m, y=34m (corrected from earlier error)
- ✅ Power law calibration: xG = 0.76 × (angle/36.8°)^1.5
- ✅ Penalty spot (94m, 34m) → 36.8° → 76% xG (realistic)
- ✅ Shooting constraint: xG=0 for x<75m (>30m from goal)
- ✅ Bayesian shrinkage: P_smoothed = (α·P_prior + n·P_empirical)/(α + n)
- ✅ Prior strength α=10 balances sparse data smoothing with empirical preservation

**Notebook Progress** (`pass_decision_analysis.ipynb`):
- **Sections 1-5 fully executed** with all validations
- **Total actions**: 460,373 (up from 276,293)
  - Passes: 267,616 (58.1%)
  - Shots: 8,677 (1.9%, 1,071 goals = 12.34%)
  - Carries: 184,080 (40.0%, 91.6% success)
- **State-action coverage**: 87.4% (6,537 / 7,480 possible pairs)
- **Visualizations**: 3-panel heatmaps (passes in YlOrRd, shots in Reds, carries in Blues)
- Zone analysis reveals tactical patterns (defensive backward 49.7%, attacking forward 50.2%)
- **MDP matrices saved**: 20 teams × 3 files (P, R, π) = 60 .npy files
- **Ready for**: Pass success modeling (Van Roy Method 3), fundamental matrix, policy analysis

---

## 🎯 FINAL PROJECT: Pass Decision Analysis Using MDP Framework

### Project Overview
Applying Markov Decision Process (MDP) modeling to analyze short vs. long pass decisions in soccer, inspired by Van Roy et al.'s "Leaving Goals on the Pitch" paper on shooting decisions. Using Premier League 2024 season event data.

### Research Questions
1. ✅ **ANSWERED (Nov 9)**: In which zones do short progressive passes lead to better goal-scoring chances than long direct balls? → **Section 7 Optimal Action Analysis complete with 3-team comparison**
2. 🎯 **PRIORITY #1 (Nov 10)**: How would altering pass type policies (e.g., 10-20% more long forward passes in specific zones) affect expected goals scored? → **Section 9 Counterfactual Policy Analysis**
3. 🔄 **PRIORITY #2 (Nov 11)**: What is the probability of scoring after sequences of short passes vs. one long pass from midfield? → **Section 8 Sequential Action Analysis**
4. 📊 **PRIORITY #3 (Nov 12)**: What is the quality-quantity trade-off when teams increase pass frequency of certain types? → **Section 10 Quality-Quantity Trade-offs**

### Technical Approach

**MDP Components**:
- **State Space**: Full-field grid discretization (22×34 cells = 748 states) + 3 absorbing states (goal, no_goal, loss_possession)
  - ✅ Implemented in `src/state_action.py` (FieldGrid class)
  - Grid size: 3.09m × 3.09m cells (square cells for uniform discretization)
  - ✅ Absorbing states defined (state indices 748, 749, 750)
- **Action Space**: 10 actions per state (✅ FULLY IMPLEMENTED):
  1. `short_backward` (≤10m, dx < -5m)
  2. `short_lateral` (≤10m, |dx| ≤ 5m)
  3. `short_forward` (≤10m, dx > 5m)
  4. `medium_backward` (10-25m, dx < -5m)
  5. `medium_lateral` (10-25m, |dx| ≤ 5m) - includes rare long_lateral passes
  6. `medium_forward` (10-25m, dx > 5m)
  7. `long_backward` (>25m, dx < -5m)
  8. `long_forward` (>25m, dx > 5m)
  9. `shoot`
  10. `carry` (dribbling with ball) ← ✅ NEW!
- **Action Masking**: ✅ IMPLEMENTED - Reduces sparsity by 8.82%
  - Shooting masked when >30m from goal (x < 75m) → 528 states
  - Backward passes masked at defensive edge (col=0) → 22 states  
  - Forward passes masked at attacking edge (col=33) → 22 states
  - Total: 660 invalid (state, action) pairs removed
- **Transition Function**: ✅ COMPLETE - Learned from event data with Laplace smoothing (α=2)
  - Shape: (751, 10, 751) - 748 field states + 3 absorbing → 751 total
  - Absorbing transitions: successful shots → 748, failed shots → 749, failed passes/carries → 750
  - Built for all 20 Premier League teams
- **Policy**: ✅ COMPLETE - π(a | s) = probability of selecting action a in state s
  - Shape: (748, 10) with action masking applied
  - Learned empirically from observed team behavior
- **Reward Function**: ✅ COMPLETE - R = 1 for goals (state 748), 0 otherwise
  - Shape: (751, 10) reward matrix per team
- **Success Rate Modeling**: Quality-quantity trade-off modeling (Method 3) (TODO: Nov 7-8)

**Analysis Methods**:
1. **Fundamental Matrix Approach**: Compute expected goals under different policies
2. **Probabilistic Model Checking**: Compare action sequences (optional if time permits)
3. **Counterfactual Policy Analysis**: Evaluate "what-if" scenarios

### 🔍 Critical Lessons Learned (Nov 5-6, 2025)

**The Importance of Reading Provider Documentation**:

During debugging on Nov 5-6, we discovered a fundamental misunderstanding of SkillCorner's coordinate system that was causing "backward forward pass" bugs. This taught us that **ALWAYS consult the data provider's documentation FIRST** before making assumptions.

**Key Discoveries from SkillCorner Documentation**:

1. **Coordinate Meaning** (`20250216 - Dynamic Events CSV Specifications.pdf`):
   - `x_start, y_start`: Where player **first touches** ball (possession start)
   - `x_end, y_end`: Where player **releases the pass** (possession end) ← **NOT where player ends up after moving**
   - `player_targeted_x_reception, y_reception`: Where **receiver gets the ball** ← **ONLY for successful passes**
   
2. **What We Got Wrong**:
   - ❌ **Initial assumption**: `(x_start, y_start) → (x_end, y_end)` = pass distance
   - ❌ **Reality**: This is the **passer's movement while dribbling**, not the pass!
   - ✅ **Correct**: `(x_end, y_end) → (player_targeted_x_reception, y_reception)` = actual pass distance
   
3. **Why It Matters**:
   - A player can dribble **forward** 5m, then pass **backward** 20m
   - Our old calculation: `dx = 5 - 0 = 5` → classified as "forward" ❌
   - Correct calculation: `dx = 20 - 5 = 15` → classified as "backward" ✅
   
4. **Coordinate System Differences**:
   - **Event CSV data**: Coordinates are **mirrored per-team** (each team always attacks left→right)
     - From docs: "coordinates are already mirrored in the csv, so that the team is always attacking from left to right"
   - **Tracking JSON data**: Coordinates are **absolute pitch positions** (not per-team normalized)
   - **Implication**: Cannot directly compare event coordinates to tracking coordinates without de-normalization
   
5. **Unsuccessful Passes**:
   - `player_targeted_x_reception` and `y_reception` are **NULL** for unsuccessful passes
   - SkillCorner does NOT provide `pass_range` or `pass_direction` for unsuccessful passes
   - **Solution**: Must infer/estimate target locations for unsuccessful passes to build complete MDP transitions

**Validation Approach That Worked**:
- Event-to-event validation: Compare our calculated distances to SkillCorner's `pass_distance` column
- Result: **0.95-0.99 correlation** (excellent!)
- Tracking data validation failed due to coordinate system mismatch (but not needed)

**Takeaway for Future Work**:
- 📚 **ALWAYS read data provider documentation thoroughly**
- 🔍 **Validate assumptions with actual data before building entire pipelines**
- 🧪 **Test coordinate calculations on small samples first**
- 📊 **Use correlation with provider's pre-computed fields as validation**
- ⚠️ **Don't assume coordinate systems are the same across different data sources**

### Project Timeline (Nov 1-12, 2025)

**Weekend 1: Foundation (Nov 1-2)** ✅ COMPLETE!
- ✅ Saturday Nov 1 (COMPLETED):
  - ✅ Project skeleton creation and setup verification (`Project/` directory structure)
  - ✅ Data exploration and structure understanding (Chapter 2 in notebook)
  - ✅ Define grid discretization (22×34 = 748 states) and pass classification thresholds (10m, 25m, ±5m direction)
  - ✅ Create state encoding (`FieldGrid` in `state_action.py`) and action classification functions (`classify_pass_type` in `data_processing.py`)
  - ✅ Initial data quality validation (zero-distance filtering, coordinate normalization validation)
  - ✅ **BONUS**: Clearance filtering added (`end_type == 'clearance'` removed - 3,733 events)
  - ✅ **BONUS**: Zone-based tactical analysis completed (Section 2.6 in notebook)
  - ✅ Build pass classification pipeline (complete 8-type classification: short/medium/long × forward/lateral/backward)
  - ✅ Extract and classify all passes from Premier League season (267,616 passes after filtering)
  - **Final Dataset**: 378 matches, 267,616 passes, 79.20% overall success rate
  
- ✅ Sunday Nov 2 (COMPLETED - ALL FOUNDATION WORK DONE!):
  - ✅ Fixed shot extraction (lead_to_goal column → 12.34% goal rate, 1,071 goals)
  - ✅ Added carry extraction (184,080 carries, 91.6% success rate)
  - ✅ Implemented 3-way action combination (passes + shots + carries)
  - ✅ Expanded action space to 10 actions (added carry as action 9)
  - ✅ Defined absorbing states (goal, no_goal, loss_possession)
  - ✅ Implemented action availability masking (shooting >30m, edge constraints)
  - ✅ Map all actions to (state_from, action, state_to) tuples (460,373 actions)
  - ✅ Visualize action distributions with 3-panel heatmaps (passes, shots, carries)
  - ✅ State coverage analysis: 87.4% coverage (6,537 / 7,480 pairs observed)
  - ✅ Updated all notebook cells (Chapters 1-4) to include carries
  - ✅ Validated complete pipeline with all 10 actions
  - **Final Encoded Dataset**: 460,373 actions across 748 states with masking applied
  - **Saved**: `data/actions_encoded.parquet` ready for MDP construction

**Week 2: MDP Construction & Validation (Nov 3-8)**
- ✅ Monday Nov 3 (COMPLETED): MDP construction for all 20 teams
  - Built transition matrices P, reward matrices R, optimal policies π
  - Saved all team MDPs to disk (60 .npy files)
  - Visualized Man City spatial policy
  
- ✅ Tuesday Nov 4 (COMPLETED): Sparse data analysis & smoothing comparison
  - Identified sparse shooting data problem
  - Compared smoothing approaches (threshold vs Bayesian)
  - Selected Bayesian shrinkage with α=10
  
- ✅ Wednesday Nov 5 (COMPLETED): Position-based xG model
  - Built geometric xG model using viewing angle
  - Applied Bayesian shrinkage to shooting probabilities
  - Created src/xg_model.py utility file
  - Generated comparison visualizations
  
- ✅ Thursday Nov 6 (COMPLETED): Debugging & validation - Day 1
  - Manual review of Section 5 MDP results
  - Discovered coordinate interpretation bug
  - Validated transition probabilities and state mappings
  - Began investigation of SkillCorner documentation
  
- ✅ Friday Nov 7 (COMPLETED): Debugging & validation - Day 2
  - Fixed pass coordinate calculations (dx/dy now use ball trajectory)
  - Updated `classify_pass_type()` to use correct coordinates
  - Validated fix with event data correlation (0.95-0.99)
  - **Built XGBoost model for predicting pass types in unsuccessful passes**
    - Trained on 272,335 successful passes with known types
    - Features: distance, angle, position, player attributes
    - Predicted types for 56,378 unsuccessful passes
    - Saved predictions to `passes_with_types_complete.parquet`
  - Prepared for full pipeline rebuild

- ✅ Saturday Nov 8 (COMPLETED): MAJOR FIXES - Multiple critical issues resolved
  - Fixed unsuccessful pass prediction merge (memory error from Cartesian product)
  - Updated `classify_pass_type()` to use composite key merging
  - Fixed `add_state_action_encoding()` coordinate semantics
  - Corrected action IDs (shoot=6, carry=7 instead of 8, 9)
  - Added absorbing state logic for failed actions
  - Updated notebook cells for 7×11 grid and 8-action space
  - Ready for complete pipeline rebuild

**🚀 NEXT STEPS (Nov 10-12)** - Final Sprint to Finish Line:

**⚠️ CRITICAL ISSUE DISCOVERED (Nov 10)**: DataFrame confusion in notebook
- **Problem**: The `actions` variable gets overwritten somewhere in the notebook and becomes a `list` instead of `pandas.DataFrame`
- **Symptom**: `TypeError: list indices must be integers or slices, not str` when trying to access columns like `actions['team_id']`
- **Affected cells**: Section 9 counterfactual analysis (all-teams loop)
- **Root cause**: Unknown - needs investigation to find where `actions` is reassigned
- **Workaround attempted**: Using `actions_raw` or `actions_clean` instead
  - `actions_raw`: Has `first_player_possession_in_team_possession` but LACKS `state_from` column → KeyError
  - `actions_clean`: Created with subset of columns, may or may not have needed columns
- **What's needed**: 
  1. Find where/why `actions` becomes a list
  2. Determine which DataFrame actually has BOTH:
     - `first_player_possession_in_team_possession` (for counting possessions)
     - `state_from` (for state distribution)
     - `team_id` (for filtering)
  3. Use correct DataFrame name consistently throughout Section 9
- **Status**: BLOCKING PROGRESS - must be resolved before counterfactual analysis can proceed

**Monday Nov 10 (8-10h)**: 🎯 SECTION 9 - COUNTERFACTUAL POLICY ANALYSIS (PRIORITY #1) - ⚠️ BLOCKED
- **"What if" scenarios**: Modify team policies and compute impact
  - "What if Manchester City shot 20% more often from edge of box?"
  - "What if Liverpool carried 15% less and passed forward instead?"
  - "What if Nottingham increased long forward passes by 20% in midfield?"
- **Implementation**:
  - ⚠️ FIRST: Fix DataFrame variable confusion (see CRITICAL ISSUE above)
  - Create modified policy matrices (π_modified) for each scenario
  - Recompute fundamental matrices under new policies
  - Calculate expected goals difference (E[goals_new] - E[goals_current])
  - Generate spatial heatmaps showing zones where changes have biggest impact
- **Deliverables**: 
  - 3-5 counterfactual scenarios per team
  - Side-by-side policy comparison visualizations
  - Expected goals impact quantification
  - Tactical recommendations based on findings

**Tuesday Nov 11 (8-10h)**: 🔄 SECTION 8 - SEQUENTIAL ACTION ANALYSIS (PRIORITY #2)
- **Research Question**: Do multiple short passes yield higher E[goals] than one long pass?
- **Path comparison**: 
  - State A → State B via one long_forward pass
  - State A → State B via two short_forward passes through intermediate state
  - Compare expected goals for both paths
- **Multi-step analysis**:
  - Compute E[goals | s, sequence] for different action sequences
  - Build "build-up tree" from defensive third to attacking third
  - Compare expected values of different tactical approaches
- **Visualizations**:
  - Sankey diagrams showing flow through states
  - Heatmaps of optimal path choices by starting position
  - Distribution of E[goals] by sequence length
- **Key insight**: When does patient build-up beat direct play?

**Wednesday Nov 12 (10-12h)**: 📊 SECTION 10 - QUALITY-QUANTITY TRADE-OFFS + FINAL POLISH (PRIORITY #3)
- **Morning (6-8h)**: Quality-Quantity Analysis
  - **Van Roy Method 3 implementation**:
    - Rank actions by quality metric (expected threat or goal outcome)
    - Compute quality distributions per (state, action) pair
    - Model: P_success(quantity) = baseline_success × (1 - diminishing_factor × extra_attempts)
  - **Trade-off curves**:
    - Plot success rate vs. action frequency for each action type
    - Identify optimal frequency for each (state, action) pair
    - Compare teams: do high-possession teams show steeper drop-offs?
  - **Analysis**:
    - "If Man City increased long passes by 20%, how much would success rate drop?"
    - Marginal value of additional attempts
    - Identify saturation points where more attempts hurt more than help
  - **Deliverables**: Trade-off curves, marginal value plots, team comparisons

- **Afternoon (4-6h)**: 🎬 FINAL POLISH & COMPLETION
  - **Summary & conclusions**:
    - Write executive summary at top of notebook
    - Key findings section (3-5 main insights)
    - Tactical recommendations for each team analyzed
    - Limitations and future work
    - References and acknowledgments
  
  - **Final review and polish**:
    - Re-run all notebook cells for clean output
    - Verify all visualizations render correctly
    - Check that all sections have clear narrative
    - Proofread markdown cells for typos/clarity
    - Ensure all code is documented
  
  - **Final checks**:
    - ✅ All cells execute without errors
    - ✅ All figures have clear titles and legends
    - ✅ Research questions all answered
    - ✅ Code is clean and documented
    - ✅ Results are reproducible
  
  - **Evening**: ✅ **PROJECT COMPLETE AND READY FOR SUBMISSION!**

**Thursday Nov 13 (BUFFER DAY)**: Final review if needed, otherwise project is done!

### Estimated Total Hours: ~30-36 hours over 3 days = Intense but achievable with focused work!
  
- **Final checks**:
  - ✅ All cells execute without errors
  - ✅ All figures have clear titles and legends
  - ✅ Research questions all answered
  - ✅ Code is clean and documented
  - ✅ Results are reproducible
  
- **Submit by 11:59 PM**: Upload to course platform

### Estimated Total Hours: ~22-28 hours over 4 days = Reasonable pace with buffer time!

### Sections Completed ✅
1. ✅ **Section 1-2**: Setup & Data Loading (Nov 1-2)
2. ✅ **Section 3**: State-Action Space Definition (Nov 2)
3. ✅ **Section 4**: MDP Construction (Nov 3-4)
4. ✅ **Section 5**: Position-based xG Model (Nov 5)
5. ✅ **Section 6**: Data Fixes & Validation (Nov 6-8)
6. ✅ **Section 7**: Optimal Action Selection (Nov 9) ⭐ **MAJOR ACHIEVEMENT!**
   - Computed E[goals | s, a] for all state-action pairs
   - Identified optimal action per state for Manchester City
   - Created beautiful 3-team comparison visualization (City, Liverpool, Nottingham)
   - Validated optimal actions respect physical constraints
   - Fixed action label bug (forward/backward were swapped!)

### Sections Remaining 🎯
7. 🎯 **Section 9**: Counterfactual Policy Analysis (Nov 10) - PRIORITY 1 - ⚠️ BLOCKED
   - **Blocking issue**: DataFrame variable confusion (`actions` becomes a list)
   - **Needed**: Identify correct DataFrame with `team_id`, `first_player_possession_in_team_possession`, AND `state_from`
   - **Goal**: Answer "What if teams changed their pass policies by 10-20%?"
   
8. 🔄 **Section 8**: Sequential Action Analysis (Nov 11) - PRIORITY 2
   - **Goal**: Compare E[goals] for different action sequences (short build-up vs long balls)
   
9. 📊 **Section 10**: Quality-Quantity Trade-offs (Nov 12) - PRIORITY 3
   - **Goal**: Model how success rates change when teams increase pass frequency

10. 🎬 **Section 11**: Summary & Submission (Nov 13)

### Work Summary (Nov 10 Evening)
**What was attempted**:
- Created counterfactual analysis cell for all 20 teams
- Tried to filter `actions` DataFrame by `team_id` 
- Encountered multiple DataFrame-related errors

**Errors encountered**:
1. `TypeError: list indices must be integers or slices, not str` → `actions` is a list, not DataFrame
2. `KeyError: 'state_from'` → `actions_raw` lacks the state encoding column
3. Confusion about which DataFrame has which columns

**What needs to be done next session**:
1. **Investigate the notebook cell-by-cell** to find where `actions` gets reassigned to a list
2. **Check the kernel variables** to see what DataFrames actually exist:
   - `actions` (currently a list - why?)
   - `actions_raw` (has possession columns but no `state_from`)
   - `actions_clean` (subset of columns - check if it has what we need)
3. **Solution options**:
   - Option A: Fix whatever is reassigning `actions` to a list
   - Option B: Use the correct DataFrame name throughout Section 9
   - Option C: Create a new properly-named DataFrame with all needed columns
4. **Once fixed**: Complete the all-teams counterfactual loop and comparison tables

### Key Deliverables
1. **Jupyter Notebook**: Complete analysis pipeline with documented code
2. **MDP Model**: Learned transition probabilities, policies, and success rates
3. **Visualizations**: Heat maps showing optimal actions per field zone
4. **Analysis Results**: 
   - Zone-specific pass recommendations
   - Expected goal impact of policy modifications
   - Quality-quantity trade-off curves
5. **Written Report**: Methodology, findings, tactical insights, limitations

### Data Specifications
- **Source**: PremierLeague_data/2024/dynamic/*.parquet (event data)
- **Scale**: 378 matches, 460,373 total actions after filtering
  - **Passes**: 267,616 (58.1%, 79.20% success rate)
  - **Shots**: 8,677 (1.9%, 12.34% goal rate = 1,071 goals)
  - **Carries**: 184,080 (40.0%, 91.6% success rate, mean distance 8.53m)
- **Filtering Applied**:
  - ✅ Event type: `player_possession` only
  - ✅ Zero-distance passes removed (91,504 ball controls filtered out)
  - ✅ Clearances removed (3,733 `end_type == 'clearance'` filtered out)
  - ✅ Coordinates normalized (all teams attack left→right, 0→105m)
  - ✅ Shots fixed: using `lead_to_goal` column for goal detection
  - ✅ Carries extracted: using `carry=True` flag with `end_type` success indicator
- **Pass Type Distribution** (after filtering):
  - short_lateral: 73.46% (196,580 passes, 81.76% success)
  - medium_forward: 6.14% (16,422 passes, 71.01% success)
  - medium_backward: 5.94% (15,890 passes, 71.10% success)
  - short_forward: 5.06% (13,536 passes, 75.24% success)
  - short_backward: 4.97% (13,305 passes, 74.63% success)
  - medium_lateral: 2.20% (5,894 passes, 77.94% success)
  - long_forward: 1.13% (3,034 passes, 60.15% success)
  - long_backward: 1.10% (2,955 passes, 58.88% success)
- **Action Distribution** (all 10 actions):
  - Action 1 (short_lateral): 42.70% (196,580 actions)
  - Action 9 (carry): 39.98% (184,080 actions)
  - Action 5 (medium_forward): 3.57% (16,422 actions)
  - Action 3 (medium_backward): 3.45% (15,890 actions)
  - Action 2 (short_forward): 2.94% (13,536 actions)
  - Action 0 (short_backward): 2.89% (13,305 actions)
  - Action 8 (shoot): 1.88% (8,677 actions)
  - Action 4 (medium_lateral): 1.28% (5,894 actions)
  - Action 7 (long_forward): 0.66% (3,034 actions)
  - Action 6 (long_backward): 0.64% (2,955 actions)
- **State-Action Coverage**: 
  - 6,537 unique (state, action) pairs observed
  - Theoretical maximum: 748 states × 10 actions = 7,480 pairs
  - Coverage: 87.4% (excellent for MDP construction)
  - After masking: 6,820 valid pairs (660 masked as impossible)
  - Mean observations per pair: 70.4 (well above sparsity threshold)
  - Well-covered pairs (≥10 obs): 4,339 (66.4%)
- **Data Sparsity**: Significantly reduced by action masking (8.82% reduction in state-action space)
- **Key Insight**: Carries are nearly as common as passes (40% vs 58%), making them essential for realistic modeling
- **Tactical Insights**: Zone context matters more than direction (backward passes: 49.7% success in defensive third vs 79.2% in attacking third)

### Success Criteria
1. ✅ Complete MDP learned from Premier League data
2. ✅ Quality-quantity trade-off model implemented
3. ✅ Actionable tactical insights generated (zone-specific recommendations)
4. ✅ Counterfactual analysis showing impact of policy changes
5. ✅ Clear visualizations and interpretable results
6. ✅ Written documentation of methodology and findings

### Reference Materials
- **Primary Paper**: Van Roy et al. (2020) "Leaving Goals on the Pitch: Evaluating Decision Making in Soccer"
- **Course Modules**: Module 4 (Possession Values), Module 2 (Clustering/Visualization)
- **Tools**: pandas, numpy, matplotlib, mplsoccer, scipy (fundamental matrix)

---

## 🛠️ Project-Specific Guidelines

### When Working on Final Project

1. **Code Organization**:
   - Use clear section headers in notebook
   - Separate data processing, MDP construction, and analysis
   - Document all assumptions and design choices
   - Include sanity checks and validation steps

2. **Data Processing**:
   - Validate pass classifications (visual spot-checks)
   - Check for edge cases (passes near boundaries, very short/long passes)
   - Handle missing data gracefully
   - Use vectorized operations for efficiency

3. **MDP Construction**:
   - Verify probabilities sum to 1 for each (state, action)
   - Check for numerical stability in matrix operations
   - Use Laplace smoothing (α=1 or 2) consistently
   - Validate transition matrix is well-conditioned

4. **Success Rate Modeling**:
   - Rank passes by quality metric (xT added, or goal-outcome)
   - Compute quality distributions per (state, action)
   - Apply adjustments when policy changes frequency
   - Document assumptions clearly

5. **Analysis & Interpretation**:
   - Connect findings to tactical concepts
   - Compare results to reference paper where applicable
   - Acknowledge limitations and data constraints
   - Provide actionable insights for coaches/analysts

6. **Visualization**:
   - Use mplsoccer for pitch plots
   - Consistent color schemes (diverging for comparisons)
   - Clear legends and annotations
   - Heat maps for spatial patterns

### Football Analytics Context for Project
- **Pass Types**: Short build-up vs. long direct play distinction is tactically meaningful
- **Direction**: Forward passes are risky but progressive; backward passes are safe but not advancing
- **Zones**: Defensive third = safety priority, attacking third = risk-taking acceptable
- **Expected Goals**: Ultimate success metric - all analysis ties back to goal probability

---

**Last Updated**: November 1, 2025  
**Current Focus**: Final Project - Markov Decision Process Analysis of Pass Decision Making in Soccer  
**Hardware Status**: ✅ RAM Upgraded - No Memory Constraints

## Libraries to Favor
- pandas, numpy for data manipulation
- matplotlib, seaborn, plotly for visualization
- scikit-learn for traditional ML
- PyTorch or TensorFlow for deep learning
- polars for high-performance data processing (when appropriate)

## What to Avoid
- Avoid overly complex solutions when simple ones suffice
- Don't use deprecated pandas methods (e.g., append, use concat instead)
- Avoid hardcoded paths; use pathlib or os.path
- Don't ignore data validation and sanity checks
