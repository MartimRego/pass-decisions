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

**Last Updated**: November 1, 2025 (End of Day)  
**Current Focus**: Final Project - Markov Decision Process Analysis of Pass Decision Making in Soccer  
**Hardware Status**: ✅ RAM Upgraded - No Memory Constraints  
**Day 1 Progress**: ✅ All Nov 1 tasks completed + clearance filtering + zone analysis (ahead of schedule!)

---

## 📁 Project File Structure (Nov 1, 2025)

```
Project/
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # ✅ COMPLETE: Load, filter, classify passes (631 lines)
│   │   ├── load_premier_league_events()
│   │   ├── extract_pass_events()  # Filters player_possession + clearances
│   │   ├── rescale_coordinates()  # SkillCorner → FIFA (0-105m × 0-68m)
│   │   ├── normalize_attack_direction()  # All teams attack left→right
│   │   ├── classify_pass_length()  # short ≤10m, medium 10-25m, long >25m
│   │   ├── classify_pass_direction()  # forward dx>5, backward dx<-5, lateral |dx|≤5
│   │   └── classify_pass_type()  # 8 pass types + filtering
│   │
│   └── state_action.py         # ✅ COMPLETE: MDP state/action space (304 lines)
│       ├── ACTION_NAMES  # Dict mapping 0-8 to action names
│       └── FieldGrid     # 22×34 grid discretization class
│
├── pass_decision_analysis.ipynb  # ✅ CHAPTER 2 COMPLETE (8 sections executed)
│   ├── Section 1: Imports (with module reloading)
│   ├── Section 2: Data Exploration & Validation
│   │   ├── 2.1: Load events (1.8M events → 359K passes)
│   │   ├── 2.2: Extract passes (filters clearances)
│   │   ├── 2.3: Coordinate transformation
│   │   ├── 2.4: Pass classification (8 types)
│   │   ├── 2.5: Statistics (79.20% success, distributions)
│   │   ├── 2.6: Zone analysis (defensive/middle/attacking breakdown)
│   │   └── 2.7: Final summary (267,616 passes ready)
│   │
│   └── Section 3+: State-Action Encoding (TODO: Nov 2)
│
├── diagnostic_scripts/  # Validation scripts created during development
│   ├── test_clearance_filter.py
│   ├── analyze_pass_context.py
│   ├── diagnose_pass_direction.py
│   └── investigate_clearances.py
│
└── README.md  # Project overview

```

### Key Implementation Details

**Data Processing Pipeline** (`data_processing.py`):
- Event extraction: Filters `event_type == 'player_possession'` (matches Module 3 methodology)
- Clearance removal: Filters `end_type == 'clearance'` (3,733 removed)
- Zero-distance filtering: Removes passes with distance ≤ 0.01m (91,504 removed)
- Coordinate system: SkillCorner (-52 to 52, -34 to 34) → FIFA (0-105m × 0-68m)
- Normalization: Flips coordinates so all teams attack left→right
- Classification thresholds validated through diagnostic analysis

**State-Action Space** (`state_action.py`):
- Grid: 22 columns (x-axis, 4.77m) × 34 rows (y-axis, 2.00m) = 748 states
- Actions: 0=short_backward, 1=short_lateral, ..., 8=shoot
- Absorbing states planned: goal, no_goal, loss_possession (not yet implemented)

**Notebook Progress** (`pass_decision_analysis.ipynb`):
- Chapter 2 fully executed with all validations
- Zone analysis reveals tactical patterns (defensive backward 49.7%, attacking forward 50.2%)
- Ready for state-action encoding (Chapter 3)

---

## 🎯 FINAL PROJECT: Pass Decision Analysis Using MDP Framework

### Project Overview
Applying Markov Decision Process (MDP) modeling to analyze short vs. long pass decisions in soccer, inspired by Van Roy et al.'s "Leaving Goals on the Pitch" paper on shooting decisions. Using Premier League 2024 season event data.

### Research Questions
1. In which zones do short progressive passes lead to better goal-scoring chances than long direct balls?
2. What is the probability of scoring after sequences of short passes vs. one long pass from midfield?
3. How would altering pass type policies (e.g., 10-20% more long forward passes in specific zones) affect expected goals scored?
4. What is the quality-quantity trade-off when teams increase pass frequency of certain types?

### Technical Approach

**MDP Components**:
- **State Space**: Full-field grid discretization (22×34 cells = 748 states) + 3 absorbing states (goal, no_goal, loss_possession)
  - ✅ Implemented in `src/state_action.py` (FieldGrid class)
  - Grid size: 4.77m × 2.00m cells
- **Action Space**: 9 actions per state (✅ IMPLEMENTED):
  1. `short_backward` (≤10m, dx < -5m)
  2. `short_lateral` (≤10m, |dx| ≤ 5m)
  3. `short_forward` (≤10m, dx > 5m)
  4. `medium_backward` (10-25m, dx < -5m)
  5. `medium_lateral` (10-25m, |dx| ≤ 5m) - includes rare long_lateral passes
  6. `medium_forward` (10-25m, dx > 5m)
  7. `long_backward` (>25m, dx < -5m)
  8. `long_forward` (>25m, dx > 5m)
  9. `shoot`
- **Transition Function**: P(s, a, s') learned from event data with Laplace smoothing (TODO: Nov 3)
- **Policy**: π(a | s) = probability of selecting action a in state s (TODO: Nov 4)
- **Reward Function**: R = 1 for goals, 0 otherwise (TODO: Nov 5)
- **Success Rate Modeling**: Quality-quantity trade-off modeling (Method 3) (TODO: Nov 6)

**Analysis Methods**:
1. **Fundamental Matrix Approach**: Compute expected goals under different policies
2. **Probabilistic Model Checking**: Compare action sequences (optional if time permits)
3. **Counterfactual Policy Analysis**: Evaluate "what-if" scenarios

### Project Timeline (Nov 1-12, 2025)

**Weekend 1: Foundation (Nov 1-2)**
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
  
- Sunday Nov 2 (8-10h):
  - Map passes to (state_from, action, state_to) tuples
  - Visualize pass distributions and validate classifications
  - State coverage analysis and sparsity checks

**Week 2: MDP Construction (Nov 3-7)**
- Monday Nov 3 (2-3h): Transition probability estimation with Laplace smoothing
- Tuesday Nov 4 (2-3h): Policy estimation (action selection probabilities per state)
- Wednesday Nov 5 (2-3h): Position-based xG model for shooting
- Thursday Nov 6 (2-3h): Pass success rate modeling with quality distribution analysis
- Friday Nov 7 (2-3h): Fundamental matrix computation and validation

**Weekend 2: Analysis & Policy Experiments (Nov 8-9)**
- Saturday Nov 8 (8-10h):
  - Implement quality-quantity trade-off modeling for pass success rates
  - "What should players do?" analysis: optimal action per zone
  - Generate heat maps showing optimal pass types
  - Compare immediate shooting vs. different pass sequences
  
- Sunday Nov 9 (8-10h):
  - Counterfactual policy analysis: "What if?" scenarios
  - Modify policies (increase/decrease specific pass types in zones)
  - Compute expected goals under altered policies
  - Identify strategic insights and tactical recommendations

**Final Push (Nov 10-12)**
- Monday Nov 10 (3-4h):
  - Visualization refinement and tactical interpretation
  - Create compelling figures for presentation
  - Draft findings and insights
  
- Tuesday Nov 11 (4-6h):
  - Complete write-up with methodology, results, discussion
  - Final validation and code cleanup
  
- Wednesday Nov 12 (2-4h):
  - Prepare submission materials
  - Final review and polish
  - ✅ PROJECT READY FOR SUBMISSION

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
- **Scale**: 378 matches, 267,616 passes after filtering (79.20% success rate)
- **Filtering Applied**:
  - ✅ Event type: `player_possession` only
  - ✅ Zero-distance passes removed (91,504 ball controls filtered out)
  - ✅ Clearances removed (3,733 `end_type == 'clearance'` filtered out)
  - ✅ Coordinates normalized (all teams attack left→right, 0→105m)
- **Pass Type Distribution** (after filtering):
  - short_lateral: 73.46% (196,580 passes, 81.76% success)
  - medium_forward: 6.14% (16,422 passes, 71.01% success)
  - medium_backward: 5.94% (15,890 passes, 71.10% success)
  - short_forward: 5.06% (13,536 passes, 75.24% success)
  - short_backward: 4.97% (13,305 passes, 74.63% success)
  - medium_lateral: 2.20% (5,894 passes, 77.94% success)
  - long_forward: 1.13% (3,034 passes, 60.15% success)
  - long_backward: 1.10% (2,955 passes, 58.88% success)
- **Data Sparsity**: 748 states × 9 actions = 6,732 pairs → ~40 observations per pair (good coverage expected)
- **Key Insight**: Zone context matters more than direction (backward passes: 49.7% success in defensive third vs 79.2% in attacking third)

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
