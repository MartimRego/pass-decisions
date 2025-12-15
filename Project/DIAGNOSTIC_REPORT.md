# 🔍 COMPREHENSIVE NOTEBOOK DIAGNOSTIC REPORT
**Date**: November 25, 2025
**Issue**: Near-zero expected goals & broken visualizations

## 🔴 ROOT CAUSE IDENTIFIED

### **MDP matrices were built with OLD 7×11 grid, but notebook expects 22×34 grid**

### Evidence:
1. **Saved MDP shapes** (from data/team_mdps/):
   - `P_40_Manchester_City.npy`: shape (80, 8, 80)
   - `pi_40_Manchester_City.npy`: shape (77, 8)  
   - `R_40_Manchester_City.npy`: shape (80,)
   - **Interpretation**: 77 field states (7×11) + 3 absorbing = 80 total

2. **Grid definition** (src/state_action.py):
   ```python
   class FieldGrid:
       def __init__(self, n_rows: int = 22, n_cols: int = 34, ...):
   ```
   - Expected: 22×34 = 748 field states + 3 absorbing = 751 total

3. **Dimension mismatch**:
   - Expected total states: 751
   - Actual total states: 80
   - **Ratio**: 751/80 = 9.39× too small!

## 💥 CASCADING FAILURES

### 1. Expected Goals Calculations
**Symptom**: `EXPECTED GOALS PER POSSESSION: 0.0000`

**Cause**:
- Fundamental matrix N computed over 77×77 states
- Most possessions start in states 0-77 of the NEW grid (22×34 indexing)
- But in OLD grid terms, these map to completely different field locations
- State-to-coordinates conversion is broken

**Example**:
```python
# NEW grid: state 54 → row=54//34=1, col=54%34=20 → (x=61.8m, y=3.1m)
# OLD grid: state 54 → row=54//11=4, col=54%11=10 → (x=95.5m, y=24.7m)
```
Completely different locations!

### 2. Visualization Corruption
**Symptom**: Heatmap shows 2 tiny red boxes in bottom-right corner

**Cause**:
```python
eg_grid = expected_goals_per_state.reshape(grid.n_rows, grid.n_cols)
# Tries to reshape 77-element array into (22, 34) = 748 elements
# → IndexError or partial fill with zeros
```

**Fix needed**: Cannot visualize 77-state data on 748-state grid!

### 3. Policy Analysis Broken
**Symptom**: Counterfactual analysis shows ~0.00 goal impacts

**Cause**:
- Policy modifications apply to 77-state space
- But possession distributions computed from 748-state coordinates
- Zone mappings (e.g., "midfield columns 6-9") refer to different field areas in each grid

### 4. State Encoding Mismatch
**Actions DataFrame** (`actions_encoded.parquet`):
- `state_from` and `state_to` encoded using **22×34 grid**
- Values range 0-747 for field states

**MDP Matrices**:
- Built expecting states 0-76 (7×11 grid)
- Transitions for states 77-747 completely missing!

## ✅ REQUIRED FIXES

### **PRIMARY FIX: Regenerate ALL MDP matrices with 22×34 grid**

#### Step 1: Verify grid configuration in MDP generation script
```python
# In the script that builds team_mdps/P_*.npy, R_*.npy, pi_*.npy:
grid = FieldGrid(n_rows=22, n_cols=34)  # MUST be this!
```

#### Step 2: Rebuild actions encoding (if needed)
Check if `actions_encoded.parquet` was built with 22×34 grid:
```python
actions = pd.read_parquet('data/actions_encoded.parquet')
print(f"Max state_from: {actions['state_from'].max()}")  # Should be <748
print(f"Max state_to: {actions['state_to'].max()}")      # Should be <748
```

If not 22×34, rerun state encoding.

#### Step 3: Regenerate MDP matrices
Run the MDP construction script (Section 4 cells or dedicated script):
```bash
# Expected output shapes:
# P: (751, 8, 751) - 748 field + 3 absorbing, 8 actions
# R: (751,) - reward vector
# pi: (748, 8) - policy over field states only
```

#### Step 4: Validate new MDPs
```python
P = np.load('data/team_mdps/P_40_Manchester_City.npy')
assert P.shape == (751, 8, 751), f"Wrong shape: {P.shape}"

R = np.load('data/team_mdps/R_40_Manchester_City.npy')
assert R.shape == (751,), f"Wrong shape: {R.shape}"
assert R[748] == 1.0, "Goal state reward should be 1.0"

pi = np.load('data/team_mdps/pi_40_Manchester_City.npy')
assert pi.shape == (748, 8), f"Wrong shape: {pi.shape}"
```

## 🔧 SECONDARY FIXES (After MDP regeneration)

### Fix 1: Update transient_limit usage
Many cells compute:
```python
transient_limit = N_city.shape[0]  # Will be 748 after fix
```
This is correct pattern - keep using it!

### Fix 2: Visualization extent and reshaping
```python
# CORRECT (after MDP regeneration):
eg_grid = expected_goals_per_state.reshape(grid.n_rows, grid.n_cols)
# Will reshape 748-element array into (22, 34) - perfect match!

# Heatmap extent
extent = [0, 105, 0, 68]  # Correct for pitch dimensions
ax.imshow(eg_grid, extent=extent, origin='lower', ...)
```

### Fix 3: Zone definitions in counterfactual analysis
```python
# OLD midfield: columns 6-9 of 11 total
# → x-range: [57.3m, 85.9m] in 7×11 grid

# NEW midfield: columns ? of 34 total for same x-range
# → Need to recalculate column indices for 22×34 grid
```

### Fix 4: Absorbing state indices
```python
# Dynamic computation (already correct in notebook):
goal_state = R.shape[0] - 3  # Will be 748 after regeneration
no_goal_state = R.shape[0] - 2  # Will be 749
loss_state = R.shape[0] - 1  # Will be 750
```

## 📊 EXPECTED RESULTS AFTER FIX

### Before (Current):
- Expected goals per possession: **0.0000**
- Heatmap: Blank with 2 tiny boxes
- Counterfactual impacts: **±0.00 to ±0.01 goals**

### After (With correct MDPs):
- Expected goals per possession: **0.05 - 0.15** (realistic for football)
- Heatmap: Full pitch coverage, gradient from defensive (low) to attacking (high)
- Counterfactual impacts: **±0.5 to ±5.0 goals** per season (meaningful)

## 🚀 ACTION PLAN

1. **[CRITICAL]** Locate MDP generation code (Section 4 or standalone script)
2. **[CRITICAL]** Verify it uses `FieldGrid(n_rows=22, n_cols=34)`
3. **[CRITICAL]** Regenerate all 20 team MDPs
4. **[HIGH]** Validate new MDP shapes and values
5. **[MEDIUM]** Re-run notebook Sections 5-10 with new MDPs
6. **[LOW]** Update zone definitions for 22×34 grid

## ⏱️ ESTIMATED FIX TIME

- Regenerate MDPs: **10-20 minutes** (depending on data size)
- Validate + re-run notebook: **5-10 minutes**
- **Total**: ~30 minutes

## ❓ OPEN QUESTIONS

1. Why were MDPs generated with 7×11 grid if code shows 22×34 defaults?
   - Possible: MDPs were generated before grid size change
   - Possible: Generation script has hardcoded 7×11

2. Was `actions_encoded.parquet` regenerated with 22×34 grid?
   - Need to check max state values

3. Are there cached intermediate results that also need regeneration?
   - Check for any .npy/.parquet files with state-dependent data

---
**Recommendation**: Regenerate MDPs immediately. This is a blocking issue for all downstream analysis.
