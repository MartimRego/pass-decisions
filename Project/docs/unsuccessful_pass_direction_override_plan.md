# Unsuccessful Pass Direction Override + XGBoost Retraining (Implementation Handoff)

## Goal
Improve *unsuccessful pass type* labeling by using inferred opponent regain location to override **direction only** (no length override), while keeping the current XGBoost multiclass model as the fallback.

This supports downstream work (e.g., risk-aware MDP turnover penalties) without expanding state/action spaces.

## Non-negotiable constraints (decisions)
- **Pass-type schema remains unchanged.**
  - Keep the existing 6-class directional/length label space (e.g., `short_backward`, `short_lateral`, `short_forward`, `long_backward`, `long_lateral`, `long_forward`).
  - We do **not** add new classes.
- **Fallback to XGBoost only** when the direction override cannot be applied.
  - No partial overrides other than direction.
- `MAX_GAP_FRAMES`:
  - Must be **default 50**.
  - Must be **configurable via CLI**.
- Prefer **`x_end_norm/y_end_norm`** as the pass release / endpoint for direction calculations.
- **Offside counts as a failed pass**.

## Where this lives in the repo
Primary artifacts discovered:
- Dataset creation / export:
  - [twelve-deep-learning/Project/1_create_complete_dataset.py](twelve-deep-learning/Project/1_create_complete_dataset.py)
  - [twelve-deep-learning/Project/2_create_complete_dataset.ipynb](twelve-deep-learning/Project/2_create_complete_dataset.ipynb)
- Pass classifier training:
  - [twelve-deep-learning/Project/train_pass_classifier.ipynb](twelve-deep-learning/Project/train_pass_classifier.ipynb)
- Model artifacts location:
  - [PremierLeague_data/2024/pass_prediction_models/](PremierLeague_data/2024/pass_prediction_models/)
- Prior data forensics and prototype logic:
  - [twelve-deep-learning/Project/data_testing.ipynb](twelve-deep-learning/Project/data_testing.ipynb)
  - [twelve-deep-learning/Project/docs/risk_data_findings.md](twelve-deep-learning/Project/docs/risk_data_findings.md)

## Required data columns
From the processed tables used in `data_testing.ipynb`:
- Identity / ordering:
  - `match_id`, `period`, `frame_start`, `frame_end`, `team_id`
  - Stable join key is typically `(match_id, event_id, action_id)`
- Pass outcome:
  - `pass_outcome` (must allow detecting `offside`)
- Normalized coordinates:
  - `x_end_norm`, `y_end_norm` (pass endpoint / release point for this plan)
  - `x_start_norm`, `y_start_norm` (used for context / debugging)
- Success flag:
  - `success` (must treat `offside` as `0`)

New/derived features to compute:
- `team_switch_next` (boolean)
- `opp_regain_x_norm`, `opp_regain_y_norm`
- `opp_regain_gap_frames`

## Definitions
### Unsuccessful pass
A pass is considered unsuccessful when `success == 0`.

Important: ensure `pass_outcome == "offside"` is mapped to `success == 0`.

### Opponent regain (proxy)
For each action row `i`, within the same `(match_id, period)`, sort by time (`frame_start`, `frame_end`) and define:
- `next_team_id` = team_id of the next row in that sequence
- `team_switch_next = next_team_id != team_id`

When `team_switch_next` is true, define opponent regain proxy:
- `opp_regain_x_norm = next_row.x_start_norm`
- `opp_regain_y_norm = next_row.y_start_norm`
- `opp_regain_gap_frames = next_row.frame_start - i.frame_end`

This is the exact approach prototyped in [twelve-deep-learning/Project/data_testing.ipynb](twelve-deep-learning/Project/data_testing.ipynb).

## Direction override logic (core)
### When to apply override
Apply **direction override** for a pass if all are true:
- `success == 0` (includes offside)
- `team_switch_next == True`
- `opp_regain_gap_frames` is not null
- `opp_regain_gap_frames <= MAX_GAP_FRAMES` (default 50, CLI-configurable)
- `x_end_norm/y_end_norm` and `opp_regain_x_norm/opp_regain_y_norm` are present

Otherwise: **do not override** and rely on XGBoost prediction.

### How to compute direction
Compute the direction vector from pass endpoint to opponent regain:
- `dx = opp_regain_x_norm - x_end_norm`
- `dy = opp_regain_y_norm - y_end_norm`

Convert to angle and bucket into the existing direction scheme used by the project.
- Use the existing direction bucket function already used for pass labeling.
- Only replace the **direction component** of the predicted/assigned pass type.

### What remains unchanged
- **Length classification remains untouched.**
  - Keep the length component as predicted by XGBoost.
- If the project stores pass type as a single categorical label, implement the override by:
  1) mapping label → (length, direction)
  2) swapping direction
  3) mapping back to the same label namespace

## XGBoost retraining plan (add regain-derived features)
Retraining is required because the new regain-derived features are highly informative for failed-pass labeling.

### Training set
- Train on pass events.
- Target remains the current 6-class pass type label.
- Unsuccessful passes must include offside.

### Add features (minimal set)
Add features derived from opponent regain proxy:
- `opp_regain_gap_frames` (numeric)
- `has_opp_regain_within_gap` (boolean: `opp_regain_gap_frames <= MAX_GAP_FRAMES`)
- `dx_to_regain`, `dy_to_regain`
- `dist_to_regain = sqrt(dx^2 + dy^2)`
- `angle_to_regain = atan2(dy, dx)`

Notes:
- These features are available only when `team_switch_next` is true; otherwise fill with sentinel values (and include a boolean missingness flag) so the model can learn the missingness pattern.

### Outputs
Continue writing model artifacts to:
- [PremierLeague_data/2024/pass_prediction_models/](PremierLeague_data/2024/pass_prediction_models/)

### Evaluation focus
- Primary metric: confusion among directional classes on unsuccessful passes.
- Sanity check: for cases where override applies, direction accuracy should improve versus baseline.

## CLI configurability requirement
`MAX_GAP_FRAMES` must be set via CLI (default 50).

Implementation expectation:
- Add a CLI flag in [twelve-deep-learning/Project/1_create_complete_dataset.py](twelve-deep-learning/Project/1_create_complete_dataset.py), e.g.:
  - `--max-gap-frames 50`
- Ensure both:
  - dataset creation (feature computation / override application)
  - and any re-run of inference
  use the same value.

## Implementation checklist (fresh-chat ready)
1. Confirm where `success` is derived and ensure `pass_outcome == "offside"` implies `success = 0`.
2. Add CLI arg `--max-gap-frames` with default 50.
3. In dataset creation, compute `team_switch_next` and `opp_regain_*` features per action within `(match_id, period)`.
4. In the pass-type assignment step:
   - If override conditions hold, recompute direction from `(x_end_norm, y_end_norm) → (opp_regain_x_norm, opp_regain_y_norm)`.
   - Keep predicted length and recompose the existing label.
   - Else fall back to XGBoost label.
5. Retrain model in [twelve-deep-learning/Project/train_pass_classifier.ipynb](twelve-deep-learning/Project/train_pass_classifier.ipynb) with regain-derived features.
6. Regenerate outputs/parquets and verify:
   - No schema changes.
   - Override applies only to the intended subset.

## Notes / known limitations
- Opponent regain is a proxy (next opponent player-possession). It is reliable for immediate turnovers but will miss cases where intervening events exist or where the dataset omits restart-taker actions.
- The override is intentionally conservative (bounded by `MAX_GAP_FRAMES`) to avoid “teleport” direction artifacts after stoppages.
