# SkillCorner/Twelve data testing findings (turnovers & restarts)

Date: 2025-12-23

This note documents what we learned from inspecting the processed SkillCorner Dynamic Events tables in this repo and running tests in [twelve-deep-learning/Project/data_testing.ipynb](twelve-deep-learning/Project/data_testing.ipynb).

## What tables exist and what they contain

### `PremierLeague_data/2024/processed/actions_complete.parquet`
- Size: ~521k rows, ~315 columns.
- Granularity: **player_possession** events only (`event_type == 'player_possession'`).
- Contains:
  - `match_id`, `period`, `frame_start`, `frame_end`
  - `team_id`, `player_id`
  - `end_type` (how this player possession ended): observed values include `pass`, `shot`, `possession_loss`, `foul_suffered`, `clearance`, `unknown`.
  - For passes: `pass_outcome` (e.g., `successful`, `unsuccessful`, `offside`).
  - Phase model outputs: `team_in_possession_phase_type` (includes `set_play`, `build_up`, `create`, `finish`, `transition`, `direct`, `quick_break`, `chaotic`, `disruption`).
  - Coordinates: `x_start/y_start/x_end/y_end` (SkillCorner meters in the attacking-side frame).

Important limitation:
- This table **does not provide explicit restart types** (no `corner`, `throw_in`, `goal_kick`, `free_kick` labels as event types/subtypes). The restart taker action is often not a `player_possession` event.

### `PremierLeague_data/2024/processed/actions_clean.parquet`
- Size: same number of rows as actions_complete, but only ~24 columns.
- Contains the project’s derived fields used for the MDP pipeline:
  - `success` (1/0)
  - normalized coordinates `x_*_norm`, `y_*_norm`
  - `action_type`, `pass_type`, etc.

Key discovery:
- `(match_id, event_id)` is **not globally unique** across the season data. In tests, joining on it caused a many-to-many merge.
- `(match_id, event_id, action_id)` is the stable join key.

## 1) Where does the opponent regain the ball?

### Best available proxy in these tables
For an event row `i` by team A (a player possession), the **opponent regain** is approximated by:
- Find the *next* player_possession event within the same `(match_id, period)` with `team_id != team_id_i`.
- Use that next row’s `x_start_norm`, `y_start_norm` as the opponent’s regain location.

This produces:
- `opp_regain_x_norm`, `opp_regain_y_norm`
- `opp_regain_gap_frames = next_frame_start - frame_end_i`

Observed on a small sample (3 matches):
- Median regain gap was ~25 frames, with heavy tail (90% ~344 frames; 95% ~578; 99% ~977).
- Opponent regain phase types were mostly `chaotic/create/build_up/finish`, with some `set_play`.

Interpretation:
- Small gaps look like **in-play turnovers**.
- Large gaps likely indicate **stoppages** (ball out, fouls, set pieces) between possessions.

## 2) Turnover vs restart, and who gets the restart?

### What we can do reliably
We can reliably identify:
- Whether possession switches to the opponent immediately: `team_switch_next`.
- The team that gets the next possession start (proxy for restart owner): `restart_team_id = next_team_id`.

### What we cannot do explicitly from these tables
Because only `player_possession` rows are present:
- We generally cannot directly label the restart as `throw_in` vs `goal_kick` vs `corner`.
- Spatial signatures near touchline/goal line were *not* visible on sample restart-like possessions (near-touchline and near-goal-line rates were ~0). This strongly suggests the **restart execution is not represented** as a `player_possession` start.

### Practical heuristic for “restart-like” vs “turnover-like”
Given the limitation above, the notebook implements a **restart-likeness heuristic** for failures where possession switches:
- `restart_like_gap = opp_regain_gap_frames >= RESTART_GAP_FRAMES` (tested 25, 50, 100, 200, 300, 500)
- `next_is_set_play = (opp_regain_phase_type == 'set_play')`
- `restart_like = restart_like_gap OR next_is_set_play`

Notes:
- `set_play` catches corners/free-kicks/long throws in the phase model, but not necessarily goal-kicks.
- Time gap catches stoppages broadly.

On a small sample (3 matches), for failures that immediately switch possession:
- `restart_like` rate (gap>=200 OR next_is_set_play) was ~0.24.

### Recommended usage for the risk-aware MDP
For the “negative reward = opponent value at regain state” approach:
- Use the opponent’s next possession start state (`opp_regain_*_norm`) **regardless of turnover vs restart**.
- Keep `opp_regain_gap_frames` and `opp_regain_phase_type` as features/diagnostics:
  - you may later decide to treat large-gap events differently (e.g., downweight, or separate models).

If you need restart type (throw/corner/goal kick):
- You likely need to integrate a different table that includes explicit set-piece/restart events (e.g., Wyscout event types) or a separate feed.
- Within this current player-possession-only table, restart type is not consistently recoverable.

## Implementation notes (notebook fixes)

During debugging, we updated the notebook to:
- Locate the repo root dynamically (so relative paths work from VS Code).
- Join `actions_complete` and `actions_clean` using `(match_id, event_id, action_id)`.

See notebook for the exact code paths:
- [twelve-deep-learning/Project/data_testing.ipynb](twelve-deep-learning/Project/data_testing.ipynb)

## Next steps

1. Build `V_lg(s)` (league-average scoring value) from pooled data.
2. Estimate `p(opp_state | s,a,loss)` by mapping each failure to `opp_regain_state`.
3. Compute turnover cost `c(s,a) = E[V_lg(opp_regain_state) | s,a,loss]`.
4. Incorporate into reward as `-λ P(loss|s,a) c(s,a)` (no state-space blow-up).
