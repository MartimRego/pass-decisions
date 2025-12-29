# Risk Incorporation in the MDP (Implementation Handoff)

## Goal
Extend the current discrete MDP (which maximizes goals-for) to also account for **conceding risk after losing possession**, without exploding the state space.

We do this by adding a **transition-based penalty** on possession-loss outcomes that uses an **opponent continuation value** (league-average opponent model).

## Non-negotiable constraints (decisions)
- **Do not expand the state space** with “risk states per location”.
- Keep the existing discrete grid MDP and the exact evaluation approach (absorbing Markov chain / fundamental matrix).
- Use a **league-average opponent model** (not team-specific) for the opponent continuation value.
- We are **not modeling restarts separately** (treat as part of the same turnover/regain proxy; use a conservative time-gap gate if needed).

## What exists today (baseline)
- States: discrete ball-location grid (e.g., 22×34) + absorbing terminal states (goal, turnover/loss).
- Actions: discrete football actions (e.g., pass variants, carry, shoot).
- Transition probabilities: estimated from data under a behavior policy, then used to evaluate or improve a policy.
- Reward: primarily goals-for (and/or other terminal scoring signals).

## Key idea: risk as a transition reward (no state blow-up)
Instead of creating new states, add an **immediate penalty on transitions that lose possession**, proportional to the opponent’s expected value from the regain state.

### Notation
- $s$ = current state (grid cell)
- $a$ = action
- $s'$ = next state
- $V_{\text{opp}}(s)$ = opponent continuation value when the opponent has possession starting from state $s$ (league-average)

### Risk-aware immediate reward
Define a shaped reward on each transition:

$$
R(s,a,s') = R_{\text{for}}(s,a,s') - \lambda \cdot \mathbf{1}[\text{possession lost on this transition}] \cdot V_{\text{opp}}(s_{\text{regain}})
$$

Where:
- $R_{\text{for}}$ is the existing reward (goals-for etc.)
- $\lambda \ge 0$ is a tunable risk aversion weight
- $s_{\text{regain}}$ is the **opponent regain location state** inferred from data (proxy described below)

This is equivalent to adding a penalty to the expected state–action reward:

$$
\bar r(s,a) = \mathbb{E}[R(s,a,S') \mid s,a]
$$

and then evaluating the policy with the usual linear system / fundamental matrix.

## How to get opponent regain location (proxy)
We use the best-available proxy in the processed SkillCorner-derived tables:

Within each (match_id, period), sort player-possession events by time (frame_start/frame_end). For a row $i$:
- Let “next event” be row $i+1$.
- If `next_team_id != team_id`, treat that next possession start as the opponent regain.

Then:
- `opp_regain_x_norm, opp_regain_y_norm` = next row’s start coords
- `opp_regain_gap_frames` = next frame_start − current frame_end

Prototype + diagnostics exist in:
- [twelve-deep-learning/Project/data_testing.ipynb](twelve-deep-learning/Project/data_testing.ipynb)
- [twelve-deep-learning/Project/docs/risk_data_findings.md](twelve-deep-learning/Project/docs/risk_data_findings.md)

### Practical gate (recommended)
Because the proxy becomes noisy after stoppages, you can optionally only apply the turnover penalty when:
- `team_switch_next == True`, and
- `opp_regain_gap_frames <= MAX_GAP_FRAMES` (choose a conservative default, e.g. 50, and keep configurable)

If the gate fails, either:
- do not apply the penalty (simple, conservative), or
- apply a smaller penalty using a fallback (e.g., assume regain at the loss location)

## League-average opponent model (how to compute V_opp)
We want $V_{\text{opp}}(s)$: the expected “goals scored by the team in possession” starting from grid state $s$.

Two equivalent ways to obtain it (choose the simplest consistent with your current pipeline):

### Option A (recommended): mirror the MDP with possession perspective
1. Build the same grid MDP but interpret “reward” as goals-for by the team currently in possession.
2. Estimate transitions and a behavior policy from the full league dataset.
3. Evaluate the policy to obtain $V(s)$.
4. Use that $V$ as $V_{\text{opp}}$.

Because it’s league-average, you do not condition on team identity.

### Option B: use the existing model but swap perspective
If you already have a “value from a state for the team in possession”, then $V_{\text{opp}}$ is numerically the same function; the only difference is *when you look it up* (on possession loss, use the regain state).

## Where the risk penalty enters the exact evaluation
If you evaluate a fixed policy $\pi$ with an absorbing Markov chain approach, you typically have:
- $P_\pi$ = transition matrix under policy
- $r_\pi$ = expected immediate reward vector
- $V = (I - P_\pi)^{-1} r_\pi$ (for transient states)

Risk incorporation changes only how you compute $r_\pi$ (or, equivalently, the per-transition reward contributions before taking expectations). You do **not** need additional states.

## Implementation plan (fresh-chat ready)
1. Identify the exact place where the MDP reward vector is built.
2. Add opponent-regain mapping:
   - From data, compute `opp_regain_*` features and map `(opp_regain_x_norm, opp_regain_y_norm)` → grid state index.
3. Compute/serialize $V_{\text{opp}}$:
   - Build league-average opponent value function over grid states.
   - Save as an array indexed by grid cell.
4. Modify expected reward construction:
   - For each (s,a) that has a possession-loss outcome, subtract `lambda * V_opp[regain_state]`.
   - Keep existing goal rewards unchanged.
5. Add a small sweep utility for $\lambda$:
   - Evaluate several values (e.g., 0, 0.25, 0.5, 1.0) and sanity-check policy shifts.

## Sanity checks
- With $\lambda = 0$, results must match the baseline model.
- As $\lambda$ increases, actions with high turnover probability in dangerous areas should become less preferred.
- The penalty should vary by regain location: losing it near your box should hurt more than losing it high up the pitch.

## Known limitations
- The opponent-regain proxy can be wrong when the “next possession” is delayed by stoppages or missing event types; a tight gap threshold mitigates this.
- This is a league-average risk model; it will not capture team-specific counterpress/counterattack strengths unless you later segment by team.
