# Section 10 Refactor: Exact Strategy Comparison (GPU)

## Summary
Section 10 of the main notebook was refactored to replace the Monte Carlo simulation approach with an **exact, deterministic matrix method** that scales well to the larger state space.

The goal of Section 10 remains the same:

- **Strategy A:** 1 `long_forward`
- **Strategy B:** 2 `short_forward` (forced twice)

But instead of simulating many rollouts per start state, we compute the strategy values analytically using the learned transition matrices and the behaviour policy.

The implementation runs on **GPU when available** (via PyTorch CUDA), since the heavy work is linear algebra.

Notebook: `Project/pass_decision_analysis.ipynb`

---

## Key idea (what we compute)

Let:

- $P(s,a,s')$ be the learned transition probabilities.
- $\pi(a\mid s)$ be the team behaviour policy.
- The MDP has **transient (field) states** plus **3 absorbing states** (goal / no-goal / loss).

We compute a continuation value vector $V$ where:

- $V(s) = \Pr(\text{eventually reach goal} \mid S_0=s, \text{follow }\pi)$ for transient states.
- $V(\text{goal}) = 1$.
- $V(\text{no-goal}) = 0$.
- $V(\text{loss}) = 0$.

Once we have $V$, evaluating a scripted sequence is just “push $V$ backwards through the scripted transitions”.

### Strategy values
Define the per-action transition matrix $P_a$ as:

- $P_a[s,s'] = P(s,a,s')$.

Then:

- **1 long forward**: $Q_{\text{long}} = P_{\text{long}} V$
- **2 short forwards**: $Q_{\text{2short}} = P_{\text{short}}(P_{\text{short}} V)$

For each start state $s$, the decision comparison is:

- $\Delta(s) = Q_{\text{2short}}(s) - Q_{\text{long}}(s)$

Positive $\Delta(s)$ means “2 short forward passes” is better.

---

## How we compute the continuation value $V$ (theory)

Fixing a policy $\pi$ turns the MDP into an absorbing Markov chain. Over transient states, the policy-induced transition matrix is:

$$
Q(s,s') = \sum_a \pi(a\mid s) P(s,a,s')
$$

Let $\text{goal}$ be the goal absorbing state. Define the one-step probability of going directly to goal under the policy:

$$
b(s) = \sum_a \pi(a\mid s) P(s,a,\text{goal})
$$

The continuation value satisfies:

$$
V = b + QV
$$

so:

$$
(I - Q)V = b
$$

In earlier sections, this is equivalent to using the fundamental matrix $N=(I-Q)^{-1}$ and then $V = N b$.

In Section 10, we compute the same result using a **linear solve** (more GPU-friendly and avoids explicitly forming the inverse).

Implementation function (in the notebook): `compute_continuation_value_vector()`.

---

## How `evaluate_scripted_sequence()` works

The notebook function `evaluate_scripted_sequence(P, V, action_id, n_steps, device)` does:

1. Extract the per-action transition matrix $P_a = P[:, action_id, :]$.
2. Apply matrix-vector multiplication repeatedly:
   - 1 step: $P_a V$
   - 2 steps: $P_a(P_a V)$

In Python, `@` is the matrix multiplication operator, so `P_a @ V` is the standard matvec.

This is exactly computing the expected continuation value after forcing the scripted actions for `n_steps` steps.

---

## Region of interest (ROI)

To keep the analysis focused, Section 10 evaluates only states:

- In the **attacking half**
- But **outside the box**

With a 34-column grid, this corresponds to:

- Columns **17–28** inclusive (0-based)
- Columns **18–29** if counting columns starting at 1

The notebook filters the evaluated states using column indices.

---

## GPU usage

Section 10 uses PyTorch and selects:

- `cuda` when available
- otherwise CPU

The heavy operations that benefit from GPU are:

- building $Q$ and $b$
- solving $(I-Q)V=b$
- repeated dense matvecs for scripted evaluation

To avoid PCIe overhead, matrices/vectors are moved to the GPU once and kept there during computation.

---

## How to run Section 10

1. Run the notebook up to the point where `team_mdps` and `grid` are loaded/defined.
2. Run the Section 10 cells in order:
   - Setup (GPU + helper functions)
   - Single-state exact example
   - ROI-wide exact evaluation for one team
   - ROI-wide exact evaluation for all teams

Expected runtime should now be dominated by a handful of matrix operations instead of thousands of rollouts.

---

## Assumptions (important)

This refactor preserves the modelling assumptions used throughout the notebook:

- After the scripted pass(es), the team **returns to its behaviour policy** $\pi$.
- “Expected goals” is implemented as the **probability of eventually reaching the goal absorbing state**.
- Absorbing states are truly absorbing (self-loop), so sequences that lose possession naturally yield 0 additional value.

If these assumptions change (e.g., different continuation policy after the scripted actions), the continuation value $V$ must be recomputed accordingly.
