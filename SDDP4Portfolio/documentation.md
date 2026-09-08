# Documentation (for a mathematician)

Short companion to [formulation.md](formulation.md): how the model is
actually solved and validated. Doesn't restate sets/variables/objective —
only the numerical machinery in `Solution/sddp.py` and
`Solution/utils/scenarios.py`.

## Simulation: stagewise-independent bootstrap

- $\phi_t$ is drawn i.i.d. per stage from a single fixed empirical
  distribution — no parametric family, no serial correlation, no regime
  switching.
- The distribution is historical monthly log returns, resampled with
  replacement. A single draw is a *joint* vector across all ETFs (one
  historical month), preserving cross-sectional correlation.
- For a $k$-month period, returns are summed over **overlapping** rolling
  $k$-month windows, not disjoint blocks — the shortest-history ETF in the
  universe leaves too few disjoint blocks (~6-7) to bootstrap from at
  $k=12$; overlapping windows give ~69, at the cost of adjacent windows
  sharing $k-1$ months (not independent of each other). Raises `ValueError`
  if fewer than 2 windows exist.
- No tail extrapolation: simulated outcomes can't exceed the convex hull of
  historically realised joint returns raised to power-$T$ combinations.

## Solving: SDDP as approximate DP

- Each stage's LP is standard (linear constraints, Rockafellar-Uryasev
  CVaR linearisation). The hard part is the continuation value
  $Q_{t+1}(h_t, C_t)$, approximated from below by Benders cuts built from
  LP dual values at trial states.
- State is dollar value, not units (`h = \phi h_{prev} + b - u`) — keeps
  the continuation value a function of state alone, independent of price
  level.
- $h_{prev}[e]$ appears in *two* constraints (holdings balance, and cash
  balance via the franking credit term), so its cut subgradient is the sum
  of both constraints' duals — a naive single-constraint cut formula would
  under-count it.
- Forward pass: simulate `n_forward` paths, solve stage LPs along each.
  Backward pass: for each *distinct* state visited at stage $t-1$ (dupes
  collapsed), resample `n_backward` fresh $\phi_t$, average objective/duals
  into one cut per state.

## CVaR: fixed-$\zeta$ outer loop, not a state variable

- $\zeta$ is fixed for an entire inner-loop solve (baked into the terminal
  stage's objective offset), then re-estimated between outer iterations as
  the empirical $\alpha$-quantile of realised terminal losses.
- Cuts are only valid for the $\zeta$ they were built under, so `self.cuts`
  is rebuilt from scratch every outer iteration.
- $\zeta$ itself is a running average across outer iterations, weighted by
  each iteration's `n_forward` (lower-variance quantile estimates get more
  trust; since `n_forward` ramps up, this also makes later iterations
  dominate — one mechanism, not two). `self.zeta` on return always matches
  the cuts actually returned.

## Convergence diagnostics

- **Inner loop**: deterministic bound (root-stage objective under current
  cuts, `recommend_action`) vs. statistical bound (sample mean/SE of
  realised objective over the forward pass, `_statistical_bound`). Stops
  when the gap $\le$ `gap_tolerance`.
- **Outer loop**: stops when the weighted $\zeta$ average moves by
  $\le$ `zeta_tolerance`, or `n_outer` is hit.
- `n_inner`/`n_forward`/`n_backward` ramp small→large across outer
  iterations (`config.ramped`).
- The statistical bound is a *fresh* resample every inner iteration, so its
  own CI width is a noise floor on the achievable gap — an oscillating gap
  late in a run reflects that floor (scales as $1/\sqrt{n_{forward}}$), not
  a bug. Don't chase a tight `gap_tolerance` without raising `n_forward`.
- Degeneracy guards: near-duplicate cuts dropped, cuts per stage capped
  (FIFO), duplicate states collapsed to one backward solve — all to stop
  GLOP's simplex landing on an almost-singular basis.

## Testing — current state

No automated test suite exists (no `test_*.py`). Validation so far:

- Structural: every constraint in `solve_stage` is commented with the
  formulation.md constraint it implements and uses the same symbol.
- End-to-end sanity checks on live runs (e.g. franking-credit yields match
  hand-computed expectations from look-through weights; gap shrinks from
  ~900% pre-cuts to a stable band consistent with the sampling floor above).
- Three real bugs were caught this way, not by tests: a missing
  $-\lambda\zeta$ objective offset, stale cuts surviving a $\zeta$ change,
  and an incomplete subgradient (missing the franking-credit constraint's
  contribution).
- Not checked: cut validity as a true supporting hyperplane, bootstrap
  statistical properties at the sample sizes used, LP feasibility
  invariants. Treat the missing test suite as the main gap before trusting
  a modified version of the algorithm.

## Key assumptions

1. Returns are stagewise-independent draws from a fixed empirical
   distribution — no momentum, mean reversion, or regime switching.
2. No drift in the return-generating process over the horizon.
3. CVaR uses a fixed-$\zeta$-per-outer-iteration relaxation, not
   formulation.md's fully time-consistent state-augmented version.
4. Convergence is judged against sampling-noise-bounded estimates, not an
   exact certificate.
5. Validated by manual cross-checks and sanity runs, not automated tests.
