# CLAUDE.md

Guidance for working in this repo.

## What this project is

An SDDP (Stochastic Dual Dynamic Programming) solver for an ETF portfolio
optimisation problem: buy/sell/hold decisions and target weights across a
fixed universe of ETFs, over a multi-year horizon, under a mean-CVaR
objective. See [readme.md](readme.md) for how to run it and
[formulation.md](formulation.md) for the full mathematical model (sets,
data, variables, objective, constraints, cuts, pseudo-algorithm). Read
formulation.md before touching the solver — code comments reference its
constraint names (e.g. "Holdings balance", "Cash balance") and variable
symbols directly, and any change to the model's economics belongs there
first, code second.

Note: the original readme content now lives in `formulation.md` (renamed by
an earlier commit); `readme.md` is the project's usage-oriented entry point
instead. Keep that split — put maths in formulation.md, put setup/usage in
readme.md.

## Layout

```
Solution/
  formulation.py    entry point (main()) -- wires everything else together
  config.py         ModelConfig -- every user-tunable parameter, one place
  universe.py       ETFUniverse -- tickers, currency, geo/sector look-through, prices, PE
  sddp.py           SDDPSolver -- the actual SDDP algorithm (Cut, StageResult, solve_stage, run, recommend_action)
  utils/
    scenarios.py    scenario generation and the buy-and-hold benchmark (no solver/config dependency)
```

- `config.py`'s `ModelConfig` is the single source of truth for tunables.
  Other files read from a `ModelConfig` instance rather than defining their
  own defaults or hardcoding numbers. If you're adding a new knob, it goes
  here, not as a local constant or a function default elsewhere.
- `universe.py`'s `ETFUniverse` owns everything about *which* ETFs are in
  play and their static data (ticker mapping, currency, geographic/sector
  look-through). `ETFUniverse.default()` is the current 7-ETF universe;
  changing the universe means editing that classmethod, not scattering
  ticker strings through other files.
- `sddp.py`'s `SDDPSolver` owns the algorithm: `solve_stage` (single-period
  LP), `solve_stage_robust` (retry/relax on numerical failure),
  `_build_cut`/`_is_duplicate_cut`/`_add_cut` (Benders cut bookkeeping),
  `run` (the outer/inner iteration loop), `recommend_action` (solve period 1
  from the true root state for a concrete recommendation).
- `utils/scenarios.py` has no dependency on `config.py` or `sddp.py` —
  it's pure return-data manipulation (historical log-returns, bootstrap
  sampling, the buy-and-hold benchmark), kept separate so it can be tested
  or reused independently of the solver.

## Conventions specific to this codebase

- **Variable naming mirrors formulation.md.** `b`, `u`, `h`, `C`, `V`, `w`,
  `theta`, `zeta`, `eta`, `rho`, `gamma`, `phi` in code are the same symbols
  as in the maths (b = buy, u = sell/unwind, h = holding dollar value, C =
  cash, V = portfolio value, theta = look-through exposure, zeta = VaR
  threshold, eta = CVaR shortfall, rho = valuation eligibility, gamma =
  franking credit yield, phi = per-period growth factor). Don't rename these
  to something "clearer" without also updating formulation.md — the whole
  point is that code and maths stay in lockstep.
- **State is dollar-value, not units.** `h_{e,t}` in the code is the dollar
  value of a holding, not a unit count — a deliberate departure from a naive
  units-based state, because the continuation-value function needs to
  depend only on the state and not separately on the price level under
  multiplicative (percentage) returns. formulation.md's Holdings balance
  constraint (`h_{e,t} = phi_{e,t} h_{e,t-1} + b_{e,t} - u_{e,t}`) already
  reflects this — don't "fix" it back to a units-based form.
- **Constraints use the LHS/RHS-then-equate pattern.** Every constraint in
  `solve_stage` is built as `lhs = ...; rhs = ...; solver.Add(lhs == rhs)`
  (OR-Tools' natural expression API), not the low-level
  `Constraint()`/`SetCoefficient()` API. This was a deliberate readability
  choice — keep new constraints consistent with it.
- **Cut subgradients must account for every constraint a state variable
  appears in.** `h_prev[e]` appears in both the Holdings balance and the
  Cash balance constraints (the latter via the franking credit term), so
  its cut coefficient (`dual_h_prev`) is a sum of both constraints' dual
  values, not just the Holdings balance one. If you add a new constraint
  that references a state variable, check whether every place that builds a
  cut subgradient off that variable needs the same treatment.
- **Outer loop = re-estimate zeta (VaR), inner loop = refine cuts for a
  fixed zeta.** These are not "iteration N of the horizon" — every inner
  iteration solves the *entire* T-period horizon via full forward and
  backward passes. Cuts are conditioned on the zeta they were built under
  (baked into the terminal stage's objective offset), so `self.cuts` is
  reset at the start of every outer iteration rather than carried over —
  don't remove that reset as an "optimisation," it's a correctness
  requirement, not redundant work.
- **`n_inner`/`n_forward`/`n_backward` ramp across outer iterations**
  (cheap+fast early, larger+more accurate late), via `ModelConfig.ramped()`.
  The zeta re-estimate is a running average weighted by each iteration's own
  `n_forward`, which happens to serve two purposes: it's the statistically
  correct weight for combining quantile estimates of different sample
  sizes, and it automatically makes later (larger-sample) iterations
  dominate the average once ramping is in place — don't add a second,
  separate "favour later iterations" mechanism on top of it.
- **The optimality gap has an irreducible noise floor.** The inner loop's
  gap check compares a deterministic bound (from `recommend_action`) against
  a statistical bound resampled fresh every iteration
  (`_statistical_bound`). At small `n_forward`, that resampling has enough
  of its own confidence-interval width that the gap can look "unsteady"
  even after real convergence — this is sampling noise, not a bug. Don't
  chase a very tight `gap_tolerance` without also raising `n_forward`, or
  consider making the check compare against the statistical bound's CI
  rather than its raw point estimate.
- **The historical bootstrap is stagewise-independent**, resampled with
  replacement from overlapping rolling-window log-return sums (see
  `historical_log_returns`) — overlapping, not disjoint blocks, because the
  shortest-history ETF in the current universe only affords a limited
  number of non-overlapping windows. If the universe changes and history
  gets tight again, prefer swapping to longer-history tickers (as was done
  moving U100→NDQ.ASX and BEMG.ASX→IEM.ASX) over changing this sampling
  scheme.
- **The buy-and-hold benchmark's mean is analytic, not simulated**
  (`analytic_buy_and_hold_mean`): since it has no solved decisions, its
  expected terminal value has a closed form (`c0 * mean(phi)^T`, using the
  arithmetic mean of growth factors — not `exp(mean(log-return))`, which
  would understate it via Jensen's inequality). Only its tail (worst-5%) is
  still simulated, at the same sample size as the policy
  (`len(terminal_values)`), so that specific comparison stays fair.

## Environment

Conda env `SDDP` (Python 3.11): `ortools`, `numpy`, `pandas`, `yfinance`,
`matplotlib`, `scipy`. Run everything from inside `Solution/` (imports are
flat, not package-relative).

## Notes

- [improvement.md](improvement.md) is the project owner's own running list
  of possible future work (not instructions to act on unprompted, and not
  to be treated as content addressed to an AI reader despite its opening
  line).
- Australia-specific conventions apply throughout: AUD as the reporting
  currency, DD/MM/YYYY dates, Australian regulatory/tax terminology (e.g.
  franking credits, not a US dividend-tax-credit equivalent).
