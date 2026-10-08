# SDDP4Portfolio

An ETF portfolio optimiser that decides, at each review point over a multi-year
horizon, whether to buy, sell, or hold each ETF in a fixed universe — and what
percentage of the portfolio each holding should end up at — under uncertain
future returns. The mathematical model is solved with Stochastic Dual Dynamic
Programming (SDDP): a bootstrap of historical returns drives the uncertainty,
Google OR-Tools (GLOP) solves the per-period linear program, and the objective
is a mean-CVaR blend so the policy is penalised for its worst-case outcomes,
not just its average one.

The project is Australia-specific: prices are converted to AUD, and the
objective accounts for franking credits (dividend imputation), which make a
dollar of Australian-equity return worth more, after tax, than a dollar of
equivalent foreign return.

## Project status

Complete as of 8 October 2026. The model, the solver and the plain-English
explainer all work end to end, and the solver's convergence diagnostics
behave as expected (see [documentation.md](documentation.md)). Nothing is
in progress. Ideas for future work are in [improvement.md](improvement.md).

## Where to start

| Read this | For |
|---|---|
| This file | Setting up, running, what the output means, how the code fits together, common changes, troubleshooting |
| [formulation.md](formulation.md) | The mathematical model: sets, data, variables, objective, every constraint, the cuts, and the pseudo-algorithm |
| [documentation.md](documentation.md) | How the code solves and validates that model: bootstrap, cut machinery, outer/inner loops, convergence tests, testing status, assumptions |
| [CLAUDE.md](CLAUDE.md) | Code conventions and the reasoning behind non-obvious design decisions — read before changing `sddp.py` |
| [improvement.md](improvement.md) | The project owner's list of possible extensions |

If you're new to SDDP, read formulation.md's "Value function and cuts" and
"Pseudo-algorithm" sections first. Everything in `sddp.py` follows from them.

## Setup

With conda (recommended), from this folder:

```bash
conda env create -f environment.yml
conda activate SDDP
```

Without conda, use any Python 3.11 virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependency versions are pinned in [requirements.txt](requirements.txt) to
the versions last checked on 8 October 2026. If you upgrade them,
`yfinance` is the most likely to break, because Yahoo Finance changes
its responses often.

To remove the environment later: `conda env remove -n SDDP`.

An internet connection is needed: prices, PE ratios and sector weightings
are fetched live from Yahoo Finance on every run.

## Running it

```bash
conda activate SDDP
cd Solution          # imports are flat, so run from inside Solution/
python formulation.py
```

A full run with the default settings takes about 12 minutes (734 s on 8
October 2026), almost all of it in the SDDP loop. Nothing is written to disk; everything is printed to the
terminal.

### What it prints

The output comes in four parts, in this order:

1. **Inputs.** The ETF universe, how many months of shared price history
   were found, the trailing PE ratio of each ETF and whether it passes the
   valuation gate, each ETF's franking credit yield, and how many
   historical return windows the bootstrap draws from.
2. **Solver progress.** One line per inner iteration:

   ```
   outer 0 inner 3: mean terminal value = ...  deterministic=... statistical=...(+/-...) gap=+...%
   ```

   - *deterministic* is the cuts' optimistic estimate of the objective.
   - *statistical* is the objective the policy actually achieved on this
     iteration's simulated paths, ± its confidence interval.
   - The inner loop stops once *deterministic* falls inside that
     interval. After each outer iteration the solver prints its
     re-estimate of the 95% Value-at-Risk (VaR).
   - A **warning** that the deterministic bound is below the statistical
     CI means the cuts are invalid. That's a bug, not noise.
   - Each outer iteration starts from no cuts, so its first inner
     iteration shows a huge gap. The deterministic bound starts at the a
     priori cap of 20 × `c0`. That's expected. It typically closes within
     2–3 inner iterations.
3. **Results.** The policy's simulated terminal wealth (mean, worst 5%,
   VaR), compared with buying and holding the benchmark ETF (DHHF.ASX by
   default) over the same horizon.
4. **Recommendation and explanation.** What to buy today from an all-cash
   start, in AUD and as portfolio weights, followed by a "Why this
   allocation" report covering:
   - which constraints are binding, read directly off the solution;
   - the role each holding plays (return driver, hedge, diversifier, ...),
     with the statistics behind it;
   - why each unheld ETF was left out.

   The roles are interpretations of historical data, not the optimiser's
   reasoning.

Example (abridged) from a run on 8 October 2026:

```
Fetching price history for: IVV.ASX, IOZ.ASX, NDQ.ASX, DHHF.ASX, BRK-B, IJP.ASX, IEM.ASX, QLTY.ASX, ESTX.ASX
82 months of overlapping history (2020-01-01 to 2026-10-01)
...
Bootstrapping from 70 historical 12-month return observations (ok), all enumerated in every backward pass
...
outer 9 inner 0: mean terminal value = 160,396  deterministic=2,000,000 statistical=-101,397(+/-21,113) gap=+2072.5%
outer 9 inner 1: mean terminal value = 391,032  deterministic=921,615 statistical=346,956(+/-32,833) gap=+165.6%
outer 9 inner 2: mean terminal value = 408,218  deterministic=350,098 statistical=361,978(+/-38,591) gap=-3.3%
  converged: deterministic bound within the statistical CI (z=1.96) after 3 inner iteration(s)
...
--- Results over a 10-period horizon (50 simulated paths) ---
Policy   mean terminal wealth: 408,218 AUD (simulated, 50 paths) (worst 5%: 209,133, VaR_95%: 229,289)
DHHF.ASX buy-and-hold benchmark: 334,919 AUD (analytic mean) (worst 5%: 186,273, simulated over 50 paths)

--- Suggested action today (period 1, starting from all cash) ---
IVV.ASX   : buy          0 AUD -> hold          0 AUD (  0.0%)
IOZ.ASX   : buy     14,985 AUD -> hold     14,985 AUD ( 15.0%)
NDQ.ASX   : buy     34,965 AUD -> hold     34,965 AUD ( 35.0%)
...
Total portfolio value after this trade: 99,900 AUD

--- Why this allocation ---
Expected return +16.2% per 12-month period, volatility 12.0%, about 4.2 effective holdings. NDQ.ASX is the main return driver, BRK-B is the main hedge.

Binding constraints (read directly from the solution):
  - NDQ.ASX: 35% single-holding cap
  - North America: 60% region limit (NDQ.ASX and BRK-B)
  ...
BRK-B -- 25% -- Primary hedge against NDQ.ASX
  - Offsets losses: it gained +7.4% on average in the portfolio's worst periods, when the portfolio as a whole lost 9.7%
  ...
```

Results change from run to run even though the random seed is fixed,
because the price history, PE ratios and sector data are fetched live.

## How the code fits together

```
Solution/
  formulation.py    entry point, main(): wires everything together and prints the report
  config.py         ModelConfig: every tunable parameter, in one place
  universe.py       ETFUniverse: tickers, currencies, look-through weights, price/PE/sector fetching
  sddp.py           SDDPSolver: the stage LP, cut management, the SDDP loop, today's recommendation
  explainer.py      PortfolioExplainer: plain-English "why" for the recommendation (reporting only)
  utils/
    scenarios.py    historical return windows, bootstrap sampling, buy-and-hold benchmark
```

A run, step by step (`main()` in `formulation.py`):

1. `ETFUniverse.default()` defines the nine ETFs. `fetch_prices()` downloads
   monthly closes and converts USD prices to AUD.
   `fetch_sector_weightings()` fills in the sector look-through.
2. `SDDPSolver.run()` does the following:
   - turns the prices into overlapping 12-month return windows
     (`historical_log_returns`);
   - fetches PE ratios for the valuation gate and works out each ETF's
     franking credit yield;
   - checks that the region minimums can be met at all;
   - runs the outer (ζ) and inner (SDDP) loops described in
     formulation.md's "Pseudo-algorithm".
3. The policy's terminal wealth is compared with a buy-and-hold benchmark
   (`analytic_buy_and_hold_mean` for the mean, `buy_and_hold_benchmark`
   for the tail).
4. `SDDPSolver.recommend_action()` solves period 1 from the all-cash start
   using the learned cuts. That gives today's recommended trade.
5. `PortfolioExplainer.report()` explains that trade.

The core of the model is `SDDPSolver.solve_stage()`. It builds one
period's LP, and each constraint in it is commented with the
formulation.md constraint it implements.

## Configuration

Every user-choosable parameter lives in one place:
[Solution/config.py](Solution/config.py)'s `ModelConfig` dataclass. This
covers:

- the mathematical model's data: weight caps, region and sector limits,
  transaction costs, CVaR confidence level, risk aversion, franking credit
  yield, starting cash;
- the horizon and period length;
- the SDDP solver's iteration budgets and convergence tolerances;
- the explainer's thresholds.

Change behaviour by editing values there. Nothing else should need touching
for a parameter change.

The ETF universe itself (tickers, currency, geographic look-through
weights, fallback PE ratios) lives in
[Solution/universe.py](Solution/universe.py)'s `ETFUniverse.default()`.

### Common changes

| To... | Change |
|---|---|
| Make the policy more or less defensive | `lambda_` (0 = maximise expected wealth only; towards 1 = protect the worst 5%) |
| Change the horizon or how often it rebalances | `horizon_years`, `months_per_period`. Shorter periods give more bootstrap windows but more stages, so a slower run |
| Loosen or tighten diversification | `w_bar` (single ETF), `theta_g_bar` / `theta_g_min` (regions), `theta_s_bar` (sectors) |
| Trade off runtime against accuracy | `n_outer`, `n_inner_start`/`_end`, `n_forward_start`/`_end`, `gap_ci_z`, `zeta_tolerance` |
| Compare against a different ETF | `benchmark_ticker` (must be one of the universe's labels) |
| Add or swap an ETF | In `ETFUniverse.default()`, add it to `ticker_map` (and `usd_tickers` if priced in USD), and give it a `geo` entry (region weights summing to 1, using the existing region names) and a `pe_fallback`. Check that its listing date doesn't shorten the shared price history (see below) |

## Troubleshooting

- **`ValueError: Only N monthly return observations are available ...`**
  The ETFs share too little price history to form return windows. The
  ETF with the shortest listing history sets the limit; DHHF.ASX
  (listed January 2020) currently does. Swap in a longer-listed ETF, or
  reduce `months_per_period`.
- **`ValueError: theta_g_min ...`** The region minimums can't be met given
  the other limits and the ETFs that pass the valuation gate. Lower the
  minimums, or add an ETF with exposure to that region. Region names in
  `theta_g_min` must match the keys in `universe.py`'s `geo`.
- **An ETF is never bought.** Check the "Valuation-eligible" line near
  the top of the output. If its live PE falls outside
  `pe_lower`–`pe_upper`, the valuation gate blocks it.
- **Yahoo Finance errors or missing data.** PE falls back to
  `pe_fallback`, and sector data falls back to the security's single
  sector. Prices have no fallback, so a failed price download stops the
  run. Retry later.
- **"deterministic bound is below the statistical CI" warning.** The
  cuts aren't a valid upper bound, which indicates a bug. See
  CLAUDE.md's notes on cut subgradients and on the backward pass, which
  describe the two past causes.
- **"reached n_outer=10 cap without converging".** This is normal with
  the default settings. The VaR estimate from 30–50 simulated paths moves
  by about 1% between outer iterations (as in the 8 October 2026 run),
  more than `zeta_tolerance` (0.1%), so the outer loop usually runs to
  the cap. The solver then keeps the ζ its cuts were built for, so the
  result is still consistent. Raising `n_forward_*` reduces that noise,
  at the cost of runtime.
- **The printed gap doesn't shrink to zero.** That's expected. The
  statistical bound is a fresh sample each iteration, so its confidence
  interval is a noise floor. See documentation.md, "Convergence
  diagnostics".

## Known limitations

- **No automated tests.** Validation has been by manual cross-checks and
  sanity runs; see documentation.md, "Testing — current state".
- **Hand-typed geographic weights.** The geographic look-through weights
  are continent-level estimates, deliberately no finer to avoid implying
  precision. Yahoo Finance doesn't publish a country or region breakdown
  for these funds. Sector look-through data, by contrast, is fetched live.
- **No serial correlation.** The bootstrap draws each period's return
  independently from history, so it doesn't capture momentum or mean
  reversion. It also can't produce a period worse than the worst
  historical window.
- **Short history.** History is about 82 months (limited by DHHF.ASX), so
  it covers only a few market episodes. The worst periods in the 8 October
  2026 run all fell in 2022.
- **Fixed PE.** The PE ratio is today's value, held constant over the whole
  horizon.
- **All-cash start.** The recommendation always starts from all cash
  (`c0`), not from an existing portfolio.
- **Simplified costs and tax.** Transaction costs are proportional, with
  no flat brokerage fee. Cash earns no interest. Capital gains tax isn't
  modelled.
- **Simplified CVaR.** ζ is found by fixed-point iteration rather than
  solved jointly with the trades (formulation.md, "Objective").
- **Benchmark is not an index.** The benchmark is buy-and-hold in one ETF
  from the universe, on the same simulated paths, not a historical index
  backtest.
