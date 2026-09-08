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

The full mathematical formulation — sets, data, decision variables, the
mean-CVaR objective, every constraint, the cut/value-function machinery, and a
pseudo-algorithm — is written up in [formulation.md](formulation.md). This
file covers how to run the code; see [CLAUDE.md](CLAUDE.md) for a map of the
codebase and its conventions.

## Setup

```bash
conda create -n SDDP python=3.11
conda activate SDDP
pip install ortools numpy pandas yfinance matplotlib scipy
```

## Running it

```bash
conda activate SDDP
cd Solution
python formulation.py
```

This fetches price history for the ETF universe (live, via yfinance),
bootstraps a return distribution from it, runs the SDDP solver to
convergence, and prints:

- the policy's simulated terminal-wealth distribution (mean, worst 5%,
  VaR<sub>95%</sub>) against an analytic buy-and-hold benchmark on the same
  sample size,
- a concrete buy/hold recommendation for *today* (period 1, starting from
  all cash), broken down by ETF with dollar amounts and resulting weights.

A full run currently takes several minutes, dominated by the SDDP outer/inner
iteration loop — see `ModelConfig` in [Solution/config.py](Solution/config.py)
for the knobs that trade off runtime against accuracy.

## Configuration

Every user-choosable parameter — the mathematical model's data (weight caps,
transaction costs, CVaR confidence level, risk aversion, franking credit
yield, ...), the horizon and period length, and the SDDP solver's iteration
budgets and convergence tolerances — lives in one place:
[Solution/config.py](Solution/config.py)'s `ModelConfig` dataclass. Change
behaviour by editing values there; nothing else in the codebase should need
touching for a parameter change.

The ETF universe itself (tickers, currency, geographic and sector
look-through data) lives in
[Solution/universe.py](Solution/universe.py)'s `ETFUniverse.default()`.

## Known limitations

- Geographic look-through weights are hand-typed estimates, not sourced from
  a live feed — Yahoo Finance doesn't expose a country/region breakdown for
  these funds. Sector look-through data, by contrast, is fetched live.
- The bootstrap is stagewise-independent (each period's return is drawn i.i.d.
  from history), so it doesn't capture serial correlation like momentum or
  mean reversion.
- See [improvement.md](improvement.md) for a running list of possible future
  extensions.
