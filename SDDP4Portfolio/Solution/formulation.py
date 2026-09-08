"""Entry point: builds the ETF universe, fetches price history, runs the
SDDP solver, and reports policy performance against a buy-and-hold
benchmark. See readme.md for the full mathematical model, universe.py for
the ETF/ticker data, and sddp.py for the solver itself.
"""

import numpy as np

from config import ModelConfig
from sddp import SDDPSolver
from universe import ETFUniverse
from utils.scenarios import analytic_buy_and_hold_mean, buy_and_hold_benchmark, historical_log_returns


def main():
    universe = ETFUniverse.default()
    config = ModelConfig()

    print("Fetching price history for:", ", ".join(universe.etfs))
    prices_hist = universe.fetch_prices()
    print(f"{len(prices_hist)} months of overlapping history "
          f"({prices_hist.index[0].date()} to {prices_hist.index[-1].date()})")

    print("Fetching live sector composition...")
    universe.fetch_sector_weightings()

    solver = SDDPSolver(universe, config)
    T = config.horizon_periods  # readme.md: T -- horizon_years worth of months_per_period-month periods
    print(f"Horizon: {T} periods of {config.months_per_period} months "
          f"({config.horizon_years:.0f} years)")

    _cuts, zeta, _rho, terminal_values = solver.run(prices_hist, T)

    returns = historical_log_returns(prices_hist, config.months_per_period)
    benchmark_label = config.benchmark_ticker
    benchmark_index = universe.etfs.index(benchmark_label)

    # Mean: computed analytically (exact, zero sampling noise) -- unlike
    # the policy, a buy-and-hold position has no solved decisions, so its
    # expected value has a closed form under the same bootstrap assumption.
    benchmark_mean = analytic_buy_and_hold_mean(returns, T, benchmark_index, config.c0)
    # Tail: no closed form, so still simulated -- but at the same sample
    # size as the policy (config.n_forward, via len(terminal_values)) so
    # that comparison specifically stays apples-to-apples.
    benchmark_values = buy_and_hold_benchmark(
        returns, T, solver.rng, benchmark_index, config.c0,
        n_forward=len(terminal_values),
    )

    worst_pct = int((1 - config.alpha) * 100)
    policy_worst = np.quantile(terminal_values, 1 - config.alpha)
    benchmark_worst = np.quantile(benchmark_values, 1 - config.alpha)

    print(f"\n--- Results over a {T}-period horizon ({len(terminal_values)} simulated paths) ---")
    print(f"Policy   mean terminal wealth: {np.mean(terminal_values):,.0f} AUD (simulated, "
          f"{len(terminal_values)} paths) (worst {worst_pct}%: {policy_worst:,.0f}, "
          f"VaR_{config.alpha:.0%}: {-zeta:,.0f})")
    print(f"{benchmark_label} buy-and-hold benchmark: {benchmark_mean:,.0f} AUD (analytic mean) "
          f"(worst {worst_pct}%: {benchmark_worst:,.0f}, simulated over {len(benchmark_values)} paths)")

    action = solver.recommend_action(T)
    print("\n--- Suggested action today (period 1, starting from all cash) ---")
    for e in universe.etfs:
        weight = action.h[e] / action.v if action.v else 0.0
        print(f"{e:10s}: buy {action.b[e]:>10,.0f} AUD -> hold {action.h[e]:>10,.0f} AUD ({weight:6.1%})")
    cash_weight = action.c / action.v if action.v else 0.0
    print(f"{'Cash':10s}: {'':>10}{'':3}    hold {action.c:>10,.0f} AUD ({cash_weight:6.1%})")
    print(f"Total portfolio value after this trade: {action.v:,.0f} AUD")


if __name__ == "__main__":
    main()
