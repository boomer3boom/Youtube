"""Generic scenario-generation utilities: bootstrapping historical monthly
returns into growth-factor paths, and reconstructing price paths from them.

Kept separate from SDDPSolver because these are reusable independently of
the solver -- e.g. the buy-and-hold benchmark uses the same bootstrap to
simulate a naive single-ETF strategy over the same horizon.
"""

import numpy as np


def historical_log_returns(prices, months_per_period=1):
    """Historical log returns at review-point granularity.

    `prices` is always monthly data. When a review period spans more than
    one month, a period's return is the sum of its constituent monthly log
    returns (equivalent to compounding), computed over *overlapping*
    rolling windows rather than non-overlapping blocks -- with a short
    overlapping history across a universe, non-overlapping blocks would
    leave very few samples to bootstrap from; overlapping windows get
    several more out of the same data, at the cost of the samples no
    longer being independent of each other.
    """
    monthly_log_returns = np.log(prices / prices.shift(1)).dropna()
    if months_per_period == 1:
        result = monthly_log_returns.to_numpy()
    else:
        result = monthly_log_returns.rolling(months_per_period).sum().dropna().to_numpy()

    if len(result) < 2:
        raise ValueError(
            f"Only {len(monthly_log_returns)} monthly return observations are available "
            f"(from {len(prices)} months of overlapping price history across the universe), "
            f"which isn't enough to form even one overlapping {months_per_period}-month "
            f"return window with any variance to bootstrap from. Either reduce "
            f"months_per_period in config.py, or use a universe whose tickers share a "
            f"longer overlapping history (the binding constraint is whichever ETF has the "
            f"shortest listing history)."
        )
    return result


def sample_growth_factors(returns, n, rng):
    """Bootstrap n joint growth-factor vectors phi_{e,t} = P_{e,t}/P_{e,t-1}
    from history (readme.md: phi_{e,t})."""
    idx = rng.integers(0, len(returns), size=n)
    return np.exp(returns[idx])


def simulate_phi_path(returns, T, rng):
    """One sampled path of T growth-factor vectors, shape (T, n_etf)."""
    return sample_growth_factors(returns, T, rng)


def price_path_from_phi(p0, phi_path):
    return np.vstack([p0, p0 * np.cumprod(phi_path, axis=0)])


def buy_and_hold_benchmark(returns, T, rng, etf_index, c0, n_forward=200):
    """Naive comparison: hold a single ETF (by index into the universe)
    untouched over the horizon, simulated with the same bootstrap used by
    the SDDP solver's scenario generation.

    Used for tail statistics (e.g. worst-5%), which have no simple closed
    form under bootstrap resampling. For the mean, prefer
    analytic_buy_and_hold_mean() -- unlike the SDDP policy (whose mean is
    unavoidably noisy, since it depends on solved decisions with no closed
    form), a buy-and-hold position has an exact expected value, so there's
    no reason to leave simulation noise in it.
    """
    outcomes = []
    for _ in range(n_forward):
        phi_path = simulate_phi_path(returns, T, rng)
        outcomes.append(c0 * float(np.prod(phi_path[:, etf_index])))
    return outcomes


def analytic_buy_and_hold_mean(returns, T, etf_index, c0):
    """Exact expected terminal wealth for a buy-and-hold position in one
    ETF, under the same i.i.d.-bootstrap-with-replacement assumption used
    for scenario generation elsewhere: each period's growth factor is an
    independent draw from the same historical pool, so

        E[W_T] = c0 * mean(phi)^T

    using the *arithmetic* mean of the historical per-period growth
    factors phi = exp(r). That has to be mean(exp(r)), not exp(mean(r)) --
    those differ by Jensen's inequality (exp is convex), and only the
    former equals what bootstrap sampling actually averages to.
    """
    phi = np.exp(returns[:, etf_index])
    mean_phi = float(np.mean(phi))
    return c0 * mean_phi ** T
