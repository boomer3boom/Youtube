"""Every user-choosable parameter for a run, in one place.

Other files (formulation.py, sddp.py, universe.py) read from a ModelConfig
instance rather than defining their own defaults -- if you want to change
how the model behaves, this is the only file you should need to edit.

Sections below: readme.md Data/Parameters (the mathematical model itself),
SDDP run controls (iteration caps and convergence tolerances), reporting
(the buy-and-hold benchmark), and solver-internal engineering constants
(numerical robustness knobs with no readme.md counterpart -- see sddp.py
for why each one exists).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    # ------------------------------------------------------------------ #
    # readme.md "Data" and "Parameters"
    # ------------------------------------------------------------------ #
    w_bar: float = 0.35             # readme.md: w-bar, max weight in a single ETF
    theta_g_bar: float = 0.60       # readme.md: theta-bar^G_g, max region exposure
    theta_s_bar: float = 0.45       # readme.md: theta-bar^S_s, max sector exposure
    kappa_buy: float = 0.001        # readme.md: kappa^buy
    kappa_sell: float = 0.001       # readme.md: kappa^sell
    pe_lower: float = 0.0           # readme.md: PE-underline
    pe_upper: float = 45.0          # readme.md: PE-overline
    alpha: float = 0.95             # readme.md: alpha, CVaR confidence level
    lambda_: float = 0.5            # readme.md: lambda, risk-aversion weight
    c0: float = 100_000.0           # readme.md: C_0, starting cash (AUD)

    # ------------------------------------------------------------------ #
    # Run configuration (no readme.md symbol -- these fix the length and
    # granularity of the review points t in readme.md's set T)
    # ------------------------------------------------------------------ #
    months_per_period: int = 12     # review points every 12 months
    horizon_years: float = 10.0     # total investment horizon

    # ------------------------------------------------------------------ #
    # SDDP run controls (see sddp.py's SDDPSolver.run() docstring)
    # ------------------------------------------------------------------ #
    n_outer: int = 10               # cap on outer (zeta re-estimation) iterations
    n_inner: int = 20               # cap on inner (cut-refinement) iterations per outer iteration
    n_forward: int = 20             # scenario paths sampled per forward pass
    n_backward: int = 10            # scenario samples per backward-pass cut
    gap_tolerance: float = 0.05     # stop the inner loop once within this optimality gap
    zeta_tolerance: float = 0.05    # stop the outer loop once zeta moves less than this

    # ------------------------------------------------------------------ #
    # Reporting: buy-and-hold benchmark
    # ------------------------------------------------------------------ #
    benchmark_ticker: str = "IVV.ASX"   # which ETF to compare the policy against
    benchmark_min_paths: int = 200      # minimum simulated paths for the benchmark estimate

    # ------------------------------------------------------------------ #
    # Solver-internal engineering parameters (no readme.md symbol)
    # ------------------------------------------------------------------ #
    big_m_multiplier: float = 10.0        # M = big_m_multiplier * c0
    theta_upper_multiplier: float = 20.0  # loose a-priori cap on theta_t
    max_cuts_per_stage: int = 8
    cut_pi_tol: float = 3e-2
    cut_q_tol: float = 200.0
    relax_schedule: tuple = (0.0, 1.0, 10.0, 100.0, 1_000.0)

    @property
    def m(self) -> float:
        """readme.md: M, the big-M constant."""
        return self.big_m_multiplier * self.c0

    @property
    def theta_upper_bound(self) -> float:
        """A-priori upper bound on theta_t before any cuts exist."""
        return self.theta_upper_multiplier * self.c0

    @property
    def horizon_periods(self) -> int:
        """readme.md: T, the number of review points, given the horizon
        and how many months each period spans."""
        return round(self.horizon_years * 12 / self.months_per_period)
