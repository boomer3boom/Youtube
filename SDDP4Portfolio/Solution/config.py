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

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    # ------------------------------------------------------------------ #
    # readme.md "Data" and "Parameters"
    # ------------------------------------------------------------------ #
    w_bar: float = 0.35             # readme.md: w-bar, max weight in a single ETF
    theta_g_bar: float = 0.60       # readme.md: theta-bar^G_g, max region exposure
    theta_g_min: dict = field(default_factory=lambda: {
        "North America": 0.05, "Australia": 0.15, "Asia": 0.05,
        "Europe": 0.01, "South America": 0.005, "Other": 0.01,
    })                               # readme.md: theta-underline^G_g, min exposure per
                                     # region; region names must match universe.py's
                                     # geo keys, and regions left out have no minimum
    theta_s_bar: float = 0.45       # readme.md: theta-bar^S_s, max sector exposure
    kappa_buy: float = 0.001        # readme.md: kappa^buy
    kappa_sell: float = 0.001       # readme.md: kappa^sell
    pe_lower: float = 0.0           # readme.md: PE-underline
    pe_upper: float = 45.0          # readme.md: PE-overline
    alpha: float = 0.95             # readme.md: alpha, CVaR confidence level
    lambda_: float = 0.2            # readme.md: lambda, risk-aversion weight
    c0: float = 100_000.0           # readme.md: C_0, starting cash (AUD)
    gamma_au_annual: float = 0.012  # readme.md: gamma^AU, Australian equity franking
                                     # credit yield (~4% dividend yield x ~70% franked
                                     # x 30/70 gross-up at the 30% company tax rate),
                                     # a benefit only an Australian resident taxpayer
                                     # can use -- see universe.py's L_G for how this
                                     # gets attributed per ETF via its Australian
                                     # look-through weight

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
    gap_ci_z: float = 1.96          # stop the inner loop once the deterministic bound is
                                     # within this many standard errors above the
                                     # statistical mean (1.96 = 95% CI); None = always
                                     # run the full n_inner
    zeta_tolerance: float = 0.001    # stop the outer loop once zeta moves less than this

    # n_inner/n_forward are ramped linearly from *_start (on the first outer
    # iteration) to *_end (on the last), so early outer iterations are
    # cheap and fast while later ones get bigger, more accurate samples --
    # see SDDPSolver.run()'s docstring for how this also feeds the
    # weighted-average zeta estimate (weighted by each iteration's own
    # n_forward, so later, larger-sample iterations naturally dominate the
    # average too). There is no backward-pass sample size: the backward
    # pass always enumerates every scenario in the bootstrap pool (see
    # formulation.md "Value function and cuts" for why it can't sample).
    n_inner_start: int = 15           # inner-loop cap on the first outer iteration
    n_inner_end: int = 25            # inner-loop cap on the last outer iteration
    n_forward_start: int = 30        # forward-pass sample size on the first outer iteration
    n_forward_end: int = 50          # forward-pass sample size on the last outer iteration

    # ------------------------------------------------------------------ #
    # Reporting: buy-and-hold benchmark
    # ------------------------------------------------------------------ #
    # Simulated with the same n_forward as the policy (see SDDP run
    # controls above) so the two are compared on an equal sample size --
    # otherwise a noisier (smaller-sample) policy estimate can look worse
    # than the benchmark purely from sampling variance, not real
    # underperformance.
    benchmark_ticker: str = "DHHF.ASX"  # which ETF to compare the policy against

    # ------------------------------------------------------------------ #
    # Reporting: explainer (explainer.py's role heuristics -- they only
    # change how the recommendation is described, never the recommendation)
    # ------------------------------------------------------------------ #
    explain_min_weight: float = 0.005      # below this weight an ETF counts as not held
    explain_minor_weight: float = 0.10     # held below this while other caps bind -> residual allocation
    explain_hedge_corr: float = 0.0        # correlation with the primary driver below this -> hedge
    explain_diversifier_corr: float = 0.6  # correlation with the primary driver below this -> diversifier
    explain_redundant_corr: float = 0.9    # an unheld ETF this correlated with a better holding -> redundant
    explain_binding_tol: float = 1e-3      # slack (as a fraction of V) within which a constraint is binding

    # ------------------------------------------------------------------ #
    # Solver-internal engineering parameters (no readme.md symbol)
    # ------------------------------------------------------------------ #
    big_m_multiplier: float = 10.0        # M = big_m_multiplier * c0
    theta_upper_multiplier: float = 20.0  # loose a-priori cap on theta_t
    max_cuts_per_stage: int = 500         # FIFO safety valve only -- every dropped cut
                                          # loosens the deterministic bound, so keep
                                          # this well above what a run accumulates
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

    @property
    def gamma_au_per_period(self) -> float:
        """readme.md: gamma^AU, scaled from an annual rate to whatever
        months_per_period is currently set to."""
        return self.gamma_au_annual * self.months_per_period / 12

    def ramped(self, outer: int, start: float, end: float) -> int:
        """Linearly interpolate between a *_start and *_end value, given
        the current outer iteration index (0-based). Reaches *_end exactly
        on outer = n_outer - 1, regardless of whether the loop actually
        runs that long (it may stop early via zeta_tolerance)."""
        frac = outer / max(self.n_outer - 1, 1)
        return round(start + (end - start) * frac)
