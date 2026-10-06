"""Plain-English explanation of a recommended allocation: why each ETF is
held (or not), and what role it plays in the portfolio.

SDDPSolver only reports *what* to hold. PortfolioExplainer reads that
recommendation back against the same historical data and the model's own
constraints to suggest *why*: which holding drives return, which ones
diversify or hedge it, which constraints are binding, and why the
remaining ETFs were left out.

Two kinds of statement come out of this, and the report keeps them apart:

- Binding constraints (single-holding cap, geographic/sector look-through
  limits, valuation gate) are read directly off the solution, so they are
  facts about why the optimiser stopped where it did.
- Roles ("primary return driver", "hedge", "diversifier", ...) are post-hoc
  interpretations. The solver maximises mean-CVaR over the whole horizon
  and never computes correlations or labels itself; the roles summarise
  the statistics a person would look at, using the thresholds in
  config.py's "Reporting: explainer" section.

Nothing here refers to a specific ticker, region or sector, so it works for
any ETFUniverse.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from utils.scenarios import historical_log_returns


@dataclass
class BindingConstraint:
    """A constraint the recommended allocation sits on (zero slack)."""
    kind: str       # "holding" (single-holding cap), "region", "region_min" or "sector"
    subject: str    # the ETF, region or sector the limit applies to
    limit: float    # the cap, as a fraction of V
    members: list   # ETFs whose holdings make up the constrained exposure

    @property
    def display(self):
        return self.subject if self.kind == "holding" else _pretty(self.subject)

    @property
    def label(self):
        return {"holding": "single-holding cap", "region": "region limit",
                "region_min": "region minimum", "sector": "sector limit"}[self.kind]


@dataclass
class HoldingExplanation:
    etf: str
    weight: float
    role: str
    reasons: list = field(default_factory=list)


class PortfolioExplainer:
    """Explain one recommended allocation (a StageResult from
    SDDPSolver.recommend_action) in terms a person can act on.

    prices_hist : the same monthly price history the solver bootstrapped from
    rho, gamma  : the solver's valuation eligibility and franking credit
                  yields (SDDPSolver.rho / .gamma after run())
    """

    def __init__(self, universe, config, prices_hist, action, rho, gamma):
        self.universe = universe
        self.config = config
        self.rho = rho
        self.gamma = gamma

        e_set = universe.etfs
        prices = prices_hist[e_set]
        k = config.months_per_period

        # Per-period growth factors: the same pool the solver bootstraps
        # from, so "expected return" here means what the model means by it
        # (the arithmetic mean of phi, see analytic_buy_and_hold_mean).
        self.phi = np.exp(historical_log_returns(prices, k))
        self.window_ends = prices.index[k:]
        self.mu = pd.Series(self.phi.mean(axis=0) - 1, index=e_set)
        self.mu_rank = self.mu.rank(ascending=False, method="min").astype(int)
        self.cov = np.atleast_2d(np.cov(self.phi, rowvar=False))

        # Correlations from monthly returns rather than the overlapping
        # period windows: adjacent windows share k-1 months, which inflates
        # how alike any two series look.
        monthly = historical_log_returns(prices, 1)
        self.corr = pd.DataFrame(np.atleast_2d(np.corrcoef(monthly, rowvar=False)),
                                 index=e_set, columns=e_set)
        self.n_months = len(monthly)
        self.history_span = (prices.index[0], prices.index[-1])

        v = action.v
        self.w = pd.Series({e: action.h[e] / v if v else 0.0 for e in e_set})
        self.cash_weight = action.c / v if v else 0.0
        self.held = sorted((e for e in e_set if self.w[e] >= config.explain_min_weight),
                           key=lambda e: (-round(self.w[e], 4), -self.w[e] * self.mu[e]))

        # The portfolio's worst historical periods -- the (1-alpha) tail the
        # CVaR term cares about, with a floor of 3 so the average means
        # something -- and what each ETF did in them.
        r_p = (self.phi - 1) @ self.w.to_numpy()
        n_tail = max(3, math.ceil((1 - config.alpha) * len(r_p)))
        self.tail_idx = np.argsort(r_p)[:n_tail]
        self.port_tail = float(r_p[self.tail_idx].mean())
        self.etf_tail = pd.Series(self.phi[self.tail_idx].mean(axis=0) - 1, index=e_set)

    # ------------------------------------------------------------------ #
    # Portfolio-level statistics
    # ------------------------------------------------------------------ #

    @property
    def period_label(self):
        return f"{self.config.months_per_period}-month period"

    def expected_return(self):
        return float(self.w @ self.mu)

    def volatility(self):
        w = self.w.to_numpy()
        return math.sqrt(max(float(w @ self.cov @ w), 0.0))

    def risk_contributions(self):
        """Share of portfolio variance attributable to each holding:
        w_e (Sigma w)_e / (w' Sigma w). Sums to 1 across holdings."""
        w = self.w.to_numpy()
        var_p = float(w @ self.cov @ w)
        if var_p <= 0:
            return pd.Series(0.0, index=self.w.index)
        return pd.Series(w * (self.cov @ w) / var_p, index=self.w.index)

    def region_exposure(self):
        """readme.md: theta^G_g / V for the recommended allocation."""
        exposure = {g: sum(self.w[e] * self.universe.geo[e].get(g, 0.0) for e in self.universe.etfs)
                    for g in self.universe.regions}
        return pd.Series(exposure, dtype=float).sort_values(ascending=False)

    def binding_constraints(self):
        """Constraints from solve_stage with (near-)zero slack in the
        recommended allocation. Unlike the roles, these are read straight
        off the solution."""
        cfg = self.config
        universe = self.universe
        tol = cfg.explain_binding_tol
        binding = [BindingConstraint("holding", e, cfg.w_bar, [e])
                   for e in self.held if self.w[e] >= cfg.w_bar - tol]
        for kind, look_through, groups, limit in (
                ("region", universe.geo, universe.regions, cfg.theta_g_bar),
                ("sector", universe.sector, universe.sectors, cfg.theta_s_bar)):
            for grp in groups:
                share = sum(self.w[e] * look_through[e].get(grp, 0.0) for e in universe.etfs)
                if share >= limit - tol:
                    members = [e for e in self.held if look_through[e].get(grp, 0.0) > 0]
                    binding.append(BindingConstraint(kind, grp, limit, members))
        # Minimums bind from below: exposure sitting right at the floor
        # means the optimiser would hold less of that region if allowed.
        for g, minimum in cfg.theta_g_min.items():
            share = sum(self.w[e] * universe.geo[e].get(g, 0.0) for e in universe.etfs)
            if minimum > 0 and share <= minimum + tol:
                members = [e for e in self.held if universe.geo[e].get(g, 0.0) > 0]
                binding.append(BindingConstraint("region_min", g, minimum, members))
        return binding

    # ------------------------------------------------------------------ #
    # Per-ETF explanations
    # ------------------------------------------------------------------ #

    def explain(self):
        """One HoldingExplanation per ETF in the universe: held ETFs first
        (largest weight first), then the ones left out."""
        e_set = self.universe.etfs
        binding = self.binding_constraints()
        driver = max(self.held, key=lambda e: self.w[e] * self.mu[e]) if self.held else None

        explanations = [self._explain_held(e, driver, binding) for e in self.held]

        # The largest hedge is the portfolio's primary one.
        hedges = [x for x in explanations if x.role.startswith("Hedge")]
        if hedges:
            primary = max(hedges, key=lambda x: x.weight)
            primary.role = "Primary " + primary.role[0].lower() + primary.role[1:]

        for e in (e for e in e_set if e not in self.held):
            explanations.append(HoldingExplanation(e, float(self.w[e]), "Not held",
                                                   self._exclusion_reasons(e, binding)))
        return explanations

    def _explain_held(self, e, driver, binding):
        cfg = self.config
        reasons = self._binding_reasons(e, binding)
        exp_ret = self.expected_return()
        tail_explained = False

        if e == driver:
            role = "Primary return driver"
            if exp_ret > 0:
                reasons.append(f"Supplies {self.w[e] * self.mu[e] / exp_ret:.0%} of the portfolio's "
                               f"expected return of {exp_ret:+.1%} per {self.period_label}")
        else:
            c = float(self.corr.loc[e, driver])
            new_regions = [g for g, wt in self.universe.geo[e].items()
                           if wt > 0 and self.universe.geo[driver].get(g, 0.0) == 0]
            new_share = sum(self.universe.geo[e][g] for g in new_regions)
            inverse = c < cfg.explain_hedge_corr
            tail_hedge = self.etf_tail[e] > 0 > self.port_tail
            is_diversifier = c < cfg.explain_diversifier_corr or new_share >= 0.5
            strong_return = self.mu[e] >= self.mu.median()

            if inverse or tail_hedge:
                role = f"Hedge against {driver}"
                why = []
                if inverse:
                    why.append(f"tends to move opposite to {driver} (correlation {c:+.2f})")
                if tail_hedge:
                    why.append(f"gained {self.etf_tail[e]:+.1%} on average in the portfolio's worst "
                               f"periods, when the portfolio as a whole lost {-self.port_tail:.1%}")
                reasons.insert(0, "Offsets losses: it " + " and ".join(why))
                tail_explained = tail_hedge
            elif is_diversifier and strong_return:
                role = "Return with diversification"
            elif is_diversifier:
                role = "Diversifier"
            elif strong_return:
                role = "Secondary return driver"
            else:
                role = "Supporting holding"

            if not inverse:
                reasons.append(f"Moves {_describe_corr(c)} {driver} "
                               f"(correlation {c:+.2f} over {self.n_months} months)")
            if new_regions:
                reasons.append(f"Adds exposure {driver} doesn't have: "
                               f"{_format_exposure(self.universe.geo[e], new_regions)}")

            # A small position sitting behind binding caps is there because
            # it's the best remaining home for leftover capital.
            caps_hit = [b for b in binding if b.kind != "region_min" and e not in b.members]
            if not role.startswith("Hedge") and self.w[e] < cfg.explain_minor_weight and caps_hit:
                role = f"Residual allocation ({role.lower()})"
                reasons.insert(0, f"Takes the capital left over once {_describe_caps(caps_hit)}")

        reasons.append(f"Historical expected return {self.mu[e]:+.1%} per {self.period_label} "
                       f"(rank {self.mu_rank[e]} of {len(self.universe.etfs)})")
        reasons.append(self._risk_reason(self.risk_contributions()[e], self.w[e]))
        if not tail_explained:
            reasons.append(f"In the portfolio's {len(self.tail_idx)} worst historical periods it returned "
                           f"{self.etf_tail[e]:+.1%} on average (whole portfolio: {self.port_tail:+.1%})")
        if self.gamma.get(e, 0.0) > 0:
            reasons.append(f"Earns franking credits worth about {self.gamma[e]:.1%} per "
                           f"{self.period_label} on top of its price return")
        return HoldingExplanation(e, float(self.w[e]), role, reasons)

    def _binding_reasons(self, e, binding):
        reasons = []
        for b in binding:
            if e not in b.members:
                continue
            if b.kind == "holding":
                reasons.append(f"At the {_pct(b.limit)} single-holding cap: the optimiser would likely "
                               f"hold more if the cap were higher")
            elif b.kind == "region_min":
                others = [m for m in b.members if m != e]
                with_others = f" (together with {_join(others)})" if others else ""
                reasons.append(f"Needed to meet the {_pct(b.limit)} {b.display} region minimum"
                               f"{with_others}, which is binding: the optimiser holds no more "
                               f"{b.display} exposure than required")
            else:
                others = [m for m in b.members if m != e]
                with_others = f" together with {_join(others)}" if others else ""
                reasons.append(f"Fills the {_pct(b.limit)} {b.display} {b.label}{with_others}")
        return reasons

    def _risk_reason(self, rc_e, w_e):
        if rc_e < 0:
            return (f"Lowers overall portfolio variance (contribution {rc_e:.0%}) "
                    f"despite holding {w_e:.0%} of capital")
        if rc_e < w_e * 0.9:
            verdict = "less risk than its share of capital"
        elif rc_e > w_e * 1.1:
            verdict = "more risk than its share of capital"
        else:
            verdict = "risk roughly in line with its share of capital"
        return f"Accounts for {rc_e:.0%} of portfolio variance on {w_e:.0%} of capital ({verdict})"

    def _exclusion_reasons(self, e, binding):
        cfg = self.config
        geo = self.universe.geo[e]
        reasons = []
        if self.rho.get(e, 1.0) == 0:
            reasons.append(f"Failed the valuation gate: its PE is outside the "
                           f"{cfg.pe_lower:g}-{cfg.pe_upper:g} band, so it couldn't be bought")

        # Its main regions are already at their limit, filled by holdings
        # the optimiser preferred.
        for b in (b for b in binding if b.kind == "region"):
            if geo.get(b.subject, 0.0) >= 0.5:
                better = [m for m in b.members if self.mu[m] > self.mu[e]]
                filled_by = f", filled by {_join(better)} with higher expected returns" if better else ""
                reasons.append(f"{geo[b.subject]:.0%} of it is {b.display}, which is already at the "
                               f"{_pct(b.limit)} region limit{filled_by}")

        for h in self.held:
            c = float(self.corr.loc[e, h])
            if c >= cfg.explain_redundant_corr and self.mu[h] >= self.mu[e]:
                reasons.append(f"Largely redundant with {h} (correlation {c:+.2f}), which has the higher "
                               f"expected return ({self.mu[h]:+.1%} vs {self.mu[e]:+.1%})")

        # Mostly a repackaging of regions the portfolio already holds, at a
        # lower return than the portfolio gets from them.
        exposure = self.region_exposure()
        covered = [g for g, wt in geo.items() if wt > 0 and exposure.get(g, 0.0) >= cfg.explain_min_weight]
        covered_share = sum(geo[g] for g in covered)
        exp_ret = self.expected_return()
        if covered_share >= 0.5 and self.mu[e] < exp_ret and not reasons:
            reasons.append(f"{covered_share:.0%} of it is in regions the portfolio already holds "
                           f"({_format_exposure(geo, covered)}), at a lower expected return "
                           f"({self.mu[e]:+.1%}) than the portfolio's ({exp_ret:+.1%})")

        if self.mu_rank[e] == len(self.universe.etfs):
            reasons.append(f"Lowest historical expected return in the universe ({self.mu[e]:+.1%} per "
                           f"{self.period_label})")
        if self.held and self.etf_tail[e] <= self.port_tail:
            reasons.append(f"No help in a downturn: it returned {self.etf_tail[e]:+.1%} on average in the "
                           f"portfolio's worst periods, worse than the portfolio's {self.port_tail:+.1%}")

        if not reasons:
            reasons.append(f"Expected return of {self.mu[e]:+.1%} per {self.period_label} (rank "
                           f"{self.mu_rank[e]} of {len(self.universe.etfs)}) didn't earn a place in "
                           f"the portfolio once risk and constraints were weighed")
        return reasons

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #

    def report(self):
        cfg = self.config
        explanations = self.explain()
        binding = self.binding_constraints()
        # Effective number of holdings over the invested (non-cash) part.
        w = self.w.to_numpy()
        n_eff = float(w.sum() ** 2 / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else 0.0

        lines = ["", "--- Why this allocation ---"]
        if self.held:
            driver = next(x.etf for x in explanations if x.role == "Primary return driver")
            hedge = next((x.etf for x in explanations if x.role.startswith("Primary hedge")), None)
            hedge_msg = f", {hedge} is the main hedge" if hedge else ", with no clear hedge"
            lines.append(f"Expected return {self.expected_return():+.1%} per {self.period_label}, "
                         f"volatility {self.volatility():.1%}, about {n_eff:.1f} effective holdings. "
                         f"{driver} is the main return driver{hedge_msg}.")
        else:
            lines.append("Nothing is held: the whole portfolio is in cash.")
        if self.cash_weight >= cfg.explain_min_weight:
            lines.append(f"Cash: {self.cash_weight:.0%} held back. Cash earns nothing in this model, "
                         f"so it is there to limit downside, not for return.")

        lines.append("")
        lines.append("Binding constraints (read directly from the solution):")
        for b in binding:
            members = f" ({_join(b.members)})" if b.kind != "holding" else ""
            lines.append(f"  - {b.display}: {_pct(b.limit)} {b.label}{members}")
        if not binding:
            lines.append("  - none: the allocation is driven by return and risk alone")

        exposure = self.region_exposure()
        exposure = exposure[exposure >= cfg.explain_min_weight]
        if len(exposure):
            lines.append("")
            lines.append("Look-through region exposure: "
                         + ", ".join(f"{g} {x:.0%}" for g, x in exposure.items()))

        for x in explanations:
            lines.append("")
            lines.append(f"{x.etf} -- {x.weight:.0%} -- {x.role}")
            lines.extend(f"  - {r}" for r in x.reasons)

        lines.append("")
        lines.append("Notes:")
        lines.append("  - Roles are interpretations of historical statistics, not the optimiser's own "
                     "reasoning. Binding constraints are the exception.")
        history = (f"  - Statistics use {self.n_months} months of history "
                   f"({self.history_span[0]:%b %Y} to {self.history_span[1]:%b %Y}).")
        if self.held:
            ends = sorted(self.window_ends[self.tail_idx])
            history += (f" The worst periods are overlapping {self.period_label}s ending "
                        f"{_join([f'{d:%b %Y}' for d in ends])}, so they may all be one market episode.")
        lines.append(history)
        return "\n".join(lines)


def _describe_corr(c):
    if c < 0:
        return "against"
    if c < 0.3:
        return "largely independently of"
    if c < 0.7:
        return "only partly with"
    return "closely with"


def _describe_caps(caps):
    """'IOZ.ASX and NDQ.ASX hit the 35% single-holding cap and North America hit its 60% region limit'"""
    phrases = []
    for kind in ("holding", "region", "sector"):
        group = [b for b in caps if b.kind == kind]
        if not group:
            continue
        subjects = _join([b.display for b in group])
        verb_target = f"the {_pct(group[0].limit)} {group[0].label}" if kind == "holding" \
            else f"{'its' if len(group) == 1 else 'their'} {_pct(group[0].limit)} {group[0].label}"
        phrases.append(f"{subjects} hit {verb_target}")
    return _join(phrases)


def _pct(x):
    """A limit as a percentage, without rounding away a fractional one:
    0.35 -> '35%', 0.005 -> '0.5%'."""
    return f"{round(x * 100, 2):g}%"


def _format_exposure(exposure, keys, top=3):
    ranked = sorted(keys, key=lambda k: -exposure[k])
    text = ", ".join(f"{k} {exposure[k]:.0%}" for k in ranked[:top])
    return text + (f" and {len(ranked) - top} more" if len(ranked) > top else "")


def _join(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _pretty(slug):
    """Sector slugs from yfinance ('consumer_cyclical') to display text;
    leaves already-formatted names like 'North America' alone."""
    return slug.replace("_", " ").title() if slug.islower() else slug
