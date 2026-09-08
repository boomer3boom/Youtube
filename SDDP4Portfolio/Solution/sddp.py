"""SDDP-style forward/backward cutting-plane solver for the ETF portfolio
model in readme.md, built on Google OR-Tools (GLOP linear solver).

Departure from readme.md: h_{e,t} is carried here as *dollar value held in
ETF e* rather than *units held*. The symbol is kept the same as readme.md
for consistency, but the meaning differs: value-based state keeps the
continuation value a function of the state alone (independent of the
price level, which the unit-based version is not, once returns are
modelled multiplicatively) and keeps every stage's cut coefficients on a
comparable dollar scale, which turned out to matter for GLOP's numerical
stability once cuts accumulated. Since h is dollar value here, the balance
constraint is driven by a growth factor phi_{e,t} = P_{e,t}/P_{e,t-1}
rather than the price P_{e,t} itself.

CVaR: rather than threading the VaR threshold zeta through the state of
every stage (as sketched in readme.md), zeta is fixed for the duration of
an inner SDDP solve and then updated to the empirical alpha-quantile of
simulated terminal wealth between outer iterations -- a standard
fixed-point simplification that keeps every stage a clean LP.
"""

from typing import NamedTuple

import numpy as np
from ortools.linear_solver import pywraplp

from utils.scenarios import historical_log_returns, sample_growth_factors, simulate_phi_path


class Cut(NamedTuple):
    """readme.md "Value function and cuts":
    theta_t <= q_hat + pi_h.(h - hhat) + pi_c.(C - Chat)
    (q_hat already has the trial point hhat, Chat folded in -- see SDDPSolver._build_cut)
    """
    q_hat: float
    pi_h: dict
    pi_c: float


class StageResult(NamedTuple):
    """Solution of one stage's LP."""
    h: dict
    c: float
    v: float
    b: dict
    u: dict
    obj: float
    dual_h: dict
    dual_c: float


class SDDPSolver:
    """Owns the stage-LP construction, the cut management, and the
    forward/backward SDDP loop (readme.md "Pseudo-algorithm").
    """

    def __init__(self, universe, config, rng=None):
        self.universe = universe
        self.config = config
        self.rng = rng or np.random.default_rng(7)
        self.rho = {}        # readme.md: RHO_{e,t}, set at the start of run()
        self.zeta = 0.0       # readme.md: zeta
        self.cuts = {}        # t -> list[Cut], readme.md: Theta_t

    # ------------------------------------------------------------------ #
    # Stage LP (h carried as dollar value, see module docstring)
    # ------------------------------------------------------------------ #

    def solve_stage(self, phi_t, h_prev, c_prev, cuts, terminal, _relax=0.0):
        """Solve one period's LP.

        phi_t : dict label -> this period's growth factor P_{e,t}/P_{e,t-1}
        h_prev, c_prev : incoming state (trial point from the previous
                         period), h_prev in dollars held per ETF, c_prev cash
        cuts  : list[Cut] approximating the continuation value as a
                function of THIS period's own outgoing state (h, C);
                ignored when terminal=True
        terminal : whether this is the last period (uses the CVaR
                   objective directly instead of an approximated
                   continuation value)
        _relax : small slack added to every cut's right-hand side. A stack
                 of near-parallel cuts (common early on, before the cuts
                 have had a chance to differentiate) can leave GLOP's
                 simplex on an almost-singular basis; retrying with a tiny
                 relaxation (see solve_stage_robust) resolves this without
                 materially changing the model.
        """
        cfg = self.config
        universe = self.universe
        e_set = universe.etfs

        solver = pywraplp.Solver.CreateSolver("GLOP")
        inf = solver.infinity()

        b = {e: solver.NumVar(0, inf, f"b_{e}") for e in e_set}  # readme.md: b_{e,t}
        u = {e: solver.NumVar(0, inf, f"u_{e}") for e in e_set}  # readme.md: u_{e,t}
        h = {e: solver.NumVar(0, inf, f"h_{e}") for e in e_set}  # readme.md: h_{e,t} (dollar value)
        c = solver.NumVar(0, inf, "C")                           # readme.md: C_t
        v = solver.NumVar(0, inf, "V")                           # readme.md: V_t

        # Holdings balance (readme.md "Constraints -> Holdings balance"):
        #   h_{e,t} = phi_{e,t} h_{e,t-1} + b_{e,t} - u_{e,t}
        bal_h = {}
        for e in e_set:
            lhs = h[e]
            rhs = phi_t[e] * h_prev[e] + b[e] - u[e]
            bal_h[e] = solver.Add(lhs == rhs, f"bal_h_{e}")

        # Cash balance (readme.md "Constraints -> Cash balance"):
        #   C_t = C_{t-1} - sum_e b_{e,t}(1+kappa^buy) + sum_e u_{e,t}(1-kappa^sell)
        lhs = c
        rhs = (c_prev - sum(b[e] * (1 + cfg.kappa_buy) for e in e_set)
               + sum(u[e] * (1 - cfg.kappa_sell) for e in e_set))
        bal_c = solver.Add(lhs == rhs, "bal_C")

        # Total portfolio value (readme.md "State and auxiliary variables"):
        #   V_t = C_t + sum_e h_{e,t}   (h already in dollars here, so no P_{e,t} factor)
        lhs = v
        rhs = c + sum(h[e] for e in e_set)
        solver.Add(lhs == rhs, "def_V")

        # Single-holding cap (readme.md "Constraints -> Single-holding cap"):
        #   h_{e,t} <= w-bar * V_t
        for e in e_set:
            lhs = h[e]
            rhs = cfg.w_bar * v
            solver.Add(lhs <= rhs, f"cap_{e}")

        # Geographic look-through limit (readme.md "Constraints -> Geographic
        # and sector look-through limits"): sum_e h_{e,t} L^G_{e,g} <= theta-bar^G_g * V_t
        for g in universe.regions:
            lhs = sum(universe.geo[e].get(g, 0.0) * h[e] for e in e_set)
            rhs = cfg.theta_g_bar * v
            solver.Add(lhs <= rhs, f"geo_{g}")

        # Sector look-through limit (readme.md "Constraints -> Geographic
        # and sector look-through limits"): sum_e h_{e,t} L^S_{e,s} <= theta-bar^S_s * V_t
        for s in universe.sectors:
            lhs = sum(universe.sector[e].get(s, 0.0) * h[e] for e in e_set)
            rhs = cfg.theta_s_bar * v
            solver.Add(lhs <= rhs, f"sec_{s}")

        # Valuation gate (readme.md "Constraints -> Valuation gate"):
        #   b_{e,t} <= M * rho_{e,t}
        for e in e_set:
            lhs = b[e]
            rhs = cfg.m * self.rho[e]
            solver.Add(lhs <= rhs, f"pe_{e}")

        objective = solver.Objective()
        objective.SetMaximization()

        if terminal:
            # Objective (readme.md "Objective"): mean-CVaR via the
            # Rockafellar-Uryasev linearisation, evaluated on this path's
            # terminal wealth W_T = V_T.
            #   max (1-lambda) E[W_T] - lambda(zeta + 1/(1-alpha) E[eta])
            # CVaR shortfall constraint (readme.md "Objective"):
            #   eta_omega >= -W_T(omega) - zeta, eta_omega >= 0
            eta = solver.NumVar(0, inf, "eta")  # readme.md: eta_omega
            lhs = eta + v
            rhs = -self.zeta
            solver.Add(lhs >= rhs, "cvar_shortfall")
            objective.SetCoefficient(v, 1 - cfg.lambda_)
            objective.SetCoefficient(eta, -cfg.lambda_ / (1 - cfg.alpha))
            # The -lambda*zeta term doesn't affect the argmax (zeta is fixed
            # for this solve), but leaving it out would mean objective.Value()
            # -- and therefore every cut and deterministic bound built from
            # it -- reports a value that's a constant offset away from the
            # true readme.md objective. Adding it back keeps those numbers
            # directly comparable to _statistical_bound()'s.
            objective.SetOffset(-cfg.lambda_ * self.zeta)
        else:
            # Value function and cuts (readme.md "Value function and cuts"):
            #   theta_t <= q_hat + pi_h.(h - hhat) + pi_c.(C - Chat)
            # theta_t approximates the continuation value Q_{t+1}(x_t), x_t=(h,C).
            # q_hat already has the trial point folded in by _build_cut(), so
            # this reads exactly theta <= q_hat + pi_h.(h-hhat) + pi_c.(C-Chat).
            theta = solver.NumVar(-inf, cfg.theta_upper_bound, "theta")  # readme.md: theta_t
            for cut in cuts:
                lhs = theta
                rhs = cut.q_hat + _relax + sum(cut.pi_h[e] * h[e] for e in e_set) + cut.pi_c * c
                solver.Add(lhs <= rhs)
            objective.SetCoefficient(theta, 1.0)

        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            raise RuntimeError(f"stage LP not optimal, status={status}")

        # Duals above are w.r.t. the constraint RHS (phi_e*h_prev[e]); the
        # chain rule through that RHS gives the sensitivity w.r.t. h_prev[e].
        dual_h_prev = {e: phi_t[e] * bal_h[e].dual_value() for e in e_set}

        return StageResult(
            h={e: h[e].solution_value() for e in e_set},
            c=c.solution_value(),
            v=v.solution_value(),
            b={e: b[e].solution_value() for e in e_set},
            u={e: u[e].solution_value() for e in e_set},
            obj=objective.Value(),
            dual_h=dual_h_prev,
            dual_c=bal_c.dual_value(),
        )

    def solve_stage_robust(self, phi_t, h_prev, c_prev, cuts, terminal):
        """solve_stage with retries: if GLOP reports a non-optimal status on
        a near-degenerate cut stack, retry with a small relaxation on the
        cuts' right-hand side. As a last resort, drop the oldest cuts and
        retry -- this only ever discards an outer approximation of the
        value function (never a feasibility constraint), so the result
        stays a valid, just slightly less informed, solve rather than a
        crash.
        """
        last_err = None
        for relax in self.config.relax_schedule:
            try:
                return self.solve_stage(phi_t, h_prev, c_prev, cuts, terminal, _relax=relax)
            except RuntimeError as err:
                last_err = err
        for keep in (len(cuts) // 2, len(cuts) // 4, 0):
            try:
                return self.solve_stage(phi_t, h_prev, c_prev, cuts[-keep:] if keep else [],
                                         terminal, _relax=0.0)
            except RuntimeError as err:
                last_err = err
        raise last_err

    # ------------------------------------------------------------------ #
    # Cut management
    # ------------------------------------------------------------------ #

    def _build_cut(self, q_hat_raw, dual_h, dual_c, h_trial, c_trial):
        """Fold the trial point into the constant so solve_stage's cut
        constraint (`theta - pi_h.h - pi_c.C <= q_hat`) is exactly
        `theta <= q_hat_raw + pi.(x - xhat)`.
        """
        q_hat = q_hat_raw - sum(dual_h[e] * h_trial[e] for e in self.universe.etfs) - dual_c * c_trial
        return Cut(q_hat, dual_h, dual_c)

    def _is_duplicate_cut(self, candidate, existing):
        """Stacking many near-identical cuts (common in early iterations,
        before real cuts exist and trades are close to degenerate) makes
        GLOP's simplex fail numerically on an almost-singular constraint
        set. Skip a cut that duplicates one already present within a small
        tolerance.
        """
        cfg = self.config
        for cut in existing:
            if abs(candidate.q_hat - cut.q_hat) > cfg.cut_q_tol:
                continue
            if abs(candidate.pi_c - cut.pi_c) > cfg.cut_pi_tol:
                continue
            if any(abs(candidate.pi_h[e] - cut.pi_h[e]) > cfg.cut_pi_tol for e in self.universe.etfs):
                continue
            return True
        return False

    def _add_cut(self, stage, new_cut):
        stage_cuts = self.cuts.setdefault(stage, [])
        if self._is_duplicate_cut(new_cut, stage_cuts):
            return
        stage_cuts.append(new_cut)
        # Cap cuts per stage (FIFO): a long run otherwise keeps stacking
        # near-parallel cuts once the policy has converged, which is
        # exactly the degeneracy that makes GLOP fail numerically.
        if len(stage_cuts) > self.config.max_cuts_per_stage:
            self.cuts[stage] = stage_cuts[-self.config.max_cuts_per_stage:]

    # ------------------------------------------------------------------ #
    # SDDP loop (readme.md "Pseudo-algorithm")
    # ------------------------------------------------------------------ #

    def _statistical_bound(self, terminal_values):
        """Sample mean and standard error of the realized mean-CVaR
        objective (readme.md "Objective") at the current zeta, evaluated
        over the forward pass's simulated terminal wealth outcomes. This
        is the "statistical bound" in readme.md's "Pseudo-algorithm" --
        an estimate of the value actually achieved by the current policy,
        as opposed to the cuts' (optimistic) approximation of it.
        """
        cfg = self.config
        tv = np.asarray(terminal_values, dtype=float)
        eta = np.maximum(0.0, -tv - self.zeta)
        g = (1 - cfg.lambda_) * tv - cfg.lambda_ * (self.zeta + eta / (1 - cfg.alpha))
        mean = float(np.mean(g))
        se = float(np.std(g, ddof=1) / np.sqrt(len(g))) if len(g) > 1 else 0.0
        return mean, se

    def run(self, prices_hist, T, verbose=True):
        """Run the SDDP loop (readme.md "Pseudo-algorithm").

        All iteration counts and tolerances (n_outer, n_inner, n_forward,
        n_backward, gap_tolerance, zeta_tolerance) come from self.config --
        see config.py to change them.

        config.n_inner is a cap (k_max) rather than a fixed iteration
        count: each inner iteration checks the relative gap between the
        deterministic bound (the root-stage objective under the current
        cuts -- an optimistic over-approximation of the true value
        function) and the statistical bound (the sample mean of the
        policy's actually-realized objective over the forward pass -- see
        _statistical_bound). Once that gap closes to within
        config.gap_tolerance, the inner loop stops early; set
        gap_tolerance=None in config.py to always run the full n_inner
        iterations instead.

        config.n_outer is likewise a cap on the number of times zeta
        (readme.md's CVaR threshold) gets re-estimated from the realized
        terminal-wealth distribution. It stops early once the re-estimate
        barely moves from the value that was actually used to build the
        current cuts (within config.zeta_tolerance); set
        zeta_tolerance=None to always run the full n_outer iterations.
        Either way, self.zeta on return is always the value self.cuts was
        actually built for -- if the cap is hit before convergence, the
        most recent re-estimate is discarded rather than returned
        alongside cuts that don't match it.
        """
        cfg = self.config
        n_outer, n_inner = cfg.n_outer, cfg.n_inner
        n_forward, n_backward = cfg.n_forward, cfg.n_backward
        gap_tolerance, zeta_tolerance = cfg.gap_tolerance, cfg.zeta_tolerance

        e_set = self.universe.etfs
        returns = historical_log_returns(prices_hist, self.config.months_per_period)
        pe = self.universe.fetch_pe()
        self.rho = self.universe.eligibility(pe, self.config.pe_lower, self.config.pe_upper)
        if verbose:
            print("Trailing PE used:", {e: round(pe[e], 1) for e in e_set})
            print("Valuation-eligible:", self.rho)
            print(f"Bootstrapping from {len(returns)} historical "
                  f"{self.config.months_per_period}-month return observations "
                  f"({'thin -- treat results as illustrative' if len(returns) < 15 else 'ok'})")

        self.zeta = 0.0
        terminal_values = []

        for outer in range(n_outer):
            zeta_used = self.zeta  # the value this iteration's cuts are about to be built for

            # Every cut encodes value-function information for a *fixed*
            # zeta (it's baked into the terminal stage's objective offset,
            # see solve_stage). zeta changes at the end of each outer
            # iteration, which invalidates every cut built under the old
            # value -- so the cut set has to be rebuilt from scratch each
            # time zeta moves, rather than carried over.
            self.cuts = {t: [] for t in range(1, T)}  # cuts[t] approximates continuation value AT stage t

            for inner in range(n_inner):
                # ---- forward pass ----
                phi_paths = [simulate_phi_path(returns, T, self.rng) for _ in range(n_forward)]
                trial_states = []
                for phi_path in phi_paths:
                    h_val = {e: 0.0 for e in e_set}
                    c_val = self.config.c0
                    states = [(h_val, c_val)]
                    for t in range(1, T + 1):
                        phi_t = dict(zip(e_set, phi_path[t - 1]))
                        sol = self.solve_stage_robust(phi_t, h_val, c_val, self.cuts.get(t, []),
                                                       terminal=(t == T))
                        h_val, c_val = sol.h, sol.c
                        states.append((h_val, c_val))
                    trial_states.append(states)

                terminal_values = [sum(states[T][0].values()) + states[T][1] for states in trial_states]

                # ---- convergence check (readme.md "Pseudo-algorithm") ----
                deterministic_bound = self.recommend_action(T).obj
                statistical_mean, statistical_se = self._statistical_bound(terminal_values)
                gap = abs(deterministic_bound - statistical_mean) / max(abs(statistical_mean), 1e-9)
                if verbose:
                    print(f"outer {outer} inner {inner}: mean terminal value = "
                          f"{float(np.mean(terminal_values)):,.0f}  "
                          f"deterministic={deterministic_bound:,.0f} "
                          f"statistical={statistical_mean:,.0f}(+/-{1.96 * statistical_se:,.0f}) "
                          f"gap={gap:.1%}")
                if gap_tolerance is not None and gap <= gap_tolerance:
                    if verbose:
                        print(f"  converged: gap {gap:.1%} <= tolerance {gap_tolerance:.1%} "
                              f"after {inner + 1} inner iteration(s)")
                    break

                # ---- backward pass ----
                for t in range(T, 0, -1):
                    seen_states = set()
                    for phi_path, states in zip(phi_paths, trial_states):
                        h_prev, c_prev = states[t - 1]
                        # Many forward paths can land on the exact same
                        # state (e.g. whenever "no trade" is optimal, every
                        # path visits an identical trajectory). Building a
                        # cut per path in that case just stacks
                        # near-duplicate cuts at the same point, which is
                        # redundant and numerically fragile.
                        state_key = (tuple(round(h_prev[e], 2) for e in e_set), round(c_prev, 2))
                        if state_key in seen_states:
                            continue
                        seen_states.add(state_key)

                        vals, duals_h, duals_c = [], [], []
                        for phi_sample in sample_growth_factors(returns, n_backward, self.rng):
                            phi_t = dict(zip(e_set, phi_sample))
                            sol = self.solve_stage_robust(phi_t, h_prev, c_prev,
                                                           self.cuts.get(t, []), terminal=(t == T))
                            vals.append(sol.obj)
                            duals_h.append(sol.dual_h)
                            duals_c.append(sol.dual_c)
                        q_hat_raw = float(np.mean(vals))
                        dual_h_avg = {e: float(np.mean([d[e] for d in duals_h])) for e in e_set}
                        dual_c_avg = float(np.mean(duals_c))
                        if t > 1:
                            new_cut = self._build_cut(q_hat_raw, dual_h_avg, dual_c_avg, h_prev, c_prev)
                            self._add_cut(t - 1, new_cut)

            # ---- re-estimate zeta and check for outer convergence ----
            losses = [-tv for tv in terminal_values]
            zeta_candidate = float(np.quantile(losses, self.config.alpha))
            zeta_change = (abs(zeta_candidate - zeta_used) / max(abs(zeta_used), 1e-9)
                           if outer > 0 else None)
            if verbose:
                change_msg = f", change from this iteration's zeta: {zeta_change:.1%}" if zeta_change is not None else ""
                print(f"[outer {outer}] re-estimated zeta (VaR at {self.config.alpha:.0%}) = "
                      f"{-zeta_candidate:,.0f} AUD terminal wealth{change_msg}")

            converged = (zeta_tolerance is not None and zeta_change is not None
                         and zeta_change <= zeta_tolerance)
            if converged:
                if verbose:
                    print(f"  converged: zeta change {zeta_change:.1%} <= tolerance {zeta_tolerance:.1%} "
                          f"after {outer + 1} outer iteration(s)")
                break
            if outer == n_outer - 1:
                if verbose:
                    print(f"  reached n_outer={n_outer} cap without converging; keeping zeta="
                          f"{-zeta_used:,.0f} and its matching cuts rather than the latest re-estimate")
                break
            self.zeta = zeta_candidate

        return self.cuts, self.zeta, self.rho, terminal_values

    # ------------------------------------------------------------------ #
    # Recommended action
    # ------------------------------------------------------------------ #

    def recommend_action(self, T):
        """The concrete trade recommended right now: solve period 1's LP
        from the actual root state (all cash, readme.md: h_{e,0}=0,
        C_0=c0) using whatever cuts run() has learned for that stage.

        Because the incoming state is all-cash, the growth factor phi_t
        plays no role in the holdings-balance constraint here (phi_e * 0 =
        0 regardless of phi_e -- it only matters once there are existing
        holdings to carry forward), so a placeholder of 1.0 for every ETF
        is used; it does not affect the recommended trade.
        """
        e_set = self.universe.etfs
        h_prev = {e: 0.0 for e in e_set}
        c_prev = self.config.c0
        phi_placeholder = {e: 1.0 for e in e_set}
        return self.solve_stage_robust(phi_placeholder, h_prev, c_prev,
                                        self.cuts.get(1, []), terminal=(T == 1))
