# ETF Portfolio Optimisation — Mathematical Formulation

The mathematical model behind this project: the problem statement, sets,
data, variables, the mean-CVaR objective, every constraint, the cut-based
value-function approximation, and a pseudo-algorithm for solving it with
Stochastic Dual Dynamic Programming (SDDP).

Companion documents:

- [readme.md](readme.md) — how to set up and run the code, and what it prints.
- [documentation.md](documentation.md) — how the code actually solves and
  validates this model: the bootstrap, the cut machinery, the outer/inner
  loops, the convergence tests, and where the implementation departs from
  this document.

Code in `Solution/` uses the same symbols as this document (`b`, `u`, `h`,
`C`, `V`, `zeta`, `eta`, `rho`, `gamma`, `phi`, ...), and comments in
`Solution/sddp.py` name the constraint each block implements (e.g.
"Holdings balance"), so the two can be read side by side.

## Problem Overview

An investor holds a portfolio made up of a fixed universe of exchange-traded
funds (ETFs) — a mix of broad-market, thematic, and regional funds, some
listed on the ASX and others listed offshore, so unit prices are not all
denominated in the same currency. At regular intervals the investor
reviews the portfolio and must decide, for each ETF, whether to buy more
units, sell part of the existing holding, or leave the position unchanged,
and — just as importantly — what percentage of total portfolio value
should end up sitting in that ETF once the decision is acted on.

This is harder than a single "pick the best weights" allocation problem
for a few reasons:

- **Prices move.** The price of each ETF changes from one review point to
  the next, and future prices are not known with certainty at the time a
  buy or sell decision is made.
- **Trading is not free.** Buying and selling incurs transaction costs, so
  it is not optimal to fully rebalance to a theoretical "ideal" weighting
  every period — the benefit of moving closer to the ideal has to outweigh
  the cost of the trade itself.
- **Holdings are constrained, not just optimised.** No single ETF may
  dominate the portfolio, and because ETFs overlap in what they actually
  hold underneath (the same large technology stock might sit inside
  several different funds), the portfolio is also constrained on its
  *look-through* exposure — by geography and by sector/theme — not only
  on the face-value weight of each fund.
- **Valuation matters, not just price.** Some ETFs are cheap or expensive
  relative to their earnings at a point in time, and the investor wants to
  avoid buying into funds sitting outside an acceptable valuation band.
- **The same pre-tax return isn't worth the same to every ETF.** The
  investor is an Australian resident taxpayer, so dividends from
  Australian companies carry a franking credit (a refund of the 30%
  company tax already paid) that dividends from an ETF's foreign holdings
  don't. An ETF whose look-through exposure is more Australian is worth
  more, after tax, than an equally-performing ETF exposed elsewhere —
  purely because of who is holding it, not because of anything about the
  fund itself.

The problem, then, is to decide a sequence of buy/sell/hold actions and
resulting holding percentages for each ETF over time, so that the
portfolio's expected value is maximised (or its risk is controlled) net
of transaction costs, while respecting single-holding, geographic,
sector, and valuation limits at every review point.

## Sets

| Set | Description |
|---|---|
| $\mathcal{E}$ | Universe of investable ETFs, indexed by $e$ |
| $\mathcal{T} = \{0, 1, \dots, T\}$ | Review points (time periods) over the investment horizon, indexed by $t$ |
| $\mathcal{G}$ | Geographic regions (continents: North America, South America, Europe, Asia, Africa, Australia, plus an "Other" bucket for holdings a fund's published breakdown doesn't attribute) that ETF holdings can be attributed to, indexed by $g$. In the code, $\mathcal{G}$ is whichever regions appear in the universe's look-through data |
| $\mathcal{S}$ | Sectors / themes that ETF holdings can be attributed to, indexed by $s$ (in the code, the sector names Yahoo Finance publishes for each fund) |

## Data

**Prices and cash**

| Symbol | Description |
|---|---|
| $P_{e,t}$ | Unit price of ETF $e$ at review point $t$ (uncertain for $t>0$) |
| $\phi_{e,t} = P_{e,t} / P_{e,t-1}$ | Period growth factor for ETF $e$, derived from $P_{e,t}$ |
| $C_{0}$ | Initial cash available to invest, at $t=0$ |

**Look-through exposure**

| Symbol | Description |
|---|---|
| $L^{G}_{e,g} \in [0,1]$ | Proportion of ETF $e$'s underlying holdings attributed to region $g$, with $\sum_{g \in \mathcal{G}} L^{G}_{e,g} = 1$ for every $e$ |
| $L^{S}_{e,s} \in [0,1]$ | Proportion of ETF $e$'s underlying holdings attributed to sector/theme $s$, with $\sum_{s \in \mathcal{S}} L^{S}_{e,s} = 1$ for every $e$ |

**Valuation**

| Symbol | Description |
|---|---|
| $PE_{e,t}$ | Price-to-earnings ratio of ETF $e$ at review point $t$ (in the code, today's trailing PE is fetched once and held constant over the horizon, since no historical PE series is readily available) |
| $\underline{PE}, \overline{PE}$ | Lower and upper bounds on an acceptable price-to-earnings ratio |

**Costs and limits**

| Symbol | Description |
|---|---|
| $\kappa^{buy}, \kappa^{sell}$ | Proportional transaction cost applied to the value bought / sold |
| $\overline{w}$ | Maximum permitted holding weight for any single ETF |
| $\overline{\theta}^{G}_{g}$ | Maximum permitted portfolio exposure to region $g$ |
| $\underline{\theta}^{G}_{g}$ | Minimum required portfolio exposure to region $g$ (zero, i.e. no requirement, for any region not given one) |
| $\overline{\theta}^{S}_{s}$ | Maximum permitted portfolio exposure to sector/theme $s$ |
| $\gamma^{AU}$ | Average per-period franking credit yield on Australian equities (e.g. dividend yield $\times$ franking level $\times \frac{t_c}{1-t_c}$ at the $t_c=30\%$ company tax rate) |
| $\gamma_{e} = L^{G}_{e,\text{Australia}} \cdot \gamma^{AU}$ | Effective franking credit yield of ETF $e$, derived from how much of its look-through exposure is Australian |

## Parameters

| Symbol | Description |
|---|---|
| $\alpha \in (0,1)$ | CVaR confidence level (e.g. 0.95) |
| $\lambda \in [0,1]$ | Risk-aversion weight trading off expected wealth against CVaR |
| $\rho_{e,t} \in \{0,1\}$ | Valuation eligibility flag, $\rho_{e,t}=1 \iff \underline{PE} \le PE_{e,t} \le \overline{PE}$ |
| $M$ | A big-M constant, large relative to any feasible trade value |
| $p_\omega$ | Probability of sample path $\omega$ |

## Variables

### Decision variables

At each review point $t$, and for each ETF $e$, the problem chooses:

| Symbol | Description |
|---|---|
| $b_{e,t} \ge 0$ | Value of ETF $e$ bought at review point $t$ |
| $u_{e,t} \ge 0$ | Value of ETF $e$ sold at review point $t$ |
| $w_{e,t} \in [0,1]$ | Resulting holding weight of ETF $e$ — the percentage of total portfolio value held in $e$ — after acting on $b_{e,t}$ and $u_{e,t}$ |

with $w_{e,t}$ bounded above by $\overline{w}$, and the region/sector
look-through exposures $\sum_{e} w_{e,t} L^{G}_{e,g}$ and
$\sum_{e} w_{e,t} L^{S}_{e,s}$ bounded above by $\overline{\theta}^{G}_{g}$
and $\overline{\theta}^{S}_{s}$ respectively, and the region exposures
bounded below by $\underline{\theta}^{G}_{g}$, at every review point.

### State and auxiliary variables

The buy/sell decisions at $t$ change what is carried into $t+1$, so the
problem needs an explicit *state* — what is held going into a review
point — separate from the *flow* decisions $b_{e,t}$ and $u_{e,t}$ made
at that review point.

| Symbol | Description |
|---|---|
| $h_{e,t} \ge 0$ | **Dollar value** held in ETF $e$ at the end of period $t$ (state variable) |
| $C_t \ge 0$ | Cash balance at the end of period $t$ (state variable) |
| $V_t = C_t + \sum_{e \in \mathcal{E}} h_{e,t}$ | Total portfolio value at $t$ |
| $x_t = (h_{\cdot,t}, C_t)$ | State vector carried from period $t$ into period $t+1$ |
| $\zeta$ | Value-at-Risk (VaR) threshold at confidence level $\alpha$, expressed as a **loss** (negative wealth): only the worst $1-\alpha$ fraction of outcomes have a loss $-W_T$ above $\zeta$, so the VaR as a wealth level is $-\zeta$. It is not fixed data; the optimiser solves for it alongside the trading decisions, and it anchors the CVaR term in the objective (auxiliary variable). The code finds it by fixed-point iteration instead — see the note under "Objective" |
| $\eta_\omega \ge 0$ | Shortfall of path $\omega$: how far its loss exceeds $\zeta$ (auxiliary variable) |
| $\theta_t$ | Approximation of the expected value-to-go from the end of period $t$ onwards, bounded above by the Benders cuts (see "Value function and cuts"). Not related to the look-through limits $\overline{\theta}^{G}_{g}$, $\overline{\theta}^{S}_{s}$, which share the letter |

$h_{e,t}$ is deliberately carried as **dollar value** rather than units.
The two are equivalent, but dollar value keeps the continuation value a
function of the state alone — independent of the price level, which a
units-based state is not, once returns are modelled multiplicatively —
and keeps every stage's cut coefficients on a comparable scale, which
matters for numerical stability once cuts accumulate. The holding weight
introduced earlier is now just $w_{e,t} = h_{e,t} / V_t$, which stays
linear in the decisions since it is only ever used as $h_{e,t} \le
\overline{w}\, V_t$, never formed as an explicit ratio.

## Solution

### Objective

Terminal wealth on path $\omega$ is $W_T(\omega) = C_T(\omega) + \sum_e h_{e,T}(\omega)$.
The investor maximises a blend of expected terminal wealth and its
conditional value-at-risk, using the standard Rockafellar–Uryasev
linearisation of CVaR:

$$
\max \;\; (1-\lambda)\, \mathbb{E}[W_T] \; - \; \lambda \left( \zeta + \frac{1}{1-\alpha} \sum_{\omega} p_\omega\, \eta_\omega \right)
$$

$$
\text{s.t.} \quad \eta_\omega \ge -W_T(\omega) - \zeta, \qquad \eta_\omega \ge 0 \qquad \forall \omega
$$

$\lambda = 0$ recovers a purely expected-wealth objective; $\lambda \to 1$
drives the policy towards protecting the worst $1-\alpha$ tail of
outcomes. This is the main lever for how defensive the policy is
(`lambda_` in `Solution/config.py`).

*Note:* $\zeta$ is a single quantity shared by every path, not something
that varies period to period, so it is carried forward unchanged as an
extra component of the state $x_t$ (a trivial $\zeta_t = \zeta_{t-1}$
balance) purely so the terminal-stage subproblem can still reference it.
This keeps the recursion below valid, though it is worth being upfront
that a terminal CVaR handled this way is not fully *time-consistent* in
the dynamic-programming sense — a Markovian, per-stage (nested) risk
measure would be needed for that. For evaluating a single fixed horizon
this simpler version is sufficient.

*Implementation note:* the code does not carry $\zeta$ as a state
variable. It fixes $\zeta$ for an entire SDDP solve, then re-estimates it
as the empirical $\alpha$-quantile of simulated losses and solves again,
until $\zeta$ stops moving (the "outer loop" in the pseudo-algorithm
below). Each stage stays a plain LP; the cost is that every cut is only
valid for the $\zeta$ it was built under, so cuts are discarded whenever
$\zeta$ changes. See [documentation.md](documentation.md).

### Constraints

For every period $t \in \mathcal{T}\setminus\{0\}$ and every ETF $e \in \mathcal{E}$:

**Holdings balance** (last period's value is carried forward by the
period's growth factor, then adjusted by trades):

$$
h_{e,t} = \phi_{e,t}\, h_{e,t-1} + b_{e,t} - u_{e,t}
$$

**Cash balance** (buying costs $\kappa^{buy}$ extra, selling nets $\kappa^{sell}$ less,
and last period's holdings pay out a franking credit $\gamma_e$ in cash):

$$
C_t = C_{t-1} + \sum_{e} h_{e,t-1}\,\gamma_e - \sum_{e} b_{e,t}\left(1+\kappa^{buy}\right) + \sum_{e} u_{e,t}\left(1-\kappa^{sell}\right)
$$

The franking credit is earned on what was *held over* the period (hence
$h_{e,t-1}$, not $h_{e,t}$), and is separate from — additive to — the
ETF's own price return: it doesn't change $\phi_{e,t}$ or the holdings
balance above, only how much cash comes back each period. $\gamma_e$ is
zero for any ETF with no Australian look-through exposure, so this term
only ever benefits IOZ.ASX and, to a lesser extent, DHHF.ASX in this
project's universe.

**Single-holding cap:**

$$
h_{e,t} \le \overline{w}\, V_t
$$

**Geographic and sector look-through limits:**

$$
\sum_{e} h_{e,t}\, L^{G}_{e,g} \le \overline{\theta}^{G}_{g}\, V_t \qquad \forall g \in \mathcal{G}
$$

$$
\sum_{e} h_{e,t}\, L^{S}_{e,s} \le \overline{\theta}^{S}_{s}\, V_t \qquad \forall s \in \mathcal{S}
$$

**Geographic minimums** (guarantee exposure to chosen regions, rather than
only capping it):

$$
\sum_{e} h_{e,t}\, L^{G}_{e,g} \ge \underline{\theta}^{G}_{g}\, V_t \qquad \forall g \in \mathcal{G}
$$

Only regions given a positive $\underline{\theta}^{G}_{g}$ get this
constraint. Since $V_t$ includes cash, a minimum also limits how much the
policy can sit in cash. The constraint involves only this period's
$h_{e,t}$ and $V_t$, never the incoming state, so it doesn't add a term
to any cut subgradient. It can make a stage infeasible: if the only ETFs
with exposure to $g$ fail the valuation gate, or the holding and region
caps leave too little room, no allocation satisfies every minimum. Selling
is unrestricted and every limit scales with $V_t$, so if the all-cash
starting state can satisfy the minimums, every later state can too, and
one check at $t=1$ is enough.

**Valuation gate** (cannot buy into an ETF trading outside the acceptable PE band):

$$
b_{e,t} \le M\, \rho_{e,t}
$$

together with $b_{e,t}, u_{e,t}, h_{e,t}, C_t \ge 0$ and, at $t=0$,
$h_{e,0}$ and $C_0$ fixed to the investor's actual starting portfolio.
(The code currently always starts from all cash: $h_{e,0}=0$,
$C_0 = $ `c0`.)

No explicit constraint is needed to stop the model from buying and
selling the same ETF in the same period — with $\kappa^{buy}, \kappa^{sell} > 0$,
simultaneous $b_{e,t}>0$ and $u_{e,t}>0$ is always cost-dominated by
netting the two trades, so an optimal solution never does it. This
matters beyond tidiness: it keeps every stage subproblem a linear
program, which is what the duality argument for the cuts below relies on
— introducing binary buy/sell indicators would turn each stage into a
mixed-integer program and invalidate that argument.

### Value function and cuts

Write the period-$t$ problem as a function of the state inherited from
period $t-1$ and the growth-factor realisation at $t$:

$$
Q_t(x_{t-1}, \phi_t) = \max_{b_t,\, u_t,\, h_t,\, C_t} \Big\{ \text{(period-$t$ contribution)} + \theta_t \Big\}
$$

The period-$t$ contribution is zero for $t < T$: nothing is consumed along
the way, so every intermediate stage simply maximises $\theta_t$. At
$t = T$ there is no $\theta_T$; the contribution is the objective above
evaluated on this path, $(1-\lambda) W_T - \lambda\left(\zeta + \frac{\eta}{1-\alpha}\right)$.

The maximisation is subject to the balance, exposure, valuation and cap
constraints above, plus cuts approximating the expected value-to-go
$\mathcal{Q}_{t+1}(x_t) = \mathbb{E}_{\phi_{t+1}}[\,Q_{t+1}(x_t, \phi_{t+1})\,]$.
Cuts built from period $t+1$'s problem are added to period $t$'s problem:

$$
\theta_t \le \hat{Q}_{t+1}^{(k)} + \bar{\pi}_{t+1}^{(k)} \cdot \left(x_t - \hat{x}_t^{(k)}\right), \qquad k = 1,\dots,K
$$

Each cut $k$ is built by fixing the incoming state of the period-$(t+1)$
problem to a previously visited trial point $\hat{x}_t^{(k)}$ (treating
$h_{e,t}=\hat{h}_{e,t}^{(k)}$ and $C_t=\hat{C}_t^{(k)}$ as constraints
rather than free variables), solving it for **every** growth-factor
realisation $\phi_{t+1} \in \Omega_{t+1}$, and reading off:

- $\hat{Q}_{t+1}^{(k)}$ — the probability-weighted average optimal
  objective value, and
- $\bar{\pi}_{t+1}^{(k)}$ — the probability-weighted average dual price
  (shadow price) on the state-fixing constraints.

Because $\mathcal{Q}_{t+1}$ is concave and piecewise-linear in the state
(a property inherited from LP duality), every such cut is a valid outer
approximation — it lies on or above $\mathcal{Q}_{t+1}$ everywhere, so
$\theta_t$ (the minimum over all cuts) can never fall *below* the true
value-to-go, and the approximation tightens monotonically as more cuts
accumulate near the states the policy actually visits.

That validity depends on $\hat{Q}$ and $\bar{\pi}$ being the *exact*
expectation over $\Omega_{t+1}$. Estimating them from a random sample of
$\Omega_{t+1}$ instead gives noisy cuts, and since $\theta_t$ takes the
minimum over cuts, the ones whose noise happened to be low are the ones
that bind. That biases the approximation downward, the bias compounds back
through every stage, and the resulting bound can sit below the value the
policy actually achieves. Here $\Omega_{t+1}$ is the historical bootstrap
pool (one joint growth-factor vector per overlapping rolling window, each
equally likely, identical at every stage), which is small enough to
enumerate in full.

### Pseudo-algorithm

This is the algorithm as implemented in `SDDPSolver.run()`
(`Solution/sddp.py`). Snake-case names (`n_outer`, `gap_ci_z`, `c0`, ...)
are fields of `ModelConfig` (`Solution/config.py`). The outer loop exists only because of how the code
handles $\zeta$ (see the implementation note under "Objective"). The inner
loop is standard SDDP.

```
Ω     ← every joint growth-factor vector in the historical bootstrap pool
         (one per overlapping rolling window, equally likely, same at every t)
x_0   ← all cash: h_{e,0} = 0, C_0 = c0
ζ     ← 0

for outer = 0 .. n_outer-1:                          # ---- outer loop: fix ζ ----
    n_inner, n_forward ← ramped from *_start to *_end as outer increases
    Θ_t ← ∅ for every t                              # cuts are only valid for one ζ

    for inner = 0 .. n_inner-1:                      # ---- inner loop: SDDP ----

        # forward pass
        draw n_forward paths, each T growth-factor vectors sampled from Ω
                with replacement
        for each path ω, for t = 1..T:
            solve stage-t LP from x_{t-1}(ω) with φ_t(ω) and cuts Θ_t
            record trial state x_t(ω)
        statistical bound   ← mean ± z·SE of the realised objective
                              (1-λ)W_T − λ(ζ + η/(1−α)) over the n_forward paths
        deterministic bound ← objective of the period-1 LP solved from x_0
                              with cuts Θ_1 (an upper bound if the cuts are valid)

        # convergence check (z = gap_ci_z)
        if deterministic bound ≤ statistical mean + z·SE:
            break                                    # gap is within sampling noise

        # backward pass
        for t = T down to 2:
            for each distinct trial state x_{t-1}(ω) from the forward pass:
                for every φ in Ω (enumerate, never sample):
                    solve stage-t LP from x_{t-1}(ω) with φ and cuts Θ_t
                    -> optimal value, duals on the incoming state
                average values and duals over Ω -> one cut
                add the cut to Θ_{t-1} (skip near-duplicates)

    # re-estimate ζ
    ζ_sample ← α-quantile of the losses −W_T from the last forward pass
    ζ_new    ← running average of ζ_sample over outer iterations,
               weighted by each iteration's n_forward
    if outer > 0 and |ζ_new − ζ| / |ζ| ≤ zeta_tolerance: break
    if this is the last outer iteration:  break      # keep ζ that matches Θ
    ζ ← ζ_new

output: ζ and the cut sets Θ_t built under it
output: terminal-wealth distribution of the last forward pass, compared with
        a buy-and-hold position in benchmark_ticker over the same horizon
output: today's recommended trade = the period-1 LP solved from x_0
        with Θ_1, explained in plain English by PortfolioExplainer
```
