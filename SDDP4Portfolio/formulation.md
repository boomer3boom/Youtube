# ETF Portfolio Optimisation

This is a repo that stores my project on using SDDP to try and solve investment portfolio problem. I attempt to using CVAR and benchmark my solution against S and P 500 over turbulent and stable years. Below I outline the problem statement and the OR solution to this problem.

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
| $\mathcal{G}$ | Geographic regions / countries that ETF holdings can be attributed to, indexed by $g$ |
| $\mathcal{S}$ | Sectors / themes that ETF holdings can be attributed to, indexed by $s$ |

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
| $PE_{e,t}$ | Price-to-earnings ratio of ETF $e$ at review point $t$ |
| $\underline{PE}, \overline{PE}$ | Lower and upper bounds on an acceptable price-to-earnings ratio |

**Costs and limits**

| Symbol | Description |
|---|---|
| $\kappa^{buy}, \kappa^{sell}$ | Proportional transaction cost applied to the value bought / sold |
| $\overline{w}$ | Maximum permitted holding weight for any single ETF |
| $\overline{\theta}^{G}_{g}$ | Maximum permitted portfolio exposure to region $g$ |
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
and $\overline{\theta}^{S}_{s}$ respectively, at every review point.

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
| $\zeta$ | Value-at-Risk (VaR) threshold at confidence level $\alpha$ — the wealth level below which only the worst $1-\alpha$ fraction of outcomes fall. It is not fixed data; the optimiser solves for it alongside the trading decisions, and it anchors the CVaR term in the objective (auxiliary variable) |
| $\eta_\omega \ge 0$ | Shortfall of path $\omega$ below $\zeta$ (auxiliary variable) |
| $\theta_t$ | The estimated value for the benders cut $t$ |

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
outcomes — this is the lever used later to benchmark the "stable years"
vs "turbulent years" behaviour against the S&P 500.

*Note:* $\zeta$ is a single quantity shared by every path, not something
that varies period to period, so it is carried forward unchanged as an
extra component of the state $x_t$ (a trivial $\zeta_t = \zeta_{t-1}$
balance) purely so the terminal-stage subproblem can still reference it.
This keeps the recursion below valid, though it is worth being upfront
that a terminal CVaR handled this way is not fully *time-consistent* in
the dynamic-programming sense — a Markovian, per-stage (nested) risk
measure would be needed for that. For benchmarking a single fixed
horizon against the S&P 500 this simpler version is sufficient.

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

**Valuation gate** (cannot buy into an ETF trading outside the acceptable PE band):

$$
b_{e,t} \le M\, \rho_{e,t}
$$

together with $b_{e,t}, u_{e,t}, h_{e,t}, C_t \ge 0$ and, at $t=0$,
$h_{e,0}$ and $C_0$ fixed to the investor's actual starting portfolio.

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

subject to the balance, exposure, valuation and cap constraints above,
plus cuts approximating the expected value-to-go (that is, we need to apply the t+1 cut onto t)
$\mathcal{Q}_{t+1}(x_t) = \mathbb{E}_{\phi_{t+1}}[\,Q_{t+1}(x_t, \phi_{t+1})\,]$:

$$
\theta_t \le \hat{Q}_{t+1}^{(k)} + \bar{\pi}_{t+1}^{(k)} \cdot \left(x_t - \hat{x}_t^{(k)}\right), \qquad k = 1,\dots,K
$$

Each cut $k$ is built by fixing the incoming state of the period-$(t+1)$
problem to a previously visited trial point $\hat{x}_t^{(k)}$ (treating
$h_{e,t}=\hat{h}_{e,t}^{(k)}$ and $C_t=\hat{C}_t^{(k)}$ as constraints
rather than free variables), solving it for a sample of growth-factor
realisations $\phi_{t+1} \in \Omega_{t+1}$, and reading off:

- $\hat{Q}_{t+1}^{(k)}$ — the sample-average optimal objective value, and
- $\bar{\pi}_{t+1}^{(k)}$ — the sample-average dual price (shadow price)
  on the state-fixing constraints.

Because $\mathcal{Q}_{t+1}$ is concave and piecewise-linear in the state
(a property inherited from LP duality), every such cut is a valid outer
approximation — $\theta_t$ can never exceed the true value-to-go, and the
approximation tightens monotonically as more cuts accumulate near the
states the policy actually visits.

### Pseudo-algorithm

```
initialise: cut sets Θ_t ← ∅ for t = 1..T
initialise: x_0 ← investor's actual starting holdings and cash
k ← 0

repeat
    k ← k + 1

    # ---- forward pass ----
    sample a set of growth-factor paths {ω} over t = 1..T
    for each sampled path ω:
        for t = 1..T:
            solve stage-t problem Q_t(x_{t-1}(ω), φ_t(ω)) using
                the current cuts Θ_t as an approximation of θ_t
            record trial state  x_t(ω)  and stage contribution
    compute a statistical estimate (mean ± CI) of total objective
        across sampled paths            -> candidate solution bound
    read the root-stage objective (t = 0, with all current cuts)
                                         -> deterministic bound

    # ---- convergence check ----
    if gap(deterministic bound, statistical bound) ≤ ε or k = k_max:
        break

    # ---- backward pass ----
    for t = T down to 1:
        for each distinct trial state x_{t-1}(ω) visited above:
            for each growth-factor realisation φ_t in Ω_t (all, or a sample):
                fix incoming state to x_{t-1}(ω)
                solve stage-t problem -> optimal value, dual prices
            average the values and duals across realisations
            build one new cut from the averages
            add the cut to Θ_{t-1}

until convergence

output: cut sets {Θ_t} defining the value-to-go approximation
output: simulate forward once more with fixed cuts (in-sample and
         out-of-sample price paths, including the S&P 500 comparison
         window) to obtain realised buy/sell/hold decisions and
         portfolio performance
```
