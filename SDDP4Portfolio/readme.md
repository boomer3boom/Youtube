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

## Decisions

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
