"""The ETF universe: tickers, currency handling, look-through exposure data,
and price/PE retrieval.

Everything here corresponds to readme.md's sets and data: E (ETFs), G
(regions), S (sectors/themes), L_G, L_S and PE. Centralising it in one
class (rather than a handful of module-level dicts) is what lets
formulation.py and sddp.py both refer to "the universe" instead of each
importing a pile of loose constants.
"""

import pandas as pd
import yfinance as yf


class ETFUniverse:
    """readme.md: E, G, S, L_G, L_S, PE."""

    def __init__(self, ticker_map, usd_tickers, geo, pe_fallback):
        self.ticker_map = ticker_map    # label -> yfinance ticker
        self.usd_tickers = usd_tickers  # labels priced in USD, converted to AUD
        self.geo = geo                  # readme.md: L^G_{e,g}, label -> {region: weight}
        self.pe_fallback = pe_fallback  # label -> fallback PE if the live fetch fails
        self.sector = {}                # readme.md: L^S_{e,s}, populated by fetch_sector_weightings()

    @property
    def etfs(self):
        """readme.md: E, the universe of investable ETFs."""
        return list(self.ticker_map.keys())

    @property
    def regions(self):
        """readme.md: G, geographic regions ETF holdings look through to."""
        return sorted({g for exposure in self.geo.values() for g in exposure})

    @property
    def sectors(self):
        """readme.md: S, sectors/themes ETF holdings look through to."""
        return sorted({s for exposure in self.sector.values() for s in exposure})

    def franking_credit_yields(self, gamma_au_per_period):
        """readme.md: gamma_e = L^G_{e,Australia} * gamma^AU.

        Franking credits are only worth something to an Australian
        resident taxpayer, and only accrue on the Australian-equity slice
        of a fund's underlying holdings -- so this reuses the geographic
        look-through weight already in `geo` rather than needing a new
        per-ETF data table. IOZ.ASX (100% Australian) gets the full
        gamma^AU; DHHF.ASX gets roughly its ~37% Australian weight's worth;
        everything else (IVV.ASX, NDQ.ASX, IJP.ASX, IEM.ASX, BRK-B) gets
        ~0, since none of their look-through exposure is Australian.
        """
        return {e: self.geo[e].get("Australia", 0.0) * gamma_au_per_period for e in self.etfs}

    def fetch_prices(self, period="8y"):
        """readme.md: P_{e,t}, all converted to a single currency (AUD)."""
        yf_tickers = list(dict.fromkeys(list(self.ticker_map.values()) + ["AUDUSD=X"]))
        raw = yf.download(yf_tickers, period=period, interval="1mo", progress=False)["Close"]
        raw = raw.dropna(how="all").ffill().dropna()
        fx = raw["AUDUSD=X"]  # USD per AUD
        prices = pd.DataFrame(index=raw.index)
        for label, ticker in self.ticker_map.items():
            prices[label] = raw[ticker] / fx if label in self.usd_tickers else raw[ticker]
        return prices.dropna()

    def fetch_sector_weightings(self):
        """readme.md: L^S_{e,s}, fetched live from yfinance's fund sector
        breakdown (`Ticker.funds_data.sector_weightings`). Unlike the
        geographic look-through data (L_G, still hand-typed -- Yahoo simply
        doesn't expose a country/region breakdown for these funds), sector
        weightings *are* published live, so there's no need to guess.

        BRK-B isn't a fund (it's Berkshire Hathaway stock), so it has no
        `funds_data`; it falls back to its single GICS sector from
        `Ticker.info["sector"]`, weighted 1.0.
        """
        sector = {}
        for label, ticker in self.ticker_map.items():
            try:
                weights = yf.Ticker(ticker).funds_data.sector_weightings
                if not weights:
                    raise ValueError("empty sector_weightings")
                sector[label] = dict(weights)
            except Exception:
                slug = "unknown"
                try:
                    raw_sector = yf.Ticker(ticker).info.get("sector")
                    if raw_sector:
                        slug = raw_sector.strip().lower().replace(" ", "_")
                except Exception:
                    pass
                sector[label] = {slug: 1.0}
        self.sector = sector
        return sector

    def fetch_pe(self):
        """readme.md: PE_{e,t}, fetched live but held constant across the
        horizon since a genuine historical PE time series isn't readily
        available for these tickers.
        """
        pe = {}
        for label, ticker in self.ticker_map.items():
            try:
                info = yf.Ticker(ticker).info
                value = info.get("trailingPE") or info.get("forwardPE")
                pe[label] = float(value) if value else self.pe_fallback[label]
            except Exception:
                pe[label] = self.pe_fallback[label]
        return pe

    def eligibility(self, pe, pe_lower, pe_upper):
        """readme.md: RHO_{e,t} = 1 iff PE-underline <= PE_{e,t} <= PE-overline."""
        return {e: 1.0 if pe_lower <= pe[e] <= pe_upper else 0.0 for e in self.etfs}

    @classmethod
    def default(cls):
        """The seven-ETF universe used throughout this project: IVV.ASX,
        IOZ.ASX, NDQ.ASX, DHHF.ASX, BRK-B, IJP.ASX, IEM.ASX.

        NDQ.ASX and IEM.ASX replace this universe's original U100/BEMG.ASX
        picks: U100.AX only listed in August 2023 and BEMG.AX in September
        2025, both of which capped the joint overlapping history available
        for scenario bootstrapping to little more than a year. NDQ.ASX
        (listed 2015) and IEM.ASX (listed 2008) push the binding constraint
        out to DHHF.ASX's January 2020 listing instead -- about 81 months
        of overlap, enough for a real 12-month-period bootstrap.

        Geographic look-through weights (geo, below) are indicative, not
        sourced from a live factsheet feed -- Yahoo doesn't expose a
        country/region breakdown for these funds. Figures for IOZ.ASX,
        DHHF.ASX and IEM.ASX are reused from
        Stock-Stochastic/CVar/StockWithCVar.py elsewhere in this repo
        (which already used these same tickers); the rest are rough
        estimates from each fund's known style/region. Sector look-through
        data, by contrast, is fetched live -- see fetch_sector_weightings().
        BRK-B trades in USD and is converted to AUD each period via the
        AUDUSD=X spot rate.
        """
        ticker_map = {
            "IVV.ASX": "IVV.AX",
            "IOZ.ASX": "IOZ.AX",
            "NDQ.ASX": "NDQ.AX",
            "DHHF.ASX": "DHHF.AX",
            "BRK-B": "BRK-B",
            "IJP.ASX": "IJP.AX",
            "IEM.ASX": "IEM.AX",
        }
        usd_tickers = {"BRK-B"}
        geo = {
            "IVV.ASX": {"USA": 1.0},
            "IOZ.ASX": {"Australia": 1.0},
            "NDQ.ASX": {"USA": 1.0},
            "DHHF.ASX": {"USA": 0.424, "Australia": 0.37, "Japan": 0.038, "China": 0.018,
                         "Canada": 0.018, "Britain": 0.016, "India": 0.014, "Taiwan": 0.014,
                         "Germany": 0.011},
            "BRK-B": {"USA": 1.0},
            "IJP.ASX": {"Japan": 1.0},
            "IEM.ASX": {"China": 0.2766, "Taiwan": 0.1969, "India": 0.1939, "Korea": 0.09,
                        "Saudi Arabia": 0.04, "Brazil": 0.04, "South Africa": 0.0295,
                        "Mexico": 0.0179, "Malaysia": 0.0148, "Indonesia": 0.0146, "Thailand": 0.0142},
        }
        pe_fallback = {"IVV.ASX": 25.0, "IOZ.ASX": 20.0, "NDQ.ASX": 28.0, "DHHF.ASX": 22.0,
                       "BRK-B": 15.0, "IJP.ASX": 18.0, "IEM.ASX": 16.0}
        return cls(ticker_map, usd_tickers, geo, pe_fallback)
