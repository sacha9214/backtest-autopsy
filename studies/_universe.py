"""The instrument universe every study draws from.

Chosen before any result was seen, and deliberately spread across asset classes
and regions: picking instruments after noticing which ones cooperate is
selection bias at the level of the dataset, which no amount of deflation fixes.
"""

US_BROAD = ["SPY", "QQQ", "IWM", "DIA", "MDY", "VTI"]
US_SECTORS = ["XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB"]
INTERNATIONAL = ["EFA", "EEM", "EWJ", "EWG", "EWU", "EWZ", "EWY", "FXI"]
FIXED_INCOME = ["TLT", "IEF", "SHY", "LQD", "HYG", "TIP"]
COMMODITIES = ["GLD", "SLV", "USO", "DBA", "DBC"]
FACTORS = ["MTUM", "VLUE", "QUAL", "USMV", "SPLV"]
LARGE_CAPS = ["AAPL", "MSFT", "JNJ", "PG", "KO", "XOM", "GE", "IBM", "WMT", "JPM"]

UNIVERSE = (
    US_BROAD + US_SECTORS + INTERNATIONAL + FIXED_INCOME
    + COMMODITIES + FACTORS + LARGE_CAPS
)

GROUPS = {
    "US broad": US_BROAD,
    "US sectors": US_SECTORS,
    "International": INTERNATIONAL,
    "Fixed income": FIXED_INCOME,
    "Commodities": COMMODITIES,
    "Factors": FACTORS,
    "Large caps": LARGE_CAPS,
}


def group_of(ticker: str) -> str:
    for name, members in GROUPS.items():
        if ticker in members:
            return name
    return "other"
