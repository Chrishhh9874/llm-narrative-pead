"""Step 2: earnings-announcement events = original 8-K filings with Item 2.02 during S&P 500 membership.

Timing. The submissions API `acceptanceDateTime` is unreliable: against the EDGAR header
(ACCEPTANCE-DATETIME, US/Eastern) it is off by +4/+5/+8/+10 hours with no stable pattern (checked on
200 random events). We therefore read ACCEPTANCE-DATETIME from each filing's .hdr.sgml (cached).

day 0: acceptance before 16:00 ET on a trading day -> that day; otherwise the next trading day.
Trading calendar = SPY trading dates (yfinance).

Output: data/interim/events.csv
"""

import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from common import INTERIM, RAW, SAMPLE_END, SAMPLE_START, SecClient, load_submissions

ET = "America/New_York"


def trading_calendar():
    p = RAW / "prices" / "SPY.csv"
    if not p.exists():
        import yfinance as yf

        p.parent.mkdir(parents=True, exist_ok=True)
        df = yf.download("SPY", start="2013-01-01", end="2026-10-01", auto_adjust=False, progress=False)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.reset_index().to_csv(p, index=False)
    return pd.DatetimeIndex(pd.read_csv(p, parse_dates=["Date"]).Date).sort_values()




def header_acceptance(sec, cik, acc):
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{acc}.hdr.sgml"
    b = sec.get(url)
    m = re.search(rb"ACCEPTANCE-DATETIME>?:?\s*(\d{14})", b or b"")
    return pd.to_datetime(m.group(1).decode(), format="%Y%m%d%H%M%S") if m else pd.NaT


def assign_day0(accept_et, cal):
    d = accept_et.dt.normalize()
    after_close = accept_et.dt.hour >= 16
    is_td = d.isin(cal)
    # first trading day >= d, or > d when the filing came after the close
    pos = cal.searchsorted(d, side="left")
    pos = pos + (is_td & after_close).to_numpy().astype(int)
    pos = pos.clip(max=len(cal) - 1)
    return pd.Series(cal[pos], index=accept_et.index)


def main():
    sec = SecClient()
    u = pd.read_csv(INTERIM / "universe.csv", parse_dates=["start", "end"]).dropna(subset=["cik"])
    rows = []
    for cik, g in u.groupby("cik"):
        info, df = load_submissions(sec, cik)
        if info is None:
            continue
        df = df[(df.form == "8-K") & df["items"].fillna("").str.contains(r"\b2\.02\b")]
        for _, sp in g.iterrows():
            m = df[df.filingDate.between(sp.start, sp.end)]
            for _, f in m.iterrows():
                rows.append({"cik": int(cik), "ticker": sp.ticker, "accession": f.accessionNumber,
                             "filing_date": f.filingDate, "accept_api": f.acceptanceDateTime,
                             "items": f["items"], "primary_doc": f.primaryDocument})
    ev = pd.DataFrame(rows).drop_duplicates("accession")
    with ThreadPoolExecutor(4) as pool:  # SecClient's shared limiter keeps the global rate < 10/s
        ev["accept_et"] = list(pool.map(lambda r: header_acceptance(sec, r[0], r[1]),
                                        zip(ev.cik, ev.accession)))
    print("missing header acceptance:", int(ev.accept_et.isna().sum()))
    ev = ev.dropna(subset=["accept_et"])
    ev["day0"] = assign_day0(ev.accept_et, trading_calendar())
    ev["year"] = ev.day0.dt.year
    ev = ev[(ev.day0 >= SAMPLE_START) & (ev.day0 <= SAMPLE_END)].sort_values(["cik", "accept_et"])

    INTERIM.mkdir(parents=True, exist_ok=True)
    ev.to_csv(INTERIM / "events.csv", index=False)
    print(f"events={len(ev)} firms={ev.cik.nunique()}")
    print(ev.groupby("year").size().to_string())
    print("accept hour (ET) distribution:", ev.accept_et.dt.hour.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
