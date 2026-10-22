"""Step 5: align EDGAR events with the Kaggle (Zacks-sourced) earnings file, 2015 - 2021H1.

Match: same ticker, Kaggle announcement date within [-3, +3] calendar days of the EDGAR acceptance
date (nearest wins). Reports
  * date agreement (exact / within 1 day) and implied day-0 agreement where Kaggle has release_time
    (pre -> same day, post -> next trading day);
  * analyst SUE = (eps - eps_est) / P, P = unadjusted close on the last trading day before day 0.
    Kaggle EPS is on the as-announced share basis (e.g. AAPL 2019 = pre-2020-split values), so the
    unadjusted close is the consistent deflator. Zacks EPS is "street" (adjusted) EPS and is never
    mixed with XBRL GAAP EPS.

Output: data/interim/events_kaggle.csv (events_sue.csv + kaggle columns)
"""

import argparse

import numpy as np
import pandas as pd

from common import INTERIM, RAW

KAGGLE = RAW / "kaggle_tsaustin" / "stocks_latest"
KAGGLE_END = pd.Timestamp("2021-06-14")


def load_prices(tickers):
    """Unadjusted closes for the needed tickers only (the file has 24M rows; read in chunks)."""
    parts = []
    for ch in pd.read_csv(KAGGLE / "stock_prices_latest.csv", usecols=["symbol", "date", "close"],
                          chunksize=2_000_000):
        parts.append(ch[ch.symbol.isin(tickers)])
    p = pd.concat(parts)
    p["date"] = pd.to_datetime(p.date)
    return p.sort_values(["symbol", "date"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=str(INTERIM / "events_sue.csv"))
    ap.add_argument("--out", default=str(INTERIM / "events_kaggle.csv"))
    args = ap.parse_args()

    ev = pd.read_csv(args.events, parse_dates=["accept_et", "day0", "qend"])
    ev["kticker"] = ev.ticker.str.replace(".", "-", regex=False)
    sub = ev[ev.day0 <= KAGGLE_END].copy()
    sub["adate"] = sub.accept_et.dt.normalize()

    k = pd.read_csv(KAGGLE / "earnings_latest.csv", parse_dates=["date"])
    k = k[k.symbol.isin(sub.kticker.unique())].rename(columns={"date": "k_date", "symbol": "kticker"})
    m = sub[["accession", "kticker", "adate", "day0"]].merge(k, on="kticker")
    m["gap"] = (m.k_date - m.adate).dt.days
    m = m[m.gap.abs() <= 3].assign(absgap=lambda d: d.gap.abs()).sort_values(["accession", "absgap"])
    m = m.drop_duplicates("accession").drop_duplicates(["kticker", "k_date"])

    cal = pd.DatetimeIndex(pd.read_csv(RAW / "prices" / "SPY.csv", parse_dates=["Date"]).Date)
    # pre -> first trading day >= date; post -> first trading day > date
    pos = np.where(m.release_time.eq("post"), cal.searchsorted(m.k_date, "right"), cal.searchsorted(m.k_date, "left"))
    m["k_day0"] = cal[np.clip(pos, 0, len(cal) - 1)]
    m.loc[m.release_time.isna(), "k_day0"] = pd.NaT

    px = load_prices(m.kticker.unique())
    m = m.sort_values("day0")
    px = px.rename(columns={"date": "px_date", "close": "px_close"}).sort_values("px_date")
    m["pre_day"] = m.day0 - pd.Timedelta(days=1)
    m = pd.merge_asof(m.sort_values("pre_day"), px, left_on="pre_day", right_on="px_date",
                      left_by="kticker", right_by="symbol", tolerance=pd.Timedelta(days=7), direction="backward")
    m["sue_analyst"] = (m.eps - m.eps_est) / m.px_close

    keep = ["accession", "k_date", "qtr", "eps_est", "eps", "release_time", "gap", "k_day0", "px_close",
            "sue_analyst"]
    out = ev.merge(m[keep].rename(columns={"eps": "k_eps", "qtr": "k_qtr"}), on="accession", how="left")
    out.to_csv(args.out, index=False)

    s = out[out.day0 <= KAGGLE_END]
    if "matched" in s:
        s = s[s.matched]
    hit = s.k_date.notna()
    print(f"EDGAR events <= 2021-06-14: {len(s)}; matched to Kaggle: {hit.mean():.1%}")
    print("gap (Kaggle date - EDGAR acceptance date) days:", s.gap.value_counts().sort_index().to_dict())
    rt = s[s.k_day0.notna()]
    print(f"day0 agreement where release_time known (n={len(rt)}): {(rt.k_day0 == rt.day0).mean():.1%}")
    both = s.dropna(subset=["sue_analyst", "eps_sue"])
    print(f"analyst SUE available: {s.sue_analyst.notna().sum()}; "
          f"sign agreement with XBRL SUE (n={len(both)}): {(np.sign(both.sue_analyst) == np.sign(both.eps_sue)).mean():.1%}; "
          f"Spearman={both.sue_analyst.corr(both.eps_sue, method='spearman'):.2f}")


if __name__ == "__main__":
    main()
