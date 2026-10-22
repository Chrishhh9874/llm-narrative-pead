"""Step 11: point-in-time market cap per event (stratification for the stage-B subsample).

mktcap = shares outstanding (latest XBRL cover-page fact filed before day0) x close on the last trading
day before day0. yfinance Close is adjusted for every later split, so raw shares are multiplied by
the yfinance split ratios after the shares' filing date. (The EPS-detected splits of s03 are not used:
spin-offs and restatements make some of them spurious, e.g. HON, MA.) Where yfinance has no price
(delisted / acquired firms) the unadjusted Kaggle close (to 2021-06) is used with the raw share count.
Events still without a price take the previous event's market cap of the same firm (imputed=True).

Shares: the freshest of dei:EntityCommonStockSharesOutstanding (summed over share classes of one
filing), us-gaap:CommonStockSharesOutstanding, us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding.
companyfacts omits dimensional facts, so multi-class firms (e.g. MA after 2010) stop reporting the
undimensioned dei value and only the latest filing across concepts is current.

Inputs : data/interim/full_events.csv (+ dev-set event files), data/raw/prices/ (+ yfinance splits)
Outputs: data/interim/xbrl_shares/{cik}.csv.gz   extracted share facts
         data/interim/mktcap.csv                 accession, cik, ticker, day0, shares, px, mktcap, src
"""

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from common import INTERIM, RAW, SecClient

KAGGLE_PX = RAW / "kaggle_tsaustin" / "stocks_latest" / "stock_prices_latest.csv"
KAGGLE_CLOSE = INTERIM / "kaggle_close.csv.gz"  # unadjusted close for the sample tickers
SHARE_DIR = INTERIM / "xbrl_shares"
SPLIT_DIR = RAW / "prices" / "splits"
MIN_SHARES = 1e6  # placeholder values (FOX reports 1 share on some cover pages)
MAX_RATIO = 300  # splits move share counts < 100x within the sample (NFLX 70x); scale errors are >= 1000x
SOURCES = [("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding"),
           ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding")]
EVENT_FILES = ["full_events.csv", "pilot_events.csv", "audit3_events.csv", "audit4_events.csv"]


def extract_shares(sec, cik):
    p = SHARE_DIR / f"{cik}.csv.gz"
    if p.exists():
        return pd.read_csv(p, parse_dates=["end", "filed"])
    js = sec.get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json", use_cache=False)
    facts = (js or {}).get("facts", {})
    rows = [{"src": c, "end": f["end"], "val": f["val"], "accn": f["accn"], "form": f.get("form"),
             "filed": f["filed"]}
            for ns, c in SOURCES for f in facts.get(ns, {}).get(c, {}).get("units", {}).get("shares", [])]
    df = pd.DataFrame(rows, columns=["src", "end", "val", "accn", "form", "filed"])
    SHARE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return pd.read_csv(p, parse_dates=["end", "filed"])


def pit_shares(f, day0):
    """Freshest filing before day0 that reports shares: (value, source, filed); ties go to the earlier
    source. dei values of one filing are summed over share classes; the other concepts take the value
    for the latest period end."""
    f = f[(f.filed < day0) & (f.val >= MIN_SHARES)]
    if len(f):  # filer scale errors (x1000, x1e6, e.g. PKG, STX; COP weighted shares in thousands):
        # far off the firm's typical cover-page count
        dei = f[f.src == SOURCES[0][1]]
        ref = (dei if len(dei) else f).val.median()
        f = f[np.abs(np.log(f.val / ref)) < np.log(MAX_RATIO)]
    best = (np.nan, None, pd.NaT)
    for _, c in SOURCES:
        g = f[f.src == c]
        if g.empty:
            continue
        last = g[g.filed == g.filed.max()]
        if c == "EntityCommonStockSharesOutstanding":
            last = last[last.end == last.end.max()]
            val = last.drop_duplicates("val").val.sum()  # one row per class; repeated facts dropped
        else:
            val = last[last.end == last.end.max()].val.iloc[-1]
        if best[1] is None or last.filed.iloc[0] > best[2]:
            best = (val, c, last.filed.iloc[0])
    return best


def yf_splits(t):
    """yfinance split history (date, ratio), cached; spin-off price adjustments appear here too."""
    p = SPLIT_DIR / f"{t}.csv"
    if not p.exists():
        import yfinance as yf

        s = yf.Ticker(t).splits
        if s is None or not isinstance(s.index, pd.DatetimeIndex):  # delisted: no history
            s = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
        SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"date": s.index.tz_localize(None).normalize(), "ratio": s.values}).to_csv(p, index=False)
    s = pd.read_csv(p, parse_dates=["date"])
    return list(zip(s.date, s.ratio))


def split_factor(splits, filed, until=None):
    """Multiply raw shares reported on `filed` by this to reach the yfinance basis (all later splits),
    or, with `until`, the basis on that date (splits in (filed, until])."""
    return float(np.prod([r for d, r in splits or [] if d > filed and (until is None or d <= until)]))


def close_before(ticker, day0, renames, cache):
    t = renames.get(ticker, ticker)
    if t not in cache:
        p = RAW / "prices" / f"{t}.csv"
        df = pd.read_csv(p, parse_dates=["Date"]) if p.exists() else pd.DataFrame()
        if len(df) and "Close" in df:
            df = df.dropna(subset=["Close"]).set_index("Date").sort_index()
            # re-assigned symbols show up as thin series (as in s07): drop days with tiny dollar volume
            dv = (df.Close * df.Volume).rolling(20, min_periods=5).median()
            cache[t] = df.Close[dv >= 2e6]
        else:
            cache[t] = None
    return last_before(cache[t], day0)


def kaggle_close(tickers):
    if not KAGGLE_CLOSE.exists():
        parts = [ch[ch.symbol.isin(tickers)] for ch in
                 pd.read_csv(KAGGLE_PX, usecols=["symbol", "date", "close", "volume"], chunksize=2_000_000)]
        pd.concat(parts).to_csv(KAGGLE_CLOSE, index=False)
    k = pd.read_csv(KAGGLE_CLOSE, parse_dates=["date"])
    out = {}
    for t, g in k.groupby("symbol"):
        g = g.drop_duplicates("date").set_index("date").sort_index()
        dv = (g.close * g.volume).rolling(20, min_periods=5).median()
        out[t] = g.close[dv >= 2e6]
    return out


def last_before(s, day0):
    if s is None:
        return np.nan
    s = s[s.index < day0]
    return s.iloc[-1] if len(s) and (day0 - s.index[-1]).days <= 7 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    ev = pd.concat([pd.read_csv(INTERIM / f, parse_dates=["day0"]) for f in EVENT_FILES], ignore_index=True)
    ev = ev.drop_duplicates("accession")[["accession", "cik", "ticker", "day0"]]
    sec = SecClient()
    ciks = sorted(ev.cik.unique())
    with ThreadPoolExecutor(args.workers) as ex:  # SecClient throttles to < 10 req/s across threads
        facts = dict(zip(ciks, ex.map(lambda c: extract_shares(sec, c), ciks)))
    print(f"share facts for {len(ciks)} CIKs, {sum(len(f) == 0 for f in facts.values())} empty")

    renames = pd.read_csv(RAW / "manual" / "ticker_renames.csv").set_index("old_ticker").new_ticker.to_dict()
    yft = sorted({renames.get(t, t) for t in ev.ticker})
    with ThreadPoolExecutor(args.workers) as ex:
        splits = dict(zip(yft, ex.map(yf_splits, yft)))
    kg = kaggle_close(set(ev.ticker.str.replace(".", "-")))
    cache, rows = {}, []
    for r in ev.itertuples():
        sh, src, filed = pit_shares(facts[r.cik], r.day0)
        spl = splits.get(renames.get(r.ticker, r.ticker))
        adj = split_factor(spl, filed)
        px_yf = close_before(r.ticker, r.day0, renames, cache)
        px_kg = last_before(kg.get(r.ticker.replace(".", "-")), r.day0)
        mc_yf, mc_kg = sh * adj * px_yf, sh * split_factor(spl, filed, until=r.day0) * px_kg
        rows.append({"accession": r.accession, "cik": r.cik, "ticker": r.ticker, "day0": r.day0.date(),
                     "shares": sh, "shares_src": src, "shares_filed": filed, "split_adj": adj,
                     "mktcap_yf": mc_yf, "mktcap_kg": mc_kg, "px_src": "yfinance" if mc_yf > 0 else
                     ("kaggle" if mc_kg > 0 else None), "mktcap": mc_yf if mc_yf > 0 else mc_kg})
    out = pd.DataFrame(rows)
    both = out.mktcap_yf.notna() & out.mktcap_kg.notna()
    ratio = (out.mktcap_yf / out.mktcap_kg)[both]
    print(f"yfinance vs Kaggle on {both.sum()} events: within 5% {(abs(np.log(ratio)) < 0.05).mean():.1%}, "
          f"off by > 1.8x {(abs(np.log(ratio)) > np.log(1.8)).sum()}")

    out = out.sort_values(["cik", "day0"])
    out["imputed"] = out.mktcap.isna()
    out["mktcap"] = out.groupby("cik").mktcap.ffill()

    # continuity check: a missed split or a wrong share class shows up as a jump between consecutive events
    lm = np.log(out.mktcap)
    out["jump"] = (lm - lm.groupby(out.cik).shift()).abs() > np.log(1.8)
    out.to_csv(INTERIM / "mktcap.csv", index=False)

    print(f"{len(out)} events: mktcap for {out.mktcap.notna().sum()} "
          f"(imputed from the firm's previous event {(out.imputed & out.mktcap.notna()).sum()}, "
          f"still missing {out.mktcap.isna().sum()})")
    print("price source:", out.px_src.value_counts().to_dict())
    print("shares source:", out.shares_src.value_counts().to_dict())
    print(f"jumps > 1.8x between consecutive events: {out.jump.sum()}")
    print("mktcap $bn quantiles:", (out.mktcap.quantile([0.01, 0.25, 0.5, 0.75, 0.99]) / 1e9).round(1).to_dict())


if __name__ == "__main__":
    main()
