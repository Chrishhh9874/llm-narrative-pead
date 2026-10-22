"""Step 7: cumulative abnormal returns around day 0.

Prices: yfinance adjusted close (primary, queried under the current symbol of renamed tickers);
Kaggle `close_adjusted` (to 2021-06-14) as fallback and as an identity check against re-used symbols,
judged per event on its [-250, +60] window (yfinance kept if >= 90% of overlapping daily returns agree
within 0.5%, else Kaggle).
Market proxy = SPY.
Data guards: Kaggle daily |r| > 50% and yfinance days with 20-day median dollar volume < $2M are set
to missing (a window needs all its days, so affected events get no CAR).
Abnormal returns (all models estimated by OLS on trading days [-250, -30], >= 120 obs)
  market-adjusted  AR = r_i - r_m                                   (car_*)
  market model     AR = r_i - (a + b r_m)                           (car_*_mm)
  beta-only        AR = r_i - b r_m                                 (car_*_mb)
  FF5 + UMD        AR = r_i - rf - sum_k b_k f_k, alpha estimated but not subtracted   (car_*_ff)
  The market model's estimation-window alpha (`alpha`, daily) carries the stock's pre-event run-up into the
  drift windows (about 59 x alpha over [2,60]), so it is mechanically correlated with SUE and tone; the
  alpha-free models (mb, ff) are the ones used for drift.
Pre-event returns (controls): `mom` = market-adjusted sum over [-250, -30], `pre_m29_m1` over [-29, -1].
Windows (trading days relative to day 0): [0,1], [2,5], [2,20], [2,60]; a window needs all its days.
--path: instead of the CAR table, daily ARs over days [-20, 60] (FF5 + UMD without alpha, and market-adjusted)
for the cumulative-CAR path figure; a missing day stays missing.

Factors: Kenneth French daily FF5 (2x3) and momentum, data/raw/ff/ (in percent, converted).

Outputs
  data/raw/prices/{ticker}.csv      cached daily prices
  data/interim/car_{tag}.csv        one row per event
  data/interim/car_path_{tag}.parquet  (--path) one row per event: ar_ff_{k}, ar_ma_{k} for day k
"""

import argparse
import warnings

import numpy as np
import pandas as pd

from common import INTERIM, RAW

PX_DIR = RAW / "prices"
KAGGLE_PX = RAW / "kaggle_tsaustin" / "stocks_latest" / "stock_prices_latest.csv"
WINDOWS = {"car_0_1": (0, 1), "car_2_5": (2, 5), "car_2_20": (2, 20), "car_2_60": (2, 60)}
EST = (-250, -30)
FF_DIR = RAW / "ff"
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
PATH = (-20, 60)


def load_ff():
    def read(name):
        lines = (FF_DIR / name).read_text().splitlines()
        hdr = next(i for i, l in enumerate(lines) if l.startswith(","))
        rows = [l.split(",") for l in lines[hdr + 1:] if l[:8].strip().isdigit() and len(l.split(",")) > 1]
        cols = [c.strip() for c in lines[hdr].split(",")[1:]]
        df = pd.DataFrame([[float(x) for x in r[1:]] for r in rows], columns=cols,
                          index=pd.to_datetime([r[0].strip() for r in rows]))
        return df / 100
    ff = read("F-F_Research_Data_5_Factors_2x3_daily.csv").join(read("F-F_Momentum_Factor_daily.csv"))
    return ff.rename(columns={"Mom   ": "Mom"})


def yf_prices(ticker):
    p = PX_DIR / f"{ticker}.csv"
    if p.exists():
        df = pd.read_csv(p, parse_dates=["Date"])
        return df if len(df) else None
    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(ticker.replace(".", "-"), start="2013-06-01", end="2026-10-01", auto_adjust=False,
                         progress=False)
    if df is None or df.empty:
        pd.DataFrame(columns=["Date", "Adj Close"]).to_csv(p, index=False)
        return None
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()[["Date", "Adj Close", "Close", "Volume"]]
    df.to_csv(p, index=False)
    return df


def kaggle_prices(tickers):
    parts = []
    for ch in pd.read_csv(KAGGLE_PX, usecols=["symbol", "date", "close_adjusted"], chunksize=2_000_000):
        parts.append(ch[ch.symbol.isin(tickers)])
    k = pd.concat(parts) if parts else pd.DataFrame(columns=["symbol", "date", "close_adjusted"])
    k["date"] = pd.to_datetime(k.date)
    return {t: g.set_index("date").close_adjusted.sort_index() for t, g in k.groupby("symbol")}


def returns_for(tickers):
    """Daily returns per (historical) ticker from both sources: ({ticker: yfinance}, {ticker: kaggle}).
    yfinance is queried under the current symbol (rename map)."""
    renames = pd.read_csv(RAW / "manual" / "ticker_renames.csv").set_index("old_ticker").new_ticker.to_dict()
    k = kaggle_prices([t.replace(".", "-") for t in tickers])
    yfr, kgr = {}, {}
    for t in tickers:
        ks = k.get(t.replace(".", "-"))
        if ks is not None:
            kr = ks.pct_change()
            kgr[t] = kr.mask(kr.abs() > 0.5).dropna()  # Kaggle splices re-used symbols / corporate actions
        df = yf_prices(renames.get(t, t))
        if df is not None:
            df = df.set_index("Date").sort_index()
            yr = df["Adj Close"].pct_change()
            if "Volume" in df:
                # a re-assigned symbol (e.g. PARA after 2021) shows up as a thin, odd-priced series:
                # drop days whose 20-day median dollar volume is below $2M
                dv = (df["Close"] * df["Volume"]).rolling(20, min_periods=5).median()
                yr = yr.mask(dv < 2e6)
            yr = yr.dropna()
            if len(yr):
                yfr[t] = yr[~yr.index.duplicated(keep="last")]
    return yfr, kgr


def pick_series(yr, kr, cal, i0):
    """Source for one event, judged on its own window [-250, +60]: yfinance unless Kaggle covers the
    window and the two disagree (< 90% of overlapping days within 0.5%), which means the yfinance
    symbol belonged to another company at that time (e.g. IR before 2020-03, now Trane = TT)."""
    lo, hi = cal[max(0, i0 + EST[0])], cal[min(len(cal) - 1, i0 + 60)]
    y = yr.loc[lo:hi] if yr is not None else None
    k = kr.loc[lo:hi] if kr is not None else None
    if y is not None and len(y) and k is not None and len(k):
        both = pd.concat([y, k], axis=1, join="inner")
        if len(both) >= 100 and ((both.iloc[:, 0] - both.iloc[:, 1]).abs() < 0.005).mean() < 0.9:
            return k, "kaggle"
        return pd.concat([k[k.index < y.index.min()], y]), "yfinance"  # Kaggle only before yfinance starts
    if y is not None and len(y):
        return y, "yfinance"
    if k is not None and len(k):
        return k, "kaggle"
    return None, None


def event_car(ri, rm, day0, ff, path=False):
    cal = rm.index
    if day0 not in cal:
        return {}
    i0 = cal.get_loc(day0)
    r = pd.concat([ri.reindex(cal), rm, ff.reindex(cal)], axis=1)
    r.columns = ["ri", "rm"] + list(ff.columns)
    res = {}
    est = r.iloc[max(0, i0 + EST[0]): i0 + EST[1] + 1].dropna(subset=["ri", "rm"])
    a = b = np.nan
    bf = None
    if len(est) >= 120:
        b, a = np.polyfit(est.rm, est.ri, 1)
        ef = est.dropna(subset=FACTORS + ["RF"])
        if len(ef) >= 120:
            X = np.column_stack([np.ones(len(ef)), ef[FACTORS].to_numpy()])
            bf = np.linalg.lstsq(X, (ef.ri - ef.RF).to_numpy(), rcond=None)[0][1:]
    res["beta"], res["alpha"] = b, a
    if path:
        days = range(PATH[0], PATH[1] + 1)
        w = r.reset_index(drop=True).reindex([i0 + k for k in days])  # days past the data end -> NaN
        ma = (w.ri - w.rm).to_numpy()
        fa = ((w.ri - w.RF - w[FACTORS].to_numpy() @ bf).to_numpy() if bf is not None
              else np.full(len(w), np.nan))
        return {**{f"ar_ff_{k}": v for k, v in zip(days, fa)}, **{f"ar_ma_{k}": v for k, v in zip(days, ma)}}
    pre = r.iloc[max(0, i0 + EST[0]): i0]
    for name, (s, e) in {"mom": EST, "pre_m29_m1": (-29, -1)}.items():
        w = pre.iloc[s - EST[0]: e - EST[0] + 1] if i0 + EST[0] >= 0 else None
        ok = w is not None and len(w) == e - s + 1 and w.ri.notna().mean() >= 0.9
        res[name] = float((w.ri - w.rm).sum()) if ok else np.nan
    for name, (s, e) in WINDOWS.items():
        w = r.iloc[i0 + s: i0 + e + 1] if i0 + e < len(r) else None
        if w is None or w.ri.isna().any() or len(w) != e - s + 1:
            for sfx in ("", "_mm", "_mb", "_ff"):
                res[name + sfx] = np.nan
            continue
        res[name] = float((w.ri - w.rm).sum())
        res[name + "_mm"] = float((w.ri - (a + b * w.rm)).sum()) if np.isfinite(b) else np.nan
        res[name + "_mb"] = float((w.ri - b * w.rm).sum()) if np.isfinite(b) else np.nan
        fw = w[FACTORS + ["RF"]]
        res[name + "_ff"] = (float((w.ri - w.RF - fw[FACTORS].to_numpy() @ bf).sum())
                             if bf is not None and fw.notna().all().all() else np.nan)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=str(INTERIM / "pilot_events.csv"))
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--path", action="store_true", help="write daily ARs (car_path_{tag}.parquet) instead")
    args = ap.parse_args()

    ev = pd.read_csv(args.events, parse_dates=["day0"])
    spy = pd.read_csv(PX_DIR / "SPY.csv", parse_dates=["Date"]).set_index("Date")["Adj Close"].pct_change()
    yfr, kgr = returns_for(sorted(ev.ticker.unique()))
    cal = spy.index
    ff = load_ff()
    rows = []
    for r in ev.itertuples():
        ri, src = (None, None)
        if r.day0 in cal:
            ri, src = pick_series(yfr.get(r.ticker), kgr.get(r.ticker), cal, cal.get_loc(r.day0))
        res = event_car(ri, spy, r.day0, ff, args.path) if ri is not None else {}
        rows.append({"accession": r.accession, "price_src": src, **res})
    car = pd.DataFrame(rows)
    if args.path:
        car.to_parquet(INTERIM / f"car_path_{args.tag}.parquet", index=False)
        print(f"events={len(car)} with AR_ff on day 60: {car.ar_ff_60.notna().sum()}")
        return
    car.to_csv(INTERIM / f"car_{args.tag}.csv", index=False)
    print(f"events={len(car)} with CAR[2,20]={car.car_2_20.notna().sum()}  price source={car.price_src.value_counts(dropna=False).to_dict()}")
    print(car[[c for c in car.columns if c.startswith("car_")]].describe().round(4).to_string())


if __name__ == "__main__":
    main()
