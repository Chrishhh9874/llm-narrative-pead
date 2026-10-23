"""Step 22: checks requested in the first round of review (online appendix A-E, APPENDIX.md). No API calls.

--prep reads local caches this repository does not ship (EDGAR headers, the Kaggle / Zacks file, daily
volume, XBRL facts, the raw LLM responses) and writes small event-level files to data/interim/:
  event_timing.csv   accession, report_date (8-K <PERIOD>, the date of the release), accept_hr (ET),
                     zacks_shift (Zacks-implied day 0 minus ours, trading days; NaN without release time),
                     report_before_day0 (the report date falls on or before the trading day preceding day 0),
                     vol_peak (day in {-1, 0, 1} with the largest log volume over its [-60, -11] mean)
  sue_checks.csv     accession, eps_cur_derived (current-quarter EPS = annual - nine months), eps_win_derived (any
                     derived quarter among the SUE inputs), split_unconfirmed (SUE inputs straddle a detected split
                     that neither yfinance nor Kaggle lists)
  q4_validation.csv  fiscal fourth quarters with both a directly reported three-month value and annual - nine-month
                     (first-reported values, s03 split and scale cleaning), the XBRL concept of each of the three
                     facts and whether the direct value comes from the annual value's filing
  splits_check.csv   the s03 splits with the exchange records they were checked against
  llm_runs.csv       one row per LLM task: calls, model, server fingerprints, first / last call, request settings,
                     median / mean reasoning tokens, calls that hit max_tokens

Rows of results/checks/R4_referee_checks.csv (key, n, coef, t, ci_lo, ci_hi, p, note), all on the main `_ff` CAR.
Coefficients in percentage points (Stage B: per weighted s.d.); `ci_*` are 90% intervals for the TOST rows.
  c1_*   event dates: agreement with Zacks timing, the 8-K date of report and the volume peak; suspect events;
         c1_{var}_{A|B}_{y}: T2 spec (5) tone coefficient (A) / Master CB - HB and RM - CM (B) under
         drop (suspect events removed), redate (day 0 from Zacks, else the volume peak in [-1, 0] of suspect
         events) and m1 (announcement window CAR[-1,1], also as the drift-window control)
  c2_*   EPS construction: derived-Q4 validation, split cross-check, the same tests without derived quarters or
         without events touched by an unconfirmed split
  c3_*   Stage B design: Kish effective sample size by label, sum of 1/pi against population counts by stratum
  c4_*   model runs: calls, fingerprints, inference window; Stage A tone drift after the knowledge cutoff
  c5_*   equivalence tests (TOST, 5% level): bounds = 25% of the signal's own announcement effect (ann25) or half
         the SUE drift in the same regression (sue50); `p` = TOST p, `coef` = estimate, ci = 90% CI, `ci_hi` of the
         c5_*_min row = smallest bound the test passes
"""

import argparse
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import norm

from common import INTERIM, RAW, ROOT
from s15_analysis import CTRL, load, ols
from s16_stageB import CUTOFF, build, lab

OUT = ROOT / "results" / "checks"
C0 = [c for c in CTRL if c != "car01"]
BASE = ["sue_r", "tone_r", "sue_x_tone"]
LATE_HR = 16.5  # acceptance after 16:30 ET: release time unconfirmed without Zacks (8-K may trail the wire)
YS = ("car_0_1", "car_2_20", "car_2_60")


# ================================================================ prep (local caches only)
def prep_timing():
    from common import SecClient

    sec = SecClient()
    e = pd.read_csv(INTERIM / "events_sue.csv", parse_dates=["accept_et", "day0"])

    def period(cik, acc):
        p = sec._cache_path(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{acc}.hdr.sgml")
        m = re.search(rb"<PERIOD>(\d{8})", p.read_bytes()[:3000]) if p.exists() else None
        return m.group(1).decode() if m else None

    e["report_date"] = pd.to_datetime([period(c, a) for c, a in zip(e.cik, e.accession)], format="%Y%m%d")
    e["accept_hr"] = e.accept_et.dt.hour + e.accept_et.dt.minute / 60
    cal = pd.DatetimeIndex(pd.read_csv(RAW / "prices" / "SPY.csv", parse_dates=["Date"]).Date)
    k = pd.read_csv(INTERIM / "events_kaggle.csv", parse_dates=["k_day0"])[["accession", "k_day0"]].dropna()
    k["k_pos"] = cal.searchsorted(k.k_day0)
    e = e.merge(k, on="accession", how="left")
    e["zacks_shift"] = e.k_pos - cal.searchsorted(e.day0)
    e["report_before_day0"] = cal.searchsorted(e.report_date) < cal.searchsorted(e.day0)  # report on a trading day before day 0

    # abnormal log volume on days -1, 0, 1 (yfinance, Kaggle volume where s07 used Kaggle prices)
    src = pd.read_csv(INTERIM / "car_full.csv", usecols=["accession", "price_src"])
    e = e.merge(src, on="accession", how="left")
    ren = pd.read_csv(RAW / "manual" / "ticker_renames.csv").set_index("old_ticker").new_ticker.to_dict()
    kv = pd.read_csv(INTERIM / "kaggle_close.csv.gz", parse_dates=["date"])
    kv = {t: g.set_index("date").volume for t, g in kv.groupby("symbol")}
    peak = {}
    for (t, s), g in e[e.price_src.notna()].groupby(["ticker", "price_src"]):
        if s == "yfinance":
            p = RAW / "prices" / f"{ren.get(t, t)}.csv"
            vol = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").Volume if p.exists() else None
        else:
            vol = kv.get(t.replace(".", "-"))
        if vol is None:
            continue
        v = np.log1p(vol[~vol.index.duplicated()].reindex(cal)).to_numpy()
        for r in g.itertuples():
            i = cal.searchsorted(r.day0)
            if 60 <= i < len(cal) - 1:
                w = v[i - 1: i + 2] - np.nanmean(v[i - 60: i - 10])
                if not np.isnan(w).any():
                    peak[r.accession] = int(np.argmax(w)) - 1
    e["vol_peak"] = e.accession.map(peak)
    e[["accession", "report_date", "accept_hr", "zacks_shift", "report_before_day0", "vol_peak"]].to_csv(
        INTERIM / "event_timing.csv", index=False)


def exchange_splits(tickers):
    """{ticker: [(date, ratio)]} from yfinance (data/raw/prices/splits) and the Kaggle split coefficients."""
    ren = pd.read_csv(RAW / "manual" / "ticker_renames.csv").set_index("old_ticker").new_ticker.to_dict()
    parts = [ch[ch.split_coefficient != 1] for ch in pd.read_csv(
        RAW / "kaggle_tsaustin" / "stocks_latest" / "stock_prices_latest.csv",
        usecols=["symbol", "date", "split_coefficient"], parse_dates=["date"], chunksize=3_000_000)]
    kg = pd.concat(parts)
    out = {}
    for t in tickers:
        rec = []
        p = RAW / "prices" / "splits" / f"{ren.get(t, t)}.csv"
        if p.exists():
            y = pd.read_csv(p, parse_dates=["date"])
            rec += [(d, r, "yfinance") for d, r in zip(y.date, y.ratio)]
        g = kg[kg.symbol == t.replace(".", "-")]
        rec += [(d, r, "kaggle") for d, r in zip(g.date, g.split_coefficient)]
        out[t] = rec
    return out


def prep_sue():
    from s03_xbrl_sue import FACT_DIR, clean_eps, detect_splits, extract_facts, first_reported, to_common_basis

    # (1) fiscal Q4s with both a direct three-month value and annual - nine months
    rows = []
    for p in sorted(FACT_DIR.glob("*.csv.gz")):
        cik = int(p.name.split(".")[0])
        f = extract_facts(None, cik)
        for kind in ("eps", "rev"):
            x = f[f.kind == kind]
            if x.empty:
                continue
            if kind == "eps":
                x = clean_eps(to_common_basis(x, detect_splits(x)))
            x = first_reported(x)
            x = (x.sort_values(["start", "end", "filed", "val"], ascending=[True, True, True, False]) if kind == "rev"
                 else x.sort_values(["start", "end", "filed", "prio"]))
            fr = x.drop_duplicates(["start", "end"])  # as in s03 quarterly_series
            q = fr[fr.dur.between(80, 100)].drop_duplicates("end").set_index("end")
            for _, a in fr[fr.dur.between(340, 380)].iterrows():
                b = fr[(fr.start == a.start) & ((a.end - fr.end).dt.days.between(80, 100))]
                if len(b) and a.end in q.index:
                    d = q.loc[a.end]
                    rows.append({"cik": cik, "kind": kind, "qend": a.end, "derived": a.val - b.val.iloc[0],
                                 "direct": d.val, "direct_form": d.form, "annual_concept": a.concept,
                                 "nine_concept": b.concept.iloc[0], "direct_concept": d.concept,
                                 "direct_in_annual_filing": d.accn == a.accn})
    pd.DataFrame(rows).to_csv(INTERIM / "q4_validation.csv", index=False)

    # (2) detected splits against exchange records: one split or a product of splits in (lo - 5d, hi + 5d]
    sp = pd.read_csv(INTERIM / "xbrl_splits.csv", parse_dates=["lo", "hi"])
    u = pd.read_csv(INTERIM / "universe.csv")
    tk = {c: sorted(set(g.ticker)) for c, g in u.groupby("cik")}
    rec = exchange_splits(sorted(u.ticker.unique()))
    rows = []
    for r in sp.itertuples():
        hits = sorted({(d, x, s) for t in tk.get(r.cik, []) for d, x, s in rec[t]
                       if r.lo - pd.Timedelta(days=5) < d <= r.hi + pd.Timedelta(days=5)})
        ok = False
        for s in ("yfinance", "kaggle"):
            h = {d: x for d, x, ss in hits if ss == s}
            ok |= bool(h) and abs(np.log(np.prod(list(h.values())) / r.ratio)) < np.log(1.05)
        rows.append({"cik": r.cik, "tickers": "|".join(tk.get(r.cik, [])), "lo": r.lo.date(), "hi": r.hi.date(),
                     "ratio": r.ratio, "records": "; ".join(f"{d.date()} {x:g} ({s})" for d, x, s in hits),
                     "has_record": any(rec[t] for t in tk.get(r.cik, [])), "confirmed": ok})
    chk = pd.DataFrame(rows)

    # (3) event flags; SUE inputs = quarters qend - 36 months .. qend first reported before day 0 (+ the current one)
    q = pd.read_csv(INTERIM / "quarters.csv", parse_dates=["qend", "eps_filed"])[["cik", "qend", "eps_filed", "eps_src"]]
    e = pd.read_csv(INTERIM / "events_sue.csv", parse_dates=["day0", "qend"])
    e = e[e.eps_sue.notna()][["accession", "cik", "qend", "day0", "eps_src"]]
    chk["events_affected"] = 0
    out = []
    for cik, g in e.groupby("cik"):
        qq = q[q.cik == cik]
        bad = chk[(chk.cik == cik) & ~chk.confirmed]
        for x in g.itertuples():
            w = qq[(qq.qend >= x.qend - pd.DateOffset(months=36, days=20)) & (qq.qend <= x.qend) &
                   ((qq.eps_filed < x.day0) | (qq.qend == x.qend))].dropna(subset=["eps_filed"])
            hit = [i for i, lo in zip(bad.index, pd.to_datetime(bad.lo)) if w.eps_filed.min() <= lo < w.eps_filed.max()]
            chk.loc[hit, "events_affected"] += 1
            out.append({"accession": x.accession, "eps_cur_derived": x.eps_src == "derived",
                        "eps_win_derived": (w.eps_src == "derived").any(), "split_unconfirmed": bool(hit)})
    s = pd.DataFrame(out)
    chk.to_csv(INTERIM / "splits_check.csv", index=False)
    s.to_csv(INTERIM / "sue_checks.csv", index=False)


def prep_llm():
    rows = []
    for d in sorted((INTERIM / "llm_raw").iterdir()):
        recs = []
        for p in d.glob("*.json"):
            j = json.loads(p.read_text())
            rq = j.get("request", {})
            u = (j.get("response") or {}).get("usage") or {}
            recs.append({"ts": j.get("ts"), "model": (j.get("response") or {}).get("model"),
                         "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                         "truncated": bool(j.get("truncated_usage")),
                         "fp": j.get("system_fingerprint"), "temperature": rq.get("temperature"),
                         "max_tokens": rq.get("max_tokens"), "reasoning_effort": rq.get("reasoning_effort"),
                         "json": (rq.get("response_format") or {}).get("type")})
        r = pd.DataFrame(recs)
        ts = pd.to_datetime(r.ts, unit="s", utc=True)
        rows.append({"task": d.name, "calls": len(r), "models": "|".join(sorted(r.model.dropna().unique())),
                     "fingerprints": "|".join(sorted(r.fp.dropna().unique())), "no_fingerprint": int(r.fp.isna().sum()),
                     "first_utc": ts.min(), "last_utc": ts.max(), "reasoning_median": r.reasoning.median(),
                     "reasoning_mean": r.reasoning.mean(), "truncated": int(r.truncated.sum()),
                     **{c: "|".join(map(str, sorted(r[c].dropna().unique()))) for c in
                        ("temperature", "max_tokens", "reasoning_effort", "json")}})
    pd.DataFrame(rows).to_csv(INTERIM / "llm_runs.csv", index=False)


# ================================================================ analysis
def res(key, r, x, n, note=None, level=0.05):
    ci = r.conf_int(level).loc[x] * 100
    return {"key": key, "n": n, "coef": r.params[x] * 100, "t": r.tvalues[x], "ci_lo": ci[0], "ci_hi": ci[1],
            "p": r.pvalues[x], "note": note}


def a_spec5(d, y, ann="car_0_1"):
    """T2 spec (5): CAR on SUE, tone, SUE x tone and controls; `ann` replaces CAR[0,1] as the drift control."""
    d = d.assign(car01=d[ann]) if ann != "car_0_1" else d
    xs = [*BASE, *(C0 if y in ("car_0_1", "car_m1_1") else CTRL)]
    return ols(d, y, xs, fe="yq")


def b_diff(b, y, a, c):
    """Master-label weighted mean difference a - c (neutral = confirming), s.e. clustered by day 0."""
    s = b.assign(grp=[lab(x, m, "confirmed") for x, m in zip(b.eps_sue, b.master_dir)])
    s = s[s.grp.isin([a, c])].assign(is_a=lambda z: (z.grp == a).astype(float))
    return ols(s, y, ["is_a"], cluster=("day0",), weights="w")


def tests(rows, tag, a, b, ann="car_0_1"):
    """Main tests on (possibly altered) Stage A / B frames: tone (A) and CB - HB, RM - CM (B)."""
    for y in (ann, "car_2_20", "car_2_60"):
        r, n = a_spec5(a, y, ann)
        rows.append(res(f"{tag}_A_{y}", r, "tone_r", n))
        for name, (g1, g2) in {"H1": ("CB", "HB"), "H2": ("RM", "CM")}.items():
            r, n = b_diff(b, y, g1, g2)
            rows.append(res(f"{tag}_B{name}_{y}", r, "is_a", n))


def shifted_cars(shift, also_m1=False):
    """CARs from the daily FF ARs (s07 --path) with day 0 moved by `shift` trading days (Series by accession).
    A window needs all its days; the [2,60] window of an event moved forward stops at day 60 (path end)."""
    p = pd.read_parquet(INTERIM / "car_path_full.parquet").set_index("accession")
    s = shift.reindex(p.index).fillna(0).astype(int).to_numpy()
    out = pd.DataFrame(index=p.index)
    wins = {"car_0_1": (0, 1), "car_2_20": (2, 20), "car_2_60": (2, 60)}
    if also_m1:
        wins["car_m1_1"] = (-1, 1)
    for name, (lo, hi) in wins.items():
        vals = np.full(len(p), np.nan)
        for k in np.unique(s):
            m = s == k
            cols = [f"ar_ff_{j}" for j in range(lo + k, min(hi + k, 60) + 1)]
            w = p.loc[m, cols]
            vals[m] = w.sum(axis=1).where(w.notna().all(axis=1)).to_numpy()
        out[name] = vals
    return out


def with_cars(d, cars):
    d = d.drop(columns=[c for c in cars.columns if c in d], errors="ignore").merge(
        cars, left_on="accession", right_index=True, how="left")
    d["car01"] = d.car_0_1
    return d


def c1_timing(rows, a, b):
    t = pd.read_csv(INTERIM / "event_timing.csv", parse_dates=["report_date"])
    e = pd.read_csv(INTERIM / "events_sue.csv", parse_dates=["accept_et", "day0"])[["accession", "accept_et", "day0",
                                                                                     "matched"]].merge(t, on="accession")
    e["dor_gap"] = (e.accept_et.dt.normalize() - e.report_date).dt.days
    # release dated before the filing day, and possibly before day 0
    e["dor_early"] = (e.dor_gap > 0) & e.report_before_day0
    e["suspect"] = (e.zacks_shift.fillna(0) != 0) | (e.zacks_shift.isna() & ((e.accept_hr >= LATE_HR) | e.dor_early))
    for tag, s in (("all", e[e.matched]), ("A", e[e.accession.isin(a.accession)])):
        z = s.dropna(subset=["zacks_shift"])
        rows += [{"key": f"c1_{tag}_zacks_agree", "n": len(z), "coef": (z.zacks_shift == 0).mean()},
                 {"key": f"c1_{tag}_zacks_late1", "n": int((z.zacks_shift == -1).sum())},
                 {"key": f"c1_{tag}_zacks_early1", "n": int((z.zacks_shift == 1).sum())},
                 {"key": f"c1_{tag}_zacks_2plus", "n": int((z.zacks_shift.abs() >= 2).sum())},
                 {"key": f"c1_{tag}_dor_same", "n": int(s.report_date.notna().sum()), "coef": (s.dor_gap == 0).mean()},
                 {"key": f"c1_{tag}_dor_after", "n": int((s.dor_gap < 0).sum()), "note": "report date after acceptance"},
                 {"key": f"c1_{tag}_dor_before", "n": int((s.dor_gap > 0).sum()), "note": "report date before acceptance"},
                 {"key": f"c1_{tag}_dor_before_zacks_agree", "n": int(((s.dor_gap > 0) & s.zacks_shift.notna()).sum()),
                  "coef": (s[(s.dor_gap > 0)].dropna(subset=["zacks_shift"]).zacks_shift == 0).mean()},
                 {"key": f"c1_{tag}_after16", "n": len(s), "coef": (s.accept_hr >= 16).mean()},
                 {"key": f"c1_{tag}_16_1630", "n": len(s), "coef": s.accept_hr.between(16, LATE_HR, "left").mean()},
                 {"key": f"c1_{tag}_intraday", "n": len(s), "coef": s.accept_hr.between(9.5, 16, "left").mean()},
                 {"key": f"c1_{tag}_suspect", "n": int(s.suspect.sum()), "coef": s.suspect.mean()}]
        v = s.dropna(subset=["vol_peak"])
        for cond, m in {"": v.index == v.index, "_zagree": v.zacks_shift == 0, "_zlate": v.zacks_shift == -1,
                        "_zearly": v.zacks_shift == 1}.items():
            vv = v[m]
            rows += [{"key": f"c1_{tag}_volpeak{k}{cond}", "n": len(vv), "coef": (vv.vol_peak == j).mean()}
                     for k, j in (("m1", -1), ("0", 0), ("p1", 1))]
        for lo, hi in ((0, 9.5), (9.5, 16), (16, 16.5), (16.5, 17), (17, 24)):
            z = s[s.accept_hr.between(lo, hi, "left")].dropna(subset=["zacks_shift"])
            rows.append({"key": f"c1_{tag}_zacks_late_hr{lo:g}-{hi:g}", "n": len(z), "coef": (z.zacks_shift < 0).mean()})

    sus = set(e.accession[e.suspect])
    tests(rows, "c1_drop", a[~a.accession.isin(sus)], b[~b.accession.isin(sus)])
    shift = e.set_index("accession").zacks_shift
    vol = e.set_index("accession").vol_peak
    fill = np.where(e.set_index("accession").suspect & (vol == -1), -1, 0)
    shift = shift.fillna(pd.Series(fill, index=shift.index))
    rows.append({"key": "c1_redate_moved", "n": int((shift.reindex(a.accession) != 0).sum()),
                 "note": "Stage A events whose day 0 moves"})
    cars = shifted_cars(shift)
    tests(rows, "c1_redate", with_cars(a, cars), with_cars(b, cars))
    m1 = shifted_cars(pd.Series(dtype=float), also_m1=True)[["car_m1_1"]]
    tests(rows, "c1_m1", a.merge(m1, left_on="accession", right_index=True, how="left"),
          b.merge(m1, left_on="accession", right_index=True, how="left"), ann="car_m1_1")


def c2_sue(rows, a, b):
    v = pd.read_csv(INTERIM / "q4_validation.csv", parse_dates=["qend"])
    e = pd.read_csv(INTERIM / "events_sue.csv", parse_dates=["qend"])
    e = e[e.accession.isin(a.accession)]
    for kind in ("eps", "rev"):
        x = v[v.kind == kind]
        dif = (x.derived - x.direct).abs()
        rows += [{"key": f"c2_q4_{kind}_n", "n": len(x)},
                 {"key": f"c2_q4_{kind}_exact", "n": len(x), "coef": (dif < 0.005 if kind == "eps" else dif == 0).mean()}]
        if kind == "eps":
            rows += [{"key": f"c2_q4_eps_within{c}c", "n": len(x), "coef": (dif < c / 100 + 0.005).mean()} for c in (1, 2)]
        else:
            rows.append({"key": "c2_q4_rev_within0.5pct", "n": len(x), "coef": (dif / x.direct.abs() < 0.005).mean()})
            # derived - direct = (annual - direct) - nine months: with one concept and the direct value in the 10-K,
            # a gap is the 10-K's revision of the first-reported nine-month value
            off = x[dif > 0]
            one = (off.annual_concept == off.nine_concept) & (off.nine_concept == off.direct_concept)
            rows += [{"key": "c2_q4_rev_gap_same_concept", "n": len(off), "coef": one.mean()},
                     {"key": "c2_q4_rev_gap_same_concept_10k", "n": len(off),
                      "coef": (one & off.direct_in_annual_filing).mean()}]
        # events whose current quarter is a direct Q4: SUE with the derived value instead
        m = e.merge(x[["cik", "qend", "derived"]], on=["cik", "qend"])
        m = m[m[f"{kind}_sue"].notna() & (m[f"{kind}_src"] == "q")]
        s1, s2 = m[f"{kind}_sue"], (m.derived - m[f"{kind}_lag4"]) / m[f"{kind}_sd8"]
        g = lambda z: np.select([z >= 0.5, z <= -0.5], [1, -1], 0)
        rows += [{"key": f"c2_q4_{kind}_sue_sign", "n": len(m), "coef": (np.sign(s1) == np.sign(s2)).mean()},
                 {"key": f"c2_q4_{kind}_sue_spearman", "n": len(m), "coef": s1.corr(s2, method="spearman")},
                 {"key": f"c2_q4_{kind}_sue_group", "n": len(m), "coef": (g(s1) == g(s2)).mean()}]
    sp = pd.read_csv(INTERIM / "splits_check.csv")
    rows += [{"key": "c2_splits", "n": len(sp)}, {"key": "c2_splits_has_record", "n": int(sp.has_record.sum())},
             {"key": "c2_splits_confirmed", "n": int(sp.confirmed.sum()), "coef": sp.confirmed.mean()}]
    f = pd.read_csv(INTERIM / "sue_checks.csv")
    fa = f[f.accession.isin(a.accession)]
    for c in ("eps_cur_derived", "eps_win_derived", "split_unconfirmed"):
        rows.append({"key": f"c2_n_{c}", "n": int(fa[c].sum()), "coef": fa[c].mean(), "note": "Stage A share"})
        drop = set(f.accession[f[c]])
        tests(rows, f"c2_no_{c}", a[~a.accession.isin(drop)], b[~b.accession.isin(drop)])



def strata(a):
    """s12's stratum of every Stage A event (post, hb_cand, rm_cand, random)."""
    a = a[["accession", "day0", "eps_sue", "narr_tone"]].copy()
    a["yr"] = a.day0.dt.year
    a["grp"] = np.select([a.eps_sue >= 0.5, a.eps_sue <= -0.5], ["beat", "miss"], "inline")
    a["tone_q"] = a.groupby(["yr", "grp"]).narr_tone.rank(pct=True)
    return np.select([a.day0 >= CUTOFF, (a.grp == "beat") & (a.tone_q <= 0.2), (a.grp == "miss") & (a.tone_q > 0.6)],
                     ["post", "hb_cand", "rm_cand"], "random")


def c3_design(rows, b):
    des = pd.read_csv(INTERIM / "stageB_design.csv")
    pop = pd.Series(strata(pd.read_csv(INTERIM / "agents_A_full.csv", parse_dates=["day0"]))).value_counts()
    w = 1 / des.pi
    rows.append({"key": "c3_sum_w", "n": len(des), "coef": w.sum(), "ci_lo": pop.sum(), "note": "ci_lo = population"})
    for s, g in des.groupby("stratum"):
        rows.append({"key": f"c3_stratum_{s}", "n": len(g), "coef": (1 / g.pi).sum(), "ci_lo": pop.get(s, 0),
                     "ci_hi": int(g.in_gate.sum()), "t": g.p_stratum.mean(), "p": g.pi.mean(),
                     "note": "coef = sum 1/pi, ci_lo = population, ci_hi = in gate, t = mean p_s, p = mean pi"})
    kish = lambda x: x.sum() ** 2 / (x ** 2).sum()
    rows.append({"key": "c3_kish_all", "n": len(b), "coef": kish(b.w)})
    grp = pd.Series([lab(x, m, "confirmed") for x, m in zip(b.eps_sue, b.master_dir)], index=b.index)
    for g in ("CB", "HB", "IL", "RM", "CM"):
        rows.append({"key": f"c3_kish_{g}", "n": int((grp == g).sum()), "coef": kish(b.w[grp == g])})


def c4_model(rows, a):
    r = pd.read_csv(INTERIM / "llm_runs.csv", parse_dates=["first_utc", "last_utc"])
    fps = set("|".join(r.fingerprints.dropna()).split("|")) - {""}
    rows += [{"key": "c4_calls", "n": int(r.calls.sum())},
             {"key": "c4_fingerprints", "n": len(fps), "note": "|".join(sorted(fps))},
             {"key": "c4_no_fingerprint", "n": int(r.no_fingerprint.sum())},
             {"key": "c4_models", "n": r.models.nunique(), "note": "|".join(sorted(set(r.models.dropna())))},
             {"key": "c4_first_utc", "note": str(r.first_utc.min())}, {"key": "c4_last_utc", "note": str(r.last_utc.max())}]
    for c in ("temperature", "max_tokens", "reasoning_effort", "json"):
        rows.append({"key": f"c4_{c}", "note": "|".join(sorted(set("|".join(r[c].astype(str)).split("|"))))})
    m = r.set_index("task").loc["agent_master_anon"]
    rows += [{"key": "c4_master_reasoning", "n": int(m.calls), "coef": m.reasoning_median, "t": m.reasoning_mean,
              "note": "coef = median, t = mean reasoning tokens per Master call"},
             {"key": "c4_truncated", "n": int(r.truncated.sum()), "note": "calls that hit max_tokens (all tasks)"}]
    post = a[a.day0 >= CUTOFF]
    for y in ("car_2_20", "car_2_60"):
        rr, n = a_spec5(post, y)
        rows.append(res(f"c4_A_post_{y}", rr, "tone_r", n))


def tost(key, est, se, bound, n, note=None):
    p = max(norm.sf((est + bound) / se), norm.cdf((est - bound) / se))
    return {"key": key, "n": n, "coef": est, "t": est / se, "ci_lo": est - 1.645 * se, "ci_hi": est + 1.645 * se,
            "p": p, "note": f"bound {bound:.3f}" + (f"; {note}" if note else "")}


def c5_tost(rows, a, b):
    from s16_stageB import ablation

    r0, _ = a_spec5(a, "car_0_1")
    ann = r0.params.tone_r * 100
    for y in ("car_2_20", "car_2_60"):
        r, n = a_spec5(a, y)
        est, se = r.params.tone_r * 100, r.bse.tone_r * 100
        rows.append(tost(f"c5_A_{y}_ann25", est, se, 0.25 * ann, n, "25% of tone on CAR[0,1]"))
        rows.append(tost(f"c5_A_{y}_sue50", est, se, 0.5 * r.params.sue_r * 100, n, "half the SUE drift"))
        rows.append({"key": f"c5_A_{y}_min", "n": n, "ci_hi": abs(est) + 1.645 * se, "note": "smallest bound passed"})
    ab = ablation(b)
    for sig in ("narr", "single", "master"):
        spec = {"narr": "(4) +narr", "single": "(5) +single", "master": "(6) +master"}[sig]
        a0 = ab[(ab.y == "car_0_1") & (ab.spec == spec)].iloc[0][sig]
        for y in ("car_2_20", "car_2_60"):
            r = ab[(ab.y == y) & (ab.spec == spec)].iloc[0]
            se = abs(r[sig] / r[sig + "_t"])
            rows.append(tost(f"c5_B_{sig}_{y}_ann25", r[sig], se, 0.25 * a0, int(r.n), "25% of own CAR[0,1] coef"))
            rows.append({"key": f"c5_B_{sig}_{y}_min", "n": int(r.n), "ci_hi": abs(r[sig]) + 1.645 * se})
    for name, (g1, g2) in {"H1": ("CB", "HB"), "H2": ("RM", "CM")}.items():
        r0, _ = b_diff(b, "car_0_1", g1, g2)
        for y in ("car_2_20", "car_2_60"):
            r, n = b_diff(b, y, g1, g2)
            rows.append(tost(f"c5_B{name}_{y}_ann25", r.params.is_a * 100, r.bse.is_a * 100,
                             0.25 * abs(r0.params.is_a * 100), n, "25% of the CAR[0,1] difference"))
            rows.append({"key": f"c5_B{name}_{y}_min", "n": n, "ci_hi": abs(r.params.is_a * 100) + 1.645 * r.bse.is_a * 100})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true", help="rebuild the event-level inputs from the local caches")
    args = ap.parse_args()
    if args.prep:
        for f in (prep_timing, prep_sue, prep_llm):
            print("prep:", f.__name__, flush=True)
            f()
    OUT.mkdir(parents=True, exist_ok=True)
    a = load("ff")
    a["sue_x_tone"] = (a.sue_r - 0.5) * (a.tone_r - 0.5)
    b = build("ff")
    b = b.merge(a[["accession", "tone_r", "sue_x_tone"]], on="accession", how="left")
    rows = []
    tests(rows, "c0_base", a, b)
    c5_tost(rows, a, b)
    c3_design(rows, b)
    c4_model(rows, a)
    c1_timing(rows, a, b)
    c2_sue(rows, a, b)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "R4_referee_checks.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 60):
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
