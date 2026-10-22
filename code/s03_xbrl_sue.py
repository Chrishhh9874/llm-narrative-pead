"""Step 3: quarterly EPS / revenue from SEC XBRL companyfacts -> seasonal random-walk SUE, matched to events.

Point-in-time choices
  * Every period uses its AS-FIRST-REPORTED value (earliest `filed` among 10-Q/10-K facts), so lagged
    quarters are what investors knew at the announcement, not later restatements.
  * Discrete quarters come from ~90-day facts; a missing quarter (typically Q4) = YTD(k) - YTD(k-1)
    with the same fiscal-year start (e.g. FY - 9M), both first-reported.
  * Bad EPS facts (found on the development sample): |EPS| >= 1000 are scale errors (HAL "120000" for 0.12) and are
    dropped; when a period is re-reported later with a value more than 50x apart after the split adjustment
    (LIN's pre-merger shell entity: Q3 2018 first -182.16, later 1.58), every filing of it takes the latest value.
  * Stock splits: when the same period is re-reported later with a value ratio close to a split ratio
    (2, 3, 1.5, ...), a split is dated between the two filings and earlier-filed EPS values are
    divided by it. SUE is a ratio, so any consistent per-share basis works.
SUE (Livnat & Mendenhall 2006, seasonal random walk):
    SUE_q = (X_q - X_{q-4}) / sd(X_k - X_{k-4}, k = q-1..q-8), needs >= 6 of the 8 past changes.
    In events_sue.csv SUE is point-in-time: lagged quarters count only if first reported before day0.
    quarters.csv keeps the unrestricted panel version.
Event matching: quarter q is announced by the last Item 2.02 8-K with day0 in (end_q, first-report
filing date + 1 day] (fallback: first one up to + 5 days), capped at end_q + 120 days, skipping follow-ups
when another candidate remains (EX-99.1 missing / PDF only, a call transcript, financial supplement, IR packet,
AUM report, or a dividend announcement: about 7% of windows with >= 2 candidates). Earlier 2.02
8-Ks in the window are pre-announcements; their count is kept as n_8k_in_window.

Outputs
  data/interim/xbrl_facts/{cik}.csv.gz   extracted facts (raw companyfacts JSON is not kept)
  data/interim/quarters.csv              firm-quarter panel with SUE
  data/interim/events_sue.csv            events.csv + matched quarter + SUE
"""

import argparse
import warnings
import re

import numpy as np
import pandas as pd

from common import INTERIM, SecClient

FACT_DIR = INTERIM / "xbrl_facts"
EPS_CONCEPTS = ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"]
REV_CONCEPTS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
                "SalesRevenueGoodsNet", "SalesRevenueServicesNet", "RevenuesNetOfInterestExpense",
                "RegulatedAndUnregulatedOperatingRevenue", "ElectricUtilityRevenue"]
FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT"}
SPLIT_RATIOS = np.array([1.25, 4 / 3, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 40, 50])
SPLIT_RATIOS = np.concatenate([SPLIT_RATIOS, 1 / SPLIT_RATIOS])  # reverse splits


# ---------------------------------------------------------------- extraction
def extract_facts(sec, cik):
    p = FACT_DIR / f"{cik}.csv.gz"
    if p.exists():
        return pd.read_csv(p, parse_dates=["start", "end", "filed"])
    js = sec.get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json", use_cache=False)
    rows = []
    gaap = (js or {}).get("facts", {}).get("us-gaap", {})
    for kind, concepts, unit in (("eps", EPS_CONCEPTS, "USD/shares"), ("rev", REV_CONCEPTS, "USD")):
        for c in concepts:
            for f in gaap.get(c, {}).get("units", {}).get(unit, []):
                if f.get("form") in FORMS and f.get("start"):
                    rows.append({"kind": kind, "concept": c, "start": f["start"], "end": f["end"],
                                 "val": f["val"], "accn": f["accn"], "form": f["form"], "filed": f["filed"]})
    df = pd.DataFrame(rows, columns=["kind", "concept", "start", "end", "val", "accn", "form", "filed"])
    FACT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return pd.read_csv(p, parse_dates=["start", "end", "filed"])


# ---------------------------------------------------------------- per-share split adjustment
def detect_splits(eps):
    """Split events (lo, hi, ratio): ratio = old/new per-share value; split happened in (lo, hi]."""
    ev = []
    for _, g in eps.groupby(["start", "end"]):
        g = g.sort_values("filed").drop_duplicates("filed")
        v, f = g.val.to_numpy(), g.filed.to_numpy()
        for i in range(1, len(g)):
            if v[i] == 0 or v[i - 1] == 0 or np.sign(v[i]) != np.sign(v[i - 1]):
                continue
            r = v[i - 1] / v[i]
            j = np.argmin(np.abs(np.log(SPLIT_RATIOS) - np.log(r)))
            if abs(np.log(r) - np.log(SPLIT_RATIOS[j])) < 0.03 and abs(v[i - 1]) >= 0.05:
                ev.append((f[i - 1], f[i], SPLIT_RATIOS[j]))
    if not ev:
        return []
    ev = pd.DataFrame(ev, columns=["lo", "hi", "ratio"]).sort_values("hi")
    merged = []
    for ratio, g in ev.groupby("ratio"):
        # cluster windows of the same ratio that overlap; keep the intersection
        cur = None
        for lo, hi in sorted(zip(g.lo, g.hi), key=lambda x: x[1]):
            if cur and lo < cur[1]:
                cur = [max(cur[0], lo), min(cur[1], hi), cur[2] + 1]
            else:
                if cur:
                    merged.append((cur[0], cur[1], ratio, cur[2]))
                cur = [lo, hi, 1]
        merged.append((cur[0], cur[1], ratio, cur[2]))
    # require >= 2 periods agreeing on a split, to ignore isolated restatements
    return [(lo, hi, r) for lo, hi, r, n in merged if n >= 2]


def to_common_basis(eps, splits):
    """Express every EPS value on the basis after all detected splits."""
    adj = np.ones(len(eps))
    for lo, hi, r in splits:
        adj = np.where(eps.filed.to_numpy() <= lo, adj * r, adj)
    out = eps.copy()
    out["val"] = out.val / adj
    return out


def clean_eps(eps, max_abs=1000, max_ratio=50, floor=0.05):
    """Drop scale errors and replace values that a later filing of the same period puts on another share basis."""
    eps = eps[eps.val.abs() < max_abs].copy()
    last = eps.sort_values("filed").groupby(["start", "end"]).val.transform("last")
    lo = np.maximum(np.minimum(eps.val.abs(), last.abs()), floor)
    off = np.maximum(eps.val.abs(), last.abs()) / lo > max_ratio
    eps.loc[off, "val"] = last[off]
    return eps


# ---------------------------------------------------------------- quarterly series
def first_reported(f):
    f = f.copy()
    f["dur"] = (f.end - f.start).dt.days
    # concept priority, then earliest filing
    f["prio"] = f.concept.map({c: i for i, c in enumerate(EPS_CONCEPTS + REV_CONCEPTS)})
    return f


def quarterly_series(f, kind):
    """DataFrame indexed by quarter end: val, filed (first report), src (q|derived)."""
    empty = pd.DataFrame(columns=["val", "filed", "src"], index=pd.DatetimeIndex([], name="end"))
    if f.empty:
        return empty
    f = first_reported(f)
    if kind == "rev":
        # revenue: per period keep the largest of the candidate concepts within the first-reporting filing
        f = f.sort_values(["start", "end", "filed", "val"], ascending=[True, True, True, False])
    else:
        f = f.sort_values(["start", "end", "filed", "prio"])
    fr = f.drop_duplicates(["start", "end"])  # as-first-reported per exact period

    q = fr[fr.dur.between(80, 100)].copy()
    q["src"] = "q"
    ytd = fr[fr.dur.between(170, 380)]
    derived = []
    for _, a in ytd.iterrows():
        # previous YTD with same start, ending ~1 quarter earlier
        b = fr[(fr.start == a.start) & (fr.end < a.end) & ((a.end - fr.end).dt.days.between(80, 100))]
        if len(b):
            b = b.iloc[0]
            derived.append({"start": b.end + pd.Timedelta(days=1), "end": a.end, "val": a.val - b.val,
                            "filed": max(a.filed, b.filed), "src": "derived"})
    d = pd.DataFrame(derived, columns=["start", "end", "val", "filed", "src"])
    allq = pd.concat([q[["start", "end", "val", "filed", "src"]], d], ignore_index=True)
    # one value per quarter end: prefer direct quarterly fact, then earliest filed
    if allq.empty:
        return empty
    allq["end"] = pd.to_datetime(allq.end)
    allq["src_rank"] = (allq.src != "q").astype(int)
    allq = allq.sort_values(["end", "src_rank", "filed"]).drop_duplicates("end")
    # collapse near-duplicate ends (52/53-week calendars, +-6 days)
    allq = allq[allq.end.diff().dt.days.fillna(999) > 6]
    return allq.set_index("end")[["val", "filed", "src"]]


def lag_value(s, end, months):
    """Value of the quarter ending ~`months` months before `end` (+-20 days)."""
    target = end - pd.DateOffset(months=months)
    idx = s.index[(s.index >= target - pd.Timedelta(days=20)) & (s.index <= target + pd.Timedelta(days=20))]
    return s.loc[idx[0], "val"] if len(idx) else np.nan


def sue_panel(s, min_hist=6):
    if s.empty:
        return pd.DataFrame()
    s = s.copy()
    s["lag4"] = [lag_value(s, e, 12) for e in s.index]
    s["chg"] = s.val - s.lag4
    sd, n = [], []
    for e in s.index:
        past = [lag_value(s.assign(val=s.chg), e, 3 * k) for k in range(1, 9)]
        past = [x for x in past if pd.notna(x)]
        n.append(len(past))
        sd.append(np.std(past, ddof=1) if len(past) >= min_hist else np.nan)
    s["sd8"], s["n_hist"] = sd, n
    s["sue"] = np.where(s.sd8 > 0, s.chg / s.sd8, np.nan)
    return s


# ---------------------------------------------------------------- main
def process_cik(sec, cik):
    f = extract_facts(sec, cik)
    if f.empty:
        return pd.DataFrame(), []
    eps = f[f.kind == "eps"]
    splits = detect_splits(eps)
    eps = clean_eps(to_common_basis(eps, splits))
    e = sue_panel(quarterly_series(eps, "eps")).add_prefix("eps_")
    r = sue_panel(quarterly_series(f[f.kind == "rev"], "rev")).add_prefix("rev_")
    out = e.join(r, how="outer")
    out.index.name = "qend"
    out = out.reset_index()
    out.insert(0, "cik", cik)
    return out, splits


def pit_sue(q, qend, day0, kind, min_hist=6):
    """Event-level SUE that only uses lagged quarters first reported before day0 (no look-ahead).
    q: one firm's quarters (qend, {kind}_val, {kind}_filed). Returns (lag4, chg, sd8, n_hist, sue)."""
    s = q[["qend", f"{kind}_val", f"{kind}_filed"]].dropna()
    s = s[(s.qend == qend) | (s[f"{kind}_filed"] < day0)].set_index("qend").rename(columns={f"{kind}_val": "val"})
    if qend not in s.index:
        return (np.nan,) * 4 + (np.nan,)
    lag = lambda e, m: lag_value(s, e, m)
    lag4 = lag(qend, 12)
    chg = s.loc[qend, "val"] - lag4
    past = [lag(e, 0) - lag(e, 12) for e in [qend - pd.DateOffset(months=3 * k) for k in range(1, 9)]]
    past = [x for x in past if pd.notna(x)]
    sd = np.std(past, ddof=1) if len(past) >= min_hist else np.nan
    return lag4, chg, sd, len(past), (chg / sd if sd > 0 else np.nan)


RELEASE_HEAD = re.compile(r"(reports?|announces?|posts?|delivers?|releases?)\b.{0,120}\b(results|earnings|quarter)|"
                          r"\b(first|second|third|fourth)[- ]quarter.{0,60}(results|earnings)|"
                          r"results for the (first|second|third|fourth) quarter", re.I | re.S)


# follow-up 8-Ks after the release, still under Item 2.02: call transcripts, IR packets / supplements, AUM reports
FOLLOWUP_HEAD = re.compile(r"\btranscript\b|held a conference call|call participants|prepared remarks|"
                           r"financial supplement|statistical supplement|supplemental (reportable )?segment information|"
                           r"investor (packet|presentation)|assets under management|(\.\s?){8,}", re.I)  # dot leaders: IR packet index
DIVIDEND = re.compile(r"\bdeclare[sd]?\b.{0,80}\bdividends?\b|\bdividends?\b.{0,40}\b(declared|payment)\b", re.I | re.S)
RESULTS_WORD = re.compile(r"\bresults\b|\bearnings\b|net income|per diluted share|\brevenues?\b", re.I)


warnings.filterwarnings("ignore", message=".*XML document.*")


def exhibit_head(sec, cik, acc):
    """First 1,500 characters of the filing's EX-99.1 text ("" if there is none or it is not HTML/text)."""
    from bs4 import BeautifulSoup
    from s04_ex991 import exhibit_url
    url, _ = exhibit_url(sec, cik, acc)
    b = sec.get(url) if url and not url.lower().endswith(".pdf") else None
    return re.sub(r"\s+", " ", BeautifulSoup(b, "lxml").get_text(" "))[:1500] if b else ""


def release_like(sec, cik, acc):
    """Does the filing's EX-99.1 open like an earnings release (first 1,500 characters)?"""
    return bool(RELEASE_HEAD.search(exhibit_head(sec, cik, acc)))


def followup_like(sec, cik, acc):
    """A 2.02 8-K that is not the release itself: no (or an almost empty) EX-99.1, a transcript / supplement / IR
    packet / AUM report, or a dividend announcement without results wording ("ANNOUNCES FIRST QUARTER 2016 DIVIDEND
    PAYMENT": APH, EXPD)."""
    h = exhibit_head(sec, cik, acc)
    body = re.sub(r"^.{0,80}?exhibit 99\.?1\s*", "", h, flags=re.I)
    return (len(body) < 200 or bool(FOLLOWUP_HEAD.search(h[:400])) or
            (bool(DIVIDEND.search(h)) and not RESULTS_WORD.search(h)))


def tie_order(sec, ev):
    """Order of 8-Ks sharing a day0, so that the one picked as "last" is the earnings release: filings whose
    EX-99.1 reads like a release come after those that do not (CFO commentary, slides); among releases the
    latest (a same-day corrected re-filing) wins; if none reads like a release, the earliest wins.
    (143 such groups 2015-2025; when exactly one is a release it is the earliest filing in 74% of them.)"""
    ev = ev.copy()
    dup = ev.duplicated(["cik", "day0"], keep=False)
    ev["rel"] = False
    if sec is not None:
        ev.loc[dup, "rel"] = [release_like(sec, r.cik, r.accession) for r in ev[dup].itertuples()]
    t = ev.accept_et.astype("int64")
    ev["tie"] = np.where(ev.rel, t, -t)
    return ev.sort_values(["cik", "day0", "rel", "tie"])


def match_events(ev, quarters, sec=None):
    rows = []
    ev_sorted = tie_order(sec, ev)
    for cik, g in ev_sorted.groupby("cik", sort=False):
        q = quarters[quarters.cik == cik].sort_values("qend")
        used = set()
        for _, x in q.iterrows():
            filed = x.eps_filed if pd.notna(x.eps_filed) else x.rev_filed
            if pd.isna(filed):
                continue
            cap = x.qend + pd.Timedelta(days=120)
            if filed > cap:  # first XBRL report long after the fact (e.g. post-merger comparatives)
                continue
            win = g[(g.day0 > x.qend) & (g.day0 <= min(filed + pd.Timedelta(days=5), cap)) & ~g.accession.isin(used)]
            # the release is the LAST 2.02 8-K up to the 10-Q/10-K filing (+1 day); earlier ones in the
            # window are pre-announcements (e.g. AAPL 2019-01-02 guidance letter). Fallback: first after.
            pre = win[win.day0 <= filed + pd.Timedelta(days=1)]
            # ... but not a follow-up filed after the release (transcript, IR packet, dividend 8-K)
            if len(pre) >= 2 and sec is not None:
                keep = pre[[not followup_like(sec, cik, a) for a in pre.accession]]
                pre = keep if len(keep) else pre
            pick = pre.iloc[-1] if len(pre) else (win.iloc[0] if len(win) else None)
            if pick is not None:
                used.add(pick.accession)
                rows.append({"accession": pick.accession, "qend": x.qend, "n_8k_in_window": len(win)})
    mt = pd.DataFrame(rows, columns=["accession", "qend", "n_8k_in_window"])
    out = ev.merge(mt, on="accession", how="left").merge(quarters, on=["cik", "qend"], how="left")
    out["matched"] = out.qend.notna()
    # replace panel SUE with point-in-time SUE (panel version may use history first filed after day0)
    byc = {c: g for c, g in quarters.groupby("cik")}
    for kind in ("eps", "rev"):
        cols = [f"{kind}_{c}" for c in ("lag4", "chg", "sd8", "n_hist", "sue")]
        vals = [pit_sue(byc[r.cik], r.qend, r.day0, kind) if r.matched else (np.nan,) * 5
                for r in out[["cik", "qend", "day0", "matched"]].itertuples()]
        out[cols] = pd.DataFrame(vals, index=out.index, columns=cols)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ciks", type=str, default="", help="comma-separated subset (default: all event firms)")
    ap.add_argument("--rate", type=float, default=8.0, help="SEC requests per second")
    ap.add_argument("--rematch", action="store_true", help="reuse quarters.csv; only redo the event matching")
    args = ap.parse_args()

    sec = SecClient(max_per_sec=args.rate)
    ev = pd.read_csv(INTERIM / "events.csv", parse_dates=["filing_date", "accept_et", "day0"])
    ciks = [int(c) for c in args.ciks.split(",")] if args.ciks else sorted(ev.cik.unique())
    ev = ev[ev.cik.isin(ciks)]

    if args.rematch:
        quarters = pd.read_csv(INTERIM / "quarters.csv", parse_dates=["qend", "eps_filed", "rev_filed"])
        ciks = []
    panels, split_log = [], []
    for i, cik in enumerate(ciks):
        try:
            p, sp = process_cik(sec, cik)
        except Exception as e:  # keep going; failures are listed at the end
            print(f"cik {cik} failed: {type(e).__name__}: {e}", flush=True)
            p, sp = pd.DataFrame(), []
        panels.append(p)
        split_log += [{"cik": cik, "lo": lo, "hi": hi, "ratio": r} for lo, hi, r in sp]
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(ciks)} firms", flush=True)
    if not args.rematch:
        quarters = pd.concat(panels, ignore_index=True)
        quarters.to_csv(INTERIM / "quarters.csv", index=False)
        pd.DataFrame(split_log).to_csv(INTERIM / "xbrl_splits.csv", index=False)

    out = match_events(ev, quarters, sec)
    out.to_csv(INTERIM / "events_sue.csv", index=False)
    m = out[out.matched]
    print(f"events={len(out)} matched={len(m)} ({len(m) / len(out):.1%}); "
          f"with eps SUE={m.eps_sue.notna().sum()} rev SUE={m.rev_sue.notna().sum()}"
          + ("" if args.rematch else f"; splits={len(split_log)}"))
    print("eps SUE quantiles:", m.eps_sue.quantile([.01, .1, .25, .5, .75, .9, .99]).round(2).to_dict())


if __name__ == "__main__":
    main()
