"""Step 18: calendar-time portfolios (paper Section 7): Long CB / Short HB, Long RM / Short CM.

Each trading day, a group's portfolio holds every event whose day is inside the holding window ([2, 20] or
[2, 60] trading days after day 0); daily return = equal-weighted (stage B: 1/pi-weighted) mean raw return.
Raw returns come from s07 --path: r_i = ar_ma_k + r_SPY on that date. A leg needs >= MIN_N stocks that day,
otherwise the day is dropped. The long-short daily return is regressed on FF5 + UMD; alpha in % per day and
annualised (x 252), Newey-West s.e. with NW_LAGS lags. Long legs alone are regressed on excess returns.

Groups
  A-rel   stage A, SUE group x within-quarter tone tercile (s15 groups "rel"; same as T1 / figure 1)
  A-prev  stage A, tone terciles from the previous calendar quarter's breakpoints (no look-ahead in formation)
  B       stage B, Master labels, neutral -> confirmed (s16 lab), 1/pi weights
  plus Long beat / Short miss (|SUE| >= 0.5, stage A) as a power check: SUE alone drifts (T2)

Output: results/calendar/C1_calendar_ff.csv
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import INTERIM, ROOT
from s07_car import FACTORS, PX_DIR, load_ff
from s15_analysis import groups, load
from s16_stageB import build, lab

OUT = ROOT / "results" / "calendar"
HOLD = {"[2,20]": (2, 20), "[2,60]": (2, 60)}
MIN_N = 5
NW_LAGS = 10


def long_returns(ev, path, cal, rm):
    """One row per (event, holding day k in 2..60): date and raw return."""
    p = path.merge(ev[["accession", "day0"]], on="accession")
    p = p[p.day0.isin(cal)]
    i0 = cal.get_indexer(p.day0)
    rows = []
    for k in range(2, 61):
        idx = i0 + k
        ok = idx < len(cal)
        dates = cal[idx[ok]]
        rows.append(pd.DataFrame({"accession": p.accession.to_numpy()[ok], "k": k, "date": dates,
                                  "ret": p[f"ar_ma_{k}"].to_numpy()[ok] + rm.reindex(dates).to_numpy()}))
    return pd.concat(rows).dropna(subset=["ret"])


def portfolio(r, members, lo, hi):
    """Daily weighted mean return of `members` (accession -> weight) held over days [lo, hi]; plus count."""
    s = r[r.k.between(lo, hi)].merge(members, on="accession")
    s["wr"] = s.ret * s.w
    g = s.groupby("date")
    return pd.DataFrame({"ret": g.wr.sum() / g.w.sum(), "n": g.size()})


def regress(y, ff):
    d = pd.concat([y.rename("y"), ff], axis=1, join="inner").dropna()
    r = sm.OLS(d.y, sm.add_constant(d[FACTORS])).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    return {"days": len(d), "alpha_d": r.params.const * 100, "alpha_ann": r.params.const * 252 * 100,
            "t": r.tvalues.const, "ci_lo_ann": r.conf_int().loc["const", 0] * 252 * 100,
            "ci_hi_ann": r.conf_int().loc["const", 1] * 252 * 100, "r2": r.rsquared,
            **{f"b_{f}": r.params[f] for f in FACTORS}}


def prev_quarter_terciles(d):
    """Tone tercile using the previous calendar quarter's 1/3 and 2/3 breakpoints."""
    q = d.groupby("yq").narr_tone.quantile([1 / 3, 2 / 3]).unstack()
    q.index = pd.PeriodIndex(q.index, freq="Q") + 1
    q.index = q.index.astype(str)
    lo, hi = d.yq.map(q[1 / 3]), d.yq.map(q[2 / 3])
    t = np.where(d.narr_tone > hi, "hi", np.where(d.narr_tone < lo, "lo", "mid"))
    return pd.Series(np.where(lo.isna(), None, t), index=d.index)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rm = pd.read_csv(PX_DIR / "SPY.csv", parse_dates=["Date"]).set_index("Date")["Adj Close"].pct_change()
    cal = rm.index
    ff = load_ff()
    path = pd.read_parquet(INTERIM / "car_path_full.parquet", columns=["accession", *[f"ar_ma_{k}" for k in range(2, 61)]])

    a = load("ff")
    a["grp_rel"] = groups(a, 0.5, "rel")
    a["grp_prev"] = groups(a.assign(tone_t=prev_quarter_terciles(a)), 0.5, "rel").where(
        prev_quarter_terciles(a).notna())
    a["grp_sue"] = np.where(a.eps_sue >= 0.5, "beat", np.where(a.eps_sue <= -0.5, "miss", None))
    b = build("ff")
    b["grp"] = [lab(s, x, "confirmed") for s, x in zip(b.eps_sue, b.master_dir)]

    ra = long_returns(a, path, cal, rm)
    rb = long_returns(b, path, cal, rm)
    sets = {"A-rel": (ra, a, "grp_rel", None), "A-prev": (ra, a, "grp_prev", None), "B": (rb, b, "grp", "w")}
    tests = {"CB-HB": ("CB", "HB"), "RM-CM": ("RM", "CM")}

    rows = []
    for (hname, (lo, hi)) in HOLD.items():
        jobs = [(s, t, *tests[t]) for s in sets for t in tests] + [("A-sue", "beat-miss", "beat", "miss")]
        for sname, tname, gl, gs in jobs:
            r, d, col, wcol = sets["A-rel"][:2] + ("grp_sue", None) if sname == "A-sue" else sets[sname]
            w = d[wcol] if wcol else pd.Series(1.0, index=d.index)
            leg = {g: portfolio(r, pd.DataFrame({"accession": d.accession[d[col] == g], "w": w[d[col] == g]}), lo, hi)
                   for g in (gl, gs)}
            both = leg[gl].join(leg[gs], lsuffix="_l", rsuffix="_s", how="inner")
            both = both[(both.n_l >= MIN_N) & (both.n_s >= MIN_N)]
            base = {"hold": hname, "sample": sname, "portfolio": tname,
                    "n_long_avg": both.n_l.mean(), "n_short_avg": both.n_s.mean()}
            rows.append({**base, "leg": "long-short", **regress(both.ret_l - both.ret_s, ff)})
            for g, sfx in ((gl, "_l"), (gs, "_s")):
                ex = both["ret" + sfx] - ff.RF.reindex(both.index)
                rows.append({**base, "leg": g, **regress(ex, ff)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "C1_calendar_ff.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(t[["hold", "sample", "portfolio", "leg", "days", "n_long_avg", "n_short_avg", "alpha_d", "alpha_ann",
             "t", "ci_lo_ann", "ci_hi_ann"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
