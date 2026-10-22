"""Step 15: H1-H3 on the full stage-A sample (paper Sections 6-7).

Sample: agents_A_full.csv (19,940 non-dev events with Narrative tone), merged with CARs, market cap and
pre-event returns. No dev-set event enters any table.

Drift CAR model (--model, default ff): FF5 + UMD betas, alpha not subtracted (s07 `_ff`). The market model
with alpha (`_mm`) is shown only as a diagnostic: its estimation-window alpha is the pre-event run-up and
correlates with both SUE and tone (s07 docstring).

Tables (results/stageA/*.csv and printed)
  T1 grouped CARs: SUE group (|SUE| >= IN_LINE) x narrative group, two narrative definitions
       abs   tone < 0 negative / tone > 0 positive / tone == 0 neutral (absolute direction)
       rel   bottom / top tercile of tone within the calendar quarter (tone < 0 is only ~6% of events)
     H1 = CB - HB, H2 = RM - CM, difference tests by OLS on a group dummy, s.e. clustered by day 0
  T2 regressions: CAR = SUE + Tone + SUE x Tone + controls + year-quarter FE, s.e. clustered by firm and
     day 0. SUE and Tone are ranks scaled to [0, 1] within the calendar quarter, so a coefficient is the
     bottom-to-top return spread. Controls: log market cap, rev SUE rank, mom [-250,-30], pre [-29,-1],
     CAR[0,1]. H3b: Tone x divergence (beat & tone below quarter median, or miss & above).
  T3 in-line threshold robustness: H1 / H2 for IN_LINE in 0.25 / 0.5 / 1.0
  T4 sub-periods (2015-19, 2020-21, 2022-25) for the T2 main regression
  T5 text baselines on the full sample: CAR ~ SUE + <LM | FinBERT | Narrative tone> + controls + FE,
     each tone ranked within the quarter; `dr2` = R2 gain over SUE + controls (needs baselines.csv, s14)
  T6 Fama-MacBeth: the T2 spec (5) and the T5 "all three" horse race estimated by OLS within each calendar
     quarter (>= FM_MIN events); coefficient = time-series mean, t from the plain FM s.e. and Newey-West (FM_NW lags)
"""

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import INTERIM, ROOT

OUT = ROOT / "results" / "stageA"
WINDOWS = ["car_0_1", "car_2_5", "car_2_20", "car_2_60"]
PERIODS = {"2015-19": (2015, 2019), "2020-21": (2020, 2021), "2022-25": (2022, 2025)}
FM_MIN, FM_NW = 100, 2


def load(model):
    a = pd.read_csv(INTERIM / "agents_A_full.csv", parse_dates=["day0"])
    car = pd.read_csv(INTERIM / "car_full.csv")
    mc = pd.read_csv(INTERIM / "mktcap.csv")[["accession", "cik", "mktcap"]]
    d = a.merge(car, on="accession", how="left").merge(mc, on="accession", how="left")
    sfx = "" if model == "ma" else f"_{model}"
    for w in WINDOWS:
        d[w] = d[w + sfx]
    d["car01"] = d["car_0_1" + sfx]
    d["yq"] = d.day0.dt.to_period("Q").astype(str)
    d["yr"] = d.day0.dt.year
    d["size"] = np.log(d.mktcap)
    for c in ("eps_sue", "narr_tone", "rev_sue"):
        d[f"r_{c}"] = d.groupby("yq")[c].rank(pct=True)
    d["sue_r"], d["tone_r"], d["rev_r"] = d.r_eps_sue, d.r_narr_tone, d.r_rev_sue
    d["tone_t"] = d.groupby("yq").narr_tone.transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=["lo", "mid", "hi"]))
    return d


def ols(df, y, xs, fe=None, cluster=("cik", "day0"), weights=None):
    """OLS (or WLS) with optional fixed-effect dummies; s.e. clustered on 1 or 2 dimensions."""
    cols = [y, *xs, *cluster] + ([fe] if fe else []) + ([weights] if weights else [])
    df = df.dropna(subset=[c for c in cols if c])
    X = df[xs].astype(float)
    if fe:
        X = pd.concat([X, pd.get_dummies(df[fe], prefix="fe", drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X)
    g = np.column_stack([pd.factorize(df[c])[0] for c in cluster])
    g = g[:, 0] if g.shape[1] == 1 else g
    mod = sm.WLS(df[y], X, weights=df[weights]) if weights else sm.OLS(df[y], X)
    return mod.fit(cov_type="cluster", cov_kwds={"groups": g}), len(df)


def groups(d, thr, how):
    sue = np.where(d.eps_sue >= thr, "beat", np.where(d.eps_sue <= -thr, "miss", "inline"))
    if how == "abs":
        narr = np.where(d.narr_tone > 0, "pos", np.where(d.narr_tone < 0, "neg", "neu"))
    else:
        narr = np.where(d.tone_t == "hi", "pos", np.where(d.tone_t == "lo", "neg", "mid"))
    lab = {("beat", "pos"): "CB", ("beat", "neg"): "HB", ("miss", "pos"): "RM", ("miss", "neg"): "CM"}
    return pd.Series([lab.get((s, n), "IL" if s == "inline" else f"{s}_{n}") for s, n in zip(sue, narr)],
                     index=d.index)


def diff_test(d, w, a, b, weights=None):
    """mean(a) - mean(b) of column w with s.e. clustered by day 0."""
    s = d[d.grp.isin([a, b])].copy()
    s["is_a"] = (s.grp == a).astype(float)
    if s.is_a.nunique() < 2:
        return {"diff": np.nan, "t": np.nan}
    r, _ = ols(s, w, ["is_a"], cluster=("day0",), weights=weights)
    return {"diff": r.params.is_a, "t": r.tvalues.is_a, "p": r.pvalues.is_a}


def t1_grouped(d, thr):
    rows = []
    for how in ("abs", "rel"):
        d = d.assign(grp=groups(d, thr, how))
        for g, s in d.groupby("grp"):
            rows.append({"def": how, "grp": g, "n": len(s),
                         **{w: s[w].mean() * 100 for w in WINDOWS}})
        for name, (a, b) in {"H1 CB-HB": ("CB", "HB"), "H2 RM-CM": ("RM", "CM")}.items():
            row = {"def": how, "grp": name}
            for w in WINDOWS:
                t = diff_test(d, w, a, b)
                row[w] = t["diff"] * 100
                row[w + "_t"] = t["t"]
            rows.append(row)
    return pd.DataFrame(rows)


CTRL = ["size", "rev_r", "mom", "pre_m29_m1", "car01"]


def t2_regressions(d):
    d = d.copy()
    d["sue_x_tone"] = (d.sue_r - 0.5) * (d.tone_r - 0.5)
    d["div"] = (((d.eps_sue >= 0.5) & (d.tone_r < 0.5)) | ((d.eps_sue <= -0.5) & (d.tone_r >= 0.5))).astype(float)
    d["tone_x_div"] = (d.tone_r - 0.5) * d["div"]
    specs = {
        "(1) SUE": ["sue_r"],
        "(2) Tone": ["tone_r"],
        "(3) SUE+Tone": ["sue_r", "tone_r"],
        "(4) +SUExTone": ["sue_r", "tone_r", "sue_x_tone"],
        "(5) +controls": ["sue_r", "tone_r", "sue_x_tone", *CTRL],
        "(6) +div": ["sue_r", "tone_r", "div", "tone_x_div", *CTRL],
    }
    rows = []
    for w in ("car_2_20", "car_2_60", "car_0_1"):
        for name, xs in specs.items():
            xs = [x for x in xs if not (w == "car_0_1" and x == "car01")]
            r, n = ols(d, w, xs, fe="yq")
            row = {"y": w, "spec": name, "n": n, "r2": r.rsquared}
            for x in xs:
                row[x] = r.params[x] * 100
                row[x + "_t"] = r.tvalues[x]
            rows.append(row)
    return pd.DataFrame(rows)


def t3_threshold(d):
    rows = []
    for thr in (0.25, 0.5, 1.0):
        for how in ("abs", "rel"):
            dd = d.assign(grp=groups(d, thr, how))
            n = dd.grp.value_counts()
            for name, (a, b) in {"H1 CB-HB": ("CB", "HB"), "H2 RM-CM": ("RM", "CM")}.items():
                for w in ("car_2_20", "car_2_60"):
                    t = diff_test(dd, w, a, b)
                    rows.append({"thr": thr, "def": how, "test": name, "y": w, "n_a": n.get(a, 0),
                                 "n_b": n.get(b, 0), "diff": t["diff"] * 100, "t": t["t"]})
    return pd.DataFrame(rows)


def t4_periods(d):
    rows = []
    d = d.assign(sue_x_tone=(d.sue_r - 0.5) * (d.tone_r - 0.5))
    for p, (lo, hi) in PERIODS.items():
        s = d[d.yr.between(lo, hi)]
        for w in ("car_2_20", "car_2_60"):
            r, n = ols(s, w, ["sue_r", "tone_r", "sue_x_tone", *CTRL], fe="yq")
            rows.append({"period": p, "y": w, "n": n,
                         **{k: r.params[k] * 100 for k in ("sue_r", "tone_r", "sue_x_tone")},
                         **{k + "_t": r.tvalues[k] for k in ("sue_r", "tone_r", "sue_x_tone")}})
    return pd.DataFrame(rows)


def t5_baselines(d):
    b = pd.read_csv(INTERIM / "baselines.csv")[["accession", "lm_tone", "fb_tone"]]
    d = d.merge(b, on="accession", how="left")
    for c in ("lm_tone", "fb_tone"):
        d[f"{c}_r"] = d.groupby("yq")[c].rank(pct=True)
    rows = []
    for w in ("car_0_1", "car_2_20", "car_2_60"):
        ctrl = [c for c in CTRL if not (w == "car_0_1" and c == "car01")]
        s = d.dropna(subset=["lm_tone_r", "fb_tone_r"])
        base = None
        for name, sig in {"SUE only": [], "+LM": ["lm_tone_r"], "+FinBERT": ["fb_tone_r"], "+Narrative": ["tone_r"],
                          "all three": ["lm_tone_r", "fb_tone_r", "tone_r"]}.items():
            r, n = ols(s, w, ["sue_r", *sig, *ctrl], fe="yq")
            base = r.rsquared if base is None else base
            rows.append({"y": w, "spec": name, "n": n, "r2": r.rsquared, "dr2": r.rsquared - base,
                         **{k: r.params[k] * 100 for k in ["sue_r", *sig]},
                         **{k + "_t": r.tvalues[k] for k in ["sue_r", *sig]}})
    return pd.DataFrame(rows)


def t6_fama_macbeth(d):
    b = pd.read_csv(INTERIM / "baselines.csv")[["accession", "lm_tone", "fb_tone"]]
    d = d.merge(b, on="accession", how="left").assign(sue_x_tone=(d.sue_r - 0.5) * (d.tone_r - 0.5))
    for c in ("lm_tone", "fb_tone"):
        d[f"{c}_r"] = d.groupby("yq")[c].rank(pct=True)
    rows = []
    for w in ("car_0_1", "car_2_20", "car_2_60"):
        ctrl = [c for c in CTRL if not (w == "car_0_1" and c == "car01")]
        for name, xs in {"(5) +controls": ["sue_r", "tone_r", "sue_x_tone", *ctrl],
                         "all three": ["sue_r", "lm_tone_r", "fb_tone_r", "tone_r", *ctrl]}.items():
            s = d.dropna(subset=[w, *xs])
            coef = pd.DataFrame({q: sm.OLS(g[w], sm.add_constant(g[xs].astype(float))).fit().params
                                 for q, g in s.groupby("yq") if len(g) >= FM_MIN}).T
            row = {"y": w, "spec": name, "quarters": len(coef), "n": len(s)}
            for x in xs[:4] if name == "all three" else xs[:3]:
                c = coef[x]
                nw = sm.OLS(c.to_numpy(), np.ones(len(c))).fit(cov_type="HAC", cov_kwds={"maxlags": FM_NW})
                row[x] = c.mean() * 100
                row[x + "_t"] = c.mean() / (c.std(ddof=1) / np.sqrt(len(c)))
                row[x + "_tnw"] = nw.tvalues[0]
                row[x + "_pos"] = (c > 0).mean()
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ff", choices=["ff", "mb", "ma", "mm"])
    ap.add_argument("--in-line", type=float, default=0.5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = load(args.model)
    print(f"model {args.model}: {len(d)} events, CAR[2,20] {d.car_2_20.notna().sum()}, "
          f"size {d['size'].notna().sum()}, mom {d.mom.notna().sum()}")
    pd.set_option("display.width", 250, "display.max_columns", 40)
    for name, f in (("T1_grouped", lambda: t1_grouped(d, args.in_line)), ("T2_regressions", lambda: t2_regressions(d)),
                    ("T3_threshold", lambda: t3_threshold(d)), ("T4_periods", lambda: t4_periods(d)),
                    ("T5_baselines", lambda: t5_baselines(d)), ("T6_fama_macbeth", lambda: t6_fama_macbeth(d))):
        t = f()
        t.to_csv(OUT / f"{name}_{args.model}.csv", index=False)
        print(f"\n== {name} ({args.model})\n{t.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
