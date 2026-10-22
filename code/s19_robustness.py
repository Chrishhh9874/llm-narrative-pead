"""Step 19: additional robustness checks reported in the text (cutoff split, guidance controls, drift CIs, probe tests). No API calls.

Rows of results/checks/R1_checks_ff.csv (key, n, coef, t, ci_lo, ci_hi, p), all on the main `_ff` CAR:
  A_drift_{y}[_nocar01]   Stage A tone-rank coefficient on drift CARs (T2 spec 5) with 95% CI, with / without the
                          CAR[0,1] control
  A_car01_guid            T2 spec 5 on CAR[0,1] plus Narrative guidance-direction dummies; A_car01_base without them
  A_guid_{g}              T2 spec 5 on CAR[2,20] within each guidance-direction subsample
  A_car01_{pre,post}      T2 spec 5 on CAR[0,1] before / after the knowledge cutoff (day 0 >= 2025-08-01)
  B_car01_{sig}_{pre,post}, B_car01_{sig}_xhit
                          Stage B Table 3 spec on CAR[0,1] by cutoff, and the signal x EPS-recall interaction
  B_dr2_single_minus_master
                          dR2(Single) - dR2(Master) on CAR[0,1]; CI from a day-0 cluster bootstrap
  probe_acc, probe_auc    return probe: accuracy (binomial p vs the chance rate implied by the marginals) and AUC
                          (Mann-Whitney p)
  n_*                     sample-flow and firm counts used in the text
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binomtest, mannwhitneyu

from common import INTERIM, ROOT
from s15_analysis import CTRL, load, ols
from s16_stageB import CUTOFF, build

OUT = ROOT / "results" / "checks"
C0 = [c for c in CTRL if c != "car01"]
N_BOOT = 500


def row(key, r, x, n):
    ci = r.conf_int().loc[x] * 100
    return {"key": key, "n": n, "coef": r.params[x] * 100, "t": r.tvalues[x], "ci_lo": ci[0], "ci_hi": ci[1],
            "p": r.pvalues[x]}


def stage_a(rows):
    d = load("ff")
    d["sue_x_tone"] = (d.sue_r - 0.5) * (d.tone_r - 0.5)
    base = ["sue_r", "tone_r", "sue_x_tone"]
    for y in ("car_2_20", "car_2_60"):
        r, n = ols(d, y, [*base, *CTRL], fe="yq")
        rows.append(row(f"A_drift_{y}", r, "tone_r", n))
        r, n = ols(d, y, [*base, *C0], fe="yq")
        rows.append(row(f"A_drift_{y}_nocar01", r, "tone_r", n))
    g = pd.get_dummies(d.narr_guidance, prefix="g", dtype=float).drop(columns="g_none")
    dg = pd.concat([d, g], axis=1)
    r, n = ols(dg, "car_0_1", [*base, *C0], fe="yq")
    rows.append(row("A_car01_base", r, "tone_r", n))
    r, n = ols(dg, "car_0_1", [*base, *C0, *g.columns], fe="yq")
    rows.append(row("A_car01_guid", r, "tone_r", n))
    for gv in sorted(d.narr_guidance.dropna().unique()):
        r, n = ols(d[d.narr_guidance == gv], "car_2_20", [*base, *CTRL], fe="yq")
        rows.append(row(f"A_guid_{gv}", r, "tone_r", n))
    post = d.day0 >= CUTOFF
    for tag, s in (("pre", d[~post]), ("post", d[post])):
        r, n = ols(s, "car_0_1", [*base, *C0], fe="yq")
        rows.append(row(f"A_car01_{tag}", r, "tone_r", n))


def stage_b(rows):
    b = build("ff")
    for sig in ("narr", "single", "master"):
        for tag, s in (("pre", b[~b.post]), ("post", b[b.post])):
            r, n = ols(s, "car_0_1", ["sue_r", sig, *C0], fe="yq", weights="w")
            rows.append(row(f"B_car01_{sig}_{tag}", r, sig, n))
        s = b.dropna(subset=["eps_hit"]).copy()
        s["hit"] = s.eps_hit.astype(float)
        s["x"] = s[sig] * s.hit
        r, n = ols(s, "car_0_1", ["sue_r", sig, "hit", "x", *C0], fe="yq", weights="w")
        rows.append(row(f"B_car01_{sig}_xhit", r, "x", n))

    s = b.dropna(subset=["car_0_1", "single", "master", "sue_r", *C0]).reset_index(drop=True)

    def r2(df, sig):
        X = pd.concat([df[["sue_r", sig, *C0]], pd.get_dummies(df.yq, drop_first=True, dtype=float)], axis=1)
        return sm.WLS(df.car_0_1, sm.add_constant(X), weights=df.w).fit().rsquared

    rng = np.random.default_rng(0)
    idx = s.groupby("day0").indices
    days = np.array(list(idx))
    boot = []
    for _ in range(N_BOOT):
        bs = s.iloc[np.concatenate([idx[k] for k in rng.choice(days, len(days))])]
        boot.append(r2(bs, "single") - r2(bs, "master"))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rows.append({"key": "B_dr2_single_minus_master", "n": len(s), "coef": r2(s, "single") - r2(s, "master"),
                 "ci_lo": lo, "ci_hi": hi, "p": np.mean(np.array(boot) <= 0)})


def probe(rows):
    p = pd.read_csv(INTERIM / "probes_ret2000_anon.csv").dropna(subset=["car_2_20", "ret_p", "ret_correct"])
    up = (p.car_2_20 > 0).astype(int)
    pred_up = p.ret_dir.astype(str).str.lower().isin(["up", "outperform", "positive"])
    chance = pred_up.mean() * up.mean() + (1 - pred_up.mean()) * (1 - up.mean())
    k = int(p.ret_correct.sum())
    rows.append({"key": "probe_acc", "n": len(p), "coef": k / len(p), "ci_lo": chance,
                 "p": binomtest(k, len(p), 0.5).pvalue})
    u = mannwhitneyu(p.ret_p[up == 1], p.ret_p[up == 0])
    rows.append({"key": "probe_auc", "n": len(p), "coef": u.statistic / (up.sum() * (1 - up).sum()), "p": u.pvalue})
    rows.append({"key": "probe_pred_up", "n": len(p), "coef": pred_up.mean()})
    rows.append({"key": "probe_base_up", "n": len(p), "coef": up.mean()})


def counts(rows):
    u = pd.read_csv(INTERIM / "universe.csv")
    e = pd.read_csv(INTERIM / "events_sue.csv")
    b = pd.read_csv(INTERIM / "agents_stageB_all.csv")
    gate = b.accession.isin(pd.read_csv(INTERIM / "gate_events.csv").accession)
    for key, n in {"n_universe_cik": u.cik.nunique(), "n_events": len(e), "n_event_cik": e.cik.nunique(),
                   "n_eps_sue": e.eps_sue.notna().sum(), "n_gate": gate.sum(),
                   **{f"n_B_{k}": v for k, v in b[~gate].stratum.value_counts().items()},
                   "n_B_post_all": (pd.to_datetime(b.day0) >= CUTOFF).sum()}.items():
        rows.append({"key": key, "n": int(n)})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in (counts, probe, stage_a, stage_b):
        step(rows)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "R1_checks_ff.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(df.round(4))


if __name__ == "__main__":
    main()
