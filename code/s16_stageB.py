"""Step 16: H4, ablations and the Master-label H1 / H2 on the stage-B sub-sample (paper Section 8).

Sample: agents_stageB_all.csv (5,000 non-dev events, stratified; weights 1/pi restore the population).
Every regression and mean is weighted by 1/pi; s.e. clustered by firm and day 0 (means: by day 0).

Signals are weighted z-scores within stage B, so each coefficient is the CAR change per 1 s.d.:
  lm (LM tone), fb (FinBERT tone), narr (Narrative tone), single (Single score),
  master (full debate), noreb (Master without the rebuttal round).
SUE enters as its full-sample rank within the calendar quarter (s15 `sue_r`), plus the s15 controls.

Tables (results/stageB/*.csv)
  B1 ablation: CAR ~ SUE + <signal> + controls + year-quarter FE, one signal at a time,
     plus a horse race (all signals together) and Master vs Single head-to-head; `dr2` = R2 gain over
     spec (1). Under the paper's framing the CAR[0,1] rows are the main result, CAR[2,20] / [2,60] the drift test
  B2 labels: weighted mean CARs of the 2x2 + In-line labels from master / noreb / single, and H1 = CB - HB,
     H2 = RM - CM, under two treatments of a neutral narrative: "confirmed" (s10 label(): neutral -> CB / CM)
     and "drop" (neutral non-in-line events removed)
  B3 robustness: (a) repeat-stable labels and modal score (500 repeat events); (b) memory split by eps_hit
     (signal x eps_hit); (c) post-cutoff events only (day 0 >= 2025-08-01); (d) sub-periods;
     (e) without truncated Master outputs
"""

import argparse
import json

import numpy as np
import pandas as pd

from common import INTERIM, ROOT
from s15_analysis import CTRL, PERIODS, load, ols

OUT = ROOT / "results" / "stageB"
SIGNALS = {"lm": "lm_tone", "fb": "fb_tone", "narr": "narr_tone", "single": "single_score",
           "master": "master_score", "noreb": "master_noreb_score"}
CUTOFF = "2025-08-01"
IN_LINE = 0.5


def wz(x, w):
    m = np.average(x[x.notna()], weights=w[x.notna()])
    s = np.sqrt(np.average((x[x.notna()] - m) ** 2, weights=w[x.notna()]))
    return (x - m) / s


def truncated(role):
    """accessions whose repeat-0 call of `role` hit max_tokens and was re-sent with a larger limit.

    Read from the raw LLM cache when it exists (and saved to truncated_{role}.csv), else from that csv."""
    raw, csv = INTERIM / "llm_raw" / f"agent_{role}_anon", INTERIM / f"truncated_{role}.csv"
    if not raw.exists():
        return set(pd.read_csv(csv).accession)
    out = set()
    for p in raw.glob("*__r0.json"):
        if json.loads(p.read_text()).get("truncated_usage"):
            out.add(p.name.split("__")[0])
    pd.DataFrame({"accession": sorted(out)}).to_csv(csv, index=False)
    return out


def build(model):
    a = load(model)[["accession", "cik", "yq", "yr", "sue_r", "rev_r", "size", "mom", "pre_m29_m1", "car01",
                     "car_0_1", "car_2_5", "car_2_20", "car_2_60"]]
    b = pd.read_csv(INTERIM / "agents_stageB_all.csv", parse_dates=["day0"])
    base = pd.read_csv(INTERIM / "baselines.csv")[["accession", "lm_tone", "fb_tone"]]
    cut = pd.read_csv(INTERIM / "cutoff_probe_stageB.csv")[["accession", "eps_hit"]]
    d = b.merge(a, on="accession", how="left").merge(base, on="accession", how="left").merge(cut, on="accession", how="left")
    d["w"] = 1 / d.pi
    for k, c in SIGNALS.items():
        d[k] = wz(d[c], d.w)
    d["post"] = d.day0 >= CUTOFF
    return d


def ablation(d, ys=("car_2_20", "car_2_60", "car_0_1")):
    rows = []
    specs = {"(1) SUE only": [], **{f"({i}) +{k}": [k] for i, k in enumerate(SIGNALS, 2)},
             "horse race": list(SIGNALS), "master vs single": ["master", "single"],
             "master vs narr": ["master", "narr"]}
    for y in ys:
        for name, sig in specs.items():
            xs = ["sue_r", *sig, *[c for c in CTRL if not (y == "car_0_1" and c == "car01")]]
            r, n = ols(d, y, xs, fe="yq", weights="w")
            row = {"y": y, "spec": name, "n": n, "r2": r.rsquared}
            for x in ["sue_r", *sig]:
                row[x] = r.params[x] * 100
                row[x + "_t"] = r.tvalues[x]
            rows.append(row)
    df = pd.DataFrame(rows)
    df["dr2"] = df.r2 - df.y.map(df[df.spec == "(1) SUE only"].set_index("y").r2)  # gain over SUE + controls
    return df


def lab(sue, direction, neutral):
    if pd.isna(sue) or pd.isna(direction):
        return None
    if abs(sue) < IN_LINE:
        return "IL"
    if direction == "neutral":
        return None if neutral == "drop" else ("CB" if sue > 0 else "CM")
    return {(True, "positive"): "CB", (True, "negative"): "HB",
            (False, "positive"): "RM", (False, "negative"): "CM"}[(sue > 0, direction)]


def wdiff(d, y, a, b):
    s = d[d.grp.isin([a, b])].copy()
    s["is_a"] = (s.grp == a).astype(float)
    r, n = ols(s, y, ["is_a"], cluster=("day0",), weights="w")
    return r.params.is_a * 100, r.tvalues.is_a


def labels(d, ys=("car_0_1", "car_2_20", "car_2_60")):
    rows = []
    for role in ("master", "master_noreb", "single"):
        for neutral in ("confirmed", "drop"):
            dd = d.assign(grp=[lab(s, x, neutral) for s, x in zip(d.eps_sue, d[f"{role}_dir"])])
            for g, s in dd.dropna(subset=["grp"]).groupby("grp"):
                rows.append({"role": role, "neutral": neutral, "grp": g, "n": len(s),
                             **{y: np.average(s[y].dropna(), weights=s.w[s[y].notna()]) * 100 for y in ys}})
            for name, (a, b) in {"H1 CB-HB": ("CB", "HB"), "H2 RM-CM": ("RM", "CM")}.items():
                row = {"role": role, "neutral": neutral, "grp": name}
                for y in ys:
                    row[y], row[y + "_t"] = wdiff(dd, y, a, b)
                rows.append(row)
    return pd.DataFrame(rows)


def robustness(d, model):
    rows = []

    def reg(tag, s, sig, y="car_2_20", extra=()):
        xs = ["sue_r", sig, *extra, *CTRL]
        r, n = ols(s, y, xs, fe="yq", weights="w")
        rows.append({"check": tag, "signal": sig, "y": y, "n": n, "coef": r.params[sig] * 100,
                     "t": r.tvalues[sig], "ci_lo": r.conf_int().loc[sig, 0] * 100,
                     "ci_hi": r.conf_int().loc[sig, 1] * 100,
                     **{f"{e}_coef": r.params[e] * 100 for e in extra}, **{f"{e}_t": r.tvalues[e] for e in extra}})

    # (a) repeat stability: modal direction over 3 runs, keep events whose 3 runs agree
    rep = pd.read_csv(INTERIM / "agents_repeat_B.csv")
    g = rep.groupby("accession")
    stab = pd.DataFrame({"n_dir": g.master_dir.nunique(), "master_mean3": g.master_score.mean()}).reset_index()
    s = d.merge(stab, on="accession")
    s["master_mean3"] = wz(s.master_mean3, s.w)
    for y in ("car_2_20", "car_2_60"):
        reg("(a) repeat events, run 0", s, "master", y)
        reg("(a) repeat events, mean of 3 runs", s, "master_mean3", y)
        reg("(a) repeat events, 3 runs agree", s[s.n_dir == 1], "master", y)
    # (b) memory: does the signal work better where the model recalls the EPS?
    s = d.dropna(subset=["eps_hit"]).copy()
    s["hit"] = s.eps_hit.astype(float)
    for sig in ("master", "single", "narr"):
        s[f"{sig}_x_hit"] = s[sig] * s.hit
        for y in ("car_2_20", "car_2_60"):
            reg("(b) memory interaction", s, sig, y, extra=("hit", f"{sig}_x_hit"))
    # (c) post-cutoff events only (2025-08 onwards: CAR[2,60] partly missing)
    for sig in ("master", "single", "narr"):
        reg("(c) post-cutoff", d[d.post], sig)
        reg("(c) pre-cutoff", d[~d.post], sig)
    # (d) sub-periods
    for p, (lo, hi) in PERIODS.items():
        for sig in ("master", "single"):
            reg(f"(d) {p}", d[d.yr.between(lo, hi)], sig)
    # (e) drop truncated Master outputs
    tr = truncated("master") | truncated("master_noreb")
    for sig in ("master", "noreb"):
        reg(f"(e) no truncated master ({len(tr & set(d.accession))} dropped)", d[~d.accession.isin(tr)], sig)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ff", choices=["ff", "mb", "ma", "mm"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = build(args.model)
    print(f"stage B: {len(d)} events, CAR[2,20] {d.car_2_20.notna().sum()}, lm {d.lm.notna().sum()}, "
          f"fb {d.fb.notna().sum()}, eps_hit {d.eps_hit.notna().sum()}, post {d.post.sum()}")
    print("weighted corr of signals:\n", pd.DataFrame(
        np.cov(d[list(SIGNALS)].dropna().T, aweights=d.dropna(subset=list(SIGNALS)).w), index=list(SIGNALS),
        columns=list(SIGNALS)).round(2))
    pd.set_option("display.width", 250, "display.max_columns", 40)
    for name, f in (("B1_ablation", ablation), ("B2_labels", labels),
                    ("B3_robustness", lambda x: robustness(x, args.model))):
        t = f(d)
        t.to_csv(OUT / f"{name}_{args.model}.csv", index=False)
        print(f"\n== {name} ({args.model})\n{t.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
