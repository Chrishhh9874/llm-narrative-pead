"""Step 17: paper figure 1, cumulative abnormal return paths of the five groups (CB / HB / IL / RM / CM).

Daily ARs from s07 --path (car_path_full.parquet, FF5 + UMD, alpha not subtracted = main model `_ff`).
Constant composition: an event enters only if its AR exists on every day 0..60.
  Stage A (full sample): SUE group x within-quarter tone tercile (s15 groups(..., "rel")), equal weights
  Stage B (5,000 events): Master labels, neutral -> confirmed (s16 lab), weighted by 1/pi
Top row: group mean CAR cumulated from day 0 (anchored at 0 on day -1, so the announcement jump shows).
Bottom row: drift spreads CB - HB and RM - CM of CAR[2, t] with 95% CI, s.e. clustered by day 0 (as T1 / B2).
Encoding: colour = narrative direction (blue positive, orange negative), line style = earnings
(solid beat, dashed miss), grey = in line, so the figure survives grayscale print.

Drawn at its printed size (6.0 in = 15.2 cm wide), so 7.5 pt in the figure is 7.5 pt on paper.
Outputs: results/figures/fig1_car_path.{pdf,png}, results/figures/fig1_car_path.csv (the plotted CAARs)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import INTERIM, ROOT
from s15_analysis import groups, load, ols
from s16_stageB import build, lab

OUT = ROOT / "results" / "figures"
DAYS = range(0, 61)
STYLE = {"CB": ("#2a78d6", "-"), "HB": ("#eb6834", "-"), "RM": ("#2a78d6", "--"), "CM": ("#eb6834", "--"),
         "IL": ("#8a8985", ":")}
NAMES = {"CB": "Confirmed beat", "HB": "Hollow beat", "IL": "In line", "RM": "Resilient miss",
         "CM": "Confirmed miss"}


SPREADS = {"CB − HB": ("CB", "HB", "#1baf7a", "-"), "RM − CM": ("RM", "CM", "#6250d6", "--")}


def caar(ar, grp, w):
    """Weighted mean AR per day within each group, cumulated from day 0 (in %), with day -1 = 0."""
    rows = []
    for g in STYLE:
        m = grp == g
        c0 = np.r_[0, np.cumsum(np.average(ar[m], axis=0, weights=w[m]) * 100)]
        rows += [{"grp": g, "n": int(m.sum()), "day": k, "car": v} for k, v in zip([-1, *DAYS], c0)]
    return pd.DataFrame(rows)


def spreads(d, cols, weights=None):
    """CAR[2, t] spread between the two groups of each SPREADS pair for t = 2..60, s.e. clustered by day 0."""
    c2 = pd.DataFrame(d[cols[2:]].to_numpy().cumsum(axis=1), index=d.index, columns=list(DAYS)[2:])
    rows = []
    for name, (a, b, _, _) in SPREADS.items():
        s = d[d.grp.isin([a, b])].assign(is_a=lambda x: (x.grp == a).astype(float))
        for t in c2.columns:
            r, _ = ols(s.assign(y=c2.loc[s.index, t]), "y", ["is_a"], cluster=("day0",), weights=weights)
            lo, hi = r.conf_int().loc["is_a"]
            rows.append({"spread": name, "day": t, "diff": r.params.is_a * 100, "lo": lo * 100, "hi": hi * 100,
                         "t": r.tvalues.is_a})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = pd.read_parquet(INTERIM / "car_path_full.parquet")
    cols = [f"ar_ff_{k}" for k in DAYS]
    path = path.dropna(subset=cols)[["accession", *cols]]

    a = load("ff")
    a = a.assign(grp=groups(a, 0.5, "rel")).merge(path, on="accession")
    b = build("ff")
    b = b.assign(grp=[lab(s, x, "confirmed") for s, x in zip(b.eps_sue, b.master_dir)]).merge(path, on="accession")
    res = pd.concat([
        caar(a[cols].to_numpy(), a.grp.to_numpy(), np.ones(len(a))).assign(stage="A"),
        caar(b[cols].to_numpy(), b.grp.to_numpy(), b.w.to_numpy()).assign(stage="B")])
    spr = pd.concat([spreads(a, cols).assign(stage="A"), spreads(b, cols, "w").assign(stage="B")])
    res.to_csv(OUT / "fig1_car_path.csv", index=False)
    spr.to_csv(OUT / "fig1_spreads.csv", index=False)
    print(res[res.day.isin([1, 20, 60])].pivot_table(index=["stage", "grp", "n"], columns="day",
                                                       values="car").round(2).to_string())
    print(spr[spr.day.isin([5, 20, 40, 60])].round(2).to_string(index=False))
    print("max |t| of any spread, any day:", spr.groupby(["stage", "spread"]).t.apply(lambda x: x.abs().max()).round(2).to_dict())

    plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "pdf.fonttype": 42})
    fig, ax = plt.subplots(2, 2, figsize=(6.0, 4.4), sharex=True, sharey="row")
    titles = {"A": "Stage A: full sample, tone terciles", "B": "Stage B: Master labels, 1/π-weighted"}
    for j, st in enumerate("AB"):
        for x in ax[:, j]:
            x.axhline(0, color="#c3c2b7", lw=0.8, zorder=0)
            x.grid(axis="y", color="#e8e7e3", lw=0.5)
            x.set_xlim(-1, 60)
            x.set_xticks([0, 10, 20, 30, 40, 50, 60])
        x = ax[0, j]
        x.axvspan(-1, 1, color="#f0efec", lw=0, zorder=-1)  # announcement window [0, 1]
        ends = []
        for g, (c, ls) in STYLE.items():
            s = res[(res.stage == st) & (res.grp == g)]
            x.plot(s.day, s.car, color=c, ls=ls, lw=1.6, label=f"{NAMES[g]} ({g})")
            ends.append([s.car.iat[-1], f"n={s.n.iat[0]:,}"])
        ends.sort()
        for k in range(1, len(ends)):  # push end labels apart
            ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.25)
        for y, txt in ends:
            x.annotate(txt, (60, y), xytext=(3, 0), textcoords="offset points", fontsize=6.3, color="#52514e",
                       va="center", annotation_clip=False)
        x.set_title(titles[st], fontsize=8, loc="left", color="#0b0b0b")
        x = ax[1, j]
        for name, (_, _, c, ls) in SPREADS.items():
            s = spr[(spr.stage == st) & (spr.spread == name)]
            x.fill_between(s.day, s.lo, s.hi, color=c, alpha=0.15, lw=0)
            x.plot(s.day, s["diff"], color=c, ls=ls, lw=1.6, label=f"{name}, 95% CI")
        x.set_xlabel("Trading days after the announcement")
    ax[0, 0].set_ylabel("Mean CAR from day 0 (%)")
    ax[1, 0].set_ylabel("Drift spread, CAR[2, t] (%)")
    ax[1, 0].legend(frameon=False, fontsize=6.8, loc="upper left", handlelength=2.6)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6.8, handlelength=2.6, columnspacing=1.6,
               bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_car_path.{ext}", dpi=300)
    print("saved", OUT / "fig1_car_path.pdf")


if __name__ == "__main__":
    main()
