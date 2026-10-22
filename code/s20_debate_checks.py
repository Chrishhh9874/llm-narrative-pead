"""Step 20: further robustness checks (SUE re-import by the debaters, stage B without the gate batch, sample flow). No API calls.

Rows of results/checks/R2_checks_ff.csv (key, n, coef, t), all on the main `_ff` CAR:
  sue_load_{sig}          weighted regression of the z-scored Single / Master score on Narrative tone (z) and the SUE
                          rank: the SUE-rank coefficient measures how much of the surprise the role re-imports beyond
                          the narrative it was given (Single and the debaters see the numbers; Narrative does not)
  flip_{grp}_{sig}        weighted share of a direction among beats with Narrative tone < 0 (hb) and misses with tone
                          > 0 (rm): pos / neg
  nogate_{y}_{spec}_{sig} Table 3 specs without the 996 gate-batch events, reweighted by 1/p_stratum (s12: the
                          non-gate events are drawn with p_stratum from a random 1 - p_gate share of the population)
  nogate_{H1,H2}_{y}      Master-label CB - HB / RM - CM without the gate batch
  loss_*                  why 19,940 Stage A events fall to 18,477 with returns and controls
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import INTERIM, ROOT
from s15_analysis import load
from s16_stageB import SIGNALS, ablation, build, labels, wz

OUT = ROOT / "results" / "checks"


def sue_load(rows, d):
    for sig in ("single", "master", "noreb"):
        s = d.dropna(subset=[sig, "narr", "sue_r"])
        r = sm.WLS(s[sig], sm.add_constant(s[["narr", "sue_r"]]), weights=s.w).fit(cov_type="HC1")
        rows.append({"key": f"sue_load_{sig}", "n": len(s), "coef": r.params.sue_r, "t": r.tvalues.sue_r,
                     "narr": r.params.narr, "r2": r.rsquared})
    for grp, m in {"hb": (d.eps_sue >= 0.5) & (d.narr_tone < 0), "rm": (d.eps_sue <= -0.5) & (d.narr_tone > 0)}.items():
        s = d[m]
        for sig in ("single", "master", "master_noreb"):
            for v in ("positive", "negative"):
                rows.append({"key": f"flip_{grp}_{sig}_{v[:3]}", "n": len(s),
                             "coef": np.average(s[f"{sig}_dir"] == v, weights=s.w)})


def no_gate(rows, d):
    des = pd.read_csv(INTERIM / "stageB_design.csv")[["accession", "in_gate", "p_stratum"]]
    s = d.merge(des, on="accession", how="left")
    s = s[~s.in_gate].copy()
    s["w"] = 1 / s.p_stratum
    for k, c in SIGNALS.items():
        s[k] = wz(s[c], s.w)
    b = ablation(s, ys=("car_0_1", "car_2_20"))
    for _, r in b.iterrows():
        for sig in ("narr", "single", "master"):
            if pd.notna(r.get(sig)):
                rows.append({"key": f"nogate_{r.y}_{r.spec}_{sig}", "n": r.n, "coef": r[sig], "t": r[sig + "_t"],
                             "dr2": r.dr2})
    lb = labels(s, ys=("car_0_1", "car_2_20"))
    lb = lb[(lb.role == "master") & (lb.neutral == "confirmed")]
    for h, g in (("H1", "H1 CB-HB"), ("H2", "H2 RM-CM")):
        r = lb[lb.grp == g].iloc[0]
        for y in ("car_0_1", "car_2_20"):
            rows.append({"key": f"nogate_{h}_{y}", "n": len(s), "coef": r[y], "t": r[y + "_t"]})


def loss(rows):
    d = load("ff")
    nocar = d.car_0_1.isna()
    ctrl = d.mktcap.isna() | d.rev_r.isna() | d.mom.isna() | d.pre_m29_m1.isna()
    for key, m in {"loss_all": nocar | ctrl, "loss_no_car": nocar, "loss_no_price": nocar & d.price_src.isna(),
                   "loss_short_history": nocar & d.price_src.notna(), "loss_ctrl_only": ~nocar & ctrl,
                   "loss_rev_only": ~nocar & d.rev_r.isna()}.items():
        rows.append({"key": key, "n": int(m.sum())})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = build("ff")
    rows = []
    sue_load(rows, d)
    no_gate(rows, d)
    loss(rows)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "R2_checks_ff.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
