"""Step 12: draw the rest of the stage-B subsample.

Stage B = gate batch (996, simple random, run first) plus, from the remaining non-dev events:
  post      all events with day0 >= 2025-08-01 (after the model's knowledge cutoff)
  hb_cand   all beats (EPS SUE >= 0.5) whose Narrative tone is in the bottom 20% of their year's beats
  rm_cand   1,000 of the misses (EPS SUE <= -0.5) whose tone is in the top 40% of their year's misses
  random    the rest of the 5,000, stratified by year x within-year market-cap tercile (proportional)
Tone percentiles come from stage A (Narrative never sees SUE); thresholds 0.5 / 20% / 40% were picked on
the gate batch's Master directions (no CAR). `pi` is each event's inclusion probability in stage B
(gate and stratum draws are independent), for inverse-probability weighting.

Inputs : agents_A_full.csv, agents_gate_B.csv, full_events.csv, mktcap.csv
Outputs: data/interim/stageB_events.csv (events still to run), stageB_design.csv (all 5,000 + pi)
"""

import argparse

import numpy as np
import pandas as pd

from common import INTERIM

N_TOTAL = 5000
N_RM = 1000
SUE_CUT = 0.5
HB_Q, RM_Q = 0.2, 0.6
CUTOFF = "2025-08-01"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    a = pd.read_csv(INTERIM / "agents_A_full.csv", parse_dates=["day0"])[["accession", "day0", "eps_sue", "narr_tone"]]
    a = a.merge(pd.read_csv(INTERIM / "mktcap.csv")[["accession", "mktcap"]], on="accession", how="left")
    a["yr"] = a.day0.dt.year
    a["grp"] = np.select([a.eps_sue >= SUE_CUT, a.eps_sue <= -SUE_CUT], ["beat", "miss"], "inline")
    a["tone_q"] = a.groupby(["yr", "grp"]).narr_tone.rank(pct=True)
    a["size"] = a.groupby("yr").mktcap.transform(lambda x: pd.qcut(x, 3, labels=["S", "M", "L"])).astype(str)
    a.loc[a.mktcap.isna(), "size"] = "NA"

    gate = set(pd.read_csv(INTERIM / "agents_gate_B.csv").accession)
    a["in_gate"] = a.accession.isin(gate)
    p_gate = len(gate) / len(a)
    a["stratum"] = np.select([a.day0 >= CUTOFF, (a.grp == "beat") & (a.tone_q <= HB_Q),
                              (a.grp == "miss") & (a.tone_q > RM_Q)], ["post", "hb_cand", "rm_cand"], "random")

    pool = a[~a.in_gate]
    pick, p_str = [], {}
    for s in ("post", "hb_cand"):
        pick.append(pool[pool.stratum == s])
        p_str[s] = 1.0
    rm = pool[pool.stratum == "rm_cand"]
    pick.append(rm.sample(N_RM, random_state=rng.integers(1 << 31)))
    p_str["rm_cand"] = N_RM / len(rm)
    n_rand = N_TOTAL - len(gate) - sum(len(x) for x in pick)
    rest = pool[pool.stratum == "random"]
    frac = n_rand / len(rest)
    # proportional allocation over year x size cells, largest-remainder rounding to hit n_rand exactly
    cells = rest.groupby(["yr", "size"]).size()
    quota = np.floor(cells * frac).astype(int)
    quota[(cells * frac - quota).sort_values(ascending=False).index[: n_rand - quota.sum()]] += 1
    for (yr, sz), q in quota.items():
        c = rest[(rest.yr == yr) & (rest["size"] == sz)]
        pick.append(c.sample(q, random_state=rng.integers(1 << 31)))
    new = pd.concat(pick)

    a["p_stratum"] = a.stratum.map(p_str)
    cell_p = (quota / cells).rename("p_cell").reset_index()
    a = a.merge(cell_p, on=["yr", "size"], how="left")
    a.loc[a.stratum == "random", "p_stratum"] = a.p_cell
    a["pi"] = 1 - (1 - p_gate) * (1 - a.p_stratum)
    a["in_stageB"] = a.in_gate | a.accession.isin(new.accession)
    design = a[a.in_stageB].drop(columns="p_cell")
    design.to_csv(INTERIM / "stageB_design.csv", index=False)

    ev = pd.read_csv(INTERIM / "full_events.csv")
    ev[ev.accession.isin(new.accession)].to_csv(INTERIM / "stageB_events.csv", index=False)
    print(f"stage B: {len(design)} events = gate {len(gate)} + new {len(new)}")
    print(design.groupby(["stratum", "in_gate"]).size().unstack(fill_value=0).to_string())
    print("new random by size:", new[new.stratum == "random"]["size"].value_counts().to_dict())
    print(f"pi range {design.pi.min():.4f}-{design.pi.max():.4f}; p_gate {p_gate:.4f}, p_rm {p_str['rm_cand']:.3f}, "
          f"p_random {frac:.4f}")


if __name__ == "__main__":
    main()
