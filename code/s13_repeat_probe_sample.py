"""Step 13: samples for the repeat runs and the expanded return probe.

repeat_events.csv    500 stage-B events, stratified by stage-B stratum (proportional), run with
                     `s10_agents.py --repeats 3` (repeat 0 is the cached main run)
probe_ret_events.csv 2,000 non-dev events with text, stratified by year (proportional), for the
                     return probe `s08_probes.py --return-only`
"""

import argparse

import numpy as np
import pandas as pd

from common import INTERIM


def stratified(df, by, n, rng):
    """Proportional allocation over `by` with largest-remainder rounding."""
    cells = df.groupby(by).size()
    q = np.floor(cells * n / len(df)).astype(int)
    q[(cells * n / len(df) - q).sort_values(ascending=False).index[: n - q.sum()]] += 1
    return pd.concat([g.sample(q[k], random_state=rng.integers(1 << 31)) for k, g in df.groupby(by)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    ev = pd.read_csv(INTERIM / "full_events.csv")

    b = pd.read_csv(INTERIM / "agents_stageB_all.csv")[["accession", "stratum"]]
    rep = stratified(b, "stratum", 500, rng)
    ev[ev.accession.isin(rep.accession)].to_csv(INTERIM / "repeat_events.csv", index=False)
    print("repeat sample by stratum:", rep.stratum.value_counts().to_dict())

    a = pd.read_csv(INTERIM / "agents_A_full.csv", parse_dates=["day0"])[["accession", "day0"]]
    a["yr"] = a.day0.dt.year
    pr = stratified(a, "yr", 2000, rng)
    ev[ev.accession.isin(pr.accession)].to_csv(INTERIM / "probe_ret_events.csv", index=False)
    print("return-probe sample by year:", pr.yr.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
