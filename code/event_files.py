"""Event files between pipeline steps and the merges of the per-batch s10 outputs.

  full      data/interim/full_events.csv: matched events with an EPS SUE (s03), minus the development
            samples (pilot_events, audit3_events, audit4_events; drawn with sample_events.py), sorted by
            day 0 and CIK; plus data/interim/stageA/events_{year}.csv, one file per day-0 year for the stage-A runs
  merge-a   data/interim/agents_A_full.csv = the yearly stage-A outputs agents_A_{year}_A.csv stacked
  merge-b   data/interim/agents_stageB_all.csv = gate batch (agents_gate_B.csv) + the rest of stage B
            (agents_B_main_B.csv), with `stratum` and `pi` from stageB_design.csv (s12)
"""

import argparse

import pandas as pd

from common import INTERIM

DEV = ["pilot_events.csv", "audit3_events.csv", "audit4_events.csv"]
YEARS = range(2015, 2026)


def full():
    ev = pd.read_csv(INTERIM / "events_sue.csv")
    dev = pd.concat([pd.read_csv(INTERIM / f).accession for f in DEV])
    f = ev[ev.matched & ev.eps_sue.notna() & ~ev.accession.isin(dev)]
    f = f.sort_values(["day0", "cik"], kind="stable")
    f.to_csv(INTERIM / "full_events.csv", index=False)
    (INTERIM / "stageA").mkdir(exist_ok=True)
    yr = pd.to_datetime(f.day0).dt.year
    for y in YEARS:
        f[yr == y].to_csv(INTERIM / "stageA" / f"events_{y}.csv", index=False)
    print(f"full_events.csv n={len(f)} (dev excluded: {dev.nunique()})")


def merge_a():
    a = pd.concat([pd.read_csv(INTERIM / f"agents_A_{y}_A.csv") for y in YEARS], ignore_index=True)
    assert a.accession.is_unique
    a.to_csv(INTERIM / "agents_A_full.csv", index=False)
    print(f"agents_A_full.csv n={len(a)}")


def merge_b():
    b = pd.concat([pd.read_csv(INTERIM / "agents_gate_B.csv"), pd.read_csv(INTERIM / "agents_B_main_B.csv")],
                  ignore_index=True)
    des = pd.read_csv(INTERIM / "stageB_design.csv")[["accession", "stratum", "pi"]]
    b = b.merge(des, on="accession", how="left")
    assert b.accession.is_unique and b.pi.notna().all()
    b.to_csv(INTERIM / "agents_stageB_all.csv", index=False)
    print(f"agents_stageB_all.csv n={len(b)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["full", "merge-a", "merge-b"])
    args = ap.parse_args()
    {"full": full, "merge-a": merge_a, "merge-b": merge_b}[args.what]()


if __name__ == "__main__":
    main()
