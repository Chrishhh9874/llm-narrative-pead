"""Draw the 200-event pilot: matched earnings events with an XBRL EPS SUE, stratified by year.

Output: data/interim/pilot_events.csv
  --out / --exclude: a further sample outside earlier ones, e.g. the second development sample
    (--n 100 --seed 20261024 --exclude data/interim/pilot_events.csv --out audit3_events.csv)
  --need-text: the stage-B gate batch draws only events whose text went through s06
"""

import argparse

import pandas as pd

from common import INTERIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--out", default="pilot_events.csv", help="file name under data/interim/")
    ap.add_argument("--exclude", nargs="*", default=[], help="event files whose accessions are left out")
    ap.add_argument("--need-text", action="store_true", help="only events with an anonymized text in data/interim/anon/")
    args = ap.parse_args()

    ev = pd.read_csv(INTERIM / "events_sue.csv", parse_dates=["day0", "qend"])
    pool = ev[ev.matched & ev.eps_sue.notna()]
    if args.need_text:
        pool = pool[[(INTERIM / "anon" / f"{a}.txt").exists() for a in pool.accession]]
    for f in args.exclude:
        pool = pool[~pool.accession.isin(pd.read_csv(f).accession)]
    years = sorted(pool.year.unique())
    per = args.n // len(years)
    parts = [g.sample(min(per, len(g)), random_state=args.seed) for _, g in pool.groupby("year")]
    s = pd.concat(parts)
    if len(s) < args.n:  # top up from the remaining pool
        s = pd.concat([s, pool.drop(s.index).sample(args.n - len(s), random_state=args.seed)])
    s = s.sort_values(["day0", "cik"])
    s.to_csv(INTERIM / args.out, index=False)
    print(f"{args.out} n={len(s)} firms={s.cik.nunique()} from pool={len(pool)}")
    print(s.year.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
