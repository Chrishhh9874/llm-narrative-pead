"""Step 9: estimate the LLM's effective knowledge cutoff from recall of reported earnings.

DeepSeek does not disclose the training cutoff of V4.1-Flash, so it is measured: for named firms
(real name + ticker, NOT anonymized) the model is asked for the GAAP diluted EPS reported for a given
fiscal quarter. Recall accuracy by calendar quarter of the quarter end should collapse to the
no-knowledge level after the cutoff; events after it are out-of-sample for the model.

Sample: matched events with an EPS value, n per calendar quarter (default 2022Q1 - 2025Q4), excluding
firms with a detected stock split after the quarter end (XBRL EPS is on the latest share basis, so the
announced value would differ). Hit = |guess - true| <= max(0.02, 5% of |true|).
Also asks for the YoY direction of EPS (sign hit vs sign of eps_chg) and a self-rated "knows" flag.

--events FILE runs the same probe on given events instead (per-event EPS recall for the stage-B
recall-strength split). Split firms are then kept: a guess also counts as a hit on the as-announced
basis (XBRL value x yfinance splits after the quarter end), and `split_after` flags them.

Outputs: data/interim/cutoff_probe.csv (or cutoff_probe_{tag}.csv) ; raw calls under llm_raw/probe_cutoff/
"""

import argparse
import asyncio

import numpy as np
import pandas as pd

from common import INTERIM
from llm import LLM, usage_summary

SYS = "You are a financial data assistant. Answer only with a JSON object."
PROMPT = """Company: {name} (ticker {ticker}).
Question: what GAAP diluted earnings per share (EPS) did this company report for its fiscal quarter
ended {qend}? Answer from memory; do not guess wildly - if you do not know, say so.

Return JSON: {{"knows": true or false, "diluted_eps": number or null,
"eps_up_vs_same_quarter_prior_year": true or false or null}}"""


async def run(ev, concurrency=16):
    llm = LLM(concurrency=concurrency, max_tokens=1500)
    tasks = [llm.chat_json("probe_cutoff", r.accession, SYS,
                           PROMPT.format(name=r.name, ticker=r.ticker, qend=r.qend.strftime("%B %d, %Y")))
             for r in ev.itertuples()]
    res = await asyncio.gather(*tasks, return_exceptions=True)
    await llm.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="events per calendar quarter")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--events", help="run on these events (accession column) instead of sampling")
    ap.add_argument("--tag", help="output cutoff_probe_{tag}.csv")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    ev = pd.read_csv(INTERIM / "events_sue.csv", parse_dates=["qend", "day0"])
    ev = ev[ev.matched & ev.eps_val.notna()]
    u = pd.read_csv(INTERIM / "universe.csv").drop_duplicates("cik").set_index("cik")
    ev["name"] = ev.cik.map(u.wiki_name.fillna(u.edgar_name))
    ev["cq"] = ev.qend.dt.to_period("Q").astype(str)
    if args.events:
        ev = ev[ev.accession.isin(pd.read_csv(args.events).accession)]
        renames = pd.read_csv(INTERIM.parent / "raw" / "manual" / "ticker_renames.csv")
        renames = renames.set_index("old_ticker").new_ticker.to_dict()
        split_dir = INTERIM.parent / "raw" / "prices" / "splits"

        def later(r):
            p = split_dir / f"{renames.get(r.ticker, r.ticker)}.csv"
            s = pd.read_csv(p, parse_dates=["date"]) if p.exists() else pd.DataFrame(columns=["date", "ratio"])
            return float(np.prod(s.ratio[s.date > r.qend]))

        ev["split_after"] = ev.apply(later, axis=1)
    else:
        ev = ev[ev.qend.between(args.start, args.end)]
        sp = pd.read_csv(INTERIM / "xbrl_splits.csv", parse_dates=["lo", "hi"])
        later_split = ev.apply(lambda r: ((sp.cik == r.cik) & (sp.hi > r.qend)).any(), axis=1)
        ev = ev[~later_split]
        ev = pd.concat([g.sample(min(args.n, len(g)), random_state=args.seed) for _, g in ev.groupby("cq")])
        ev["split_after"] = 1.0

    res = asyncio.run(run(ev, args.concurrency))
    rows = []
    for r, x in zip(ev.itertuples(), res):
        j = x[0] if not isinstance(x, Exception) and x[0] else {}
        g = j.get("diluted_eps")
        g = float(g) if isinstance(g, (int, float)) else np.nan
        up = j.get("eps_up_vs_same_quarter_prior_year")
        hit = lambda t: bool(np.isfinite(g) and abs(g - t) <= max(0.02, 0.05 * abs(t)))  # noqa: E731
        rows.append({"accession": r.accession, "ticker": r.ticker, "qend": r.qend, "cq": r.cq,
                     "eps_true": r.eps_val, "eps_guess": g, "knows": j.get("knows"), "split_after": r.split_after,
                     "eps_hit": hit(r.eps_val) or hit(r.eps_val * r.split_after),
                     "dir_hit": (up == (r.eps_chg > 0)) if isinstance(up, bool) and pd.notna(r.eps_chg) else np.nan})
    pr = pd.DataFrame(rows)
    pr.to_csv(INTERIM / (f"cutoff_probe_{args.tag}.csv" if args.tag else "cutoff_probe.csv"), index=False)
    s = pr.groupby("cq").agg(n=("eps_hit", "size"), eps_hit=("eps_hit", "mean"),
                             knows=("knows", lambda k: (k == True).mean()), dir_hit=("dir_hit", "mean"))  # noqa: E712
    print(s.round(2).to_string())
    print("probe_cutoff", usage_summary("probe_cutoff"))


if __name__ == "__main__":
    main()
