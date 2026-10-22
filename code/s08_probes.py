"""Step 8: look-ahead-bias probes on the anonymized press releases (paper Section 9).

identity probe  ask the model to name the company / ticker and the fiscal period of the release.
                Hit = guessed ticker equals the true (or renamed) ticker, or guessed name matches an
                alias (name similarity >= 0.8). Period hit = guessed calendar year == quarter-end year.
return probe    ask the model, from the anonymized text only, whether the stock will beat the market
                over trading days [2, 20]; accuracy vs. sign of market-adjusted CAR[2,20].
                ~50% accuracy means the model cannot "recall" outcomes from anonymized text.

--anon anon_masked probes the amount-masked variant; --identity-only skips the return probe,
--return-only skips the identity probe (expanded return probe on 2,000 events).
Task names get the variant suffix (probe_identity_anon_masked, ...) so cached calls never mix variants.

Outputs: data/interim/probes_{tag}.csv ; raw calls under data/interim/llm_raw/probe_*/
"""

import argparse
import asyncio
import re

import numpy as np
import pandas as pd

from common import INTERIM
from llm import LLM, usage_summary
from s01_universe import name_sim

MAX_WORDS = 6000

SYS = "You are an experienced equity analyst. Answer only with a JSON object."
RELEASE_BLOCK = """Earnings press release (company, product and person names, tickers and dates are masked):
<<<
{text}
>>>

"""
ID_PROMPT = RELEASE_BLOCK + """Task: use any clue in the text above (industry, segments, figures, wording) to identify it.

Return JSON: {{"company_name": string or null, "ticker": string or null,
"fiscal_quarter": "Q1".."Q4" or null, "calendar_year": integer or null,
"confidence": number between 0 and 1}}"""
RET_PROMPT = RELEASE_BLOCK + """Task: based only on the text above, will the company's stock outperform the overall
market over the period from the 2nd to the 20th trading day after the announcement?

Return JSON: {{"direction": "outperform" or "underperform", "probability_outperform": number between 0 and 1}}"""


def load_text(anon_dir, acc):
    w = (anon_dir / f"{acc}.txt").read_text().split(" ")
    return " ".join(w[:MAX_WORDS])


async def run(ev, anon_dir, t_id, t_ret, concurrency=16):
    llm = LLM(concurrency=concurrency, max_tokens=2000)
    tasks = []
    for r in ev.itertuples():
        text = load_text(anon_dir, r.accession)
        tasks.append(llm.chat_json(t_id, r.accession, SYS, ID_PROMPT.format(text=text)) if t_id else asyncio.sleep(0, ({}, None)))
        tasks.append(llm.chat_json(t_ret, r.accession, SYS, RET_PROMPT.format(text=text)) if t_ret else asyncio.sleep(0, ({}, None)))
    res = await asyncio.gather(*tasks, return_exceptions=True)
    await llm.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=str(INTERIM / "pilot_events.csv"))
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--anon", default="anon", help="text dir under data/interim (anon | anon_masked)")
    ap.add_argument("--identity-only", action="store_true")
    ap.add_argument("--return-only", action="store_true")
    ap.add_argument("--car", default=None, help="CAR file (default car_{tag}.csv)")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    anon_dir = INTERIM / args.anon
    sfx = f"_{args.anon}"
    t_id = None if args.return_only else f"probe_identity{sfx}"
    t_ret = None if args.identity_only else f"probe_return{sfx}"
    ev = pd.read_csv(args.events, parse_dates=["qend"])
    ev = ev[[(anon_dir / f"{a}.txt").exists() for a in ev.accession]]
    res = asyncio.run(run(ev, anon_dir, t_id, t_ret, args.concurrency))

    u = pd.read_csv(INTERIM / "universe.csv")
    renames = pd.read_csv(INTERIM.parent / "raw" / "manual" / "ticker_renames.csv")
    car = pd.read_csv(args.car or INTERIM / f"car_{args.tag}.csv")
    rows = []
    for i, r in enumerate(ev.itertuples()):
        idr, ret = res[2 * i], res[2 * i + 1]
        idj = idr[0] if not isinstance(idr, Exception) and idr[0] else {}
        rj = ret[0] if not isinstance(ret, Exception) and ret[0] else {}
        true_t = {r.ticker} | set(renames.loc[renames.old_ticker == r.ticker, "new_ticker"])
        names = u.loc[u.cik == r.cik, ["wiki_name", "edgar_name"]].stack().tolist()
        g_t = str(idj.get("ticker") or "").upper().strip().replace("NYSE:", "").replace("NASDAQ:", "").strip()
        g_n = idj.get("company_name") or ""
        rows.append({
            "accession": r.accession, "ticker": r.ticker, "qend": r.qend,
            "id_ticker": g_t, "id_name": g_n, "id_conf": idj.get("confidence"),
            "id_year": idj.get("calendar_year"), "id_quarter": idj.get("fiscal_quarter"),
            "id_hit": bool(g_t in true_t) or (bool(g_n) and max([name_sim(g_n, x) for x in names] or [0]) >= 0.8),
            "id_year_hit": idj.get("calendar_year") == r.qend.year,
            "ret_dir": rj.get("direction"), "ret_p": rj.get("probability_outperform"),
        })
    pr = pd.DataFrame(rows).merge(car[["accession", "car_2_20"]], on="accession", how="left")
    pr["ret_correct"] = np.where(pr.car_2_20.notna() & pr.ret_dir.notna(),
                                 (pr.ret_dir == "outperform") == (pr.car_2_20 > 0), np.nan)
    pr.to_csv(INTERIM / f"probes_{args.tag}{sfx}.csv", index=False)

    n = len(pr)
    print(f"n={n}  identity hit={pr.id_hit.mean():.1%}  year hit={pr.id_year_hit.mean():.1%}  "
          f"guessed-any-name={(pr.id_name.str.len() > 0).mean():.1%}")
    v = pr.dropna(subset=["ret_correct"])
    k = int(v.ret_correct.sum())
    from scipy.stats import binomtest
    if len(v):
        print(f"return probe: accuracy={k / max(len(v), 1):.1%} (n={len(v)}, binomial p vs 0.5="
              f"{binomtest(k, len(v), 0.5).pvalue:.3f}); share 'outperform'={(v.ret_dir == 'outperform').mean():.1%}")
    for t in filter(None, (t_id, t_ret)):
        print(t, usage_summary(t))


if __name__ == "__main__":
    main()
