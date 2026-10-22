"""Step 10: multi-agent adjudication of the anonymized press releases (paper Section 4).

Agents (one model, deepseek-flash, for every LLM role; roles differ only by the instruction block)
  numeric     Python, no LLM: EPS / revenue SUE and year-on-year changes -> a short numeric summary
  narrative   LLM, text only (never sees the numeric summary, so its tone is independent of SUE):
              tone, guidance direction, risk flags, evidence paragraphs
  single      LLM, one call reading text + numeric summary -> same verdict schema as master (Single ablation)
  bull / bear LLM, text + numbers + narrative output -> opening arguments
  bull_reb / bear_reb   one rebuttal round: each side answers the other's opening
  master      LLM, everything above -> narrative verdict, adjudicated score, evidence paragraphs
  master_noreb  master without the rebuttal round (no-rebuttal ablation: does the debate itself matter)

Stage A (full sample): narrative only, 1 call per event.  Stage B (sub-sample): all 8 calls.
The 2x2 + in-line label is formed only here, from the SUE sign and the narrative direction (`label()`); the
prompts carry neither the in-line threshold nor any beat / miss category, so IN_LINE is an analysis
parameter that can change without re-running the LLM.

Token economy: the document is trimmed to MAX_WORDS (headline, opening paragraphs, quotes and
guidance first, then the remaining narrative in document order) and placed right after an identical
system prompt in every call, so all calls of an event share one cacheable prefix. Calls of one event are
sequenced (narrative first) so later calls can hit the prefix cache written by the first one.

Calls are cached on disk by llm.py (task = agent_{role}_{anon}); re-runs only send missing calls.
Real runs refuse to start in DeepSeek peak hours unless --allow-peak.

  --dry-run N   print the prompts of N events and run the whole pipeline on a fake LLM (no API call)

Outputs: data/interim/agents_{tag}.csv ; raw calls under data/interim/llm_raw/agent_*/
"""

import argparse
import asyncio
import json
import re
import time

import numpy as np
import pandas as pd

from common import INTERIM
from llm import LLM, is_peak, usage_summary

MAX_WORDS = 1500
MIN_WORDS = 50  # s04 extracted (almost) no narrative from 37 filings (e.g. DRE 2017-19): skip them
IN_LINE = 0.5  # analysis parameter only (never in a prompt): |EPS SUE| below this is labelled In-line
KEEP = {"narrative", "guidance", "heading"}
QUOTE = re.compile(r"[“\"]|\bsaid\b")
ROLES_A = ["narrative"]
ROLES_B = ["narrative", "single", "bull", "bear", "bull_reb", "bear_reb", "master", "master_noreb"]
GUIDANCE_DIR = ["raise", "maintain", "lower", "mixed", "initiate", "none"]
RISKS = ["demand", "pricing", "margin", "costs", "supply_chain", "fx", "rates_credit", "regulation_legal",
         "impairment_writedown", "restructuring", "management_change", "macro", "competition", "liquidity_leverage"]

SYS = """You are a senior equity research analyst reviewing a quarterly earnings press release. \
Answer only with a JSON object.

Conventions of the text you will see:
- The release is anonymized: the issuer is "Company X"; other people, organizations and products are \
"Executive A", "Organization A", "Product A", ...; full dates are "[DATE]"; addresses, tickers and links are removed.
- Years are shifted by one common offset: "Year t" is the fiscal year being reported, "Year t-1" the year \
before, "Year t+1" the next. These labels are shifted calendar years; the company's fiscal year may differ \
from the calendar year (e.g. "June Year t" can fall in fiscal Year t+1).
- Paragraphs are numbered [P1], [P2], ...; cite them by number. Financial statements and legal boilerplate \
were removed, and long releases were shortened, so some numbered paragraphs may be missing.
- Judge only what the text and the numbers given to you say. Do not predict stock prices or returns, and \
do not use any knowledge you may have about which company this is or what happened afterwards."""

# judging rules shared by single and master (set on the development sample), so the Single ablation stays like-for-like;
# narrative_direction is the narrative's own sign (the paper's 2x2), not its sign relative to the surprise
RULES = """How to judge:
- "The earnings surprise" is the EPS SUE in the numeric summary (GAAP diluted EPS vs. the same quarter last \
year), not revenue and not adjusted EPS.
- First separate disclosed one-off items in either period (tax remeasurements, gains or losses on sales, \
impairments, legal items, deal costs, mark-to-market effects) from recurring operations, and judge the \
recurring business. A GAAP beat that comes only from one-offs or a depressed prior-year base is not by itself \
positive evidence; a GAAP miss caused by one-offs is not by itself negative evidence. Do not count a one-off again as a forward-looking negative unless the text says it will recur. \
Writedowns caused by an ongoing price or demand downturn are a fundamental signal, not a one-off.
- Check the comparison basis of every change you rely on (year-on-year vs. sequential); judge seasonal \
businesses against the same quarter last year.
- Guidance: raised = positive; lowered or withdrawn = negative; reaffirmed, or narrowed around the same \
midpoint = neutral, unless the text shows the rest of the year implied by it is clearly weaker or stronger than \
the current quarter. Slower growth that the text attributes to an acquisition anniversary or a tough comparison \
period is not deterioration.
- Upbeat management tone is the norm in press releases, not evidence by itself. Answer "neutral" when there is \
no forward guidance and the positives are mainly tone, or when positives and negatives weigh about the same.
- Your narrative_direction must agree with your own rationale.
"""

DOC = """Earnings press release:
<<<
{doc}
>>>

"""

NUMERIC = """Numeric summary (computed from the company's XBRL filings, not from the text):
{numeric}

"""

TASK = {
    "narrative": """Task (Narrative analyst): read only the release above and assess management's narrative: \
tone of the discussion, any forward-looking guidance, and risks or weaknesses it reveals, including ones that \
are downplayed or buried. Do not judge whether the reported numbers are good or bad.

Return JSON:
{{"tone": number in [-1, 1] (-1 very negative, 0 neutral, 1 very positive),
"guidance_dir": one of {gdir} (direction of the company's own outlook vs. its prior outlook; "initiate" if new \
numeric guidance without a prior reference; "none" if no numeric guidance or reaffirmation),
"risk_flags": list of items from {risks} (empty if none),
"evidence": up to 3 paragraph numbers that drive the assessment, e.g. [3, 12],
"confidence": number in [0, 1],
"rationale": at most 40 words}}""",

    "single": """Task (Analyst, single pass): weigh the reported numbers against management's narrative. Setting aside \
one-off items, does the qualitative information (tone, guidance, risks) point to better or worse future fundamentals for the company? Judge the narrative's own direction, not \
whether it is better or worse than the earnings surprise: after a beat, a positive narrative confirms it and a \
negative one undercuts it; after a miss, a positive narrative points to recovery and a negative one confirms it.

{rules}
Return JSON:
{{"narrative_direction": "positive" | "negative" | "neutral",
"score": number in [-1, 1] (net signal of the narrative about future fundamentals),
"evidence": 1-3 paragraph numbers,
"confidence": number in [0, 1],
"rationale": at most 50 words}}""",

    "bull": """Narrative analyst's assessment of the release:
{narrative}

Task (Bull advocate): build the strongest evidence-based case that the release is GOOD news for the \
company's future fundamentals, beyond what the earnings surprise shows. Use only facts in the text and the \
numbers; do not invent facts.

Return JSON:
{{"thesis": at most 25 words,
"points": 2-4 items of {{"claim": at most 25 words, "evidence": paragraph numbers}},
"strength": number in [0, 1] (how strong your own case honestly is)}}""",

    "bear": """Narrative analyst's assessment of the release:
{narrative}

Task (Bear advocate): build the strongest evidence-based case that the release is BAD news for the \
company's future fundamentals, beyond what the earnings surprise shows (e.g. low-quality beats, one-offs, \
weak guidance, buried risks). Use only facts in the text and the numbers; do not invent facts.

Return JSON:
{{"thesis": at most 25 words,
"points": 2-4 items of {{"claim": at most 25 words, "evidence": paragraph numbers}},
"strength": number in [0, 1] (how strong your own case honestly is)}}""",

    "bull_reb": """Your opening case (Bull):
{own}

Opposing case (Bear):
{other}

Task (Bull advocate, rebuttal): answer the Bear's points. Concede the points that the text supports and \
rebut the ones it does not, citing paragraphs.

Return JSON:
{{"rebuttals": 1-4 items of {{"target": index of the Bear point (1-based), "response": at most 30 words, \
"evidence": paragraph numbers}},
"concessions": list of Bear point indices you accept,
"strength_after": number in [0, 1]}}""",

    "bear_reb": """Your opening case (Bear):
{own}

Opposing case (Bull):
{other}

Task (Bear advocate, rebuttal): answer the Bull's points. Concede the points that the text supports and \
rebut the ones it does not, citing paragraphs.

Return JSON:
{{"rebuttals": 1-4 items of {{"target": index of the Bull point (1-based), "response": at most 30 words, \
"evidence": paragraph numbers}},
"concessions": list of Bull point indices you accept,
"strength_after": number in [0, 1]}}""",

    "master": """Narrative analyst:
{narrative}

Bull case:
{bull}

Bear case:
{bear}
{rebuttals}
Task (Chief analyst, adjudication): weigh the debate against the text and the numbers. Rule on which side \
the evidence supports. Then, setting aside one-off items, decide whether the qualitative information (tone, \
guidance, risks) points to better or worse future fundamentals for the company. Judge the narrative's own direction, not \
whether it is better or worse than the earnings surprise: after a beat, a positive narrative confirms it and a \
negative one undercuts it; after a miss, a positive narrative points to recovery and a negative one confirms it. Your verdict must rest on \
paragraphs of the release, not on the advocates' wording: check every advocate claim against the paragraph \
it cites before relying on it, and do not adopt the advocates' adjectives.

{rules}
Return JSON:
{{"narrative_direction": "positive" | "negative" | "neutral",
"score": number in [-1, 1] (net signal of the narrative about future fundamentals),
"winner": "bull" | "bear" | "draw",
"evidence": 1-3 paragraph numbers,
"confidence": number in [0, 1],
"rationale": at most 50 words}}""",
}
REBUTTALS = """
Bull rebuttal:
{bull_reb}

Bear rebuttal:
{bear_reb}
"""


# ---------------------------------------------------------------- inputs

def align_types(anon, kept):
    """Paragraph types for the anonymized paragraphs. s06 drops paragraphs that become empty, so when the
    counts differ, match each anonymized paragraph to the next kept paragraph with the most shared words."""
    if len(anon) == len(kept):
        return [k["type"] for k in kept]
    out, j = [], 0
    for a in anon:
        wa = set(a.lower().split())
        cands = range(j, min(j + 4, len(kept)))
        best = max(cands, key=lambda i: len(wa & set(kept[i]["text"].lower().split())), default=None)
        if best is None:
            out.append("narrative")
            continue
        out.append(kept[best]["type"])
        j = best + 1
    return out


def build_doc(acc, anon_dir, max_words=MAX_WORDS):
    """Numbered, trimmed document text. Priority: headings and the first two narrative paragraphs, guidance,
    quotes, then the remaining narrative in document order; headings are kept only if a paragraph under
    them is kept. Returns (doc, n_paragraphs, words_kept, words_total)."""
    paras = (anon_dir / f"{acc}.txt").read_text().split("\n\n")
    kept = [p for p in json.loads((INTERIM / "ex991" / f"{acc}.json").read_text())["paragraphs"]
            if p["type"] in KEEP]
    types = align_types(paras, kept)
    words = [len(p.split()) for p in paras]
    total = sum(words)
    first_narr = [i for i, t in enumerate(types) if t == "narrative"][:2]
    tiers = [
        [i for i, t in enumerate(types) if t == "heading" and i < 3] + first_narr,
        [i for i, t in enumerate(types) if t == "guidance"],
        [i for i, t in enumerate(types) if t == "narrative" and QUOTE.search(paras[i])],
        [i for i, t in enumerate(types) if t in ("narrative", "guidance")],
    ]
    take, used = set(), 0
    for tier in tiers:
        for i in tier:
            if i not in take and used + words[i] <= max_words:
                take.add(i)
                used += words[i]
    # a heading is useful only if something under it survived; headline headings (i < 3) always stay
    for i, t in enumerate(types):
        if t == "heading" and i >= 3:
            nxt = next((k for k in range(i + 1, len(types)) if types[k] != "heading"), None)
            if nxt in take and used + words[i] <= max_words:
                take.add(i)
                used += words[i]
    doc = "\n\n".join(f"[P{i + 1}] {paras[i]}" for i in sorted(take))
    return doc, len(paras), used, total


def numeric_summary(r):
    def pct(chg, base):
        return f"{chg / abs(base):+.1%}" if pd.notna(chg) and pd.notna(base) and base != 0 else "n/a"

    def sue(x):
        return f"{x:+.2f}" if pd.notna(x) else "n/a"

    return "\n".join([
        f"- Diluted EPS (GAAP) vs. same quarter last year: change {r.eps_chg:+.2f} per share "
        f"({pct(r.eps_chg, r.eps_lag4)}); standardized surprise SUE = {sue(r.eps_sue)}",
        f"- Revenue vs. same quarter last year: {pct(r.rev_chg, r.rev_lag4)}; revenue SUE = {sue(r.rev_sue)}",
        "- SUE = year-on-year change divided by its std over the prior 8 quarters; "
        "positive = above the seasonal random-walk expectation",
    ])


def compact(x):
    return json.dumps(x, ensure_ascii=False, separators=(",", ":")) if x is not None else "null"


def prompts_for(role, doc, numeric, out):
    """(user prompt) of one role. `out` holds the parsed outputs of the roles that ran before."""
    def opts(xs):
        return " | ".join(f'"{x}"' for x in xs)

    fill = dict(gdir=opts(GUIDANCE_DIR), risks=opts(RISKS), narrative=compact(out.get("narrative")),
                rules=RULES)
    head = DOC.format(doc=doc)
    if role == "narrative":
        return head + TASK[role].format(**fill)
    head += NUMERIC.format(numeric=numeric)
    if role in ("bull_reb", "bear_reb"):
        me, other = role[:4], ("bear" if role.startswith("bull") else "bull")
        return head + TASK[role].format(own=compact(out.get(me)), other=compact(out.get(other)))
    if role.startswith("master"):
        reb = "" if role == "master_noreb" else REBUTTALS.format(bull_reb=compact(out.get("bull_reb")),
                                                                 bear_reb=compact(out.get("bear_reb")))
        return head + TASK["master"].format(bull=compact(out.get("bull")), bear=compact(out.get("bear")),
                                            rebuttals=reb, **fill)
    return head + TASK[role].format(**fill)


# ---------------------------------------------------------------- pipeline

class FakeLLM:
    """Stand-in for --dry-run: returns a schema-shaped placeholder, records the prompts, sends nothing."""

    def __init__(self):
        self.calls = []

    async def chat_json(self, task, key, system, user, repeat=0):
        self.calls.append((task, key, repeat, system, user))
        role = task.split("_", 1)[1].rsplit("_anon", 1)[0]
        fake = {"narrative": {"tone": 0.2, "guidance_dir": "maintain", "risk_flags": ["margin"], "evidence": [1],
                              "confidence": 0.7, "rationale": "dry run"},
                "bull": {"thesis": "dry run", "points": [{"claim": "x", "evidence": [1]}], "strength": 0.6},
                "bull_reb": {"rebuttals": [{"target": 1, "response": "x", "evidence": [1]}], "concessions": [],
                             "strength_after": 0.5}}
        fake["bear"], fake["bear_reb"] = fake["bull"], fake["bull_reb"]
        verdict = {"narrative_direction": "positive", "score": 0.3, "winner": "bull",
                   "evidence": [1], "confidence": 0.6, "rationale": "dry run"}
        return fake.get(role, verdict), None

    async def close(self):
        pass


async def run_event(llm, r, roles, anon, anon_dir, repeat):
    doc, n_par, w_kept, w_total = build_doc(r.accession, anon_dir)
    numeric = numeric_summary(r)
    out = {}

    async def call(role):
        p, _ = await llm.chat_json(f"agent_{role}_{anon}", r.accession, SYS,
                                   prompts_for(role, doc, numeric, out), repeat=repeat)
        out[role] = p

    # narrative first: it writes the shared prefix to the cache and feeds every later role
    await call("narrative")
    later = [x for x in ("single", "bull", "bear") if x in roles]
    await asyncio.gather(*map(call, later))
    if "bull_reb" in roles:
        await asyncio.gather(call("bull_reb"), call("bear_reb"))
    await asyncio.gather(*map(call, [x for x in ("master", "master_noreb") if x in roles]))
    return {"n_par": n_par, "words_kept": w_kept, "words_total": w_total}, out


async def run(ev, roles, anon, repeats, llm):
    async def one(r, k):
        try:
            return await run_event(llm, r, roles, anon, INTERIM / anon, k)
        except Exception as e:  # one failed event must not stop the batch; it is re-sent on the next run
            return {"error": repr(e)[:200]}, {}

    res = await asyncio.gather(*[one(r, k) for k in range(repeats) for r in ev.itertuples()])
    await llm.close()
    return res


def cited_ok(ev_list, n_par):
    try:
        if isinstance(ev_list, str):  # once written as one string "P7, P11, P20"
            ev_list = re.split(r"[,\s]+", ev_list.strip())
        ids = [int(str(x).lstrip("P")) for x in ev_list]  # the model writes 5 or "P5"
    except (TypeError, ValueError):
        return False
    return bool(ids) and all(1 <= i <= n_par for i in ids)


def flatten(r, k, meta, out):
    row = {"accession": r.accession, "ticker": r.ticker, "day0": r.day0, "repeat": k,
           "eps_sue": r.eps_sue, "rev_sue": r.rev_sue, **meta}
    n = out.get("narrative") or {}
    row.update(narr_tone=n.get("tone"), narr_guidance=n.get("guidance_dir"),
               narr_risks=";".join(n.get("risk_flags") or []), narr_conf=n.get("confidence"),
               narr_evidence_ok=cited_ok(n.get("evidence"), meta.get("n_par", 0)))
    for role in ("single", "master", "master_noreb"):
        v = out.get(role) or {}
        row.update({f"{role}_dir": v.get("narrative_direction"), f"{role}_score": v.get("score"),
                    f"{role}_conf": v.get("confidence"),
                    f"{role}_evidence_ok": cited_ok(v.get("evidence"), meta.get("n_par", 0)) if v else None})
        if role.startswith("master"):
            row[f"{role}_winner"] = v.get("winner")
    for side in ("bull", "bear"):
        row[f"{side}_strength"] = (out.get(side) or {}).get("strength")
        row[f"{side}_strength_after"] = (out.get(f"{side}_reb") or {}).get("strength_after")
    return row


def label(sue, direction):
    if pd.isna(sue) or direction not in ("positive", "negative", "neutral"):
        return None
    if abs(sue) < IN_LINE:
        return "In-line"
    if direction == "neutral":  # neutral narrative: the numbers speak for themselves
        return "Confirmed Beat" if sue > 0 else "Confirmed Miss"
    return {(True, "positive"): "Confirmed Beat", (True, "negative"): "Hollow Beat",
            (False, "positive"): "Resilient Miss", (False, "negative"): "Confirmed Miss"}[(sue > 0, direction)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=str(INTERIM / "pilot_events.csv"))
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--anon", default="anon", help="text dir under data/interim (main pipeline: anon)")
    ap.add_argument("--stage", choices=["A", "B"], default="B")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, help="first N events only")
    ap.add_argument("--concurrency", type=int, default=500)
    ap.add_argument("--dry-run", type=int, metavar="N", help="print prompts of N events; fake LLM, no API call")
    ap.add_argument("--allow-peak", action="store_true")
    args = ap.parse_args()

    roles = ROLES_A if args.stage == "A" else ROLES_B
    ev = pd.read_csv(args.events)
    def has_text(a):
        p = INTERIM / args.anon / f"{a}.txt"
        return p.exists() and len(p.read_text().split()) >= MIN_WORDS

    ev = ev[np.array([has_text(a) for a in ev.accession]) & ev.eps_sue.notna()]
    if args.limit:
        ev = ev.head(args.limit)

    if args.dry_run:
        ev = ev.head(args.dry_run)
        llm = FakeLLM()
        res = asyncio.run(run(ev, roles, args.anon, 1, llm))
        for task, key, _, system, user in llm.calls:
            print(f"\n{'=' * 30} {task} / {key}  (~{int(1.35 * len((system + user).split()))} tokens)\n")
            print(user if task.endswith("narrative_" + args.anon) else user[user.index("Numeric summary"):])
        print(f"\n{'=' * 30} SYSTEM\n{SYS}")
        print(pd.DataFrame([flatten(r, 0, *res[i]) for i, r in enumerate(ev.itertuples())]).T)
        return

    if is_peak(time.time()) and not args.allow_peak:
        raise SystemExit("DeepSeek peak hours (Beijing weekday 9-12 / 14-18): run later or pass --allow-peak")
    llm = LLM(concurrency=args.concurrency, max_tokens=4000)
    t0 = time.time()
    res = asyncio.run(run(ev, roles, args.anon, args.repeats, llm))
    print(f"wall time {(time.time() - t0) / 60:.1f} min at concurrency {args.concurrency}; "
          f"retries: {dict(llm.retries) or 'none'}")

    rows, i = [], 0
    for k in range(args.repeats):
        for r in ev.itertuples():
            rows.append(flatten(r, k, *res[i]))
            i += 1
    df = pd.DataFrame(rows)
    for role in ("single", "master", "master_noreb"):
        df[f"{role}_label"] = [label(s, d) for s, d in zip(df.eps_sue, df[f"{role}_dir"])]
    out = INTERIM / f"agents_{args.tag}_{args.stage}.csv"
    df.to_csv(out, index=False)

    print(f"{len(ev)} events x {args.repeats} repeats -> {out}")
    if "error" in df:
        print(f"failed events: {df.error.notna().sum()} (re-run to retry)")
    print(f"words kept: median {df.words_kept.median():.0f} / total {df.words_total.median():.0f}")
    if "master" in roles:
        print("master label:", df.master_label.value_counts().to_dict())
    print(f"narrative tone: mean {df.narr_tone.mean():+.2f}, sd {df.narr_tone.std():.2f}, "
          f"missing {df.narr_tone.isna().sum()}")
    tot = {"calls": 0, "cost_cny": 0.0}
    for role in roles:
        u = usage_summary(f"agent_{role}_{args.anon}")
        hit = u["cache_hit"] / max(u["prompt"], 1)
        print(f"  {role:13s} calls {u['calls']:5d}  in {u['prompt'] / max(u['calls'], 1):6.0f}  "
              f"cache-hit {hit:5.1%}  out {u['completion'] / max(u['calls'], 1):5.0f}  "
              f"(reasoning {u['reasoning'] / max(u['calls'], 1):4.0f})  cost {u['cost_cny']:.3f} CNY")
        tot["calls"] += u["calls"]
        tot["cost_cny"] += u["cost_cny"]
    print(f"  total calls {tot['calls']}, cost {tot['cost_cny']:.3f} CNY "
          f"({tot['cost_cny'] / max(len(ev) * args.repeats, 1):.4f} per event)")


if __name__ == "__main__":
    main()
