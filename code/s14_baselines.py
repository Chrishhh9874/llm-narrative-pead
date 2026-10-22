"""Step 14: text-tone baselines for the ablation table (LM and FinBERT, paper Sections 6 and 8).

Both baselines read exactly the text the agents read: the anonymized release trimmed by s10's build_doc
(MAX_WORDS, same paragraph priority), with the [Pn] markers removed. LM is also computed on the full
anonymized text (`lm_*_full`) as a check that the trimming does not drive the baseline.

  LM       Loughran-McDonald master dictionary 1993-2025 (data/raw/lm/). tone = (pos - neg) / (pos + neg),
           plus pos / neg / uncertainty shares of all words. Words are upper-cased alphabetic tokens.
  FinBERT  ProsusAI/finbert, sentence level (Huang, Wang & Yang 2023). Per event: mean(p_pos - p_neg) over
           sentences (`fb_tone`), and the count version (n_pos - n_neg) / (n_pos + n_neg) (`fb_tone_cnt`).

Events: every event with anonymized text of >= MIN_WORDS words (full sample + dev set), keyed by accession.

  --limit N   first N events only (timing check)

Output: data/interim/baselines.csv
"""

import argparse
import re
import time

import numpy as np
import pandas as pd

from common import INTERIM, RAW
from s10_agents import MIN_WORDS, build_doc

LM_FILE = RAW / "lm" / "Loughran-McDonald_MasterDictionary_1993-2025.csv"
FINBERT = "ProsusAI/finbert"
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z\"“(])|\n+")
MARK = re.compile(r"^\[P\d+\] ", re.M)


def load_lm():
    d = pd.read_csv(LM_FILE, keep_default_na=False)
    return {c: set(d.Word[d[c] > 0].str.upper()) for c in ("Positive", "Negative", "Uncertainty")}


def lm_scores(text, lm, suffix=""):
    w = [x.upper() for x in WORD.findall(text)]
    n = max(len(w), 1)
    pos = sum(x in lm["Positive"] for x in w)
    neg = sum(x in lm["Negative"] for x in w)
    unc = sum(x in lm["Uncertainty"] for x in w)
    return {f"lm_tone{suffix}": (pos - neg) / (pos + neg) if pos + neg else 0.0,
            f"lm_pos{suffix}": pos / n, f"lm_neg{suffix}": neg / n, f"lm_unc{suffix}": unc / n,
            f"n_words{suffix}": len(w)}


def sentences(text):
    out = []
    for s in SENT.split(text):
        s = s.strip()
        if len(WORD.findall(s)) >= 4:  # drop table fragments, datelines and bare headings
            out.append(s)
    return out


def finbert_scores(docs, batch=128):
    """docs: list of sentence lists -> list of dicts. Sentences are sorted by length for fast batching."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(FINBERT)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT).to(dev).eval()
    lab = {v.lower(): k for k, v in model.config.id2label.items()}
    flat = [(i, s) for i, ss in enumerate(docs) for s in ss]
    order = sorted(range(len(flat)), key=lambda k: len(flat[k][1]))
    probs = np.zeros((len(flat), 3), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            enc = tok([flat[k][1] for k in idx], padding=True, truncation=True, max_length=128,
                      return_tensors="pt").to(dev)
            probs[idx] = torch.softmax(model(**enc).logits, -1).float().cpu().numpy()
            if (b // batch) % 500 == 0:
                print(f"  finbert {b:,}/{len(flat):,} sentences, {(time.time() - t0) / 60:.1f} min", flush=True)
    doc_of = np.array([i for i, _ in flat])
    pp, pn = probs[:, lab["positive"]], probs[:, lab["negative"]]
    arg = probs.argmax(1)
    out = []
    for i in range(len(docs)):
        m = doc_of == i
        if not m.any():
            out.append({"fb_tone": np.nan, "fb_tone_cnt": np.nan, "fb_n_sent": 0})
            continue
        npos, nneg = (arg[m] == lab["positive"]).sum(), (arg[m] == lab["negative"]).sum()
        out.append({"fb_tone": float((pp[m] - pn[m]).mean()),
                    "fb_tone_cnt": (npos - nneg) / (npos + nneg) if npos + nneg else 0.0,
                    "fb_pos": float((arg[m] == lab["positive"]).mean()),
                    "fb_neg": float((arg[m] == lab["negative"]).mean()), "fb_n_sent": int(m.sum())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    anon = INTERIM / "anon"
    accs = sorted(p.stem for p in anon.glob("*.txt"))
    accs = [a for a in accs if len((anon / f"{a}.txt").read_text().split()) >= MIN_WORDS]
    if args.limit:
        accs = accs[:args.limit]
    lm = load_lm()
    print(f"{len(accs)} events; LM words: pos {len(lm['Positive'])}, neg {len(lm['Negative'])}")

    rows, docs = [], []
    for a in accs:
        doc = MARK.sub("", build_doc(a, anon)[0])
        rows.append({"accession": a, **lm_scores(doc, lm), **lm_scores((anon / f"{a}.txt").read_text(), lm, "_full")})
        docs.append(sentences(doc))
    t0 = time.time()
    fb = finbert_scores(docs)
    print(f"finbert: {sum(map(len, docs)):,} sentences in {(time.time() - t0) / 60:.1f} min")
    df = pd.concat([pd.DataFrame(rows), pd.DataFrame(fb)], axis=1)
    out = INTERIM / ("baselines.csv" if not args.limit else "baselines_test.csv")
    df.to_csv(out, index=False)
    print(f"-> {out}")
    print(df.drop(columns="accession").describe().T[["mean", "std", "min", "50%", "max"]].round(3))
    print("corr:\n", df[["lm_tone", "lm_tone_full", "lm_neg", "fb_tone", "fb_tone_cnt"]].corr(method="spearman").round(2))


if __name__ == "__main__":
    main()
