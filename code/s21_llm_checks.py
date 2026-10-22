"""Step 21: LLM-side statistics of Tables 3-4 that no other step writes: repeat-run stability and probe hit rates.

Inputs: agents_repeat_B.csv (s10 --repeats 3 on 500 stage-B events; repeat 0 is the main run),
probes_pilot_anon_masked.csv (s08), cutoff_probe_stageB.csv (s09), agents_stageB_all.csv. No API calls.

Rows of results/checks/R3_llm_checks.csv (key, field, value, n):
  kappa         Fleiss' kappa of a categorical field over the three runs
  all_agree     share of events with the same value in all three runs
  corr          mean pairwise Pearson correlation of a continuous field across runs
  keep_{label}  share of events given `label` in at least one run that keep it in all three runs
  id_hit        identity probe on the amount-masked development texts: firm named
  eps_hit_pre   EPS recall, stage B, quarter end before PRE_QEND; eps_hit_post: post-cutoff stratum
"""

from itertools import combinations

import numpy as np
import pandas as pd

from common import INTERIM, ROOT

OUT = ROOT / "results" / "checks"
CATEGORICAL = ["master_dir", "master_noreb_dir", "single_dir", "master_winner", "master_label", "single_label",
               "narr_guidance"]
CONTINUOUS = ["narr_tone", "master_score", "single_score"]
PRE_QEND = "2025-06-01"  # quarters ending before 2025Q2, clear of the mid-2025 knowledge cutoff


def fleiss(wide):
    """Fleiss' kappa for an events x raters frame of category values."""
    counts = pd.get_dummies(wide.stack()).groupby(level=0).sum()
    n = wide.shape[1]
    p_i = ((counts ** 2).sum(axis=1) - n) / (n * (n - 1))
    p_j = counts.sum() / counts.values.sum()
    pe = (p_j ** 2).sum()
    return (p_i.mean() - pe) / (1 - pe)


def main():
    r = pd.read_csv(INTERIM / "agents_repeat_B.csv")
    rows = []
    for c in CATEGORICAL:
        w = r.pivot(index="accession", columns="repeat", values=c).dropna()
        rows += [("kappa", c, fleiss(w), len(w)), ("all_agree", c, (w.nunique(axis=1) == 1).mean(), len(w))]
    for c in CONTINUOUS:
        w = r.pivot(index="accession", columns="repeat", values=c).dropna()
        rows.append(("corr", c, np.mean([w[a].corr(w[b]) for a, b in combinations(w.columns, 2)]), len(w)))
    w = r.pivot(index="accession", columns="repeat", values="master_label")
    for lab in ("Hollow Beat", "Resilient Miss"):
        g = w[(w == lab).any(axis=1)]
        rows.append((f"keep_{lab}", "master_label", (g == lab).all(axis=1).mean(), len(g)))
    idp = pd.read_csv(INTERIM / "probes_pilot_anon_masked.csv").id_hit.dropna()
    rows.append(("id_hit", "id_hit", idp.mean(), len(idp)))
    c = pd.read_csv(INTERIM / "cutoff_probe_stageB.csv", parse_dates=["qend"]).merge(
        pd.read_csv(INTERIM / "agents_stageB_all.csv")[["accession", "stratum"]], on="accession")
    for key, g in (("eps_hit_pre", c[c.qend < PRE_QEND]), ("eps_hit_post", c[c.stratum == "post"])):
        rows.append((key, "eps_hit", g.eps_hit.mean(), g.eps_hit.notna().sum()))
    out = pd.DataFrame(rows, columns=["key", "field", "value", "n"])
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "R3_llm_checks.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
