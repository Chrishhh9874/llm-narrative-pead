"""Step 1: point-in-time S&P 500 universe 2015-2025 with ticker -> CIK mapping.

Inputs
  data/raw/sp500_fja05680/components.csv   (fja05680/sp500, MIT; daily snapshots of members)
  data/raw/wikipedia/sp500_rev*.html       (current table with CIK + historical changes table)
  SEC company_tickers.json                 (current registrants)
CIK candidates: manual override -> Wikipedia CIK column -> SEC current tickers -> ticker renames ->
EDGAR entity search on the Wikipedia security name. A candidate is accepted only if the CIK filed
>= 2 10-K/10-Q during the membership spell (name-search hits also need name similarity >= 0.6).

Output
  data/interim/universe.csv  one row per membership spell: ticker, start, end, cik, edgar_name, map_source, ...
"""

import difflib
import re
import urllib.parse

import pandas as pd

from common import INTERIM, RAW, SAMPLE_END, SAMPLE_START, SecClient, load_submissions

WIKI = RAW / "wikipedia" / "sp500_rev1329075976.html"  # revision of 2025-12-23
OVERRIDES = RAW / "manual" / "ticker_cik_overrides.csv"  # ticker,cik,valid_from,valid_to (trusted; blank cik = exclude)
RENAMES = RAW / "manual" / "ticker_renames.csv"  # old_ticker,new_ticker


def membership_spells():
    c = pd.read_csv(RAW / "sp500_fja05680" / "components.csv", parse_dates=["date"]).sort_values("date")
    # include the last snapshot before the sample so firms already in the index on 2015-01-01 count
    first = c.loc[c.date < SAMPLE_START, "date"].max()
    c = c[(c.date >= first) & (c.date <= SAMPLE_END)].reset_index(drop=True)
    rows = []
    for i, r in c.iterrows():
        nxt = c.date.iloc[i + 1] - pd.Timedelta(days=1) if i + 1 < len(c) else pd.Timestamp(SAMPLE_END)
        for t in r.tickers.split(","):
            rows.append((t, max(r.date, pd.Timestamp(SAMPLE_START)), nxt))
    df = pd.DataFrame(rows, columns=["ticker", "s", "e"]).sort_values(["ticker", "s"])
    spells = []
    for t, g in df.groupby("ticker"):
        cur_s, cur_e = None, None
        for s, e in zip(g.s, g.e):
            if cur_s is None:
                cur_s, cur_e = s, e
            elif s <= cur_e + pd.Timedelta(days=1):
                cur_e = max(cur_e, e)
            else:
                spells.append((t, cur_s, cur_e))
                cur_s, cur_e = s, e
        spells.append((t, cur_s, cur_e))
    return pd.DataFrame(spells, columns=["ticker", "start", "end"])


def wiki_tables():
    t = pd.read_html(WIKI)
    cur = t[0][["Symbol", "Security", "CIK"]].rename(columns={"Symbol": "ticker", "Security": "name", "CIK": "cik"})
    ch = t[1]
    ch.columns = ["date", "add_t", "add_n", "rem_t", "rem_n", "reason"]
    ch["date"] = pd.to_datetime(ch.date, errors="coerce")
    names = pd.concat([
        ch[["date", "add_t", "add_n"]].set_axis(["date", "ticker", "name"], axis=1),
        ch[["date", "rem_t", "rem_n"]].set_axis(["date", "ticker", "name"], axis=1),
    ]).dropna(subset=["ticker", "name"]).reset_index(drop=True)
    return cur, names


NAME_STOP = {"inc", "corp", "corporation", "co", "company", "plc", "ltd", "llc", "lp", "the", "holdings",
             "holding", "group", "de", "nv", "new", "international", "incorporated", "limited", "sa"}


def norm_name(x):
    x = re.sub(r"\(.*?\)|\[.*?\]|/[a-z]{2,3}/?", " ", str(x).lower())
    x = re.sub(r"[^a-z0-9 ]", " ", x.replace("&", " and "))
    return " ".join(w for w in x.split() if w not in NAME_STOP)


def name_sim(a, b):
    """1.0 if one name's tokens contain the other's (or are its initials), else a sequence ratio."""
    a, b = norm_name(a), norm_name(b)
    if not a or not b:
        return 0.0
    ta, tb = set(a.split()), set(b.split())
    if ta <= tb or tb <= ta:
        return 1.0
    initials = lambda s: "".join(w[0] for w in str(s).lower().split() if w.isalpha())
    if a.replace(" ", "") in (initials(b), initials(b.replace("machines", "x"))) or len(a) <= 4 and a == initials(b):
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def edgar_entity_search(sec, name, k=10):
    """Candidate (cik, entity) pairs from EDGAR's entity autocomplete index (current names only)."""
    q = re.sub(r"\(.*?\)|\[.*?\]", "", name).strip()
    out = []
    for query in dict.fromkeys([q, " ".join(q.split()[:2])]):
        js = sec.get_json("https://efts.sec.gov/LATEST/search-index?keysTyped=" + urllib.parse.quote(query))
        for h in (js or {}).get("hits", {}).get("hits", [])[:k]:
            out.append((int(h["_id"]), h["_source"]["entity"]))
    return list(dict.fromkeys(out))


def edgar_fts_ticker(sec, ticker, start, end):
    """CIKs of 10-K filers whose documents mention "(TICKER)"-style text during the spell."""
    url = ("https://efts.sec.gov/LATEST/search-index?q=" + urllib.parse.quote(f'"{ticker}"')
           + f"&forms=10-K&dateRange=custom&startdt={start:%Y-%m-%d}&enddt={end:%Y-%m-%d}")
    js = sec.get_json(url)
    hits = (js or {}).get("hits", {}).get("hits", [])
    return list(dict.fromkeys(int(c) for h in hits for c in h["_source"]["ciks"][:1]))


_sub_cache = {}


def periodic_filings_in_spell(sec, cik, start, end):
    """(# of 10-K/10-Q filed during the spell, EDGAR name, former names) for validation."""
    if cik not in _sub_cache:
        _sub_cache[cik] = load_submissions(sec, cik)
    info, df = _sub_cache[cik]
    if info is None:
        return 0, None, []
    m = df.form.isin(["10-K", "10-Q", "10-K405", "10-KT"]) & df.filingDate.between(start, end)
    former = [f["name"] for f in (info.get("formerNames") or [])]
    return int(m.sum()), info["name"], former


def apply_overrides(spells, ov):
    """Split spells at override validity bounds and attach the forced CIK (NaN = exclude)."""
    rows = []
    for _, r in spells.iterrows():
        o = ov[ov.ticker == r.ticker]
        if o.empty:
            rows.append({**r.to_dict(), "override": False, "override_cik": None})
            continue
        for _, x in o.iterrows():
            s = max(r.start, x.valid_from) if pd.notna(x.valid_from) else r.start
            e = min(r.end, x.valid_to) if pd.notna(x.valid_to) else r.end
            if s <= e:
                rows.append({**r.to_dict(), "start": s, "end": e, "override": True, "override_cik": x.cik})
        covered = o.valid_from.isna().any() and o.valid_to.isna().any() or len(o) > 1
        if not covered:  # single bounded override: rest of the spell goes through normal matching
            x = o.iloc[0]
            if pd.notna(x.valid_to) and x.valid_to < r.end:
                rows.append({**r.to_dict(), "start": max(r.start, x.valid_to + pd.Timedelta(days=1)),
                             "override": False, "override_cik": None})
            if pd.notna(x.valid_from) and x.valid_from > r.start:
                rows.append({**r.to_dict(), "end": min(r.end, x.valid_from - pd.Timedelta(days=1)),
                             "override": False, "override_cik": None})
    out = pd.DataFrame(rows)
    return out[out.start <= out.end].reset_index(drop=True)


def main():
    sec = SecClient()
    spells = membership_spells()

    sec_map = sec.get_json("https://www.sec.gov/files/company_tickers.json")
    sec_map = {v["ticker"].upper(): int(v["cik_str"]) for v in sec_map.values()}
    wiki_cur, wiki_names = wiki_tables()
    wiki_cik = {r.ticker.replace(".", "-"): int(r.cik) for _, r in wiki_cur.iterrows()}
    renames = pd.read_csv(RENAMES).set_index("old_ticker").new_ticker.to_dict()
    ov = pd.read_csv(OVERRIDES, parse_dates=["valid_from", "valid_to"])
    spells = apply_overrides(spells, ov)

    out = []
    for _, r in spells.iterrows():
        t = r.ticker.replace(".", "-")
        cand = wiki_names[wiki_names.ticker == r.ticker]
        wname = None
        if len(cand):
            d = pd.concat([(cand.date - r.end).abs(), (cand.date - r.start).abs()], axis=1).min(axis=1)
            wname = cand.loc[d.idxmin(), "name"]
        if wname is None and (wiki_cur.ticker == r.ticker).any():
            wname = wiki_cur.loc[wiki_cur.ticker == r.ticker, "name"].iloc[0]

        # ordered candidates: (source, cik, min name similarity vs. EDGAR current/former names)
        cands = []
        if r.override:
            if pd.isna(r.override_cik):
                out.append({**r.to_dict(), "wiki_name": wname, "cik": None, "edgar_name": None,
                            "map_source": "excluded", "n_periodic": 0, "name_sim": None})
                continue
            cands.append(("override", int(r.override_cik), 0.0))
        if t in wiki_cik:
            cands.append(("wikipedia", wiki_cik[t], 0.5))
        if t in sec_map:
            cands.append(("sec_current", sec_map[t], 0.5))
        if r.ticker in renames:
            nt = renames[r.ticker]
            for src, m in (("rename_wiki", wiki_cik), ("rename_sec", sec_map)):
                if nt in m:
                    cands.append((src, m[nt], 0.5))
        if wname:
            for cik, ent in edgar_entity_search(sec, wname):
                if name_sim(wname, ent) >= 0.5:
                    cands.append(("name_search", cik, 0.6))

        if wname and not any(c[0] in ("override", "wikipedia", "sec_current") for c in cands):
            cands += [("fts_ticker", cik, 0.6) for cik in edgar_fts_ticker(sec, r.ticker, r.start, r.end)]
        current_member = r.end >= pd.Timestamp("2025-06-30")

        best = None
        for src, cik, min_sim in dict.fromkeys(cands):
            n, ename, former = periodic_filings_in_spell(sec, cik, r.start, r.end)
            sim = max([name_sim(wname, x) for x in [ename, *former]] or [0]) if wname else None
            need_n = 1 if (r.end - r.start).days < 270 else 2
            trusted = src == "override" or (src == "wikipedia" and current_member)
            if trusted or (n >= need_n and (sim is None or sim >= min_sim)):
                score = (src not in ("name_search", "fts_ticker"), sim or 0, n)
                if best is None or score > best[0]:
                    best = (score, src, cik, ename, n, sim)
                if src not in ("name_search", "fts_ticker"):
                    break  # trust the first validated direct mapping
        rec = {**r.to_dict(), "wiki_name": wname}
        if best:
            _, src, cik, ename, n, sim = best
            rec.update(cik=cik, edgar_name=ename, map_source=src, n_periodic=n, name_sim=sim)
        else:
            rec.update(cik=None, edgar_name=None, map_source="unmapped", n_periodic=0, name_sim=None)
        out.append(rec)

    u = pd.DataFrame(out).drop(columns=["override", "override_cik"])
    u["cik"] = u.cik.astype("Int64")
    INTERIM.mkdir(parents=True, exist_ok=True)
    u.to_csv(INTERIM / "universe.csv", index=False)
    print(f"spells={len(u)} tickers={u.ticker.nunique()} ciks={u.cik.nunique()}")
    print(u.map_source.value_counts().to_string())
    print("unmapped:", u.loc[u.cik.isna(), ["ticker", "wiki_name"]].values.tolist())


if __name__ == "__main__":
    main()
