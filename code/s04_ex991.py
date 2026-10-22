"""Step 4: download the earnings press release (EX-99.1) of each event and split it into typed paragraphs.

Exhibit choice: from the filing's -index.htm document table, the first document typed EX-99.1
(else EX-99.01 / EX-99 / first EX-99.*). Other EX-99.* documents are recorded but not parsed.

Cleaning
  * Tables (zero-width / no-break spaces ignored), in this order:
      statement table  a statement title in its first rows -> "[STATEMENT]" marker (starts the appendix)
      guidance table   <= 40 rows right after Outlook/Guidance wording ("... guidance ... as listed below:"), or
                       with >= 30% range-valued cells ("$710M- $740M", "~$336M") -> one guidance paragraph per row
      text table       >= 60% of rows carry a sentence (>= 5 words) and no number-only cell, e.g.
                       one-row "Highlights" bullets, or a one-bullet table with a figure ("Coal down 43 percent")
                       -> one paragraph per row
      key-value table  <= 8 rows of "label | value(s)" with mostly $ / % values ("Organic revenue growth | 13.7%")
                       -> "label: value | value" per row
      numeric table    >= 40% of cells numeric -> dropped ("[TABLE]")
      other            layout table -> kept as text
  * Paragraph types:
      safe_harbor  forward-looking-statement disclaimers (PSLRA language) and the risk-factor list after them
      boilerplate  conference call / webcast, "About <company>" section, non-GAAP / term definitions,
                   legal notices (merger solicitation, trademarks)
      fragment     page numbers / headers (running headers repeated >= 3 times, "Slide 4", "NEWS",
                   "... and Subsidiaries", "... page 3"), exhibit labels, contact blocks and address
                   lines, footnotes, table row labels left between numbers, anything < 6 words unless
                   it carries a figure ("Comparable restaurant sales decreased 4.0%")
      guidance     sentence-level: guidance/outlook/forecast/target wording with a figure, or a
                   forward-looking verb + figure + future period; withdrawals ("withdraws guidance")
                   and backward references ("exceeded guidance") do not count. Under an
                   "Outlook"/"Guidance" heading, short (<= 40 words) paragraphs with a figure and list
                   items after a colon also count; the section ends at the next non-Outlook heading, a
                   backward-looking item ("Net sales decreased 3 percent") or the first long paragraph
                   that is neither guidance nor about guidance. Rows of guidance tables (their footnote
                   rows are fragments).
      narrative    everything else (before the financial statements)
      appendix     anything after the first financial statement (statement table, or a title paragraph
                   followed by a table within the next 4 paragraphs: a "Balance Sheet" section heading over
                   prose is not a statement), once >= 3 substantive narrative paragraphs were seen (guards
                   against a table of contents; >= 1 when the table sits right below the title), or the index of
                   attached schedules. A summary-table start ("Selected Financial Information") followed by >= 5
                   body paragraphs is undone (WFC). A run of >= 2 prose paragraphs reporting results inside
                   the appendix (segment discussion between tables) is narrative again.
  * has_guidance = any guidance paragraph.
  * image_based: the exhibit is page images with a hidden text layer (Workiva); the hidden text is split
    into paragraphs at runs of >= 4 spaces and leading page numbers are dropped. Flagged in the index.

Outputs
  data/raw/ex991/{accession}.htm       raw exhibit
  data/interim/ex991/{accession}.json  {"paragraphs": [{"type", "text"}], ...}
  data/interim/ex991_index.csv         one row per event: exhibit file, counts, has_guidance
"""

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from bs4 import BeautifulSoup

from common import INTERIM, RAW, SecClient

RAW_DIR = RAW / "ex991"
OUT_DIR = INTERIM / "ex991"

SAFE_HARBOR = re.compile(r"private securities litigation reform act|safe harbor|forward[- ]looking statements?"
                         r".{0,200}(risks?|uncertaint|actual results)|undue reliance|"
                         r"(undertakes?|assumes?) no (obligation|duty)|forward[- ]looking statements? speak only|"
                         r"(do|does) not (undertake|intend|assume).{0,40}(update|revise)", re.I | re.S)
BOILER = re.compile(r"conference call|webcast|dial[- ]in|replay|recording of (the|this|our) .{0,20}call|investor relations|media contact|"
                    r"^about [A-Z]|non-gaap (financial )?measures? (is|are|should|used|as defined)|"
                    r"reconciliation.{0,40}(non-gaap|gaap)|for more information|www\.|https?://|"
                    r"press release.{0,30}(attached|furnished)|additional information and where to find it|"
                    r"participants in the solicitation|no offer or solicitation|does not constitute an offer|"
                    r"proxy statement/prospectus|registration statement on form s-4|all rights reserved|©|"
                    r"(registered )?trademarks? (of|are the property)|used under licen[cs]e|regulation fd|member finra|"
                    r"past performance (is no|does not)|not (fdic|federally) insured|securities offered through", re.I)
# company description outside an "About" section ("People's United Financial, Inc., a diversified financial
# services company with $40 billion in total assets, provides ...")
ABOUT_TEXT = re.compile(r"^(?:(?:[A-Z][\w.&’'\-]*|of|and|&),?\s){1,7}(?:\([^)]{1,40}\),?\s)?(?:is\s(?:a|an|one of the|the)|a|an|one of the) "
                        r"[\w\-’', ]{0,80}\b(company|corporation|provider|leader|manufacturer|bank|holding company|firm|"
                        r"reit|trust|operator|developer|producer|distributor|retailer|insurer|subsidiary)\b")
CALL_INFO = re.compile(r"conference call|webcast|dial[- ]in|replay|passcode|access code|recording of (the|this|our) .{0,20}call", re.I)
# "The following forward-looking statements are based on current expectations ... Quanta now expects revenues to be
# between $12.55 billion and ...": a disclaimer sentence leading guidance is guidance unless PSLRA language is present
SAFE_STRONG = re.compile(r"private securities litigation|safe harbor|undue reliance|(undertakes?|assumes?) no (obligation|duty)|"
                         r"speak only as of", re.I)
HEADLINE = re.compile(r"\b(announces?|reports?|initiates?|raises?|reaffirms?|delivers?|posts?|provides|results|outlook|"
                      r"guidance)\b", re.I)
NOTE_LINE = re.compile(r"^notes?:\s", re.I)
# definitional text (non-GAAP measures, glossary terms) -> boilerplate unless it reports figures
DEFINITION = re.compile(r"\b(is|are) defined as|\bwe define\b|\b(is|are) calculated (as|by)|\bdefinitions? of\b|"
                        r"management believes .{0,120}(useful|meaningful|helpful)|should not be considered (as )?"
                        r"(a substitute|an alternative|in isolation|superior)|not (be )?comparable to similarly titled|"
                        r"provides? (useful|meaningful) (supplemental )?information to (investors|management)|"
                        r"\bwe (exclude|use) .{0,60}(non-gaap|adjusted) (financial )?measures?|"
                        r"refer(s)? to .{0,40}\b(means|represents)\b|"
                        r"without unreasonable efforts?|(is|are) (not able|unable) to (provide|present) .{0,40}reconcil|"
                        r"\btables? (below|above)\b.{0,60}\b(are|is) (reported|presented|prepared|in accordance)|"
                        r"amounts? .{0,40}(are )?(reported|presented|shown|expressed) in (millions|thousands|billions)", re.I)
# scope notes ("Unless otherwise noted, all growth rates refer to ..."), only in short paragraphs
SCOPE_NOTE = re.compile(r"unless otherwise (noted|stated|indicated)", re.I)
RISK_LEAD = re.compile(r"^(our (ability|inability)|the (risks?|effects?|impact)|possible|potential|changes?|fluctuations?|"
                       r"failure|other risks|risks?|uncertaint)", re.I)
RISK_ITEM = re.compile(r"\b(risks?|uncertaint\w*|our ability to|failure to|changes in|fluctuations|volatility|"
                       r"adverse|litigation|competition|regulatory|economic conditions)\b", re.I)
FIGURES = re.compile(r"\$\s?\d|\d(\.\d+)?\s?(%|percent\b)|\d\s(million|billion)\b", re.I)
# guidance (sentence level) needs (a) explicit guidance/outlook/forecast/target wording with a figure, or
# (b) a forward-looking verb + a figure + a future-period marker. "expected ranges", "second quarter
# revenue was ...", "exceeded our guidance", "withdrawing our guidance" do not count.
GUIDE_NOUN = re.compile(r"\b(guidance|outlook|forecast range|financial targets?|"
                        r"(cagr|growth|eps|earnings|margin|leverage|payout|return|revenue|sales)\s+targets?|"
                        r"capital (plan|budget|program|expenditure plan)|"
                        r"(rais|lower|narrow|reaffirm|reiterat|updat|maintain|confirm|increas|initiat|introduc|issu|"
                        r"revis)\w*\b(?:\W+\w+){0,6}?\W+(guidance|outlook|forecasts?|targets?|ranges?|estimates?|"
                        r"expectations?))\b", re.I)
FWD_VERB = re.compile(r"\b(expects?|expecting|anticipates?|anticipating|forecasts?|forecasting|estimates?|targets?|"
                      r"targeting|sees|intends?|plans? to|guides?|guiding|"
                      r"(?-i:projects?)|is projected to|are projected to|is expected to|are expected to|"
                      r"will be in the range|will (increase|decrease|grow|decline|reach|deliver|generate|be)\b.{0,60}"
                      r"\b(by|to|of) (approximately |about |roughly )?\$?\d)\b", re.I)
FIGURE = re.compile(r"\$\s?\d|\d+(\.\d+)?\s?(%|percent|basis points|bps)|\b(range|between \S+ and)\b|"
                    r"\b(low|mid|high)[- ](single|double)[- ]digits?\b|\b(low|mid|high)[- ]teens\b|\bdouble[- ]digits?\b|"
                    r"\b(approximately|about|roughly|essentially) flat\b|\d+(\.\d+)? (million|billion)\b", re.I)
FUTURE = re.compile(r"\b(full[- ]year|fiscal( year)? ?(20)?\d{2}|for the (fiscal )?year|year ending|quarter ending|"
                    r"next (quarter|year|fiscal|twelve months|12 months)|coming (quarter|year)s?|remainder of|"
                    r"second half|first half|(first|second|third|fourth)( fiscal)? quarter( of)? (fiscal )?(20)?\d{2}|long[- ]term|"
                    r"through (fiscal )?(20)?\d{2}|by (the end of )?(fiscal )?(20)?\d{2}|(in|for) (fiscal )?20\d{2}|"
                    r"(three|five|3|5)[- ]year|multi-year|cagr|annually|medium[- ]term|per annum|"
                    r"next (\w+ )?(months|years|quarters))\b", re.I)
WITHDRAW = re.compile(r"\b(withdr[ae]w\w*|suspend\w*|no longer (applicable|valid|in effect|be relied)|"
                      r"(not|no longer) (be )?(providing|provide|issuing|issue|giving)\b.{0,40}(guidance|outlook|forecast))",
                      re.I)
BACKWARD = re.compile(r"\b(exceed(s|ed|ing)?|beat|ahead of|above|below|within|in line with|top end of|high end of|low end of)\b"
                      r".{0,30}\b(guidance|outlook|expectations?|forecast)\b|\bas (anticipated|expected|previously "
                      r"(announced|disclosed))\b|\b(repurchased|returned|paid|declared|completed)\b", re.I)
# a results item, not a forecast ("Net sales decreased 3 percent to $1,828 million")
PAST = re.compile(r"\b(increased|decreased|was|were|grew|declined|rose|fell|improved|expanded|contracted|returned|"
                  r"declared|delivered|achieved)\b", re.I)
DEAL_CLOSE = re.compile(r"\b(expected|anticipated|expects?|anticipates?) (the transaction |the acquisition |it )?to "
                        r"(close|be completed|complete)\b", re.I)
RESULT_VERB = re.compile(r"\b(totaled|totalled|was|were|increased|decreased|grew|declined|rose|fell|improved)\b", re.I)
GUIDE_UPDATE = re.compile(r"\b(rais|lower|narrow|reaffirm|reiterat|updat|maintain|confirm|revis|introduc|initiat)\w*\b"
                          r"(?:\W+\w+){0,6}?\W+(guidance|outlook|forecasts?|targets?|ranges?)", re.I)
GUIDE_WORD = re.compile(r"\b(guidance|outlook|forecasts?|expects?)\b", re.I)
SENT_SPLIT = re.compile(r"(?<=[.;!?])\s+(?=[A-Z“\"(•])")


def is_guidance(p):
    """Sentence level, except that the future-period marker may sit anywhere in the paragraph
    ("we are updating our full year guidance. We anticipate revenue of approximately $7.1 billion")."""
    future = FUTURE.search(p)
    for s in SENT_SPLIT.split(p):
        if WITHDRAW.search(s) or DEAL_CLOSE.search(s) and not GUIDE_NOUN.search(s):
            continue
        fwd = FWD_VERB.search(s) and FIGURE.search(s) and future
        if (BACKWARD.search(s) or RESULT_VERB.search(s) and not GUIDE_UPDATE.search(s)) and not fwd:
            continue  # "capital program in 2019 totaled $3.2 billion", "Gross margin was 49.2%"
        intro = s.rstrip().endswith(":") and (GUIDE_NOUN.search(s) or (FWD_VERB.search(s) and future))
        if fwd or intro or (GUIDE_NOUN.search(s) and FIGURE.search(s)):
            return True
    return False


GUIDE_HEAD = re.compile(r"^(?!(provides|raises|reaffirms|updates|increases|lowers|narrows|issues|maintains|announces|"
                        r"initiates|reiterates|introduces|confirms)\b)([\w\-’']+\s){0,4}"
                        r"(outlook|guidance|forecast|financial targets|expectations)\b.{0,40}$", re.I)
ANY_HEAD = re.compile(r"^[A-Z][A-Za-z0-9&,'\-\s]{2,60}$")
ABOUT_HEAD = re.compile(r"^about\b", re.I)
NONGAAP_HEAD = re.compile(r"non-gaap|use of (non-gaap|adjusted)|adjusted (financial )?measures|^definitions?\b|glossary|"
                          r"^reconciliations?\b|forward[- ]looking non-gaap", re.I)
CONTACT_HEAD = re.compile(r"^((investor|media|press|analyst|financial)s?\s*(relations\s*)?(contacts?|inquiries)?|"
                          r"contacts?|corporate communications|for further information)\s*:?$", re.I)
FRAGMENT_PAT = re.compile(r"^(page \d+( of \d+)?|-?\s?\d{1,3}\s?-?|exhibit( 99(\.\d+)?)?|document|for immediate release|"
                          r"(news|press|earnings) release|\(?-?more-?\)?|(#\s?){3}|ex-99.*|[\W\d_\s]+|slide \d+|news|"
                          r"table [a-z0-9]{1,2}|.{0,50}\band subsidiaries|.{0,40}\bpage (\d+|[a-z]+)|"
                          r".{0,40}\b(news|press|earnings) release|[A-Z]{2,5}-[A-Z]|\d{1,3}\s+(www\.)?[\w\-]+\.(com|net|org))$", re.I)
# "(3) Excludes", "* ", "18 Book value", "2) Includes", "____ (1) Adjusted" (a page that begins with its page
# number is not a footnote: only short paragraphs qualify, see classify)
FOOTNOTE = re.compile(r"^(_{2,}\s*|\(\w{1,4}\)|\*+|\d{1,2}\)\s?|\d{1,2}\s?(?=[A-Z])|[a-z]\.\s)")
CONTACT_LINE = re.compile(r"\bcontacts?\s*:|@\w+\.\w|\(\d{3}\)\s?\d{3}|\d{3}[-.]\d{3}[-.]\d{4}", re.I)
ADDRESS_LINE = re.compile(r"\b[A-Z][a-z]+\.?,\s(?:[A-Z]{2}|[A-Z][a-z]+)\s\d{5}\b|\((?:NYSE|NASDAQ|Nasdaq)\s*:|"
                          r"\b(?:NYSE|NASDAQ|Nasdaq)\s*:\s*[A-Z]{1,5}\b")
BULLET_LEAD = re.compile(r"^[•▪●◦·\-–—*]\s*")
NUMERIC_TOKEN = re.compile(r"^[\s$()\d.,%\-–—x*]+$")
PAGE_MARK = re.compile(r"\bPage \d+ of \d+\b|^-\s?\d{1,3}\s?-\s+", re.I)


def exhibit_url(sec, cik, acc):
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/"
    b = sec.get(base + f"{acc}-index.htm")
    if b is None:
        return None, []
    soup = BeautifulSoup(b, "lxml")
    docs = []
    for tr in soup.select("table.tableFile tr"):
        td = [x.get_text(" ", strip=True) for x in tr.find_all("td")]
        a = tr.find("a")
        if len(td) >= 4 and a:
            docs.append((td[3].upper(), a["href"].split("/")[-1]))
    ex = [(t, d) for t, d in docs if t.startswith("EX-99")]
    for pref in ("EX-99.1", "EX-99.01", "EX-99"):
        hit = [d for t, d in ex if t == pref]
        if hit:
            return base + hit[0], [t for t, _ in ex]
    return (base + ex[0][1] if ex else None), [t for t, _ in ex]


ZW = re.compile(r"[​‌‍⁠﻿\xa0]")
SYMBOL_CELLS = {"$", "%", ")", "(", "•", "▪", "●", "-", "–", "—", "*", "·"}
GUIDE_SENTINEL = "⟦G⟧ "  # marks paragraphs made from guidance-table rows
TITLED = "⟦T⟧ "  # ... of a guidance table whose own header rows say outlook / guidance (kept inside the appendix too)


def table_rows(tbl):
    rows = []
    for tr in tbl.find_all("tr"):
        cells = [re.sub(r"\s+", " ", ZW.sub(" ", c.get_text(" ", strip=True))).strip() for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c and c not in SYMBOL_CELLS]
        if cells:
            rows.append(cells)
    return rows


# a forecast value: "$710M- $740M", "66% - 68%", "~$336M", "4%-6%" (a figure with $ / % / unit, not a phone number)
RANGE_CELL = re.compile(r"^[~≈]\s?\$?\d|^\$?\d[\d.,]*\s?(%|[MBK]\b|million|billion)?\s?(-|–|to)\s?\$?\d[\d.,]*\s?(%|[MBK]\b|million|billion)", re.I)


def table_kind(tbl, rows):
    """statement / guidance / text / keyvalue / numeric / layout (see module docstring)."""
    if not rows:
        return "numeric"
    head = " ".join(" ".join(r) for r in rows[:4])
    before = " ".join(s.strip() for s in reversed(tbl.find_all_previous(string=True, limit=12)) if s.strip())[-250:]
    n_title = sum(any(is_statement_title(c) for c in r) for r in rows)
    page_end = sum(bool(re.fullmatch(r"\d{1,3}(\s?-\s?\d{1,3})?", r[-1])) for r in rows) / len(rows)
    if (n_title >= 2 and page_end >= 0.5) or re.search(r"table of contents(\W|\[TABLE\])*$", before, re.I):
        return "numeric"  # a table of contents lists statement titles; it must not end the release
    if any(is_statement_title(c) for r in rows[:4] for c in r) or \
            re.search(r"\btotal assets\b.*\btotal liabilities\b|net cash (provided|used) (by|in) operating.*"
                      r"net cash (provided|used) (by|in) investing", " ".join(" ".join(r) for r in rows), re.I | re.S):
        return "statement"
    cells = [c for r in rows for c in r]
    num = sum(bool(re.search(r"[\d$%]", c)) and len(c) < 40 for c in cells) / len(cells)
    vals = [c for r in rows for c in r[1:] if re.search(r"\d", c)]
    ranged = sum(bool(RANGE_CELL.search(c)) for c in vals) / len(vals) if vals else 0
    if len(rows) <= 40 and (re.search(r"\b(outlook|guidance|forecast|financial targets)\b", head, re.I) or
                            re.search(r"\b(outlook|guidance|forecasts?|targets?)\b[^.]{0,80}[.:]?\s*$", before, re.I) or
                            re.search(r"\b(outlook|guidance|forecasts?)\b[^.]{0,200}\b(below|follows)\s*:\s*$", before, re.I) or
                            re.search(r"\b(expects?|anticipates?|projects?)\b[^.]{0,40}\bthe following\b[^.]{0,100}:\s*$", before, re.I) or
                            ranged >= 0.3) and num >= 0.2:
        return "guidance"
    if len(rows) <= 2 and all(len(r) <= 2 for r in rows) and any(len(re.findall(r"[A-Za-z]{2,}", c)) >= 3 and FIGURES.search(c) for r in rows for c in r):
        return "text"  # one-bullet table: "Coal down 43 percent", "AUM of $1.8 trillion"
    if 2 <= len(rows) <= 8 and all(len(r) <= 3 for r in rows) and \
            sum(len(re.findall(r"[A-Za-z]{2,}", r[0])) >= 2 for r in rows) >= len(rows) - 1 and \
            vals and sum(bool(re.search(r"[$%]", c)) for c in vals) >= 0.5 * len(vals):
        return "keyvalue"  # small metrics table: "Organic revenue growth | 13.7%"
    text_rows = sum(any(len(c.split()) >= 5 for c in r) and not any(NUMERIC_TOKEN.match(c) for c in r) for r in rows)
    if text_rows / len(rows) >= 0.6:
        return "text"
    return "numeric" if num >= 0.4 else "layout"


BLOCK_TAGS = ["p", "div", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "table",
              "center", "blockquote", "pre"]
STATEMENT_HEAD = re.compile(
    r"^(\S+\s){0,6}?((unaudited|condensed|consolidated|preliminary|summary|selected|interim)\s)*"
    r"(statements? of (consolidated )?(operations|income|earnings|financial (position|condition)|cash flows?|"
    r"comprehensive (income|loss)|(stockholders|shareholders)['’]? equity|condition)|income statements?|balance sheets?|"
    r"comprehensive income statements?|cash flow statements?|(consolidated|condensed) financial summary)\b|"
    r"^(supplemental (financial )?(information|schedules?|data|tables?)|financial (supplement|tables)|"
    r"selected( [\w,]+){0,4} (information|data)|appendix( [a-z])?|\(?(financial )?tables (follow|to follow)\)?)\b", re.I)


HIDDEN = re.compile(r"color:\s*(white|#fff\b|#ffffff)", re.I)


def image_layer(soup):
    """Paragraphs of a page-image exhibit (>= 2 images, >= 80% of the text in a hidden white-font layer), else None."""
    total = len(soup.get_text())
    hidden = [e for e in soup.find_all(style=HIDDEN) if not e.find_parent(style=HIDDEN)]
    if len(soup.find_all("img")) < 2 or not total or sum(len(e.get_text()) for e in hidden) < 0.8 * total:
        return None
    paras = []
    for e in hidden:
        parts = [re.sub(r"\s+", " ", ZW.sub(" ", x)).strip() for x in re.split(r"\s{4,}", e.get_text())]
        parts = [x for x in parts if x]
        if parts:
            parts[0] = re.sub(r"^\d{1,3}\s+(?=\S)", "", parts[0])  # page number
        paras += [x for x in parts if not re.fullmatch(r"\d{1,3}", x)]
    return paras


def row_text(r):
    """"label: v1 | v2" for a table row."""
    return r[0].rstrip(": ") + ": " + " | ".join(r[1:]) if len(r) > 1 else r[0]


def html_to_paragraphs(html):
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    img = image_layer(soup)
    if img is not None:
        return merge_blocks(img), 0, True
    n_dropped = 0
    for tbl in soup.find_all("table"):
        if tbl.find_parent("table") is not None:
            continue
        rows = table_rows(tbl)
        kind = table_kind(tbl, rows)
        if kind == "statement":
            tbl.replace_with(soup.new_string("\n\n[STATEMENT]\n\n"))
            n_dropped += 1
        elif kind == "numeric":
            tbl.replace_with(soup.new_string("\n\n[TABLE]\n\n"))
            n_dropped += 1
        elif kind == "guidance":
            titled = TITLED if re.search(r"\b(outlook|guidance|projected|forecast)\b",
                                         " ".join(" ".join(r) for r in rows[:3]), re.I) else ""
            lines = [GUIDE_SENTINEL + titled + row_text(r) for r in rows]
            tbl.replace_with(soup.new_string("\n\n" + "\n\n".join(lines) + "\n\n"))
        elif kind == "text":
            tbl.replace_with(soup.new_string("\n\n" + "\n\n".join(" ".join(r) for r in rows) + "\n\n"))
        elif kind == "keyvalue":
            lines = [row_text(r) for r in rows]
            tbl.replace_with(soup.new_string("\n\n" + "\n\n".join(lines) + "\n\n"))
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for el in soup.find_all(BLOCK_TAGS):  # block boundaries -> blank lines; inline tags concatenate
        el.insert_before("\n\n")
        el.insert_after("\n\n")
    text = soup.get_text("")
    text = re.sub(r"[ \t\xa0\u200b\u200c\u200d\u2060\ufeff]+", " ", text)
    blocks = [re.sub(r"\s*\n\s*", " ", b).strip() for b in re.split(r"\n\s*\n", text)]
    return merge_blocks([b for b in blocks if b]), n_dropped, False


BULLET = re.compile(r"^[\u2022\u25e6\u25aa\u25cf\u2013\u2014\u00b7o*\-]$")


def merge_blocks(blocks):
    """Attach lone bullet glyphs to the next block; re-join lines split mid-sentence by layout markup."""
    out, pending_bullet = [], False
    for b in blocks:
        open_sentence = out and len(out[-1].split()) >= 8 and not re.search(r"[.:;!?\u201d\"]$", out[-1]) \
            and out[-1] not in ("[TABLE]", "[STATEMENT]")
        if open_sentence and re.fullmatch(r"-?\s?\d{1,3}\s?-?|\d{1,3}\s?\|", b):
            continue  # page number between the two halves of a sentence
        if open_sentence:
            b = re.sub(r"^\d{1,3}\s?\|?\s+(?=[a-z])", "", b)  # "4 these and other factors", "1 | commercial"
        if BULLET.match(b):
            pending_bullet = True
            continue
        if pending_bullet:
            b, pending_bullet = "\u2022 " + b, False
        elif (out and out[-1] not in ("[TABLE]", "[STATEMENT]") and b not in ("[TABLE]", "[STATEMENT]")
              and re.match(r"^[a-z(]", b) and not re.search(r"[.:;!?\u201d\"]$", out[-1])
              and not out[-1].startswith(GUIDE_SENTINEL) and not b.startswith(GUIDE_SENTINEL)):
            out[-1] = out[-1] + " " + b
            continue
        out.append(b)
    return out


TITLE_SUFFIX = re.compile(r"^(\s*([\-\u2013\u2014,]\s*)?(\([^)]*\)|unaudited|continued|preliminary|"
                          r"(for the )?((three|six|nine|twelve)( and (three|six|nine|twelve))? months|(quarters?|years?)( and (years?|quarters?))?) ended.*|(in|dollars in) (millions|thousands|billions).*|"
                          r"\w+ \d{1,2},? \d{4}|q[1-4] ?\d{2,4}))*\s*$", re.I)


def is_statement_title(p, words=None):
    """A financial-statement title: the statement noun phrase, optionally preceded by a few qualifier words
    (company name, "U.S. GAAP Condensed Consolidated") and followed only by suffixes such as "(Unaudited)",
    "(In millions, except per share amounts)", "(continued)". "Other Income Statement Items" and
    "Balance Sheet and Cash Flow" are section headings, not statements."""
    words = len(p.split()) if words is None else words
    if words > 40 or p.rstrip().endswith((".", ":")):
        return False
    m = STATEMENT_HEAD.search(p)
    return bool(m) and bool(TITLE_SUFFIX.match(p[m.end():]))


def statement_follows(paragraphs, i, k=4):
    """A statement title paragraph is a statement only if a table follows within the next 4 paragraphs
    (CFG's "Balance Sheet" section heading is followed by prose)."""
    nxt = paragraphs[i + 1:i + 1 + k]
    return any(x in ("[TABLE]", "[STATEMENT]") or x.startswith(GUIDE_SENTINEL) for x in nxt)


# index of the attached schedules ("Quarter 4 2017 Earnings Release Schedules Index") also ends the release
SCHEDULE_INDEX = re.compile(r"^(\S+\s){0,6}?(schedules? index|index (to|of) (the )?(financial )?schedules|earnings release schedules( and supplementa\w+ information)?)$", re.I)


WEAK_START = re.compile(r"\b(selected|summary|highlights|supplemental)\b", re.I)


def is_body_prose(p):
    """A paragraph of the release body (results or a quote), used to undo a statement start that came too early."""
    return (len(p.split()) >= 25 and re.search(r"[.”\"]$", p.rstrip()) and (p[0].isupper() or p[0] in "“\"")
            and (PAST.search(p) or p[0] in "“\"") and not FOOTNOTE.match(p) and not DEFINITION.search(p)
            and not BOILER.search(p) and not SAFE_HARBOR.search(p) and not RISK_LEAD.match(p))


def is_result_prose(p):
    """Segment discussion between appendix tables ("Health & Civil revenues were $1.33 billion ...")."""
    words = len(p.split())
    return (words >= 20 and p.rstrip().endswith(".") and len(FIGURES.findall(p)) >= 2 and PAST.search(p)
            and not FOOTNOTE.match(p) and not DEFINITION.search(p) and not BOILER.search(p)
            and not SAFE_HARBOR.search(p))


def classify(paragraphs):
    """Type each paragraph; text after the first financial statement (statement table, or a title followed by a
    table) is `appendix`, once >= 3 substantive narrative paragraphs were seen (a table of contents must not end
    the release). A summary-table start ("Selected Financial Information", "... Highlights") followed by >= 5
    paragraphs of release body before the next statement candidate (WFC's page-3 summary table) is undone and the
    next candidate is tried."""
    skip = 0
    for _ in range(10):
        typed, start = _classify(paragraphs, skip)
        if start is None or not WEAK_START.search(paragraphs[start]):
            break
        nxt = next((j for j in range(start + 1, len(paragraphs)) if paragraphs[j] == "[STATEMENT]" or
                    is_statement_title(paragraphs[j]) and statement_follows(paragraphs, j)), len(paragraphs))
        if sum(bool(is_body_prose(PAGE_MARK.sub("", x).strip())) for x in paragraphs[start + 1:nxt]) < 5:
            break
        skip = start + 1
    return typed


def _classify(paragraphs, skip=0):
    """One pass of classify(); statement candidates before index `skip` are ignored. Returns (typed, start index)."""
    counts = Counter(PAGE_MARK.sub("", x).strip() for x in paragraphs)
    seen = set()
    typed, appendix, n_body, start = [], False, 0, None
    section = None  # None / "guidance" / "about" / "nongaap" / "safe" / "contact"
    in_list = False
    for i, raw in enumerate(paragraphs):
        p = PAGE_MARK.sub("", raw).strip()
        words = len(p.split())
        if raw in ("[TABLE]", "[STATEMENT]"):
            if raw == "[STATEMENT]" and n_body >= 3 and i >= skip and not appendix:
                appendix, start = True, i
            typed.append({"type": "table", "text": raw})
            continue
        if appendix and not p.startswith(GUIDE_SENTINEL + TITLED):
            typed.append({"type": "appendix", "text": p.removeprefix(GUIDE_SENTINEL)})
            continue
        if not p:
            continue
        if p.startswith(GUIDE_SENTINEL):  # guidance tables count even inside the schedules (UNH "2025 Outlook")
            t = p[len(GUIDE_SENTINEL):].removeprefix(TITLED)
            prose = len(t.split()) > 25 and t.rstrip().endswith(".") and re.search(r"[a-z]{3,} [a-z]{3,} [a-z]{3,}", t)
            lettered = re.match(r"^[a-z]\s(?=[A-Z0-9])", t)  # "a 2022 outlook projections are ...", "b Attributable"
            foot = (FOOTNOTE.match(t) or prose) and not is_guidance(t) or lettered and len(t.split()) > 12
            head = not re.search(r"\d", t) and len(t.split()) <= 8  # column header row ("First-Quarter")
            typed.append({"type": "fragment" if foot else "heading" if head else "guidance", "text": t})
            continue
        # a statement title over a table (right below it: >= 1 body paragraph suffices, PGR's monthly release),
        # or the index of the attached schedules
        if i >= skip and (is_statement_title(p, words) and (n_body >= 3 and statement_follows(paragraphs, i) or
                                                             n_body >= 1 and statement_follows(paragraphs, i, 2))
                          or n_body >= 3 and words <= 15 and SCHEDULE_INDEX.match(p)):
            appendix, start = True, i
            typed.append({"type": "appendix", "text": p})
            continue
        core = BULLET_LEAD.sub("", p)
        repeated = words <= 15 and (p in seen or counts[p] >= 3)  # running headers
        seen.add(p)
        if repeated or FRAGMENT_PAT.match(core):
            typed.append({"type": "fragment", "text": p})
            continue
        if words <= 8 and CONTACT_HEAD.match(core):
            section, in_list = "contact", False
            typed.append({"type": "fragment", "text": p})
            continue
        if section == "contact" and HEADLINE.search(p) and words >= 3 and not BULLET_LEAD.match(p):
            section = None  # the release headline right after a contact block ("Initiates Q1 and FY 2022 Outlook")
        if section == "contact":
            if words <= 10 and not FIGURES.search(p) or re.search(r"@|\(\d{3}\)|\d{3}[-.]\d{3}[-.]\d{4}", p):
                typed.append({"type": "fragment", "text": p})
                continue
            section = None
        if words <= 30 and CONTACT_LINE.search(p) and not is_guidance(p) or \
                words <= 20 and ADDRESS_LINE.search(p) and not p.rstrip().endswith(".") and not ABOUT_HEAD.match(core):
            typed.append({"type": "fragment", "text": p})  # "PR Contact: ... (949) 524-4035", mastheads
            continue
        # a colon-ended Outlook header ("•2025 Outlook:") opens a guidance list
        if words <= 8 and GUIDE_HEAD.match(core.rstrip(": ")):
            section, in_list = "guidance", core.rstrip().endswith(":")
            typed.append({"type": "heading", "text": p})
            continue
        backward = PAST.search(p) and not FWD_VERB.search(p) and not GUIDE_WORD.search(p) and not is_guidance(p)
        if section == "guidance" and backward:
            section, in_list = None, False  # results bullets after the Outlook block
        if in_list and 3 <= words <= 40 and not SAFE_HARBOR.search(p) and not BOILER.search(p):
            typed.append({"type": "guidance", "text": p})
            continue
        in_list = False
        if section == "guidance" and words <= 40 and (FIGURE.search(p) or p.rstrip().endswith(":") or
                                                      BULLET_LEAD.match(p) and re.search(r"\d", p)) and \
                not (FOOTNOTE.match(p) and not is_guidance(p)):
            typed.append({"type": "guidance", "text": p})  # "◦Q3 FY20: -25% to -35%", "•At least 500 net new stores"
            in_list = p.rstrip().endswith(":")
            continue
        if words <= 8 and ANY_HEAD.match(core) or (words <= 8 and section == "guidance" and not BULLET_LEAD.match(p)
                                                   and not p.rstrip().endswith((".", ":"))):
            section = ("about" if ABOUT_HEAD.match(core) else "nongaap" if NONGAAP_HEAD.search(core)
                       else "safe" if SAFE_HARBOR.search(core) or re.search(r"forward[- ]looking|cautionary", core, re.I)
                       else None)
            typed.append({"type": "boilerplate" if section == "about" else "heading", "text": p})
            continue
        n_fig = len(FIGURES.findall(p))
        if SAFE_HARBOR.search(p) and not (n_fig >= 2 and not SAFE_STRONG.search(p) and
                                          (is_guidance(p) or any(is_guidance(x) and not SAFE_HARBOR.search(x)
                                                                 for x in SENT_SPLIT.split(p)))):
            t, section = "safe_harbor", "safe"
        elif words <= 30 and NOTE_LINE.match(core) and not is_guidance(p):
            t = "boilerplate"  # "Note: Amounts may not sum due to rounding."
        elif section == "safe" and (BULLET_LEAD.match(p) or FOOTNOTE.match(p) or p[0].islower() or
                                    re.match(r"^\d+\.\s", p)) and words < 150:
            t = "safe_harbor"  # risk-factor list after the disclaimer
        elif section == "safe" and words < 80 and RISK_ITEM.search(p) and (re.search(r"[;,]( and| or)?$", p) or
                                                                             RISK_LEAD.match(p)):
            t = "safe_harbor"  # risk items without a bullet ("Our ability to implement our business strategy ...;")
        elif section == "about" or (ABOUT_TEXT.match(p) and words < 150 and
                                    not re.search(r"\b(today|reported|announced|reports|announces)\b", p)):
            t = "boilerplate"
        elif BOILER.search(p) and words < 150 and (n_fig < 3 or CALL_INFO.search(p)):
            t = "boilerplate"
        elif words < 250 and n_fig < 3 and (DEFINITION.search(p) or section == "nongaap" or
                                            words < 60 and SCOPE_NOTE.search(p)) and not is_guidance(p):
            t = "boilerplate"
        elif words < 6 and not (words >= 3 and FIGURES.search(p)):
            t = "fragment"
        elif FOOTNOTE.match(p) and words <= 150 and not is_guidance(p):
            t = "fragment"  # table footnotes "(3) Excludes ...", "18 Book value per share ..."
        elif is_guidance(p):
            t = "guidance"
            in_list = p.rstrip().endswith(":")
        else:
            t = "narrative"
            if words > 40 and section == "guidance" and not GUIDE_WORD.search(p):
                section = None  # a long paragraph that is not about guidance ends an Outlook section
        if t in ("narrative", "guidance") and words >= 15:
            n_body += 1
        typed.append({"type": t, "text": p})
    # short heading / narrative leftovers sitting between numbers are table row labels
    for i in range(1, len(typed) - 1):
        cur = typed[i]
        if cur["type"] in ("heading", "narrative") and len(cur["text"].split()) <= 8 and not FIGURES.search(cur["text"]):
            nb = [typed[j] for j in (i - 1, i + 1)]
            if all(x["type"] in ("fragment", "table") and NUMERIC_TOKEN.match(x["text"].replace("[TABLE]", "0")) for x in nb):
                cur["type"] = "fragment"
    # >= 2 results paragraphs between two appendix tables are segment discussion -> narrative
    block = []
    for x in typed + [{"type": "table", "text": ""}]:
        if x["type"] == "table":
            prose = [y for y in block if is_result_prose(y["text"])]
            if len(prose) >= 2:
                for y in prose:
                    y["type"] = "narrative"
            block = []
        elif x["type"] == "appendix":
            block.append(x)
    return typed, start


def process_event(sec, r, force=False):
    out_json = OUT_DIR / f"{r.accession}.json"
    if out_json.exists() and not force:
        d = json.loads(out_json.read_text())
        return {k: v for k, v in d.items() if k != "paragraphs"}
    url, ex_types = exhibit_url(sec, r.cik, r.accession)
    rec = {"accession": r.accession, "cik": int(r.cik), "exhibit_url": url, "ex_types": ";".join(ex_types)}
    if url is None:
        rec["status"] = "no_ex99"
        return rec
    html = sec.get(url)  # disk-cached, so --force re-parses without new requests
    if html is None:
        rec["status"] = "fetch_failed"
        return rec
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"{r.accession}{'.htm' if url.lower().endswith(('htm', 'html')) else '.txt'}").write_bytes(html)
    paras, n_tables, image_based = html_to_paragraphs(html)
    typed = classify(paras)
    cnt = pd.Series([p["type"] for p in typed]).value_counts().to_dict()
    words = lambda t: sum(len(p["text"].split()) for p in typed if p["type"] == t)
    rec.update(status="ok", n_tables_dropped=n_tables, image_based=image_based,
               **{f"n_{k}": cnt.get(k, 0) for k in ("narrative", "guidance", "safe_harbor", "boilerplate")},
               words_narrative=words("narrative"), words_guidance=words("guidance"),
               has_guidance=cnt.get("guidance", 0) > 0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({**rec, "paragraphs": typed}, ensure_ascii=False, indent=1))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=str(INTERIM / "pilot_events.csv"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="re-parse events whose JSON already exists")
    ap.add_argument("--index", default="ex991_index.csv", help="index file name under data/interim/")
    ap.add_argument("--rate", type=float, default=8.0, help="SEC requests per second (split it across parallel shards)")
    args = ap.parse_args()

    sec = SecClient(max_per_sec=args.rate)
    ev = pd.read_csv(args.events)
    with ThreadPoolExecutor(args.workers) as pool:
        recs = list(pool.map(lambda r: process_event(sec, r, args.force), ev.itertuples()))
    idx = pd.DataFrame(recs)
    idx.to_csv(INTERIM / args.index, index=False)
    print(idx.status.value_counts().to_string())
    ok = idx[idx.status == "ok"]
    print(f"has_guidance={ok.has_guidance.mean():.1%}  median narrative words={ok.words_narrative.median():.0f}  "
          f"median guidance words={ok.words_guidance.median():.0f}  median tables dropped={ok.n_tables_dropped.median():.0f}")


if __name__ == "__main__":
    main()
