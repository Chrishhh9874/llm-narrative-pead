"""Step 6: anonymize the LLM-facing text of each press release and measure residual identifiers.

LLM-facing text = paragraphs typed narrative / guidance / heading (s04); safe harbor, boilerplate,
fragments, tables and the post-statement appendix are excluded.

Replacements (consistent within a document: every string masked once is masked everywhere)
  firm aliases (EDGAR current + former names, Wikipedia name, short forms, "E*TRADE"/"Domino's"
    spellings; acronyms the release defines right after a name, "Texas Instruments (TI)", "HP Inc.
    (“HP”)"; two-letter EDGAR names such as "HP", case-sensitive)  -> "Company X"
  subsidiaries named after the firm ("Company X Partners", "Company X Capital") -> "Company X"
  ticker in exchange tags "(NYSE: WFC)" and as a standalone upper-case token, also the firm's other EDGAR tickers
    and share classes ("VIA" next to "VIAB") and lower-case brand forms ("pep+")   -> "Company X"
  a generic first word of the name used alone as subject or possessive ("Republic expects", "Republic's") -> "Company X"
  spaCy ORG not in a public-institution whitelist           -> "Organization A/B/..."
  spaCy (run on a straight-apostrophe copy: "Chris O’Connell"; all-caps contact lines "JAIME MARCUS" tagged in Title
    case; storm names "Hurricanes Helene and Milton" skipped) PERSON (and the surname and first name of every masked full name; a lone first or last name
    gets the full name's placeholder; "Mark Company X" for a founder-family CEO) -> "Executive A/B/..."
  spaCy PRODUCT, ®/™-marked names (every later mention too, with or without the mark) -> "Product A/B/..."
    entities are NOT masked when they are generic: acronyms of <= 5 letters (FFO, TPV, CCAR), units
    (Bcfe, MMBtu), spans with digits, spans made only of common words (wordfreq Zipf >= 3: "Card
    Services", "Regulated Businesses", "Notebooks", "Henry Hub"), spans containing a verb (headline
    fragments "Simplifies Movement of Data") or a generic term (GAAP, Act, Program, Guidance)
  name words: rare words of the firm name (wordfreq Zipf < 3.9, e.g. "Nabors", "Sands") and the name's first word
    ("Regions", "Duke"), case-sensitive; not the headquarters city ("Boston")  -> "Company X"
  datelines in any common form ("ATLANTA, Feb. 26, 2019 --", "Burlington, NC, [DATE] -", "NEW YORK; [DATE] —",
    "ARLINGTON, Va. and LONDON, [DATE] (GLOBE NEWSWIRE) --",
    "NEW YORK--(BUSINESS WIRE)--", "Minneapolis/[DATE]/", "McLean, Va. (")  -> removed
  headquarters: "City, ST 12345" / "City, State 12345" addresses, P.O. boxes, short address lines with a
    street ("83 Tower Road North") or a UK-style postcode ("BS30 8XP") -> removed;
    "headquartered in <City>[, <State>]" and the EDGAR business/mailing-address city -> "[LOCATION]"
  --mask-amounts: dollar amounts and "N million/billion"    -> "[AMT]" (percentages kept)
  full dates ("February 26, 2019")                          -> "[DATE]"
  founding years before 1990 ("Founded in 1907") -> "[YEAR]"; accounting standard numbers ("ASU No. 2016-09") -> "ASU [STANDARD]"
  years 1990-2049, not inside model numbers ("MHM-2020") (also FY24, Q1'23 / Q1’23, 4Q17, 1H19, 9 Mos’20, ACC.25, and "20301" = 2030 + footnote 1): Y -> "Year t",
    "Year t-1", "Year t+1" relative to the reporting year t = the fiscal year the release states in its first
    paragraphs ("fiscal 2024", "FY24"), else the fiscal year of the matched quarter end (EDGAR fiscalYearEnd:
    a December quarter of a September-FYE firm is t = next calendar year; a Q4 release published in January
    keeps t = the year reported), else the most frequent year in the first three paragraphs
  URLs (also without scheme, "tylertech.com") / e-mails / @handles / phone numbers (also international) -> removed
  exchange tags with or without parentheses ("NASDAQ: ZIONZ")      -> removed

Residual check (automatic, deliberately broader than the masking rules): any word (>= 4 letters) of
any firm name left in capitalized form regardless of frequency, the ticker as an upper-case token,
years 1990-2039, ZIP addresses and Title-case datelines. `residual_*` columns flag
documents that still contain any; a random 100-document manual audit is done separately.

Outputs
  data/interim/anon[_masked]/{accession}.txt  anonymized LLM input
  data/interim/anon_index[_masked].csv        per-document counts and residual flags
"""

import argparse
import json
import re
from collections import Counter

import pandas as pd
import spacy
from wordfreq import zipf_frequency

from common import INTERIM, SecClient, load_submissions

IN_DIR = INTERIM / "ex991"
KEEP_TYPES = {"narrative", "guidance", "heading"}

INSTITUTIONS = {
    "sec", "securities and exchange commission", "fda", "u.s. food and drug administration", "food and drug administration",
    "federal reserve", "the federal reserve", "fed", "fdic", "occ", "cfpb", "fasb", "iasb", "gaap", "u.s. gaap", "irs",
    "epa", "doj", "department of justice", "treasury", "u.s. treasury", "opec", "who", "cdc", "eu", "european union",
    "nyse", "nasdaq", "new york stock exchange", "s&p", "standard & poor's", "moody's", "fitch", "congress",
    "supreme court", "ftc", "federal trade commission", "faa", "ferc", "nrc", "cms", "medicare", "medicaid", "nhtsa",
    "department of defense", "dod", "u.s. government", "imf", "world bank", "ecb", "bank of england", "pcaob",
    "ebitda", "eps", "roe", "roa", "rotce", "cet1", "ceo", "cfo", "coo", "board", "board of directors", "the board",
    "company", "the company", "covid-19", "covid", "non-gaap", "r&d", "sg&a", "capex", "q1", "q2", "q3", "q4",
    "nymex", "comex", "pjm", "ercot", "miso", "caiso", "iso", "lme", "cboe", "finra", "naic", "nerc", "puc",
    "securities exchange act", "dodd-frank", "dodd-frank act", "cares act", "brexit", "basel iii", "ccar",
    "tax cuts and jobs act", "inflation reduction act", "affordable care act",
}
# entity spans containing one of these words are generic, not names
GENERIC_TERMS = {"approx", "yoy", "gaap", "non-gaap", "eps", "act", "regulation", "program", "programme", "guidance", "outlook", "results",
                 "report", "supplement", "segment", "quarter", "revenue", "revenues", "income", "earnings", "measures",
                 "ratio", "margin", "plan", "basel", "reform", "rule", "standard", "statement", "statements", "table",
                 "webcast", "conference", "relations", "communications", "highlights", "overview", "summary", "update",
                 "fiscal", "annual", "adjusted", "operating", "net", "total", "organic", "core", "ebitda", "ffo", "noi",
                 "repurchase", "repurchases", "dividend", "dividends", "share", "shares", "stock", "buyback",
                 "buybacks", "capital", "cash", "downturn", "response", "unchanged", "updates", "common", "newswire",
                 "wire", "box", "release", "call", "contact", "contacts", "ir", "vp", "investor", "investors", "media",
                 # metric acronyms and segment / channel words that spaCy tags as names
                 "revpar", "adr", "qoq", "asp", "asps", "voi", "medtech", "omnichannel", "ethernet", "gigabit",
                 "saas", "paas", "iaas", "ngfm", "ngfms", "passcode", "gbe", "pcie", "multifamily", "midstream",
                 "upstream", "downstream", "broking", "reinsurance", "urology", "hurricane", "hurricanes", "storm",
                 # diseases and biology terms (eponyms are not company names)
                 "disease", "syndrome", "crohn", "sjögren", "sjogren", "alzheimer", "parkinson", "huntington",
                 "hodgkin", "duchenne", "graves", "cushing", "igg", "ige", "iga"}
UNITS = {"bcf", "bcfe", "mcf", "mcfe", "mmcf", "mmcfe", "tcf", "mmbtu", "btu", "bbl", "bbls", "bbld", "mbbl", "mbbld",
         "boe", "boed", "boepd", "mboe", "mboed", "mmboe", "bopd", "mbopd", "ngl", "ngls", "lng", "mw", "mwh", "gw",
         "gwh", "kwh", "twh", "dth", "mdth", "mtpa", "teu", "asm", "asms", "rpm", "rasm", "casm", "alb", "albd", "bps"}
LEGAL_SUFFIX = r"(,?\s+(inc|incorporated|corp|corporation|co|company|plc|ltd|limited|llc|l\.p\.|lp|n\.v\.|s\.a\.|ag|se|"
LEGAL_SUFFIX += r"holdings?|group|& co|and company|international)\.?)*"
MONTHS = r"(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|june?|july?|aug(ust)?|sep(t(ember)?)?|oct(ober)?|nov(ember)?|dec(ember)?)\.?"
FULL_DATE = re.compile(rf"\b{MONTHS}\s+\d{{1,2}}(st|nd|rd|th)?,?\s+(19|20)\d{{2}}\b", re.I)
MONTH_YEAR = re.compile(rf"\b{MONTHS}\s+(19|20)\d{{2}}\b", re.I)
YEAR = re.compile(r"(?<![\d$.,])(?<!Suite )(?<!Box )(?<!No\. )(?<!#)(?<![A-Z]-)((?:199|20[0-4])\d)(?![\d%])")
# accounting standard numbers ("ASU No. 2016-09", "ASU 2016-02") would turn into "ASU Year t-6-09"
ASU = re.compile(r"\b(ASUs?|Accounting Standards Updates?)(\s+No\.)?\s+(?:19|20)\d{2}-\d{2,3}\b")
FY_SHORT = re.compile(r"\b(FY|Q[1-4]\s?|[1-4]Q|[12]H|\d{1,2} ?Mos\.?(?=['’])|[A-Z]{3,6}\.(?=\d{2}\b))['’]?(\d{2})\b(?!\.\d)")
# founding years before 1990 identify the firm ("Founded in 1907"); later years are mapped to Year t+-k
FOUNDED = re.compile(r"\b((?:founded|established|incorporated|formed|organized|since|dating back to)(?: in)?) (1[6-9][0-8]\d)\b", re.I)
YEAR_FOOTNOTE = re.compile(r"\b(in|by|of|through|for|year|fiscal|to|since|until|and|end) ((?:199|20[0-3])\d)\d(?=[\s,.;:)])")
FISCAL_STATED = re.compile(r"\b(?:fiscal(?: year)?|FY)\s?['’]?(20\d{2}|\d{2})\b", re.I)
DATELINE_END = re.compile(r"\s(?:--|[\-\u2013\u2014])\s|/[^/]{3,30}/\s*(?:--|[\-\u2013\u2014])?\s*")
EXCH_TAG = re.compile(r"\((NYSE|NASDAQ|Nasdaq|NasdaqGS|NYSE American|Nasdaq Global Select Market|NYSE Arca)\s*:\s*[A-Z.\-]{1,6}\)|"
                      r"\b(NYSE|NASDAQ|Nasdaq|NasdaqGS|NYSE American|NYSE Arca)\s*:\s*[A-Z][A-Z.\-]{0,6}\b", re.I)
# "said Mark Company X, Chair of the Board": a CEO whose surname is the firm's name (spaCy misses it once masked)
FOUNDER = re.compile(r"(?<!\bThe )\b[A-Z][a-z]+ Company X(?=,\s(?:the\s)?(?:Chair|Chairman|Chief|President|CEO|Executive|"
                     r"Vice|Senior|Founder|Co-Founder))")
STORM = re.compile(r"\b(?:Hurricanes?|Tropical Storms?|Winter Storms?|Typhoons?|Storms?)\s+([A-Z][a-z]+)(?:,?\s+(?:and\s+)?([A-Z][a-z]+))?")
SUBSIDIARY = re.compile(r"Company X (?:Partners|Capital|Midstream|Logistics|Shipbuilding|Holdings|Funding|Finance|"
                        r"Cruise Lines?|Lines?|Bank|Airlines)\b"
                        r"(?:,? (?:L\.P\.|LP|LLC|Inc\.?))?")
STREET = re.compile(r"\b\d{1,5}(?:st|nd|rd|th)? [A-Z][a-z]+(?: [A-Z][a-z]+){0,2} (?:Road|Street|Avenue|Drive|Way|Boulevard|"
                    r"Blvd|St|Ave|Lane|Esplanade|Parkway|Place|Plaza|Court|Square|Floor)\b\.?")
UK_POSTCODE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}\b")
DEFINED_ACRONYM = r"\s*\(\s*(?:the\s+)?[“\"]?([A-Z][A-Za-z&]{1,5})[”\"]?(?:\s+or\s+[“\"]?[A-Z][\w&]*[”\"]?)?\s*\)"
CONTACT = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\w+|\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}|(?<!\w)@\w{3,}|"
                     r"\+\d{1,3}[\s.\-]?\(?\d{1,4}\)?(?:[\s.\-]?\d{2,4}){2,4}\b|"
                     r"\b(?:[a-z0-9\-]+\.)+(?:com|net|org|io|co|us)(?:/\S*)?\b", re.I)
ZIP_ADDR = re.compile(r"(?:\b[A-Z][A-Za-z.'\-]+(?:\s[A-Z][A-Za-z.'\-]+){0,3},\s)?\b(?:[A-Z]{2}|[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)"
                      r",?\s\d{5}(?:-\d{4})?\b")
PO_BOX = re.compile(r"\bP\.?\s?O\.?\s?Box\s\d+,?\s?", re.I)
HQ_PHRASE = re.compile(r"\b(headquartered|based|headquarters|head office) (in|at|from|to) ((?:[A-Z][A-Za-z.'\-]+\s?){1,3}(?:,\s?[A-Z][A-Za-z.]+){0,2})")
WIRE = r"(?:\(?(?:BUSINESS\s?WIRE|GLOBE\s?NEWSWIRE|PR\s?NEWSWIRE|PRNewswire|ACCESSWIRE|MARKETWIRED?|Marketwire|GLOBE NEWSWIRE)\)?)"
TITLE_DATELINE = re.compile(r"^[A-Z][A-Za-z.'\-]+(?:\s[A-Z][A-Za-z.'\-]+){0,3},\s(?:[A-Z]{2}|[A-Z][a-z]+\.?)(?:,?\s[A-Z][a-z]+)?"
                            r",?\s*(?:\[DATE\])?\s*(?:/[^/]{3,30}/)?\s*(?:--|[\-\u2013\u2014])\s+")
AMOUNT = re.compile(r"\(?\$\s?\d[\d,]*(?:\.\d+)?\)?(?:\s?(?:million|billion|thousand|trillion|mm|bn|[mbk])\b)?"
                    r"|\b\d[\d,]*(?:\.\d+)?\s(?:million|billion|trillion)\b", re.I)
LEGAL_WORDS = {"inc", "incorporated", "corp", "corporation", "company", "holdings", "holding", "group", "limited",
               "international", "the", "and", "trust", "plc", "ltd", "llc"}
GENERIC_FIRST = {"american", "general", "united", "first", "national", "international", "public", "global", "new",
                 "southern", "northern", "western", "eastern", "central", "pacific", "atlantic", "federal", "state",
                 "the", "mid", "health", "life", "home", "best", "universal", "digital", "consolidated", "republic"}
US_STATES = {"alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware", "florida",
             "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
             "maryland", "massachusetts", "michigan", "minnesota", "mississippi", "missouri", "montana", "nebraska",
             "nevada", "ohio", "oklahoma", "oregon", "pennsylvania", "tennessee", "texas", "utah", "vermont",
             "virginia", "washington", "wisconsin", "wyoming", "carolina", "dakota", "jersey", "york", "america"}
BAD_WIKI = [("NWSA", "21st Century Fox")]  # Wikipedia name of a different company (checked by hand)
MARKED = re.compile(r"\b([A-Z][\w\-]*(?:\s+[A-Z][\w\-]*){0,3})\s?[®™™®]")


def strip_dateline(p):
    """Drop a leading "CITY, State, [DATE] --" dateline: the part before the first dash must start
    with an all-caps place name (>= 4 letters) and contain nothing but place/date tokens."""
    m = DATELINE_END.search(p[:90])
    if not m:
        return p, 0
    head = p[:m.start()].strip()
    city = re.split(r",|;|\(", head)[0].strip()
    rest = head[len(city):]
    if (sum(c.isalpha() for c in city) >= 4 and city == city.upper() and re.fullmatch(r"[A-Z .'&\-]+", city)
            and re.fullmatch(r"[\w .,;()\[\]\-\u2013\u2014]*", rest) and len(rest.split()) <= 8):
        p = p[m.end():].lstrip()
        return re.sub(r"^\[DATE\]\s*(?:/[^/]{3,30}/)?\s*(?:--|[\-\u2013\u2014])\s*", "", p), 1
    return p, 0


DATELINE_HEAD = re.compile(
    rf"^(?P<place>(?:[A-Z][A-Za-z.'\-]+\s?){{1,4}}(?:,\s?(?:[A-Z][A-Za-z.]+\s?){{1,2}}){{0,2}}),?\s*"
    rf"(?P<date>(?:[\-–—]+\s?)?\(?\s?\[DATE\]\s?\)?)?\s*(?P<wire>/[^/]{{2,40}}/|--\s?{WIRE}\s?--|\(?{WIRE}\)?)?\s*"
    rf"(?P<sep>--|[\-–—](?=\s)|:)?\s*")
STATE_TAIL = re.compile(r",\s?(?:[A-Z]{2}|[A-Z][a-z]{1,4}\.|[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s*$")


def strip_dateline_any(p):
    """Drop a leading dateline in the forms the all-caps rule misses: "NEW YORK--(BUSINESS WIRE)--",
    "Minneapolis/[DATE]/", "McLean, Va. ([DATE]) -", "Burlington, NC, [DATE] -". The head must be a short
    place phrase followed by a date or wire tag, or end in a state and be followed by a dash."""
    m = DATELINE_HEAD.match(p[:160])
    if not m or len(m.group("place").split()) > 6 or re.match(
            r"(Three|Six|Nine|Twelve|Quarter|Year|Fiscal|For|As|On|At|Effective|Through|Since|Period)\b", p):
        return p, 0
    has_tag = bool(m.group("date") or m.group("wire"))
    state = bool(STATE_TAIL.search(m.group("place")))
    if (has_tag and (m.group("sep") or m.group("wire") or state)) or (state and m.group("sep") and m.group("sep") != ":"):
        return p[m.end():].lstrip(" -–—"), 1
    return p, 0


def name_variants(names):
    out = set()
    for n in names:
        if not isinstance(n, str) or not n.strip():
            continue
        n = re.sub(r"\s*/[A-Z]{2,3}/?\s*$", "", n.strip())  # EDGAR state tags "/DE/"
        n = re.sub(r"\s*\(.*?\)", "", n)
        base = re.sub(LEGAL_SUFFIX + r"$", "", n, flags=re.I).strip(" ,.")
        forms = {n, base, re.sub(r"^the\s+", "", base, flags=re.I)}
        for a, b in [(r"\bCorp\b\.?", "Corporation"), (r"\bCo\b\.?", "Company"), (r"\s&\s", " and "),
                     (r"\bIntl\b\.?", "International"), (r"\bCos\b\.?", "Companies")]:
            forms |= {re.sub(a, b, f, flags=re.I) for f in forms}
        # EDGAR drops punctuation: "E TRADE" is spelled "E*TRADE" / "E-Trade"; "DOMINOS" is "Domino's"
        for f in list(forms):
            if re.search(r"\b[A-Za-z] [A-Za-z]{2,}", f):
                forms |= {re.sub(r"\b([A-Za-z]) ([A-Za-z]{2,})", rf"\1{j}\2", f) for j in ("*", "-", "")}
            if re.search(r"[A-Za-z]{4,}s\b", f, re.I):
                forms |= {re.sub(r"\b([A-Za-z]{4,})([sS])\b", rf"\1{q}\2", f) for q in ("'", "’")}
            if "'" in f:  # "PEOPLE'S UNITED" is printed "People’s United"
                forms.add(f.replace("'", "’"))
        for v in forms:
            v = v.strip(" ,.")
            if len(v) >= 3 and v.lower() not in INSTITUTIONS:
                out.add(v)
                out.add(v.title())
        # first two words of a multi-word name (e.g. "Wells Fargo" from "Wells Fargo & Company")
        w = base.split()
        if len(w) >= 3 and len(" ".join(w[:2])) >= 6 and all(x[0].isupper() and x.lower() not in {"of", "the", "and"} for x in w[:2]):
            out |= {" ".join(w[:2]), " ".join(w[:2]).replace("'", "’")}
    return sorted(out, key=len, reverse=True)


def name_words(names, hq=()):
    """Case-sensitive single-word aliases: rare words of the firm name and the name's first word
    (not the headquarters city: "Boston" in Boston Properties stays a place)."""
    hq = {h.lower() for h in hq}
    out = set()
    for n in names:
        if not isinstance(n, str):
            continue
        n = re.sub(r"['’]s\b", "", re.sub(r"\s*/[A-Z]{2,3}/?\s*$", "", n))  # "Domino's" -> "Domino"
        ws = re.findall(r"[A-Za-z]{4,}", n)
        for i, w in enumerate(ws):
            lw = w.lower()
            if lw in LEGAL_WORDS or lw in US_STATES or lw in hq:
                continue
            f = zipf_frequency(lw, "en")
            if lw in INSTITUTIONS and f >= 3.9:  # the firm's own rare name stays maskable ("Cboe")
                continue
            if f < 3.9 or (i == 0 and f < 5.0 and lw not in GENERIC_FIRST):
                out.add(w[0].upper() + w[1:].lower())
                out.add(w.upper())  # "AFLAC"; all-caps headlines ("CONSTELLATION REPORTS ...")
    return sorted(out, key=len, reverse=True)


# a generic first word used alone as the firm's name: "Republic expects revenue ...", "Republic's guidance"
SUBJECT_VERB = (r"(?=['’]s\b|\s(?:expects?|continues?|reported|reports|delivered|delivers|remains|announced|announces|"
                r"believes|generated|generates|achieved|completed|acquired|plans|anticipates|is|was|has|will|intends)\b)")


def generic_first(names):
    """First words of the firm's names that name_words() leaves out as too generic ("Republic", "United")."""
    out = set()
    for n in names:
        if isinstance(n, str):
            w = re.findall(r"[A-Za-z]{4,}", re.sub(r"\s*/[A-Z]{2,3}/?\s*$", "", n))
            if w and w[0].lower() in GENERIC_FIRST - {"the", "first", "best", "home", "life", "health"}:
                out.add(w[0][0].upper() + w[0][1:].lower())
    return out


class Placeholder:
    def __init__(self):
        self.maps = {}

    def get(self, kind, text):
        m = self.maps.setdefault(kind, {})
        key = text.lower().strip()
        if key not in m:
            i = len(m)
            m[key] = f"{kind} {chr(65 + i % 26)}{'' if i < 26 else i // 26}"
        return m[key]


def reference_year(paras, qend, fye=None):
    """t = fiscal year of the matched quarter end (EDGAR fiscalYearEnd "MMDD"; a 52/53-week year ending in the first
    days of January counts as a December year end). For January/February year ends, whose fiscal-year naming varies
    by firm (Walmart: by end year, Target: by start year), the fiscal year the release states ("fiscal 2016",
    "FY16") wins, then the most frequent year in the first three paragraphs."""
    fy_q = None
    if isinstance(qend, str) or (qend is not None and not pd.isna(qend)):
        q = pd.Timestamp(qend)
        m, d = (int(str(fye)[:2]), int(str(fye)[2:4])) if fye and re.fullmatch(r"\d{4}", str(fye)) else (12, 31)
        if m == 1 and d <= 10:
            m, d = 12, 31
        end = pd.Timestamp(q.year, m, min(d, 28 if m == 2 else d)) + pd.Timedelta(days=10)
        fy_q = q.year if q <= end else q.year + 1
        if m >= 3:
            return fy_q
    stated = Counter(int(y) if len(y) == 4 else 2000 + int(y) for p in paras[:10] for y in FISCAL_STATED.findall(p))
    if stated:
        return stated.most_common(1)[0][0]
    yrs = Counter(int(y) for p in paras[:3] for y in YEAR.findall(p))
    if yrs:
        return yrs.most_common(1)[0][0]
    return fy_q


def rel_year(y, t):
    d = int(y) - t
    return "Year t" if d == 0 else f"Year t{d:+d}"


ENT_EDGE = re.compile(r"^[\W\d_]+|[\W_]+$")


def clean_entity(text, label, span=None):
    """The maskable core of a spaCy entity, or None when the entity is generic (see module docstring)."""
    t = ENT_EDGE.sub("", re.sub(r"^(the|The)\s+|^[a-z]\.\s?(?=[A-Z])", "", text.strip()))  # "e.Working": footnote e
    t = re.sub(r"['’]s$", "", t)
    if label == "PERSON" and re.fullmatch(r"(?:[A-Z][a-z]+\.? ){1,2}Company X", t):
        return t  # "Mark Company X": a CEO whose surname is the firm's name
    if len(t) < 2 or re.search(r"\d|Company X|\[|Year t", t):
        return None
    if label != "PERSON" and span is not None and any(x.pos_ in ("VERB", "AUX") for x in span):
        return None  # headline fragments: "Simplifies Movement of Data", "Wirestorm Delivers DevOps"
    low = t.lower()
    if low in INSTITUTIONS:
        return None
    toks = re.findall(r"[A-Za-z][A-Za-z&'\-.]*", t)
    ltoks = [x.lower().strip(".") for x in toks]
    if not toks or any(x in GENERIC_TERMS for x in ltoks) or any(x in UNITS for x in ltoks):
        return None
    if re.fullmatch(r"[A-Z&.\-]{1,5}", t):  # acronyms: FFO, TPV, CCAR, OEM
        return None
    content = [x for x in ltoks if x not in {"of", "and", "the", "&", "for"}]
    if label == "PERSON":
        if any(x in LEGAL_WORDS for x in ltoks):
            return None  # "Cameron Group" is a business, not a person
        if t.isupper() or (len(toks) == 1 and all(zipf_frequency(x, "en") >= 4.0 for x in content)):
            return None
    elif all(zipf_frequency(x, "en") >= 3.0 for x in content):
        return None  # "Regulated Businesses", "Notebooks", "Specialty Materials"
    return t


def defined_acronyms(paras, aliases):
    """Short names the release defines right after the firm's name: "Texas Instruments (TI)", "HP Inc. (“HP”)"."""
    if not aliases:
        return set()
    pat = re.compile(r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")" + LEGAL_SUFFIX + DEFINED_ACRONYM, re.I)
    out = set()
    for p in paras:
        for m in pat.finditer(p):
            a = m.group(m.lastindex)
            if a and sum(c.isupper() for c in a) >= 2 and a.lower() not in INSTITUTIONS:
                out.add(a)
    return out


def anonymize(paras, aliases, words, tickers, qend, nlp, mask_amounts=False, hq=(), fye=None, short=(), generic=()):
    t_ref = reference_year(paras, qend, fye) or pd.Timestamp.now().year
    ph = Placeholder()
    words = sorted(set(words) | set(short) | defined_acronyms(paras, aliases), key=len, reverse=True)
    alias_re = re.compile(r"\b(" + "|".join(re.escape(a) for a in aliases) + r")('s|’s)?(?!\w)", re.I) if aliases else None
    word_re = re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")('s|’s)?\b") if words else None
    tickers = sorted({x for x in tickers if isinstance(x, str) and len(x) >= 2}, key=len, reverse=True)
    # every class of the firm's stock ("VIA" next to "VIAB"); lower case with a "+" is a brand ("pep+")
    ticker_re = re.compile(r"(?<![\w$])(" + "|".join(re.escape(x) for x in tickers) + r")(?![\w])|(?<![\w$])(" +
                           "|".join(re.escape(x.lower()) for x in tickers) + r")(?=\+)") if tickers else None
    generic_re = re.compile(r"\b(" + "|".join(re.escape(w) for w in sorted(generic)) + r")" + SUBJECT_VERB) if generic else None
    hq_re = re.compile(r"\b(" + "|".join(re.escape(h) for h in sorted(hq, key=len, reverse=True)) + r")\b(,\s?[A-Z][A-Za-z.]+)?",
                       re.I) if hq else None
    stats = Counter()
    marked = set()
    # pass 1: contacts, dates, datelines, addresses, the firm's own names
    tmp = []
    for i, p in enumerate(paras):
        p, n = EXCH_TAG.subn("", p)
        stats["exch_tag"] += n
        p = CONTACT.sub("", p)
        p, n = FULL_DATE.subn("[DATE]", p)
        stats["date"] += n
        # the dateline often follows a run of headline bullets, so every paragraph is checked; the looser
        # forms only where a date or wire tag sits in the first 160 characters
        p, n = strip_dateline(p)  # "ATLANTA, [DATE] --" / "SAN FRANCISCO --"
        if not n and (i < 8 or re.search(rf"\[DATE\]|{WIRE}", p[:160])):
            p, n = strip_dateline_any(p)
            if not n:
                p, n = TITLE_DATELINE.subn("", p, count=1)
        stats["dateline"] += n
        # the firm's own names first, so that "Boston Properties" is not cut by the HQ city "Boston"
        if alias_re:
            p, n = alias_re.subn(lambda m: "Company X" + ("'s" if m.group(2) else ""), p)
            stats["alias"] += n
        if ticker_re:
            p, n = ticker_re.subn("Company X", p)
            stats["ticker"] += n
        if word_re:
            p, n = word_re.subn(lambda m: "Company X" + ("'s" if m.group(2) else ""), p)
            stats["name_word"] += n
        if generic_re:
            p, n = generic_re.subn("Company X", p)
            stats["name_word"] += n
        p = PO_BOX.sub("", p)
        p, n = ZIP_ADDR.subn("", p)
        stats["zip_addr"] += n
        if len(p.split()) <= 30 and (STREET.search(p) or UK_POSTCODE.search(p)):
            p = ""  # address line of a contact block ("83 Tower Road North, Warmley, Bristol, England, BS30 8XP")
            stats["address"] += 1
        p, n = HQ_PHRASE.subn(lambda m: f"{m.group(1)} {m.group(2)} [LOCATION]", p)
        stats["hq"] += n
        if re.search(r"headquarter", p, re.I):  # "relocate its headquarters from Omaha, NE to Chicago, IL"
            p, n = re.subn(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?,\s?[A-Z]{2}\b", "[LOCATION]", p)
            stats["hq"] += n
        if hq_re:  # the HQ city next to a dateline-like position or "headquarters"
            if i < 3 or re.search(r"headquarter", p, re.I):
                p, n = hq_re.subn("[LOCATION]", p)
                stats["hq"] += n
        p = SUBSIDIARY.sub("Company X", p)
        p = re.sub(r"Company X(?:\s+Company X)+", "Company X", p)
        p = re.sub(r"Company X\s?[®™™®]", "Company X", p)
        for m in MARKED.finditer(p):  # ®/™ brands: masked in every mention, marked or not (pass 2)
            if "Company X" not in m.group(1) and m.group(1).lower() not in INSTITUTIONS:
                marked.add(m.group(1))
        p = re.sub(r"\s?[®™]", "", p)
        tmp.append(p)
    # pass 2: named entities over the whole document -> one placeholder per string
    found = {m: ph.get("Product", m) for m in sorted(marked, key=len, reverse=True)}
    stats["marked"] += len(found)
    storms = {w for m in STORM.finditer(" ".join(tmp)) for w in m.groups() if w}
    for p in tmp:  # an all-caps contact line spaCy cannot tag ("JAIME MARCUS"): tag its Title-case form
        if 2 <= len(p.split()) <= 4 and re.fullmatch(r"[A-Z][A-Z.'’\- ]+", p):
            ents = nlp(p.title()).ents
            if len(ents) == 1 and ents[0].label_ == "PERSON" and ents[0].text == p.title() and p not in found:
                found[p] = ph.get("Executive", p.title())
    # spaCy misses names with a curly apostrophe ("Chris O’Connell"): tag a straight-quote copy, take the original span
    for orig, doc in zip(tmp, nlp.pipe(x.replace("’", "'") for x in tmp)):
        for e in doc.ents:
            kind = {"ORG": "Organization", "PERSON": "Executive", "PRODUCT": "Product"}.get(e.label_)
            if not kind:
                continue
            t = clean_entity(orig[e.start_char:e.end_char], e.label_, e)
            if t in storms:
                continue  # "Hurricanes Helene and Milton"
            if t and t not in found:
                found[t] = ph.get(kind, t)
                if kind == "Executive" and len(t.split()) >= 2:  # surname alone ("Heminger")
                    last = t.split()[-1]
                    if len(last) >= 4 and zipf_frequency(last.lower(), "en") < 5.0 and last not in found:
                        found[last] = found[t]
    # first and last names of masked executives ("Susan, John, David, and Nick"), unless the word is also used in lower
    # case ("will", "mark"); a lone first or last name takes the full name's placeholder (consistent IDs)
    low_text = " ".join(tmp)
    for full, tag in list(found.items()):
        parts = [x for x in full.split() if not re.fullmatch(r"[A-Z]\.?", x)]  # skip initials ("M. Susan Hardwick")
        if not tag.startswith("Executive") or len(parts) < 2 or "Company X" in full:
            continue
        for part in (parts[0], parts[-1]):
            if len(part) < 3 or not re.fullmatch(r"[A-Z][a-z’'\-]+", part) or part in {"Mr", "Mrs", "Ms", "Dr"}:
                continue
            if re.search(rf"\b{re.escape(part.lower())}\b", low_text) or part.lower() in LEGAL_WORDS | GENERIC_TERMS:
                continue
            if part == parts[-1] and zipf_frequency(part.lower(), "en") >= 5.0:
                continue
            if part not in found or (found[part].startswith("Executive") and found[part] != tag):
                found[part] = tag
    stats.update({k: sum(v.startswith(k) for v in found.values()) for k in ("Organization", "Executive", "Product")})
    # not the first half of a hyphenated name ("Dodd-Frank"); "Brand-related" is still masked
    ent_re = re.compile(r"(?<![\w])(" + "|".join(re.escape(k) for k in sorted(found, key=len, reverse=True)) + r")(?![\w]|-[A-Z])") \
        if found else None
    out = []
    for p in tmp:
        if ent_re:
            p = ent_re.sub(lambda m: found[m.group(1)], p)
        p = FOUNDER.sub(lambda m: ph.get("Executive", m.group(0)), p)
        # pass 3: amounts, years
        if mask_amounts:
            p, n = AMOUNT.subn("[AMT]", p)
            stats["amount"] += n
        p = MONTH_YEAR.sub(lambda m: f"{m.group(1)} {rel_year(m.group(0)[-4:], t_ref)}", p)
        p = FY_SHORT.sub(lambda m: f"{m.group(1).strip()} {rel_year('20' + m.group(2), t_ref)}", p)
        p = FOUNDED.sub(lambda m: f"{m.group(1)} [YEAR]", p)
        p = ASU.sub(lambda m: m.group(1) + " [STANDARD]", p)
        p = YEAR_FOOTNOTE.sub(lambda m: f"{m.group(1)} {rel_year(m.group(2), t_ref)}", p)
        p, n = YEAR.subn(lambda m: rel_year(m.group(1), t_ref), p)
        stats["year"] += n
        p = re.sub(r"\s{2,}", " ", p).strip(" ,;-")
        if p:
            out.append(p)
    return out, stats, t_ref


def residuals(text, names, ticker):
    toks = {w for n in names if isinstance(n, str) for w in re.findall(r"[A-Za-z]{4,}", n)
            if w.lower() not in LEGAL_WORDS and w.lower() not in GENERIC_FIRST and w.lower() not in US_STATES}
    hit_alias = sorted(w for w in {t[0].upper() + t[1:].lower() for t in toks} if re.search(rf"\b{w}\b", text))
    hit_place = bool(ZIP_ADDR.search(text)) or any(TITLE_DATELINE.match(p) for p in text.split("\n\n"))
    hit_ticker = bool(ticker and len(ticker) >= 2 and re.search(rf"(?<![\w$]){re.escape(ticker)}(?!\w)", text))
    hit_year = YEAR.findall(text)
    return hit_alias, hit_ticker, hit_year, hit_place


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=str(INTERIM / "pilot_events.csv"))
    ap.add_argument("--mask-amounts", action="store_true", help="also replace dollar amounts by [AMT]")
    ap.add_argument("--index", default="anon_index", help="index file stem under data/interim/")
    args = ap.parse_args()
    sfx = "_masked" if args.mask_amounts else ""
    out_dir = INTERIM / f"anon{sfx}"

    nlp = spacy.load("en_core_web_md", disable=["lemmatizer"])
    sec = SecClient()
    ev = pd.read_csv(args.events)
    u = pd.read_csv(INTERIM / "universe.csv")
    # a Wikipedia name shared by two tickers belongs to the better-matching one (MPC got "Marathon Oil")
    best = u.sort_values("name_sim").drop_duplicates("wiki_name", keep="last").set_index("wiki_name").ticker
    bad = u.wiki_name.map(best).ne(u.ticker) | u.set_index(["ticker", "wiki_name"]).index.isin(BAD_WIKI)
    u.loc[bad, "wiki_name"] = None
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in ev.itertuples():
        f = IN_DIR / f"{r.accession}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        paras = [p["text"] for p in d["paragraphs"] if p["type"] in KEEP_TYPES]
        info, _ = load_submissions(sec, r.cik)
        names = [info["name"], *[x["name"] for x in info.get("formerNames") or []]] if info else []
        names += u.loc[u.cik == r.cik, ["wiki_name", "edgar_name"]].stack().tolist()
        aliases = name_variants(names)
        addr = (info or {}).get("addresses") or {}
        hq = {a["city"].title() for a in addr.values() if a and a.get("city") and len(a["city"]) >= 4}
        short = {b for b in (re.sub(LEGAL_SUFFIX + r"$", "", re.sub(r"\s*/[A-Z]{2,3}/?\s*$", "", n), flags=re.I).strip(" ,.")
                             for n in names if isinstance(n, str)) if re.fullmatch(r"[A-Z]{2}", b)}  # "HP INC" -> "HP"
        tickers = [r.ticker, *[re.sub(r"[-.].*$", "", x) for x in (info or {}).get("tickers") or []]]
        if len(r.ticker) >= 4 and r.ticker[-1] in "ABKL":  # other share classes: VIAB -> VIA, GOOGL -> GOOG
            tickers += [r.ticker[:-1]] + [r.ticker[:-1] + c for c in "ABK"]
        anon, stats, t_ref = anonymize(paras, aliases, name_words(names, hq), tickers, r.qend, nlp, args.mask_amounts,
                                       hq, (info or {}).get("fiscalYearEnd"), short, generic_first(names))
        text = "\n\n".join(anon)
        (out_dir / f"{r.accession}.txt").write_text(text)
        ha, ht, hy, hp = residuals(text, names, r.ticker)
        rows.append({"accession": r.accession, "cik": r.cik, "ticker": r.ticker, "t_ref": t_ref,
                     "words_in": sum(len(p.split()) for p in paras), "words_out": len(text.split()),
                     **{f"n_{k}": v for k, v in stats.items()},
                     "residual_alias": ";".join(ha), "residual_ticker": ht, "residual_year": len(hy),
                     "residual_place": hp})
    idx = pd.DataFrame(rows)
    idx.to_csv(INTERIM / f"{args.index}{sfx}.csv", index=False)
    print(f"docs={len(idx)}  residual name word={idx.residual_alias.fillna('').ne('').mean():.1%}  "
          f"ticker={idx.residual_ticker.mean():.1%}  year={(idx.residual_year > 0).mean():.1%}  "
          f"place={idx.residual_place.mean():.1%}")
    print(idx.filter(like="n_").sum().to_string())


if __name__ == "__main__":
    main()
