"""Shared paths, environment loading and a polite, disk-cached SEC HTTP client."""

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
SEC_CACHE = RAW / "sec_cache"

SAMPLE_START = "2015-01-01"
SAMPLE_END = "2025-12-31"


def load_env():
    """Read KEY=VALUE lines from ROOT/.env into os.environ (existing vars win)."""
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "PEAD Research research@example.com")


class SecClient:
    """GET with SEC fair-access throttling (<10 req/s), retries, and a disk cache."""

    def __init__(self, max_per_sec=8, cache_dir=SEC_CACHE):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"})
        self.min_gap = 1.0 / max_per_sec
        self.last = 0.0
        self.lock = threading.Lock()
        self.cache_dir = Path(cache_dir)

    def _cache_path(self, url):
        h = hashlib.sha1(url.encode()).hexdigest()
        return self.cache_dir / h[:2] / h

    def get(self, url, use_cache=True):
        """Return response bytes, or None on 404."""
        p = self._cache_path(url)
        if use_cache and p.exists():
            return p.read_bytes()
        for attempt in range(5):
            with self.lock:
                wait = self.min_gap - (time.time() - self.last)
                if wait > 0:
                    time.sleep(wait)
                self.last = time.time()
            try:
                r = self.s.get(url, timeout=30)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 404:
                return None
            if r.status_code in (429, 503) or r.status_code >= 500:
                time.sleep(2 ** attempt * 2)
                continue
            r.raise_for_status()
            if use_cache:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(r.content)
            return r.content
        raise RuntimeError(f"SEC GET failed after retries: {url}")

    def get_json(self, url, use_cache=True):
        b = self.get(url, use_cache)
        return None if b is None else json.loads(b)


SUB_COLS = ["accessionNumber", "filingDate", "acceptanceDateTime", "form", "items", "primaryDocument"]


def load_submissions(sec, cik):
    """All filings of a CIK (recent block + paginated older files) as a DataFrame, plus header info."""
    import pandas as pd

    js = sec.get_json(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json")
    if js is None:
        return None, pd.DataFrame(columns=SUB_COLS)
    parts = [pd.DataFrame(js["filings"]["recent"])]
    for f in js["filings"].get("files", []):
        more = sec.get_json("https://data.sec.gov/submissions/" + f["name"])
        if more:
            parts.append(pd.DataFrame(more))
    df = pd.concat(parts, ignore_index=True)
    df = df[[c for c in SUB_COLS if c in df.columns]].drop_duplicates("accessionNumber")
    df["filingDate"] = pd.to_datetime(df.filingDate)
    info = {k: js.get(k) for k in ("name", "tickers", "exchanges", "sic", "sicDescription", "formerNames", "addresses",
                                    "fiscalYearEnd")}
    return info, df
