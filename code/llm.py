"""Async DeepSeek client: one model for every agent, bounded concurrency, retries, resumable raw logs.

Every call is keyed by (task, key, repeat); the full request + response is written to
data/interim/llm_raw/{task}/{key}__r{repeat}.json and re-used on re-runs (resume after interruption).
A cached record is re-used only if its request (model, settings, messages) hashes to the same value as the
new one; otherwise it is renamed to *.json.stale-{time} (kept for the record, ignored by usage_summary)
and the call is sent again, so editing a prompt can never silently return answers to the old prompt.
Model settings are fixed here so all agents share them: deepseek-flash, temperature 0, JSON mode.
"""

import asyncio
import hashlib
import json
import os
import random
import re
import time
from collections import Counter

import httpx

from common import INTERIM

MODEL = "deepseek-flash"  # DeepSeek-V4.1-Flash; response system_fingerprint is logged per call
TEMPERATURE = 0.0
REASONING_EFFORT = "low"
RAW_DIR = INTERIM / "llm_raw"


BARE_PARA = re.compile(r"(?<=[\[,\s])P(\d+)(?=\s*[\],])")
TRAILING_COMMA = re.compile(r",(\s*[\]}])")


def parse_json(content):
    """json.loads, tolerating paragraph ids written as bare P12 inside lists ([P4,P12] -> [4,12])
    and trailing commas ({..}, ] -> {..}])."""
    fixed = TRAILING_COMMA.sub(r"\1", BARE_PARA.sub(r"\1", content or ""))
    for text in (content, fixed):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return None


def request_hash(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


class LLM:
    def __init__(self, concurrency=16, max_tokens=4000, reasoning_effort=REASONING_EFFORT):
        self.base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/")
        self.key = os.environ["DEEPSEEK_API_KEY"]
        self.sem = asyncio.Semaphore(concurrency)
        self.retries = Counter()  # retried responses by status (429 = DeepSeek rate limit), or "network"
        self.max_tokens = max_tokens
        self.effort = reasoning_effort
        # httpx pools at most 100 connections by default, which would cap concurrency above 100
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(300.0),
                                        limits=httpx.Limits(max_connections=concurrency,
                                                            max_keepalive_connections=concurrency))

    async def close(self):
        await self.client.aclose()

    async def chat_json(self, task, key, system, user, repeat=0):
        """Return (parsed JSON or None, record). Cached on disk by (task, key, repeat)."""
        path = RAW_DIR / task / f"{key}__r{repeat}.json"
        body = {"model": MODEL, "temperature": TEMPERATURE, "max_tokens": self.max_tokens,
                "reasoning_effort": self.effort, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        h = request_hash(body)
        if path.exists():
            rec = json.loads(path.read_text())
            if (rec.get("request_hash") or request_hash(rec["request"])) == h:
                parsed = rec.get("parsed")
                if parsed is None:  # records written before parse_json learned a repair
                    parsed = rec["parsed"] = parse_json(rec["response"]["choices"][0]["message"].get("content"))
                if parsed is not None or rec["response"]["choices"][0].get("finish_reason") != "length":
                    return parsed, rec
            path.rename(path.with_name(f"{path.name}.stale-{int(time.time())}"))
        t0, send, truncated = time.time(), body, []
        resp = await self._post(send)
        # reasoning ran out of room: resend with a larger budget; the cache key stays the base request's hash
        while resp["choices"][0].get("finish_reason") == "length" and send["max_tokens"] < 4 * self.max_tokens:
            truncated.append(resp["usage"])  # billed too, counted by usage_summary
            send = {**send, "max_tokens": 2 * send["max_tokens"]}
            resp = await self._post(send)
        content = resp["choices"][0]["message"].get("content") or ""
        parsed = parse_json(content)
        rec = {"task": task, "key": key, "repeat": repeat, "ts": t0, "request": send, "response": resp,
               "parsed": parsed, "latency_s": round(time.time() - t0, 2),
               "system_fingerprint": resp.get("system_fingerprint"), "request_hash": h,
               "truncated_usage": truncated}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        return parsed, rec

    async def _post(self, body):
        async with self.sem:
            for attempt in range(6):
                try:
                    r = await self.client.post(f"{self.base}/chat/completions", json=body,
                                               headers={"Authorization": f"Bearer {self.key}"})
                    if r.status_code in (429, 500, 502, 503, 504):
                        self.retries[r.status_code] += 1
                        raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                    r.raise_for_status()
                    resp = r.json()
                    break
                except (httpx.HTTPError, json.JSONDecodeError) as e:
                    if not isinstance(e, httpx.HTTPStatusError):
                        self.retries["network"] += 1
                    if attempt == 5:
                        raise
                    await asyncio.sleep(2 ** attempt + random.random())
        return resp


# CNY per 1M tokens (DeepSeek pricing page): (cache-hit input, cache-miss input, output)
PRICE_PEAK = (0.04, 2.0, 8.0)
PRICE_OFFPEAK = (0.02, 1.0, 4.0)  # weekends/holidays and weekday hours outside 9-12, 14-18 Beijing


def is_peak(ts):
    """Beijing weekday 09:00-12:00 / 14:00-18:00. Chinese public holidays are not modelled (-> peak)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    t = datetime.fromtimestamp(ts, ZoneInfo("Asia/Shanghai"))
    return t.weekday() < 5 and (9 <= t.hour < 12 or 14 <= t.hour < 18)


def usage_summary(task):
    """Token totals and cost (CNY) over all cached calls of a task."""
    tot = {"calls": 0, "prompt": 0, "cache_hit": 0, "completion": 0, "reasoning": 0, "cost_cny": 0.0,
           "peak_calls": 0}
    for p in (RAW_DIR / task).glob("*.json"):
        rec = json.loads(p.read_text())
        peak = is_peak(rec.get("ts", time.time()))
        ph, pm, po = PRICE_PEAK if peak else PRICE_OFFPEAK
        tot["calls"] += 1
        tot["peak_calls"] += peak
        for u in [rec["response"]["usage"], *rec.get("truncated_usage", [])]:
            hit = u.get("prompt_cache_hit_tokens", 0)
            miss = u.get("prompt_cache_miss_tokens", u.get("prompt_tokens", 0) - hit)
            out = u.get("completion_tokens", 0)
            tot["prompt"] += u.get("prompt_tokens", 0)
            tot["cache_hit"] += hit
            tot["completion"] += out
            tot["reasoning"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
            tot["cost_cny"] += (hit * ph + miss * pm + out * po) / 1e6
    tot["cost_cny"] = round(tot["cost_cny"], 4)
    return tot
