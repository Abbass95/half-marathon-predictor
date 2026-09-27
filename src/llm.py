"""Thin Claude API helper: structured predictions, on-disk response cache, token/cost accounting."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

import anthropic

from .common import CFG, OUT, load_env

MODEL = CFG["llm"]["model"]
SMALL_MODEL = CFG["llm"]["small_model"]
SAMPLES = CFG["llm"]["samples"]
# USD per million tokens (input, output) - Anthropic first-party list prices.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-haiku-4-5": (1.0, 5.0)}
CACHE_FILE = OUT / "llm_responses.json"

SYSTEM = (
    "You are an experienced running coach and exercise physiologist. You predict race finish times "
    "from a runner's data. The data cutoff is the evening of 2026-09-19, the day before the race: "
    "you have no information about the race result, so reason only from the evidence provided. "
    "Give your single best estimate of the official finish time plus a realistic 90% interval."
)

PREDICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "predicted_seconds": {"type": "integer"},
        "low_seconds": {"type": "integer"},
        "high_seconds": {"type": "integer"},
        "reasoning": {"type": "string"},
    },
    "required": ["predicted_seconds", "low_seconds", "high_seconds", "reasoning"],
    "additionalProperties": False,
}

_client = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        load_env()
        _client = anthropic.Anthropic()
    return _client


def _load_cache() -> dict:
    return json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")


def cost_usd(model: str, usage: dict) -> float:
    pin, pout = PRICES[model]
    cached = usage.get("cache_read_input_tokens") or 0
    written = usage.get("cache_creation_input_tokens") or 0
    return (usage["input_tokens"] * pin + written * pin * 1.25 + cached * pin * 0.1
            + usage["output_tokens"] * pout) / 1e6


def cache_key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def predict(tag: str, user: str, model: str = MODEL, sample: int = 0, system: str = SYSTEM,
            effort: str = "high") -> dict:
    """One structured prediction. Cached by (model, prompts, sample index) -> reruns are free."""
    cache = _load_cache()
    key = cache_key(model, system, user, sample)
    if key in cache:
        return cache[key]
    output_config = {"format": {"type": "json_schema", "schema": PREDICTION_SCHEMA}}
    if model != SMALL_MODEL:  # effort is not supported on Haiku 4.5
        output_config["effort"] = effort
    r = client().messages.create(
        model=model, max_tokens=16000, system=system,
        cache_control={"type": "ephemeral"},  # samples share the long prompt prefix
        messages=[{"role": "user", "content": user}],
        output_config=output_config,
    )
    usage = r.usage.to_dict()
    entry = {"tag": tag, "model": model, "sample": sample, "stop_reason": r.stop_reason,
             "usage": usage, "cost_usd": cost_usd(model, usage),
             "created": datetime.now().isoformat(timespec="seconds")}
    if r.stop_reason == "refusal":
        entry["output"] = None
    else:
        entry["output"] = json.loads(next(b.text for b in r.content if b.type == "text"))
    cache[key] = entry
    _save_cache(cache)
    return entry


def record(key: str, entry: dict) -> None:
    """Store an arbitrary entry (e.g. an agent transcript) in the same cache file."""
    cache = _load_cache()
    cache[key] = entry
    _save_cache(cache)


def lookup(key: str) -> dict | None:
    return _load_cache().get(key)
