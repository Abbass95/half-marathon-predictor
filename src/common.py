"""Shared helpers: config, paths, secrets, leakage guard, prediction registry."""
from __future__ import annotations

import os
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

RACE_DATE: date = CFG["race"]["date"]
RACE_KM: float = CFG["race"]["distance_km"]
RACE_ID: int = CFG["race"]["strava_activity_id"]

STRAVA_DIR = ROOT / CFG["paths"]["strava_export"]
RAW = ROOT / CFG["paths"]["raw"]
PROC = ROOT / CFG["paths"]["processed"]
OUT = ROOT / CFG["paths"]["outputs"]
for _p in (RAW, PROC, OUT):
    _p.mkdir(parents=True, exist_ok=True)

# Windows consoles default to cp1252; French names contain characters it cannot print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_env() -> None:
    """Minimal .env loader (tolerates 'KEY = value' with spaces and quotes)."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def assert_pre_race(df: pd.DataFrame, col: str, name: str) -> None:
    """Leakage guard: no row may be dated on/after race day."""
    d = pd.to_datetime(df[col])
    if d.dt.tz is not None:
        d = d.dt.tz_convert(CFG["race"]["timezone"]).dt.tz_localize(None)
    bad = d.dt.date >= RACE_DATE
    if bad.any():
        raise AssertionError(f"LEAKAGE in {name}: {bad.sum()} rows dated >= {RACE_DATE}")


def fmt_hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def fmt_pace(sec_per_km: float) -> str:
    s = int(round(sec_per_km))
    return f"{s // 60}:{s % 60:02d}/km"


PRED_FILE = OUT / "predictions.csv"
PRED_COLS = ["family", "method", "predicted_s", "low_s", "high_s", "notes", "created"]


def save_predictions(rows: list[dict], family: str) -> pd.DataFrame:
    """Upsert this family's rows into outputs/predictions.csv (other families untouched)."""
    new = pd.DataFrame(rows)
    new["family"] = family
    new["created"] = datetime.now().isoformat(timespec="seconds")
    for c in PRED_COLS:
        if c not in new:
            new[c] = None
    new = new[PRED_COLS]
    if PRED_FILE.exists():
        old = pd.read_csv(PRED_FILE)
        old = old[old["family"] != family]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(PRED_FILE, index=False)
    return new
