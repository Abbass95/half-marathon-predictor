"""Pull Garmin Connect wellness data with the unofficial `garminconnect` library.

Credentials come from .env (GARMIN_EMAIL / GARMIN_PASSWORD). OAuth tokens are cached in
~/.garminconnect (outside the repo) so later runs don't need the password or MFA.

Outputs
  data/raw/garmin/day_<date>.json      raw API payloads per day (cache)
  data/raw/garmin/profile.json         user profile / settings (age, weight, height)
  data/raw/garmin/race_predictions.json
  data/processed/garmin_daily.csv      tidy daily table (morning metrics keyed to the *night before*)
"""
from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .common import CFG, PROC, RACE_DATE, RAW, load_env

GDIR = RAW / "garmin"
GDIR.mkdir(parents=True, exist_ok=True)
TOKENS = Path.home() / ".garminconnect"

DAILY_CALLS = {
    "sleep": "get_sleep_data",
    "hrv": "get_hrv_data",
    "rhr": "get_rhr_day",
    "stats": "get_stats",
    # get_training_readiness / get_training_status return nothing for this watch model.
    "max_metrics": "get_max_metrics",
}


def connect():
    from garminconnect import Garmin

    load_env()
    # login(tokenstore) reuses cached tokens, falls back to credentials, and saves new tokens.
    g = Garmin(os.environ["GARMIN_EMAIL"], os.environ["GARMIN_PASSWORD"],
               prompt_mfa=lambda: input("Garmin MFA code: "))
    g.login(str(TOKENS))
    return g


def safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}


def fetch(g) -> None:
    start = CFG["garmin"]["start_date"]
    # Race-day morning metrics (last night's sleep/HRV) are known before the 10:18 start,
    # so we fetch through race day and re-key them to the night before in `tidy`.
    days = [start + timedelta(d) for d in range((RACE_DATE - start).days + 1)]
    for d in days:
        f = GDIR / f"day_{d}.json"
        if f.exists():
            continue
        payload = {k: safe(getattr(g, m), d.isoformat()) for k, m in DAILY_CALLS.items()}
        f.write_text(json.dumps(payload, default=str), encoding="utf-8")
        print(f"fetched {d}")

    prof = GDIR / "profile.json"
    if not prof.exists():
        payload = {"settings": safe(g.get_userprofile_settings),
                   "weigh_ins": safe(g.get_weigh_ins, start.isoformat(), RACE_DATE.isoformat())}
        prof.write_text(json.dumps(payload, default=str), encoding="utf-8")

    rp = GDIR / "race_predictions.json"
    if not rp.exists():
        payload = safe(g.get_race_predictions, start.isoformat(),
                       (RACE_DATE - timedelta(1)).isoformat(), "daily")
        rp.write_text(json.dumps(payload, default=str), encoding="utf-8")


def g_(d, *path):
    """Nested get that tolerates missing keys / lists / error payloads."""
    for p in path:
        if isinstance(d, list):
            d = d[0] if d else None
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def tidy() -> pd.DataFrame:
    rows = []
    for f in sorted(GDIR.glob("day_*.json")):
        d = date.fromisoformat(f.stem[4:])
        j = json.loads(f.read_text(encoding="utf-8"))
        s = g_(j, "sleep", "dailySleepDTO") or {}
        rows.append({
            "date": d,
            # Morning metrics describe the night that ended on `d` -> attribute to d-1 evening.
            "night_of": d - timedelta(1),
            "sleep_h": (s.get("sleepTimeSeconds") or 0) / 3600 or None,
            "sleep_score": g_(s, "sleepScores", "overall", "value"),
            "deep_sleep_h": (s.get("deepSleepSeconds") or 0) / 3600 or None,
            "hrv_overnight_ms": g_(j, "hrv", "hrvSummary", "lastNightAvg"),
            "hrv_weekly_ms": g_(j, "hrv", "hrvSummary", "weeklyAvg"),
            "hrv_status": g_(j, "hrv", "hrvSummary", "status"),
            "resting_hr": g_(j, "stats", "restingHeartRate"),
            "stress_avg": g_(j, "stats", "averageStressLevel"),
            "body_battery_max": g_(j, "stats", "bodyBatteryHighestValue"),
            "body_battery_wake": g_(j, "stats", "bodyBatteryAtWakeTime"),
            "steps": g_(j, "stats", "totalSteps"),
            "vo2max": g_(j, "max_metrics", "generic", "vo2MaxPreciseValue")
                      or g_(j, "max_metrics", "generic", "vo2MaxValue"),
        })
    df = pd.DataFrame(rows).sort_values("date")
    df.to_csv(PROC / "garmin_daily.csv", index=False)
    return df


def tidy_race_predictions() -> pd.DataFrame:
    rp = json.loads((GDIR / "race_predictions.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(rp)[["calendarDate", "time5K", "time10K", "timeHalfMarathon", "timeMarathon"]]
    df = df.rename(columns={"calendarDate": "date"}).dropna(subset=["timeHalfMarathon"])
    df.to_csv(PROC / "garmin_race_predictions.csv", index=False)
    return df


def main() -> None:
    try:
        fetch(connect())
    except KeyError as e:
        raise SystemExit(f"Missing {e} in .env") from e
    df = tidy()
    rp = tidy_race_predictions()
    cols = ["sleep_h", "sleep_score", "hrv_overnight_ms", "resting_hr", "body_battery_wake", "vo2max"]
    print(f"Garmin daily rows: {len(df)} ({df.date.min()} -> {df.date.max()})")
    print("Non-null counts:", df[cols].notna().sum().to_dict())
    print(df[["date"] + cols].tail(8).to_string(index=False))
    print(f"Garmin race predictor: {len(rp)} days; last pre-race HM prediction "
          f"{rp.iloc[-1].date} = {rp.iloc[-1].timeHalfMarathon:.0f} s")


if __name__ == "__main__":
    main()
