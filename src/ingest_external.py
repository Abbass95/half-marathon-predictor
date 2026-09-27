"""External data: Vickers & Vertosick (2016) survey, BMClab 2019 training logs, Open-Meteo weather.

Outputs
  data/processed/vickers.csv          one row per runner with a half-marathon time (+ shorter race)
  data/raw/run_ww_2019_d.parquet      BMClab daily training log (downloaded once, 155 MB)
  data/processed/weather_hourly.csv   hourly observed weather for the training period
  data/processed/race_forecast.csv    race-morning forecast as issued the day before (no leakage)
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import requests

from .common import CFG, PROC, RACE_DATE, RAW, ROOT

MILE_KM = 1.609344
BMCLAB_URL = "https://ndownloader.figshare.com/files/33801866"  # run_ww_2019_d.parquet
BMCLAB_FILE = RAW / "run_ww_2019_d.parquet"
LAT, LON, TZ = CFG["race"]["lat"], CFG["race"]["lon"], CFG["race"]["timezone"]
HOURLY = "temperature_2m,apparent_temperature,relative_humidity_2m,dew_point_2m,wind_speed_10m"


def load_vickers() -> pd.DataFrame:
    """Keep runners with a half-marathon time; express everything in metric units.

    Uses raw times (not difficulty-adjusted) plus the course-difficulty rating as a feature,
    because Abbass's race time is raw too.
    """
    v = pd.read_excel(ROOT / CFG["paths"]["vickers_xlsx"], sheet_name="Sheet1")
    v = v[v["mh_ti"].notna()].copy()
    out = pd.DataFrame({
        "id": v["id"],
        "age": v["age"],
        "bmi": v["bmi"],
        "female": v["female"],
        "typical_km_week": v["typical"] * MILE_KM,
        "max_km_week": v["max"] * MILE_KM,
        "sprint": v["sprint"],
        "tempo": v["tempo"],
        "hm_s": v["mh_ti"],
        "hm_difficulty": v["mh_di"],
        "hm_fitness": v["mh_tr"],
        "k5_s": v["k5_ti"],
        "k10_s": v["k10_ti"],
        "m10_s": v["m10_ti"],  # 10 miles
        "group": v["group"],
    })
    # Plausibility filter: HM between 1:05 and 3:30, reported mileage within survey range.
    out = out[(out.hm_s.between(3900, 12600)) & (out.typical_km_week > 0)]
    out.to_csv(PROC / "vickers.csv", index=False)
    return out


def download_bmclab() -> None:
    if BMCLAB_FILE.exists():
        return
    print("Downloading BMClab 2019 daily training log (~155 MB)...")
    with requests.get(BMCLAB_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = BMCLAB_FILE.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.rename(BMCLAB_FILE)


def fetch_weather() -> pd.DataFrame:
    runs = pd.read_csv(PROC / "runs.csv", parse_dates=["date"])
    start = runs.date.min().date()
    end = RACE_DATE - timedelta(1)
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": LAT, "longitude": LON, "start_date": start.isoformat(),
        "end_date": end.isoformat(), "hourly": HOURLY, "timezone": TZ}, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"]).rename(columns={"time": "time_local"})
    df.to_csv(PROC / "weather_hourly.csv", index=False)
    return df


def fetch_race_forecast() -> pd.DataFrame:
    """Race-morning forecast as it looked on the eve of the race (Previous Runs API, day-1 lead)."""
    vars_ = HOURLY.split(",")
    r = requests.get("https://previous-runs-api.open-meteo.com/v1/forecast", params={
        "latitude": LAT, "longitude": LON, "start_date": RACE_DATE.isoformat(),
        "end_date": RACE_DATE.isoformat(), "timezone": TZ,
        "hourly": ",".join(f"{v}_previous_day1" for v in vars_)}, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"]).rename(columns={"time": "time_local"})
    df.columns = [c.replace("_previous_day1", "") for c in df.columns]
    df.to_csv(PROC / "race_forecast.csv", index=False)
    return df


def main() -> None:
    v = load_vickers()
    print(f"Vickers: {len(v)} runners with a half-marathon; "
          f"with 10k: {v.k10_s.notna().sum()}, with 5k: {v.k5_s.notna().sum()}, "
          f"with 10mi: {v.m10_s.notna().sum()}")
    print(f"  HM median {v.hm_s.median() / 60:.0f} min, typical weekly km median {v.typical_km_week.median():.0f}")

    download_bmclab()
    b = pd.read_parquet(BMCLAB_FILE)
    print(f"BMClab 2019 daily: {len(b):,} rows, {b.athlete.nunique():,} athletes, columns {list(b.columns)}")

    w = fetch_weather()
    print(f"Weather hourly: {len(w)} rows ({w.time_local.iloc[0]} -> {w.time_local.iloc[-1]})")
    f = fetch_race_forecast()
    morning = f[f.time_local.str[11:13].isin(["10", "11", "12"])]
    print("Race-morning forecast (issued day before):")
    print(morning.to_string(index=False))


if __name__ == "__main__":
    main()
