"""Parse the Strava bulk export: activities.csv + per-activity FIT/GPX tracks.

Outputs
  data/processed/runs.csv              one row per run (race included, flagged is_race)
  data/processed/streams/<id>.parquet  1 Hz-ish stream per run: t (s), dist_m, hr, cad, alt, speed
"""
from __future__ import annotations

import gzip
import io
import math
import re

import gpxpy
import numpy as np
import pandas as pd
from fitparse import FitFile

from .common import PROC, RACE_DATE, RACE_ID, STRAVA_DIR

FR_MONTHS = {"janv.": 1, "févr.": 2, "mars": 3, "avr.": 4, "mai": 5, "juin": 6,
             "juil.": 7, "août": 8, "sept.": 9, "oct.": 10, "nov.": 11, "déc.": 12}

STREAMS = PROC / "streams"
STREAMS.mkdir(parents=True, exist_ok=True)


def parse_fr_date(s: str) -> pd.Timestamp:
    """'20 sept. 2026, 08:18:47' (Strava export, UTC) -> tz-aware UTC timestamp."""
    s = s.replace(" ", " ").replace("\xa0", " ")
    m = re.match(r"(\d{1,2}) (\S+) (\d{4}), (\d{2}):(\d{2}):(\d{2})", s)
    if not m:
        raise ValueError(f"Unparseable date: {s!r}")
    d, mon, y, hh, mm, ss = m.groups()
    return pd.Timestamp(int(y), FR_MONTHS[mon], int(d), int(hh), int(mm), int(ss), tz="UTC")


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def read_fit(path) -> pd.DataFrame:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    rows = []
    for rec in FitFile(io.BytesIO(raw)).get_messages("record"):
        v = {f.name: f.value for f in rec}
        rows.append({
            "time": v.get("timestamp"),
            "dist_m": v.get("distance"),
            "hr": v.get("heart_rate"),
            "cad": v.get("cadence"),
            "alt": v.get("enhanced_altitude", v.get("altitude")),
            "speed": v.get("enhanced_speed", v.get("speed")),
        })
    df = pd.DataFrame(rows).dropna(subset=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def read_gpx(path) -> pd.DataFrame:
    g = gpxpy.parse(path.read_text(encoding="utf-8"))
    pts = [(p.time, p.latitude, p.longitude, p.elevation)
           for t in g.tracks for s in t.segments for p in s.points]
    df = pd.DataFrame(pts, columns=["time", "lat", "lon", "alt"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    # Drop GPS jumps: points implying > 6 m/s (2:47/km) from the previous kept point.
    while True:
        step = haversine_m(df.lat.shift(), df.lon.shift(), df.lat, df.lon)
        v = step / df.time.diff().dt.total_seconds().clip(lower=1)
        bad = v > 6.0
        if not bad.any():
            break
        df = df[~bad].reset_index(drop=True)
    step = haversine_m(df.lat.shift(), df.lon.shift(), df.lat, df.lon).fillna(0)
    df["dist_m"] = step.cumsum()
    df["hr"] = np.nan
    df["cad"] = np.nan
    dt = df.time.diff().dt.total_seconds()
    df["speed"] = (step / dt.replace(0, np.nan))
    return df.drop(columns=["lat", "lon"])


def normalise_stream(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("time").reset_index(drop=True)
    df["dist_m"] = df["dist_m"].ffill().fillna(0).cummax()
    df["t"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
    # Moving clock: only count intervals where the runner actually moved (> 1 m/s),
    # so traffic lights / auto-pause gaps don't pollute splits and best efforts.
    dt = df["t"].diff().fillna(0)
    step_speed = df["dist_m"].diff().fillna(0) / dt.replace(0, np.nan)
    df["t_moving"] = (dt * (step_speed > 1.0)).cumsum()
    return df[["time", "t", "t_moving", "dist_m", "hr", "cad", "alt", "speed"]]


def load_activities() -> pd.DataFrame:
    a = pd.read_csv(STRAVA_DIR / "activities.csv", encoding="utf-8")
    # Duplicate French headers are mangled by pandas as 'Name', 'Name.1'.
    runs = a[a["Type d'activité"] == "Course à pied"].copy()
    out = pd.DataFrame({
        "id": runs["ID de l'activité"].astype("int64"),
        "start_utc": runs["Date de l'activité"].map(parse_fr_date),
        "name": runs["Nom de l'activité"],
        "elapsed_s": runs["Temps écoulé.1"].astype(float),
        "moving_s": runs["Durée de déplacement"].astype(float),
        "distance_m": runs["Distance.1"].astype(float),
        "elev_gain_m": runs["Dénivelé positif"].astype(float),
        "hr_avg": runs["Fréquence cardiaque moyenne"],
        "hr_max": runs["Fréquence cardiaque max..1"],
        "cadence_avg": runs["Cadence moyenne"],
        "calories": runs["Calories"],
        "file": runs["Nom du fichier"],
    })
    out["date"] = out["start_utc"].dt.tz_convert("Europe/Paris").dt.date
    out["is_race"] = out["id"] == RACE_ID
    out["source"] = np.select(
        [out["file"].isna(), out["file"].str.contains(r"\.fit", na=False)],
        ["manual", "garmin_fit"], "phone_gpx")
    return out.sort_values("start_utc").reset_index(drop=True)


def main() -> None:
    runs = load_activities()
    stats = []
    for r in runs.itertuples():
        if r.source == "manual":  # manually logged run: summary only, no track
            continue
        path = STRAVA_DIR / r.file
        df = read_fit(path) if ".fit" in path.name else read_gpx(path)
        df = normalise_stream(df)
        if r.source == "phone_gpx":  # raw phone GPS overestimates; trust Strava's cleaned total
            df["dist_m"] *= r.distance_m / df["dist_m"].iloc[-1]
            df = normalise_stream(df)
        df.to_parquet(STREAMS / f"{r.id}.parquet", index=False)
        stats.append({"id": r.id, "stream_rows": len(df),
                      "stream_km": df.dist_m.iloc[-1] / 1000,
                      "stream_s": df.t.iloc[-1],
                      "stream_moving_s": df.t_moving.iloc[-1],
                      "stream_start_utc": df.time.iloc[0]})
    runs = runs.merge(pd.DataFrame(stats), on="id", how="left")
    runs.to_csv(PROC / "runs.csv", index=False)

    pre = runs[runs.date < RACE_DATE]
    race = runs[runs.is_race].iloc[0]
    print(f"Runs parsed: {len(runs)}  (pre-race: {len(pre)}, race: {int(runs.is_race.sum())})")
    print(f"Sources: {runs.source.value_counts().to_dict()}")
    print(f"Pre-race span: {pre.date.min()} -> {pre.date.max()}, "
          f"total {pre.distance_m.sum() / 1000:.0f} km, longest {pre.distance_m.max() / 1000:.1f} km")
    print(f"Race: {race['name']} | CSV {race.distance_m / 1000:.2f} km / {race.elapsed_s:.0f} s | "
          f"stream {race.stream_km:.2f} km / {race.stream_s:.0f} s | start UTC csv {race.start_utc} vs fit {race.stream_start_utc}")
    gap = (runs.stream_km - runs.distance_m / 1000).abs()
    print(f"Stream vs CSV distance mismatch > 3%: {int((gap / (runs.distance_m / 1000) > 0.03).sum())} runs")


if __name__ == "__main__":
    main()
