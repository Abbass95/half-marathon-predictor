"""Feature engineering on pre-race data only.

Outputs (data/processed/)
  run_features.csv      one row per pre-race run: pace, HR, load, weather, previous-night wellness
  splits.csv            one row per full km of every pre-race run (~330 rows)
  best_efforts.csv      fastest continuous 1/3/5/10/15 km inside each pre-race run
  race_context.json     what was knowable on race morning (forecast, sleep, HRV, VO2max, course)
  training_summary.json compact log used by the LLM / hybrid methods
  race_splits.csv       per-km splits of the race itself (evaluation only, never a feature)
"""
from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd

from .common import CFG, PROC, RACE_DATE, RACE_KM, assert_pre_race, fmt_hms, fmt_pace

STREAMS = PROC / "streams"
EFFORT_KM = [1, 3, 5, 10, 15]
TZ = CFG["race"]["timezone"]
WELLNESS = ["sleep_h", "sleep_score", "hrv_overnight_ms", "resting_hr", "body_battery_wake"]


def load_runs() -> tuple[pd.DataFrame, pd.Series]:
    runs = pd.read_csv(PROC / "runs.csv", parse_dates=["start_utc"])
    runs["date"] = pd.to_datetime(runs["date"]).dt.date
    race = runs[runs.is_race].iloc[0]
    pre = runs[runs.date < RACE_DATE].copy()
    return pre, race


def km_splits(stream: pd.DataFrame) -> pd.DataFrame:
    """Moving time/HR/elevation for each complete kilometre."""
    d, t = stream.dist_m.to_numpy(), stream.t_moving.to_numpy()
    n_km = int(d[-1] // 1000)
    if n_km == 0:
        return pd.DataFrame()
    marks = np.arange(0, n_km + 1) * 1000.0
    t_at = np.interp(marks, d, t)
    alt = stream.alt.rolling(15, center=True, min_periods=1).mean().to_numpy()  # smooth GPS/baro noise
    rows = []
    for k in range(n_km):
        m = (d >= marks[k]) & (d < marks[k + 1])
        seg_alt = alt[m]
        gain = np.clip(np.diff(seg_alt), 0, None).sum() if m.sum() > 1 and not np.isnan(seg_alt).all() else np.nan
        hr = stream.hr.to_numpy()[m]
        rows.append({"km": k + 1, "pace_s": t_at[k + 1] - t_at[k], "elev_gain_m": gain,
                     "hr": np.nanmean(hr) if m.any() and not np.isnan(hr.astype(float)).all() else np.nan})
    return pd.DataFrame(rows)


def best_effort(stream: pd.DataFrame, km: float) -> float | None:
    """Fastest moving time over any continuous `km` window (two-pointer via interpolation)."""
    d, t = stream.dist_m.to_numpy(), stream.t_moving.to_numpy()
    target = km * 1000
    if d[-1] < target * 0.995:
        return None
    ends = d + target
    ok = ends <= d[-1]
    t_end = np.interp(ends[ok], d, t)
    return float((t_end - t[ok]).min())


def weather_at(start_utc: pd.Timestamp, hourly: pd.DataFrame) -> dict:
    local = start_utc.tz_convert(TZ).tz_localize(None).floor("h")
    row = hourly.loc[hourly.time_local == local]
    if row.empty:
        return {"temp_c": np.nan, "humidity": np.nan}
    r = row.iloc[0]
    return {"temp_c": r.temperature_2m, "humidity": r.relative_humidity_2m}


def build() -> dict:
    pre, race = load_runs()
    garmin = pd.read_csv(PROC / "garmin_daily.csv", parse_dates=["date", "night_of"])
    hourly = pd.read_csv(PROC / "weather_hourly.csv", parse_dates=["time_local"])

    # ---- per-run features -------------------------------------------------------------
    pre = pre.sort_values("start_utc").reset_index(drop=True)
    pre["pace_s"] = pre.moving_s / (pre.distance_m / 1000)
    pre["days_before_race"] = [(RACE_DATE - d).days for d in pre.date]
    day_km = pre.groupby("date").distance_m.sum() / 1000
    for w in (28, 84):
        pre[f"km_last_{w}d"] = [
            day_km[(day_km.index < d) & (day_km.index >= d - timedelta(w))].sum() for d in pre.date]
    pre["longest_to_date_km"] = pre.distance_m.cummax().shift().fillna(0) / 1000
    wx = pd.DataFrame([weather_at(s, hourly) for s in pre.start_utc])
    pre = pd.concat([pre, wx], axis=1)
    # Morning metrics recorded on the run day describe the night before the run.
    g = garmin.set_index(garmin.date.dt.date)[WELLNESS]
    pre = pre.join(g, on="date")
    assert_pre_race(pre, "date", "run_features")
    pre.to_csv(PROC / "run_features.csv", index=False)

    # ---- per-km splits + best efforts ---------------------------------------------------
    split_rows, effort_rows = [], []
    for r in pre[pre.source != "manual"].itertuples():
        s = pd.read_parquet(STREAMS / f"{r.id}.parquet")
        sp = km_splits(s)
        if not sp.empty:
            sp["run_id"], sp["date"], sp["run_km"] = r.id, r.date, r.distance_m / 1000
            split_rows.append(sp)
        for k in EFFORT_KM:
            be = best_effort(s, k)
            if be:
                effort_rows.append({"run_id": r.id, "date": r.date, "km": k, "time_s": be,
                                    "days_before_race": r.days_before_race})
    splits = pd.concat(split_rows, ignore_index=True)
    feat_cols = ["id", "days_before_race", "km_last_28d", "km_last_84d", "longest_to_date_km",
                 "temp_c", "humidity"] + WELLNESS
    splits = splits.merge(pre[feat_cols].rename(columns={"id": "run_id"}), on="run_id")
    # Drop GPS glitches (implausible split paces) before modelling.
    splits = splits[splits.pace_s.between(200, 720)]
    assert_pre_race(splits, "date", "splits")
    splits.to_csv(PROC / "splits.csv", index=False)
    efforts = pd.DataFrame(effort_rows)
    assert_pre_race(efforts, "date", "best_efforts")
    efforts.to_csv(PROC / "best_efforts.csv", index=False)

    # Race splits saved for evaluation charts only (not used by any predictor).
    race_stream = pd.read_parquet(STREAMS / f"{race.id}.parquet")
    km_splits(race_stream).to_csv(PROC / "race_splits.csv", index=False)

    # ---- race-morning context (all knowable before the 10:18 start) ---------------------
    fc = pd.read_csv(PROC / "race_forecast.csv")
    fc_race = fc[fc.time_local.str[11:13].isin(["10", "11", "12"])]
    morning = garmin[garmin.date.dt.date == RACE_DATE]
    vo2 = garmin[(garmin.date.dt.date < RACE_DATE) & garmin.vo2max.notna()].vo2max
    gp = pd.read_csv(PROC / "garmin_race_predictions.csv")
    ctx = {
        "race": {"name": CFG["race"]["name"], "date": str(RACE_DATE), "start_local": "10:18",
                 "distance_km": RACE_KM,
                 # Published course is flat-ish; ~100 m cumulative gain per the organiser's profile.
                 "course_elev_gain_m": 100},
        "forecast_issued_day_before": {
            "temp_c": round(fc_race.temperature_2m.mean(), 1),
            "apparent_temp_c": round(fc_race.apparent_temperature.mean(), 1),
            "humidity_pct": round(fc_race.relative_humidity_2m.mean()),
            "wind_kmh": round(fc_race.wind_speed_10m.mean(), 1)},
        "race_morning_wellness": morning[WELLNESS].iloc[0].round(1).to_dict() if not morning.empty else {},
        "garmin_vo2max_last_pre_race": float(vo2.iloc[-1]) if len(vo2) else None,
        "garmin_race_predictor_hm_s_2026_09_19": int(gp[gp.date < str(RACE_DATE)].iloc[-1].timeHalfMarathon),
    }
    (PROC / "race_context.json").write_text(json.dumps(ctx, indent=2), encoding="utf-8")

    summary = training_summary(pre, efforts, garmin, ctx)
    (PROC / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                                  encoding="utf-8")
    return {"pre": pre, "splits": splits, "efforts": efforts, "ctx": ctx, "summary": summary}


def best_efforts_table(efforts: pd.DataFrame, since_days: int | None = None) -> pd.DataFrame:
    e = efforts if since_days is None else efforts[efforts.days_before_race <= since_days]
    idx = e.groupby("km").time_s.idxmin()
    return e.loc[idx].sort_values("km").reset_index(drop=True)


def training_summary(pre, efforts, garmin, ctx) -> dict:
    wk = pre.assign(week=pd.to_datetime(pre.date).dt.to_period("W-SUN").dt.start_time.dt.date)
    weekly = wk.groupby("week").distance_m.sum().div(1000).round(1)
    all_weeks = pd.period_range(end=pd.Timestamp(RACE_DATE - timedelta(1)), periods=16, freq="W-SUN")
    last16 = {str(p.start_time.date()): float(weekly.get(p.start_time.date(), 0.0)) for p in all_weeks}
    g_last = garmin[(garmin.date.dt.date < RACE_DATE) &
                    (garmin.date.dt.date >= RACE_DATE - timedelta(28))]
    be_all, be_12w = best_efforts_table(efforts), best_efforts_table(efforts, 84)
    return {
        "runner": {"sex": CFG["runner"]["sex"], "age": CFG["runner"]["age"],
                   "weight_kg": CFG["runner"]["weight_kg"], "height_cm": CFG["runner"]["height_cm"],
                   "experience": "first half marathon; running regularly since Sept 2025",
                   "self_rated_fitness": "not well prepared (4 on a 1-4 scale)"},
        "data_cutoff": str(RACE_DATE - timedelta(1)),
        "totals": {
            "runs_last_12_months": int(len(pre)),
            "km_last_12_months": round(pre.distance_m.sum() / 1000, 1),
            "runs_last_12_weeks": int((pre.days_before_race <= 84).sum()),
            "km_last_12_weeks": round(pre[pre.days_before_race <= 84].distance_m.sum() / 1000, 1),
            "avg_km_per_week_last_12_weeks": round(pre[pre.days_before_race <= 84].distance_m.sum() / 1000 / 12, 1),
            "longest_run_ever_km": round(pre.distance_m.max() / 1000, 1),
            "longest_run_last_8_weeks_km": round(pre[pre.days_before_race <= 56].distance_m.max() / 1000, 1),
            "runs_over_10km_last_12_weeks": int(((pre.days_before_race <= 84) & (pre.distance_m > 10000)).sum()),
        },
        "weekly_km_last_16_weeks": last16,
        "best_efforts_all_time": {f"{int(r.km)}km": fmt_hms(r.time_s) + f" ({r.date})" for r in be_all.itertuples()},
        "best_efforts_last_12_weeks": {f"{int(r.km)}km": fmt_hms(r.time_s) + f" ({r.date})" for r in be_12w.itertuples()},
        "runs": [{"date": str(r.date), "km": round(r.distance_m / 1000, 2), "moving_time": fmt_hms(r.moving_s),
                  "pace": fmt_pace(r.pace_s),
                  "avg_hr": None if pd.isna(r.hr_avg) else int(r.hr_avg),
                  "elev_gain_m": None if pd.isna(r.elev_gain_m) else int(r.elev_gain_m),
                  "temp_c": None if pd.isna(r.temp_c) else round(r.temp_c)}
                 for r in pre.itertuples()],
        "wellness_last_4_weeks_avg": g_last[WELLNESS].mean().round(1).to_dict(),
        "race_context": ctx,
    }


def main() -> None:
    f = build()
    pre, splits, efforts, s = f["pre"], f["splits"], f["efforts"], f["summary"]
    print(f"run_features: {len(pre)} runs | splits: {len(splits)} km-rows from {splits.run_id.nunique()} runs "
          f"| best efforts: {len(efforts)} rows")
    print("Leakage checks passed (no feature dated >= race day).")
    print("\nBest efforts (all time):")
    for k, v in s["best_efforts_all_time"].items():
        print(f"  {k:>5}: {v}")
    print("Best efforts (last 12 weeks):")
    for k, v in s["best_efforts_last_12_weeks"].items():
        print(f"  {k:>5}: {v}")
    print("\nTotals:", json.dumps(s["totals"], indent=1))
    print("Race context:", json.dumps(f["ctx"], indent=1))


if __name__ == "__main__":
    main()
