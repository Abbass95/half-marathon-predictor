"""M2 - Personal ML on per-km splits, then simulate the 21.1 km race.

Two framings:
  M2a "training-load model": all 41 tracked runs (phone + Garmin), no heart rate.
      Learns how *training* pace varies with distance, fatigue, terrain, heat and fitness.
  M2b "effort-aware model": Garmin runs only, adds heart rate so pace can be simulated at
      race effort (the hardest average HR sustained in any pre-race run >= 5 km).

Validation: leave-one-run-out (GroupKFold) + a time split (train < Aug, test Aug-Sept).
Also: does sleep / HRV explain run-to-run pace residuals? (Garmin period only.)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import OUT, PROC, RACE_KM, fmt_hms, fmt_pace, save_predictions

BASE = ["km", "run_km", "elev_gain_m", "days_before_race", "km_last_84d", "longest_to_date_km", "temp_c"]
WELL = ["sleep_h", "sleep_score", "hrv_overnight_ms", "resting_hr", "body_battery_wake"]
# Candidate feature sets for the effort-aware model; the winner is picked by LORO MAE only
# (pre-race data), never by closeness to the race result.
M2B_CANDIDATES = {
    "full": BASE + ["hr"],
    "physio+trend": ["km", "elev_gain_m", "temp_c", "hr", "days_before_race"],
    "physio": ["km", "elev_gain_m", "temp_c", "hr"],
    "hr+km": ["km", "hr"],
    "hr only": ["hr"],
}


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         RidgeCV(alphas=np.logspace(-2, 3, 30)))


def gbm(loss="squared_error", q=None):
    kw = {"quantile": q} if q is not None else {}
    return HistGradientBoostingRegressor(loss=loss, max_depth=3, learning_rate=0.05, max_iter=250,
                                         min_samples_leaf=10, random_state=0, **kw)


def cv_mae(model_fn, X, y, groups) -> float:
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=groups.nunique()).split(X, y, groups):
        m = model_fn().fit(X.iloc[tr], y.iloc[tr])
        pred[te] = m.predict(X.iloc[te])
    return mean_absolute_error(y, pred)


def time_split_mae(model_fn, X, y, dates) -> float:
    tr = pd.to_datetime(dates) < "2026-08-01"
    m = model_fn().fit(X[tr], y[tr])
    return mean_absolute_error(y[~tr], m.predict(X[~tr]))


def race_rows(ctx: dict, runs: pd.DataFrame, cols: list[str], hr: float | None = None) -> pd.DataFrame:
    """21 synthetic km rows describing race day (all values knowable before the start)."""
    n = int(np.ceil(RACE_KM))
    last84 = runs[runs.days_before_race <= 84].distance_m.sum() / 1000
    row = {
        "run_km": RACE_KM,
        "elev_gain_m": ctx["race"]["course_elev_gain_m"] / RACE_KM,
        "days_before_race": 0,
        "km_last_84d": last84,
        "longest_to_date_km": runs.distance_m.max() / 1000,
        "temp_c": ctx["forecast_issued_day_before"]["temp_c"],
        **ctx["race_morning_wellness"],
    }
    if hr is not None:
        row["hr"] = hr
    df = pd.DataFrame([{**row, "km": k} for k in range(1, n + 1)])
    return df[cols]


def total_time(split_paces: np.ndarray) -> float:
    """Sum 21 full km + the last 97.5 m at the final km's pace."""
    frac = RACE_KM - np.floor(RACE_KM)
    return float(split_paces[:-1].sum() + split_paces[-1] * frac)


def bootstrap_interval(model_fn, X, y, groups, Xr, n=300, seed=0):
    rng = np.random.default_rng(seed)
    ids = groups.unique()
    out = []
    for _ in range(n):
        pick = rng.choice(ids, len(ids), replace=True)
        idx = np.concatenate([np.flatnonzero(groups.values == g) for g in pick])
        m = model_fn().fit(X.iloc[idx], y.iloc[idx])
        out.append(total_time(m.predict(Xr)))
    return np.percentile(out, [5, 95])


def wellness_analysis(g: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Leave-one-run-out residuals of the model *without* wellness vs previous-night wellness."""
    X, y, grp = g[cols], g.pace_s, g.run_id
    resid = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=grp.nunique()).split(X, y, grp):
        resid[te] = y.iloc[te] - ridge().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
    per_run = g.assign(resid=resid).groupby("run_id").agg(resid=("resid", "mean"),
                                                          **{w: (w, "first") for w in WELL})
    rows = []
    for w in WELL:
        d = per_run[["resid", w]].dropna()
        rho, p = stats.spearmanr(d[w], d.resid)
        rows.append({"metric": w, "n_runs": len(d), "spearman_rho": round(rho, 3), "p_value": round(p, 3)})
    return pd.DataFrame(rows)


def main() -> None:
    splits = pd.read_csv(PROC / "splits.csv")
    runs = pd.read_csv(PROC / "run_features.csv")
    ctx = json.loads((PROC / "race_context.json").read_text(encoding="utf-8"))
    rows, report, pacing = [], {}, {}

    # ---------------- M2a: all runs, no HR -------------------------------------------
    X, y, grp = splits[BASE], splits.pace_s, splits.run_id
    Xr = race_rows(ctx, runs, BASE)
    base_mae = mean_absolute_error(y, np.full(len(y), y.mean()))
    report["M2a"] = {"n_splits": len(y), "n_runs": grp.nunique(),
                     "baseline_mean_mae": round(base_mae, 1)}
    for name, fn in (("Ridge", ridge), ("GBM", gbm)):
        mae, tmae = cv_mae(fn, X, y, grp), time_split_mae(fn, X, y, splits.date)
        m = fn().fit(X, y)
        paces = m.predict(Xr)
        pred = total_time(paces)
        pacing[f"M2a {name}"] = paces
        lo = hi = None
        if name == "Ridge":
            lo, hi = bootstrap_interval(fn, X, y, grp, Xr)
        report["M2a"][name] = {"loro_mae_s_per_km": round(mae, 1), "time_split_mae": round(tmae, 1),
                               "pred": fmt_hms(pred)}
        rows.append({"method": f"Training-load {name} (all runs, no HR)", "predicted_s": round(pred),
                     "low_s": lo and round(lo), "high_s": hi and round(hi),
                     "notes": f"LORO MAE {mae:.0f} s/km (mean baseline {base_mae:.0f}); learns *training* pace"})

    # ---------------- M2b: Garmin runs with HR ----------------------------------------
    g = splits[splits.hr.notna()].copy()
    selection = {name: round(cv_mae(ridge, g[c], g.pace_s, g.run_id), 1)
                 for name, c in M2B_CANDIDATES.items()}
    best_set = min(selection, key=selection.get)
    cols = M2B_CANDIDATES[best_set]
    garmin_runs = runs[runs.source == "garmin_fit"]
    race_hr = float(garmin_runs[garmin_runs.distance_m >= 5000].hr_avg.max())
    Xg, yg, gg = g[cols], g.pace_s, g.run_id
    Xrg = race_rows(ctx, runs, cols, hr=race_hr)
    report["M2b"] = {"feature_set_loro_mae": selection, "chosen_feature_set": best_set,
                     "n_splits": len(yg), "n_runs": gg.nunique(), "race_effort_hr": race_hr,
                     "baseline_mean_mae": round(mean_absolute_error(yg, np.full(len(yg), yg.mean())), 1)}
    for name, fn in (("Ridge", ridge), ("GBM", gbm)):
        mae = cv_mae(fn, Xg, yg, gg)
        m = fn().fit(Xg, yg)
        paces = m.predict(Xrg)
        pred = total_time(paces)
        pacing[f"M2b {name}"] = paces
        lo = hi = None
        if name == "Ridge":
            lo, hi = bootstrap_interval(fn, Xg, yg, gg, Xrg)
            coef = m[-1].coef_ / m[1].scale_  # s/km per raw unit
            report["M2b"]["ridge_coef_s_per_unit"] = dict(zip(cols, np.round(coef, 2)))
        report["M2b"][name] = {"loro_mae_s_per_km": round(mae, 1), "pred": fmt_hms(pred)}
        rows.append({"method": f"Effort-aware {name} (HR {race_hr:.0f})", "predicted_s": round(pred),
                     "low_s": lo and round(lo), "high_s": hi and round(hi),
                     "notes": f"Garmin runs only; features '{best_set}' chosen by LORO MAE; "
                              f"simulated at hardest pre-race avg HR; LORO MAE {mae:.0f} s/km"})

    # Quantile GBM band for the effort-aware model
    q = {a: total_time(gbm("quantile", a).fit(Xg, yg).predict(Xrg)) for a in (0.1, 0.9)}
    report["M2b"]["GBM_quantile_10_90"] = [fmt_hms(q[0.1]), fmt_hms(q[0.9])]
    rows[-1]["low_s"], rows[-1]["high_s"] = round(q[0.1]), round(q[0.9])

    # ---------------- Wellness: does sleep/HRV move my pace? ---------------------------
    well = wellness_analysis(g, BASE + ["hr"])
    well.to_csv(OUT / "m2_wellness_effect.csv", index=False)
    well_cols = BASE + ["hr"]
    gw = g.dropna(subset=["sleep_h"])
    m = gbm().fit(gw[well_cols + WELL], gw.pace_s)
    pi = permutation_importance(m, gw[well_cols + WELL], gw.pace_s, n_repeats=20, random_state=0)
    imp = pd.Series(pi.importances_mean, index=well_cols + WELL).sort_values(ascending=False)
    imp.round(2).to_csv(OUT / "m2_permutation_importance.csv", header=["importance"])

    pd.DataFrame(pacing, index=range(1, len(Xr) + 1)).rename_axis("km").to_csv(OUT / "m2_pacing.csv")
    (OUT / "m2_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    save_predictions(rows, "M2 Personal ML")

    print(json.dumps(report, indent=1, default=float))
    print("\nWellness vs pace residual (negative rho = better wellness -> faster):")
    print(well.to_string(index=False))
    print("\nPermutation importance (GBM with wellness, in-sample):")
    print(imp.round(2).to_string())
    print("\nPredicted pacing (s/km):")
    print(pd.DataFrame(pacing).round(0).T.to_string())
    for r in rows:
        band = f" [{fmt_hms(r['low_s'])} - {fmt_hms(r['high_s'])}]" if r["low_s"] else ""
        print(f"{r['method']:<42} {fmt_hms(r['predicted_s'])}{band}")


if __name__ == "__main__":
    main()
