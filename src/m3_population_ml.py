"""M3 - Population models (transfer from other runners) + Bayesian personal calibration.

Vickers & Vertosick (2016): ~1,570 recreational runners with a half-marathon time plus a 5k
and/or 10k, weekly mileage, age, sex, BMI, training habits and self-rated fitness.
  -> learn log(HM time) from log(shorter race time) + training/profile features.

BMClab 2019 (36k athletes, daily logs): weak labels = each athlete's fastest ~21.1 km day,
features from the 12 weeks before it (volume, longest run, fastest 5-10 km day pace).

Calibration: population prediction = prior, personalised Riegel = likelihood (log-normal).
"""
from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import CFG, OUT, PROC, RACE_DATE, RACE_KM, RAW, fmt_hms, save_predictions
from .features import best_efforts_table
from .m1_formulas import fit_personal_riegel

VICKERS_FEATS = ["log_short", "log_typical", "log_max", "age", "bmi", "female", "sprint", "tempo",
                 "hm_fitness", "hm_difficulty"]
Z90 = 1.645


def vickers_xy(v: pd.DataFrame, short_col: str):
    d = v[v[short_col].notna()].copy()
    d["log_short"] = np.log(d[short_col])
    d["log_typical"] = np.log(d.typical_km_week)
    d["log_max"] = np.log(d.max_km_week.clip(lower=d.typical_km_week))
    d = d.dropna(subset=VICKERS_FEATS)
    return d[VICKERS_FEATS], np.log(d.hm_s), d


def models():
    return {
        "linear": lambda: make_pipeline(StandardScaler(), LinearRegression()),
        "GBM": lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300,
                                                     min_samples_leaf=20, random_state=0),
    }


def abbass_profile(runs: pd.DataFrame, efforts: pd.DataFrame, v: pd.DataFrame) -> dict:
    r = CFG["runner"]
    last84 = runs[runs.days_before_race <= 84]
    wk = last84.assign(week=pd.to_datetime(last84.date).dt.to_period("W-SUN")).groupby("week").distance_m.sum() / 1000
    season = best_efforts_table(efforts).set_index("km")
    bmi = (r["weight_kg"] / (r["height_cm"] / 100) ** 2) if r.get("height_cm") else float(v.bmi.median())
    return {
        "k5_s": float(season.loc[5, "time_s"]),
        "k10_s": float(season.loc[10, "time_s"]),
        "typical_km_week": float(last84.distance_m.sum() / 1000 / 12),
        "max_km_week": float(wk.max()),
        "age": float(r["age"]) if r.get("age") else float(v.age.median()),
        "bmi": bmi,
        "female": 0.0 if r["sex"] == "M" else 1.0,
        # No structured interval or tempo sessions in the log -> 0.
        "sprint": 0.0, "tempo": 0.0,
        "hm_fitness": float(r["fitness_rating"]),
        "hm_difficulty": 3.0,  # flat urban course -> "average"
        "age_is_imputed": not r.get("age"), "bmi_is_imputed": not r.get("height_cm"),
    }


def abbass_x(p: dict, short_col: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "log_short": np.log(p[short_col]), "log_typical": np.log(p["typical_km_week"]),
        "log_max": np.log(max(p["max_km_week"], p["typical_km_week"])), "age": p["age"], "bmi": p["bmi"],
        "female": p["female"], "sprint": p["sprint"], "tempo": p["tempo"],
        "hm_fitness": p["hm_fitness"], "hm_difficulty": p["hm_difficulty"]}])[VICKERS_FEATS]


# ---------------------------------------------------------------------------------------
def bmclab(p_recent: dict) -> dict:
    """Weak-label population model + where Abbass's volume sits among 21 km finishers."""
    cache = PROC / "bmclab_hm_samples.csv"
    if not cache.exists():
        b = pd.read_parquet(RAW / "run_ww_2019_d.parquet", columns=["datetime", "athlete", "distance", "duration"])
        b = b[b.distance > 0]
        b["pace"] = b.duration * 60 / b.distance  # s/km
        hm = b[b.distance.between(21.0, 21.5) & b.pace.between(180, 600)]
        best = hm.loc[hm.groupby("athlete").pace.idxmin()]  # fastest ~HM day per athlete
        best = best[best.datetime >= "2019-04-01"]  # need a full 12-week history before it
        prior = b.merge(best[["athlete", "datetime"]].rename(columns={"datetime": "hm_day"}), on="athlete")
        prior = prior[(prior.datetime < prior.hm_day) & (prior.datetime >= prior.hm_day - pd.Timedelta(days=84))]
        mid = prior[prior.distance.between(5, 10.5) & prior.pace.between(150, 600)]
        feats = pd.DataFrame({
            "km_per_week_12w": prior.groupby("athlete").distance.sum() / 12,
            "longest_12w": prior.groupby("athlete").distance.max(),
            "best_5_10k_pace": mid.groupby("athlete").pace.min(),
        })
        samples = best.set_index("athlete")[["pace"]].rename(columns={"pace": "hm_pace"}).join(feats, how="inner").dropna()
        samples.to_csv(cache)
    s = pd.read_csv(cache, index_col=0)
    X, y = s[["km_per_week_12w", "longest_12w", "best_5_10k_pace"]], s.hm_pace
    gbm = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=50,
                                        random_state=0)
    cv = cross_val_predict(gbm, X, y, cv=KFold(5, shuffle=True, random_state=0))
    gbm.fit(X, y)
    xa = pd.DataFrame([p_recent])[X.columns]
    pace = float(gbm.predict(xa)[0])
    return {
        "n_athletes": int(len(s)),
        "cv_mae_s": float(np.mean(np.abs(cv - y)) * RACE_KM),
        "pred_s": pace * RACE_KM,
        "volume_percentile": float((s.km_per_week_12w < p_recent["km_per_week_12w"]).mean() * 100),
        "longest_percentile": float((s.longest_12w < p_recent["longest_12w"]).mean() * 100),
        "median_km_per_week": float(s.km_per_week_12w.median()),
        "abbass": p_recent,
    }


def main() -> None:
    v = pd.read_csv(PROC / "vickers.csv")
    runs = pd.read_csv(PROC / "run_features.csv")
    efforts = pd.read_csv(PROC / "best_efforts.csv")
    prof = abbass_profile(runs, efforts, v)
    rows, report = [], {"abbass_features": prof}

    # ---- Vickers population models ----------------------------------------------------
    pop_preds, pop_log_sd = [], []
    for short_col, label in (("k5_s", "5k"), ("k10_s", "10k")):
        X, y, d = vickers_xy(v, short_col)
        xa = abbass_x(prof, short_col)
        for mname, fn in models().items():
            cv = cross_val_predict(fn(), X, y, cv=KFold(5, shuffle=True, random_state=0))
            mae_s = float(np.mean(np.abs(np.exp(cv) - np.exp(y))))
            resid_sd = float(np.std(y - cv))
            low = d.typical_km_week < 25
            mae_low = float(np.mean(np.abs(np.exp(cv[low.values]) - np.exp(y[low]))))
            m = fn().fit(X, y)
            pred = float(np.exp(m.predict(xa)[0]))
            lo, hi = pred * np.exp(-Z90 * resid_sd), pred * np.exp(Z90 * resid_sd)
            report[f"vickers_{label}_{mname}"] = {"n": int(len(y)), "cv_mae_s": round(mae_s),
                                                  "cv_mae_s_low_volume_runners": round(mae_low),
                                                  "n_low_volume": int(low.sum()), "pred": fmt_hms(pred)}
            if mname == "linear":
                coefs = dict(zip(VICKERS_FEATS, np.round(m[-1].coef_ / m[0].scale_, 4)))
                report[f"vickers_{label}_{mname}"]["coef_per_unit_log_time"] = coefs
                pop_preds.append(np.log(pred)); pop_log_sd.append(resid_sd)
            rows.append({"method": f"Population {mname} from {label} (Vickers, n={len(y)})",
                         "predicted_s": round(pred), "low_s": round(lo), "high_s": round(hi),
                         "notes": f"5-fold CV MAE {mae_s / 60:.1f} min on population; "
                                  f"your season-best {label} {fmt_hms(prof[short_col])}, {prof['typical_km_week']:.1f} km/wk"})

    # ---- Bayesian calibration: population prior x personal likelihood ------------------
    prior_mu, prior_sd = float(np.mean(pop_preds)), float(np.mean(pop_log_sd))
    p_r, lo_r, hi_r, b, _ = fit_personal_riegel(efforts)
    like_mu, like_sd = np.log(p_r), (np.log(hi_r) - np.log(lo_r)) / (2 * Z90)
    w_prior, w_like = 1 / prior_sd ** 2, 1 / like_sd ** 2
    post_mu = (w_prior * prior_mu + w_like * like_mu) / (w_prior + w_like)
    post_sd = (w_prior + w_like) ** -0.5
    post = float(np.exp(post_mu))
    report["bayes"] = {"prior": fmt_hms(np.exp(prior_mu)), "prior_log_sd": round(prior_sd, 4),
                       "likelihood": fmt_hms(p_r), "likelihood_log_sd": round(like_sd, 4),
                       "weight_on_personal": round(w_like / (w_prior + w_like), 3), "posterior": fmt_hms(post)}
    rows.append({"method": "Population prior x personal curve (Bayesian)", "predicted_s": round(post),
                 "low_s": round(np.exp(post_mu - Z90 * post_sd)), "high_s": round(np.exp(post_mu + Z90 * post_sd)),
                 "notes": f"prior = mean of Vickers linear models, likelihood = personalised Riegel; "
                          f"{w_like / (w_prior + w_like):.0%} weight on personal data"})

    # ---- BMClab weak-label model -------------------------------------------------------
    last84 = runs[runs.days_before_race <= 84]
    mid = last84[last84.distance_m.between(5000, 10500)]
    p_recent = {"km_per_week_12w": last84.distance_m.sum() / 1000 / 12,
                "longest_12w": last84.distance_m.max() / 1000,
                "best_5_10k_pace": float(mid.pace_s.min())}
    bm = bmclab(p_recent)
    report["bmclab"] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in bm.items()}
    rows.append({"method": f"BMClab weak-label GBM (n={bm['n_athletes']:,})", "predicted_s": round(bm["pred_s"]),
                 "notes": f"labels = fastest ~21 km day per athlete (2019); CV MAE {bm['cv_mae_s'] / 60:.1f} min; "
                          f"your 12-wk volume is at the {bm['volume_percentile']:.0f}th percentile"})

    (OUT / "m3_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    save_predictions(rows, "M3 Population ML")
    print(json.dumps(report, indent=1, default=float))
    for r in rows:
        band = f" [{fmt_hms(r['low_s'])} - {fmt_hms(r['high_s'])}]" if r.get("low_s") else ""
        print(f"{r['method']:<50} {fmt_hms(r['predicted_s'])}{band}")


if __name__ == "__main__":
    main()
