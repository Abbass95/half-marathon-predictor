"""M1 - Classic running formulas + Garmin's commercial predictor (domain baselines)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .common import PROC, RACE_KM, fmt_hms, save_predictions
from .features import best_efforts_table

RIEGEL_EXP = 1.06
FIT_KM = [3, 5, 10, 15]  # 1 km efforts are anaerobic and GPS-noisy -> excluded from curve fits


def riegel(t1: float, d1: float, d2: float = RACE_KM, b: float = RIEGEL_EXP) -> float:
    return t1 * (d2 / d1) ** b


def cameron(t1: float, d1_km: float, d2_km: float = RACE_KM) -> float:
    def a(d_m):
        return 13.49681 - 0.000030363 * d_m + 835.7114 / d_m ** 0.7905
    d1, d2 = d1_km * 1000, d2_km * 1000
    return t1 / d1 * (a(d1) / a(d2)) * d2


# ---- Jack Daniels / Gilbert VDOT ------------------------------------------------------
def _vo2(v_m_min):
    return -4.60 + 0.182258 * v_m_min + 0.000104 * v_m_min ** 2


def _pct_max(t_min):
    return 0.8 + 0.1894393 * np.exp(-0.012778 * t_min) + 0.2989558 * np.exp(-0.1932605 * t_min)


def vdot_from_race(t_s: float, d_km: float) -> float:
    t_min = t_s / 60
    return _vo2(d_km * 1000 / t_min) / _pct_max(t_min)


def time_from_vdot(vdot: float, d_km: float = RACE_KM) -> float:
    f = lambda t_min: _vo2(d_km * 1000 / t_min) / _pct_max(t_min) - vdot
    return brentq(f, 30, 600) * 60


# ---- Personalised Riegel ----------------------------------------------------------------
def fit_personal_riegel(efforts: pd.DataFrame, n_boot: int = 2000, seed: int = 0):
    """log T = log a + b log D on the season-best curve; bootstrap over the underlying runs."""
    e = efforts[efforts.km.isin(FIT_KM)]
    best = best_efforts_table(e)
    b, loga = np.polyfit(np.log(best.km), np.log(best.time_s), 1)
    point = float(np.exp(loga) * RACE_KM ** b)

    rng = np.random.default_rng(seed)
    runs = e.run_id.unique()
    boots = []
    for _ in range(n_boot):
        sample = e[e.run_id.isin(rng.choice(runs, len(runs), replace=True))]
        bb = best_efforts_table(sample)
        if bb.km.nunique() < 3:
            continue
        b_, a_ = np.polyfit(np.log(bb.km), np.log(bb.time_s), 1)
        boots.append(np.exp(a_) * RACE_KM ** b_)
    lo, hi = np.percentile(boots, [5, 95])
    return point, float(lo), float(hi), float(b), best


def main() -> None:
    efforts = pd.read_csv(PROC / "best_efforts.csv")
    ctx = json.loads((PROC / "race_context.json").read_text(encoding="utf-8"))
    season = best_efforts_table(efforts).set_index("km")
    recent = best_efforts_table(efforts, since_days=84).set_index("km")

    rows = []

    def add(method, pred, notes, lo=None, hi=None):
        rows.append({"method": method, "predicted_s": round(pred), "low_s": lo and round(lo),
                     "high_s": hi and round(hi), "notes": notes})

    for label, tbl in (("season best", season), ("last 12 wk", recent)):
        for k in (5, 10):
            if k in tbl.index:
                t = tbl.loc[k, "time_s"]
                add(f"Riegel from {k}k ({label})", riegel(t, k),
                    f"{k}k {fmt_hms(t)} on {tbl.loc[k, 'date']}, exponent 1.06")
    t15 = season.loc[15, "time_s"]
    add("Riegel from 15k (season best)", riegel(t15, 15), f"15k {fmt_hms(t15)} on {season.loc[15, 'date']}")
    add("Cameron from 10k (season best)", cameron(season.loc[10, "time_s"], 10), "Cameron (1998) formula")

    p, lo, hi, b, _ = fit_personal_riegel(efforts)
    add("Personalised Riegel (own fatigue curve)", p, f"fitted exponent b={b:.3f} on 3/5/10/15k season bests; "
        "90% bootstrap interval over runs", lo, hi)

    v10 = vdot_from_race(season.loc[10, "time_s"], 10)
    add("Daniels VDOT from 10k (season best)", time_from_vdot(v10), f"VDOT {v10:.1f}")
    v10r = vdot_from_race(recent.loc[10, "time_s"], 10)
    add("Daniels VDOT from 10k (last 12 wk)", time_from_vdot(v10r), f"VDOT {v10r:.1f}")
    vo2 = ctx["garmin_vo2max_last_pre_race"]
    add("Daniels VDOT from Garmin VO2max", time_from_vdot(vo2), f"Garmin VO2max {vo2} used as VDOT")

    add("Garmin Race Predictor", ctx["garmin_race_predictor_hm_s_2026_09_19"],
        "Garmin Connect prediction on 2026-09-19 (commercial black box)")

    df = pd.DataFrame(rows)
    save_predictions(rows, "M1 Formulas")
    print(f"Personalised Riegel exponent: {b:.3f} (standard 1.06)")
    for r in df.itertuples():
        band = f"  [{fmt_hms(r.low_s)} - {fmt_hms(r.high_s)}]" if pd.notna(r.low_s) else ""
        print(f"{r.method:<42} {fmt_hms(r.predicted_s)}{band}   {r.notes}")


if __name__ == "__main__":
    main()
