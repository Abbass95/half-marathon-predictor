"""Evaluation: the ONLY module that loads the actual race result. Scores every prediction and draws charts.

Outputs (outputs/)
  results.csv / results.md   every method: prediction, interval, error, within-interval
  family_summary.csv         per family: best / median absolute error, LLM cost
  chart1_hero.png            headline methods vs the actual time (LinkedIn image)
  chart1_all_methods.png     every method (README)
  chart2_fatigue_curve.png   personal best-effort curve, Riegel vs personalised Riegel
  chart3_race_pacing.png     actual km splits vs model-predicted pacing
  chart4_training.png        weekly volume + sleep/HRV before the race
  chart5_population.png      where this runner sits among other runners
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from .common import OUT, PROC, RACE_KM, fmt_hms, fmt_pace
from .features import best_efforts_table
from .m1_formulas import fit_personal_riegel

# ---- style: reference palette (light surface), families in fixed slot order ----------------
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
FAMILY_COLORS = {"M1 Formulas": "#2a78d6", "M2 Personal ML": "#eb6834", "M3 Population ML": "#1baf7a",
                 "M4 Claude only": "#eda100", "M5 Hybrid": "#e87ba4"}
FAMILY_LABELS = {"M1 Formulas": "Formulas & Garmin", "M2 Personal ML": "Personal ML",
                 "M3 Population ML": "Population ML", "M4 Claude only": "Claude (LLM) only",
                 "M5 Hybrid": "Hybrid ML + LLM"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8,
})

# Headline methods for the LinkedIn chart: 1-2 per family, always including the family's best.
HERO = [
    "Daniels VDOT from 10k (season best)", "Garmin Race Predictor",
    "Training-load Ridge (all runs, no HR)",
    "Population prior x personal curve (Bayesian)", "Population linear from 5k (Vickers, n=816)",
    "Claude Opus 5.5 - full training log", "Claude Opus 5.5 - minimal profile",
    "Median of all M1-M3 predictions (n=21)", "Claude agent with Python tools",
]
SHORT = {
    "Riegel from 10k (season best)": "Riegel formula (best 10k)",
    "Daniels VDOT from 10k (season best)": "Daniels VDOT (best 10k)",
    "Personalised Riegel (own fatigue curve)": "Riegel fitted to my own curve",
    "Garmin Race Predictor": "Garmin Race Predictor",
    "Training-load Ridge (all runs, no HR)": "Ridge on 316 km-splits",
    "Effort-aware GBM (HR 168)": "Gradient boosting + heart rate",
    "Population linear from 5k (Vickers, n=816)": "Regression on 816 runners",
    "Population prior x personal curve (Bayesian)": "Bayesian: population x my curve",
    "Claude Opus 5.5 - minimal profile": "Claude, 5-line profile",
    "Claude Opus 5.5 - full training log": "Claude, full training log",
    "Claude + retrieved similar runners (RAG)": "Claude + 20 similar runners (RAG)",
    "Claude judging M1-M3 model outputs": "Claude judging the ML models",
    "Claude agent with Python tools": "Claude agent with Python tools",
    "Median of all M1-M3 predictions (n=21)": "Ensemble median (no LLM)",
}


def actual_time() -> float:
    runs = pd.read_csv(PROC / "runs.csv")
    return float(runs[runs.is_race].elapsed_s.iloc[0])


def fmt_err(e: float) -> str:
    """Signed error as m:ss (minutes may exceed 59)."""
    a = int(round(abs(e)))
    return f"{'+' if e >= 0 else '−'}{a // 60}:{a % 60:02d}"


def hm_fmt(x, _=None) -> str:
    x = int(round(x))
    return f"{x // 3600}:{x % 3600 // 60:02d}"


# ------------------------------------------------------------------------------ scoring
def score(actual: float) -> pd.DataFrame:
    p = pd.read_csv(OUT / "predictions.csv")
    p["error_s"] = p.predicted_s - actual
    p["abs_error_s"] = p.error_s.abs()
    p["error_pct"] = (p.error_s / actual * 100).round(2)
    p["within_interval"] = np.where(p.low_s.notna(), (p.low_s <= actual) & (actual <= p.high_s), None)
    p["cost_usd"] = p.notes.str.extract(r"cost \$([\d.]+)").astype(float)
    p.to_csv(OUT / "results.csv", index=False)

    fam = p.groupby("family", sort=False).agg(
        n_methods=("method", "size"),
        best_method=("abs_error_s", lambda s: p.loc[s.idxmin(), "method"]),
        best_abs_error=("abs_error_s", "min"),
        median_abs_error=("abs_error_s", "median"),
        interval_coverage=("within_interval", lambda s: f"{int(s.dropna().sum())}/{s.notna().sum()}"),
        llm_cost_usd=("cost_usd", "sum"))
    fam.to_csv(OUT / "family_summary.csv")

    lines = [f"# Results - actual time {fmt_hms(actual)}\n",
             "| Family | Method | Predicted | 90% interval | Error | Error % | Actual in interval |",
             "|---|---|---|---|---|---|---|"]
    for r in p.itertuples():
        band = f"{fmt_hms(r.low_s)}–{fmt_hms(r.high_s)}" if pd.notna(r.low_s) else "—"
        inside = {True: "yes", False: "no"}.get(r.within_interval, "—")
        lines.append(f"| {r.family} | {r.method} | {fmt_hms(r.predicted_s)} | {band} | "
                     f"{fmt_err(r.error_s)} | {r.error_pct:+.1f}% | {inside} |")
    lines += ["", "## By family", "", "| Family | Best method | Best error | Median error | Interval coverage | LLM cost |",
              "|---|---|---|---|---|---|"]
    for f, r in fam.iterrows():
        lines.append(f"| {f} | {r.best_method} | {fmt_err(r.best_abs_error)[1:]} | "
                     f"{fmt_err(r.median_abs_error)[1:]} | {r.interval_coverage} | ${r.llm_cost_usd:.2f} |")
    (OUT / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return p


# ------------------------------------------------------------------------------ charts
def dot_chart(p: pd.DataFrame, actual: float, fname: str, title: str, labels: dict | None,
              lo_lim=1.75 * 3600, hi_lim=2.58 * 3600, width=12, row_h=0.42):
    p = p.copy()
    p["fam_order"] = p.family.map({f: i for i, f in enumerate(FAMILY_COLORS)})
    p = p.sort_values(["fam_order", "predicted_s"], ascending=[True, True]).reset_index(drop=True)
    # Leave a gap row between families for the family header.
    ys, y, headers, prev = [], 0, [], None
    for r in p.itertuples():
        if r.family != prev:
            y += 1.0 if prev is not None else 0
            headers.append((y, r.family))
            y += 1
            prev = r.family
        ys.append(y)
        y += 1
    p["y"] = ys
    fig, ax = plt.subplots(figsize=(width, 1.6 + row_h * (y + 1)))
    ax.axvline(actual, color=INK, lw=2, zorder=1)
    ax.text(actual + 45, -1.1, f"Actual  {fmt_hms(actual)}", ha="left", va="bottom", fontsize=11,
            fontweight="bold", color=INK)
    for yh, f in headers:
        ax.text(lo_lim - 90, yh, FAMILY_LABELS[f].upper(), ha="right", va="center", fontsize=10,
                fontweight="bold", color=INK2)
        ax.plot([lo_lim + 60, lo_lim + 360], [yh, yh], color=FAMILY_COLORS[f], lw=6,
                solid_capstyle="round", clip_on=False)
    for r in p.itertuples():
        c = FAMILY_COLORS[r.family]
        name = (labels or {}).get(r.method, r.method)
        if pd.notna(r.low_s):
            ax.plot([max(r.low_s, lo_lim), min(r.high_s, hi_lim)], [r.y, r.y], color=c, lw=2, alpha=0.45,
                    solid_capstyle="round", zorder=2)
        off = r.predicted_s < lo_lim or r.predicted_s > hi_lim
        x = min(max(r.predicted_s, lo_lim + 60), hi_lim - 60)
        ax.scatter([x], [r.y], s=70, color=c, edgecolor=SURFACE, linewidth=2, zorder=3,
                   marker="<" if r.predicted_s < lo_lim else (">" if r.predicted_s > hi_lim else "o"))
        err = r.predicted_s - actual
        tag = f"{fmt_hms(r.predicted_s)}  ({fmt_err(err)})"
        if off:
            tag = f"{fmt_hms(r.predicted_s)} (off scale)"
        ax.text(hi_lim + 90, r.y, tag, va="center", fontsize=9.5, color=INK2)
        ax.text(lo_lim - 90, r.y, name, va="center", ha="right", fontsize=10, color=INK)
    ax.set_xlim(lo_lim, hi_lim)
    ax.set_ylim(y, -1.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(hm_fmt))
    ax.set_xticks(np.arange(np.ceil(lo_lim / 300) * 300, hi_lim + 1, 600))
    ax.grid(axis="x")
    ax.set_xlabel("Predicted finish time (h:mm)  -  bar = 90% interval, (±m:ss) = error vs actual")
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.01, 0.995 - 0.55 / fig.get_figheight(), "Semi-marathon Marcq-en-Barœul, 20 Sept 2026 · "
             "every method only saw data up to the day before", ha="left", fontsize=10.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.62 / fig.get_figheight()))
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def chart_fatigue(actual: float):
    e = pd.read_csv(PROC / "best_efforts.csv")
    best = best_efforts_table(e)
    best = best[best.km >= 3]
    _, _, _, b, _ = fit_personal_riegel(e)
    d = np.linspace(3, RACE_KM, 200)
    loga = np.polyfit(np.log(best.km), np.log(best.time_s), 1)[1]
    t10 = best.set_index("km").loc[10, "time_s"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(d, t10 * (d / 10) ** 1.06 / d, color="#2a78d6", lw=2, label="Riegel (standard exponent 1.06, from 10 km)")
    ax.plot(d, np.exp(loga) * d ** b / d, color="#eb6834", lw=2, ls="--",
            label=f"Riegel fitted to my own efforts (exponent {b:.3f})")
    ax.scatter(best.km, best.time_s / best.km, s=70, color=INK2, zorder=3, label="My season-best training efforts")
    for r in best.itertuples():
        ax.annotate(f"{int(r.km)} km\n{fmt_pace(r.time_s / r.km)}", (r.km, r.time_s / r.km),
                    textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9, color=INK2)
    ax.scatter([RACE_KM], [actual / RACE_KM], s=180, marker="*", color=INK, zorder=4, label="Actual race")
    ax.annotate(f"race {fmt_pace(actual / RACE_KM)}", (RACE_KM, actual / RACE_KM), textcoords="offset points",
                xytext=(-14, 4), ha="right", fontsize=10, fontweight="bold")
    ax.axvspan(15, RACE_KM + 0.5, color=GRID, alpha=0.5, lw=0)
    ax.text(18.1, ax.get_ylim()[0] + 3, "never trained\nbeyond 15 km", ha="center", fontsize=9, color=INK2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt_pace(x).replace("/km", "")))
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Pace (min/km, lower = faster)")
    ax.set_xlim(2, RACE_KM + 0.5)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    ax.set_title("Extrapolating my fatigue curve from ≤15 km to 21.1 km", loc="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "chart2_fatigue_curve.png", dpi=150)
    plt.close(fig)


def chart_pacing(actual: float, p: pd.DataFrame):
    race = pd.read_csv(PROC / "race_splits.csv")
    pace = pd.read_csv(OUT / "m2_pacing.csv").iloc[:21]
    first, second = race[race.km <= 10].pace_s.mean(), race[race.km > 11].pace_s.mean()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(pace.km, pace["M2a Ridge"], color="#eb6834", lw=2, marker="o", ms=5,
            label="Personal ML (Ridge) - expected pacing")
    ax.plot(race.km, race.pace_s, color=INK, lw=2, marker="o", ms=6, label="My actual km splits (watch)")
    ax.hlines(first, 1, 10, color="#2a78d6", lw=2, ls="--", label=f"Actual mean km 1-10: {fmt_pace(first)}")
    ax.hlines(second, 12, 21, color="#2a78d6", lw=2, ls=":", label=f"Actual mean km 12-21: {fmt_pace(second)}")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt_pace(x).replace("/km", "")))
    ax.set_xticks(range(1, 22, 2))
    ax.set_xlabel("Kilometre")
    ax.set_ylabel("Pace (min/km, higher = slower)")
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", ncol=2)
    ax.set_ylim(race.pace_s.min() - 15, pace["M2a Ridge"].max() + 25)
    ax.set_title(f"The personal model learned my easy-run pace; race pace was {fmt_pace(race.pace_s.mean())} "
                 f"with a {second - first:.0f} s/km late fade", loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "chart3_race_pacing.png", dpi=150)
    plt.close(fig)


def chart_training():
    runs = pd.read_csv(PROC / "run_features.csv", parse_dates=["date"])
    g = pd.read_csv(PROC / "garmin_daily.csv", parse_dates=["date"])
    wk = runs.set_index("date").distance_m.resample("W-SUN").sum() / 1000
    g = g[(g.date >= "2026-06-01") & (g.date < "2026-09-20")]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1, 1]})
    ax = axes[0]
    ax.bar(wk.index, wk.values, width=5, color="#2a78d6", edgecolor=SURFACE, lw=1)
    ax.axvline(pd.Timestamp("2026-09-20"), color=INK, lw=1.5)
    ax.text(pd.Timestamp("2026-09-18"), wk.max() * 0.95, "race", ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel("km per week")
    ax.grid(axis="y")
    ax.set_title("12 months of training: 333 km in total, 9.4 km/week in the last 12 weeks",
                 loc="left", fontsize=13, fontweight="bold")
    for ax, col, lab in ((axes[1], "sleep_h", "Sleep (h)"), (axes[2], "hrv_overnight_ms", "Overnight HRV (ms)")):
        ax.plot(g.date, g[col], color="#8a8984", lw=1)
        ax.plot(g.date, g[col].rolling(7, min_periods=3).mean(), color="#1baf7a", lw=2)
        ax.set_ylabel(lab)
        ax.grid(axis="y")
        ax.axvline(pd.Timestamp("2026-09-20"), color=INK, lw=1.5)
    axes[1].set_title("Garmin wellness (grey = daily, green = 7-day mean): no measurable effect on my pace",
                      loc="left", fontsize=10.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "chart4_training.png", dpi=150)
    plt.close(fig)


def chart_population(actual: float):
    s = pd.read_csv(PROC / "bmclab_hm_samples.csv")
    v = pd.read_csv(PROC / "vickers.csv")
    rep = json.loads((OUT / "m3_report.json").read_text(encoding="utf-8"))
    me_vol = rep["bmclab"]["abbass"]["km_per_week_12w"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    bins = np.logspace(np.log10(3), np.log10(250), 40)
    ax.hist(s.km_per_week_12w.clip(3, 250), bins=bins, color="#1baf7a", edgecolor=SURFACE, lw=1)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100, 200])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.axvline(me_vol, color=INK, lw=2)
    ax.text(me_vol * 1.08, ax.get_ylim()[1] * 0.9, f"me: {me_vol:.1f} km/wk\n{rep['bmclab']['volume_percentile']:.0f}th percentile",
            fontsize=10, fontweight="bold")
    ax.set_xlabel("Average km/week in the 12 weeks before their fastest ~21 km (log scale)")
    ax.set_ylabel("Athletes")
    ax.set_title(f"BMClab: {len(s):,} runners", loc="left", fontsize=12, fontweight="bold")
    ax = axes[1]
    ax.scatter(v.typical_km_week, v.hm_s, s=10, color="#1baf7a", alpha=0.35, lw=0)
    ax.scatter([me_vol], [actual], s=200, marker="*", color=INK, zorder=3)
    ax.annotate(f"me: {fmt_hms(actual)}", (me_vol, actual), textcoords="offset points", xytext=(10, 8),
                fontsize=10, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_yticks(np.arange(3600, 3.5 * 3600, 900))
    ax.yaxis.set_major_formatter(FuncFormatter(hm_fmt))
    ax.set_xlabel("Typical km/week before the race (log scale)")
    ax.set_ylabel("Half-marathon time (h:mm)")
    ax.set_title(f"Vickers survey: {len(v):,} half-marathon runners", loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "chart5_population.png", dpi=150)
    plt.close(fig)


def main() -> None:
    actual = actual_time()
    p = score(actual)
    hero = p[p.method.isin(HERO)]
    missing = set(HERO) - set(hero.method)
    if missing:
        print("WARNING: hero methods missing:", missing)
    dot_chart(hero, actual, "chart1_hero.png", "Predicting my first half marathon: formulas vs ML vs Claude",
              SHORT, width=11, row_h=0.55)
    dot_chart(p, actual, "chart1_all_methods.png", f"All {len(p)} predictions vs the actual time", None,
              lo_lim=1.7 * 3600, width=14)
    chart_fatigue(actual)
    chart_pacing(actual, p)
    chart_training()
    chart_population(actual)

    fam = pd.read_csv(OUT / "family_summary.csv")
    print(f"Actual: {fmt_hms(actual)}\n")
    print(p.sort_values("abs_error_s")[["family", "method", "predicted_s", "error_s", "within_interval"]]
          .assign(predicted=lambda d: d.predicted_s.map(fmt_hms),
                  error=lambda d: d.error_s.map(fmt_err))
          [["family", "method", "predicted", "error", "within_interval"]].to_string(index=False))
    print("\n", fam.to_string(index=False))
    print("\nCharts written to", OUT)


if __name__ == "__main__":
    main()
