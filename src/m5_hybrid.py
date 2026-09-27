"""M5 - Hybrid ML + LLM.

5a RAG few-shot: retrieve the 20 most similar Vickers runners (kNN) -> Claude predicts from them.
5b ML -> LLM:    Claude reviews every M1-M3 model output + context the models cannot see.
5c Agent:        Claude calls Python tools (formulas, population model, data lookups) itself.
5d Ensemble:     equal-weight median and inverse-variance mean of M1-M3 (no LLM).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from anthropic import beta_tool
from sklearn.preprocessing import StandardScaler

from .common import CFG, OUT, PROC, RACE_KM, fmt_hms, save_predictions
from .features import best_efforts_table
from .llm import MODEL, SAMPLES, SYSTEM, cache_key, client, cost_usd, lookup, predict, record
from .m1_formulas import riegel, time_from_vdot, vdot_from_race
from .m3_population_ml import abbass_profile, abbass_x, models, vickers_xy

K = 20
AGENT_SAMPLES = 3
Z90 = 1.645


def load():
    summary = json.loads((PROC / "training_summary.json").read_text(encoding="utf-8"))
    summary["race_context"].pop("garmin_race_predictor_hm_s_2026_09_19", None)
    runs = pd.read_csv(PROC / "run_features.csv")
    efforts = pd.read_csv(PROC / "best_efforts.csv")
    v = pd.read_csv(PROC / "vickers.csv")
    return summary, runs, efforts, v


def compact_context(summary: dict) -> str:
    keep = {k: summary[k] for k in ("runner", "totals", "weekly_km_last_16_weeks", "best_efforts_all_time",
                                    "best_efforts_last_12_weeks", "wellness_last_4_weeks_avg", "race_context")}
    return json.dumps(keep, indent=1, ensure_ascii=False)


def summarize(name: str, outs: list[dict], extra_notes: str = "") -> dict:
    ok = [o for o in outs if o["output"]]
    p = np.array([o["output"]["predicted_seconds"] for o in ok])
    return {"method": name, "predicted_s": round(float(p.mean())),
            "low_s": round(float(np.mean([o["output"]["low_seconds"] for o in ok]))),
            "high_s": round(float(np.mean([o["output"]["high_seconds"] for o in ok]))),
            "notes": f"{extra_notes}mean of {len(ok)} samples (spread {fmt_hms(p.min())}-{fmt_hms(p.max())}); "
                     f"cost ${sum(o['cost_usd'] for o in outs):.3f}"}


# ---------------------------------------------------------------------- 5a RAG few-shot
def rag(summary, runs, efforts, v) -> tuple[list[dict], float]:
    prof = abbass_profile(runs, efforts, v)
    pool = v[v.k5_s.notna()].copy()
    feats = pd.DataFrame({"log_5k": np.log(pool.k5_s), "log_typical": np.log(pool.typical_km_week),
                          "age": pool.age, "bmi": pool.bmi, "female": pool.female, "fitness": pool.hm_fitness})
    me = pd.DataFrame([{"log_5k": np.log(prof["k5_s"]), "log_typical": np.log(prof["typical_km_week"]),
                        "age": prof["age"], "bmi": prof["bmi"], "female": prof["female"],
                        "fitness": prof["hm_fitness"]}])
    sc = StandardScaler().fit(feats)
    w = np.array([2.0, 1.5, 0.5, 0.5, 1.0, 1.0])  # 5k time and volume matter most
    d = np.sqrt((((sc.transform(feats) - sc.transform(me)) * w) ** 2).sum(axis=1))
    nn = pool.iloc[np.argsort(d)[:K]]
    table = pd.DataFrame({
        "age": nn.age.round().astype(int), "sex": np.where(nn.female == 1, "F", "M"), "bmi": nn.bmi.round(1),
        "typical_km_week": nn.typical_km_week.round(0), "max_km_week": nn.max_km_week.round(0),
        "intervals": nn.sprint.map({1: "yes", 0: "no"}), "tempo": nn.tempo.map({1: "yes", 0: "no"}),
        "fitness_1best_4worst": nn.hm_fitness.astype(int), "hm_course_1hard_5fast": nn.hm_difficulty.astype(int),
        "5k_time": nn.k5_s.map(fmt_hms), "half_marathon_time": nn.hm_s.map(fmt_hms)})
    table.to_csv(OUT / "m5a_neighbours.csv", index=False)
    knn_ratio = float(np.median(nn.hm_s / nn.k5_s) * prof["k5_s"])

    prompt = (
        "Target runner (first half marathon on 2026-09-20):\n"
        f"```json\n{compact_context(summary)}\n```\n\n"
        f"For reference, here are the {K} most similar runners from a published survey of 2,300 "
        "recreational runners (Vickers & Vertosick 2016), matched on 5 km time, weekly volume, age, BMI, "
        "sex and self-rated fitness, with their actual half-marathon times. Their 5 km times are race times; "
        "the target runner's 5 km is a training effort.\n\n"
        f"{table.to_csv(index=False)}\n"
        "Using these comparable runners as evidence, predict the target runner's half-marathon time."
    )
    outs = [predict("M5a RAG", prompt, sample=i) for i in range(SAMPLES)]
    rows = [
        {"method": f"kNN baseline (k={K}, median HM/5k ratio)", "predicted_s": round(knn_ratio),
         "notes": "no LLM: nearest Vickers runners' HM/5k ratio x your 5k"},
        summarize("Claude + retrieved similar runners (RAG)", outs),
    ]
    return rows, sum(o["cost_usd"] for o in outs)


# ---------------------------------------------------------------------- 5b ML -> LLM
def ml_to_llm(summary) -> tuple[list[dict], float]:
    preds = pd.read_csv(OUT / "predictions.csv")
    preds = preds[preds.family.str.match(r"M[123]")]
    lines = []
    for r in preds.itertuples():
        band = f" (90% interval {fmt_hms(r.low_s)}-{fmt_hms(r.high_s)})" if pd.notna(r.low_s) else ""
        lines.append(f"- [{r.family}] {r.method}: {fmt_hms(r.predicted_s)}{band}. {r.notes}")
    prompt = (
        "Several models have produced half-marathon predictions for this runner. Your job is to act as "
        "the final judge: weigh the models by how credible they are for THIS runner, correct for what "
        "they cannot see (endurance gap, first-race pacing, conditions, wellness), and give one final "
        "prediction.\n\nModel outputs:\n" + "\n".join(lines) +
        f"\n\nRunner context:\n```json\n{compact_context(summary)}\n```"
    )
    outs = [predict("M5b ML->LLM", prompt, sample=i) for i in range(SAMPLES)]
    return [summarize("Claude judging M1-M3 model outputs", outs)], sum(o["cost_usd"] for o in outs)


# ---------------------------------------------------------------------- 5c Agent with tools
def build_tools(summary, runs, efforts, v, trace: list):
    prof = abbass_profile(runs, efforts, v)
    pop = {}
    for short_col, label in (("k5_s", "5k"), ("k10_s", "10k")):
        X, y, _ = vickers_xy(v, short_col)
        m = models()["linear"]().fit(X, y)
        pop[label] = (m, short_col, float(np.std(y - m.predict(X))))

    def log(name, args, result):
        trace.append({"tool": name, "input": args, "output": result})
        return result

    @beta_tool
    def get_runner_profile() -> str:
        """Age, sex, height, weight, self-rated fitness and running experience of the runner."""
        return log("get_runner_profile", {}, json.dumps(summary["runner"]))

    @beta_tool
    def get_training_overview() -> str:
        """Training totals (12 months, 12 weeks, longest runs) and weekly km for the last 16 weeks."""
        return log("get_training_overview", {}, json.dumps({"totals": summary["totals"],
                                                            "weekly_km": summary["weekly_km_last_16_weeks"]}))

    @beta_tool
    def list_runs(last_n_weeks: int) -> str:
        """List individual runs (date, km, moving time, pace, avg HR, elevation, temperature).

        Args:
            last_n_weeks: How many weeks back from the race to include (1-52).
        """
        cutoff = str((pd.Timestamp(CFG["race"]["date"]) - pd.Timedelta(weeks=last_n_weeks)).date())
        out = [r for r in summary["runs"] if r["date"] >= cutoff]
        return log("list_runs", {"last_n_weeks": last_n_weeks}, json.dumps(out))

    @beta_tool
    def get_best_efforts(period: str) -> str:
        """Fastest continuous 1/3/5/10/15 km segments found inside training runs (not races).

        Args:
            period: "all" for the last 12 months or "last_12_weeks".
        """
        t = best_efforts_table(efforts, 84 if period == "last_12_weeks" else None)
        res = {f"{int(r.km)}km": {"time_seconds": round(r.time_s), "time": fmt_hms(r.time_s), "date": r.date}
               for r in t.itertuples()}
        return log("get_best_efforts", {"period": period}, json.dumps(res))

    @beta_tool
    def predict_riegel(time_seconds: float, distance_km: float, exponent: float = 1.06) -> str:
        """Riegel formula: extrapolate a performance to the half marathon (21.0975 km).

        Args:
            time_seconds: Time of the reference performance in seconds.
            distance_km: Distance of the reference performance in km.
            exponent: Fatigue exponent (1.06 standard; higher = more slowdown with distance).
        """
        t = riegel(time_seconds, distance_km, b=exponent)
        return log("predict_riegel", {"time_seconds": time_seconds, "distance_km": distance_km, "exponent": exponent},
                   json.dumps({"half_marathon_seconds": round(t), "half_marathon": fmt_hms(t)}))

    @beta_tool
    def predict_vdot(time_seconds: float, distance_km: float) -> str:
        """Jack Daniels VDOT: compute VDOT from a performance and the equivalent half-marathon time.

        Args:
            time_seconds: Time of the reference performance in seconds.
            distance_km: Distance of the reference performance in km.
        """
        vd = vdot_from_race(time_seconds, distance_km)
        t = time_from_vdot(vd)
        return log("predict_vdot", {"time_seconds": time_seconds, "distance_km": distance_km},
                   json.dumps({"vdot": round(vd, 1), "half_marathon_seconds": round(t), "half_marathon": fmt_hms(t)}))

    @beta_tool
    def predict_population_model(reference: str, reference_time_seconds: float, weekly_km: float) -> str:
        """Regression trained on ~1,570 recreational runners (Vickers & Vertosick 2016) predicting
        half-marathon time from a shorter race time, weekly volume, age, sex, BMI and fitness rating.
        Returns the prediction and the population 90% interval.

        Args:
            reference: "5k" or "10k".
            reference_time_seconds: The runner's time over that distance in seconds.
            weekly_km: Typical weekly running volume in km.
        """
        m, short_col, sd = pop["10k" if reference == "10k" else "5k"]
        p = dict(prof, typical_km_week=weekly_km, **{short_col: reference_time_seconds})
        t = float(np.exp(m.predict(abbass_x(p, short_col))[0]))
        return log("predict_population_model", {"reference": reference, "reference_time_seconds": reference_time_seconds,
                                                "weekly_km": weekly_km},
                   json.dumps({"half_marathon_seconds": round(t), "half_marathon": fmt_hms(t),
                               "interval_90": [fmt_hms(t * np.exp(-Z90 * sd)), fmt_hms(t * np.exp(Z90 * sd))]}))

    @beta_tool
    def get_wellness() -> str:
        """Garmin sleep, HRV, resting HR and body battery: last-4-week average and race morning."""
        return log("get_wellness", {}, json.dumps({
            "last_4_weeks_avg": summary["wellness_last_4_weeks_avg"],
            "race_morning": summary["race_context"]["race_morning_wellness"],
            "garmin_vo2max": summary["race_context"]["garmin_vo2max_last_pre_race"]}))

    @beta_tool
    def get_race_conditions() -> str:
        """Race course, start time and the weather forecast issued the day before."""
        rc = summary["race_context"]
        return log("get_race_conditions", {}, json.dumps({"race": rc["race"],
                                                           "forecast": rc["forecast_issued_day_before"]}))

    submitted = {}

    @beta_tool
    def submit_prediction(predicted_seconds: int, low_seconds: int, high_seconds: int, reasoning: str) -> str:
        """Submit the final half-marathon prediction. Call exactly once, at the end.

        Args:
            predicted_seconds: Best estimate of the finish time in seconds.
            low_seconds: Fast end of the 90% interval in seconds.
            high_seconds: Slow end of the 90% interval in seconds.
            reasoning: Short justification.
        """
        submitted.update(predicted_seconds=predicted_seconds, low_seconds=low_seconds,
                         high_seconds=high_seconds, reasoning=reasoning)
        return log("submit_prediction", dict(submitted), "Prediction recorded.")

    tools = [get_runner_profile, get_training_overview, list_runs, get_best_efforts, predict_riegel,
             predict_vdot, predict_population_model, get_wellness, get_race_conditions, submit_prediction]
    return tools, submitted


def agent(summary, runs, efforts, v) -> tuple[list[dict], float]:
    task = ("Predict the finish time of this runner's first half marathon on 2026-09-20. "
            "Investigate the data with the tools, run whichever calculations you find credible, "
            "then call submit_prediction once with your final answer.")
    outs = []
    for i in range(AGENT_SAMPLES):
        key = cache_key("agent", MODEL, task, i)
        entry = lookup(key)
        if entry is None:
            trace: list = []
            tools, submitted = build_tools(summary, runs, efforts, v, trace)
            runner = client().beta.messages.tool_runner(
                model=MODEL, max_tokens=16000, system=SYSTEM, tools=tools,
                output_config={"effort": "high"},
                messages=[{"role": "user", "content": task}])
            usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                     "cache_creation_input_tokens": 0}
            turns = 0
            for message in runner:
                turns += 1
                for k in usage:
                    usage[k] += getattr(message.usage, k, 0) or 0
            entry = {"tag": "M5c agent", "model": MODEL, "sample": i, "turns": turns, "usage": usage,
                     "cost_usd": cost_usd(MODEL, usage), "output": dict(submitted) or None, "trace": trace}
            record(key, entry)
        outs.append(entry)
    (OUT / "m5c_agent_traces.json").write_text(json.dumps(
        [{"sample": o["sample"], "turns": o["turns"], "tool_calls": [t["tool"] for t in o["trace"]],
          "trace": o["trace"]} for o in outs], indent=1, ensure_ascii=False), encoding="utf-8")
    calls = np.mean([len(o["trace"]) for o in outs])
    return [summarize("Claude agent with Python tools", outs, f"{calls:.0f} tool calls/run avg; ")], \
        sum(o["cost_usd"] for o in outs)


# ---------------------------------------------------------------------- 5d Ensemble
def ensemble() -> list[dict]:
    p = pd.read_csv(OUT / "predictions.csv")
    p = p[p.family.str.match(r"M[123]")]
    med = float(np.exp(np.log(p.predicted_s).median()))
    # Inverse-variance over rows with an interval; skip the Bayesian row (already a combination).
    iv = p[p.low_s.notna() & ~p.method.str.contains("Bayesian")].copy()
    iv["sd"] = (np.log(iv.high_s) - np.log(iv.low_s)) / (2 * Z90)
    wts = 1 / iv.sd ** 2
    mu = float((wts * np.log(iv.predicted_s)).sum() / wts.sum())
    sd = float(wts.sum() ** -0.5)
    return [
        {"method": f"Median of all M1-M3 predictions (n={len(p)})", "predicted_s": round(med),
         "notes": "equal weight, robust to outliers; weights cannot be learned from one race"},
        {"method": f"Inverse-variance mean (n={len(iv)} with intervals)", "predicted_s": round(np.exp(mu)),
         "low_s": round(np.exp(mu - Z90 * sd)), "high_s": round(np.exp(mu + Z90 * sd)),
         "notes": "each model weighted by 1/variance of its own 90% interval (assumes independence -> too narrow)"},
    ]


def main() -> None:
    summary, runs, efforts, v = load()
    rows, cost = [], 0.0
    for fn in (rag, ml_to_llm):
        r, c = fn(summary, runs, efforts, v) if fn is rag else fn(summary)
        rows += r
        cost += c
    r, c = agent(summary, runs, efforts, v)
    rows += r
    cost += c
    rows += ensemble()
    save_predictions(rows, "M5 Hybrid")
    for r in rows:
        band = f" [{fmt_hms(r['low_s'])}-{fmt_hms(r['high_s'])}]" if r.get("low_s") else ""
        print(f"{r['method']:<48} {fmt_hms(r['predicted_s'])}{band}  {r['notes']}")
    print(f"\nTotal M5 API cost: ${cost:.3f}")


if __name__ == "__main__":
    main()
