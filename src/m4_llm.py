"""M4 - Claude only: predict from (a) a minimal profile, (b) the full training log.

Each variant is sampled several times (thinking-model outputs vary run to run) and with a large
and a small model. Responses are cached in outputs/llm_responses.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import CFG, OUT, PROC, fmt_hms, save_predictions
from .features import best_efforts_table
from .llm import MODEL, SAMPLES, SMALL_MODEL, predict


def minimal_prompt(summary: dict) -> str:
    r, t = CFG["runner"], summary["totals"]
    be = summary["best_efforts_all_time"]
    return (
        f"Runner: {r['age']}-year-old man, {r['height_cm']} cm, {r['weight_kg']} kg.\n"
        f"Race: first-ever half marathon (21.1 km), 2026-09-20, flat city course, ~18 °C.\n"
        f"Training: about {t['avg_km_per_week_last_12_weeks']} km per week over the last 12 weeks; "
        f"longest run in the last 8 weeks {t['longest_run_last_8_weeks_km']} km.\n"
        f"Best 10 km in the last 12 months: {be['10km'].split(' ')[0]} (February 2026).\n\n"
        "Predict the half-marathon finish time."
    )


def full_prompt(summary: dict) -> str:
    s = json.loads(json.dumps(summary))
    # Keep Garmin's own race prediction out: it is a competing method (M1), not raw data.
    s["race_context"].pop("garmin_race_predictor_hm_s_2026_09_19", None)
    return (
        "Here is the complete training log and context for a runner's first half marathon "
        "(21.0975 km) on 2026-09-20. Best efforts are the fastest continuous segments found inside "
        "training runs (not races). Wellness values come from a Garmin watch.\n\n"
        f"```json\n{json.dumps(s, indent=1, ensure_ascii=False)}\n```\n\n"
        "Predict the half-marathon finish time."
    )


def run_variant(name: str, prompt: str, model: str) -> tuple[dict, list[dict]]:
    outs = [predict(f"M4 {name}", prompt, model=model, sample=i) for i in range(SAMPLES)]
    ok = [o for o in outs if o["output"]]
    preds = np.array([o["output"]["predicted_seconds"] for o in ok])
    row = {
        "method": f"Claude {'Opus 5.5' if model == MODEL else 'Haiku 4.5'} - {name}",
        "predicted_s": round(float(preds.mean())),
        "low_s": round(float(np.mean([o["output"]["low_seconds"] for o in ok]))),
        "high_s": round(float(np.mean([o["output"]["high_seconds"] for o in ok]))),
        "notes": f"mean of {len(ok)} samples (spread {fmt_hms(preds.min())}-{fmt_hms(preds.max())}); "
                 f"cost ${sum(o['cost_usd'] for o in outs):.3f}",
    }
    if preds.mean() < 3600:  # a half marathon under 1 h is impossible for this runner
        row["notes"] = ("IMPLAUSIBLE - arithmetic/unit error: the reasoning estimates ~5:50-6:05/km "
                        "but converts 21.1 km to ~33 min. ") + row["notes"]
    return row, outs


def main() -> None:
    summary = json.loads((PROC / "training_summary.json").read_text(encoding="utf-8"))
    prompts = {"minimal profile": minimal_prompt(summary), "full training log": full_prompt(summary)}
    (OUT / "m4_prompts.md").write_text(
        "\n\n".join(f"## {k}\n\n{v}" for k, v in prompts.items()), encoding="utf-8")

    rows, total_cost, examples = [], 0.0, {}
    for name, prompt in prompts.items():
        for model in (MODEL, SMALL_MODEL):
            row, outs = run_variant(name, prompt, model)
            rows.append(row)
            total_cost += sum(o["cost_usd"] for o in outs)
            examples[row["method"]] = outs[0]["output"]["reasoning"] if outs[0]["output"] else "REFUSED"
            print(f"{row['method']:<45} {fmt_hms(row['predicted_s'])} "
                  f"[{fmt_hms(row['low_s'])}-{fmt_hms(row['high_s'])}]  {row['notes']}")
    (OUT / "m4_reasoning_examples.json").write_text(json.dumps(examples, indent=1, ensure_ascii=False),
                                                    encoding="utf-8")
    save_predictions(rows, "M4 Claude only")
    print(f"\nTotal M4 API cost: ${total_cost:.3f}")
    print("\nSample reasoning (Opus 5.5, full log):\n", examples[f"Claude Opus 5.5 - full training log"])


if __name__ == "__main__":
    main()
