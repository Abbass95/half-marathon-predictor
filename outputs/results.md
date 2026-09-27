# Results - actual time 2:02:11

| Family | Method | Predicted | 90% interval | Error | Error % | Actual in interval |
|---|---|---|---|---|---|---|
| M1 Formulas | Riegel from 5k (season best) | 1:55:15 | — | −6:56 | -5.7% | — |
| M1 Formulas | Riegel from 10k (season best) | 2:01:35 | — | −0:36 | -0.5% | — |
| M1 Formulas | Riegel from 5k (last 12 wk) | 2:00:31 | — | −1:40 | -1.4% | — |
| M1 Formulas | Riegel from 10k (last 12 wk) | 2:16:49 | — | +14:38 | +12.0% | — |
| M1 Formulas | Riegel from 15k (season best) | 1:59:23 | — | −2:48 | -2.3% | — |
| M1 Formulas | Cameron from 10k (season best) | 2:01:30 | — | −0:41 | -0.6% | — |
| M1 Formulas | Personalised Riegel (own fatigue curve) | 2:00:58 | 2:00:34–2:11:03 | −1:13 | -1.0% | yes |
| M1 Formulas | Daniels VDOT from 10k (season best) | 2:02:11 | — | +0:00 | +0.0% | — |
| M1 Formulas | Daniels VDOT from 10k (last 12 wk) | 2:17:20 | — | +15:09 | +12.4% | — |
| M1 Formulas | Daniels VDOT from Garmin VO2max | 1:29:40 | — | −32:31 | -26.6% | — |
| M1 Formulas | Garmin Race Predictor | 1:58:35 | — | −3:36 | -3.0% | — |
| M2 Personal ML | Training-load Ridge (all runs, no HR) | 2:18:12 | 2:09:56–2:28:06 | +16:01 | +13.1% | no |
| M2 Personal ML | Training-load GBM (all runs, no HR) | 2:29:07 | — | +26:56 | +22.0% | — |
| M2 Personal ML | Effort-aware Ridge (HR 168) | 2:24:07 | 2:16:00–2:32:59 | +21:56 | +17.9% | no |
| M2 Personal ML | Effort-aware GBM (HR 168) | 2:19:27 | 1:59:26–2:33:37 | +17:16 | +14.1% | yes |
| M3 Population ML | Population linear from 5k (Vickers, n=816) | 2:07:00 | 1:53:09–2:22:32 | +4:49 | +3.9% | yes |
| M3 Population ML | Population GBM from 5k (Vickers, n=816) | 2:06:04 | 1:52:12–2:21:39 | +3:53 | +3.2% | yes |
| M3 Population ML | Population linear from 10k (Vickers, n=526) | 2:10:00 | 1:54:55–2:27:05 | +7:49 | +6.4% | yes |
| M3 Population ML | Population GBM from 10k (Vickers, n=526) | 2:07:50 | 1:52:44–2:24:59 | +5:39 | +4.6% | yes |
| M3 Population ML | Population prior x personal curve (Bayesian) | 2:01:46 | 1:57:04–2:06:39 | −0:25 | -0.3% | yes |
| M3 Population ML | BMClab weak-label GBM (n=17,093) | 2:01:39 | — | −0:32 | -0.4% | — |
| M4 Claude only | Claude Opus 5.5 - minimal profile | 2:10:24 | 1:57:30–2:32:00 | +8:13 | +6.7% | yes |
| M4 Claude only | Claude Haiku 4.5 - minimal profile | 0:33:20 | 0:31:22–0:35:32 | −88:51 | -72.7% | no |
| M4 Claude only | Claude Opus 5.5 - full training log | 2:02:24 | 1:52:22–2:20:04 | +0:13 | +0.2% | yes |
| M4 Claude only | Claude Haiku 4.5 - full training log | 2:17:39 | 2:10:48–2:25:00 | +15:28 | +12.7% | no |
| M5 Hybrid | kNN baseline (k=20, median HM/5k ratio) | 2:04:02 | — | +1:51 | +1.5% | — |
| M5 Hybrid | Claude + retrieved similar runners (RAG) | 2:07:28 | 1:55:22–2:26:20 | +5:17 | +4.3% | yes |
| M5 Hybrid | Claude judging M1-M3 model outputs | 2:11:48 | 1:59:36–2:30:48 | +9:37 | +7.9% | yes |
| M5 Hybrid | Claude agent with Python tools | 2:06:30 | 1:56:13–2:24:00 | +4:19 | +3.5% | yes |
| M5 Hybrid | Median of all M1-M3 predictions (n=21) | 2:02:11 | — | +0:00 | +0.0% | — |
| M5 Hybrid | Inverse-variance mean (n=8 with intervals) | 2:10:11 | 2:06:47–2:13:39 | +8:00 | +6.5% | no |

## By family

| Family | Best method | Best error | Median error | Interval coverage | LLM cost |
|---|---|---|---|---|---|
| M1 Formulas | Daniels VDOT from 10k (season best) | 0:00 | 2:48 | 1/1 | $0.00 |
| M2 Personal ML | Training-load Ridge (all runs, no HR) | 16:01 | 19:36 | 1/3 | $0.00 |
| M3 Population ML | Population prior x personal curve (Bayesian) | 0:25 | 4:21 | 5/5 | $0.00 |
| M4 Claude only | Claude Opus 5.5 - full training log | 0:13 | 11:50 | 2/4 | $0.20 |
| M5 Hybrid | Median of all M1-M3 predictions (n=21) | 0:00 | 4:48 | 3/4 | $0.58 |