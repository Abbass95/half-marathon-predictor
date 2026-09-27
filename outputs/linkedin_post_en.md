I ran my first half marathon last week: 2:02:11, on about 9 km of training per week 😅

Then I did what a data person does: I tried to predict that time as if it were the night before the race. Strict rule: no model saw anything from race day.

5 approaches, same data (Strava + Garmin):
🔹 Running formulas (Riegel, Daniels VDOT) + Garmin's own predictor
🔹 Personal ML on 316 km-splits from my 42 runs
🔹 Population ML: transfer learning from 1,571 runners + a Bayesian blend with my own fatigue curve
🔹 Claude (Opus 5.5) as a coach, with little vs full context
🔹 Hybrids: RAG over similar runners, and a Claude agent calling my Python models as tools

What I learned:

1️⃣ Small data isn't ML-ready. My personal models learned my easy-run pace and missed by 16+ min. A decades-old formula (VDOT) from my best 10 km hit the time to the second. That's luck with n=1, but all three formulas based on that 10 km landed within a minute.

2️⃣ Context matters a lot for LLMs. Claude with a 5-line profile: +8 min. With my full training log: +13 s.

3️⃣ Uncertainty beats point estimates. Population data × my own curve (Bayesian) was 25 s off, with the narrowest 90% interval that still contained the real time.

4️⃣ The hybrids were over-cautious. Seeing low-volume runners made Claude expect me to fade badly. I only slowed by 3 s/km.

Bonus: sleep and HRV had no measurable effect on my pace, and Garmin's VO2max implies a 1:29 half marathon. I wish 🙃

Total LLM cost for the study: $0.78.

Code (Python, scikit-learn, Anthropic API) 👇
https://github.com/Abbass95/half-marathon-predictor

#MachineLearning #DataScience #LLM #Claude #Running #HalfMarathon
