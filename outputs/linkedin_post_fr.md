J'ai couru mon premier semi-marathon la semaine dernière : 2:02:11, avec environ 9 km d'entraînement par semaine 😅

Ensuite, j'ai fait ce que fait un data scientist : essayer de prédire ce temps comme si on était la veille de la course. Règle stricte : aucun modèle n'a vu la moindre donnée du jour J.

5 approches, mêmes données (Strava + Garmin) :
🔹 Formules de course (Riegel, VDOT de Daniels) + le prédicteur de Garmin
🔹 ML personnel sur 316 kilomètres issus de mes 42 sorties
🔹 ML sur une population : transfer learning à partir de 1 571 coureurs + un mélange bayésien avec ma propre courbe de fatigue
🔹 Claude (Opus 5.5) en coach, avec peu ou beaucoup de contexte
🔹 Hybrides : RAG sur des coureurs similaires, et un agent Claude qui appelle mes modèles Python comme outils

Ce que j'en retiens :

1️⃣ Peu de données, ce n'est pas du ML. Mes modèles personnels ont appris mon allure d'endurance et se sont trompés de plus de 16 min. Une formule vieille de plusieurs décennies (VDOT), à partir de mon meilleur 10 km, a trouvé le temps à la seconde près. C'est de la chance avec n=1, mais les trois formules basées sur ce 10 km sont toutes à moins d'une minute.

2️⃣ Pour un LLM, le contexte change tout. Claude avec un profil de 5 lignes : +8 min. Avec tout mon historique : +13 s.

3️⃣ L'incertitude compte plus qu'un chiffre unique. Population × ma courbe (bayésien) : 25 s d'écart, avec l'intervalle à 90 % le plus étroit qui contenait le vrai temps.

4️⃣ Les hybrides ont été trop prudents. En voyant des coureurs peu entraînés, Claude s'attendait à ce que je craque. Je n'ai ralenti que de 3 s/km.

Bonus : le sommeil et la VFC n'ont eu aucun effet mesurable sur mon allure, et la VO2max de Garmin prédit un semi en 1:29. J'aimerais bien 🙃

Coût total des appels LLM : 0,78 $.

Le code (Python, scikit-learn, API Anthropic) 👇
https://github.com/Abbass95/half-marathon-predictor

#MachineLearning #DataScience #LLM #Claude #Running #SemiMarathon
