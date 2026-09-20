# Rapport d'Évaluation Exhaustif — 8 260 Exemples Holdout

Modèle évalué : `checkpoints/qwen_merged`
Date & Heure : 2026-09-21 00:19:35
Mode : Système Complet (LoRA fusionné + debiasing $k=3$ + calibration post-hoc)

---

## 1. Métriques Globales (Ensemble des 8 260 exemples non vus)

| Métrique | Valeur | Interprétation |
|---|:---:|---|
| **Exemples Totaux** | **8260** | 100 % du jeu d'évaluation holdout réservé |
| **Accuracy Globale** | **80.12 %** | Taux moyen de prédictions exactes en pur zéro-shot |
| **NLL (Negative Log-Likelihood)** | **0.4933** | Certitude probabiliste moyenne |
| **Brier Score** | **0.2575** | Précision probabiliste quadratique (proche de 0 = idéal) |
| **ECE (Expected Calibration Error)** | **0.0176** | Écart de calibration sur 15 bins |
| **ACE (Adaptive Calibration Error)** | **0.0186** | Écart de calibration avec bins équipopulés |
| **AURC (Area Under Risk-Coverage)** | **0.0479** | Capacité à rejeter les erreurs via le seuil de certitude |
| **Latence p50** | **58.6 ms** | 50 % des décisions prises sous ce délai |
| **Latence p95** | **96.5 ms** | Comportement sous charge / pire cas |
| **Débit Réel** | **14.1 q/s** | Requêtes par seconde sur un seul GPU NVIDIA RTX 5070 Ti |
| **Temps Total d'Exécution** | **9.8 min** | (585.5 secondes) |

---

## 2. Décomposition par Jeu de Données Source

| Dataset | Type de Tâche | Nb Exemples | Accuracy | ECE | Brier |
|---|---|:---:|:---:|:---:|:---:|
| **banking77** | Choice (77 classes) | 3076 | **91.48 %** | 0.0147 | 0.1298 |
| **massive** | Choice (60 classes) | 2974 | **91.49 %** | 0.0157 | 0.1253 |
| **sst5_eval** | Score (5 niveaux) | 2210 | **49.00 %** | 0.0613 | 0.6130 |

---

## 3. Focus Ordinal & Régression Continue ($\mathbb{E}[	ext{Score}]$)

| Dataset | Exact Match | Off-by-one ($\pm 1$) | MAE Discrète | MAE Continue $\mathbb{E}[S]$ | Spearman $\rho$ | Taux Biais Médian |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **sst5_eval** | **49.00 %** | **95.79 %** | 0.557 | 0.582 | **0.841** | 29.32 % |

---

## 4. Analyse & Comparaison

- **Performance Globale** : Accuracy de **80.12 %** sur le holdout exhaustif de 8 260 exemples avec une calibration remarquable ($ECE = 0.0176$).
- **Déterminisme d'API & Sécurité** : Taux d'erreur de schéma = **0.0 %** (aucun risque d'incompatibilité JSON ou d'hallucination de syntaxe).
- **Vitesse vs LLM génératif classique** : 58.6 ms vs ~300 ms pour un appel autorégressif JSON standard (**~5.1× plus rapide**).
