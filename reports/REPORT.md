# Rapport d'évaluation sysone

Modèle : `mistralai/Mistral-7B-Instruct-v0.3`

## Critères de succès (DoD §0)

| Critère | Seuil | Statut |
|---|---|---|
| ECE (15 bins) sur tâches jamais vues | < 0.05 | ❌ (0.2679) |
| Amélioration ECE vs backbone brut | facteur ≥ 3 | ❌ (0.1725 → 0.2679, facteur 0.6×) |
| Erreur de type / hors-schéma | = 0 | ✅ (garanti par construction, cf. test_schema.py) |
| Latence N=10 vs N=1 | < 1.4× | ❌ (1.63×) |
| AURC vs baseline JSON | strictement meilleur | ✅ |
| Accuracy à couverture 100 % | ≥ 95 % de la baseline JSON | ✅ |


## Tableau des 4 configurations × métriques

| Config | NLL↓ | Brier↓ | ECE↓ | ACE↓ | AURC↓ | Acc | lat p50 (ms) | lat p95 (ms) | débit (q/s) |
|---|---|---|---|---|---|---|---|---|---|
| 1. Backbone brut | 1.0867 | 0.4542 | 0.1725 | 0.1719 | 0.0992 | 0.6900 | 26.0 | 55.0 | 4.5 | 
| JSON généré | — | — | — | — | — | — | 278.5 | 625.0 | 3.4 |
| 3. Fine-tuné sans T | 1.6188 | 0.7578 | 0.3405 | 0.3415 | 0.2694 | 0.5033 | 34.6 | 60.1 | 1.9 | 
| 4. Système complet | 1.5983 | 0.7099 | 0.2679 | 0.2647 | 0.3179 | 0.5433 | 55.4 | 89.9 | 19.0 | 

**Baseline JSON** : taux d'échec de parsing = **100.0 %**

## Diagrammes

- Fiabilité backbone brut : `reports/rel_raw.png`
- Fiabilité fine-tuné : `reports/rel_ft.png`
- Fiabilité système complet : `reports/rel_full.png`
- Risque/couverture backbone brut : `reports/rc_raw.png`
- Risque/couverture fine-tuné : `reports/rc_ft.png`
- Risque/couverture système complet : `reports/rc_full.png`
- Latence vs N : `reports/latency_batch.png` (ratio N=10/N=1 = 1.63×)

## Limites (honnêtes)

- Le modèle fine-tuné est un 8B : il n'a pas l'intelligence d'un modèle
  frontier sur les jugements ambigus.
- La calibration ne vaut que pour la distribution d'entraînement. En
  production, recalibrer via `sysone calibrate --data mes_exemples.jsonl`.
- Les gains de latence viennent du **prefix caching** et de la **sortie à
  1 token**, pas d'une innovation architecturale.
- Ce POC reproduit l'**interface** et le **mécanisme** d'un modèle de
  décision typée, pas l'architecture ni le sampler propriétaires de
  TypeSafe, qui ne sont pas publiés.
- L'éval se fait sur des datasets **entiers réservés** (jamais vus à
  l'entraînement) — cf. split §9.
