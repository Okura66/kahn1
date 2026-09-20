# Phase 0 — Spike de validation

Modèle : `mistralai/Mistral-7B-Instruct-v0.3`
Paramètre de restriction : `allowed_token_ids`

## Assertions

1. **Somme des probabilités ≈ 1.0** : orig=1.000000, perm=1.000000 → ✅

2. **Biais de position détecté** : delta sur option 'remboursement' = 0.0242 → ✅

3. **Surconfiance** : p_max(orig)=0.8502 → ✅

## Distributions brutes

Ordre original (A=remboursement, B=bug, C=annuler, D=facturation) :

```json
{
  "29509": 0.0014712146513953981,
  "29528": 0.0005944244533123724,
  "29511": 0.8501929749451994,
  "29525": 0.1477413859500928
}
```

Ordre permuté (A=facturation, B=annuler, C=bug, D=remboursement) :

```json
{
  "29509": 0.029058477889673844,
  "29528": 0.5314240438699648,
  "29511": 0.4138734615089012,
  "29525": 0.02564401673146012
}
```

## Verdict

**Toutes les assertions sont confirmées.** Le reste de la spec est valide, on peut construire l'architecture.
