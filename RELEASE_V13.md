# V13 XGBoost Release Candidate

V13 is the frozen XGBoost release candidate derived from V10's development-only
diagnostic analysis. It has not been evaluated on a final holdout and is not a
production release.

## Frozen operating point

- Profile: `models/configs/xgb_v13_feature30_candidate.json`
- Decision threshold: `0.26`
- Policy: recall >= `0.92`; false-positive rate <= `0.005`
- Single controlled change from V10: `f_classif` selected features, `20 -> 30`

## Development-only cross-validation evidence

Five-fold nested stratified cross-validation used `train_data.csv` only. Every
outer fold used the frozen threshold of `0.26`; all five folds met both policy
targets.

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Recall | 0.9216 | 0.0007 |
| False-positive rate | 0.00362 | 0.00004 |
| Precision | 0.9853 | 0.0002 |
| F1-score | 0.9524 | 0.0004 |
| ROC-AUC | 0.9914 | 0.0001 |
| PR-AUC | 0.9803 | 0.0002 |

Compared with V10 at the same fixed threshold, V13 detected 790 more attacks
and produced 1,942 fewer false alerts across the five outer folds. This is
development evidence only, not a final deployment estimate.

## Next validation boundary

The V10 final holdout is consumed and must not be used for V13. Train a
canonical V13 artifact with the frozen profile, then evaluate it once on a new
independent holdout or external traffic dataset. The training pipeline honors
the frozen profile threshold rather than reselecting a different operating
point.

```powershell
.\venv\Scripts\python.exe -m src.train `
  --config models\configs\xgb_v13_feature30_candidate.json `
  --data data\processed\train_data.csv
```

Before performing final evaluation, record the new holdout source, its checksum,
and the fact that it was not used during V13 development.
