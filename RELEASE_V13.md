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

## Canonical artifact verification

- Artifact: `models/saved/xgb_v13_feature30.pkl`
- Artifact SHA-256:
  `CDE4E1A9439D7B9F5B96150E5EF838FC4B9C3012C85DD7843B798EA437ED7054`
- Metadata: `models/saved/xgb_v13_feature30_metadata.json`

The artifact was trained on `train_data.csv` with the frozen threshold of
`0.26`. Its internal development split achieved 92.18% attack recall and a
0.351% FPR. A read-only inference smoke test successfully loaded the artifact,
applied its saved MinMax scaler and 30-feature schema, and produced predictions
for 100 input rows.

This is an artifact-integrity check, not a final independent evaluation.

## Next validation boundary

The V10 final holdout is consumed and must not be used for V13. The canonical
V13 artifact is already trained. Evaluate that exact artifact once on a new
independent holdout or external traffic dataset. The training pipeline honors
the frozen profile threshold rather than reselecting a different operating
point.

Before performing final evaluation, record the new holdout source, its checksum,
and the fact that it was not used during V13 development.

## CIC-IDS2017 compatibility preflight

The supplied CIC-IDS2017 `MachineLearningCSV` files were inspected without
running model evaluation. All eight files have a label column and can safely
map the older CICFlowMeter names for 29 of V13's 30 required features. However,
every supplied CSV lacks `Protocol`.

Do not impute a value for `Protocol` or run `final-evaluate` with these files:
the frozen V13 artifact was trained using that feature, so doing so would make
the reported result invalid.

Valid paths forward are:

1. Generate a new independent capture in a controlled environment with a
   CICFlowMeter export that includes all V13 features, including `Protocol`.
2. Obtain the CIC-IDS2017 PCAPs and regenerate compatible flows with a
   documented CICFlowMeter process and reliable labels.
3. Create a separately documented cross-dataset model that deliberately omits
   `Protocol`; that would be a new candidate, not V13's final evaluation.

The header-only report is at
`models/evaluation/xgb_v13_cicids2017_preflight/schema_preflight.csv`.
