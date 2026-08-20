# ML-NIDS Experiments & Results Log

## Purpose
Track all model experiments, hyperparameter tuning, and results for reproducibility and comparison.

---

## Experiment Template

```
### Experiment V1: [Brief Description]

**Date**: YYYY-MM-DD  
**Model**: [random_forest/gradient_boosting/svm]  
**Dataset**: [CICIDS2018/NSL-KDD/etc.]  
**Dataset Size**: [Number of samples]  

#### Hyperparameters
- n_estimators: 100
- max_depth: 20
- learning_rate: 0.1
- [others...]

#### Data Preprocessing
- Normalization: [minmax/zscore/robust]
- Feature Selection: [Yes/No, method if yes]
- Class Balancing: [None/SMOTE/Other]
- Train/Test Split: 80/20

#### Results
- Accuracy: 0.XXXX
- Precision: 0.XXXX
- Recall: 0.XXXX
- F1-Score: 0.XXXX
- ROC-AUC: 0.XXXX
- Training Time: XXX seconds
- Memory Used: XXX MB

#### Confusion Matrix
```
                 Predicted Benign    Predicted Attack
Actual Benign    TN                  FP
Actual Attack    FN                  TP
```

#### Notes
- What worked well
- What could be improved
- Observations about the data
- Next steps

#### Model Location
`models/saved/nids_v1.pkl`

#### Evaluation Report
`models/evaluation/nids_v1_metrics.json`

---

## Experiments

### Experiment V1: Random Forest Baseline

**Date**: 2026-08-12  
**Model**: Random Forest  
**Dataset**: CICIDS2018, binary classification (0 = benign, 1 = attack)  
**Dataset Size**: 1,018,036 evaluation samples  

#### Hyperparameters
- n_estimators: 100
- max_depth: 20
- min_samples_split: 5
- min_samples_leaf: 2
- max_features: sqrt
- class_weight: balanced
- random_state: 42
- n_jobs: -1

#### Data Preprocessing
- Removed duplicate records and constant features.
- Retained statistical outliers so legitimate attack traffic was not discarded.
- Used feature selection and MinMax scaling fitted on training data only.
- Used a stratified 80/20 train-test split.
- Class Balancing: Random Forest `class_weight="balanced"`.

#### Results
- Accuracy: 0.9798
- Precision: 0.9773
- Recall (attack): 0.9243
- F1-Score: 0.9501
- ROC-AUC: 0.9893
- PR-AUC: 0.9774
- Specificity: 0.9944
- False Positive Rate: 0.0056
- False Negative Rate: 0.0757

#### Confusion Matrix

```
                 Predicted Benign    Predicted Attack
Actual Benign          801,217                 4,548
Actual Attack           16,064               196,207
```

#### Notes
- The baseline achieved strong discrimination and a very low false-positive rate,
  which helps limit alert fatigue.
- It missed 16,064 attacks (7.57% of attack samples); improving attack recall is
  the primary objective for the XGBoost comparison.
- Compare future models using the same split and preprocessing settings.

#### Model Location
`models/saved/baseline_rf.pkl`

#### Evaluation Report
`models/saved/baseline_rf_results/metrics.csv`

---

### Experiment V2: XGBoost Baseline Comparison

**Date**: 2026-08-12  
**Model**: XGBoost  
**Dataset**: CICIDS2018, binary classification (0 = benign, 1 = attack)  
**Dataset Size**: 1,018,036 evaluation samples  

#### Hyperparameters
- n_estimators: 100
- max_depth: 20
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.8
- min_child_weight: 1.0
- reg_lambda: 1.0
- scale_pos_weight: 1.0
- random_state: 42
- n_jobs: -1
- tree_method: hist

#### Data Preprocessing
- Removed duplicate records and constant features.
- Retained statistical outliers so legitimate attack traffic was not discarded.
- Used feature selection and MinMax scaling fitted on training data only.
- Used a stratified 80/20 train-test split.
- Class Balancing: None beyond the model's default `scale_pos_weight=1.0`.

#### Results
- Accuracy: 0.9809
- Precision: 0.9912
- Recall (attack): 0.9167
- F1-Score: 0.9525
- ROC-AUC: 0.9905
- PR-AUC: 0.9783
- Specificity: 0.9979
- False Positive Rate: 0.0021
- False Negative Rate: 0.0833

#### Confusion Matrix

```
                 Predicted Benign    Predicted Attack
Actual Benign          804,044                 1,721
Actual Attack           17,692               194,579
```

#### Notes
- XGBoost improved precision, F1-score, ROC-AUC, PR-AUC, specificity, and the
  false-positive rate compared with the Random Forest baseline.
- False positives decreased from 4,548 to 1,721, reducing potential alert
  fatigue substantially.
- Attack recall decreased from 0.9243 to 0.9167, resulting in 1,628 additional
  missed attacks. The next experiment should tune XGBoost for higher recall.

#### Model Location
`models/saved/baseline_xgb.pkl`

#### Evaluation Report
`models/saved/baseline_xgb_results/metrics.csv`

---

### Experiment V3: XGBoost Recall Trade-off

**Date**: 2026-08-12  
**Model**: XGBoost  
**Dataset**: CICIDS2018, binary classification (0 = benign, 1 = attack)  
**Dataset Size**: 1,018,036 evaluation samples  

#### Results
- Accuracy: 0.9771
- Precision: 0.9638
- Recall (attack): 0.9247
- F1-Score: 0.9438
- ROC-AUC: 0.9900
- PR-AUC: 0.9775
- Specificity: 0.9908
- False Positive Rate: 0.0092
- False Negative Rate: 0.0753

#### Confusion Matrix

```
                 Predicted Benign    Predicted Attack
Actual Benign          798,390                 7,375
Actual Attack           15,987               196,284
```

#### Notes
- Attack recall improved from 0.9167 in V2 to 0.9247, detecting 1,705 more
  attack samples.
- This required a substantial trade-off: false positives increased from 1,721
  to 7,375 (+5,654), and the false-positive rate increased from 0.21% to 0.92%.
- Precision, F1-score, ROC-AUC, and PR-AUC were all lower than V2.
- Keep V2 as the preferred low-alert baseline. Retain V3 as the recall-focused
  candidate when missed attacks are more costly than additional analyst alerts.

#### Model Location
`models/saved/xgb_recall_v3.pkl`

#### Evaluation Report
`models/saved/xgb_recall_v3_results/metrics.csv`

---

### Experiment V4: XGBoost Recall Optimization

**Date**: 2026-08-13
**Model**: XGBoost
**Dataset**: CICIDS2018, binary classification (0 = benign, 1 = attack)
**Dataset Size**: 1,018,036 evaluation samples

#### Results
- Accuracy: 0.9801
- Precision: 0.9835
- Recall (attack): 0.9198
- F1-Score: 0.9506
- ROC-AUC: 0.9902
- PR-AUC: 0.9779
- Specificity: 0.9959
- False Positive Rate: 0.0041
- False Negative Rate: 0.0802

#### Confusion Matrix

```
                 Predicted Benign    Predicted Attack
Actual Benign          802,499                 3,266
Actual Attack           17,014               195,257
```

#### Notes
- This is an intermediate recall-versus-alert-volume operating point between
  V2 and V3.
- Compared with V2, it detected 678 additional attacks (17,014 versus 17,692
  false negatives) but generated 1,545 additional false alerts (3,266 versus
  1,721 false positives).
- Compared with V3, it reduced false positives by 4,109 while accepting 1,027
  more missed attacks.
- V2 remains the best low-alert option; V4 is a reasonable compromise when a
  modest recall increase is worth a false-positive rate of 0.41%.

#### Model Location
`models/saved/xgb_recall_v4.pkl`

#### Evaluation Report
`models/saved/xgb_recall_v4_results/metrics.csv`

---

### Experiment V5: Validation-Based Threshold Tuning

**Status**: Completed.
**Date**: 2026-08-13
**Model**: XGBoost
**Dataset**: CICIDS2018, binary classification (0 = benign, 1 = attack)

#### Hyperparameters
- n_estimators: 100
- max_depth: 20
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.8
- min_child_weight: 1.0
- reg_lambda: 1.0
- scale_pos_weight: 1.0
- random_state: 42
- n_jobs: -1
- tree_method: hist

#### Methodology
- Uses a stratified 70% training, 15% validation, and 15% final-test split.
- Fits data cleaning decisions, feature selection, and MinMax scaling only on
  training data; validation and test data use the saved training schema.
- Trains XGBoost on the training partition, then evaluates attack-probability
  thresholds of 0.40, 0.45, 0.50, and 0.55 on validation data.
- Selects the highest threshold meeting recall >= 0.92 and false-positive rate
  <= 0.5%; if none meet both targets, selects the best recall/FPR trade-off and
  records that outcome.
- Evaluates the selected threshold once on the untouched final test partition.
- Persists the selected threshold, threshold policy, model version, fitted
  preprocessor, and feature schema with the trained model artifact.

#### Validation Threshold Selection

| Threshold | Precision | Recall | False-Positive Rate | False Positives | False Negatives | Meets Targets |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.40 | 0.9874 | 0.9178 | 0.0031 | 1,868 | 13,086 | No |
| 0.45 | 0.9898 | 0.9166 | 0.0025 | 1,498 | 13,282 | No |
| 0.50 | 0.9917 | 0.9157 | 0.0020 | 1,226 | 13,415 | No |
| 0.55 | 0.9967 | 0.9112 | 0.0008 | 478 | 14,135 | No |

No candidate met both the recall target (>= 0.92) and the FPR target (<= 0.5%).
The workflow therefore selected **0.40**, the candidate with the highest
validation recall and an FPR below the policy limit.

#### Final Test Results
- Selected threshold: 0.40
- Accuracy: 0.9806
- Precision: 0.9872
- Recall (attack): 0.9189
- F1-Score: 0.9518
- ROC-AUC: 0.9905
- PR-AUC: 0.9784
- Specificity: 0.9969
- False Positive Rate: 0.0031
- False Negative Rate: 0.0811

#### Confusion Matrix

```
                 Predicted Benign    Predicted Attack
Actual Benign          602,421                 1,903
Actual Attack           12,907               146,296
```

#### Notes
- The validation decision generalizes closely to the untouched test split:
  recall was 0.9178 on validation and 0.9189 on test, while FPR was 0.31% on
  both splits.
- V5 is a threshold-policy experiment, not a direct one-to-one comparison with
  V1-V4, because it uses a different 70/15/15 split rather than an 80/20 split.
- It did not reach the 92% recall target. The next iteration should broaden the
  threshold candidates below 0.40 or adjust XGBoost training parameters, while
  selecting against the same validation policy.

#### Model Location
`models/saved/xgb_v5_threshold_tuning.pkl`

#### Evaluation Report
`models/saved/xgb_v5_threshold_tuning_results/metrics.csv`

---

### Experiment V6: Expanded Threshold Search

**Date**: 2026-08-13
**Model**: XGBoost
**Dataset**: CICIDS2018, binary classification (0 = benign, 1 = attack)

#### Results
- Selected threshold: 0.30
- Accuracy: 0.9797
- Precision: 0.9798
- Recall (attack): 0.9214
- F1-Score: 0.9497
- ROC-AUC: 0.9905
- PR-AUC: 0.9784
- False Positive Rate: 0.0050
- False Negative Rate: 0.0786

#### Confusion Matrix

```
                 Predicted Benign    Predicted Attack
Actual Benign          601,303                 3,021
Actual Attack           12,513               146,690
```

#### Notes
- Expanded threshold search identified 0.30 as the only tested validation
  threshold meeting both policy targets: recall >= 0.92 and FPR <= 0.5%.
- This is the first threshold-tuned run to satisfy the defined alert policy.

#### Model Location
`models/saved/xgb_v6_expanded_thresholds.pkl`

#### Evaluation Report
`models/saved/xgb_v6_expanded_thresholds_results/metrics.csv`

---

### Experiment V7: XGBoost Class-Weight Tuning

**Date**: 2026-08-13
**Model**: XGBoost
**Dataset**: CICIDS2018, binary classification (0 = benign, 1 = attack)

#### Changed Configuration
- scale_pos_weight: 1.25 (V6 used 1.0)
- All other XGBoost settings and the 70/15/15 threshold-tuning workflow were retained.

#### Results
- Selected threshold: 0.20
- Accuracy: 0.9744
- Precision: 0.9478
- Recall (attack): 0.9284
- F1-Score: 0.9380
- ROC-AUC: 0.9905
- PR-AUC: 0.9783
- False Positive Rate: 0.0135
- False Negative Rate: 0.0716

#### Confusion Matrix

```
                 Predicted Benign    Predicted Attack
Actual Benign          596,190                 8,134
Actual Attack           11,398               147,805
```

#### Notes
- Increasing class weight improved recall by 0.70 percentage points versus V6
  and detected 1,115 additional attacks.
- It increased false positives by 5,113 and failed the 0.5% FPR policy target.
- Retain as a recall-focused comparison; V6 remains the preferred policy-compliant model.

#### Model Location
`models/saved/xgb_v7_scale_weight_125.pkl`

#### Evaluation Report
`models/saved/xgb_v7_scale_weight_125_results/metrics.csv`

---

### Experiment V8: Fine Threshold Search with Class Weight 1.25

**Date**: 2026-08-13
**Model**: XGBoost
**Dataset**: CICIDS2018, binary classification (0 = benign, 1 = attack)

#### Changed Configuration
- Retained V7's `scale_pos_weight=1.25`.
- Searched fine-grained validation thresholds from 0.30 to 0.35.

#### Results
- Selected threshold: 0.30
- Accuracy: 0.9788
- Precision: 0.9744
- Recall (attack): 0.9226
- F1-Score: 0.9478
- ROC-AUC: 0.9905
- PR-AUC: 0.9783
- False Positive Rate: 0.0064
- False Negative Rate: 0.0774

#### Confusion Matrix

```
                 Predicted Benign    Predicted Attack
Actual Benign          600,465                 3,859
Actual Attack           12,319               146,884
```

#### Notes
- Fine threshold search improved the V7 alert volume substantially, reducing
  false positives from 8,134 to 3,859 while retaining recall above 92%.
- No tested threshold met both policy targets. At 0.33, validation FPR was
  0.522%—close to the 0.5% limit—but recall was 92.04%; at 0.35, FPR passed
  at 0.488% but recall dropped to 91.99%.
- V8 is close to policy compliance but does not outperform V6, which already
  meets both targets with fewer false positives.

#### Model Location
`models/saved/xgb_v8_fine_threshold.pkl`

#### Evaluation Report
`models/saved/xgb_v8_fine_threshold_results/metrics.csv`

---

### Experiment V9: Regularized XGBoost (Depth 10)

**Date**: 2026-08-13
**Model**: XGBoost
**Dataset**: CICIDS2018, binary classification (0 = benign, 1 = attack)

#### Changed Configuration
- n_estimators: 300
- max_depth: 10
- learning_rate: 0.05
- min_child_weight: 3.0
- scale_pos_weight: 1.0
- Retained `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, and the
  training-only preprocessing and 70/15/15 split workflow.

#### Validation Threshold Selection

| Threshold | Precision | Recall | False-Positive Rate | False Positives | False Negatives | Meets Targets |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.30 | 0.9865 | 0.9189 | 0.0033 | 1,995 | 12,919 | No |
| 0.31 | 0.9872 | 0.9187 | 0.0031 | 1,899 | 12,951 | No |
| 0.32 | 0.9886 | 0.9181 | 0.0028 | 1,679 | 13,035 | No |
| 0.33 | 0.9889 | 0.9179 | 0.0027 | 1,633 | 13,067 | No |
| 0.34 | 0.9894 | 0.9176 | 0.0026 | 1,564 | 13,111 | No |
| 0.35 | 0.9897 | 0.9175 | 0.0025 | 1,516 | 13,140 | No |

None of the validation candidates reached the recall target of 0.92, so the
fallback policy selected threshold **0.30** for its highest validation recall.

#### Final Test Results
- Selected threshold: 0.30
- Accuracy: 0.9807
- Precision: 0.9866
- Recall (attack): 0.9201
- F1-Score: 0.9522
- ROC-AUC: 0.9906
- PR-AUC: 0.9787
- Specificity: 0.9967
- False Positive Rate: 0.0033
- False Negative Rate: 0.0799

#### Confusion Matrix

```
                 Predicted Benign    Predicted Attack
Actual Benign          602,328                 1,996
Actual Attack           12,723               146,480
```

#### Notes
- V9 improved on V6 in accuracy, precision, F1-score, ROC-AUC, PR-AUC, and
  false-positive rate.
- Compared with V6, it produced 1,025 fewer false alerts (1,996 versus 3,021)
  while missing 210 additional attacks (12,723 versus 12,513).
- The final test recall exceeded 0.92, but the threshold was selected from
  validation data where it narrowly missed the recall target; therefore V9 is
  promising but not strictly policy-compliant under the predeclared selection
  rule. Retain V6 as the policy-compliant reference.

#### Model Location
`models/saved/xgb_v9_regularized_depth10.pkl`

#### Evaluation Report
`models/saved/xgb_v9_regularized_depth10_results/metrics.csv`

---

### Experiment V10: Regularized XGBoost Fine Threshold Selection

**Date**: 2026-08-13
**Model**: XGBoost
**Dataset**: CICIDS2018, binary classification (0 = benign, 1 = attack)

#### Changed Configuration
- Retained V9's regularized model: `n_estimators=300`, `max_depth=10`,
  `learning_rate=0.05`, `min_child_weight=3.0`, and `scale_pos_weight=1.0`.
- Narrowed the validation threshold search to 0.24--0.30 to locate a
  policy-compliant operating point near V9's boundary.
- Retained training-only preprocessing and the stratified 70/15/15
  train/validation/test split.

#### Validation Threshold Selection

| Threshold | Precision | Recall | False-Positive Rate | False Positives | False Negatives | Meets Targets |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.24 | 0.9805 | 0.9210 | 0.0048 | 2,917 | 12,579 | Yes |
| 0.25 | 0.9819 | 0.9205 | 0.0045 | 2,702 | 12,654 | Yes |
| 0.26 | 0.9834 | 0.9200 | 0.0041 | 2,471 | 12,731 | Yes |
| 0.27 | 0.9846 | 0.9196 | 0.0038 | 2,293 | 12,799 | No |
| 0.28 | 0.9855 | 0.9193 | 0.0036 | 2,160 | 12,850 | No |
| 0.29 | 0.9860 | 0.9190 | 0.0034 | 2,073 | 12,896 | No |
| 0.30 | 0.9865 | 0.9189 | 0.0033 | 1,995 | 12,919 | No |

Thresholds 0.24--0.26 met both validation targets (recall >= 0.92 and
false-positive rate <= 0.005). Per the selection policy, threshold **0.26**
was chosen because it is the highest qualifying threshold.

#### Final Test Results
- Selected threshold: 0.26
- Accuracy: 0.9803
- Precision: 0.9835
- Recall (attack): 0.9211
- F1-Score: 0.9513
- ROC-AUC: 0.9906
- PR-AUC: 0.9787
- Specificity: 0.9959
- False Positive Rate: 0.0041
- False Negative Rate: 0.0789

#### Confusion Matrix

```
                 Predicted Benign    Predicted Attack
Actual Benign          601,868                 2,456
Actual Attack           12,563               146,640
```

#### Notes
- V10 is policy-compliant on validation data and confirms that the V9 model can
  satisfy the recall and false-positive-rate requirements at threshold 0.26.
- Compared with V6, V10 generated 565 fewer false alerts (2,456 versus 3,021)
  while missing only 50 additional attacks (12,563 versus 12,513).
- V10 improves accuracy, precision, F1-score, and false-positive rate over V6;
  V6 retains a marginally higher attack recall.

#### Model Location
`models/saved/xgb_v10_regularized_fine_threshold.pkl`

#### Evaluation Report
`models/saved/xgb_v10_regularized_fine_threshold_results/metrics.csv`

#### Post-Freeze Validation

The frozen V10 profile was evaluated with five-fold nested stratified
cross-validation on development data. Every fold used the saved deployment
threshold of `0.26`; the outer-fold data was not used to select a threshold.

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Accuracy | 0.9803 | 0.0002 |
| Precision | 0.9834 | 0.0001 |
| Recall (attack) | 0.9209 | 0.0008 |
| F1-Score | 0.9511 | 0.0004 |
| ROC-AUC | 0.9905 | 0.0001 |
| PR-AUC | 0.9784 | 0.0002 |
| False-Positive Rate | 0.00410 | 0.00003 |

- The fixed operating point is stable: FPR ranged from 0.405% to 0.415% across
  folds, remaining under the 0.50% limit in all five folds.
- Recall ranged from 91.95% to 92.14%. Four of five folds satisfied the strict
  92.00% recall target; the remaining fold was 0.05 percentage points below it.
- The independent final-holdout evaluation at the same threshold achieved
  91.91% recall and 0.41% FPR (244,577 true positives, 21,528 false negatives,
  and 4,379 false positives). It passed the FPR target but missed the recall
  target by 0.09 percentage points.
- Therefore, V10 remains a frozen low-alert comparison candidate, but it is not
  promoted as a fully policy-compliant release under the current rule that both
  recall >= 0.92 and FPR <= 0.005 must be met on final holdout.

#### Post-Freeze Validation Reports

- `models/evaluation/xgb_v10_fixed_threshold_cv/fixed_threshold_summary.csv`
- `models/evaluation/xgb_v10_fixed_threshold_cv/fixed_threshold_fold_metrics.csv`
- `models/evaluation/xgb_v10_outer_holdout/metrics.csv`

#### Development-Only Diagnostic Analysis

Five-fold V10 cross-validation was rerun with out-of-fold diagnostics enabled.
This analysis used `train_data.csv` only; it did not access the final holdout.

- All 20 selected features appeared in all five folds, showing strong feature
  selection stability. The most influential features were `Fwd Seg Size Min`,
  `Init Fwd Win Bytes`, `Fwd Packet Length Max`, `Init Bwd Win Bytes`, and
  `RST Flag Count`.
- The outer-fold predictions contained 83,960 false negatives (1.65% of all
  development samples) and 16,516 false positives (0.32%).
- False negatives had a median attack probability of 0.0647, far below the
  0.26 operating threshold; only 2,819 (3.36%) were within 0.05 of it.
  Lowering the threshold alone is therefore unlikely to recover enough missed
  attacks to satisfy the recall target.
- False positives had a median attack probability of 0.4104. Only 4,022
  (24.35%) were within 0.05 of the threshold, indicating that many false alerts
  are also high-confidence model decisions rather than borderline cases.
- The evidence supports a targeted feature/model experiment for V13 instead of
  further threshold-only or class-weight-only tuning.

#### Diagnostic Reports

- `models/evaluation/xgb_v10_diagnostics_cv/feature_importance_summary.csv`
- `models/evaluation/xgb_v10_diagnostics_cv/out_of_fold_error_summary.csv`
- `models/evaluation/xgb_v10_diagnostics_cv/out_of_fold_predictions.csv`

---

### Experiment V11: Regularized XGBoost with Moderate Class Weight

**Date**: 2026-08-13
**Model**: XGBoost
**Dataset**: CICIDS2018, binary classification (0 = benign, 1 = attack)

#### Changed Configuration
- Retained V10's regularized model: `n_estimators=300`, `max_depth=10`,
  `learning_rate=0.05`, and `min_child_weight=3.0`.
- Increased `scale_pos_weight` moderately from 1.0 to **1.10** to emphasize
  attack samples without using V7's more aggressive weight of 1.25.
- Searched validation thresholds from 0.24 to 0.35; retained the training-only
  preprocessing and stratified 70/15/15 train/validation/test split.

#### Validation Threshold Selection

| Threshold | Precision | Recall | False-Positive Rate | False Positives | False Negatives | Meets Targets |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.24 | 0.9777 | 0.9218 | 0.0055 | 3,344 | 12,450 | No |
| 0.25 | 0.9790 | 0.9214 | 0.0052 | 3,146 | 12,512 | No |
| 0.26 | 0.9810 | 0.9208 | 0.0047 | 2,837 | 12,607 | Yes |
| 0.27 | 0.9824 | 0.9204 | 0.0043 | 2,619 | 12,680 | Yes |
| 0.28 | 0.9835 | 0.9199 | 0.0041 | 2,458 | 12,747 | No |
| 0.29 | 0.9845 | 0.9196 | 0.0038 | 2,299 | 12,806 | No |
| 0.30 | 0.9853 | 0.9193 | 0.0036 | 2,180 | 12,843 | No |
| 0.31 | 0.9861 | 0.9190 | 0.0034 | 2,063 | 12,888 | No |
| 0.32 | 0.9867 | 0.9188 | 0.0033 | 1,966 | 12,927 | No |
| 0.33 | 0.9872 | 0.9186 | 0.0031 | 1,901 | 12,967 | No |
| 0.34 | 0.9886 | 0.9180 | 0.0028 | 1,681 | 13,058 | No |
| 0.35 | 0.9890 | 0.9178 | 0.0027 | 1,623 | 13,082 | No |

Thresholds 0.26 and 0.27 met both validation targets (recall >= 0.92 and
false-positive rate <= 0.005). Per the selection policy, threshold **0.27**
was chosen because it is the highest qualifying threshold.

#### Final Test Results
- Selected threshold: 0.27
- Accuracy: 0.9802
- Precision: 0.9826
- Recall (attack): 0.9215
- F1-Score: 0.9511
- ROC-AUC: 0.9906
- PR-AUC: 0.9787
- Specificity: 0.9957
- False Positive Rate: 0.0043
- False Negative Rate: 0.0785

#### Confusion Matrix

```
                 Predicted Benign    Predicted Attack
Actual Benign          601,732                 2,592
Actual Attack           12,504               146,699
```

#### Notes
- V11 is policy-compliant: the selected threshold met both targets on
  validation data and the independent test results also satisfy them.
- Relative to V10, V11 detected 59 additional attacks (fewer false negatives)
  but produced 136 more false alerts. Its validation recall margin above the
  0.92 target is larger, while its false-positive-rate margin is smaller.
- Choose V11 when slightly higher attack detection is worth a small increase
  in alerts; retain V10 when minimizing false positives is the priority.

#### Model Location
`models/saved/xgb_v11_scale_weight_110.pkl`

#### Evaluation Report
`models/saved/xgb_v11_scale_weight_110_results/metrics.csv`

---

### Experiment V12: Pre-Registered Moderate Recall Candidate

**Date**: 2026-08-20
**Model**: XGBoost
**Dataset**: CICIDS2018, binary classification (0 = benign, 1 = attack)

#### Controlled Change
- Retains V10's preprocessing, feature selection, XGBoost architecture, random
  seed, and policy targets.
- Changes only `scale_pos_weight` from `1.00` to **`1.05`**.
- Pre-registers threshold `0.26` for fixed-threshold cross-validation, while
  separately recording the threshold selected from each fold's inner validation
  data.

#### Hypothesis
- The modest additional positive-class weight may increase attack recall without
  exceeding the 0.50% false-positive-rate limit.

#### Evaluation Protocol
- Run five-fold nested cross-validation on `data/processed/train_data.csv` only.
- Do not use the consumed V10 final holdout for any V12 tuning or comparison.
- Freeze V12 only after reviewing its development-data cross-validation report;
  then use a new independent holdout or external traffic dataset once.

#### Candidate Profile
`models/configs/xgb_v12_scale_weight_105_candidate.json`

#### Status
Rejected after development-data cross-validation — not evaluated on a final
holdout.

#### Fixed-Threshold Cross-Validation Results

All five outer folds used the pre-registered threshold of `0.26`.

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Accuracy | 0.9801 | 0.0002 |
| Precision | 0.9823 | 0.0003 |
| Recall (attack) | 0.9212 | 0.0008 |
| F1-Score | 0.9508 | 0.0004 |
| ROC-AUC | 0.9904 | 0.0001 |
| PR-AUC | 0.9784 | 0.0002 |
| False-Positive Rate | 0.00436 | 0.00008 |

#### Comparison with Frozen V10

- V12 increased mean attack recall by 0.03 percentage points (92.09% to
  92.12%), or roughly 65 additional attacks detected per outer fold.
- This trade-off added roughly 211 false alerts per outer fold: mean FPR rose
  from 0.410% to 0.436%, while precision and F1-score both decreased.
- Four of five folds met both policy targets. Fold 1 recall was 91.97%, so V12
  still lacks a reliable recall margin above the 92.00% requirement.
- The small recall gain does not justify the additional alert volume or reduced
  FPR margin. V12 is rejected; it must not be evaluated on a final holdout.

#### Evaluation Reports

- `models/evaluation/xgb_v12_scale_weight_105_cv/fixed_threshold_summary.csv`
- `models/evaluation/xgb_v12_scale_weight_105_cv/fixed_threshold_fold_metrics.csv`

---

### Experiment V13: Pre-Registered Feature-Set Expansion Candidate

**Date**: 2026-08-20
**Model**: XGBoost
**Dataset**: CICIDS2018, binary classification (0 = benign, 1 = attack)

#### Controlled Change
- Retains V10's preprocessing policy, XGBoost hyperparameters, random seed,
  `scale_pos_weight=1.0`, and threshold policy.
- Expands `f_classif` feature selection from **20 to 30 features**.
- Pre-registers threshold `0.26` for fixed-threshold cross-validation while
  separately recording each fold's validation-selected threshold.

#### Hypothesis
- V10's false negatives were generally far below the decision threshold, so
  additional discriminative flow features may improve their attack probability
  without relying on a lower threshold or a higher class weight.

#### Evaluation Protocol
- Run five-fold nested cross-validation on `data/processed/train_data.csv` only.
- Compare fixed-threshold V13 results against fixed-threshold V10 results.
- Do not use V10's consumed final holdout for V13 design, tuning, or comparison.
- Freeze V13 only if it provides a meaningful recall improvement while retaining
  FPR <= 0.005 and low fold-to-fold variation.

#### Candidate Profile
`models/configs/xgb_v13_feature30_candidate.json`

#### Status
Frozen release candidate after development-only cross-validation; not evaluated
on a final holdout.

#### Fixed-Threshold Cross-Validation Results

All five outer folds used the frozen threshold of `0.26` and met both policy
targets (recall >= 0.92 and FPR <= 0.005).

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Accuracy | 0.9808 | 0.0002 |
| Precision | 0.9853 | 0.0002 |
| Recall (attack) | 0.9216 | 0.0007 |
| F1-Score | 0.9524 | 0.0004 |
| ROC-AUC | 0.9914 | 0.0001 |
| PR-AUC | 0.9803 | 0.0002 |
| False-Positive Rate | 0.00362 | 0.00004 |

#### Comparison with Frozen V10

- V13 increased mean recall by 0.07 percentage points (92.09% to 92.16%) and
  reduced mean FPR by 0.05 percentage points (0.410% to 0.362%).
- Across the outer-fold predictions, V13 detected 790 more attacks and produced
  1,942 fewer false alerts than V10 at the same threshold.
- Accuracy, precision, F1-score, ROC-AUC, and PR-AUC all improved. The
  improvements were stable across folds, and V13 met both targets in all five
  fixed-threshold outer-fold evaluations.
- Freeze V13 as the preferred release candidate. Its next evaluation must use a
  new independent holdout or external traffic dataset; V10's holdout is not
  eligible for reuse.

#### Evaluation Reports

- `models/evaluation/xgb_v13_feature30_cv/fixed_threshold_summary.csv`
- `models/evaluation/xgb_v13_feature30_cv/fixed_threshold_fold_metrics.csv`
- `models/evaluation/xgb_v13_feature30_cv/feature_importance_summary.csv`
- `models/evaluation/xgb_v13_feature30_cv/out_of_fold_error_summary.csv`

---

## Performance Comparison

| Version | Model Type | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Date | Notes |
|---------|------------|----------|-----------|--------|----------|---------|------|-------|
| V1 | Random Forest | 0.9798 | 0.9773 | 0.9243 | 0.9501 | 0.9893 | 2026-08-12 | Baseline; 0.56% false-positive rate |
| V2 | XGBoost | 0.9809 | 0.9912 | 0.9167 | 0.9525 | 0.9905 | 2026-08-12 | Lowest FPR (0.21%); recall needs tuning |
| V3 | XGBoost | 0.9771 | 0.9638 | 0.9247 | 0.9438 | 0.9900 | 2026-08-12 | Higher recall; FPR rose to 0.92% |
| V4 | XGBoost | 0.9801 | 0.9835 | 0.9198 | 0.9506 | 0.9902 | 2026-08-13 | Balanced compromise; FPR 0.41% |
| V5 | XGBoost | 0.9806 | 0.9872 | 0.9189 | 0.9518 | 0.9905 | 2026-08-13 | Threshold 0.40; 0.31% FPR; target recall not met |
| V6 | XGBoost | 0.9797 | 0.9798 | 0.9214 | 0.9497 | 0.9905 | 2026-08-13 | Policy-compliant: threshold 0.30, FPR 0.50% |
| V7 | XGBoost | 0.9744 | 0.9478 | 0.9284 | 0.9380 | 0.9905 | 2026-08-13 | Higher recall; FPR 1.35%, not compliant |
| V8 | XGBoost | 0.9788 | 0.9744 | 0.9226 | 0.9478 | 0.9905 | 2026-08-13 | Fine search; FPR 0.64%, still not compliant |
| V9 | XGBoost | 0.9807 | 0.9866 | 0.9201 | 0.9522 | 0.9906 | 2026-08-13 | Lowest FPR (0.33%); validation recall target narrowly missed |
| V10 | XGBoost | 0.9803 | 0.9835 | 0.9211 | 0.9513 | 0.9906 | 2026-08-13 | Frozen threshold 0.26; final holdout FPR passed but recall was 91.91% |
| V11 | XGBoost | 0.9802 | 0.9826 | 0.9215 | 0.9511 | 0.9906 | 2026-08-13 | Policy-compliant: threshold 0.27, FPR 0.43% |
| V12 | XGBoost (5-fold CV) | 0.9801 | 0.9823 | 0.9212 | 0.9508 | 0.9904 | 2026-08-20 | Rejected: marginal recall gain; FPR increased to 0.44% |
| V13 | XGBoost (5-fold CV) | 0.9808 | 0.9853 | 0.9216 | 0.9524 | 0.9914 | 2026-08-20 | Frozen candidate; all fixed-threshold folds policy-compliant, FPR 0.36% |

---

## Key Insights

- **Best Model**: [Model type and why]
- **Worst Model**: [Model type and why]
- **Dataset Challenges**: [Class imbalance, missing values, etc.]
- **Feature Importance**: [Top 5 features]
- **Bottlenecks**: [Training time, memory, etc.]

---

## Future Experiments

- [ ] Try ensemble methods (stacking, voting)
- [x] Implement cross-validation
- [ ] Test hyperparameter tuning (GridSearchCV)
- [ ] Apply SMOTE for class imbalance
- [ ] Feature importance analysis
- [ ] Neural network baseline
- [ ] Compare with production NIDS systems

---

**Last Updated**: 2026-08-20
**Total Experiments**: 13
