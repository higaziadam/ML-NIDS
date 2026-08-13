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

## Performance Comparison

| Version | Model Type | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Date | Notes |
|---------|------------|----------|-----------|--------|----------|---------|------|-------|
| V1 | Random Forest | 0.9798 | 0.9773 | 0.9243 | 0.9501 | 0.9893 | 2026-08-12 | Baseline; 0.56% false-positive rate |
| V2 | XGBoost | 0.9809 | 0.9912 | 0.9167 | 0.9525 | 0.9905 | 2026-08-12 | Lowest FPR (0.21%); recall needs tuning |
| V3 | XGBoost | 0.9771 | 0.9638 | 0.9247 | 0.9438 | 0.9900 | 2026-08-12 | Higher recall; FPR rose to 0.92% |

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
- [ ] Implement cross-validation
- [ ] Test hyperparameter tuning (GridSearchCV)
- [ ] Apply SMOTE for class imbalance
- [ ] Feature importance analysis
- [ ] Neural network baseline
- [ ] Compare with production NIDS systems

---

**Last Updated**: 2026-08-12  
**Total Experiments**: 3
