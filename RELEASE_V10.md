# V10 XGBoost Release Candidate

V10 is the frozen low-alert XGBoost candidate for this project. It is not a
production release. Its fixed-threshold cross-validation and one final-holdout
evaluation are complete, but the final holdout did not meet the strict recall
target required for promotion.

## Release identity

- Profile: `models/configs/xgb_v10_candidate.json`
- Artifact: `models/saved/xgb_v10_regularized_fine_threshold.pkl`
- Artifact SHA-256:
  `8C736F8FD2327C1F555D69A89E977B0BE23335835D0932341672AB270BEE9D97`
- Decision threshold: `0.26`
- Validation policy: recall >= `0.92`; false-positive rate <= `0.005`

## Evaluation status

The recorded V10 test result was 92.11% attack recall and a 0.41% false-positive
rate. This result is useful for experiment comparison, but it is not the final
unbiased deployment estimate because earlier experiments were compared on the
same test partition.

Five-fold fixed-threshold cross-validation evaluated the frozen threshold of
`0.26` without changing it per fold. Mean recall was 92.09% (standard deviation
0.08 percentage points) and mean FPR was 0.410% (standard deviation 0.003
percentage points). FPR met the 0.50% limit in every fold; four of five folds
met the 92.00% recall target.

The single final-holdout evaluation at threshold `0.26` produced 91.91% attack
recall and a 0.411% FPR. It met the FPR requirement but missed the recall
requirement by 0.09 percentage points. V10 must therefore remain a frozen
comparison artifact rather than being promoted under the current policy.

Reports:

- `models/evaluation/xgb_v10_fixed_threshold_cv/fixed_threshold_summary.csv`
- `models/evaluation/xgb_v10_fixed_threshold_cv/fixed_threshold_fold_metrics.csv`
- `models/evaluation/xgb_v10_outer_holdout/metrics.csv`

## Reproduce without overwriting V10

Install the pinned runtime:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-v10.txt
```

Train a separate reproduction artifact using the frozen profile:

```powershell
.\venv\Scripts\python.exe -m src.train `
  --config models\configs\xgb_v10_candidate.json `
  --data data\processed\train_data.csv `
  --name xgb_v10_reproduction
```

The training command refuses to overwrite existing model, result, metadata, or
split paths unless `--overwrite` is explicitly supplied. Do not use
`--overwrite` with the canonical V10 artifact.

Each saved run writes `<model_name>_metadata.json` beside the artifact. It
records source/profile checksums, selected threshold, split row counts, package
versions, and the Git commit.

## Final holdout and cross-validation workflow

Use this workflow only with a new, single labeled source dataset. Do not combine
or re-split earlier experiment outputs after observing their metrics.

First create a source dataset without learned preprocessing. This can require
substantial disk space because the CICIDS2018 source is large:

```powershell
.\venv\Scripts\python.exe scripts\prepare_kaggle_data.py `
  --single-source `
  --output-file cicids2018_labeled.csv
```

Create one development/final-holdout split. The command refuses to replace an
existing holdout unless `--overwrite` is supplied:

```powershell
.\venv\Scripts\python.exe -m src.validation create-holdout `
  --data data\processed\cicids2018_labeled.csv `
  --name v10_final_holdout `
  --holdout-size 0.15
```

Run nested cross-validation on `development.csv` only. It fits preprocessing,
feature selection, scaling, threshold selection, and the model separately in
each fold; it never accesses `final_holdout.csv`:

```powershell
.\venv\Scripts\python.exe -m src.validation cross-validate `
  --data data\final_holdout\v10_final_holdout\development.csv `
  --config models\configs\xgb_v10_candidate.json `
  --name xgb_v10_nested_cv `
  --folds 5
```

Each nested-CV run now writes two outer-fold reports:

- `summary.csv`: performance when each fold chooses its own threshold from its
  inner validation data. This evaluates the threshold-selection procedure.
- `fixed_threshold_summary.csv`: performance when every outer fold uses V10's
  frozen threshold of `0.26`. This evaluates the exact operating point used by
  the saved V10 artifact.

Use the fixed-threshold report when deciding whether the frozen V10 operating
threshold has enough recall and false-positive-rate margin. It does not tune
the threshold and it does not access the final holdout.

Only after deciding that cross-validation results are acceptable should you run
the frozen V10 artifact against `final_holdout.csv` once:

```powershell
.\venv\Scripts\python.exe -m src.validation final-evaluate `
  --data data\final_holdout\v10_final_holdout\final_holdout.csv `
  --model models\saved\xgb_v10_regularized_fine_threshold.pkl `
  --config models\configs\xgb_v10_candidate.json `
  --name xgb_v10_final_holdout
```

For a strictly unbiased final score, the source used for this final holdout must
not have contributed to V10's earlier training or experiment comparisons.

Do not adjust V10's threshold or hyperparameters based on its final-holdout
result. A future candidate must be developed using development data only and
evaluated on a new independent holdout or external traffic dataset.
