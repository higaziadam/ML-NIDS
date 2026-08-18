# V10 XGBoost Release Candidate

V10 is the frozen low-alert XGBoost candidate for this project. It is a release
candidate, not a production release: it still requires cross-validation and a
single evaluation on a genuinely untouched final holdout.

## Release identity

- Profile: `models/configs/xgb_v10_candidate.json`
- Artifact: `models/saved/xgb_v10_regularized_fine_threshold.pkl`
- Artifact SHA-256:
  `8C736F8FD2327C1F555D69A89E977B0BE23335835D0932341672AB270BEE9D97`
- Decision threshold: `0.26`
- Validation policy: recall >= `0.92`; false-positive rate <= `0.005`

## Reported evaluation

The recorded V10 test result was 92.11% attack recall and a 0.41% false-positive
rate. This result is useful for experiment comparison, but it is not the final
unbiased deployment estimate because earlier experiments were compared on the
same test partition.

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
