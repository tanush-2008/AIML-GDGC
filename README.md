# Network Intrusion Detection Challenge

## Selected result

The best confirmed public Kaggle score in the submission history available for this project is **0.84588**. Two files tied:

- `outputs/submission_blend_candidate.csv`: 75% profile weighted model + 25% domain weighted model.
- `outputs/submission_blend_alpha_030.csv`: 70% profile weighted model + 30% domain weighted model.

Both files have the required `id,Target` columns and 33,161 predictions in test row order. This is the best score observed among the submissions tried for this account; it is not a claim of first place. See [`outputs/score_history.json`](outputs/score_history.json) for the recorded scores.

## Repository map

| Path | Purpose |
|---|---|
| `fit_blend_candidate.py` | Reproduces the selected probability blend and alpha sweep. |
| `fit_domain_candidate.py` | Shared feature and density ratio helpers; can also fit the lower scoring domain-only candidate. |
| `train_model.py` | Profile weighted baseline and shared label, feature, and profile weighting utilities. |
| `experiments/` | Optional alternatives and validation diagnostics; none is needed to reproduce the selected blend. See [`experiments/README.md`](experiments/README.md). |
| `outputs/` | Scored prediction files and score history are tracked; generated diagnostics and unscored alternatives are ignored by Git. |
| `data/raw/README.md` | Describes where to place the official competition files. The data itself is excluded. |

## Problem and data

The task is four class classification of network traffic into `Availability`, `Benign`, `Reconnaissance`, and `Spoofing`. The training set has 84,713 labeled rows and 82 predictors; the test set has 33,161 rows. `Availability` is the majority class (80,072 training rows), so a random row split or accuracy alone can give a misleading view of performance.

Train and test traffic profiles differ sharply. For example, `tcp|-` is about 94.5% of training but 46.0% of test; `udp|dns` is about 1.35% of training and 24.6% of test. A train-versus-test classifier reached AUC about 0.99998, indicating severe covariate shift. This does not prove that labels are stable under the shift, and very large density ratios can be noisy.

Place the organizer-provided files here to retrain:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

## Feature engineering

The pipeline uses a small set of deliberate transformations suited to CatBoost:

- Drop opaque row `id` from the model inputs, but preserve it unchanged in the output and retain test row order.
- Keep `proto` and `service` as categorical strings; replace missing category values with a sentinel.
- Convert numeric positive and negative infinity values to missing values. Numeric features do not need scaling for these tree models.
- Form a `proto|service` traffic profile to estimate how representative each training row is of the test distribution. Use this profile for sample weights, not as a target-derived feature.
- Do not use `Target` or `id` as predictors. The selected blend uses no target encoding or label-dependent test features.

## Models and reasoning

The selected output averages probabilities from two CatBoost models:

1. **Profile weighted model:** weight each training row by its test profile share divided by its training profile share, emphasizing traffic profiles that appear more often in test. This model uses 238 iterations.
2. **Domain weighted model:** train a classifier to distinguish train rows from test rows. Use three out-of-fold predictions to estimate density ratios for labeled rows; clip raw ratios to `[0.10, 20]`, then normalize their mean to one. Fit a second CatBoost model with these weights for 175 iterations.
3. **Probability blend:** average the two models' class probabilities at 75% / 25%. A 70% / 30% blend changed only a small number of labels and tied the same public score.

Both target models use depth 8, learning rate 0.05, L2 leaf regularization 8, random strength 0.3, seed 42, and four CPU threads. Domain classifiers use depth 6 and 180 iterations. The blend was selected because its observed Kaggle score was highest, not because an internal estimate looked best.

## Evaluation methodology

`train_model.py` holds out 20% of labeled rows within `proto|service|target` strata, preserving each observed profile/class combination where possible. It reports ordinary macro-F1 and macro-F1 weighted by estimated test/train profile shares. It compares profile-only, class-balanced, and combined weighting, selects by profile-weighted macro-F1 (ordinary macro-F1 breaks ties), then refits the baseline on all labeled rows.

`experiments/validate_blend_alpha.py` uses a separate profile/target-stratified 20% holdout. It estimates train-versus-test density ratios out of fold on the fit rows, weights the held-out evaluation by estimated test-domain likelihood, and compares ordinary and domain weighted macro-F1 for several blend weights. This is diagnostic: source holdout labels cannot guarantee performance on the shifted hidden test set.

Final selection used public Kaggle scores from actual submissions. The 0.25 and 0.30 blends both scored 0.84588. The public leaderboard is only a partial evaluation and private performance may differ. In the leaderboard screenshot available during the challenge, this score was not first place.

## Recorded results

| Candidate | Public score | Status |
|---|---:|---|
| 75% profile + 25% domain probability blend | **0.84588** | Confirmed best |
| 70% profile + 30% domain probability blend | **0.84588** | Confirmed tie |
| Profile weighted baseline | 0.84551 | Scored |
| Port category model | 0.84403 | Scored |
| LightGBM alternative | 0.84368 | Scored |
| Domain weighted model alone | 0.84347 | Scored |

Weight clipping and pseudo label experiments are unscored. Local validation or agreement with other models is not evidence that they improve the Kaggle score; they are not selected as final predictions.

## Reproduce the selected blend

Use Python 3.10 or later. From the repository root:

```powershell
python -m pip install -r requirements.txt
python fit_blend_candidate.py
```

This fits both selected components and writes the 0.25 blend, 0.10–0.50 alpha sweep, class count report, and probability cache under `outputs/`. It also writes `submission_blend_alpha_030.csv`, which tied the confirmed best score. Training can take several minutes on CPU. To fit only the profile weighted baseline, run `python train_model.py`.

This package excludes the competition data. The included prediction files allow review without retraining. The supplied `sample_submission.csv` is used by the baseline script to verify ID order and output format.

## Interview notes

Be ready to explain why the train/test shift motivated profile and density ratio weighting, why extreme ratios were clipped, how the probability blend was selected, and why the public Kaggle score was more reliable for model selection than the optimistic holdout estimates. This work used AI assistance; review and understand the code before presenting the methodology.
