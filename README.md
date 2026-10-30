# Network Intrusion Detection Challenge

## Selected result

The best **confirmed public Kaggle score** in the submission history available for this project is **0.84588**. Two prediction files tied at that score:

- `outputs/submission_blend_candidate.csv` — 75% profile weighted model + 25% domain weighted model.
- `outputs/submission_blend_alpha_030.csv` — 70% profile weighted model + 30% domain weighted model.

Both have the required `id,Target` columns and 33,161 prediction rows. The 0.25 blend was the first file at this score; the 0.30 blend was later submitted and tied it. This is the best score observed for this account among the approaches tried; it is not a claim of first place. Other candidates below were not confirmed improvements.

## Problem and data

The task is four class classification of network traffic into `Availability`, `Benign`, `Reconnaissance`, and `Spoofing`. The supplied training file has 84,713 labeled rows, 82 predictors, and an opaque `id` column. The test file has 33,161 rows. `Availability` is the majority class (80,072 training rows), so ordinary accuracy or a random row split can give a misleading picture of performance.

The train and test traffic profiles differ sharply. For example, `tcp|-` is about 94.5% of training but 46.0% of test; `udp|dns` is about 1.35% of training and 24.6% of test. A train-versus-test classifier reached AUC about 0.99998. This is evidence of severe covariate shift. It does not prove that labels are stable under the shift, and extreme density ratios can be noisy.

Place the official competition files here before running the models:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

Competition data is excluded from this code package. `id` is preserved in predictions but dropped from model features. `proto` and `service` are passed to CatBoost as categorical features; numeric infinities are converted to missing values.

## Feature engineering

The pipeline uses a small set of deliberate transformations suited to CatBoost rather than scaling numeric variables or expanding them into one-hot columns:

- Drop `id` from the model inputs because it is an opaque row identifier; retain it unchanged in the output and preserve test row order.
- Keep `proto` and `service` as categorical strings, replacing missing values with a dedicated sentinel. Replace positive and negative numeric infinity with missing values.
- Form a `proto|service` traffic profile for estimating how much each training row represents the test distribution. This profile drives sample weights; it is not an extra target-derived feature.
- Do not use `Target` or row ID as predictors. No target encoding or label-dependent test features are used in the selected blend.

## Models and reasoning

The selected predictions average probabilities from two CatBoost models:

1. **Profile weighted model.** For each `proto|service` profile, assign training rows a weight equal to the test share divided by the training share. This shifts emphasis toward profiles common in test data. It uses 238 iterations.
2. **Domain weighted model.** Train a classifier to distinguish train rows from test rows. Use three out of fold predictions to estimate density ratios for the labeled rows, clip the raw ratios to `[0.10, 20]`, and normalize weights to mean one. Fit a second CatBoost classifier for 175 iterations.
3. **Probability blend.** Average the class probabilities at 75% / 25% for the selected `submission_blend_candidate.csv`. A 70% / 30% version changed only a small number of labels and tied the same public score.

Both target models use CatBoost with depth 8, learning rate 0.05, L2 leaf regularization 8, random strength 0.3, seed 42, and four CPU threads. The domain classifiers use depth 6 and 180 iterations. The blend was chosen because it had the highest observed Kaggle score, not because the internal validation estimate was highest.

## Evaluation methodology

`train_model.py` reserves 20% of the labeled data using a split within `proto|service|target` strata, so each observed profile/class combination is represented in validation when possible. It reports ordinary macro-F1 and macro-F1 weighted by estimated test/train profile shares. It compares profile-only, class-balanced, and combined weighting, selects by profile-weighted macro-F1 (ordinary macro-F1 breaks ties), then refits on all labeled rows.

`validate_blend_alpha.py` uses a separate profile/target-stratified 20% holdout. It estimates train-vs-test density ratios out of fold on the fit rows, weights the held-out score by estimated test-domain likelihood, and compares ordinary and domain-weighted macro-F1 across blend weights. This holdout uses labels from the same source sample and cannot establish the hidden-test score under severe shift. Its results are diagnostic only.

Final model selection used the public Kaggle scores from actual submissions. The 0.25 and 0.30 blends both scored 0.84588, above the other scored candidates. The public leaderboard is only a partial evaluation; private leaderboard performance may differ. The competition screenshot showed that this score was not first place, so this README calls it the best confirmed result from the tested submissions for this account, not a winning result.

## Results of tested alternatives

| Candidate | Public score | Status |
|---|---:|---|
| 75% profile + 25% domain probability blend | **0.84588** | Confirmed best |
| 70% profile + 30% domain probability blend | **0.84588** | Confirmed tie |
| Profile weighted baseline (`outputs/submission.csv`) | 0.84551 | Scored |
| Port category model | 0.84403 | Scored |
| LightGBM alternative | 0.84368 | Scored |
| Domain weighted model alone | 0.84347 | Scored |

The capped density ratio and consensus pseudo label experiments in the working outputs are **unscored**. Their local validation or agreement with other models cannot establish that they improve the Kaggle score, so they are not selected as the final result.

## Reproduce

Use Python 3.10 or later. In PowerShell, from this project directory:

```powershell
python -m pip install -r requirements.txt
python train_model.py
python fit_blend_candidate.py
```

`train_model.py` evaluates the profile weighted baseline and writes a Kaggle format baseline submission. `fit_blend_candidate.py` independently fits the two selected components, writes the selected 0.25 blend, the 0.10–0.50 sweep files, the 0.30 tied-best file, a report, and cached probabilities under `outputs/`. The run may take several minutes on CPU. The provided best prediction files are included so the result can be reviewed without retraining.

Optional experiments have separate scripts. `fit_clipped_domain_candidate.py` and `fit_consensus_pseudo_candidate.py` create unscored alternatives. `validate_blend_alpha.py` evaluates blend weights on a test-profile-weighted holdout; this is diagnostic and does not replace a Kaggle score. The LightGBM experiment requires the optional dependencies in `requirements_lightgbm.txt`.

## Package contents and interview notes

The ZIP contains the source code, dependency lists, this explanation, the two tied-best prediction files, and the score history. It excludes the competition data and the exploratory prediction files that did not beat the confirmed score. The training code can regenerate the best predictions when the official CSV files are placed under `data/raw/`.

For an interview, be prepared to explain why the train/test shift motivated profile and density ratio weighting, why the density ratios were clipped, how class probabilities were blended, and why weighted holdout results were not treated as a substitute for the Kaggle public score. The final selected output is `id,Target`; retain the opaque test IDs in the exact provided row order.

This work used AI assistance. Review and understand the code before presenting the methodology.
