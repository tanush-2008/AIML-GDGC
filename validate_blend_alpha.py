"""Compare probability blend weights on a profile-stratified holdout.

The validation is diagnostic only: it does not reproduce the severe train/test
shift measured by the domain classifier or replace a Kaggle score.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from fit_domain_candidate import density_ratio
from train_model import (
    CATEGORICAL_COLUMNS,
    CLASSES,
    LABEL_TO_CODE,
    macro_f1,
    prepare_features,
    profile_importance_weights,
    stratified_profile_split,
)


SEED = 42
THREADS = 4
DOMAIN_FOLDS = 3
ALPHAS = (0.0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0)


def main() -> None:
    root = Path(__file__).resolve().parent
    train = pd.read_csv(root / "data/raw/train.csv")
    test = pd.read_csv(root / "data/raw/test.csv")
    y = train["Target"].map(LABEL_TO_CODE).to_numpy(dtype=int)
    X = prepare_features(train)
    X_test = prepare_features(test)
    categorical = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
    fit_idx, valid_idx = stratified_profile_split(train, y, 0.20, SEED)

    # Predict each fit row's train-vs-test likelihood out of fold so its
    # density-ratio weight is not calculated from a domain model that saw it.
    rng = np.random.default_rng(SEED)
    fit_folds = np.array_split(rng.permutation(fit_idx), DOMAIN_FOLDS)
    fit_domain_weights = np.zeros(len(fit_idx), dtype=float)
    fit_position = {row: pos for pos, row in enumerate(fit_idx)}
    valid_domain_probability = np.zeros(len(valid_idx), dtype=float)
    domain_fit_sizes = []
    for number, heldout in enumerate(fit_folds, start=1):
        source_keep = np.setdiff1d(fit_idx, heldout, assume_unique=False)
        domain_x = pd.concat([X.iloc[source_keep], X_test], ignore_index=True)
        domain_y = np.r_[np.zeros(len(source_keep), dtype=int), np.ones(len(test), dtype=int)]
        domain_model = CatBoostClassifier(
            iterations=180,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            l2_leaf_reg=8.0,
            random_seed=SEED + number,
            thread_count=THREADS,
            allow_writing_files=False,
            verbose=60,
        )
        print(f"Fitting domain fold {number}/{DOMAIN_FOLDS}")
        domain_model.fit(Pool(domain_x, domain_y, cat_features=categorical))
        heldout_probability = domain_model.predict_proba(X.iloc[heldout])[:, 1]
        heldout_weights = density_ratio(heldout_probability, len(source_keep), len(test))
        for row, weight in zip(heldout, heldout_weights, strict=True):
            fit_domain_weights[fit_position[int(row)]] = weight
        valid_domain_probability += domain_model.predict_proba(X.iloc[valid_idx])[:, 1] / DOMAIN_FOLDS
        domain_fit_sizes.append(len(source_keep))

    fit_domain_weights /= fit_domain_weights.mean()
    source_size = int(round(float(np.mean(domain_fit_sizes))))
    valid_domain_weights = density_ratio(valid_domain_probability, source_size, len(test))

    profile_weights = profile_importance_weights(train, test, training=True)
    base = CatBoostClassifier(
        iterations=350,
        depth=8,
        learning_rate=0.05,
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        l2_leaf_reg=8.0,
        random_strength=0.3,
        random_seed=SEED,
        thread_count=THREADS,
        allow_writing_files=False,
        verbose=100,
    )
    print("Fitting baseline holdout component")
    base.fit(
        Pool(X.iloc[fit_idx], y[fit_idx], cat_features=categorical, weight=profile_weights[fit_idx]),
        eval_set=Pool(X.iloc[valid_idx], y[valid_idx], cat_features=categorical, weight=valid_domain_weights),
        early_stopping_rounds=60,
        use_best_model=True,
    )
    base_probability = base.predict_proba(X.iloc[valid_idx])

    shifted = CatBoostClassifier(
        iterations=350,
        depth=8,
        learning_rate=0.05,
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        l2_leaf_reg=8.0,
        random_strength=0.3,
        random_seed=SEED,
        thread_count=THREADS,
        allow_writing_files=False,
        verbose=100,
    )
    print("Fitting domain-weighted holdout component")
    shifted.fit(
        Pool(X.iloc[fit_idx], y[fit_idx], cat_features=categorical, weight=fit_domain_weights),
        eval_set=Pool(X.iloc[valid_idx], y[valid_idx], cat_features=categorical, weight=valid_domain_weights),
        early_stopping_rounds=60,
        use_best_model=True,
    )
    shifted_probability = shifted.predict_proba(X.iloc[valid_idx])

    result = []
    for alpha in ALPHAS:
        pred = ((1.0 - alpha) * base_probability + alpha * shifted_probability).argmax(axis=1)
        ordinary, _ = macro_f1(y[valid_idx], pred)
        weighted, per_class = macro_f1(y[valid_idx], pred, valid_domain_weights)
        result.append({
            "domain_blend_weight": alpha,
            "ordinary_macro_f1": ordinary,
            "domain_weighted_macro_f1": weighted,
            "domain_weighted_per_class": per_class,
        })

    report = {
        "validation_method": "profile/target-stratified 20% holdout; 3-fold OOF train-vs-test weighting",
        "base_best_iteration": int(base.get_best_iteration() + 1),
        "domain_best_iteration": int(shifted.get_best_iteration() + 1),
        "results": result,
        "caveat": "Internal validation is optimistic under the severe observed train/test shift. Use Kaggle public score to choose the final blend.",
    }
    output = root / "outputs/blend_alpha_validation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
