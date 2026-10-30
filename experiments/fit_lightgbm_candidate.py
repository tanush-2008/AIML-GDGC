"""Train a LightGBM candidate using profile-aware weighting and port categories."""

from __future__ import annotations

import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_model import (
    CLASSES,
    LABEL_TO_CODE,
    macro_f1,
    prepare_features,
    profile_importance_weights,
    profile_keys,
    stratified_profile_split,
)


SEED = 42
THREADS = 4


def features_with_categories(train: pd.DataFrame, test: pd.DataFrame):
    X = prepare_features(train)
    X_test = prepare_features(test)
    category_columns = ["proto", "service", "id.resp_p"]
    for column in category_columns:
        all_values = pd.concat([X[column], X_test[column]], ignore_index=True)
        categories = pd.Index(all_values.astype("string").fillna("__MISSING__").unique())
        X[column] = pd.Categorical(
            X[column].astype("string").fillna("__MISSING__"), categories=categories
        )
        X_test[column] = pd.Categorical(
            X_test[column].astype("string").fillna("__MISSING__"), categories=categories
        )
    return X, X_test, category_columns


def model_params() -> dict:
    return {
        "objective": "multiclass",
        "num_class": len(CLASSES),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 40,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "feature_fraction": 0.85,
        "lambda_l2": 8.0,
        "seed": SEED,
        "num_threads": THREADS,
        "verbosity": -1,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    train = pd.read_csv(root / "data/raw/train.csv")
    test = pd.read_csv(root / "data/raw/test.csv")
    sample = pd.read_csv(root / "data/raw/sample_submission.csv")
    if not test["id"].equals(sample["id"]):
        raise ValueError("test.csv IDs do not match sample_submission.csv")

    X, X_test, categorical = features_with_categories(train, test)
    y = train["Target"].map(LABEL_TO_CODE).to_numpy(dtype=int)
    train_weights = profile_importance_weights(train, test, training=True)
    fit_idx, valid_idx = stratified_profile_split(train, y, 0.20, SEED)
    tr_share = profile_keys(train).value_counts(normalize=True)
    te_share = profile_keys(test).value_counts(normalize=True)
    ratios = (te_share / tr_share).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    valid_weights = profile_keys(train.iloc[valid_idx]).map(ratios).fillna(0.0).to_numpy(float)

    train_set = lgb.Dataset(
        X.iloc[fit_idx],
        label=y[fit_idx],
        weight=train_weights[fit_idx],
        categorical_feature=categorical,
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        X.iloc[valid_idx],
        label=y[valid_idx],
        weight=valid_weights,
        categorical_feature=categorical,
        reference=train_set,
        free_raw_data=False,
    )
    print("Fitting LightGBM validation model...")
    model = lgb.train(
        model_params(),
        train_set,
        num_boost_round=1200,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(100)],
    )
    best_iteration = int(model.best_iteration or model.current_iteration())
    valid_prob = model.predict(X.iloc[valid_idx], num_iteration=best_iteration)
    valid_codes = valid_prob.argmax(axis=1)
    ordinary, ordinary_detail = macro_f1(y[valid_idx], valid_codes)
    weighted, weighted_detail = macro_f1(y[valid_idx], valid_codes, valid_weights)
    iterations = max(200, int(math.ceil(best_iteration / (len(fit_idx) / len(train)))))

    final_set = lgb.Dataset(
        X,
        label=y,
        weight=train_weights,
        categorical_feature=categorical,
        free_raw_data=False,
    )
    print(f"Fitting final LightGBM model with {iterations} iterations...")
    final_model = lgb.train(model_params(), final_set, num_boost_round=iterations)
    probability = final_model.predict(X_test, num_iteration=iterations)
    codes = probability.argmax(axis=1)
    submission = pd.DataFrame({
        "id": test["id"],
        "Target": [CLASSES[int(code)] for code in codes],
    })
    output = root / "outputs/submission_lightgbm_candidate.csv"
    submission.to_csv(output, index=False)
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": final_model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    importance.to_csv(root / "outputs/lightgbm_feature_importance.csv", index=False)
    report = {
        "model": "LightGBM multiclass with proto, service, and responder port as categorical",
        "categorical_features": categorical,
        "validation": {
            "best_iteration": best_iteration,
            "ordinary_macro_f1": ordinary,
            "profile_weighted_macro_f1": weighted,
            "ordinary_per_class": ordinary_detail,
            "profile_weighted_per_class": weighted_detail,
            "caveat": "The random holdout does not reproduce the observed train/test shift.",
        },
        "final_iterations": iterations,
        "predicted_class_counts": submission["Target"].value_counts().to_dict(),
        "submission_path": str(output),
    }
    (root / "outputs/lightgbm_candidate_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
