"""CatBoost candidate that treats source/destination ports as nominal categories only."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

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
MAX_ITERATIONS = 140
EARLY_STOPPING_ROUNDS = 35


def features_with_port_categories(frame: pd.DataFrame) -> pd.DataFrame:
    features = prepare_features(frame)
    for column in ("id.orig_p", "id.resp_p"):
        values = pd.to_numeric(features[column], errors="coerce").round()
        features[f"{column}_cat"] = (
            values.astype("Int64").astype("string").fillna("__MISSING__").astype(str)
        )
    return features.drop(columns=["id.orig_p", "id.resp_p"])


def make_model(iterations: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=8,
        learning_rate=0.05,
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        l2_leaf_reg=8.0,
        random_strength=0.3,
        max_ctr_complexity=1,
        one_hot_max_size=10,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        random_seed=SEED,
        thread_count=THREADS,
        allow_writing_files=False,
        verbose=100,
    )


def main() -> None:
    root = Path(__file__).resolve().parent
    train = pd.read_csv(root / "data/raw/train.csv")
    test = pd.read_csv(root / "data/raw/test.csv")
    sample = pd.read_csv(root / "data/raw/sample_submission.csv")
    if not test["id"].equals(sample["id"]):
        raise ValueError("test.csv IDs do not match sample_submission.csv")

    X = features_with_port_categories(train)
    X_test = features_with_port_categories(test)
    y = train["Target"].map(LABEL_TO_CODE).to_numpy(dtype=int)
    categorical = ["proto", "service", "id.orig_p_cat", "id.resp_p_cat"]
    train_weights = profile_importance_weights(train, test, training=True)
    fit_idx, valid_idx = stratified_profile_split(train, y, 0.20, SEED)
    tr_share = profile_keys(train).value_counts(normalize=True)
    te_share = profile_keys(test).value_counts(normalize=True)
    ratios = (te_share / tr_share).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    valid_weights = profile_keys(train.iloc[valid_idx]).map(ratios).fillna(0.0).to_numpy(float)

    validation_model = make_model(MAX_ITERATIONS)
    print("Fitting port-category-only validation model...")
    validation_model.fit(
        Pool(X.iloc[fit_idx], y[fit_idx], cat_features=categorical, weight=train_weights[fit_idx]),
        eval_set=Pool(X.iloc[valid_idx], y[valid_idx], cat_features=categorical, weight=valid_weights),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )
    valid_pred = np.asarray(validation_model.predict(X.iloc[valid_idx])).reshape(-1).astype(int)
    ordinary, ordinary_details = macro_f1(y[valid_idx], valid_pred)
    weighted, weighted_details = macro_f1(y[valid_idx], valid_pred, valid_weights)
    best_iteration = int(validation_model.get_best_iteration() + 1)
    fit_fraction = len(fit_idx) / len(train)
    final_iterations = max(150, int(math.ceil(best_iteration / fit_fraction)))

    final_model = make_model(final_iterations)
    print(f"Fitting final port-category-only model with {final_iterations} iterations")
    final_model.fit(Pool(X, y, cat_features=categorical, weight=train_weights))
    codes = np.asarray(final_model.predict(X_test)).reshape(-1).astype(int)
    submission = pd.DataFrame({
        "id": test["id"],
        "Target": [CLASSES[int(code)] for code in codes],
    })
    output = root / "outputs/submission_port_category_only_candidate.csv"
    submission.to_csv(output, index=False)
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": final_model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    importance.to_csv(root / "outputs/port_category_only_importance.csv", index=False)
    report = {
        "model": "CatBoost with nominal source/destination ports; numeric port columns removed",
        "validation": {
            "best_iteration": best_iteration,
            "ordinary_macro_f1": ordinary,
            "profile_weighted_macro_f1": weighted,
            "ordinary_per_class": ordinary_details,
            "profile_weighted_per_class": weighted_details,
            "caveat": "Random held-out rows do not reproduce the observed train/test shift.",
        },
        "final_iterations": final_iterations,
        "predicted_class_counts": submission["Target"].value_counts().to_dict(),
        "test_port21_predictions": submission.loc[test["id.resp_p"].eq(21), "Target"].value_counts().to_dict(),
        "submission_path": str(output),
    }
    (root / "outputs/port_category_only_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
