"""CatBoost experiment treating network ports as categorical features."""

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
MAX_ITERATIONS = 350
EARLY_STOPPING_ROUNDS = 60


def add_port_categories(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    for column in ("id.orig_p", "id.resp_p"):
        values = pd.to_numeric(features[column], errors="coerce").round()
        features[f"{column}_cat"] = (
            values.astype("Int64").astype("string").fillna("__MISSING__").astype(str)
        )
    return features


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    train = pd.read_csv(root / "data/raw/train.csv")
    test = pd.read_csv(root / "data/raw/test.csv")
    sample = pd.read_csv(root / "data/raw/sample_submission.csv")
    if not test["id"].equals(sample["id"]):
        raise ValueError("test.csv IDs do not match sample_submission.csv")

    X = add_port_categories(prepare_features(train))
    X_test = add_port_categories(prepare_features(test))
    y = train["Target"].map(LABEL_TO_CODE).to_numpy(dtype=int)
    categorical = ["proto", "service", "id.orig_p_cat", "id.resp_p_cat"]
    train_weights = profile_importance_weights(train, test, training=True)

    fit_idx, valid_idx = stratified_profile_split(train, y, 0.20, SEED)
    shares_train = profile_keys(train).value_counts(normalize=True)
    shares_test = profile_keys(test).value_counts(normalize=True)
    ratio = (shares_test / shares_train).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    valid_weights = profile_keys(train.iloc[valid_idx]).map(ratio).fillna(0.0).to_numpy(float)

    def new_model(iterations: int) -> CatBoostClassifier:
        return CatBoostClassifier(
            iterations=iterations,
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

    validation_model = new_model(MAX_ITERATIONS)
    print("Fitting port-categorical validation model...")
    validation_model.fit(
        Pool(X.iloc[fit_idx], y[fit_idx], cat_features=categorical, weight=train_weights[fit_idx]),
        eval_set=Pool(X.iloc[valid_idx], y[valid_idx], cat_features=categorical, weight=valid_weights),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )
    valid_pred = np.asarray(validation_model.predict(X.iloc[valid_idx])).reshape(-1).astype(int)
    ordinary_f1, ordinary_details = macro_f1(y[valid_idx], valid_pred)
    weighted_f1, weighted_details = macro_f1(y[valid_idx], valid_pred, valid_weights)
    best_iteration = int(validation_model.get_best_iteration() + 1)
    fit_fraction = len(fit_idx) / len(train)
    final_iterations = max(200, int(math.ceil(best_iteration / fit_fraction)))

    final_model = new_model(final_iterations)
    print(f"Fitting final port-categorical model; iterations={final_iterations}")
    final_model.fit(Pool(X, y, cat_features=categorical, weight=train_weights))
    probabilities = final_model.predict_proba(X_test)
    prediction = probabilities.argmax(axis=1)
    submission = pd.DataFrame({
        "id": test["id"],
        "Target": [CLASSES[int(code)] for code in prediction],
    })
    output = root / "outputs/submission_port_category_candidate.csv"
    submission.to_csv(output, index=False)
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": final_model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    importance.to_csv(root / "outputs/port_category_feature_importance.csv", index=False)

    port_counts = {}
    for port in (21, 53, 80, 443, 1883, 5353):
        rows = test["id.resp_p"].eq(port).to_numpy()
        port_counts[str(port)] = {
            "test_rows": int(rows.sum()),
            "predictions": submission.loc[rows, "Target"].value_counts().to_dict(),
        }

    report = {
        "model": "CatBoost with numeric and categorical source/destination ports",
        "port_21_train_target_counts": train.loc[train["id.resp_p"].eq(21), "Target"].value_counts().to_dict(),
        "validation": {
            "best_iteration": best_iteration,
            "ordinary_macro_f1": ordinary_f1,
            "profile_weighted_macro_f1": weighted_f1,
            "ordinary_per_class": ordinary_details,
            "weighted_per_class": weighted_details,
            "caveat": "Random held-out rows do not reproduce the observed train/test domain shift.",
        },
        "final_iterations": final_iterations,
        "test_port_predictions": port_counts,
        "predicted_class_counts": submission["Target"].value_counts().to_dict(),
        "submission_path": str(output),
    }
    (root / "outputs/port_category_candidate_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
