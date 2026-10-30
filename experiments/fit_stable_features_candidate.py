"""Train a candidate after removing highly train/test-shifted rate/time features."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from train_model import (
    CATEGORICAL_COLUMNS,
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


def remove_shifted_features(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    drop = [
        column for column in features.columns
        if column == "flow_duration"
        or column == "payload_bytes_per_second"
        or "_per_sec" in column
        or column.startswith(("flow_iat.", "fwd_iat.", "bwd_iat.", "active.", "idle."))
        or column.endswith(".std")
    ]
    return features.drop(columns=drop), drop


def make_model(iterations: int) -> CatBoostClassifier:
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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    train = pd.read_csv(root / "data/raw/train.csv")
    test = pd.read_csv(root / "data/raw/test.csv")
    sample = pd.read_csv(root / "data/raw/sample_submission.csv")
    if not test["id"].equals(sample["id"]):
        raise ValueError("test.csv IDs do not match sample_submission.csv")

    X, dropped = remove_shifted_features(prepare_features(train))
    X_test, _ = remove_shifted_features(prepare_features(test))
    y = train["Target"].map(LABEL_TO_CODE).to_numpy(dtype=int)
    cat = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
    profile_weights = profile_importance_weights(train, test, training=True)

    fit_idx, valid_idx = stratified_profile_split(train, y, 0.20, SEED)
    tr_share = profile_keys(train).value_counts(normalize=True)
    te_share = profile_keys(test).value_counts(normalize=True)
    ratios = (te_share / tr_share).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    valid_weights = profile_keys(train.iloc[valid_idx]).map(ratios).fillna(0.0).to_numpy(float)

    model = make_model(350)
    print(f"Dropped {len(dropped)} rate/time/variance features")
    model.fit(
        Pool(X.iloc[fit_idx], y[fit_idx], cat_features=cat, weight=profile_weights[fit_idx]),
        eval_set=Pool(X.iloc[valid_idx], y[valid_idx], cat_features=cat, weight=valid_weights),
        early_stopping_rounds=60,
        use_best_model=True,
    )
    valid_prediction = np.asarray(model.predict(X.iloc[valid_idx])).reshape(-1).astype(int)
    ordinary, ordinary_detail = macro_f1(y[valid_idx], valid_prediction)
    weighted, weighted_detail = macro_f1(y[valid_idx], valid_prediction, valid_weights)
    best_iteration = int(model.get_best_iteration() + 1)
    iterations = max(150, int(math.ceil(best_iteration / (len(fit_idx) / len(train)))))

    final_model = make_model(iterations)
    print(f"Refitting stable-feature model with {iterations} iterations")
    final_model.fit(Pool(X, y, cat_features=cat, weight=profile_weights))
    codes = np.asarray(final_model.predict(X_test)).reshape(-1).astype(int)
    submission = pd.DataFrame({
        "id": test["id"],
        "Target": [CLASSES[int(code)] for code in codes],
    })
    out = root / "outputs/submission_stable_features_candidate.csv"
    submission.to_csv(out, index=False)
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": final_model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    importance.to_csv(root / "outputs/stable_features_importance.csv", index=False)
    report = {
        "dropped_features": dropped,
        "remaining_feature_count": len(X.columns),
        "validation": {
            "best_iteration": best_iteration,
            "ordinary_macro_f1": ordinary,
            "profile_weighted_macro_f1": weighted,
            "ordinary_per_class": ordinary_detail,
            "profile_weighted_per_class": weighted_detail,
            "caveat": "Random held-out rows do not reproduce the observed train/test shift.",
        },
        "final_iterations": iterations,
        "predicted_class_counts": submission["Target"].value_counts().to_dict(),
        "submission_path": str(out),
    }
    (root / "outputs/stable_features_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
