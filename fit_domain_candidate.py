"""Fit a separate test-domain-weighted candidate without overwriting submission.csv."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


CLASSES = ["Availability", "Benign", "Reconnaissance", "Spoofing"]
LABEL_TO_CODE = {label: code for code, label in enumerate(CLASSES)}
CATEGORICAL_COLUMNS = ["proto", "service"]


SEED = 42
THREADS = 4
DOMAIN_FOLDS = 3
DOMAIN_ITERATIONS = 180
MODEL_ITERATIONS = 175


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.drop(columns=["id", "Target"], errors="ignore").copy()
    for column in CATEGORICAL_COLUMNS:
        features[column] = (
            features[column].astype("string").fillna("__MISSING__").astype(str)
        )
    numeric = [c for c in features.columns if c not in CATEGORICAL_COLUMNS]
    features[numeric] = features[numeric].replace([np.inf, -np.inf], np.nan)
    return features


def density_ratio(probability: np.ndarray, n_source: int, n_target: int) -> np.ndarray:
    eps = 1e-5
    p = np.clip(probability, eps, 1.0 - eps)
    ratio = (p / (1.0 - p)) * (n_source / n_target)
    return np.clip(ratio, 0.10, 20.0)


def auc_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    root = Path(__file__).resolve().parent
    train = pd.read_csv(root / "data/raw/train.csv")
    test = pd.read_csv(root / "data/raw/test.csv")
    X = prepare_features(train)
    X_test = prepare_features(test)
    y = train["Target"].map(LABEL_TO_CODE).to_numpy(dtype=int)
    categorical = [c for c in CATEGORICAL_COLUMNS if c in X.columns]

    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(len(X)), DOMAIN_FOLDS)
    oof_probability = np.zeros(len(X), dtype=float)
    test_probability = np.zeros(len(X_test), dtype=float)
    weights = np.zeros(len(X), dtype=float)
    source_sizes = []

    for fold_number, heldout in enumerate(folds, start=1):
        keep = np.ones(len(X), dtype=bool)
        keep[heldout] = False
        n_source = int(keep.sum())
        source_sizes.append(n_source)
        domain_x = pd.concat([X.iloc[np.flatnonzero(keep)], X_test], ignore_index=True)
        domain_y = np.r_[np.zeros(n_source, dtype=int), np.ones(len(X_test), dtype=int)]
        domain_model = CatBoostClassifier(
            iterations=DOMAIN_ITERATIONS,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            l2_leaf_reg=8.0,
            random_seed=SEED + fold_number,
            thread_count=THREADS,
            allow_writing_files=False,
            verbose=60,
        )
        print(f"Fitting domain fold {fold_number}/{DOMAIN_FOLDS}")
        domain_model.fit(Pool(domain_x, domain_y, cat_features=categorical))
        holdout_p = domain_model.predict_proba(X.iloc[heldout])[:, 1]
        oof_probability[heldout] = holdout_p
        weights[heldout] = density_ratio(holdout_p, n_source, len(X_test))
        test_probability += domain_model.predict_proba(X_test)[:, 1] / DOMAIN_FOLDS

    weights /= weights.mean()
    model = CatBoostClassifier(
        iterations=MODEL_ITERATIONS,
        depth=8,
        learning_rate=0.05,
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        l2_leaf_reg=8.0,
        random_strength=0.3,
        random_seed=SEED,
        thread_count=THREADS,
        allow_writing_files=False,
        verbose=50,
    )
    print(f"Fitting final domain-weighted model ({MODEL_ITERATIONS} iterations)")
    model.fit(Pool(X, y, cat_features=categorical, weight=weights))
    prediction_codes = np.asarray(model.predict(X_test)).reshape(-1).astype(int)
    predictions = pd.DataFrame({
        "id": test["id"],
        "Target": [CLASSES[int(code)] for code in prediction_codes],
    })
    candidate_path = root / "outputs/submission_domain_candidate.csv"
    predictions.to_csv(candidate_path, index=False)
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    importance.to_csv(root / "outputs/domain_candidate_feature_importance.csv", index=False)

    report = {
        "approach": "3-fold out-of-fold train-vs-test density-ratio weighting",
        "domain_classifier_auc": auc_rank(
            np.r_[oof_probability, test_probability],
            np.r_[np.zeros(len(oof_probability), dtype=int), np.ones(len(test_probability), dtype=int)],
        ),
        "domain_weight_quantiles": {
            str(q): float(np.quantile(weights, q)) for q in (0, 0.5, 0.9, 0.99, 1)
        },
        "model_iterations": MODEL_ITERATIONS,
        "predicted_class_counts": predictions["Target"].value_counts().to_dict(),
        "submission_path": str(candidate_path),
        "warning": (
            "Exploratory alternative. Near-perfect domain separability means density-ratio "
            "weights can be unstable; only an external leaderboard score can establish "
            "whether this candidate improves on the first submission."
        ),
    }
    (root / "outputs/domain_candidate_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
