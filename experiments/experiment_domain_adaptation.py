"""Offline experiment: importance-weight training rows toward test-like features.

This is an experiment only. It does not overwrite the submitted prediction file.
"""

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
    class_balance_weights,
    macro_f1,
    prepare_features,
    profile_keys,
    stratified_profile_split,
)


SEED = 42
THREADS = 4


def density_ratio(probability: np.ndarray, n_source: int, n_target: int) -> np.ndarray:
    """Convert P(target-domain|x) to clipped source-to-target density ratios."""
    eps = 1e-5
    probability = np.clip(probability, eps, 1.0 - eps)
    ratio = (probability / (1.0 - probability)) * (n_source / n_target)
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
    root = Path(__file__).resolve().parents[1]
    train_frame = pd.read_csv(root / "data/raw/train.csv")
    test_frame = pd.read_csv(root / "data/raw/test.csv")
    y = train_frame["Target"].map(LABEL_TO_CODE).to_numpy(dtype=int)
    X = prepare_features(train_frame)
    X_test = prepare_features(test_frame)
    fit_indices, valid_indices = stratified_profile_split(
        train_frame, y, validation_fraction=0.20, seed=SEED
    )
    X_fit, X_valid = X.iloc[fit_indices], X.iloc[valid_indices]
    y_fit, y_valid = y[fit_indices], y[valid_indices]
    categorical = [c for c in CATEGORICAL_COLUMNS if c in X.columns]

    # Out-of-fold domain predictions avoid using in-sample domain scores as
    # training importance weights, which would overstate train/test separation.
    fold_rng = np.random.default_rng(SEED)
    shuffled = fold_rng.permutation(len(X_fit))
    folds = np.array_split(shuffled, 3)
    domain_train_w = np.zeros(len(X_fit), dtype=float)
    valid_prob_sum = np.zeros(len(X_valid), dtype=float)
    test_prob_sum = np.zeros(len(X_test), dtype=float)
    source_fold_sizes = []
    print("Training 3-fold out-of-fold train-vs-test domain classifiers...")
    for fold_number, heldout in enumerate(folds, start=1):
        source_keep = np.ones(len(X_fit), dtype=bool)
        source_keep[heldout] = False
        X_domain_train = pd.concat(
            [X_fit.iloc[np.flatnonzero(source_keep)], X_test], ignore_index=True
        )
        y_domain = np.r_[
            np.zeros(source_keep.sum(), dtype=int),
            np.ones(len(X_test), dtype=int),
        ]
        domain_model = CatBoostClassifier(
            iterations=180,
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
        print(f"Domain fold {fold_number}/3")
        domain_model.fit(Pool(X_domain_train, y_domain, cat_features=categorical))
        heldout_prob = domain_model.predict_proba(X_fit.iloc[heldout])[:, 1]
        domain_train_w[heldout] = density_ratio(
            heldout_prob, int(source_keep.sum()), len(X_test)
        )
        valid_prob_sum += domain_model.predict_proba(X_valid)[:, 1] / len(folds)
        test_prob_sum += domain_model.predict_proba(X_test)[:, 1] / len(folds)
        source_fold_sizes.append(int(source_keep.sum()))

    average_source_size = int(round(float(np.mean(source_fold_sizes))))
    valid_prob = valid_prob_sum
    test_prob = test_prob_sum
    domain_valid_w = density_ratio(valid_prob, average_source_size, len(X_test))
    domain_train_w /= domain_train_w.mean()

    all_train_profiles = profile_keys(train_frame)
    test_profiles = profile_keys(test_frame)
    train_share = all_train_profiles.value_counts(normalize=True)
    test_share = test_profiles.value_counts(normalize=True)
    profile_ratio = (test_share / train_share).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    profile_train_w = (
        profile_keys(train_frame.iloc[fit_indices])
        .map(profile_ratio)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    profile_train_w = np.clip(profile_train_w, 0.10, 12.0)
    profile_train_w /= profile_train_w.mean()
    domain_profile_train_w = domain_train_w * profile_train_w
    domain_profile_train_w /= domain_profile_train_w.mean()

    # Use the domain density ratio alone for validation scoring: it targets the
    # feature distribution of the hidden test set, not just its profile mix.
    candidates = [
        ("domain_only", domain_train_w, None),
        (
            "domain_profile_sqrt_balanced",
            domain_profile_train_w,
            "sqrt",
        ),
    ]
    results = []
    for name, fit_weights, balance in candidates:
        class_weights = None
        if balance == "sqrt":
            class_weights = class_balance_weights(y_fit, fit_weights, 0.5)
        model = CatBoostClassifier(
            iterations=350,
            depth=8,
            learning_rate=0.05,
            loss_function="MultiClass",
            eval_metric="TotalF1:average=Macro",
            class_weights=class_weights,
            l2_leaf_reg=8.0,
            random_strength=0.3,
            random_seed=SEED,
            thread_count=THREADS,
            allow_writing_files=False,
            verbose=100,
        )
        print(f"\nTraining {name}; class_weights={class_weights}")
        model.fit(
            Pool(X_fit, y_fit, cat_features=categorical, weight=fit_weights),
            eval_set=Pool(
                X_valid,
                y_valid,
                cat_features=categorical,
                weight=domain_valid_w,
            ),
            early_stopping_rounds=60,
            use_best_model=True,
        )
        pred = np.asarray(model.predict(X_valid)).reshape(-1).astype(int)
        ordinary, _ = macro_f1(y_valid, pred)
        shifted, details = macro_f1(y_valid, pred, domain_valid_w)
        results.append({
            "name": name,
            "best_iteration": int(model.get_best_iteration() + 1),
            "class_weights": class_weights,
            "ordinary_macro_f1": ordinary,
            "domain_weighted_macro_f1": shifted,
            "domain_weighted_per_class": details,
        })
        print(
            f"{name}: ordinary macro F1={ordinary:.5f}; "
            f"domain-weighted macro F1={shifted:.5f}"
        )

    # Produce a separate candidate file using out-of-fold domain weights on all
    # labeled rows. Preserve outputs/submission.csv from the already-submitted run.
    full_shuffled = fold_rng.permutation(len(X))
    full_folds = np.array_split(full_shuffled, 3)
    full_domain_weights = np.zeros(len(X), dtype=float)
    for fold_number, heldout in enumerate(full_folds, start=1):
        source_keep = np.ones(len(X), dtype=bool)
        source_keep[heldout] = False
        X_domain_train = pd.concat(
            [X.iloc[np.flatnonzero(source_keep)], X_test], ignore_index=True
        )
        y_domain = np.r_[
            np.zeros(source_keep.sum(), dtype=int),
            np.ones(len(X_test), dtype=int),
        ]
        domain_model = CatBoostClassifier(
            iterations=180,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            l2_leaf_reg=8.0,
            random_seed=SEED + 100 + fold_number,
            thread_count=THREADS,
            allow_writing_files=False,
            verbose=60,
        )
        print(f"Full-data domain-weight fold {fold_number}/3")
        domain_model.fit(Pool(X_domain_train, y_domain, cat_features=categorical))
        heldout_prob = domain_model.predict_proba(X.iloc[heldout])[:, 1]
        full_domain_weights[heldout] = density_ratio(
            heldout_prob, int(source_keep.sum()), len(X_test)
        )
    full_domain_weights /= full_domain_weights.mean()
    selected = results[0]
    final_iterations = max(
        100,
        int(math.ceil(selected["best_iteration"] / (len(fit_indices) / len(X)))),
    )
    final_model = CatBoostClassifier(
        iterations=final_iterations,
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
    print(
        f"\nFitting separate domain-weighted candidate; "
        f"iterations={final_iterations}"
    )
    final_model.fit(Pool(X, y, cat_features=categorical, weight=full_domain_weights))
    test_pred = np.asarray(final_model.predict(X_test)).reshape(-1).astype(int)
    candidate = pd.DataFrame({
        "id": test_frame["id"],
        "Target": [CLASSES[int(code)] for code in test_pred],
    })
    candidate_path = root / "outputs/submission_domain_candidate.csv"
    candidate.to_csv(candidate_path, index=False)
    report_candidate = {
        "path": str(candidate_path),
        "iterations": final_iterations,
        "predicted_class_counts": candidate["Target"].value_counts().to_dict(),
        "domain_weight_quantiles_full_train": {
            str(q): float(np.quantile(full_domain_weights, q))
            for q in (0, 0.5, 0.9, 0.99, 1)
        },
    }

    report = {
        "domain_classifier_auc": auc_rank(
            np.r_[valid_prob, test_prob],
            np.r_[np.zeros(len(valid_prob), dtype=int), np.ones(len(test_prob), dtype=int)],
        ),
        "domain_weight_quantiles": {
            str(q): float(np.quantile(domain_train_w, q)) for q in (0, 0.5, 0.9, 0.99, 1)
        },
        "validation_rows": len(valid_indices),
        "results": results,
        "separate_domain_weighted_candidate": report_candidate,
        "note": (
            "This is exploratory only. A domain classifier estimates covariate "
            "shift and does not correct arbitrary changes in P(Target|features)."
        ),
    }
    out_path = root / "outputs/domain_adaptation_experiment.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
