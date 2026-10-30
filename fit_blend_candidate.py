"""Fit a conservative probability blend of the first model and domain-weighted model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from fit_domain_candidate import auc_rank, density_ratio, prepare_features
from train_model import (
    CATEGORICAL_COLUMNS,
    CLASSES,
    LABEL_TO_CODE,
    profile_importance_weights,
)


SEED = 42
THREADS = 4
DOMAIN_FOLDS = 3
DOMAIN_ITERATIONS = 180
BASE_ITERATIONS = 238
DOMAIN_CLASSIFIER_ITERATIONS = 175
DOMAIN_BLEND = 0.25
ALPHA_SWEEP = (0.10, 0.20, 0.30, 0.40, 0.50)


def main() -> None:
    root = Path(__file__).resolve().parent
    train = pd.read_csv(root / "data/raw/train.csv")
    test = pd.read_csv(root / "data/raw/test.csv")
    X = prepare_features(train)
    X_test = prepare_features(test)
    y = train["Target"].map(LABEL_TO_CODE).to_numpy(dtype=int)
    categorical = [c for c in CATEGORICAL_COLUMNS if c in X.columns]

    # First-model component: the exact full-data profile-weighted CatBoost fit.
    profile_weights = profile_importance_weights(train, test, training=True)
    base_model = CatBoostClassifier(
        iterations=BASE_ITERATIONS,
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
    print("Fitting first-model component...")
    base_model.fit(Pool(X, y, cat_features=categorical, weight=profile_weights))
    base_prob = base_model.predict_proba(X_test)

    # Estimate density ratios out of fold so each training weight is predicted
    # by a domain model that did not train on that row.
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(len(X)), DOMAIN_FOLDS)
    oof_prob = np.zeros(len(X), dtype=float)
    test_prob = np.zeros(len(X_test), dtype=float)
    domain_weights = np.zeros(len(X), dtype=float)
    for fold_number, heldout in enumerate(folds, start=1):
        keep = np.ones(len(X), dtype=bool)
        keep[heldout] = False
        n_source = int(keep.sum())
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
        heldout_p = domain_model.predict_proba(X.iloc[heldout])[:, 1]
        oof_prob[heldout] = heldout_p
        domain_weights[heldout] = density_ratio(heldout_p, n_source, len(X_test))
        test_prob += domain_model.predict_proba(X_test)[:, 1] / DOMAIN_FOLDS
    domain_weights /= domain_weights.mean()

    domain_model = CatBoostClassifier(
        iterations=DOMAIN_CLASSIFIER_ITERATIONS,
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
    print("Fitting domain-weighted component...")
    domain_model.fit(Pool(X, y, cat_features=categorical, weight=domain_weights))
    shifted_prob = domain_model.predict_proba(X_test)

    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outputs_dir / "blend_probabilities.npz",
        baseline=base_prob,
        domain_weighted=shifted_prob,
        ids=test["id"].astype(str).to_numpy(dtype=str),
    )

    sweep_reports = []
    for alpha in (DOMAIN_BLEND, *ALPHA_SWEEP):
        blended_prob = (1.0 - alpha) * base_prob + alpha * shifted_prob
        prediction_codes = blended_prob.argmax(axis=1)
        candidate = pd.DataFrame({
            "id": test["id"],
            "Target": [CLASSES[int(code)] for code in prediction_codes],
        })
        if alpha == DOMAIN_BLEND:
            candidate_path = outputs_dir / "submission_blend_candidate.csv"
        else:
            alpha_label = f"{alpha:.2f}".replace(".", "")
            candidate_path = outputs_dir / f"submission_blend_alpha_{alpha_label}.csv"
        candidate.to_csv(candidate_path, index=False)
        sweep_reports.append({
            "domain_blend_weight": alpha,
            "predicted_class_counts": candidate["Target"].value_counts().to_dict(),
            "agreement_with_current_best": float(
                np.mean(
                    candidate["Target"].to_numpy()
                    == pd.read_csv(outputs_dir / "submission_blend_candidate.csv")["Target"].to_numpy()
                )
            ) if alpha != DOMAIN_BLEND else 1.0,
            "submission_path": str(candidate_path),
        })

    best_candidate_path = outputs_dir / "submission_blend_candidate.csv"
    best_candidate = pd.read_csv(best_candidate_path)

    base_codes = base_prob.argmax(axis=1)
    baseline_submission_path = outputs_dir / "submission.csv"
    baseline_agreement = None
    if baseline_submission_path.exists():
        baseline_labels = pd.read_csv(baseline_submission_path)["Target"].map(LABEL_TO_CODE)
        if baseline_labels.notna().all() and len(baseline_labels) == len(base_codes):
            baseline_agreement = float(np.mean(base_codes == baseline_labels.to_numpy(dtype=int)))
    report = {
        "method": "75% profile-weighted baseline probability + 25% domain-weighted probability",
        "domain_blend_weight": DOMAIN_BLEND,
        "probability_file": str(outputs_dir / "blend_probabilities.npz"),
        "alpha_sweep": sweep_reports,
        "baseline_agreement_with_saved_submission": baseline_agreement,
        "train_test_domain_auc": auc_rank(
            np.r_[oof_prob, test_prob],
            np.r_[np.zeros(len(oof_prob), dtype=int), np.ones(len(test_prob), dtype=int)],
        ),
        "predicted_class_counts": best_candidate["Target"].value_counts().to_dict(),
        "submission_path": str(best_candidate_path),
        "note": (
            "The 0.25 and 0.30 blends tied at public Kaggle score 0.84588 on "
            "2026-10-30. Other alpha-sweep outputs are unscored."
        ),
    }
    (root / "outputs/blend_candidate_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
