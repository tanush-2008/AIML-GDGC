"""Self-train on high-confidence test rows where two models agree."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from train_model import (
    CATEGORICAL_COLUMNS,
    CLASSES,
    LABEL_TO_CODE,
    prepare_features,
    profile_importance_weights,
)


SEED = 42
THREADS = 4
CONFIDENCE_THRESHOLD = 0.90
PSEUDO_WEIGHT = 0.35
MODEL_ITERATIONS = 238
BLEND_ALPHA = 0.50


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    outputs = root / "outputs"
    train = pd.read_csv(root / "data/raw/train.csv")
    test = pd.read_csv(root / "data/raw/test.csv")
    sample = pd.read_csv(root / "data/raw/sample_submission.csv")
    if not test["id"].equals(sample["id"]):
        raise ValueError("test.csv IDs do not match sample_submission.csv")

    with np.load(outputs / "blend_probabilities.npz") as archive:
        baseline_probability = archive["baseline"]
        domain_probability = archive["domain_weighted"]
        cached_ids = archive["ids"].astype(str)
    if not np.array_equal(cached_ids, test["id"].astype(str).to_numpy()):
        raise ValueError("Cached probabilities do not match test.csv IDs.")

    baseline_codes = baseline_probability.argmax(axis=1)
    domain_codes = domain_probability.argmax(axis=1)
    both_confident = (
        (baseline_codes == domain_codes)
        & (baseline_probability.max(axis=1) >= CONFIDENCE_THRESHOLD)
        & (domain_probability.max(axis=1) >= CONFIDENCE_THRESHOLD)
    )
    pseudo_indices = np.flatnonzero(both_confident)
    pseudo_codes = baseline_codes[pseudo_indices]

    X = prepare_features(train)
    X_test = prepare_features(test)
    y = train["Target"].map(LABEL_TO_CODE).to_numpy(dtype=int)
    categorical = [column for column in CATEGORICAL_COLUMNS if column in X.columns]
    pseudo_X = X_test.iloc[pseudo_indices].reset_index(drop=True)
    fit_X = pd.concat([X.reset_index(drop=True), pseudo_X], ignore_index=True)
    fit_y = np.r_[y, pseudo_codes]
    original_weights = profile_importance_weights(train, test, training=True)
    pseudo_weights = np.full(len(pseudo_indices), PSEUDO_WEIGHT, dtype=float)
    fit_weights = np.r_[original_weights, pseudo_weights]

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
        verbose=60,
    )
    print(
        f"Fitting self-training model with {len(pseudo_indices)} consensus pseudo-labels; "
        f"threshold={CONFIDENCE_THRESHOLD}, weight={PSEUDO_WEIGHT}"
    )
    model.fit(Pool(fit_X, fit_y, cat_features=categorical, weight=fit_weights))
    pseudo_probability = model.predict_proba(X_test)

    candidates = []
    for alpha in (BLEND_ALPHA, 1.0):
        blended = (1.0 - alpha) * baseline_probability + alpha * pseudo_probability
        codes = blended.argmax(axis=1)
        submission = pd.DataFrame({
            "id": test["id"],
            "Target": [CLASSES[int(code)] for code in codes],
        })
        label = "direct" if alpha == 1.0 else f"blend_{alpha:.2f}".replace(".", "p")
        path = outputs / f"submission_consensus_pseudo_{label}.csv"
        submission.to_csv(path, index=False)
        candidates.append({
            "baseline_blend_weight": 1.0 - alpha,
            "pseudo_model_weight": alpha,
            "predicted_class_counts": submission["Target"].value_counts().to_dict(),
            "agreement_with_current_best": float(
                np.mean(
                    submission["Target"].to_numpy()
                    == pd.read_csv(outputs / "submission_blend_candidate.csv")["Target"].to_numpy()
                )
            ),
            "path": str(path),
        })

    report = {
        "method": "CatBoost self-training using test rows with baseline/domain agreement and confidence above threshold",
        "confidence_threshold_for_each_model": CONFIDENCE_THRESHOLD,
        "pseudo_label_rows": int(len(pseudo_indices)),
        "pseudo_label_class_counts": {
            CLASSES[int(code)]: int(count)
            for code, count in zip(*np.unique(pseudo_codes, return_counts=True), strict=True)
        },
        "pseudo_sample_weight": PSEUDO_WEIGHT,
        "model_iterations": MODEL_ITERATIONS,
        "candidates": candidates,
        "caveat": "This self-training approach is exploratory. Public Kaggle scores are required to establish whether it helps under the hidden test distribution.",
    }
    (outputs / "consensus_pseudo_candidate_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
