"""Train a stabilized domain-adaptation blend with capped importance weights."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from fit_domain_candidate import auc_rank, density_ratio, prepare_features
from train_model import CATEGORICAL_COLUMNS, CLASSES, LABEL_TO_CODE


SEED = 42
THREADS = 4
DOMAIN_FOLDS = 3
DOMAIN_ITERATIONS = 180
MODEL_ITERATIONS = 175
BLEND_ALPHA = 0.25


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-cap", type=float, default=5.0)
    args = parser.parse_args()
    weight_cap = args.weight_cap
    if weight_cap <= 0:
        raise ValueError("--weight-cap must be positive")
    root = Path(__file__).resolve().parents[1]
    outputs = root / "outputs"
    train = pd.read_csv(root / "data/raw/train.csv")
    test = pd.read_csv(root / "data/raw/test.csv")
    X = prepare_features(train)
    X_test = prepare_features(test)
    y = train["Target"].map(LABEL_TO_CODE).to_numpy(dtype=int)
    categorical = [column for column in CATEGORICAL_COLUMNS if column in X.columns]

    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(len(X)), DOMAIN_FOLDS)
    domain_oof = np.zeros(len(X), dtype=float)
    domain_test = np.zeros(len(X_test), dtype=float)
    weights = np.zeros(len(X), dtype=float)
    for fold_number, heldout in enumerate(folds, start=1):
        keep = np.ones(len(X), dtype=bool)
        keep[heldout] = False
        n_source = int(keep.sum())
        domain_X = pd.concat([X.iloc[np.flatnonzero(keep)], X_test], ignore_index=True)
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
        domain_model.fit(Pool(domain_X, domain_y, cat_features=categorical))
        probability = domain_model.predict_proba(X.iloc[heldout])[:, 1]
        domain_oof[heldout] = probability
        weights[heldout] = density_ratio(probability, n_source, len(X_test))
        domain_test += domain_model.predict_proba(X_test)[:, 1] / DOMAIN_FOLDS

    weights /= weights.mean()
    uncapped_quantiles = {str(q): float(np.quantile(weights, q)) for q in (0, .5, .9, .99, .999, 1)}
    weights = np.minimum(weights, weight_cap)
    # A global weight scale does not change the fitted objective, so keep the
    # cap exact instead of re-normalizing and raising the maximum again.
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
    print(f"Fitting capped domain-weighted model; cap={weight_cap}")
    model.fit(Pool(X, y, cat_features=categorical, weight=weights))
    shifted_probability = model.predict_proba(X_test)

    saved_probabilities = outputs / "blend_probabilities.npz"
    with np.load(saved_probabilities) as base_archive:
        baseline_probability = base_archive["baseline"]
        saved_ids = base_archive["ids"].astype(str)
    if not np.array_equal(saved_ids, test["id"].astype(str).to_numpy()):
        raise ValueError("Cached baseline probabilities do not match test.csv IDs.")
    blended_probability = (
        (1.0 - BLEND_ALPHA) * baseline_probability
        + BLEND_ALPHA * shifted_probability
    )
    codes = blended_probability.argmax(axis=1)
    candidate = pd.DataFrame({
        "id": test["id"],
        "Target": [CLASSES[int(code)] for code in codes],
    })
    cap_label = f"{weight_cap:g}".replace(".", "p")
    output = outputs / f"submission_clipped_domain_cap_{cap_label}_candidate.csv"
    candidate.to_csv(output, index=False)
    np.savez_compressed(
        outputs / f"clipped_domain_probabilities_cap_{cap_label}.npz",
        capped_domain=shifted_probability,
        baseline=baseline_probability,
        ids=test["id"].astype(str).to_numpy(dtype=str),
    )
    report = {
        "method": "25% capped density-ratio CatBoost probability blended with 75% original baseline",
        "weight_cap_after_normalization": weight_cap,
        "uncapped_domain_weight_quantiles": uncapped_quantiles,
        "capped_weight_quantiles": {
            str(q): float(np.quantile(weights, q)) for q in (0, .5, .9, .99, .999, 1)
        },
        "domain_classifier_auc": auc_rank(
            np.r_[domain_oof, domain_test],
            np.r_[np.zeros(len(X), dtype=int), np.ones(len(X_test), dtype=int)],
        ),
        "model_iterations": MODEL_ITERATIONS,
        "blend_alpha": BLEND_ALPHA,
        "predicted_class_counts": candidate["Target"].value_counts().to_dict(),
        "candidate_path": str(output),
        "note": "Unscored until submitted to Kaggle; local validation metrics are not substitutes for the leaderboard.",
    }
    (outputs / f"clipped_domain_cap_{cap_label}_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
