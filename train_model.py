"""Train and submit a CatBoost model for the Network Intrusion Detection challenge.

The script uses only the supplied competition CSV files. It holds out validation
rows within each observed protocol/service/target profile, evaluates both ordinary
and test-profile-weighted macro F1, selects a class-weighting strategy, then fits
on all labeled rows and writes a Kaggle-format submission.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


CLASSES = ["Availability", "Benign", "Reconnaissance", "Spoofing"]
LABEL_TO_CODE = {label: code for code, label in enumerate(CLASSES)}
CODE_TO_LABEL = {code: label for label, code in LABEL_TO_CODE.items()}
CATEGORICAL_COLUMNS = ["proto", "service"]


def profile_keys(frame: pd.DataFrame) -> pd.Series:
    """Return a coarse traffic profile from protocol and service."""
    proto = frame["proto"].astype("string").fillna("__MISSING__").astype(str)
    service = frame["service"].astype("string").fillna("__MISSING__").astype(str)
    return proto + "|" + service


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop opaque row IDs and preserve the two categorical network fields."""
    features = frame.drop(columns=["id", "Target"], errors="ignore").copy()
    for column in CATEGORICAL_COLUMNS:
        features[column] = (
            features[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )
    numeric_columns = [
        column for column in features.columns
        if column not in CATEGORICAL_COLUMNS
    ]
    features[numeric_columns] = features[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    return features


def stratified_profile_split(
    frame: pd.DataFrame,
    y: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split within each profile/target stratum so small classes are represented."""
    profiles = profile_keys(frame).to_numpy()
    strata = np.char.add(np.char.add(profiles.astype(str), "|"), y.astype(str))
    rng = np.random.default_rng(seed)
    validation_indices: list[int] = []

    for stratum in np.unique(strata):
        indices = np.flatnonzero(strata == stratum)
        rng.shuffle(indices)
        if len(indices) < 2:
            continue
        holdout_count = max(1, int(round(len(indices) * validation_fraction)))
        holdout_count = min(holdout_count, len(indices) - 1)
        validation_indices.extend(indices[:holdout_count].tolist())

    validation = np.asarray(sorted(validation_indices), dtype=int)
    train_mask = np.ones(len(frame), dtype=bool)
    train_mask[validation] = False
    training = np.flatnonzero(train_mask)
    return training, validation


def profile_importance_weights(
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    *,
    training: bool,
) -> np.ndarray:
    """Estimate train-to-test profile weights using only observed test features."""
    train_profiles = profile_keys(train_frame)
    eval_profiles = profile_keys(evaluation_frame)
    train_share = train_profiles.value_counts(normalize=True)
    eval_share = eval_profiles.value_counts(normalize=True)
    ratio = (eval_share / train_share).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    target_profiles = train_profiles if training else eval_profiles
    weights = target_profiles.map(ratio).to_numpy(dtype=float)

    if training:
        # Keep training-only profiles available to the model while emphasizing
        # profiles that occur more often in the unlabeled evaluation features.
        weights = np.clip(weights, 0.10, 12.0)
        weights /= weights.mean()
    return weights


def class_balance_weights(
    y: np.ndarray,
    sample_weights: np.ndarray,
    exponent: float,
) -> list[float] | None:
    """Compute class weights from profile-weighted support."""
    if exponent == 0.0:
        return None
    mass = np.bincount(y, weights=sample_weights, minlength=len(CLASSES))
    base = mass.mean() / np.maximum(mass, 1e-12)
    weights = base ** exponent
    return [float(value) for value in weights]


def macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> tuple[float, list[dict[str, float]]]:
    """Compute macro F1 and per-class precision, recall, and F1."""
    if sample_weights is None:
        sample_weights = np.ones(len(y_true), dtype=float)
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=float)
    np.add.at(matrix, (y_true, y_pred), sample_weights)

    details = []
    scores = []
    for code, label in enumerate(CLASSES):
        true_positive = matrix[code, code]
        false_positive = matrix[:, code].sum() - true_positive
        false_negative = matrix[code, :].sum() - true_positive
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0 else 0.0
        )
        recall = (
            true_positive / recall_denominator
            if recall_denominator > 0 else 0.0
        )
        f1_denominator = precision + recall
        f1 = 2.0 * precision * recall / f1_denominator if f1_denominator else 0.0
        scores.append(f1)
        details.append({
            "class": label,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "weighted_support": float(recall_denominator),
        })
    return float(np.mean(scores)), details


def make_model(
    *,
    iterations: int,
    class_weights: list[float] | None,
    seed: int,
    thread_count: int,
) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=8,
        learning_rate=0.05,
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        class_weights=class_weights,
        l2_leaf_reg=8.0,
        random_strength=0.3,
        random_seed=seed,
        thread_count=thread_count,
        allow_writing_files=False,
        verbose=100,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=120)
    parser.add_argument("--thread-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"
    sample_path = args.data_dir / "sample_submission.csv"
    for path in (train_path, test_path, sample_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required competition file not found: {path}")

    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_path)
    if not set(train_frame["Target"].dropna().unique()).issubset(LABEL_TO_CODE):
        raise ValueError("train.csv contains an unexpected Target label.")
    if train_frame["Target"].isna().any():
        raise ValueError("train.csv contains missing Target values.")
    if not test_frame["id"].equals(sample_submission["id"]):
        raise ValueError("test.csv IDs do not match sample_submission.csv order.")

    X = prepare_features(train_frame)
    X_test = prepare_features(test_frame)
    y = train_frame["Target"].map(LABEL_TO_CODE).to_numpy(dtype=int)
    profile = profile_keys(train_frame)
    train_profile_weights = profile_importance_weights(
        train_frame, test_frame, training=True
    )
    validation_train, validation_indices = stratified_profile_split(
        train_frame, y, args.validation_fraction, args.seed
    )
    X_fit = X.iloc[validation_train]
    X_valid = X.iloc[validation_indices]
    y_fit = y[validation_train]
    y_valid = y[validation_indices]
    fit_profile_weights = train_profile_weights[validation_train]
    # Map the test/train profile ratios onto the held-out training rows.
    full_profiles = profile_keys(train_frame)
    test_profiles = profile_keys(test_frame)
    train_shares = full_profiles.value_counts(normalize=True)
    test_shares = test_profiles.value_counts(normalize=True)
    profile_ratios = (test_shares / train_shares).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    valid_profile_weights = (
        full_profiles.iloc[validation_indices]
        .map(profile_ratios)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cat_features = [column for column in CATEGORICAL_COLUMNS if column in X.columns]
    candidates = [
        ("profile_only", 0.0, True),
        ("profile_sqrt_balanced", 0.5, True),
        ("profile_balanced", 1.0, True),
        ("sqrt_balanced_only", 0.5, False),
    ]
    candidate_reports = []
    best_candidate: dict | None = None

    for name, exponent, use_profile_weights in candidates:
        fit_weights = (
            fit_profile_weights
            if use_profile_weights
            else np.ones(len(validation_train), dtype=float)
        )
        class_weights = class_balance_weights(y_fit, fit_weights, exponent)
        model = make_model(
            iterations=args.iterations,
            class_weights=class_weights,
            seed=args.seed,
            thread_count=args.thread_count,
        )
        fit_pool = Pool(
            X_fit,
            y_fit,
            cat_features=cat_features,
            weight=fit_weights,
        )
        valid_pool = Pool(
            X_valid,
            y_valid,
            cat_features=cat_features,
            weight=valid_profile_weights,
        )
        print(f"\nTraining candidate: {name}; class_weights={class_weights}")
        model.fit(
            fit_pool,
            eval_set=valid_pool,
            early_stopping_rounds=args.early_stopping_rounds,
            use_best_model=True,
        )
        valid_pred = np.asarray(model.predict(X_valid)).reshape(-1).astype(int)
        ordinary_score, ordinary_details = macro_f1(y_valid, valid_pred)
        weighted_score, weighted_details = macro_f1(
            y_valid, valid_pred, valid_profile_weights
        )
        report = {
            "name": name,
            "class_weights": class_weights,
            "best_iteration": int(model.get_best_iteration() + 1),
            "ordinary_macro_f1": ordinary_score,
            "test_profile_weighted_macro_f1": weighted_score,
            "ordinary_per_class": ordinary_details,
            "weighted_per_class": weighted_details,
        }
        candidate_reports.append(report)
        print(
            f"{name}: ordinary macro F1={ordinary_score:.5f}; "
            f"test-profile-weighted macro F1={weighted_score:.5f}; "
            f"best iteration={report['best_iteration']}"
        )
        if best_candidate is None or (
            weighted_score,
            ordinary_score,
        ) > (
            best_candidate["test_profile_weighted_macro_f1"],
            best_candidate["ordinary_macro_f1"],
        ):
            best_candidate = report

    assert best_candidate is not None
    selected = next(item for item in candidates if item[0] == best_candidate["name"])
    selected_name, selected_exponent, selected_uses_profile = selected
    final_weights = (
        train_profile_weights
        if selected_uses_profile
        else np.ones(len(train_frame), dtype=float)
    )
    final_class_weights = class_balance_weights(y, final_weights, selected_exponent)
    fit_fraction = len(validation_train) / len(train_frame)
    final_iterations = int(math.ceil(best_candidate["best_iteration"] / fit_fraction))
    final_iterations = max(200, final_iterations)

    final_model = make_model(
        iterations=final_iterations,
        class_weights=final_class_weights,
        seed=args.seed,
        thread_count=args.thread_count,
    )
    print(
        f"\nTraining final model: {selected_name}; "
        f"iterations={final_iterations}; class_weights={final_class_weights}"
    )
    final_model.fit(
        Pool(X, y, cat_features=cat_features, weight=final_weights)
    )

    test_pred = np.asarray(final_model.predict(X_test)).reshape(-1).astype(int)
    submission = pd.DataFrame({
        "id": test_frame["id"],
        "Target": [CODE_TO_LABEL[int(code)] for code in test_pred],
    })
    if list(submission.columns) != ["id", "Target"]:
        raise AssertionError("Submission columns are not in the required order.")
    if len(submission) != len(test_frame) or submission["Target"].isna().any():
        raise AssertionError("Submission row count or predictions are invalid.")
    if not set(submission["Target"].unique()).issubset(CLASSES):
        raise AssertionError("Submission contains an invalid class label.")

    submission_path = args.output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": final_model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    importance.to_csv(args.output_dir / "feature_importance.csv", index=False)

    train_profile_counts = profile.value_counts().to_dict()
    test_profile_counts = profile_keys(test_frame).value_counts().to_dict()
    profile_summary = []
    for key in sorted(set(train_profile_counts) | set(test_profile_counts)):
        train_count = int(train_profile_counts.get(key, 0))
        test_count = int(test_profile_counts.get(key, 0))
        profile_summary.append({
            "profile": key,
            "train_rows": train_count,
            "test_rows": test_count,
            "train_share": train_count / len(train_frame),
            "test_share": test_count / len(test_frame),
        })

    report = {
        "rows": {"train": len(train_frame), "test": len(test_frame)},
        "input_features": len(X.columns),
        "categorical_features": cat_features,
        "classes": CLASSES,
        "class_counts": {
            label: int((train_frame["Target"] == label).sum()) for label in CLASSES
        },
        "validation": {
            "method": "20% holdout within proto/service/target strata",
            "weighted_profiles": "test profile share divided by train profile share",
            "selected_candidate": selected_name,
            "selected_candidate_report": best_candidate,
            "candidates": candidate_reports,
            "interpretation": (
                "The profile-weighted score is a covariate-shift estimate, not a "
                "guarantee of hidden-test performance. It assumes labels remain "
                "reasonably stable within the observed proto/service profiles."
            ),
        },
        "profile_distribution": profile_summary,
        "final_model": {
            "iterations": final_iterations,
            "class_weights": final_class_weights,
            "random_seed": args.seed,
            "depth": 8,
            "learning_rate": 0.05,
        },
        "submission": {
            "path": str(submission_path),
            "rows": len(submission),
            "predicted_class_counts": {
                label: int((submission["Target"] == label).sum()) for label in CLASSES
            },
        },
    }
    with (args.output_dir / "validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nSelected candidate: {selected_name}")
    print(f"Validation report: {args.output_dir / 'validation_report.json'}")
    print(f"Feature importance: {args.output_dir / 'feature_importance.csv'}")
    print(f"Kaggle submission: {submission_path}")
    print("Predicted class counts:")
    print(submission["Target"].value_counts().reindex(CLASSES, fill_value=0).to_string())


if __name__ == "__main__":
    main()
