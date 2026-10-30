"""CatBoost candidate with cross-fitted, smoothed port/profile target encodings."""

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
ENCODING_FOLDS = 5
SMOOTHING = {
    "src_port": 25.0,
    "dst_port": 15.0,
    "profile": 10.0,
    "dst_service": 20.0,
    "src_dst_port": 50.0,
}


def make_keys(frame: pd.DataFrame) -> pd.DataFrame:
    source_port = pd.to_numeric(frame["id.orig_p"], errors="coerce").round()
    dest_port = pd.to_numeric(frame["id.resp_p"], errors="coerce").round()
    src = source_port.astype("Int64").astype("string").fillna("__MISSING__")
    dst = dest_port.astype("Int64").astype("string").fillna("__MISSING__")
    service = frame["service"].astype("string").fillna("__MISSING__")
    return pd.DataFrame({
        "src_port": src.astype(str),
        "dst_port": dst.astype(str),
        "profile": profile_keys(frame).astype(str),
        "dst_service": (dst + "|" + service).astype(str),
        "src_dst_port": (src + "|" + dst).astype(str),
    }, index=frame.index)


def encode_from_source(
    source_keys: pd.DataFrame,
    y_source: np.ndarray,
    query_keys: pd.DataFrame,
) -> pd.DataFrame:
    prior = np.bincount(y_source, minlength=len(CLASSES)).astype(float)
    prior /= prior.sum()
    result = pd.DataFrame(index=query_keys.index)
    for key, alpha in SMOOTHING.items():
        source = pd.DataFrame({"key": source_keys[key].astype(str).to_numpy(), "y": y_source})
        counts = pd.crosstab(source["key"], source["y"]).reindex(
            columns=range(len(CLASSES)), fill_value=0
        )
        query_keys_as_str = query_keys[key].astype(str)
        aligned = counts.reindex(query_keys_as_str.to_numpy(), fill_value=0)
        values = aligned.to_numpy(dtype=float)
        totals = values.sum(axis=1)
        probabilities = (values + alpha * prior) / (totals[:, None] + alpha)
        for code, label in enumerate(CLASSES):
            result[f"te_{key}_{label}"] = probabilities[:, code]
        result[f"te_{key}_log_count"] = np.log1p(totals)
    return result


def out_of_fold_encodings(keys: pd.DataFrame, y: np.ndarray, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fold_id = np.zeros(len(y), dtype=int)
    for class_code in range(len(CLASSES)):
        rows = np.flatnonzero(y == class_code)
        rng.shuffle(rows)
        fold_id[rows] = np.arange(len(rows)) % ENCODING_FOLDS
    encoded = None
    for fold in range(ENCODING_FOLDS):
        valid_rows = np.flatnonzero(fold_id == fold)
        train_rows = np.flatnonzero(fold_id != fold)
        current = encode_from_source(keys.iloc[train_rows], y[train_rows], keys.iloc[valid_rows])
        if encoded is None:
            encoded = pd.DataFrame(index=keys.index, columns=current.columns, dtype=float)
        encoded.iloc[valid_rows] = current.to_numpy()
    assert encoded is not None
    return encoded.astype(float)


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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    train = pd.read_csv(root / "data/raw/train.csv")
    test = pd.read_csv(root / "data/raw/test.csv")
    sample = pd.read_csv(root / "data/raw/sample_submission.csv")
    if not test["id"].equals(sample["id"]):
        raise ValueError("test.csv IDs do not match sample_submission.csv")
    y = train["Target"].map(LABEL_TO_CODE).to_numpy(dtype=int)
    X = prepare_features(train)
    X_test = prepare_features(test)
    train_keys = make_keys(train)
    test_keys = make_keys(test)
    categorical = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
    weights = profile_importance_weights(train, test, training=True)

    fit_idx, valid_idx = stratified_profile_split(train, y, 0.20, SEED)
    fit_keys = train_keys.iloc[fit_idx]
    fit_y = y[fit_idx]
    fit_oof = out_of_fold_encodings(fit_keys.reset_index(drop=True), fit_y, SEED)
    valid_enc = encode_from_source(fit_keys, fit_y, train_keys.iloc[valid_idx])
    X_fit = pd.concat([X.iloc[fit_idx].reset_index(drop=True), fit_oof.reset_index(drop=True)], axis=1)
    X_valid = pd.concat([X.iloc[valid_idx].reset_index(drop=True), valid_enc.reset_index(drop=True)], axis=1)

    train_share = profile_keys(train).value_counts(normalize=True)
    test_share = profile_keys(test).value_counts(normalize=True)
    ratio = (test_share / train_share).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    valid_weights = profile_keys(train.iloc[valid_idx]).map(ratio).fillna(0.0).to_numpy(float)
    model = new_model(350)
    print("Fitting target-encoding validation model...")
    model.fit(
        Pool(X_fit, fit_y, cat_features=categorical, weight=weights[fit_idx]),
        eval_set=Pool(X_valid, y[valid_idx], cat_features=categorical, weight=valid_weights),
        early_stopping_rounds=60,
        use_best_model=True,
    )
    valid_pred = np.asarray(model.predict(X_valid)).reshape(-1).astype(int)
    ordinary, ordinary_detail = macro_f1(y[valid_idx], valid_pred)
    weighted, weighted_detail = macro_f1(y[valid_idx], valid_pred, valid_weights)
    best_iteration = int(model.get_best_iteration() + 1)
    final_iterations = max(150, int(math.ceil(best_iteration / (len(fit_idx) / len(train)))))

    all_oof = out_of_fold_encodings(train_keys.reset_index(drop=True), y, SEED)
    test_enc = encode_from_source(train_keys, y, test_keys)
    X_full = pd.concat([X.reset_index(drop=True), all_oof.reset_index(drop=True)], axis=1)
    X_test_full = pd.concat([X_test.reset_index(drop=True), test_enc.reset_index(drop=True)], axis=1)
    final_model = new_model(final_iterations)
    print(f"Fitting final target-encoding model with {final_iterations} iterations")
    final_model.fit(Pool(X_full, y, cat_features=categorical, weight=weights))
    codes = np.asarray(final_model.predict(X_test_full)).reshape(-1).astype(int)
    submission = pd.DataFrame({
        "id": test["id"],
        "Target": [CLASSES[int(code)] for code in codes],
    })
    output = root / "outputs/submission_port_target_encoding_candidate.csv"
    submission.to_csv(output, index=False)
    importance = pd.DataFrame({
        "feature": X_full.columns,
        "importance": final_model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    importance.to_csv(root / "outputs/port_target_encoding_importance.csv", index=False)
    report = {
        "method": "Out-of-fold smoothed target encodings for source/destination ports and profiles",
        "smoothing": SMOOTHING,
        "validation": {
            "best_iteration": best_iteration,
            "ordinary_macro_f1": ordinary,
            "profile_weighted_macro_f1": weighted,
            "ordinary_per_class": ordinary_detail,
            "profile_weighted_per_class": weighted_detail,
            "caveat": "Random holdout scores do not reproduce the observed public leaderboard shift.",
        },
        "final_iterations": final_iterations,
        "predicted_class_counts": submission["Target"].value_counts().to_dict(),
        "submission_path": str(output),
    }
    (root / "outputs/port_target_encoding_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
