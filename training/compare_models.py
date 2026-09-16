"""
RF vs XGBoost comparison for PDAD employment type prediction.

READ-ONLY: this script trains models in memory and prints metrics.
It does NOT write, overwrite, or touch any file in ml_service/artifacts/.
The deployed RF model is left completely untouched.

Preprocessing mirrors the DEPLOYED model exactly (verified against
artifacts/rf_model.pkl: 7 classes, 108 features, 1,268 training rows):
  1. Drop rows where Type_of_Employment == "None/Others"
     (these are not employment types; the deployed model excludes them)
  2. Merge "Permanent/Regular" into "Permanent"
  3. Current_Assistive_Device NaN -> "None"
  4. pd.get_dummies(drop_first=False) on the 10 categorical features
  5. train_test_split(test_size=0.2, random_state=42, stratify=y)

Both models get identical data, identical split, identical seed, and
class-balancing. The only difference is the algorithm.

Usage (from repo root, in the ml_service venv + xgboost installed):
    python ml_service/training/compare_models.py \
        --input-csv database/seeders/data/pdad_registry.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42

RAW_FEATURES = [
    "Age",
    "Sex",
    "Civil_Status",
    "Disability_Type",
    "Disability_Visibility",
    "Cause_of_Disability",
    "Educational_Attainment",
    "Skills",
    "Mobility_Status",
    "Current_Assistive_Device",
    "Occupation_Group",
]
TARGET_COLUMN = "Type_of_Employment"
CATEGORICAL_COLS = [f for f in RAW_FEATURES if f != "Age"]

# Target values excluded / merged to match the deployed model.
EXCLUDED_TARGETS = ["None/Others"]
TARGET_MERGES = {"Permanent/Regular": "Permanent"}

# Deployed-model fingerprint, for the reproduction check.
EXPECTED_CLASSES = 7
EXPECTED_FEATURES = 108
EXPECTED_TRAIN_ROWS = 1268


def normalise_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept either the notebook's PascalCase headers (Disability_Type) or the
    repository CSV's lowercase headers (disability_type), and rename to the
    PascalCase names this script uses internally.

    Matching ignores case, spaces and hyphens, mirroring
    PdadRegistrySeeder::normalizeHeader() on the Laravel side.
    """
    wanted = RAW_FEATURES + [TARGET_COLUMN]

    def key(name: str) -> str:
        return str(name).strip().lower().replace(" ", "_").replace("-", "_")

    lookup = {key(c): c for c in df.columns}
    rename, missing = {}, []
    for target in wanted:
        source = lookup.get(key(target))
        if source is None:
            missing.append(target)
        elif source != target:
            rename[source] = target

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Columns found in CSV: {sorted(df.columns.tolist())}"
        )

    if rename:
        print(f"Header style:           lowercase -> renamed {len(rename)} columns")
    return df.rename(columns=rename)


def load_and_prepare(input_csv: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = normalise_headers(pd.read_csv(input_csv))

    df = df[RAW_FEATURES + [TARGET_COLUMN]].copy()
    rows_raw = len(df)

    # 1. Drop null and non-employment target values.
    df = df.dropna(subset=[TARGET_COLUMN])
    df = df[~df[TARGET_COLUMN].isin(EXCLUDED_TARGETS)]

    # 2. Merge equivalent target labels.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].replace(TARGET_MERGES)

    # 3. Data hygiene (matches deployed: device NaN -> "None").
    df["Current_Assistive_Device"] = df["Current_Assistive_Device"].fillna("None")

    print(f"Rows in CSV:            {rows_raw:,}")
    print(f"Rows after exclusions:  {len(df):,}")
    print(f"Classes:                {df[TARGET_COLUMN].nunique()}")
    print()
    print("Class distribution:")
    counts = df[TARGET_COLUMN].value_counts()
    for label, n in counts.items():
        print(f"  {label:<16} {n:>5,}  {n / len(df) * 100:5.1f}%")
    print(f"  {'imbalance ratio':<16} {counts.max() / counts.min():>5.1f}:1")
    print()

    X = pd.get_dummies(df[RAW_FEATURES], columns=CATEGORICAL_COLS, drop_first=False)
    y = df[TARGET_COLUMN]
    return X, y


def check_reproduction(X: pd.DataFrame, y: pd.Series, n_train: int) -> None:
    """Confirm this preprocessing reproduces the deployed artifact's shape."""
    print("Reproduction check vs deployed rf_model.pkl:")
    checks = [
        ("classes", y.nunique(), EXPECTED_CLASSES),
        ("encoded features", X.shape[1], EXPECTED_FEATURES),
        ("training rows", n_train, EXPECTED_TRAIN_ROWS),
    ]
    all_ok = True
    for name, got, want in checks:
        ok = got == want
        all_ok &= ok
        print(f"  {name:<18} {got:>6}   expected {want:>6}   {'OK' if ok else 'MISMATCH'}")
    if not all_ok:
        print("\n  WARNING: preprocessing does not match the deployed model.")
        print("  Metrics below are still valid as an RF-vs-XGB comparison,")
        print("  but they do not describe the model currently in production.")
    print()


def top_k_accuracy(model, X_test, y_test, k: int) -> float:
    """Fraction of rows where the true label is among the model's top k."""
    proba = model.predict_proba(X_test)
    classes = np.asarray(model.classes_)
    top_idx = np.argsort(proba, axis=1)[:, -k:]
    truth = np.asarray(y_test).reshape(-1, 1)
    return float((classes[top_idx] == truth).any(axis=1).mean())


def prevalence_baseline(y, k: int) -> float:
    """Score from always naming the k most common classes. The honest
    floor that any top-k number must be compared against."""
    counts = pd.Series(y).value_counts()
    return float(counts.iloc[:k].sum() / len(y))


def evaluate(name, model, X_train, X_test, y_train, y_test, labels, fit_kwargs=None):
    model.fit(X_train, y_train, **(fit_kwargs or {}))
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    t2 = top_k_accuracy(model, X_test, y_test, 2)
    t3 = top_k_accuracy(model, X_test, y_test, 3)

    print(f"--- {name} ---")
    print(f"Accuracy (top-1): {acc:.4f}")
    print(f"Top-2 accuracy:   {t2:.4f}   <- what the recommender ranks on")
    print(f"Top-3 accuracy:   {t3:.4f}")
    print(f"Macro F1:         {macro:.4f}")
    print(f"Weighted F1:      {weighted:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))
    return {"name": name, "accuracy": acc, "top2": t2, "top3": t3,
            "macro_f1": macro, "weighted_f1": weighted}


def cross_validate(name, make_model, X, y_enc, balanced: bool) -> tuple[float, float]:
    """5-fold stratified CV on macro F1 — more robust than one split."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, test_idx in skf.split(X, y_enc):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y_enc[train_idx], y_enc[test_idx]
        model = make_model()
        if balanced:
            w = compute_sample_weight("balanced", y_tr)
            model.fit(X_tr, y_tr, sample_weight=w)
        else:
            model.fit(X_tr, y_tr)
        scores.append(f1_score(y_te, model.predict(X_te), average="macro", zero_division=0))
    return float(np.mean(scores)), float(np.std(scores))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Random Forest vs XGBoost (read-only, no artifacts written)."
    )
    parser.add_argument("--input-csv", required=True, help="Path to training CSV file.")
    parser.add_argument("--skip-cv", action="store_true", help="Skip 5-fold cross-validation.")
    args = parser.parse_args()

    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise SystemExit(
            "xgboost is not installed.\n"
            "Install it in a SEPARATE analysis venv, not ml_service/.venv:\n"
            "    pip install xgboost"
        )

    print("=" * 68)
    print("PDAD EMPLOYMENT TYPE PREDICTION - RF vs XGBoost")
    print("=" * 68)
    print()

    X, y = load_and_prepare(Path(args.input_csv))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    check_reproduction(X, y, len(X_train))

    print(f"Train / test split:     {len(X_train):,} / {len(X_test):,}")
    print()
    print("NOTE: the smallest classes have very few test rows, so their")
    print("per-class F1 is noisy. Cross-validation below is more reliable.")
    print()

    results = []

    # --- Random Forest: identical hyperparameters to the deployed model ---
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    labels = sorted(y.unique())
    results.append(evaluate("RANDOM FOREST (deployed config)", rf,
                            X_train, X_test, y_train, y_test, labels))

    # --- XGBoost: needs integer labels + explicit balancing ---
    le = LabelEncoder().fit(y)
    y_train_enc, y_test_enc = le.transform(y_train), le.transform(y_test)

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(le.classes_),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="mlogloss",
        tree_method="hist",
    )
    results.append(evaluate(
        "XGBOOST", xgb, X_train, X_test, y_train_enc, y_test_enc,
        list(le.classes_),
        fit_kwargs={"sample_weight": compute_sample_weight("balanced", y_train_enc)},
    ))

    # --- Cross-validation ---
    if not args.skip_cv:
        print("5-fold stratified cross-validation (macro F1):")
        y_all_enc = le.transform(y)
        rf_mean, rf_std = cross_validate(
            "RF",
            lambda: RandomForestClassifier(
                n_estimators=200, min_samples_split=5, min_samples_leaf=2,
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            X, y_all_enc, balanced=False)
        xgb_mean, xgb_std = cross_validate(
            "XGB",
            lambda: XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.8, objective="multi:softprob",
                num_class=len(le.classes_), random_state=RANDOM_STATE, n_jobs=-1,
                eval_metric="mlogloss", tree_method="hist"),
            X, y_all_enc, balanced=True)
        print(f"  Random Forest   {rf_mean:.4f} +/- {rf_std:.4f}")
        print(f"  XGBoost         {xgb_mean:.4f} +/- {xgb_std:.4f}")
        print()

    # --- Summary table (paste-ready for the paper) ---
    print("=" * 68)
    print("SUMMARY")
    print("=" * 68)
    print(f"{'Model':<32}{'top-1':>9}{'top-2':>9}{'macroF1':>10}{'weightF1':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<32}{r['accuracy']:>9.4f}{r['top2']:>9.4f}"
              f"{r['macro_f1']:>10.4f}{r['weighted_f1']:>10.4f}")
    b1, b2 = prevalence_baseline(y, 1), prevalence_baseline(y, 2)
    print(f"{'prevalence baseline':<32}{b1:>9.4f}{b2:>9.4f}{'-':>10}{'-':>10}")
    print()
    print("Gain over prevalence baseline (the honest comparison):")
    for r in results:
        print(f"   {r['name']:<32}top-1 {r['accuracy'] - b1:+.4f}   "
              f"top-2 {r['top2'] - b2:+.4f}")
    print()

    best = max(results, key=lambda r: r["macro_f1"])
    delta = abs(results[0]["macro_f1"] - results[1]["macro_f1"])
    t2delta = abs(results[0]["top2"] - results[1]["top2"])
    print(f"Higher macro F1: {best['name']}  (difference: {delta:.4f})")
    best_t2 = max(results, key=lambda r: r["top2"])
    print(f"Higher top-2:    {best_t2['name']}  (difference: {t2delta:.4f})")
    if delta < 0.02:
        print()
        print("Macro F1 difference is within noise for this dataset size.")
        print("Decide on top-2, since that is the signal the recommender uses.")
    print()
    print("No artifacts were written. The deployed model is unchanged.")


if __name__ == "__main__":
    main()
