"""
RF class-weight diagnostic for PDAD employment type prediction.

WHY THIS EXISTS
    The deployed Random Forest uses class_weight="balanced". On this dataset
    that over-corrects: the largest class (Self-employed, 38% of rows) is
    almost never predicted (recall ~0.02) while tiny classes are massively
    over-predicted. This script tests whether a different class_weight fixes
    it, which would keep the deployed architecture and require replacing only
    rf_model.pkl.

READ-ONLY: trains in memory and prints metrics. Writes no artifacts.

Usage (from repo root):
    python ml_service/training/tune_rf.py \
        --input-csv database/seeders/data/pdad_registry.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split

RANDOM_STATE = 42

RAW_FEATURES = [
    "Age", "Sex", "Civil_Status", "Disability_Type", "Disability_Visibility",
    "Cause_of_Disability", "Educational_Attainment", "Skills",
    "Mobility_Status", "Current_Assistive_Device", "Occupation_Group",
]
TARGET_COLUMN = "Type_of_Employment"
CATEGORICAL_COLS = [f for f in RAW_FEATURES if f != "Age"]

EXCLUDED_TARGETS = ["None/Others"]
TARGET_MERGES = {"Permanent/Regular": "Permanent"}

MAJORITY_CLASS = "Self-employed"


def normalise_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Accept PascalCase or lowercase CSV headers."""
    def key(name: str) -> str:
        return str(name).strip().lower().replace(" ", "_").replace("-", "_")

    lookup = {key(c): c for c in df.columns}
    rename, missing = {}, []
    for target in RAW_FEATURES + [TARGET_COLUMN]:
        source = lookup.get(key(target))
        if source is None:
            missing.append(target)
        elif source != target:
            rename[source] = target
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df.rename(columns=rename)


def load(input_csv: Path):
    df = normalise_headers(pd.read_csv(input_csv))
    df = df[RAW_FEATURES + [TARGET_COLUMN]].copy()
    df = df.dropna(subset=[TARGET_COLUMN])
    df = df[~df[TARGET_COLUMN].isin(EXCLUDED_TARGETS)]
    df[TARGET_COLUMN] = df[TARGET_COLUMN].replace(TARGET_MERGES)
    df["Current_Assistive_Device"] = df["Current_Assistive_Device"].fillna("None")
    X = pd.get_dummies(df[RAW_FEATURES], columns=CATEGORICAL_COLS, drop_first=False)
    return X, df[TARGET_COLUMN]


def make_rf(class_weight):
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test RF class_weight variants (read-only)."
    )
    parser.add_argument("--input-csv", required=True)
    args = parser.parse_args()

    X, y = load(Path(args.input_csv))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("=" * 78)
    print("RANDOM FOREST class_weight DIAGNOSTIC")
    print("=" * 78)
    print(f"Rows: {len(X):,}   train/test: {len(X_train):,}/{len(X_test):,}   "
          f"classes: {y.nunique()}")
    print(f"Majority class: {MAJORITY_CLASS} "
          f"({(y == MAJORITY_CLASS).mean() * 100:.1f}% of rows)")
    print()

    variants = {
        "balanced (DEPLOYED)": "balanced",
        "None (no weighting)": None,
        "balanced_subsample": "balanced_subsample",
    }

    print(f"{'class_weight':<24}{'Accuracy':>10}{'Macro F1':>10}"
          f"{'Weighted':>10}{'SelfEmp recall':>16}")
    print("-" * 78)

    rows = []
    for label, cw in variants.items():
        model = make_rf(cw)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        macro = f1_score(y_test, pred, average="macro", zero_division=0)
        weighted = f1_score(y_test, pred, average="weighted", zero_division=0)
        maj = recall_score(y_test, pred, labels=[MAJORITY_CLASS],
                           average="macro", zero_division=0)
        rows.append((label, cw, acc, macro, weighted, maj))
        print(f"{label:<24}{acc:>10.4f}{macro:>10.4f}{weighted:>10.4f}{maj:>16.4f}")

    print()
    print("5-fold stratified CV (macro F1, more reliable than one split):")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_arr = y.to_numpy()
    for label, cw, *_ in rows:
        scores = []
        for tr, te in skf.split(X, y_arr):
            m = make_rf(cw)
            m.fit(X.iloc[tr], y_arr[tr])
            scores.append(f1_score(y_arr[te], m.predict(X.iloc[te]),
                                   average="macro", zero_division=0))
        print(f"  {label:<24}{np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    print()
    print("=" * 78)
    best_weighted = max(rows, key=lambda r: r[4])
    deployed = rows[0]
    print(f"Deployed (balanced):  weighted F1 {deployed[4]:.4f}, "
          f"{MAJORITY_CLASS} recall {deployed[5]:.4f}")
    print(f"Best weighted F1:     {best_weighted[0]} -> {best_weighted[4]:.4f}, "
          f"{MAJORITY_CLASS} recall {best_weighted[5]:.4f}")
    if best_weighted[0] != deployed[0]:
        print()
        print("If the winner is not the deployed config, retraining RF with that")
        print("class_weight changes ONLY rf_model.pkl. training_columns.pkl,")
        print("categorical_cols.pkl, raw_features.pkl and model_classes.pkl stay")
        print("identical, so the vocabulary validator contract is unaffected.")
    print()
    print("No artifacts were written.")


if __name__ == "__main__":
    main()
