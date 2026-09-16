"""
Decision table for the PDAD employment type predictor.

WHY THIS EXISTS
    Three separate questions have to be answered together before deciding
    what to deploy and what to tell the technical adviser:

      1. 7-class (deployed) vs 3-class merge (proven in Colab, never deployed)
      2. class_weight: "balanced" (deployed) vs "balanced_subsample"
      3. top-1 vs top-2 vs top-3 accuracy

    Question 3 matters because the system does NOT use the prediction as a
    verdict. The recommender uses it as a ranking signal (matching jobs get a
    similarity boost; non-matching jobs are still shown). For a ranking
    system, "is the true class in the model's top 2?" is a legitimate and
    arguably more faithful metric than top-1.

    Top-k is NOT a trick. The API already returns all_probabilities for every
    class, so top-k reflects information the system genuinely has and uses.
    It must, however, be reported explicitly as top-k, never as plain
    "accuracy".

READ-ONLY: trains in memory, prints a table. Writes no artifacts.

Usage (from repo root):
    python ml_service/training/decision_table.py \
        --input-csv database/seeders/data/pdad_registry.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
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

# 3-class permanence-based grouping (from the Colab experiment).
CLASS_3_MAP = {
    "Permanent": "Regular",
    "Probationary": "Regular",
    "Casual": "Non-permanent",
    "Contractual": "Non-permanent",
    "Job Order": "Non-permanent",
    "Seasonal": "Non-permanent",
    "Self-employed": "Self-employed",
}


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


def top_k_accuracy(model, X_test, y_test, k: int) -> float:
    """Fraction of rows where the true label is among the model's top k."""
    proba = model.predict_proba(X_test)
    classes = np.asarray(model.classes_)
    # Indices of the k highest probabilities per row.
    top_idx = np.argsort(proba, axis=1)[:, -k:]
    top_labels = classes[top_idx]
    truth = np.asarray(y_test).reshape(-1, 1)
    return float((top_labels == truth).any(axis=1).mean())


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


def cv_macro_f1(class_weight, X, y) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_arr = np.asarray(y)
    scores = []
    for tr, te in skf.split(X, y_arr):
        m = make_rf(class_weight).fit(X.iloc[tr], y_arr[tr])
        scores.append(f1_score(y_arr[te], m.predict(X.iloc[te]),
                               average="macro", zero_division=0))
    return float(np.mean(scores)), float(np.std(scores))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decision table: class granularity x class_weight x top-k."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()

    X, y7 = load(Path(args.input_csv))
    y3 = y7.map(CLASS_3_MAP)

    if y3.isna().any():
        bad = sorted(y7[y3.isna()].unique())
        raise ValueError(f"Labels missing from CLASS_3_MAP: {bad}")

    print("=" * 76)
    print("PDAD EMPLOYMENT PREDICTOR - DECISION TABLE")
    print("=" * 76)
    print(f"Rows: {len(X):,}   Encoded features: {X.shape[1]}")
    print()
    print("7-class (deployed):")
    for label, n in y7.value_counts().items():
        print(f"   {label:<16}{n:>6,}  {n / len(y7) * 100:5.1f}%")
    print()
    print("3-class (permanence-based merge):")
    for label, n in y3.value_counts().items():
        print(f"   {label:<16}{n:>6,}  {n / len(y3) * 100:5.1f}%")
    print()

    targets = [("7-class", y7), ("3-class", y3)]
    weights = [("balanced", "balanced"),
               ("balanced_subsample", "balanced_subsample")]

    print("=" * 76)
    print("HOLDOUT RESULTS (20% stratified split, random_state=42)")
    print("=" * 76)
    header = (f"{'target':<9}{'class_weight':<21}{'top-1':>9}{'top-2':>9}"
              f"{'top-3':>9}{'macroF1':>10}{'random':>9}")
    print(header)
    print("-" * 76)

    rows = []
    for tname, yv in targets:
        n_classes = yv.nunique()
        for wname, cw in weights:
            Xtr, Xte, ytr, yte = train_test_split(
                X, yv, test_size=0.2, random_state=RANDOM_STATE, stratify=yv)
            model = make_rf(cw).fit(Xtr, ytr)
            t1 = top_k_accuracy(model, Xte, yte, 1)
            t2 = top_k_accuracy(model, Xte, yte, 2)
            t3 = top_k_accuracy(model, Xte, yte, 3)
            mf1 = f1_score(yte, model.predict(Xte), average="macro",
                           zero_division=0)
            rnd = 1.0 / n_classes
            rows.append((tname, wname, n_classes, t1, t2, t3, mf1))
            print(f"{tname:<9}{wname:<21}{t1:>9.4f}{t2:>9.4f}{t3:>9.4f}"
                  f"{mf1:>10.4f}{rnd:>9.4f}")

    print()
    if not args.skip_cv:
        print("5-fold CV macro F1 (more reliable than a single split):")
        for tname, yv in targets:
            for wname, cw in weights:
                mean, std = cv_macro_f1(cw, X, yv)
                print(f"   {tname:<9}{wname:<21}{mean:.4f} +/- {std:.4f}")
        print()

    print("=" * 76)
    print("HOW TO READ THIS")
    print("=" * 76)
    print("top-1  = the single predicted class is correct.")
    print("         This is what MODEL_CARD.md currently reports.")
    print("top-2  = the true class is in the model's two most likely classes.")
    print("         Legitimate for this system ONLY because the recommender")
    print("         ranks rather than decides, and the API already exposes")
    print("         all_probabilities. Must be labelled 'top-2 accuracy',")
    print("         never plain 'accuracy'.")
    print("random = accuracy of guessing uniformly, for baseline comparison.")
    print()
    print("3-class merges Permanent+Probationary -> Regular and")
    print("Casual+Contractual+Job Order+Seasonal -> Non-permanent.")
    print("It scores higher, but the recommender compares the predicted type")
    print("against job_posts.employment_type, which stores the ORIGINAL 7")
    print("labels. Deploying 3-class therefore requires a group-mapping step")
    print("in the recommender, or it will never match any job.")
    print()
    print("No artifacts were written. The deployed model is unchanged.")


if __name__ == "__main__":
    main()
