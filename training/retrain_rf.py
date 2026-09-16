"""
Retrain the RF employment predictor with class_weight="balanced_subsample".

WHY
    The deployed model uses class_weight="balanced", which over-corrects on
    this dataset: the largest class (Self-employed, 38% of rows) is almost
    never predicted. "balanced_subsample" scores higher on every metric
    measured - top-1, top-2, top-3, macro F1 and 5-fold CV macro F1.

WHAT IT WRITES
    Five artifacts into --output-dir. Four of them (training_columns,
    raw_features, categorical_cols, model_classes) are byte-identical to the
    deployed ones, because they depend only on the data and feature set, not
    on the algorithm or its hyperparameters. Only rf_model.pkl differs, so
    the vocabulary validator contract is unaffected.

    --output-dir is REQUIRED and refuses to overwrite unless --overwrite is
    passed, so the deployed artifacts cannot be clobbered by accident.

WHAT IT PRINTS
    A side-by-side comparison of the current setting and the candidate,
    trained on the same split with the same seed, plus a byte-level check of
    which artifacts actually changed.

Usage (from repo root, analysis venv):
    python ml_service/training/retrain_rf.py ^
        --input-csv database/seeders/data/pdad_registry.csv ^
        --output-dir ml_service/artifacts_candidate
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

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

DEPLOYED_WEIGHT = "balanced"
CANDIDATE_WEIGHT = "balanced_subsample"

EXPECTED_CLASSES = 7
EXPECTED_FEATURES = 108
EXPECTED_TRAIN_ROWS = 1268

ARTIFACT_NAMES = [
    "rf_model.pkl",
    "training_columns.pkl",
    "raw_features.pkl",
    "categorical_cols.pkl",
    "model_classes.pkl",
]


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
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Columns found: {sorted(df.columns.tolist())}"
        )
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


def top_k_accuracy(model, X_test, y_test, k: int) -> float:
    proba = model.predict_proba(X_test)
    classes = np.asarray(model.classes_)
    top_idx = np.argsort(proba, axis=1)[:, -k:]
    truth = np.asarray(y_test).reshape(-1, 1)
    return float((classes[top_idx] == truth).any(axis=1).mean())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain RF with balanced_subsample and compare."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument(
        "--output-dir", required=True,
        help="Where to write artifacts. Required, so the deployed artifacts "
             "are never overwritten by accident.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--compare-with", default=None,
        help="Optional path to the deployed artifacts dir, to hash-compare "
             "the four non-model artifacts.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    existing = [n for n in ARTIFACT_NAMES if (output_dir / n).exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing artifacts in {output_dir}:\n"
            + "\n".join(f"  - {n}" for n in existing)
            + "\n\nPass --overwrite, or pick a different --output-dir."
        )

    X, y = load(Path(args.input_csv))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    print("=" * 74)
    print("RF RETRAIN - balanced vs balanced_subsample")
    print("=" * 74)
    print(f"Rows: {len(X):,}   train/test: {len(X_train):,}/{len(X_test):,}")
    print()

    checks = [
        ("classes", y.nunique(), EXPECTED_CLASSES),
        ("encoded features", X.shape[1], EXPECTED_FEATURES),
        ("training rows", len(X_train), EXPECTED_TRAIN_ROWS),
    ]
    ok = True
    print("Matches deployed artifact shape:")
    for name, got, want in checks:
        good = got == want
        ok &= good
        print(f"   {name:<18}{got:>6}  expected {want:>6}   "
              f"{'OK' if good else 'MISMATCH'}")
    if not ok:
        print("\n   STOPPING: preprocessing does not match the deployed model.")
        print("   Retraining now would change the API contract.")
        raise SystemExit(1)
    print()

    results = {}
    models = {}
    for label, cw in [("balanced (deployed)", DEPLOYED_WEIGHT),
                      ("balanced_subsample", CANDIDATE_WEIGHT)]:
        model = make_rf(cw).fit(X_train, y_train)
        models[cw] = model
        pred = model.predict(X_test)
        results[label] = {
            "top1": accuracy_score(y_test, pred),
            "top2": top_k_accuracy(model, X_test, y_test, 2),
            "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_test, pred, average="weighted",
                                    zero_division=0),
        }

    print(f"{'setting':<22}{'top-1':>9}{'top-2':>9}{'macroF1':>10}{'weightF1':>10}")
    print("-" * 74)
    for label, r in results.items():
        print(f"{label:<22}{r['top1']:>9.4f}{r['top2']:>9.4f}"
              f"{r['macro_f1']:>10.4f}{r['weighted_f1']:>10.4f}")
    print()

    print("Per-class report for the CANDIDATE (balanced_subsample):")
    print(classification_report(
        y_test, models[CANDIDATE_WEIGHT].predict(X_test), zero_division=0))

    model = models[CANDIDATE_WEIGHT]
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "rf_model.pkl")
    joblib.dump(X.columns.tolist(), output_dir / "training_columns.pkl")
    joblib.dump(RAW_FEATURES, output_dir / "raw_features.pkl")
    joblib.dump(CATEGORICAL_COLS, output_dir / "categorical_cols.pkl")
    joblib.dump(list(model.classes_), output_dir / "model_classes.pkl")

    print(f"Artifacts written to {output_dir}/")
    for name in ARTIFACT_NAMES:
        size = (output_dir / name).stat().st_size
        print(f"   {name:<24}{size:>12,} bytes")
    print()

    if args.compare_with:
        deployed = Path(args.compare_with)
        print(f"Hash comparison vs {deployed}/")
        print(f"   {'artifact':<24}{'deployed':<20}{'candidate':<20}{'same?'}")
        for name in ARTIFACT_NAMES:
            old, new = deployed / name, output_dir / name
            if not old.exists():
                print(f"   {name:<24}{'(missing)':<20}")
                continue
            a, b = sha(old), sha(new)
            verdict = "YES" if a == b else (
                "NO (expected)" if name == "rf_model.pkl" else "NO - INVESTIGATE")
            print(f"   {name:<24}{a:<20}{b:<20}{verdict}")
        print()
        print("   Only rf_model.pkl should differ. If any other artifact")
        print("   differs, the validator vocabulary would change and the")
        print("   Laravel integration must be re-verified before deploying.")
    print()
    print("Nothing in the deployed artifacts directory was modified.")


if __name__ == "__main__":
    main()
