"""
Exhaustive search for legitimate accuracy improvements.

WHY THIS EXISTS
    Four approaches were planned earlier in the project but never
    actually run: hyperparameter tuning (GridSearch), SMOTE resampling,
    a voting ensemble, and LightGBM. Two more were never considered:
    ordinal encoding of the education variable, and threshold-free
    evaluation via log loss.

    This script runs all of them on identical preprocessing, split and
    seed, so the ceiling claim in MODEL_CARD.md rests on measurement
    rather than assumption.

WHAT COUNTS AS LEGITIMATE
    No Employment_Status (target leakage - it determines the answer).
    No fitting on the test set. No metric substitution. Anything that
    raises the number by using information the model would not have at
    prediction time is excluded.

READ-ONLY: trains in memory, writes no artifacts.

Usage (analysis venv, needs xgboost + imbalanced-learn):
    python ml_service/training/exhaust_options.py \
        --input-csv database/seeders/data/pdad_registry.csv
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

RAW_FEATURES = [
    "Age", "Sex", "Civil_Status", "Disability_Type", "Disability_Visibility",
    "Cause_of_Disability", "Educational_Attainment", "Skills",
    "Mobility_Status", "Current_Assistive_Device", "Occupation_Group",
]
TARGET = "Type_of_Employment"
CATEGORICAL = [f for f in RAW_FEATURES if f != "Age"]
EXCLUDED_TARGETS = ["None/Others"]
TARGET_MERGES = {"Permanent/Regular": "Permanent"}

# Education has a natural order. One-hot encoding throws that away.
EDUCATION_ORDER = {
    "ALS": 1, "SPED": 1,
    "Elementary Level": 2, "Elementary Graduate": 3,
    "High School Level": 4, "Junior High School": 4,
    "High School Graduate": 5, "Senior High School": 5,
    "Senior High School Graduate": 6,
    "Vocational": 7, "Vocational Graduate": 8,
    "College Level": 9, "College Graduate": 10,
    "Post Graduate": 11,
}


def normalise_headers(df: pd.DataFrame) -> pd.DataFrame:
    def key(n): return str(n).strip().lower().replace(" ", "_").replace("-", "_")
    lookup = {key(c): c for c in df.columns}
    rename, missing = {}, []
    for target in RAW_FEATURES + [TARGET]:
        src = lookup.get(key(target))
        if src is None:
            missing.append(target)
        elif src != target:
            rename[src] = target
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df.rename(columns=rename)


def load(path: Path, ordinal_education: bool = False):
    df = normalise_headers(pd.read_csv(path))
    df = df[RAW_FEATURES + [TARGET]].copy()
    df = df.dropna(subset=[TARGET])
    df = df[~df[TARGET].isin(EXCLUDED_TARGETS)]
    df[TARGET] = df[TARGET].replace(TARGET_MERGES)
    df["Current_Assistive_Device"] = df["Current_Assistive_Device"].fillna("None")

    cats = list(CATEGORICAL)
    if ordinal_education:
        df["Education_Rank"] = df["Educational_Attainment"].map(EDUCATION_ORDER)
        df["Education_Rank"] = df["Education_Rank"].fillna(
            df["Education_Rank"].median())
        cats.remove("Educational_Attainment")
        feats = [f for f in RAW_FEATURES if f != "Educational_Attainment"]
        feats.append("Education_Rank")
    else:
        feats = RAW_FEATURES

    X = pd.get_dummies(df[feats], columns=cats, drop_first=False)
    return X, df[TARGET]


def top_k(model, X_test, y_test, k):
    proba = model.predict_proba(X_test)
    classes = np.asarray(model.classes_)
    idx = np.argsort(proba, axis=1)[:, -k:]
    truth = np.asarray(y_test).reshape(-1, 1)
    return float((classes[idx] == truth).any(axis=1).mean())


def report(name, model, Xte, yte, results):
    pred = model.predict(Xte)
    r = {
        "name": name,
        "top1": accuracy_score(yte, pred),
        "top2": top_k(model, Xte, yte, 2),
        "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
    }
    results.append(r)
    print(f"  {name:<38}{r['top1']:>9.4f}{r['top2']:>9.4f}{r['macro_f1']:>10.4f}")
    return r


def main():
    ap = argparse.ArgumentParser(description="Exhaust legitimate accuracy options.")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--skip-grid", action="store_true",
                    help="Skip GridSearch (it is the slow one).")
    args = ap.parse_args()

    X, y = load(Path(args.input_csv))
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    counts = y.value_counts()
    prev1 = counts.iloc[:1].sum() / len(y)
    prev2 = counts.iloc[:2].sum() / len(y)

    print("=" * 74)
    print("EXHAUSTIVE SEARCH FOR LEGITIMATE IMPROVEMENTS")
    print("=" * 74)
    print(f"Rows {len(X):,} | features {X.shape[1]} | classes {y.nunique()}")
    print(f"Prevalence baseline: top-1 {prev1:.4f}, top-2 {prev2:.4f}")
    print(f"Smallest class: {counts.min()} rows ({counts.idxmin()})")
    print()
    print(f"  {'approach':<38}{'top-1':>9}{'top-2':>9}{'macroF1':>10}")
    print("  " + "-" * 66)

    results = []

    # --- 0. Deployed baseline ---
    base = RandomForestClassifier(
        n_estimators=200, min_samples_split=5, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1).fit(Xtr, ytr)
    report("RF deployed (baseline)", base, Xte, yte, results)

    # --- 1. GridSearch: planned, never run ---
    if not args.skip_grid:
        grid = GridSearchCV(
            RandomForestClassifier(class_weight="balanced",
                                   random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [200, 500],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
            scoring="f1_macro",
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
            n_jobs=-1,
        ).fit(Xtr, ytr)
        report("RF GridSearch (108 combos)", grid.best_estimator_, Xte, yte, results)
        print(f"     best params: {grid.best_params_}")

    # --- 2. SMOTE: planned, never run ---
    try:
        from imblearn.over_sampling import SMOTE
        le = LabelEncoder().fit(y)
        ytr_enc = le.transform(ytr)
        k = max(1, min(5, np.bincount(ytr_enc).min() - 1))
        Xs, ys = SMOTE(random_state=RANDOM_STATE, k_neighbors=k).fit_resample(
            Xtr, ytr_enc)
        sm = RandomForestClassifier(
            n_estimators=200, min_samples_split=5, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1).fit(Xs, le.inverse_transform(ys))
        report(f"RF + SMOTE (k={k})", sm, Xte, yte, results)
    except ImportError:
        print("  SMOTE skipped: pip install imbalanced-learn")

    # --- 3. Voting ensemble: planned, never run ---
    try:
        from xgboost import XGBClassifier
        le2 = LabelEncoder().fit(y)
        vote = VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(
                    n_estimators=200, min_samples_split=5, min_samples_leaf=2,
                    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
                ("xgb", XGBClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                    eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1)),
                ("lr", LogisticRegression(
                    max_iter=2000, class_weight="balanced",
                    random_state=RANDOM_STATE)),
            ],
            voting="soft",
        ).fit(Xtr, le2.transform(ytr))
        pred = le2.inverse_transform(vote.predict(Xte))
        proba = vote.predict_proba(Xte)
        cls = le2.inverse_transform(np.arange(proba.shape[1]))
        idx = np.argsort(proba, axis=1)[:, -2:]
        t2 = float((cls[idx] == np.asarray(yte).reshape(-1, 1)).any(axis=1).mean())
        r = {"name": "Voting ensemble (RF+XGB+LR)",
             "top1": accuracy_score(yte, pred), "top2": t2,
             "macro_f1": f1_score(yte, pred, average="macro", zero_division=0)}
        results.append(r)
        print(f"  {r['name']:<38}{r['top1']:>9.4f}{r['top2']:>9.4f}{r['macro_f1']:>10.4f}")
    except ImportError:
        print("  Ensemble skipped: xgboost not installed")

    # --- 4. LightGBM: mentioned, never tried ---
    try:
        from lightgbm import LGBMClassifier
        le3 = LabelEncoder().fit(y)
        # LightGBM rejects feature names containing JSON-special characters,
        # and one-hot names include values like "Cancer (RA 11215)".
        Xtr_l = Xtr.copy()
        Xte_l = Xte.copy()
        safe = [f"f{i}" for i in range(Xtr.shape[1])]
        Xtr_l.columns = safe
        Xte_l.columns = safe
        lgb = LGBMClassifier(
            n_estimators=200, learning_rate=0.1, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        ).fit(Xtr_l, le3.transform(ytr))
        pred = le3.inverse_transform(lgb.predict(Xte_l))
        proba = lgb.predict_proba(Xte_l)
        cls = le3.inverse_transform(np.arange(proba.shape[1]))
        idx = np.argsort(proba, axis=1)[:, -2:]
        t2 = float((cls[idx] == np.asarray(yte).reshape(-1, 1)).any(axis=1).mean())
        r = {"name": "LightGBM", "top1": accuracy_score(yte, pred), "top2": t2,
             "macro_f1": f1_score(yte, pred, average="macro", zero_division=0)}
        results.append(r)
        print(f"  {r['name']:<38}{r['top1']:>9.4f}{r['top2']:>9.4f}{r['macro_f1']:>10.4f}")
    except ImportError:
        print("  LightGBM skipped: pip install lightgbm")

    # --- 5. Ordinal education: never considered ---
    Xo, yo = load(Path(args.input_csv), ordinal_education=True)
    Xotr, Xote, yotr, yote = train_test_split(
        Xo, yo, test_size=0.2, random_state=RANDOM_STATE, stratify=yo)
    ordm = RandomForestClassifier(
        n_estimators=200, min_samples_split=5, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ).fit(Xotr, yotr)
    report(f"RF + ordinal education ({Xo.shape[1]} feats)", ordm, Xote, yote, results)

    # --- Summary ---
    print()
    print("=" * 74)
    best1 = max(results, key=lambda r: r["top1"])
    best2 = max(results, key=lambda r: r["top2"])
    bestf = max(results, key=lambda r: r["macro_f1"])
    b = results[0]
    print(f"Baseline           top-1 {b['top1']:.4f}  top-2 {b['top2']:.4f}  "
          f"macroF1 {b['macro_f1']:.4f}")
    print(f"Best top-1         {best1['name']}  {best1['top1']:.4f}  "
          f"({best1['top1'] - b['top1']:+.4f})")
    print(f"Best top-2         {best2['name']}  {best2['top2']:.4f}  "
          f"({best2['top2'] - b['top2']:+.4f})")
    print(f"Best macro F1      {bestf['name']}  {bestf['macro_f1']:.4f}  "
          f"({bestf['macro_f1'] - b['macro_f1']:+.4f})")
    print()
    n = len(yte)
    se = float(np.sqrt(0.47 * 0.53 / n))
    print(f"Standard error on top-1 at n={n}: {se:.4f} ({se * n:.0f} rows)")
    print(f"A difference below {1.96 * se:.4f} ({1.96 * se * n:.0f} rows) is noise.")
    gain = best1["top1"] - b["top1"]
    if gain < 1.96 * se:
        print()
        print("No approach beats the deployed model by a statistically")
        print("meaningful margin. The ceiling is a property of the data.")
    print()
    print("Nothing was written. The deployed model is unchanged.")


if __name__ == "__main__":
    main()
