"""
Does dropping the three harmful features improve the model?

WHY THIS EXISTS
    Permutation importance on the deployed model found three features
    with NEGATIVE contribution - shuffling them makes the model better:

        Civil_Status              -0.0041
        Mobility_Status           -0.0098
        Current_Assistive_Device  -0.0391

    Current_Assistive_Device is the striking one. It ranks 2nd on gini
    importance (17%) and last on permutation importance. That gap is the
    signature of a feature the model overfits to: 2,241 of 4,850 source
    rows had it missing and were filled with 'None', so it partly encodes
    "was this field blank" rather than a real attribute.

    Removing noise features is legitimate. This is not metric chasing -
    it is acting on a measurement.

WHAT IT TESTS
    Five feature sets, all on identical split and seed:
      1. All 11 (deployed)
      2. Drop Current_Assistive_Device only (the worst offender)
      3. Drop all three negatives
      4. Drop the three plus Sex (weakest positive)
      5. Occupation_Group alone (how much is one feature worth?)

    Each is evaluated on holdout AND 5x5-fold repeated CV, because a
    single split already misled us once on this project: GridSearch
    looked like +2.84 points on holdout and 86% of it was split luck.

THE COST IF IT WORKS
    Changing the feature set changes training_columns.pkl, which is the
    vocabulary validator's contract. That means a full Layer 5
    re-verification: the API would accept a different set of fields.
    Only worth it if the gain is clearly outside noise.

READ-ONLY. Writes nothing.

Usage (ml_service venv is enough - no xgboost needed):
    python ml_service/training/test_feature_removal.py \
        --input-csv database/seeders/data/pdad_registry.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

RANDOM_STATE = 42

ALL_FEATURES = [
    "Age", "Sex", "Civil_Status", "Disability_Type", "Disability_Visibility",
    "Cause_of_Disability", "Educational_Attainment", "Skills",
    "Mobility_Status", "Current_Assistive_Device", "Occupation_Group",
]
TARGET = "Type_of_Employment"
EXCLUDED_TARGETS = ["None/Others"]
TARGET_MERGES = {"Permanent/Regular": "Permanent"}

VARIANTS = {
    "all 11 (deployed)": ALL_FEATURES,
    "drop assistive_device": [f for f in ALL_FEATURES
                              if f != "Current_Assistive_Device"],
    "drop 3 negatives": [f for f in ALL_FEATURES
                         if f not in ("Civil_Status", "Mobility_Status",
                                      "Current_Assistive_Device")],
    "drop 3 negatives + Sex": [f for f in ALL_FEATURES
                               if f not in ("Civil_Status", "Mobility_Status",
                                            "Current_Assistive_Device", "Sex")],
    "Occupation_Group only": ["Occupation_Group"],
}


def normalise_headers(df):
    def key(n):
        return str(n).strip().lower().replace(" ", "_").replace("-", "_")
    lookup = {key(c): c for c in df.columns}
    rename, missing = {}, []
    for t in ALL_FEATURES + [TARGET]:
        src = lookup.get(key(t))
        if src is None:
            missing.append(t)
        elif src != t:
            rename[src] = t
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df.rename(columns=rename)


def load(path, features):
    df = normalise_headers(pd.read_csv(path))
    df = df[ALL_FEATURES + [TARGET]].copy()
    df = df.dropna(subset=[TARGET])
    df = df[~df[TARGET].isin(EXCLUDED_TARGETS)]
    df[TARGET] = df[TARGET].replace(TARGET_MERGES)
    if "Current_Assistive_Device" in df.columns:
        df["Current_Assistive_Device"] = df["Current_Assistive_Device"].fillna("None")
    cats = [f for f in features if f != "Age"]
    X = pd.get_dummies(df[features], columns=cats, drop_first=False)
    return X, df[TARGET]


def make():
    return RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=5,
        min_samples_leaf=2, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1)


def top_k(model, X_test, y_test, k):
    proba = model.predict_proba(X_test)
    classes = np.asarray(model.classes_)
    idx = np.argsort(proba, axis=1)[:, -k:]
    truth = np.asarray(y_test).reshape(-1, 1)
    return float((classes[idx] == truth).any(axis=1).mean())


def main():
    ap = argparse.ArgumentParser(
        description="Test whether dropping harmful features helps.")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--repeats", type=int, default=5,
                    help="Repeats of 5-fold CV (default 5 = 25 fits each).")
    args = ap.parse_args()

    path = Path(args.input_csv)

    print("=" * 78)
    print("FEATURE REMOVAL TEST")
    print("=" * 78)
    print("Permutation importance flagged three features as harmful:")
    print("  Civil_Status -0.0041 | Mobility_Status -0.0098 | "
          "Current_Assistive_Device -0.0391")
    print()

    # --- Holdout ---
    print("HOLDOUT (20% stratified split, random_state=42)")
    print("-" * 78)
    print(f"  {'feature set':<26}{'feats':>7}{'top-1':>9}{'top-2':>9}"
          f"{'macroF1':>10}{'vs base':>10}")
    print("  " + "-" * 71)

    holdout = {}
    base_acc = None
    for label, feats in VARIANTS.items():
        X, y = load(path, feats)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        m = make().fit(Xtr, ytr)
        pred = m.predict(Xte)
        acc = accuracy_score(yte, pred)
        t2 = top_k(m, Xte, yte, 2)
        mf1 = f1_score(yte, pred, average="macro", zero_division=0)
        if base_acc is None:
            base_acc = acc
        holdout[label] = (acc, t2, mf1)
        delta = "" if label.startswith("all 11") else f"{acc - base_acc:+.4f}"
        print(f"  {label:<26}{X.shape[1]:>7}{acc:>9.4f}{t2:>9.4f}"
              f"{mf1:>10.4f}{delta:>10}")

    # --- Repeated CV ---
    print()
    print(f"REPEATED CV ({args.repeats}x5-fold on all rows) - the reliable read")
    print("-" * 78)
    print(f"  {'feature set':<26}{'accuracy':>20}{'macro F1':>20}{'vs base':>10}")
    print("  " + "-" * 71)

    cv = {}
    base_cv = None
    for label, feats in VARIANTS.items():
        X, y = load(path, feats)
        y_arr = np.asarray(y)
        rskf = RepeatedStratifiedKFold(
            n_splits=5, n_repeats=args.repeats, random_state=RANDOM_STATE)
        accs, f1s = [], []
        for tr, te in rskf.split(X, y_arr):
            m = make().fit(X.iloc[tr], y_arr[tr])
            p = m.predict(X.iloc[te])
            accs.append(accuracy_score(y_arr[te], p))
            f1s.append(f1_score(y_arr[te], p, average="macro", zero_division=0))
        a, astd = float(np.mean(accs)), float(np.std(accs))
        f, fstd = float(np.mean(f1s)), float(np.std(f1s))
        if base_cv is None:
            base_cv = a
        cv[label] = (a, astd, f, fstd, np.array(accs))
        delta = "" if label.startswith("all 11") else f"{a - base_cv:+.4f}"
        print(f"  {label:<26}{a:>12.4f} +/-{astd:.4f}"
              f"{f:>12.4f} +/-{fstd:.4f}{delta:>10}")

    # --- Verdict ---
    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)

    from scipy.stats import ttest_rel
    base_scores = cv["all 11 (deployed)"][4]
    best_label, best_gain = None, 0.0

    for label in VARIANTS:
        if label.startswith("all 11") or label.startswith("Occupation_Group only"):
            continue
        scores = cv[label][4]
        gain = float(scores.mean() - base_scores.mean())
        t = ttest_rel(scores, base_scores)
        wins = int((scores > base_scores).sum())
        sig = "yes" if t.pvalue < 0.05 and gain > 0 else "no"
        print(f"  {label}")
        print(f"     CV gain {gain:+.4f} | won {wins}/{len(scores)} folds | "
              f"paired t-test p={t.pvalue:.4f} | significant: {sig}")
        if gain > best_gain and t.pvalue < 0.05:
            best_label, best_gain = label, gain
    print()

    single = cv["Occupation_Group only"][0]
    full = cv["all 11 (deployed)"][0]
    print(f"  Occupation_Group alone reaches {single:.4f} CV accuracy.")
    print(f"  All 11 features reach {full:.4f}.")
    print(f"  The other ten features are worth {full - single:+.4f} combined.")
    print()

    if best_label:
        print(f"  BEST: {best_label} ({best_gain:+.4f} CV accuracy, significant)")
        print()
        print("  Before acting on this, weigh the cost: changing the feature")
        print("  set changes training_columns.pkl, which is the vocabulary")
        print("  validator's contract. The API would accept a different set")
        print("  of fields, so Layer 5 needs full re-verification and the")
        print("  Laravel payload has to be updated to match.")
    else:
        print("  No variant beats the deployed feature set by a statistically")
        print("  significant margin on repeated CV.")
        print()
        print("  Report the permutation finding in MODEL_CARD.md as an")
        print("  observation and future work. Do not change the model.")
    print()
    print("Nothing was written. The deployed model is unchanged.")


if __name__ == "__main__":
    main()
