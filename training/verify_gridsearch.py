"""
Is the GridSearch improvement real, or a lucky split?

WHY THIS EXISTS
    GridSearch moved top-1 from 0.4669 to 0.4953 (+9 rows of 317), and
    top-2 and macro F1 rose with it. exhaust_options.py declared that
    "noise" using a two-proportion standard error, but that test assumes
    two independent samples. Both models were scored on the SAME 317
    rows, so the correct test is McNemar's, which only looks at the rows
    where the two models disagree and therefore has more power.

    A single 80/20 split is also a thin basis for a decision. This runs
    repeated stratified cross-validation so the comparison rests on the
    whole dataset rather than one lucky partition.

WHAT IT DOES NOT DO
    No retraining of deployed artifacts. Read-only.

Usage (analysis venv or ml_service venv - no xgboost needed):
    python ml_service/training/verify_gridsearch.py \
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

RAW_FEATURES = [
    "Age", "Sex", "Civil_Status", "Disability_Type", "Disability_Visibility",
    "Cause_of_Disability", "Educational_Attainment", "Skills",
    "Mobility_Status", "Current_Assistive_Device", "Occupation_Group",
]
TARGET = "Type_of_Employment"
CATEGORICAL = [f for f in RAW_FEATURES if f != "Age"]
EXCLUDED_TARGETS = ["None/Others"]
TARGET_MERGES = {"Permanent/Regular": "Permanent"}

DEPLOYED = dict(n_estimators=200, max_depth=None, min_samples_split=5,
                min_samples_leaf=2, max_features="sqrt")
CANDIDATE = dict(n_estimators=500, max_depth=None, min_samples_split=5,
                 min_samples_leaf=1, max_features="sqrt")


def normalise_headers(df):
    def key(n): return str(n).strip().lower().replace(" ", "_").replace("-", "_")
    lookup = {key(c): c for c in df.columns}
    rename, missing = {}, []
    for t in RAW_FEATURES + [TARGET]:
        src = lookup.get(key(t))
        if src is None:
            missing.append(t)
        elif src != t:
            rename[src] = t
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df.rename(columns=rename)


def load(path):
    df = normalise_headers(pd.read_csv(path))
    df = df[RAW_FEATURES + [TARGET]].copy()
    df = df.dropna(subset=[TARGET])
    df = df[~df[TARGET].isin(EXCLUDED_TARGETS)]
    df[TARGET] = df[TARGET].replace(TARGET_MERGES)
    df["Current_Assistive_Device"] = df["Current_Assistive_Device"].fillna("None")
    X = pd.get_dummies(df[RAW_FEATURES], columns=CATEGORICAL, drop_first=False)
    return X, df[TARGET]


def make(params, seed=RANDOM_STATE):
    return RandomForestClassifier(
        class_weight="balanced", random_state=seed, n_jobs=-1, **params)


def mcnemar(y_true, pred_a, pred_b):
    """Exact binomial McNemar on paired predictions."""
    from scipy.stats import binomtest
    a_right = np.asarray(pred_a) == np.asarray(y_true)
    b_right = np.asarray(pred_b) == np.asarray(y_true)
    b_only = int((~a_right & b_right).sum())   # candidate fixes
    a_only = int((a_right & ~b_right).sum())   # candidate breaks
    n = b_only + a_only
    if n == 0:
        return b_only, a_only, 1.0
    p = binomtest(b_only, n, 0.5).pvalue
    return b_only, a_only, float(p)


def main():
    ap = argparse.ArgumentParser(description="Verify the GridSearch result.")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--repeats", type=int, default=5,
                    help="Repeats of 5-fold CV (default 5 = 25 fits per model).")
    args = ap.parse_args()

    X, y = load(Path(args.input_csv))
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    print("=" * 72)
    print("IS THE GRIDSEARCH IMPROVEMENT REAL?")
    print("=" * 72)
    print(f"Rows {len(X):,} | train {len(Xtr):,} | test {len(Xte):,}")
    print(f"deployed : {DEPLOYED}")
    print(f"candidate: {CANDIDATE}")
    print()

    # --- Test 1: McNemar on the paired holdout predictions ---
    dep = make(DEPLOYED).fit(Xtr, ytr)
    cand = make(CANDIDATE).fit(Xtr, ytr)
    pd_, pc = dep.predict(Xte), cand.predict(Xte)

    print("TEST 1 - McNemar on the same 317 test rows")
    print("-" * 72)
    fixed, broke, p = mcnemar(yte, pd_, pc)
    print(f"  candidate correct where deployed wrong : {fixed}")
    print(f"  deployed correct where candidate wrong : {broke}")
    print(f"  net                                    : {fixed - broke:+d} rows")
    print(f"  exact binomial p-value                 : {p:.4f}")
    verdict1 = "significant at 0.05" if p < 0.05 else "NOT significant at 0.05"
    print(f"  -> {verdict1}")
    print()

    # --- Test 2: repeated CV over the whole dataset ---
    print(f"TEST 2 - {args.repeats}x5-fold CV on all {len(X):,} rows")
    print("-" * 72)
    rskf = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=args.repeats, random_state=RANDOM_STATE)
    y_arr = np.asarray(y)
    acc_d, acc_c, f1_d, f1_c = [], [], [], []

    for tr, te in rskf.split(X, y_arr):
        Xa, Xb = X.iloc[tr], X.iloc[te]
        ya, yb = y_arr[tr], y_arr[te]
        md = make(DEPLOYED).fit(Xa, ya)
        mc = make(CANDIDATE).fit(Xa, ya)
        p_d, p_c = md.predict(Xb), mc.predict(Xb)
        acc_d.append(accuracy_score(yb, p_d))
        acc_c.append(accuracy_score(yb, p_c))
        f1_d.append(f1_score(yb, p_d, average="macro", zero_division=0))
        f1_c.append(f1_score(yb, p_c, average="macro", zero_division=0))

    acc_d, acc_c = np.array(acc_d), np.array(acc_c)
    f1_d, f1_c = np.array(f1_d), np.array(f1_c)

    print(f"  {'metric':<12}{'deployed':>18}{'candidate':>18}{'diff':>10}")
    print(f"  {'accuracy':<12}{acc_d.mean():>10.4f} +/-{acc_d.std():.4f}"
          f"{acc_c.mean():>10.4f} +/-{acc_c.std():.4f}{acc_c.mean()-acc_d.mean():>10.4f}")
    print(f"  {'macro F1':<12}{f1_d.mean():>10.4f} +/-{f1_d.std():.4f}"
          f"{f1_c.mean():>10.4f} +/-{f1_c.std():.4f}{f1_c.mean()-f1_d.mean():>10.4f}")
    print()
    wins = int((acc_c > acc_d).sum())
    print(f"  candidate won {wins} of {len(acc_d)} folds on accuracy")

    # Paired t-test across folds
    from scipy.stats import ttest_rel
    t_acc = ttest_rel(acc_c, acc_d)
    t_f1 = ttest_rel(f1_c, f1_d)
    print(f"  paired t-test accuracy : p = {t_acc.pvalue:.4f}")
    print(f"  paired t-test macro F1 : p = {t_f1.pvalue:.4f}")
    print()

    # --- Test 3: seed stability on the holdout ---
    print("TEST 3 - Seed stability (holdout, 5 different random_state)")
    print("-" * 72)
    ds, cs = [], []
    for seed in [42, 7, 123, 2024, 99]:
        a = accuracy_score(yte, make(DEPLOYED, seed).fit(Xtr, ytr).predict(Xte))
        b = accuracy_score(yte, make(CANDIDATE, seed).fit(Xtr, ytr).predict(Xte))
        ds.append(a)
        cs.append(b)
        print(f"  seed {seed:<6} deployed {a:.4f}   candidate {b:.4f}   {b-a:+.4f}")
    ds, cs = np.array(ds), np.array(cs)
    print(f"  mean       deployed {ds.mean():.4f}   candidate {cs.mean():.4f}"
          f"   {cs.mean()-ds.mean():+.4f}")
    print()

    # --- Verdict ---
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    signals = {
        "McNemar p < 0.05": p < 0.05,
        "CV accuracy higher": acc_c.mean() > acc_d.mean(),
        "CV macro F1 higher": f1_c.mean() > f1_d.mean(),
        "CV accuracy t-test p < 0.05": t_acc.pvalue < 0.05,
        "won majority of folds": wins > len(acc_d) / 2,
        "higher across all seeds": bool((cs > ds).all()),
    }
    for k, v in signals.items():
        print(f"  {'YES' if v else 'no ':<4} {k}")
    score = sum(signals.values())
    print()
    if score >= 4:
        print(f"  {score}/6 signals agree: the improvement looks REAL.")
        print("  Worth retraining and redeploying rf_model.pkl.")
    elif score >= 2:
        print(f"  {score}/6 signals agree: WEAK evidence.")
        print("  Defensible either way. Report both and keep the deployed model")
        print("  unless the gain matters for the argument you are making.")
    else:
        print(f"  {score}/6 signals agree: the holdout gain was a lucky split.")
        print("  Keep the deployed model.")
    print()
    print("Nothing was written. The deployed model is unchanged.")


if __name__ == "__main__":
    main()
