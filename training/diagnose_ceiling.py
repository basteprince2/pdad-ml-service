"""
Diagnose the accuracy ceiling: is there signal left, or is 46.69% the max?

WHY THIS EXISTS
    Every model has now been tried - RF, XGBoost, LightGBM, voting
    ensemble, SMOTE, GridSearch, class merging, ordinal encoding - and
    none beat the deployed model by a meaningful margin. At that point a
    practitioner stops trying algorithms and starts diagnosing the data.

    This does not try to raise accuracy. It measures how much headroom
    exists, so the ceiling claim rests on evidence rather than exhaustion.

WHAT IT MEASURES

    1. Feature importance + permutation importance
       Which features carry signal, and which are dead weight. If
       everything is near zero, there is nothing to find.

    2. Label shuffle test (the key diagnostic)
       Retrain on randomly shuffled labels. That destroys any real
       relationship, so whatever score remains is what the model gets
       from memorising noise. The gap between real and shuffled is the
       true amount of learnable signal.

    3. Learning curve
       Does accuracy still climb with more rows, or has it flattened?
       Flat means more data would not help. Climbing means the dataset
       is too small, not the problem too hard.

    4. Bayes-error estimate via duplicate feature rows
       Rows with identical features but different labels are
       unpredictable by construction. Their frequency puts a hard upper
       bound on any model's accuracy.

    5. Per-class confusion concentration
       Whether errors are spread evenly or concentrated in specific
       class pairs. Concentrated errors sometimes point to a fixable
       labelling issue.

READ-ONLY. Writes nothing.

Usage:
    python ml_service/training/diagnose_ceiling.py \
        --input-csv database/seeders/data/pdad_registry.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

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
    return X, df[TARGET], df


def make():
    return RandomForestClassifier(
        n_estimators=200, min_samples_split=5, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)


def main():
    ap = argparse.ArgumentParser(description="Diagnose the accuracy ceiling.")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--shuffles", type=int, default=5)
    args = ap.parse_args()

    X, y, raw = load(Path(args.input_csv))
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    counts = y.value_counts()
    prevalence = counts.iloc[0] / len(y)

    print("=" * 74)
    print("CEILING DIAGNOSIS")
    print("=" * 74)
    print(f"Rows {len(X):,} | features {X.shape[1]} | classes {y.nunique()}")
    print(f"Prevalence baseline (always guess majority): {prevalence:.4f}")
    print()

    model = make().fit(Xtr, ytr)
    real_acc = accuracy_score(yte, model.predict(Xte))
    real_f1 = f1_score(yte, model.predict(Xte), average="macro", zero_division=0)
    print(f"Deployed model: accuracy {real_acc:.4f}, macro F1 {real_f1:.4f}")
    print()

    # ---------------------------------------------------------------
    print("=" * 74)
    print("1. FEATURE IMPORTANCE - where is the signal?")
    print("=" * 74)
    imp = pd.Series(model.feature_importances_, index=X.columns)
    grouped = {}
    for feat in RAW_FEATURES:
        if feat == "Age":
            grouped["Age"] = imp.get("Age", 0.0)
        else:
            cols = [c for c in X.columns if c.startswith(feat + "_")]
            grouped[feat] = imp[cols].sum() if cols else 0.0
    gs = pd.Series(grouped).sort_values(ascending=False)
    print(f"  {'feature':<28}{'gini importance':>18}{'share':>10}")
    print("  " + "-" * 56)
    for k, v in gs.items():
        bar = "#" * int(v * 100)
        print(f"  {k:<28}{v:>18.4f}{v / gs.sum() * 100:>9.1f}%  {bar}")
    print()
    print("  Gini importance is biased toward high-cardinality features,")
    print("  so permutation importance below is the more trustworthy read.")
    print()

    perm = permutation_importance(
        model, Xte, yte, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
        scoring="accuracy")
    pser = pd.Series(perm.importances_mean, index=X.columns)
    pgroup = {}
    for feat in RAW_FEATURES:
        if feat == "Age":
            pgroup["Age"] = pser.get("Age", 0.0)
        else:
            cols = [c for c in X.columns if c.startswith(feat + "_")]
            pgroup[feat] = pser[cols].sum() if cols else 0.0
    ps = pd.Series(pgroup).sort_values(ascending=False)
    print(f"  {'feature':<28}{'permutation drop':>18}")
    print("  " + "-" * 46)
    for k, v in ps.items():
        flag = "  <- no signal" if v <= 0.001 else ""
        print(f"  {k:<28}{v:>18.4f}{flag}")
    dead = [k for k, v in ps.items() if v <= 0.001]
    print()
    print(f"  {len(dead)} of {len(ps)} features contribute nothing measurable.")
    print()

    # ---------------------------------------------------------------
    print("=" * 74)
    print("2. LABEL SHUFFLE TEST - how much is real learning?")
    print("=" * 74)
    print("  Retraining on randomly shuffled labels. Any score above the")
    print("  prevalence baseline there is pure memorisation of noise.")
    print()
    rng = np.random.RandomState(RANDOM_STATE)
    shuffled = []
    for i in range(args.shuffles):
        y_shuf = pd.Series(rng.permutation(ytr.values), index=ytr.index)
        m = make().fit(Xtr, y_shuf)
        a = accuracy_score(yte, m.predict(Xte))
        shuffled.append(a)
        print(f"  shuffle {i + 1}: accuracy {a:.4f}")
    sh = float(np.mean(shuffled))
    print()
    print(f"  shuffled mean       {sh:.4f}")
    print(f"  real model          {real_acc:.4f}")
    print(f"  learnable signal    {real_acc - sh:+.4f}")
    print(f"  prevalence baseline {prevalence:.4f}")
    print()
    if real_acc - sh < 0.05:
        print("  The model barely outperforms randomised labels. Very little")
        print("  genuine structure exists between features and target.")
    else:
        print(f"  The model extracts {(real_acc - sh) * 100:.1f} points of real signal")
        print("  beyond what noise memorisation provides.")
    print()

    # ---------------------------------------------------------------
    print("=" * 74)
    print("3. LEARNING CURVE - would more data help?")
    print("=" * 74)
    print(f"  {'train rows':>12}{'CV accuracy':>14}{'change':>10}")
    print("  " + "-" * 36)
    prev = None
    for frac in [0.25, 0.5, 0.75, 1.0]:
        n = int(len(Xtr) * frac)
        idx = rng.choice(len(Xtr), n, replace=False)
        Xs, ys = Xtr.iloc[idx], ytr.iloc[idx]
        if ys.nunique() < y.nunique():
            print(f"  {n:>12}   (skipped - not all classes present)")
            continue
        skf = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        ya = np.asarray(ys)
        for tr, te in skf.split(Xs, ya):
            m = make().fit(Xs.iloc[tr], ya[tr])
            scores.append(accuracy_score(ya[te], m.predict(Xs.iloc[te])))
        sc = float(np.mean(scores))
        delta = f"{sc - prev:+.4f}" if prev is not None else "-"
        print(f"  {n:>12}{sc:>14.4f}{delta:>10}")
        prev = sc
    print()
    print("  If the last step is near zero, more rows would not help.")
    print()

    # ---------------------------------------------------------------
    print("=" * 74)
    print("4. IRREDUCIBLE ERROR - identical people, different labels")
    print("=" * 74)
    dupe = raw[RAW_FEATURES].astype(str).agg("|".join, axis=1)
    tmp = pd.DataFrame({"key": dupe, "y": raw[TARGET].values})
    grp = tmp.groupby("key")["y"]
    sizes = grp.size()
    nun = grp.nunique()
    conflicting = int(sizes[nun > 1].sum())
    groups_conflict = int((nun > 1).sum())
    print(f"  distinct feature combinations : {len(sizes):,}")
    print(f"  combinations appearing 2+ times: {int((sizes > 1).sum()):,}")
    print(f"  combinations with >1 label     : {groups_conflict:,}")
    print(f"  rows inside those groups       : {conflicting:,} "
          f"({conflicting / len(raw) * 100:.1f}%)")
    print()
    if groups_conflict:
        best = grp.apply(lambda s: s.value_counts().iloc[0]).sum()
        print(f"  Best possible accuracy if a perfect model always picked the")
        print(f"  most common label per feature combination: {best / len(raw):.4f}")
        print()
        print("  That is a hard ceiling. No model can beat it on this feature set.")
    else:
        print("  Every feature combination is unique, so this bound does not")
        print("  apply. With 108 one-hot features that is expected, and it")
        print("  means the model can in principle memorise the training set -")
        print("  which is why the shuffle test above matters more.")
    print()

    # ---------------------------------------------------------------
    print("=" * 74)
    print("5. WHERE THE ERRORS CONCENTRATE")
    print("=" * 74)
    labels = sorted(y.unique())
    cm = confusion_matrix(yte, model.predict(Xte), labels=labels)
    print(f"  {'true \\ pred':<16}" + "".join(f"{l[:7]:>9}" for l in labels))
    for i, l in enumerate(labels):
        print(f"  {l[:15]:<16}" + "".join(f"{cm[i][j]:>9}" for j in range(len(labels))))
    print()
    off = [(labels[i], labels[j], cm[i][j])
           for i in range(len(labels)) for j in range(len(labels)) if i != j]
    off.sort(key=lambda t: -t[2])
    print("  Largest confusions:")
    for a, b, n in off[:5]:
        print(f"    {a} predicted as {b}: {n} rows")
    print()
    print("=" * 74)
    print("Nothing was written. The deployed model is unchanged.")


if __name__ == "__main__":
    main()
