"""
Reproducible Random Forest training/export script for PDAD employment type prediction.

Purpose:
    Regenerates the Random Forest model artifacts used by the FastAPI ML service.

Expected input:
    A CSV dataset containing the 11 raw RF features and target column:
      - Age
      - Sex
      - Civil_Status
      - Disability_Type
      - Disability_Visibility
      - Cause_of_Disability
      - Educational_Attainment
      - Skills
      - Mobility_Status
      - Current_Assistive_Device
      - Occupation_Group
      - Type_of_Employment

Outputs:
    ml_service/artifacts/rf_model.pkl
    ml_service/artifacts/training_columns.pkl
    ml_service/artifacts/raw_features.pkl
    ml_service/artifacts/categorical_cols.pkl
    ml_service/artifacts/model_classes.pkl

Note:
    This script is intentionally explicit so retraining can be repeated with
    the same seed and artifact filenames.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


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

CATEGORICAL_COLS = [
    feature
    for feature in RAW_FEATURES
    if feature != "Age"
]


def train(input_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    missing_columns = [
        column
        for column in RAW_FEATURES + [TARGET_COLUMN]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[RAW_FEATURES + [TARGET_COLUMN]].copy()
    df = df.dropna(subset=[TARGET_COLUMN])

    # Basic data hygiene aligned with notebook export.
    df["Current_Assistive_Device"] = df["Current_Assistive_Device"].fillna("None")

    X = df[RAW_FEATURES]
    y = df[TARGET_COLUMN]

    X_encoded = pd.get_dummies(
        X,
        columns=CATEGORICAL_COLS,
        drop_first=False,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Random Forest training complete.")
    print(f"Rows: {len(df)}")
    print(f"Encoded feature count: {X_encoded.shape[1]}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, output_dir / "rf_model.pkl")
    joblib.dump(X_encoded.columns.tolist(), output_dir / "training_columns.pkl")
    joblib.dump(RAW_FEATURES, output_dir / "raw_features.pkl")
    joblib.dump(CATEGORICAL_COLS, output_dir / "categorical_cols.pkl")
    joblib.dump(list(model.classes_), output_dir / "model_classes.pkl")

    print("\nArtifacts exported:")
    print(f"- {output_dir / 'rf_model.pkl'}")
    print(f"- {output_dir / 'training_columns.pkl'}")
    print(f"- {output_dir / 'raw_features.pkl'}")
    print(f"- {output_dir / 'categorical_cols.pkl'}")
    print(f"- {output_dir / 'model_classes.pkl'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and export PDAD Random Forest employment prediction artifacts."
    )

    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to training CSV file.",
    )

    parser.add_argument(
        "--output-dir",
        default="ml_service/artifacts",
        help="Directory where model artifacts will be written.",
    )

    args = parser.parse_args()

    train(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()