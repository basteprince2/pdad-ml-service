"""
Reproducible TF-IDF recommender artifact export script.

Purpose:
    Regenerates TF-IDF recommender artifacts used by the FastAPI ML service.

Expected input:
    A CSV dataset of job postings with at least:
      - job_post_id or id
      - job_title
      - job_description
      - required_skills
      - required_education
      - disability_friendly_notes

Outputs:
    ml_service/artifacts/tfidf_vectorizer.pkl
    ml_service/artifacts/job_tfidf_matrix.pkl
    ml_service/artifacts/job_ids.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


RANDOM_STATE = 42

JOB_TEXT_COLUMNS = [
    "job_title",
    "job_description",
    "required_skills",
    "required_education",
    "disability_friendly_notes",
]

POSSIBLE_ID_COLUMNS = [
    "job_post_id",
    "job_id",
    "id",
    "Job_ID",
    "Job ID",
]


def build_job_text(row: pd.Series) -> str:
    parts = []

    for column in JOB_TEXT_COLUMNS:
        if column in row.index and pd.notna(row[column]):
            parts.append(str(row[column]))

    return " ".join(parts).strip()


def resolve_job_ids(df: pd.DataFrame) -> list:
    for column in POSSIBLE_ID_COLUMNS:
        if column in df.columns:
            return df[column].astype(int).tolist()

    return list(range(1, len(df) + 1))


def train(input_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    available_text_columns = [
        column
        for column in JOB_TEXT_COLUMNS
        if column in df.columns
    ]

    if not available_text_columns:
        raise ValueError(
            "No expected job text columns found. Expected at least one of: "
            f"{JOB_TEXT_COLUMNS}"
        )

    job_texts = df.apply(build_job_text, axis=1).fillna("").tolist()
    job_ids = resolve_job_ids(df)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )

    job_tfidf_matrix = vectorizer.fit_transform(job_texts)

    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.pkl")
    joblib.dump(job_tfidf_matrix, output_dir / "job_tfidf_matrix.pkl")
    joblib.dump(job_ids, output_dir / "job_ids.pkl")

    print("TF-IDF recommender artifact export complete.")
    print(f"Rows: {len(df)}")
    print(f"Text columns used: {available_text_columns}")
    print(f"TF-IDF matrix shape: {job_tfidf_matrix.shape}")
    print(f"Job IDs exported: {len(job_ids)}")

    print("\nArtifacts exported:")
    print(f"- {output_dir / 'tfidf_vectorizer.pkl'}")
    print(f"- {output_dir / 'job_tfidf_matrix.pkl'}")
    print(f"- {output_dir / 'job_ids.pkl'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and export PDAD TF-IDF job recommender artifacts."
    )

    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to job postings CSV file.",
    )

    parser.add_argument(
        "--output-dir",
        default="ml_service/artifacts",
        help="Directory where recommender artifacts will be written.",
    )

    args = parser.parse_args()

    train(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()