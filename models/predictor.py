import joblib
import os
import pandas as pd


class EmploymentTypePredictor:
    """
    RF employment type predictor.

    Feature engineering EXACTLY matches training pipeline:
      - 11 raw features
      - Age numeric + 10 categorical
      - pd.get_dummies with drop_first=False
      - Reindex to 108 training columns
    """

    SNAKE_TO_TRAIN = {
        "age": "Age",
        "sex": "Sex",
        "civil_status": "Civil_Status",
        "disability_type": "Disability_Type",
        "disability_visibility": "Disability_Visibility",
        "cause_of_disability": "Cause_of_Disability",
        "educational_attainment": "Educational_Attainment",
        "skills": "Skills",
        "mobility_status": "Mobility_Status",
        "current_assistive_device": "Current_Assistive_Device",
        "occupation_group": "Occupation_Group",
    }

    def __init__(self, artifacts_dir: str):
        self.model = joblib.load(os.path.join(artifacts_dir, "rf_model.pkl"))
        self.training_columns = joblib.load(
            os.path.join(artifacts_dir, "training_columns.pkl")
        )
        self.classes_ = joblib.load(os.path.join(artifacts_dir, "model_classes.pkl"))
        self.raw_features = joblib.load(os.path.join(artifacts_dir, "raw_features.pkl"))
        self.categorical_cols = joblib.load(
            os.path.join(artifacts_dir, "categorical_cols.pkl")
        )

        assert len(self.training_columns) == 108, (
            f"Expected 108 training columns, got {len(self.training_columns)}"
        )

        assert self.model.n_features_in_ == 108, (
            f"Model expects {self.model.n_features_in_} features, "
            f"but training_columns has {len(self.training_columns)}"
        )

        assert len(self.raw_features) == 11, (
            f"Expected 11 raw features, got {len(self.raw_features)}"
        )

    def predict(self, profile: dict) -> dict:
        """
        Args:
            profile: dict with snake_case keys matching PredictionRequest.

        Returns:
            dict with predicted_type, confidence, all_probabilities.
        """

        row = {
            train_col: profile[snake_col]
            for snake_col, train_col in self.SNAKE_TO_TRAIN.items()
        }

        df = pd.DataFrame([row], columns=self.raw_features)

        df_encoded = pd.get_dummies(
            df,
            columns=self.categorical_cols,
            drop_first=False,
        )

        df_aligned = df_encoded.reindex(
            columns=self.training_columns,
            fill_value=0,
        )

        recognized = int(df_aligned.iloc[0].sum() - df_aligned.iloc[0]["Age"])
        if recognized < 10:
            print(
                f"[predict] warning: only {recognized}/10 categorical values "
                f"recognized for profile. Check spelling/category vocabulary."
            )

        probs = self.model.predict_proba(df_aligned)[0]
        top_idx = probs.argmax()

        return {
            "predicted_type": str(self.classes_[top_idx]),
            "confidence": float(probs[top_idx]),
            "all_probabilities": {
                str(cls): float(p)
                for cls, p in zip(self.classes_, probs)
            },
        }