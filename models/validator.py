"""
Vocabulary validator for the RF employment type predictor.

Derives the set of valid categorical values directly from the trained
model's `training_columns.pkl` at startup, so the vocabulary always
matches the deployed model and needs no hardcoding. If the model is
retrained, the valid vocabulary updates automatically.

Design:
  - STRICT for the 9 true categorical features. An unrecognized value
    raises HTTP 422 with the list of valid options.
  - LENIENT for `Skills`. Skills reach the model as a comma-joined
    multi-skill string (Laravel: skills->pluck->implode), which does not
    match the single-value categories the RF was trained on. Rejecting
    them would break every prediction, so Skills is not validated here.
    Its encoding-mismatch is tracked separately as a known limitation.
"""

from fastapi import HTTPException


# snake_case request field -> PascalCase training column prefix.
# Must match EmploymentTypePredictor.SNAKE_TO_TRAIN exactly.
SNAKE_TO_TRAIN = {
    "sex": "Sex",
    "civil_status": "Civil_Status",
    "disability_type": "Disability_Type",
    "disability_visibility": "Disability_Visibility",
    "cause_of_disability": "Cause_of_Disability",
    "educational_attainment": "Educational_Attainment",
    "mobility_status": "Mobility_Status",
    "current_assistive_device": "Current_Assistive_Device",
    "occupation_group": "Occupation_Group",
}

# Deliberately excluded from strict validation (see module docstring).
LENIENT_FIELDS = {"skills"}


class VocabularyValidator:
    """Validates categorical inputs against the model's learned vocabulary."""

    def __init__(self, training_columns: list[str], categorical_cols: list[str]):
        # Build {training_column_prefix: {valid values}} by parsing the
        # dummy column names, e.g. "Disability_Type_Visual Disability"
        # -> feature "Disability_Type", value "Visual Disability".
        self.valid_values: dict[str, set[str]] = {}

        for train_col in categorical_cols:
            prefix = f"{train_col}_"
            values = {
                col[len(prefix):]
                for col in training_columns
                if col.startswith(prefix)
            }
            if values:
                self.valid_values[train_col] = values

    def validate(self, profile: dict) -> None:
        """
        Check each strict categorical field against the learned vocabulary.

        Raises:
            HTTPException(422) on the first unrecognized value, listing the
            valid options for that field.
        """
        for snake_field, train_col in SNAKE_TO_TRAIN.items():
            if snake_field in LENIENT_FIELDS:
                continue

            value = profile.get(snake_field)
            valid_set = self.valid_values.get(train_col)

            # If we have no vocabulary for this column, skip rather than
            # falsely reject (defensive; should not happen with 9 cats).
            if valid_set is None:
                continue

            if value not in valid_set:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "unrecognized_categorical_value",
                        "field": snake_field,
                        "received": value,
                        "valid_values": sorted(valid_set),
                    },
                )