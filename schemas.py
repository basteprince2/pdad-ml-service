from pydantic import BaseModel, Field
from typing import Optional


# ============================================================
# EMPLOYMENT TYPE PREDICTION (Random Forest)
# ============================================================


class PredictionRequest(BaseModel):
    """Input for RF employment type prediction (11 features)."""
    pwd_profile_id: int

    age: int = Field(..., ge=15, le=100)
    sex: str
    civil_status: str
    disability_type: str
    disability_visibility: str
    cause_of_disability: str
    educational_attainment: str
    skills: str
    mobility_status: str
    current_assistive_device: str
    occupation_group: str


class PredictionResponse(BaseModel):
    predicted_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    all_probabilities: dict[str, float]


# ============================================================
# JOB RECOMMENDATION (TF-IDF + Cosine Similarity + Three-Layer Filter)
# ============================================================


class JobDict(BaseModel):
    """
    Single job posting in the recommendation request payload.
    Laravel populates these fields by joining job_posts with
    employer + skills + compatibility tables before sending.
    """
    id: int
    job_title: str
    job_description: str = ""
    required_skills: str = ""
    required_education: str = ""
    employment_type: str
    disability_friendly_notes: str = ""
    compatible_disabilities: list[str] = Field(default_factory=list)


class RecommendationRequest(BaseModel):
    """
    Input for TF-IDF job recommender with three-layer filtering.

    Pipeline:
      1. Filter by predicted employment type (hard filter)
      2. Apply disability compatibility filter (three sub-layers)
      3. TF-IDF vectorize applicant + remaining jobs at runtime
      4. Cosine similarity ranking, return top_k
    """
    pwd_profile_id: int

    # Applicant profile fields for TF-IDF vectorization
    skills: str
    disability_type: str
    educational_attainment: str
    occupation_group: Optional[str] = None
    mobility_status: Optional[str] = None
    current_assistive_device: Optional[str] = "None"
    preferred_employment_type: Optional[str] = None
    preferred_job_category: Optional[str] = None

    # Filter driver — from RF prediction
    predicted_employment_type: str
    secondary_employment_type: Optional[str] = None


    # Full jobs corpus at request time
    available_jobs: list[JobDict]

    top_k: int = Field(5, ge=1, le=20)


class JobRecommendation(BaseModel):
    """Matches job_recommendations table columns."""
    job_post_id: int
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    rank_position: int = Field(..., ge=1)
    recommendation_reason: Optional[str] = None


class RecommendationMetadata(BaseModel):
    """Transparency metadata for filtering pipeline."""
    total_jobs_input: int
    after_employment_type_filter: int
    after_secondary_type_match: int = 0
    after_disability_compatibility_filter: int
    returned_count: int


class RecommendationResponse(BaseModel):
    recommendations: list[JobRecommendation]
    metadata: RecommendationMetadata


# ============================================================
# HEALTH / SERVICE META
# ============================================================


class HealthResponse(BaseModel):
    status: str
    predictor_loaded: bool
    recommender_loaded: bool
    model_version: Optional[str] = None