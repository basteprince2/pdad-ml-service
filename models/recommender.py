"""
TF-IDF job recommendation engine for Stage 2 employment matching.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("pdad_ml_service")

EDUCATION_RANK: Dict[str, int] = {
    "none": 0,
    "sped": 1,
    "elementary level": 2,
    "elementary graduate": 3,
    "als": 3,
    "junior high school": 4,
    "high school level": 4,
    "high school graduate": 5,
    "senior high school": 5,
    "senior high school graduate": 6,
    "vocational": 6,
    "vocational graduate": 7,
    "college level": 7,
    "college graduate": 8,
    "post graduate": 9,
}

EXCLUSION_RULES: Dict[str, List[str]] = {
    "Physical Disability": [
        "heavy lifting", "carrying heavy loads", "climbing ladders",
        "strenuous physical labor", "prolonged standing",
        "manual construction labor", "warehouse lifting", "scaffolding"
    ],
    "Visual Disability": [
        "driving", "delivery driver", "forklift operator",
        "operating heavy machinery", "vehicle operation",
        "fine graphic color grading", "visual inspection of micro-components",
        "crane operator", "night driving"
    ],
    "Deaf or Hard of Hearing": [
        "phone operator", "telephone support", "inbound call reception",
        "transcribing audio without captions", "telemarketing voice calls",
        "voice customer service agent"
    ],
    "Speech and Language Disability": [
        "phone operator", "telephone support", "inbound call reception",
        "telemarketing voice calls", "radio announcer", "voice customer service agent"
    ],
    "Intellectual Disability": [
        "driving", "forklift operator", "operating heavy machinery",
        "complex financial portfolio auditing", "high-voltage electrical wiring",
        "hazardous chemical handling", "heavy machinery operator"
    ],
    "Mental Disability": [
        "extreme high-risk armed security", "hazardous chemical handling",
        "emergency crisis response unit", "explosives handling"
    ],
    "Psychosocial Disability": [
        "extreme high-risk armed security", "hazardous chemical handling",
        "emergency crisis response unit", "explosives handling"
    ],
    "Cancer (RA 11215)": [
        "heavy lifting", "strenuous physical labor", "toxic chemical exposure",
        "radiation exposure", "manual construction labor"
    ],
    "Rare Disease (RA 10747)": [
        "heavy lifting", "strenuous physical labor", "toxic chemical exposure",
        "hazardous environment"
    ],
    "Learning Disability": [
        "complex legal drafting", "actuarial formula derivation",
        "high-speed complex financial auditing"
    ],
}

MOBILITY_EXCLUSIONS: Dict[str, List[str]] = {
    "Non-Ambulatory": [
        "driving", "climbing", "climbing ladders", "heavy lifting",
        "standing for long periods", "prolonged standing", "field technician",
        "delivery driver", "warehouse labor", "manual labor", "foot patrol",
        "stairs climbing", "construction worker"
    ],
    "Assisted": [
        "climbing ladders", "heavy lifting", "running", "rough terrain",
        "scaffolding", "manual construction labor"
    ],
}

DEVICE_EXCLUSIONS: Dict[str, List[str]] = {
    "Wheelchair": [
        "climbing", "climbing ladders", "heavy lifting", "prolonged standing",
        "delivery driver", "field technician", "warehouse labor", "foot patrol",
        "standing required", "scaffolding", "stairs climbing", "driving"
    ],
    "Crutches": [
        "climbing ladders", "heavy lifting", "prolonged standing", "scaffolding",
        "carrying heavy objects"
    ],
    "Walker": [
        "climbing ladders", "heavy lifting", "prolonged standing", "rough terrain",
        "delivery driver"
    ],
    "Prosthesis": [
        "extreme impact physical labor", "climbing high scaffolding"
    ],
}

DEVICE_RELAXATIONS: Dict[Tuple[str, str], List[str]] = {
    ("Deaf or Hard of Hearing", "Hearing Aid"): [
        "in-person oral communication", "face-to-face customer service", "listening to instructions"
    ],
    ("Visual Disability", "Eyeglasses"): [
        "reading printed documents", "screen viewing", "computer monitor work", "data verification"
    ],
}

DISABILITY_KEYWORDS: Dict[str, List[str]] = {
    "Deaf or Hard of Hearing": [
        "sign language interpreter", "chat-based", "written communication",
    ],
    "Physical Disability": [
        "wheelchair accessible", "remote work option",
        "ergonomic workstation", "flexible hours",
    ],
    "Speech and Language Disability": ["written communication", "chat-based"],
    "Visual Disability": ["screen reader accessible", "assistive technology", "audio support"],
    "Intellectual Disability": ["supportive supervisor", "structured routine", "inclusive workplace"],
    "Learning Disability": ["visual aids", "flexible pacing", "text-to-speech support"],
    "Psychosocial Disability": ["mental wellness support", "flexible schedule", "quiet work area"],
    "Mental Disability": ["supportive environment", "flexible work hours"],
    "Cancer (RA 11215)": ["flexible medical leave", "remote work option", "ergonomic seating"],
    "Rare Disease (RA 10747)": ["flexible schedule", "accessible facility"],
}

MOBILITY_KEYWORDS: Dict[str, List[str]] = {
    "Non-Ambulatory": ["wheelchair accessible", "ramp access", "elevator access", "ground floor"],
    "Assisted": ["wheelchair accessible", "ramp access", "accessible restroom"],
}

DEVICE_KEYWORDS: Dict[str, List[str]] = {
    "Wheelchair": ["wheelchair accessible", "ramp access", "accessible restroom", "ground floor workstation"],
    "Hearing Aid": ["quiet office environment", "written memos"],
    "Eyeglasses": ["standard lighting", "digital display"],
}


class JobRecommender:
    def __init__(self, artifacts_dir: Optional[str] = None):
        self.loaded = True

    def recommend(
        self,
        profile: Dict[str, Any],
        jobs: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        if profile is None and "profile" in kwargs:
            profile = kwargs["profile"]
        if jobs is None and "jobs" in kwargs:
            jobs = kwargs["jobs"]
        if isinstance(jobs, int):
            top_k = jobs
            jobs = None
        if jobs is None and isinstance(profile, dict):
            jobs = profile.get("available_jobs", [])
            top_k = profile.get("top_n", top_k)
        if jobs is None:
            jobs = []

        total_input = len(jobs)

        predicted_type = profile.get("predicted_employment_type", "")
        secondary_type = profile.get("secondary_employment_type") or ""
        if secondary_type == predicted_type:
            secondary_type = ""

        after_type = sum(
            1 for j in jobs
            if j.get("employment_type") == predicted_type
        )
        after_secondary = sum(
            1 for j in jobs
            if secondary_type and j.get("employment_type") == secondary_type
        )

        filtered_jobs = self._apply_disability_compatibility(jobs, profile)
        after_disability = len(filtered_jobs)

        if not filtered_jobs:
            return {
                "recommendations": [],
                "metadata": {
                    "total_jobs_input": total_input,
                    "after_employment_type_filter": after_type,
                    "after_secondary_type_match": after_secondary,
                    "after_disability_compatibility_filter": after_disability,
                    "returned_count": 0,
                },
            }

        ranked = self._vectorize_and_rank(
            profile, filtered_jobs, top_k, predicted_type, secondary_type
        )

        return {
            "recommendations": ranked,
            "metadata": {
                "total_jobs_input": total_input,
                "after_employment_type_filter": after_type,
                "after_secondary_type_match": after_secondary,
                "after_disability_compatibility_filter": after_disability,
                "returned_count": len(ranked),
            },
        }

    def _is_education_qualified(self, applicant_edu: str, required_edu: str) -> bool:
        if not required_edu or required_edu.strip().lower() in ["none", "any", "not required", "n/a", ""]:
            return True
        app_rank = EDUCATION_RANK.get(applicant_edu.strip().lower() if applicant_edu else "", 0)
        req_rank = EDUCATION_RANK.get(required_edu.strip().lower() if required_edu else "", 0)
        return app_rank >= req_rank

    def _apply_disability_compatibility(
        self,
        jobs: List[Dict[str, Any]],
        profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        applicant_disability = profile.get("disability_type", "")
        mobility_status = profile.get("mobility_status", "Ambulatory") or "Ambulatory"
        assistive_device = profile.get("current_assistive_device", "None") or "None"
        applicant_edu = profile.get("educational_attainment", "")

        compatible_jobs = []
        for job in jobs:
            # 1. Educational Attainment Qualification Filter
            req_edu = job.get("required_education", "")
            if not self._is_education_qualified(applicant_edu, req_edu):
                continue

            # 2. Explicit Job Disability Compatibility Check
            compat_list = job.get("compatible_disabilities", []) or []
            if compat_list:
                if (
                    applicant_disability
                    and applicant_disability not in compat_list
                    and "All Disabilities" not in compat_list
                ):
                    continue

            # 3. Multi-Attribute Functional Exclusion Filter
            if self._is_job_excluded(job, applicant_disability, mobility_status, assistive_device):
                continue

            compatible_jobs.append(job)

        return compatible_jobs

    def _is_job_excluded(
        self,
        job: Dict[str, Any],
        disability_type: str,
        mobility_status: str,
        assistive_device: str
    ) -> bool:
        job_text = (
            str(job.get("job_title", "")) + " " +
            str(job.get("job_description", "")) + " " +
            str(job.get("required_skills", ""))
        ).lower()

        exclusions = list(EXCLUSION_RULES.get(disability_type, []))
        relaxations = DEVICE_RELAXATIONS.get((disability_type, assistive_device), [])
        active_exclusions = [kw for kw in exclusions if kw not in relaxations]

        active_exclusions.extend(MOBILITY_EXCLUSIONS.get(mobility_status, []))
        active_exclusions.extend(DEVICE_EXCLUSIONS.get(assistive_device, []))

        for kw in active_exclusions:
            if re.search(r'\b' + re.escape(kw.lower()) + r'\b', job_text):
                return True
        return False

    def _vectorize_and_rank(
        self,
        profile: Dict[str, Any],
        jobs: List[Dict[str, Any]],
        top_k: int,
        predicted_type: str = "",
        secondary_type: str = ""
    ) -> List[Dict[str, Any]]:
        applicant_doc = self._compose_applicant_document(profile)
        job_docs = [self._compose_job_document(j) for j in jobs]

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        corpus = [applicant_doc] + job_docs
        tfidf_matrix = vectorizer.fit_transform(corpus)

        applicant_vec = tfidf_matrix[0:1]
        job_vecs = tfidf_matrix[1:]
        sims = cosine_similarity(applicant_vec, job_vecs).flatten()

        MIN_SCORE = 0.05
        PRIMARY_BOOST = 0.15
        SECONDARY_BOOST = 0.07

        boosted = []
        for job, sim in zip(jobs, sims):
            score = float(sim)
            job_type = job.get("employment_type")
            tier = None
            if predicted_type and job_type == predicted_type:
                score = min(1.0, score + PRIMARY_BOOST)
                tier = "primary"
            elif secondary_type and job_type == secondary_type:
                score = min(1.0, score + SECONDARY_BOOST)
                tier = "secondary"
            boosted.append((job, score, tier))

        boosted = [t for t in boosted if t[1] >= MIN_SCORE]
        ranked_triples = sorted(boosted, key=lambda x: x[1], reverse=True)

        return [
            {
                "job_post_id": int(job["id"]),
                "similarity_score": round(float(score), 4),
                "rank_position": i + 1,
                "recommendation_reason": self._reason_for_score(float(score), tier),
            }
            for i, (job, score, tier) in enumerate(ranked_triples[:top_k])
        ]

    @staticmethod
    def _reason_for_score(score: float, tier: Optional[str] = None) -> str:
        if score >= 0.80:
            base = "Very strong text match"
        elif score >= 0.60:
            base = "Strong text match"
        elif score >= 0.40:
            base = "Moderate text match"
        else:
            base = "Weak text match"

        if tier == "secondary":
            return f"{base}; matches second most likely employment type"
        return base

    def _compose_applicant_document(self, profile: Dict[str, Any]) -> str:
        parts = [
            profile.get("skills", ""),
            profile.get("preferred_employment_type", ""),
            profile.get("preferred_job_category", ""),
            profile.get("educational_attainment", ""),
            profile.get("occupation_group", ""),
            self._derive_disability_keywords(
                profile.get("disability_type", ""),
                profile.get("mobility_status", "Ambulatory") or "Ambulatory",
                profile.get("current_assistive_device", "None") or "None",
            ),
        ]
        return " ".join(p for p in parts if p)

    @staticmethod
    def _compose_job_document(job: Dict[str, Any]) -> str:
        parts = [
            job.get("job_title", ""),
            job.get("job_description", ""),
            job.get("required_skills", ""),
            job.get("employment_type", ""),
            job.get("required_education", ""),
            job.get("disability_friendly_notes", ""),
        ]
        return " ".join(p for p in parts if p)

    @staticmethod
    def _derive_disability_keywords(
        disability_type: str,
        mobility_status: str,
        assistive_device: str
    ) -> str:
        keywords = []
        if disability_type in DISABILITY_KEYWORDS:
            keywords.extend(DISABILITY_KEYWORDS[disability_type])
        if mobility_status in MOBILITY_KEYWORDS:
            keywords.extend(MOBILITY_KEYWORDS[mobility_status])
        if assistive_device in DEVICE_KEYWORDS:
            keywords.extend(DEVICE_KEYWORDS[assistive_device])
        return " ".join(keywords)
