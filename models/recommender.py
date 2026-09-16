import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("ml_service.recommender")

PRIMARY_BOOST = 0.05
SECONDARY_BOOST = 0.025
MIN_SCORE = 0.05
ACCOMMODATION_BOOST = 0.03

EXCLUSION_RULES: Dict[str, List[str]] = {
    "Physical Disability": [
        "heavy lifting", "lifting 25kg", "lifting 50lbs", "construction labor",
        "manual labor", "climbing stairs", "climb ladders", "standing 8 hours",
        "continuous standing", "physical patrol", "warehouse carrying", "motorcycle delivery",
        "roving inspection", "heavy material handling", "warehouse loader"
    ],
    "Visual Disability": [
        "driving", "drive vehicle", "motorcycle", "delivery rider",
        "precision visual inspection", "microscopic assembly", "graphic inspection without magnifier",
        "heavy machinery operation", "crane operator", "forklift operator"
    ],
    "Deaf or Hard of Hearing": [
        "telephone operator", "call center voice", "voice inbound", "verbal telephone",
        "telephone customer service", "audio transcribing without text", "radio dispatch",
        "oral phone calls", "inbound phone agent", "voice customer calls"
    ],
    "Speech and Language Disability": [
        "telemarketing", "verbal sales pitches", "radio announcer", "public speaking broadcast",
        "oral customer telephone support", "voice call center agent"
    ]
}

MOBILITY_EXCLUSIONS: Dict[str, List[str]] = {
    "Non-Ambulatory": [
        "delivery", "messenger", "field inspection", "climb stairs", "stairs",
        "patrol", "manual labor", "heavy lifting", "standing 8 hours", "roving",
        "climbing ladders", "warehouse loader", "motorcycle"
    ]
}

DEVICE_EXCLUSIONS: Dict[str, List[str]] = {
    "Wheelchair": [
        "climb stairs", "stairs", "climbing ladders", "motorcycle delivery",
        "delivery rider", "heavy lifting", "manual labor", "roving patrol",
        "field inspection", "standing 8 hours", "warehouse carrying"
    ],
    "Crutches": [
        "carrying heavy loads", "climbing ladders", "motorcycle delivery",
        "heavy physical labor", "running", "lifting 25kg", "stairs"
    ],
    "Walker": [
        "climbing stairs", "stairs", "motorcycle delivery", "heavy lifting",
        "field roving", "manual labor"
    ],
    "Prosthesis": [
        "heavy machinery grip", "heavy manual handling 25kg"
    ]
}

DEVICE_RELAXATIONS: Dict[Tuple[str, str], List[str]] = {
    ("Visual Disability", "Eyeglasses"): [
        "precision visual inspection", "computer screen", "reading fine print"
    ],
    ("Physical Disability", "Wheelchair"): [
        "office", "desk", "computer encoding", "clerical", "administrative"
    ],
    ("Deaf or Hard of Hearing", "Hearing Aid"): [
        "face-to-face communication", "in-person coordination"
    ]
}

ACCOMMODATION_KEYWORDS = [
    "wheelchair accessible", "accessible workplace", "pwd friendly",
    "ramp available", "elevator access", "ground floor", "work from home",
    "remote work", "assistive technology provided", "inclusive employer"
]

ALL_KNOWN_DISABILITIES = [
    "Cancer (RA 11215)", "Deaf or Hard of Hearing", "Intellectual Disability",
    "Learning Disability", "Mental Disability", "Physical Disability",
    "Psychosocial Disability", "Rare Disease (RA 10747)",
    "Speech and Language Disability", "Visual Disability"
]


class JobRecommender:
    def __init__(self, artifacts_dir: Optional[str] = None):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z0-9_-]+\b",
            min_df=1
        )

    def _normalize(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", str(text).lower().strip())

    def _is_compatible(self, profile: Dict[str, Any], job: Dict[str, Any]) -> Tuple[bool, str]:
        disability = profile.get("disability_type", "")
        mobility = profile.get("mobility_status", "")
        device = profile.get("current_assistive_device", "None")

        job_text = self._normalize(
            f"{job.get('job_title', '')} {job.get('description', '')} {job.get('required_skills', '')} {job.get('requirements', '')}"
        )

        if disability and disability not in ALL_KNOWN_DISABILITIES:
            logger.warning(f"Unmapped disability type encountered: '{disability}'")

        # 1. Mobility Status Exclusions
        if mobility in MOBILITY_EXCLUSIONS:
            for kw in MOBILITY_EXCLUSIONS[mobility]:
                if kw in job_text:
                    return False, f"Incompatible with mobility status '{mobility}' (keyword: '{kw}')"

        # 2. Assistive Device Specific Exclusions
        if device in DEVICE_EXCLUSIONS:
            for kw in DEVICE_EXCLUSIONS[device]:
                if kw in job_text:
                    return False, f"Incompatible with assistive device '{device}' (keyword: '{kw}')"

        # 3. Category-Specific Safety Exclusions
        if disability in EXCLUSION_RULES:
            relaxed_keywords = set()
            if (disability, device) in DEVICE_RELAXATIONS:
                relaxed_keywords = set(DEVICE_RELAXATIONS[(disability, device)])

            for kw in EXCLUSION_RULES[disability]:
                if kw in job_text and kw not in relaxed_keywords:
                    return False, f"Incompatible with disability '{disability}' (hazard keyword: '{kw}')"

        return True, "Compatible"

    def _has_accommodation(self, job: Dict[str, Any]) -> bool:
        job_text = self._normalize(f"{job.get('description', '')} {job.get('requirements', '')}")
        return any(akw in job_text for akw in ACCOMMODATION_KEYWORDS)

    def recommend(
        self,
        profile: Dict[str, Any],
        jobs: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Dict[str, Any]:
        total_input = len(jobs)
        if total_input == 0:
            return {
                "recommendations": [],
                "metadata": {
                    "total_jobs_input": 0,
                    "after_employment_type_filter": 0,
                    "after_secondary_type_match": 0,
                    "after_disability_compatibility_filter": 0,
                    "returned_count": 0
                }
            }

        # Step 1: Multi-Attribute Safety Filtering (Layer B)
        compatible_jobs = []
        for job in jobs:
            ok, reason = self._is_compatible(profile, job)
            if ok:
                compatible_jobs.append(job)
            else:
                logger.info(f"Job {job.get('id', job.get('job_post_id'))} excluded: {reason}")

        survived_compat = len(compatible_jobs)
        if survived_compat == 0:
            return {
                "recommendations": [],
                "metadata": {
                    "total_jobs_input": total_input,
                    "after_employment_type_filter": 0,
                    "after_secondary_type_match": 0,
                    "after_disability_compatibility_filter": 0,
                    "returned_count": 0
                }
            }

        # Step 2: TF-IDF Text Vectorization (Layer A)
        profile_text = self._normalize(
            f"{profile.get('skills', '')} {profile.get('educational_attainment', '')} "
            f"{profile.get('occupation_group', '')} {profile.get('experience', '')}"
        )

        job_texts = [
            self._normalize(
                f"{j.get('job_title', '')} {j.get('description', '')} "
                f"{j.get('required_skills', '')} {j.get('required_education', '')}"
            )
            for j in compatible_jobs
        ]

        corpus = [profile_text] + job_texts
        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            profile_vec = tfidf_matrix[0:1]
            job_vecs = tfidf_matrix[1:]
            sims = cosine_similarity(profile_vec, job_vecs).flatten()
        except Exception as e:
            logger.error(f"TF-IDF Vectorization failed: {e}")
            sims = np.zeros(len(compatible_jobs))

        primary_type = (profile.get("predicted_employment_type") or "").strip().lower()
        secondary_type = (profile.get("secondary_employment_type") or "").strip().lower()

        primary_matches = 0
        secondary_matches = 0
        ranked_items = []

        # Step 3: Two-Tier Soft Boost + Accommodation Boost (Layer C)
        for i, job in enumerate(compatible_jobs):
            base_score = float(sims[i])
            job_emp_type = (job.get("employment_type") or "").strip().lower()
            boost = 0.0
            reasons = []

            if primary_type and job_emp_type == primary_type:
                boost += PRIMARY_BOOST
                primary_matches += 1
                reasons.append(f"matches predicted type ({profile.get('predicted_employment_type')})")
            elif secondary_type and job_emp_type == secondary_type:
                boost += SECONDARY_BOOST
                secondary_matches += 1
                reasons.append(f"matches secondary predicted type ({profile.get('secondary_employment_type')})")

            if self._has_accommodation(job):
                boost += ACCOMMODATION_BOOST
                reasons.append("verified accessible workplace accommodation")

            final_score = round(min(1.0, max(0.0, base_score + boost)), 4)

            if final_score < MIN_SCORE:
                continue

            if base_score >= 0.40:
                match_desc = "Strong text match"
            elif base_score >= 0.15:
                match_desc = "Moderate text match"
            else:
                match_desc = "Weak text match"

            if reasons:
                full_reason = f"{match_desc}; {'; '.join(reasons)}"
            else:
                full_reason = match_desc

            job_id = job.get("id", job.get("job_post_id"))
            ranked_items.append({
                "job_post_id": int(job_id),
                "similarity_score": final_score,
                "recommendation_reason": full_reason
            })

        # Rank by final score descending
        ranked_items.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_items = ranked_items[:top_k]

        for rank, item in enumerate(top_items, start=1):
            item["rank_position"] = rank

        return {
            "recommendations": top_items,
            "metadata": {
                "total_jobs_input": total_input,
                "after_employment_type_filter": primary_matches,
                "after_secondary_type_match": secondary_matches,
                "after_disability_compatibility_filter": survived_compat,
                "returned_count": len(top_items)
            }
        }
