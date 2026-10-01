# ml_service/models/recommender.py
"""
Stage 2 Hybrid Recommender (TF-IDF + Multi-Attribute Functional Safety & Soft Boost)
==================================================================================
Matches PWD applicants to available job posts using:
1. Multi-Attribute Functional Safety Hard Filters (Disability compatibility, DOH/RA 7277 exclusions, Educational hierarchy qualification)
2. TF-IDF vectorization with Cosine Similarity over skills, job descriptions, and accommodations
3. Two-Tier Soft Boost for Stage 1 Random Forest Predicted Employment Type (+0.15 Primary, +0.07 Secondary)
"""

from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EDUCATION_RANK = {
    "none": 0,
    "sped": 1,
    "elementary level": 2,
    "elementary undergraduate": 2,
    "elementary graduate": 3,
    "als": 3,
    "als graduate": 3,
    "junior high school": 4,
    "high school level": 4,
    "high school undergraduate": 4,
    "high school graduate": 5,
    "senior high school": 5,
    "senior high school undergraduate": 5,
    "senior high school graduate": 6,
    "vocational": 6,
    "vocational undergraduate": 6,
    "vocational graduate": 7,
    "college level": 7,
    "college undergraduate": 7,
    "college graduate": 8,
    "post graduate": 9,
    "masteral": 9,
    "doctorate": 9,
}

EXCLUSION_RULES: Dict[str, List[str]] = {
    "Visual Disability": [
        "driving", "driver", "vehicle inspection", "visual proofing",
        "graphic design", "blueprint", "fine visual inspection",
        "visual inspection", "visual sorting", "visual acuity",
        "color grading", "color coded", "color coding", "color matching",
        "color sorting", "color vision", "delivery driver", "forklift operator",
        "operating heavy machinery", "visual merchandising"
    ],
    "Hearing Disability": [
        "telephone operator", "call center agent", "phone inquiries",
        "audio transcription", "tele sales agent", "tele sales", "voice dispatch",
        "inbound call reception", "inbound call", "outbound call", "phone operator",
        "voice customer service", "auditory alerts", "verbal phone"
    ],
    "Deaf or Hard of Hearing": [
        "telephone operator", "call center agent", "phone inquiries",
        "audio transcription", "tele sales agent", "tele sales", "voice dispatch",
        "inbound call reception", "inbound call", "outbound call", "phone operator",
        "voice customer service", "auditory alerts", "verbal phone"
    ],
    "Speech and Language Disability": [
        "voice customer service agent", "voice customer service", "phone operator",
        "spokesperson", "public announcer", "telemarketing representative",
        "telemarketing", "broadcast host", "verbal presentation",
        "telephone operator", "call center agent", "voice dispatch"
    ],
    "Speech and Language Impairment": [
        "voice customer service agent", "voice customer service", "phone operator",
        "spokesperson", "public announcer", "telemarketing representative",
        "telemarketing", "broadcast host", "verbal presentation",
        "telephone operator", "call center agent", "voice dispatch"
    ],
    "Speech Impairment": [
        "voice customer service agent", "voice customer service", "phone operator",
        "spokesperson", "public announcer", "telemarketing representative",
        "telemarketing", "broadcast host", "verbal presentation",
        "telephone operator", "call center agent", "voice dispatch"
    ],
    "Physical Disability (Orthopedic)": [
        "heavy lifting", "field manual labor", "ladder climbing", "ladder",
        "construction site laborer", "construction", "standing 8 hours",
        "heavy equipment operation", "warehouse porter", "field work",
        "scaffolding", "prolonged standing", "manual hauling"
    ],
    "Physical Disability": [
        "heavy lifting", "field manual labor", "ladder climbing", "ladder",
        "construction site laborer", "construction", "standing 8 hours",
        "heavy equipment operation", "warehouse porter", "field work",
        "scaffolding", "prolonged standing", "manual hauling"
    ],
    "Intellectual Disability": [
        "advanced financial analysis", "complex legal drafting",
        "high risk electrical engineering", "chemical lab analysis",
        "corporate compliance auditor", "high speed multitasking",
        "unsupervised heavy machinery"
    ],
    "Learning Disability": [
        "high speed complex proofreading", "real time stenography"
    ],
    "Mental Disability": [
        "hostile dispute arbitration", "extreme crisis intervention",
        "hazardous emergency response", "high conflict confrontation"
    ],
    "Mental Health Condition": [
        "hostile dispute arbitration", "extreme crisis intervention",
        "hazardous emergency response", "high conflict confrontation"
    ],
    "Psychosocial Disability": [
        "hostile dispute arbitration", "extreme crisis intervention",
        "hazardous emergency response", "high conflict confrontation"
    ],
    "Cancer (RA 11215)": [
        "heavy chemical pesticide handling", "radioactive exposure",
        "heavy physical quarrying", "unventilated toxic fume operations"
    ],
    "Rare Disease (RA 10747)": [
        "heavy hazardous mining", "extreme industrial heat exposure",
        "unventilated chemical handling", "heavy physical quarrying"
    ]
}

MOBILITY_EXCLUSIONS: Dict[str, List[str]] = {
    "Non-Ambulatory": [
        "field work", "climbing", "construction", "standing",
        "heavy lifting", "delivery driver", "ladder", "warehouse porter",
        "foot messenger", "rooftop maintenance"
    ],
    "Assisted": [
        "field work", "climbing", "construction", "standing 8 hours",
        "heavy lifting", "delivery driver", "ladder"
    ]
}

DEVICE_EXCLUSIONS: Dict[str, List[str]] = {
    "Wheelchair": [
        "climbing", "ladder", "stairs only", "standing 8 hours",
        "heavy lifting", "field messenger", "scaffolding"
    ],
    "Crutches": [
        "heavy lifting", "ladder", "climbing", "standing 8 hours", "scaffolding"
    ],
    "Walker": [
        "heavy lifting", "ladder", "climbing", "standing 8 hours"
    ],
    "Cane": [
        "heavy lifting", "ladder climbing", "scaffolding"
    ],
    "White Cane": [
        "driving", "operating heavy machinery", "fine visual inspection"
    ]
}

DEVICE_RELAXATIONS: Dict[Tuple[str, str], List[str]] = {
    ("Visual Disability", "Eyeglasses"): ["visual inspection", "visual proofing", "graphic design"],
    ("Visual Disability", "Contact Lenses"): ["visual inspection", "visual proofing", "graphic design"],
    ("Deaf or Hard of Hearing", "Hearing Aid"): ["phone inquiries", "audio transcription"],
    ("Hearing Disability", "Hearing Aid"): ["phone inquiries", "audio transcription"],
    ("Physical Disability", "Prosthesis"): ["standing"]
}

ACCOMMODATION_KEYWORDS: Dict[str, List[str]] = {
    "Visual Disability": ["screen reader compatible", "well-lit office", "braille accessible"],
    "Deaf or Hard of Hearing": [
        "sign language interpreter", "chat-based", "written communication", "visual cues"
    ],
    "Hearing Disability": [
        "sign language interpreter", "chat-based", "written communication", "visual cues"
    ],
    "Physical Disability": [
        "wheelchair accessible", "remote work option",
        "ergonomic workstation", "flexible hours", "ramp access", "elevator access"
    ],
    "Physical Disability (Orthopedic)": [
        "wheelchair accessible", "remote work option",
        "ergonomic workstation", "flexible hours", "ramp access"
    ],
    "Speech and Language Disability": ["written communication", "chat-based", "non-voice support"],
    "Speech and Language Impairment": ["written communication", "chat-based", "non-voice support"],
    "Speech Impairment": ["written communication", "chat-based", "non-voice support"],
    "Intellectual Disability": ["structured tasks", "peer buddy support", "visual guides"],
    "Learning Disability": ["flexible pacing", "assistive software", "clear instructions"],
    "Mental Health Condition": ["stress-managed environment", "supportive leadership", "regular breaks"],
    "Mental Disability": ["stress-managed environment", "supportive leadership", "regular breaks"],
    "Psychosocial Disability": ["inclusive culture", "mental wellness support", "clear milestones"],
    "Cancer (RA 11215)": ["medical leave flexibility", "climate-controlled workspace", "ergonomic seating"],
    "Rare Disease (RA 10747)": ["flexible healthcare hours", "clean indoor environment", "low physical strain"]
}

MOBILITY_KEYWORDS: Dict[str, List[str]] = {
    "Non-Ambulatory": ["wheelchair accessible", "ramp access", "ground floor", "elevator"],
    "Assisted": ["wheelchair accessible", "ramp access", "ground floor"],
    "Ambulatory": ["general office environment"]
}

DEVICE_KEYWORDS: Dict[str, List[str]] = {
    "Wheelchair": ["wheelchair accessible", "ramp access", "accessible restroom", "wide hallways"],
    "Hearing Aid": ["quiet office environment", "visual alerts"],
    "Crutches": ["elevator access", "ground floor"],
    "Walker": ["elevator access", "ground floor"],
    "Cane": ["accessible ramps"],
    "White Cane": ["tactile paths", "voice-enabled workstations"]
}

class JobRecommender:
    def __init__(self, artifacts_dir: str = None):
        self.loaded = True

    def _get_education_rank(self, edu: Optional[str]) -> int:
        if not edu:
            return 0
        norm = edu.strip().lower()
        for key, rank in sorted(EDUCATION_RANK.items(), key=lambda x: -len(x[0])):
            if key in norm:
                return rank
        return 0

    def _is_education_qualified(self, applicant_edu: Optional[str], required_edu: Optional[str]) -> bool:
        if not required_edu or required_edu.strip().lower() in ["none", "any", "not required", "n/a", ""]:
            return True
        req_rank = self._get_education_rank(required_edu)
        app_rank = self._get_education_rank(applicant_edu)
        return app_rank >= req_rank

    def _get_exclusion_keywords(self, disability_type: str, mobility_status: str, assistive_device: str) -> List[str]:
        keywords = set()
        
        # 1. Disability type exclusions
        for dt_key, rules in EXCLUSION_RULES.items():
            if dt_key.lower() == disability_type.lower() or dt_key.lower() in disability_type.lower():
                keywords.update(rules)
                
        # 2. Mobility status exclusions
        if mobility_status:
            for mob_key, rules in MOBILITY_EXCLUSIONS.items():
                if mob_key.lower() == mobility_status.lower() or mob_key.lower() in mobility_status.lower():
                    keywords.update(rules)
                    
        # 3. Device exclusions
        if assistive_device and assistive_device.lower() != "none":
            for dev_key, rules in DEVICE_EXCLUSIONS.items():
                if dev_key.lower() == assistive_device.lower() or dev_key.lower() in assistive_device.lower():
                    keywords.update(rules)
                    
            # 4. Device relaxations
            for (dt_rel, dev_rel), relaxed_rules in DEVICE_RELAXATIONS.items():
                if (dt_rel.lower() in disability_type.lower()) and (dev_rel.lower() in assistive_device.lower()):
                    for r in relaxed_rules:
                        keywords.discard(r)
                        
        return list(keywords)

    @staticmethod
    def _normalize_text(text: str) -> str:
        import re
        return re.sub(r'[^a-z0-9\s]', ' ', text.lower())

    def _is_excluded_by_heuristic(self, job: Dict[str, Any], exclusion_keywords: List[str]) -> bool:
        if not exclusion_keywords:
            return False
        raw_text = (
            job.get("job_title", "") + " " +
            job.get("job_description", "") + " " +
            job.get("required_skills", "") + " " +
            job.get("disability_friendly_notes", "")
        )
        norm_text = self._normalize_text(raw_text)
        for kw in exclusion_keywords:
            norm_kw = self._normalize_text(kw)
            if norm_kw in norm_text:
                return True
        return False

    def _apply_disability_compatibility(self, jobs: List[Dict[str, Any]], profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        applicant_disability = profile.get("disability_type", "")
        mobility_status = profile.get("mobility_status", "")
        assistive_device = profile.get("current_assistive_device", "None") or "None"
        applicant_edu = profile.get("educational_attainment", "")

        # 1. Employer-declared compatibility filter (Supports both compatible_disabilities and target_disability_types)
        jobs_a = []
        for job in jobs:
            compat_list = job.get("compatible_disabilities") or job.get("target_disability_types") or []
            if not compat_list or any(applicant_disability.lower() in c.lower() or c.lower() in applicant_disability.lower() for c in compat_list):
                jobs_a.append(job)

        # 2. Educational Hierarchy Safety Check
        jobs_b = []
        for job in jobs_a:
            req_edu = job.get("required_education", "")
            if self._is_education_qualified(applicant_edu, req_edu):
                jobs_b.append(job)

        # 3. Functional Safety Heuristic Exclusions
        exclusion_keywords = self._get_exclusion_keywords(
            applicant_disability, mobility_status, assistive_device
        )
        if not exclusion_keywords:
            return jobs_b

        return [
            job for job in jobs_b
            if not self._is_excluded_by_heuristic(job, exclusion_keywords)
        ]

    def _compose_applicant_document(self, profile: Dict[str, Any]) -> str:
        parts = [
            profile.get("skills", ""),
            profile.get("preferred_employment_type", ""),
            profile.get("preferred_job_category", ""),
            profile.get("educational_attainment", ""),
            profile.get("occupation_group", ""),
            self._derive_disability_keywords(
                profile.get("disability_type", ""),
                profile.get("mobility_status", ""),
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
    def _derive_disability_keywords(disability_type: str, mobility_status: str, assistive_device: str) -> str:
        keywords = []
        for k, vals in ACCOMMODATION_KEYWORDS.items():
            if k.lower() in disability_type.lower():
                keywords.extend(vals)
        for k, vals in MOBILITY_KEYWORDS.items():
            if k.lower() in mobility_status.lower():
                keywords.extend(vals)
        for k, vals in DEVICE_KEYWORDS.items():
            if k.lower() in assistive_device.lower():
                keywords.extend(vals)
        return " ".join(keywords)

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
        elif tier == "primary":
            return f"{base}; matches predicted employment type"
        return base

    def recommend(self, profile: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        available_jobs = profile.get("available_jobs", [])
        total_input = len(available_jobs)

        predicted_type = profile.get("predicted_employment_type", "")
        secondary_type = profile.get("secondary_employment_type") or ""
        if secondary_type == predicted_type:
            secondary_type = ""

        after_type = sum(
            1 for j in available_jobs
            if j.get("employment_type") == predicted_type
        )
        after_secondary = sum(
            1 for j in available_jobs
            if secondary_type and j.get("employment_type") == secondary_type
        )

        qualified_jobs = self._apply_disability_compatibility(available_jobs, profile)
        after_disability = len(qualified_jobs)

        if not qualified_jobs:
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
            profile, qualified_jobs, top_k, predicted_type, secondary_type
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
            max_df=1.0,
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
            raw_sim = float(sim)
            # Apply MIN_SCORE to raw text similarity so boosts do not fabricate matches from zero textual overlap
            if raw_sim < MIN_SCORE:
                continue

            score = raw_sim
            job_type = job.get("employment_type")
            tier = None
            if predicted_type and job_type == predicted_type:
                score = min(1.0, score + PRIMARY_BOOST)
                tier = "primary"
            elif secondary_type and job_type == secondary_type:
                score = min(1.0, score + SECONDARY_BOOST)
                tier = "secondary"
            boosted.append((job, score, tier))
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
