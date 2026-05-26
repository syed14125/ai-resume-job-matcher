from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.skill_extractor import compare_skills, count_skills_by_category, skill_category_lookup


EXTRA_STOPWORDS = {
    "job",
    "description",
    "role",
    "candidate",
    "work",
    "working",
    "team",
    "experience",
    "years",
    "ability",
    "skills",
    "skill",
    "required",
    "requirements",
    "responsibilities",
    "preferred",
    "qualification",
    "qualifications",
    "using",
    "use",
    "including",
    "must",
    "plus",
    "etc",
}

STOPWORDS = set(ENGLISH_STOP_WORDS).union(EXTRA_STOPWORDS)


def safe_round(value: float, digits: int = 2) -> float:
    try:
        if np.isnan(value) or np.isinf(value):
            return 0.0

        return round(float(value), digits)
    except Exception:
        return 0.0


def calculate_text_similarity(resume_text: str, job_description: str) -> float:
    """Calculate TF-IDF cosine similarity between resume and job description."""
    resume_text = resume_text or ""
    job_description = job_description or ""

    if len(resume_text.strip()) < 20 or len(job_description.strip()) < 20:
        return 0.0

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        lowercase=True,
    )

    try:
        matrix = vectorizer.fit_transform([resume_text, job_description])
        score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return safe_round(score * 100, 2)
    except Exception:
        return 0.0


def tokenize_keywords(text: str) -> list[str]:
    text = (text or "").lower()
    tokens = re.findall(r"\b[a-z][a-z0-9\+\#\.\-/]{1,}\b", text)

    cleaned_tokens = []

    for token in tokens:
        token = token.strip(".-/")

        if token and token not in STOPWORDS and len(token) > 2:
            cleaned_tokens.append(token)

    return cleaned_tokens


def extract_top_keywords(text: str, top_n: int = 30) -> list[str]:
    """Extract frequent important keywords from job description."""
    tokens = tokenize_keywords(text)
    counts = Counter(tokens)

    ranked = sorted(
        counts.items(),
        key=lambda item: (item[1], len(item[0])),
        reverse=True,
    )

    return [word for word, _count in ranked[:top_n]]


def calculate_keyword_score(
    resume_text: str,
    job_description: str,
    top_n: int = 30,
) -> dict[str, object]:
    job_keywords = extract_top_keywords(job_description, top_n=top_n)
    resume_lower = (resume_text or "").lower()

    matched_keywords = []
    missing_keywords = []

    for keyword in job_keywords:
        if keyword.lower() in resume_lower:
            matched_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)

    if job_keywords:
        keyword_score = round((len(matched_keywords) / len(job_keywords)) * 100, 2)
    else:
        keyword_score = 0.0

    return {
        "job_keywords": job_keywords,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "keyword_score": keyword_score,
    }


def calculate_ats_score(
    similarity_score: float,
    skill_score: float,
    keyword_score: float,
) -> float:
    """
    Transparent portfolio ATS-style score.

    This is not an official ATS algorithm.
    Real ATS systems use different proprietary methods.
    """
    score = (0.45 * similarity_score) + (0.40 * skill_score) + (0.15 * keyword_score)
    return safe_round(score, 2)


def get_score_label(score: float) -> str:
    if score >= 80:
        return "Excellent match"

    if score >= 65:
        return "Good match"

    if score >= 50:
        return "Moderate match"

    return "Needs improvement"


def get_priority_actions(analysis: dict[str, object], max_items: int = 7) -> list[str]:
    missing_skills = analysis.get("missing_skills", []) or []
    missing_keywords = analysis.get("missing_keywords", []) or []
    similarity_score = float(analysis.get("similarity_score", 0) or 0)

    actions = []

    if missing_skills:
        actions.append(
            "Add proof for these job-required skills if you genuinely have them: "
            + ", ".join(missing_skills[:8])
            + "."
        )

    if missing_keywords:
        actions.append(
            "Naturally include important job description keywords: "
            + ", ".join(missing_keywords[:10])
            + "."
        )

    if similarity_score < 55:
        actions.append(
            "Rewrite your resume summary and project bullets so they match the target role more clearly."
        )

    actions.extend(
        [
            "Use measurable achievements with numbers, percentages, accuracy, time saved, cost saved, or project scale.",
            "Start bullet points with strong action verbs such as built, analyzed, automated, deployed, improved, optimized, or delivered.",
            "Keep ATS formatting simple with standard headings such as Summary, Skills, Experience, Projects, and Education.",
            "Move the most relevant technical skills near the top of the resume.",
        ]
    )

    unique_actions = list(dict.fromkeys(actions))
    return unique_actions[:max_items]


def skills_to_dataframe(skills: Iterable[str], status: str) -> pd.DataFrame:
    skills = list(skills)
    lookup = skill_category_lookup(skills)

    rows = []

    for skill in skills:
        rows.append(
            {
                "Skill": skill,
                "Category": lookup.get(skill, "Other"),
                "Status": status,
            }
        )

    return pd.DataFrame(rows)


def analyze_match(resume_text: str, job_description: str) -> dict[str, object]:
    """Run full resume-job matching analysis."""
    similarity_score = calculate_text_similarity(resume_text, job_description)

    skill_results = compare_skills(resume_text, job_description)
    keyword_results = calculate_keyword_score(resume_text, job_description)

    skill_score = float(skill_results["skill_match_score"])
    keyword_score = float(keyword_results["keyword_score"])

    ats_score = calculate_ats_score(
        similarity_score=similarity_score,
        skill_score=skill_score,
        keyword_score=keyword_score,
    )

    analysis = {
        "overall_score": ats_score,
        "ats_score": ats_score,
        "similarity_score": similarity_score,
        "skill_score": skill_score,
        "keyword_score": keyword_score,
        "score_label": get_score_label(ats_score),
        **skill_results,
        **keyword_results,
    }

    analysis["priority_actions"] = get_priority_actions(analysis)

    analysis["resume_skill_category_counts"] = count_skills_by_category(
        analysis["resume_skills"]
    )

    analysis["job_skill_category_counts"] = count_skills_by_category(
        analysis["job_skills"]
    )

    return analysis