from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable


SKILL_CATEGORIES = {
    "Programming": [
        "python",
        "r",
        "sql",
        "java",
        "javascript",
        "typescript",
        "c++",
        "c#",
        "html",
        "css",
        "bash",
        "matlab",
    ],
    "Data Science": [
        "data analysis",
        "data analytics",
        "machine learning",
        "deep learning",
        "statistics",
        "predictive modeling",
        "classification",
        "regression",
        "clustering",
        "nlp",
        "natural language processing",
        "computer vision",
        "feature engineering",
        "exploratory data analysis",
        "model evaluation",
        "time series",
    ],
    "Python Libraries": [
        "pandas",
        "numpy",
        "scikit-learn",
        "sklearn",
        "matplotlib",
        "seaborn",
        "plotly",
        "tensorflow",
        "keras",
        "pytorch",
        "opencv",
        "xgboost",
        "streamlit",
        "gradio",
        "langchain",
    ],
    "Cloud and DevOps": [
        "aws",
        "azure",
        "google cloud",
        "gcp",
        "docker",
        "kubernetes",
        "linux",
        "git",
        "github",
        "gitlab",
        "ci/cd",
        "mlops",
        "fastapi",
        "flask",
        "api",
        "api integration",
    ],
    "Databases": [
        "mysql",
        "postgresql",
        "sqlite",
        "mongodb",
        "redis",
        "bigquery",
        "snowflake",
        "etl",
        "data pipeline",
        "data warehouse",
    ],
    "Business and Soft Skills": [
        "communication",
        "leadership",
        "teamwork",
        "problem solving",
        "project management",
        "stakeholder management",
        "agile",
        "scrum",
        "presentation",
        "documentation",
        "collaboration",
        "critical thinking",
    ],
    "ATS Keywords": [
        "dashboard",
        "reporting",
        "automation",
        "optimization",
        "deployment",
        "production",
        "research",
        "analysis",
        "visualization",
        "forecasting",
        "business intelligence",
        "kpi",
        "requirements",
    ],
}


ALIASES = {
    "sklearn": "scikit-learn",
    "gcp": "google cloud",
    "js": "javascript",
    "ts": "typescript",
    "nlp": "natural language processing",
}


def normalize_skill(skill: str) -> str:
    skill = skill.lower().strip()
    skill = skill.replace("_", " ")
    skill = re.sub(r"\s+", " ", skill)
    return ALIASES.get(skill, skill)


def flatten_skills() -> list[str]:
    all_skills = []

    for skills in SKILL_CATEGORIES.values():
        all_skills.extend(skills)

    return sorted(set(normalize_skill(skill) for skill in all_skills))


def skill_exists_in_text(skill: str, text: str) -> bool:
    escaped = re.escape(skill)
    escaped = escaped.replace(r"\ ", r"[\s\-]+")

    pattern = rf"(?<![a-zA-Z0-9]){escaped}(?![a-zA-Z0-9])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def extract_skills(text: str, skill_bank: Iterable[str] | None = None) -> list[str]:
    text = text or ""
    skills = list(skill_bank) if skill_bank else flatten_skills()

    found = set()

    for skill in skills:
        if skill_exists_in_text(skill, text):
            found.add(normalize_skill(skill))

    return sorted(found)


def extract_skills_by_category(text: str) -> dict[str, list[str]]:
    grouped = {}

    for category, skills in SKILL_CATEGORIES.items():
        found = extract_skills(text, skills)

        if found:
            grouped[category] = found

    return grouped


def compare_skills(resume_text: str, job_description: str) -> dict[str, object]:
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_description))

    matched_skills = sorted(resume_skills.intersection(job_skills))
    missing_skills = sorted(job_skills.difference(resume_skills))
    extra_resume_skills = sorted(resume_skills.difference(job_skills))

    if job_skills:
        skill_match_score = round((len(matched_skills) / len(job_skills)) * 100, 2)
    else:
        skill_match_score = 0.0

    return {
        "resume_skills": sorted(resume_skills),
        "job_skills": sorted(job_skills),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "extra_resume_skills": extra_resume_skills,
        "skill_match_score": skill_match_score,
        "resume_skills_by_category": extract_skills_by_category(resume_text),
        "job_skills_by_category": extract_skills_by_category(job_description),
    }


def skill_category_lookup(skills: Iterable[str]) -> dict[str, str]:
    requested = {normalize_skill(skill) for skill in skills}
    lookup = {}

    for category, category_skills in SKILL_CATEGORIES.items():
        for skill in category_skills:
            canonical = normalize_skill(skill)

            if canonical in requested:
                lookup[canonical] = category

    return lookup


def count_skills_by_category(skills: Iterable[str]) -> dict[str, int]:
    lookup = skill_category_lookup(skills)
    counts = defaultdict(int)

    for skill in skills:
        category = lookup.get(normalize_skill(skill), "Other")
        counts[category] += 1

    return dict(sorted(counts.items()))