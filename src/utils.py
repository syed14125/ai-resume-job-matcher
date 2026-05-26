from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_text_file(path: str | Path, default: str = "") -> str:
    """Load text from a file. Return default if file does not exist."""
    path = Path(path)

    if not path.exists():
        return default

    return path.read_text(encoding="utf-8", errors="ignore")


def percent_to_progress(score: float) -> float:
    """Convert percentage score into Streamlit progress value from 0.0 to 1.0."""
    try:
        score = float(score)
    except Exception:
        return 0.0

    return max(0.0, min(score / 100, 1.0))


def format_list(items: list[str], empty_message: str = "None found") -> str:
    """Format list into readable comma-separated text."""
    if not items:
        return empty_message

    return ", ".join(items)


def build_markdown_report(
    analysis: dict[str, Any],
    resume_suggestions: str = "",
    cover_letter: str = "",
    interview_questions: str = "",
    professional_summary: str = "",
) -> str:
    """Build downloadable Markdown report."""
    matched_skills = analysis.get("matched_skills", []) or []
    missing_skills = analysis.get("missing_skills", []) or []
    matched_keywords = analysis.get("matched_keywords", []) or []
    missing_keywords = analysis.get("missing_keywords", []) or []
    priority_actions = analysis.get("priority_actions", []) or []

    lines = [
        "# AI Resume & Job Description Matcher Report",
        "",
        "## Scores",
        f"- Overall ATS-style score: {analysis.get('overall_score', 0)}%",
        f"- Text similarity score: {analysis.get('similarity_score', 0)}%",
        f"- Skill match score: {analysis.get('skill_score', 0)}%",
        f"- Keyword score: {analysis.get('keyword_score', 0)}%",
        f"- Recommendation: {analysis.get('score_label', 'N/A')}",
        "",
        "## Matched Skills",
        format_list(matched_skills),
        "",
        "## Missing Skills",
        format_list(missing_skills),
        "",
        "## Matched Keywords",
        format_list(matched_keywords),
        "",
        "## Missing Keywords",
        format_list(missing_keywords),
        "",
        "## Priority Actions",
    ]

    if priority_actions:
        for index, action in enumerate(priority_actions, start=1):
            lines.append(f"{index}. {action}")
    else:
        lines.append("No priority actions generated.")

    optional_sections = [
        ("Resume Suggestions", resume_suggestions),
        ("Professional Summary", professional_summary),
        ("Cover Letter", cover_letter),
        ("Interview Questions", interview_questions),
    ]

    for title, content in optional_sections:
        if content:
            lines.extend(["", f"## {title}", content])

    return "\n".join(lines)


def analysis_to_summary_dataframe(analysis: dict[str, Any]) -> pd.DataFrame:
    """Convert score summary into a DataFrame for Streamlit display."""
    rows = [
        {
            "Metric": "Overall ATS-style score",
            "Score": analysis.get("overall_score", 0),
        },
        {
            "Metric": "Text similarity",
            "Score": analysis.get("similarity_score", 0),
        },
        {
            "Metric": "Skill match",
            "Score": analysis.get("skill_score", 0),
        },
        {
            "Metric": "Keyword match",
            "Score": analysis.get("keyword_score", 0),
        },
    ]

    return pd.DataFrame(rows)


def list_to_dataframe(items: list[str], column_name: str) -> pd.DataFrame:
    """Convert simple list into DataFrame."""
    return pd.DataFrame({column_name: items})