from __future__ import annotations

import os
import textwrap


def get_streamlit_secret(name: str) -> str:
    """Read secret from Streamlit secrets if available."""
    try:
        import streamlit as st

        value = st.secrets.get(name, "")
        return str(value).strip() if value else ""
    except Exception:
        return ""


def get_api_key(api_key: str | None = None) -> str:
    """Get Gemini API key from user input, environment variable, or Streamlit secrets."""
    if api_key and api_key.strip():
        return api_key.strip()

    env_key = os.getenv("GEMINI_API_KEY", "").strip()

    if env_key:
        return env_key

    secret_key = get_streamlit_secret("GEMINI_API_KEY")

    if secret_key:
        return secret_key

    return ""


def call_gemini(
    prompt: str,
    api_key: str | None = None,
    model_name: str = "gemini-1.5-flash",
) -> str:
    """
    Call Gemini API.

    If no API key is available, this returns an empty string.
    The app will then use fallback rule-based content.
    """
    key = get_api_key(api_key)

    if not key:
        return ""

    try:
        import google.generativeai as genai

        genai.configure(api_key=key)

        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)

        if response and response.text:
            return response.text.strip()

        return ""

    except Exception as error:
        return f"AI generation failed: {error}"


def clip_text(text: str, max_chars: int = 5000) -> str:
    """Limit long resume/job description text before sending to AI."""
    return (text or "")[:max_chars]


def fallback_resume_suggestions(analysis: dict[str, object]) -> str:
    """Rule-based resume suggestions when Gemini API is not used."""
    missing_skills = analysis.get("missing_skills", []) or []
    missing_keywords = analysis.get("missing_keywords", []) or []
    score = analysis.get("overall_score", 0)

    lines = [
        "### Resume Improvement Suggestions",
        "",
        f"Your current ATS-style match score is **{score}%**.",
        "",
        "1. Improve your professional summary by mentioning the target role and your strongest relevant skills.",
        "2. Rewrite project or experience bullet points using this format: action verb + task + tool + result.",
        "3. Add measurable results such as accuracy, time saved, cost saved, users served, or project size.",
        "4. Keep the resume ATS-friendly with simple headings and clean formatting.",
    ]

    if missing_skills:
        lines.append(
            "5. Add these missing skills only if you genuinely have them: "
            + ", ".join(missing_skills[:10])
            + "."
        )

    if missing_keywords:
        lines.append(
            "6. Include these job keywords naturally: "
            + ", ".join(missing_keywords[:12])
            + "."
        )

    lines.append(
        "7. Do not add fake skills. Add proof through projects, experience, or coursework."
    )

    return "\n".join(lines)


def generate_resume_suggestions(
    resume_text: str,
    job_description: str,
    analysis: dict[str, object],
    api_key: str | None = None,
) -> str:
    """Generate resume improvement suggestions."""
    prompt = f"""
You are an expert resume reviewer and ATS optimization assistant.

Give practical resume improvement suggestions for the candidate.

Rules:
- Do not invent experience.
- Do not add fake skills.
- Suggestions must be realistic and useful.
- Use clear headings.

Return:
1. Quick diagnosis
2. Top resume improvements
3. Rewritten bullet point examples
4. ATS keyword advice

Analysis:
{analysis}

Resume:
{clip_text(resume_text)}

Job Description:
{clip_text(job_description)}
"""

    ai_text = call_gemini(prompt, api_key=api_key)

    if ai_text:
        return ai_text

    return fallback_resume_suggestions(analysis)


def generate_cover_letter(
    resume_text: str,
    job_description: str,
    role: str = "the role",
    company: str = "your company",
    api_key: str | None = None,
) -> str:
    """Generate a cover letter."""
    prompt = f"""
Write a concise professional cover letter for {role} at {company}.

Rules:
- Use only information supported by the resume.
- Do not invent years of experience.
- Do not invent companies, degrees, or certifications.
- Keep it under 300 words.

Resume:
{clip_text(resume_text)}

Job Description:
{clip_text(job_description)}
"""

    ai_text = call_gemini(prompt, api_key=api_key)

    if ai_text:
        return ai_text

    return textwrap.dedent(
        f"""
        Dear Hiring Manager,

        I am excited to apply for {role} at {company}. My background includes hands-on experience with technical problem-solving, project work, analysis, and clear communication.

        Based on the job description, I am especially interested in contributing to work involving data analysis, reporting, process improvement, and practical solution building. I enjoy learning new tools, applying structured thinking, and turning information into useful results.

        I would welcome the opportunity to discuss how my skills, projects, and motivation can support your team.

        Sincerely,
        [Your Name]
        """
    ).strip()


def generate_interview_questions(
    job_description: str,
    analysis: dict[str, object],
    role: str = "this role",
    api_key: str | None = None,
) -> str:
    """Generate interview preparation questions."""
    prompt = f"""
Create interview preparation questions for {role}.

Return:
- 8 technical questions
- 5 behavioral questions
- 5 resume/project questions
- 5 questions the candidate can ask the employer

Analysis:
{analysis}

Job Description:
{clip_text(job_description)}
"""

    ai_text = call_gemini(prompt, api_key=api_key)

    if ai_text:
        return ai_text

    missing_skills = analysis.get("missing_skills", []) or []

    if missing_skills:
        skill_text = ", ".join(missing_skills[:5])
    else:
        skill_text = "the main tools and responsibilities in the job description"

    return textwrap.dedent(
        f"""
        ### Technical Questions

        1. Explain one project where you solved a real problem.
        2. How do you approach a new dataset or unfamiliar business problem?
        3. How do you clean and validate data before analysis?
        4. How do you explain technical findings to a non-technical person?
        5. What tools have you used for analysis, reporting, or visualization?
        6. How would you improve a dashboard or report after user feedback?
        7. What is your experience with {skill_text}?
        8. How do you check if your work is accurate?

        ### Behavioral Questions

        1. Tell me about a time you learned a new skill quickly.
        2. Describe a time you worked under pressure.
        3. Tell me about a mistake you made and what you learned.
        4. How do you prioritize multiple tasks?
        5. Describe a time you worked with a team.

        ### Resume and Project Questions

        1. Which project on your resume are you most proud of?
        2. What problem did that project solve?
        3. What tools did you use?
        4. What was the final result?
        5. What would you improve if you repeated the project?

        ### Questions to Ask the Employer

        1. What does success look like in the first 90 days?
        2. What tools does the team currently use?
        3. What are the biggest challenges in this role?
        4. How is performance measured?
        5. What growth opportunities are available?
        """
    ).strip()


def generate_professional_summary(
    resume_text: str,
    job_description: str,
    analysis: dict[str, object],
    role: str = "target role",
    api_key: str | None = None,
) -> str:
    """Generate professional summary options."""
    prompt = f"""
Write 3 ATS-friendly professional summary options for a resume targeting {role}.

Rules:
- Do not invent years of experience.
- Do not invent companies.
- Do not invent certifications.
- Use only skills supported by the resume and analysis.

Analysis:
{analysis}

Resume:
{clip_text(resume_text)}

Job Description:
{clip_text(job_description)}
"""

    ai_text = call_gemini(prompt, api_key=api_key)

    if ai_text:
        return ai_text

    matched_skills = analysis.get("matched_skills", []) or []

    if matched_skills:
        skill_phrase = ", ".join(matched_skills[:6])
    else:
        skill_phrase = "analysis, problem-solving, communication, and project delivery"

    return textwrap.dedent(
        f"""
        ### Professional Summary Option 1

        Motivated candidate targeting a {role}, with hands-on project experience in {skill_phrase}. Strong ability to learn quickly, solve problems, and communicate results clearly.

        ### Professional Summary Option 2

        Detail-oriented candidate with practical experience in {skill_phrase}. Interested in applying technical and analytical skills to real business problems and measurable outcomes.

        ### Professional Summary Option 3

        Results-focused candidate with a strong foundation in {skill_phrase}. Skilled at structured thinking, documentation, collaboration, and continuous improvement.
        """
    ).strip()