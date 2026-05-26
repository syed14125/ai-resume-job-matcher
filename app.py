from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.ai_suggestions import (
    generate_cover_letter,
    generate_interview_questions,
    generate_professional_summary,
    generate_resume_suggestions,
)
from src.matcher import analyze_match, skills_to_dataframe
from src.resume_parser import (
    estimate_word_count,
    extract_email,
    extract_phone,
    extract_text_from_uploaded_file,
)
from src.utils import (
    analysis_to_summary_dataframe,
    build_markdown_report,
    load_text_file,
    percent_to_progress,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


st.set_page_config(
    page_title="AI Resume & Job Matcher",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.16), transparent 32%),
                radial-gradient(circle at top right, rgba(236, 72, 153, 0.13), transparent 30%),
                linear-gradient(135deg, #f8fbff 0%, #eef4ff 45%, #fff7fb 100%);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 55%, #111827 100%);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: #ffffff !important;
        }

        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            color: #111827 !important;
            background-color: #ffffff !important;
        }

        .hero-card {
            padding: 34px;
            border-radius: 30px;
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 50%, #ec4899 100%);
            color: white;
            box-shadow: 0 22px 55px rgba(37, 99, 235, 0.28);
            margin-bottom: 24px;
        }

        .hero-title {
            font-size: 2.8rem;
            font-weight: 850;
            line-height: 1.05;
            margin-bottom: 12px;
        }

        .hero-subtitle {
            font-size: 1.08rem;
            line-height: 1.7;
            opacity: 0.96;
            max-width: 900px;
        }

        .hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 22px;
        }

        .hero-badge {
            padding: 9px 14px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.25);
            color: white;
            font-size: 0.88rem;
            font-weight: 600;
        }

        .section-card {
            padding: 22px;
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(226, 232, 240, 0.95);
            box-shadow: 0 14px 38px rgba(15, 23, 42, 0.08);
            margin-bottom: 18px;
        }

        .mini-card {
            padding: 18px;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid rgba(226, 232, 240, 0.95);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.07);
            height: 100%;
        }

        .score-card {
            padding: 22px;
            border-radius: 24px;
            background: white;
            border: 1px solid #e5e7eb;
            box-shadow: 0 14px 35px rgba(15, 23, 42, 0.08);
            height: 100%;
            position: relative;
            overflow: hidden;
        }

        .score-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 6px;
            background: linear-gradient(90deg, #2563eb, #7c3aed, #ec4899);
        }

        .score-label {
            color: #64748b;
            font-weight: 700;
            font-size: 0.88rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 8px;
        }

        .score-value {
            font-size: 2.2rem;
            font-weight: 850;
            margin-bottom: 6px;
        }

        .score-help {
            color: #64748b;
            font-size: 0.88rem;
            line-height: 1.5;
        }

        .status-pill {
            display: inline-block;
            padding: 9px 15px;
            border-radius: 999px;
            color: white;
            font-weight: 750;
            font-size: 0.92rem;
            margin-bottom: 14px;
        }

        .chip {
            display: inline-block;
            padding: 7px 12px;
            border-radius: 999px;
            background: #eff6ff;
            color: #1d4ed8;
            border: 1px solid #bfdbfe;
            margin: 4px 4px 4px 0;
            font-size: 0.86rem;
            font-weight: 650;
        }

        .chip-green {
            background: #ecfdf5;
            color: #047857;
            border: 1px solid #a7f3d0;
        }

        .chip-red {
            background: #fff1f2;
            color: #be123c;
            border: 1px solid #fecdd3;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px;
            margin-top: 18px;
        }

        .feature-card {
            padding: 20px;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #e5e7eb;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
        }

        .feature-title {
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 8px;
            font-size: 1.02rem;
        }

        .feature-text {
            color: #64748b;
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .action-card {
            padding: 14px 16px;
            border-radius: 18px;
            background: #ffffff;
            border-left: 5px solid #2563eb;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
            margin-bottom: 10px;
            color: #111827;
            font-weight: 500;
        }

        .footer-note {
            text-align: center;
            color: #64748b;
            margin-top: 30px;
            font-size: 0.9rem;
        }

        div.stButton > button {
            border-radius: 14px;
            font-weight: 750;
            padding: 0.75rem 1rem;
            border: none;
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: white;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.22);
        }

        div.stButton > button:hover {
            background: linear-gradient(135deg, #1d4ed8, #6d28d9);
            color: white;
            transform: translateY(-1px);
        }

        [data-testid="stMetricValue"] {
            font-weight: 850;
            color: #0f172a;
        }

        @media (max-width: 900px) {
            .feature-grid {
                grid-template-columns: 1fr;
            }

            .hero-title {
                font-size: 2.1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    sample_jd = load_text_file(DATA_DIR / "sample_job_description.txt")
    sample_resume = load_text_file(DATA_DIR / "sample_resume.txt")

    st.session_state.setdefault("job_description", sample_jd)
    st.session_state.setdefault("resume_text_input", sample_resume)
    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("resume_text_used", "")
    st.session_state.setdefault("job_description_used", "")
    st.session_state.setdefault("ai_outputs", {})


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">🚀 AI Resume & Job Description Matcher</div>
            <div class="hero-subtitle">
                A professional career intelligence tool that analyzes resume-job fit,
                detects missing skills, scores ATS-style relevance, and generates
                resume suggestions, cover letters, summaries, and interview questions.
            </div>
            <div class="hero-badges">
                <span class="hero-badge">ATS-style Score</span>
                <span class="hero-badge">Skill Gap Analysis</span>
                <span class="hero-badge">Keyword Matching</span>
                <span class="hero-badge">AI Suggestions</span>
                <span class="hero-badge">Download Report</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_color(score: float) -> str:
    if score >= 80:
        return "#059669"
    if score >= 65:
        return "#2563eb"
    if score >= 50:
        return "#f59e0b"
    return "#dc2626"


def score_level(score: float) -> str:
    if score >= 80:
        return "Excellent Match"
    if score >= 65:
        return "Good Match"
    if score >= 50:
        return "Moderate Match"
    return "Needs Improvement"


def show_score_card(title: str, score: float, help_text: str) -> None:
    color = score_color(score)

    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-label">{title}</div>
            <div class="score-value" style="color:{color};">{score}%</div>
            <div class="score-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(percent_to_progress(score))


def render_status_pill(score: float) -> None:
    color = score_color(score)
    label = score_level(score)

    st.markdown(
        f"""
        <span class="status-pill" style="background:{color};">
            {label}
        </span>
        """,
        unsafe_allow_html=True,
    )


def render_chips(items: list[str], chip_class: str = "chip", empty_text: str = "None found") -> None:
    if not items:
        st.caption(empty_text)
        return

    chips_html = " ".join(
        [f'<span class="{chip_class}">{item}</span>' for item in items]
    )
    st.markdown(chips_html, unsafe_allow_html=True)


def get_resume_text(uploaded_file, pasted_text: str) -> str:
    if uploaded_file is not None:
        return extract_text_from_uploaded_file(uploaded_file)

    return pasted_text.strip()


def render_landing_features() -> None:
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-title">🎯 Smart Matching</div>
                <div class="feature-text">
                    Uses TF-IDF similarity, skill extraction, and keyword matching to compare resume and job description.
                </div>
            </div>
            <div class="feature-card">
                <div class="feature-title">🧩 Skills Gap</div>
                <div class="feature-text">
                    Shows matched skills, missing skills, extra resume skills, and category-level breakdown.
                </div>
            </div>
            <div class="feature-card">
                <div class="feature-title">✨ Career Content</div>
                <div class="feature-text">
                    Generates resume improvements, professional summaries, cover letters, and interview questions.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_section(uploaded_resume):
    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📄 Resume Input")

        pasted_resume = st.text_area(
            "Paste resume text here if you do not upload a file",
            key="resume_text_input",
            height=350,
            placeholder="Paste your resume text here...",
        )

        if uploaded_resume is not None:
            st.success(f"Uploaded: {uploaded_resume.name}")

        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("💼 Job Description")

        job_description = st.text_area(
            "Paste target job description",
            key="job_description",
            height=350,
            placeholder="Paste the job description here...",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    return pasted_resume, job_description


def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")
        st.caption("Upload resume, load samples, and configure optional AI mode.")

        uploaded_resume = st.file_uploader(
            "Upload resume",
            type=["pdf", "docx", "txt"],
            help="Supported formats: PDF, DOCX, TXT",
        )

        st.divider()

        st.markdown("### 🧪 Sample Data")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Sample Resume", use_container_width=True):
                st.session_state["resume_text_input"] = load_text_file(
                    DATA_DIR / "sample_resume.txt"
                )
                st.success("Loaded")

        with col2:
            if st.button("Sample JD", use_container_width=True):
                st.session_state["job_description"] = load_text_file(
                    DATA_DIR / "sample_job_description.txt"
                )
                st.success("Loaded")

        st.divider()

        st.markdown("### ✨ Optional AI Mode")

        api_key = st.text_input(
            "Gemini API key",
            type="password",
            help="Optional. Without a key, the app uses rule-based suggestions.",
        )

        role = st.text_input("Target role", value="Data Analyst")
        company = st.text_input("Company name", value="the company")

        st.divider()

        st.markdown("### 📌 Project Mode")
        st.info(
            "Free mode uses TF-IDF, skill matching, keyword analysis, and rule-based suggestions."
        )

    return uploaded_resume, api_key, role, company


def run_analysis(uploaded_resume, pasted_resume: str, job_description: str) -> None:
    try:
        resume_text = get_resume_text(uploaded_resume, pasted_resume)
    except Exception as error:
        st.error(f"Could not read resume file: {error}")
        return

    if len(resume_text.strip()) < 50:
        st.error("Please upload or paste a longer resume.")
        return

    if len(job_description.strip()) < 50:
        st.error("Please paste a longer job description.")
        return

    with st.spinner("Analyzing resume and job description..."):
        analysis = analyze_match(resume_text, job_description)

    st.session_state["analysis"] = analysis
    st.session_state["resume_text_used"] = resume_text
    st.session_state["job_description_used"] = job_description
    st.session_state["ai_outputs"] = {}

    st.success("Analysis complete.")


def render_candidate_info(resume_text: str) -> None:
    st.markdown("### 👤 Candidate Information Detected")

    contact_cols = st.columns(3)

    with contact_cols[0]:
        st.markdown('<div class="mini-card">', unsafe_allow_html=True)
        st.metric("Resume Words", estimate_word_count(resume_text))
        st.caption("Estimated total words")
        st.markdown("</div>", unsafe_allow_html=True)

    with contact_cols[1]:
        st.markdown('<div class="mini-card">', unsafe_allow_html=True)
        st.metric("Email", extract_email(resume_text) or "Not found")
        st.caption("Detected from resume text")
        st.markdown("</div>", unsafe_allow_html=True)

    with contact_cols[2]:
        st.markdown('<div class="mini-card">', unsafe_allow_html=True)
        st.metric("Phone", extract_phone(resume_text) or "Not found")
        st.caption("Detected from resume text")
        st.markdown("</div>", unsafe_allow_html=True)


def render_score_overview(analysis: dict[str, object]) -> None:
    st.markdown("### 📊 Match Overview")
    render_status_pill(float(analysis["overall_score"]))

    score_cols = st.columns(4)

    with score_cols[0]:
        show_score_card(
            "Overall Score",
            float(analysis["overall_score"]),
            "Weighted ATS-style project score.",
        )

    with score_cols[1]:
        show_score_card(
            "Text Similarity",
            float(analysis["similarity_score"]),
            "Resume and job description similarity.",
        )

    with score_cols[2]:
        show_score_card(
            "Skill Match",
            float(analysis["skill_score"]),
            "Job-required skills found in resume.",
        )

    with score_cols[3]:
        show_score_card(
            "Keyword Match",
            float(analysis["keyword_score"]),
            "Important job keywords found in resume.",
        )


def render_overview_tab(analysis: dict[str, object]) -> None:
    st.subheader("🚦 Priority Actions")

    for index, action in enumerate(analysis["priority_actions"], start=1):
        st.markdown(
            f"""
            <div class="action-card">
                <strong>{index}.</strong> {action}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("📈 Score Breakdown")

    score_df = analysis_to_summary_dataframe(analysis)
    st.dataframe(score_df, use_container_width=True, hide_index=True)

    chart_df = score_df.set_index("Metric")
    st.bar_chart(chart_df)

    st.caption(
        "This score is transparent and portfolio-friendly. It is not an official ATS algorithm."
    )


def render_skills_tab(analysis: dict[str, object]) -> None:
    st.subheader("🧩 Skill Gap Analysis")

    matched_df = skills_to_dataframe(analysis["matched_skills"], "Matched")
    missing_df = skills_to_dataframe(analysis["missing_skills"], "Missing from resume")
    extra_df = skills_to_dataframe(analysis["extra_resume_skills"], "Extra in resume")

    chip_col1, chip_col2 = st.columns(2)

    with chip_col1:
        st.markdown("#### ✅ Matched Skills")
        render_chips(analysis["matched_skills"], "chip chip-green", "No matched skills found.")

    with chip_col2:
        st.markdown("#### ⚠️ Missing Skills")
        render_chips(analysis["missing_skills"], "chip chip-red", "No missing skills detected.")

    st.divider()

    table_col1, table_col2 = st.columns(2)

    with table_col1:
        st.markdown("#### Matched Skills Table")
        st.dataframe(matched_df, use_container_width=True, hide_index=True)

    with table_col2:
        st.markdown("#### Missing Skills Table")
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

    with st.expander("View extra resume skills"):
        if extra_df.empty:
            st.info("No extra resume skills detected.")
        else:
            st.dataframe(extra_df, use_container_width=True, hide_index=True)

    st.subheader("📚 Skill Categories")

    category_col1, category_col2 = st.columns(2)

    with category_col1:
        st.markdown("#### Resume Skill Categories")
        resume_counts = analysis["resume_skill_category_counts"]

        if resume_counts:
            resume_cat_df = pd.DataFrame(
                list(resume_counts.items()),
                columns=["Category", "Count"],
            ).set_index("Category")
            st.bar_chart(resume_cat_df)
        else:
            st.info("No resume skill categories found.")

    with category_col2:
        st.markdown("#### Job Skill Categories")
        job_counts = analysis["job_skill_category_counts"]

        if job_counts:
            job_cat_df = pd.DataFrame(
                list(job_counts.items()),
                columns=["Category", "Count"],
            ).set_index("Category")
            st.bar_chart(job_cat_df)
        else:
            st.info("No job skill categories found.")


def render_keywords_tab(analysis: dict[str, object]) -> None:
    st.subheader("🔑 ATS Keyword Analysis")

    keyword_cols = st.columns(2)

    with keyword_cols[0]:
        st.markdown("#### ✅ Matched Job Keywords")
        render_chips(
            analysis["matched_keywords"],
            "chip chip-green",
            "No matched keywords found.",
        )

        matched_keywords_df = pd.DataFrame(
            {"Matched Keywords": analysis["matched_keywords"]}
        )
        st.dataframe(matched_keywords_df, use_container_width=True, hide_index=True)

    with keyword_cols[1]:
        st.markdown("#### ⚠️ Missing Job Keywords")
        render_chips(
            analysis["missing_keywords"],
            "chip chip-red",
            "No missing keywords found.",
        )

        missing_keywords_df = pd.DataFrame(
            {"Missing Keywords": analysis["missing_keywords"]}
        )
        st.dataframe(missing_keywords_df, use_container_width=True, hide_index=True)


def generate_ai_outputs(
    resume_text: str,
    job_description: str,
    analysis: dict[str, object],
    api_key: str,
    role: str,
    company: str,
) -> dict[str, str]:
    resume_suggestions = generate_resume_suggestions(
        resume_text=resume_text,
        job_description=job_description,
        analysis=analysis,
        api_key=api_key,
    )

    professional_summary = generate_professional_summary(
        resume_text=resume_text,
        job_description=job_description,
        analysis=analysis,
        role=role,
        api_key=api_key,
    )

    cover_letter = generate_cover_letter(
        resume_text=resume_text,
        job_description=job_description,
        role=role,
        company=company,
        api_key=api_key,
    )

    interview_questions = generate_interview_questions(
        job_description=job_description,
        analysis=analysis,
        role=role,
        api_key=api_key,
    )

    return {
        "resume_suggestions": resume_suggestions,
        "professional_summary": professional_summary,
        "cover_letter": cover_letter,
        "interview_questions": interview_questions,
    }


def render_ai_tab(
    resume_text: str,
    job_description: str,
    analysis: dict[str, object],
    api_key: str,
    role: str,
    company: str,
) -> None:
    st.subheader("✨ AI Suggestions and Career Content")

    st.markdown(
        """
        <div class="section-card">
            <strong>Tip:</strong> This section works even without Gemini API.
            Without an API key, the app uses rule-based suggestions. With Gemini,
            it generates stronger personalized content.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("✨ Generate Suggestions and Documents", use_container_width=True):
        with st.spinner("Generating suggestions and career content..."):
            st.session_state["ai_outputs"] = generate_ai_outputs(
                resume_text=resume_text,
                job_description=job_description,
                analysis=analysis,
                api_key=api_key,
                role=role,
                company=company,
            )

    ai_outputs = st.session_state.get("ai_outputs", {})

    if not ai_outputs:
        st.info(
            "Click the button above to generate resume suggestions, summary, cover letter, and interview questions."
        )
        return

    ai_tabs = st.tabs(
        [
            "Resume Improvements",
            "Professional Summary",
            "Cover Letter",
            "Interview Questions",
        ]
    )

    with ai_tabs[0]:
        st.markdown(ai_outputs["resume_suggestions"])

    with ai_tabs[1]:
        st.markdown(ai_outputs["professional_summary"])

    with ai_tabs[2]:
        st.markdown(ai_outputs["cover_letter"])

    with ai_tabs[3]:
        st.markdown(ai_outputs["interview_questions"])


def render_report_tab(analysis: dict[str, object]) -> None:
    st.subheader("📥 Download Full Report")

    ai_outputs = st.session_state.get("ai_outputs", {})

    if not ai_outputs:
        st.warning(
            "AI content has not been generated yet. The report will include scores, skills, keywords, and priority actions only."
        )

    report = build_markdown_report(
        analysis=analysis,
        resume_suggestions=ai_outputs.get("resume_suggestions", ""),
        professional_summary=ai_outputs.get("professional_summary", ""),
        cover_letter=ai_outputs.get("cover_letter", ""),
        interview_questions=ai_outputs.get("interview_questions", ""),
    )

    st.download_button(
        label="⬇️ Download Markdown Report",
        data=report,
        file_name="resume_job_match_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

    with st.expander("Preview report"):
        st.markdown(report)


def main() -> None:
    apply_custom_css()
    initialize_state()
    render_hero()

    uploaded_resume, api_key, role, company = render_sidebar()

    pasted_resume, job_description = render_input_section(uploaded_resume)

    analyze_clicked = st.button(
        "🚀 Analyze Resume Match",
        type="primary",
        use_container_width=True,
    )

    if analyze_clicked:
        run_analysis(uploaded_resume, pasted_resume, job_description)

    analysis = st.session_state.get("analysis")

    if not analysis:
        render_landing_features()
        st.markdown(
            """
            <div class="footer-note">
                Upload or paste a resume, paste a job description, then click Analyze Resume Match.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    resume_text = st.session_state["resume_text_used"]
    job_description_used = st.session_state["job_description_used"]

    render_candidate_info(resume_text)
    render_score_overview(analysis)

    tab_overview, tab_skills, tab_keywords, tab_ai, tab_report = st.tabs(
        [
            "🏠 Overview",
            "🧩 Skills Gap",
            "🔑 ATS Keywords",
            "✨ AI Suggestions",
            "📥 Report",
        ]
    )

    with tab_overview:
        render_overview_tab(analysis)

    with tab_skills:
        render_skills_tab(analysis)

    with tab_keywords:
        render_keywords_tab(analysis)

    with tab_ai:
        render_ai_tab(
            resume_text=resume_text,
            job_description=job_description_used,
            analysis=analysis,
            api_key=api_key,
            role=role,
            company=company,
        )

    with tab_report:
        render_report_tab(analysis)


if __name__ == "__main__":
    main()
