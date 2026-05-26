# AI Resume & Job Description Matcher

An AI-powered Streamlit web app that compares a resume with a target job description and gives an ATS-style match score, missing skills, keyword gaps, and improvement suggestions.

This project is built for portfolio use and works without any paid API. It also supports optional Gemini AI mode for stronger resume suggestions, cover letter generation, interview questions, and professional summary generation.

---

## Features

- Upload resume as PDF, DOCX, or TXT
- Paste resume manually
- Paste job description
- Calculate ATS-style resume match score
- TF-IDF text similarity
- Skill gap analysis
- Job keyword matching
- Resume improvement suggestions
- Professional summary generator
- Cover letter generator
- Interview question generator
- Downloadable Markdown report
- Works without API key
- Optional Gemini API support

---

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- PyMuPDF
- python-docx
- Google Gemini API optional

---

## Project Structure

```text
ai-resume-job-matcher/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── src/
│   ├── __init__.py
│   ├── resume_parser.py
│   ├── skill_extractor.py
│   ├── matcher.py
│   ├── ai_suggestions.py
│   └── utils.py
├── data/
│   ├── sample_job_description.txt
│   └── sample_resume.txt
├── evaluation/
│   ├── test_cases.csv
│   └── evaluation_results.csv
├── reports/
│   └── .gitkeep
└── assets/
    └── .gitkeep