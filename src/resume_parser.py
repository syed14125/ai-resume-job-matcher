from __future__ import annotations

import re
from pathlib import Path
from typing import BinaryIO

import fitz
from docx import Document


def clean_text(text: str) -> str:
    """Clean extracted resume or job description text."""
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_text_from_pdf(file: BinaryIO | str | Path) -> str:
    """Extract text from a PDF file."""
    text_parts = []

    if isinstance(file, (str, Path)):
        document = fitz.open(str(file))
    else:
        file_bytes = file.read()
        document = fitz.open(stream=file_bytes, filetype="pdf")

    with document:
        for page in document:
            text_parts.append(page.get_text("text"))

    return clean_text("\n".join(text_parts))


def extract_text_from_docx(file: BinaryIO | str | Path) -> str:
    """Extract text from a DOCX file."""
    document = Document(file)
    paragraphs = [paragraph.text for paragraph in document.paragraphs]

    return clean_text("\n".join(paragraphs))


def extract_text_from_txt(file: BinaryIO | str | Path) -> str:
    """Extract text from a TXT file."""
    if isinstance(file, (str, Path)):
        return clean_text(Path(file).read_text(encoding="utf-8", errors="ignore"))

    raw = file.read()

    if isinstance(raw, bytes):
        return clean_text(raw.decode("utf-8", errors="ignore"))

    return clean_text(str(raw))


def extract_text_from_uploaded_file(uploaded_file) -> str:
    """Extract text from uploaded PDF, DOCX, or TXT file."""
    if uploaded_file is None:
        return ""

    suffix = Path(uploaded_file.name).suffix.lower()

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    if suffix == ".pdf":
        return extract_text_from_pdf(uploaded_file)

    if suffix == ".docx":
        return extract_text_from_docx(uploaded_file)

    if suffix == ".txt":
        return extract_text_from_txt(uploaded_file)

    raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")


def estimate_word_count(text: str) -> int:
    """Estimate word count."""
    return len(re.findall(r"\b\w+\b", text or ""))


def extract_email(text: str) -> str:
    """Extract the first email address from text."""
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text or "")

    if match:
        return match.group(0)

    return ""


def extract_phone(text: str) -> str:
    """Extract a likely phone number from text."""
    pattern = r"(\+?\d[\d\s\-\(\)]{7,}\d)"
    match = re.search(pattern, text or "")

    if not match:
        return ""

    return re.sub(r"\s+", " ", match.group(0)).strip()