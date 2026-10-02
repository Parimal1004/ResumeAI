"""
Core analysis services:
  - PDF extraction
  - Resume parsing & information extraction
  - Skill extraction
  - Job description analysis
  - Semantic similarity
  - Scoring
  - Improvement suggestions
  - Interview question generation
"""
from __future__ import annotations

import json
import os
import re
import string
from pathlib import Path
from typing import Any

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Lazy singletons – loaded once and reused
# ---------------------------------------------------------------------------
_nlp = None
_embedder = None
_skills_db: dict[str, Any] = {}
_all_skills: list[str] = []
_skill_aliases: dict[str, str] = {}

DATA_DIR = Path(__file__).parent.parent / "data"


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fall back to blank English model if the model isn't installed yet
            _nlp = spacy.blank("en")
    return _nlp


def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _embedder = None
    return _embedder


def _get_skills_db() -> tuple[dict, list[str], dict[str, str]]:
    global _skills_db, _all_skills, _skill_aliases
    if not _skills_db:
        skills_path = DATA_DIR / "skills.json"
        if skills_path.exists():
            with open(skills_path, "r", encoding="utf-8") as f:
                _skills_db = json.load(f)
        all_skills_set: set[str] = set()
        for key, values in _skills_db.items():
            if key != "aliases" and isinstance(values, list):
                all_skills_set.update(v.lower() for v in values)
        _all_skills = list(all_skills_set)
        _skill_aliases = {
            k.lower(): v.lower()
            for k, v in _skills_db.get("aliases", {}).items()
        }
    return _skills_db, _all_skills, _skill_aliases


# ---------------------------------------------------------------------------
# PDF EXTRACTION
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes using PyMuPDF."""
    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[str] = []

    for page in doc:
        pages.append(page.get_text())

    doc.close()

    raw = "\n".join(pages)
    return _clean_text(raw)


def extract_links_from_pdf(pdf_bytes: bytes) -> list[str]:
    """
    Extract clickable URLs embedded in PDF annotations.

    This is important for resumes where LinkedIn/GitHub are displayed
    as clickable text instead of showing the complete URL.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    links: list[str] = []

    for page in doc:
        for link in page.get_links():
            uri = link.get("uri")

            if uri and isinstance(uri, str):
                uri = uri.strip()

                if uri and uri not in links:
                    links.append(uri)

    doc.close()

    return links


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove control characters."""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove null bytes and other non-printable chars (keep newlines/tabs)
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ch.isprintable())
    return text.strip()


# ---------------------------------------------------------------------------
# RESUME INFORMATION EXTRACTION
# ---------------------------------------------------------------------------

_SECTION_HEADERS = {
    "education": ["education", "academic", "qualification", "degree", "university", "college"],
    "experience": ["experience", "work history", "employment", "professional background", "internship", "intern"],
    "skills": ["skills", "technical skills", "core competencies", "technologies", "expertise"],
    "projects": ["project", "personal project", "academic project", "portfolio"],
    "certifications": ["certification", "certificate", "license", "credential", "achievement"],
    "summary": ["summary", "objective", "profile", "about me", "overview"],
}

_ACTION_VERBS = {
    "developed", "designed", "implemented", "built", "created", "led", "managed",
    "optimized", "improved", "analyzed", "researched", "deployed", "maintained",
    "collaborated", "delivered", "engineered", "architected", "automated", "reduced",
    "increased", "achieved", "launched", "spearheaded", "coordinated", "established",
    "trained", "mentored", "presented", "published", "contributed", "integrated",
    "migrated", "refactored", "tested", "debugged", "documented", "configured",
}


def extract_resume_info(
    text: str,
    links: list[str] | None = None,
) -> dict[str, Any]:
    """Extract structured information from resume text and PDF links."""

    links = links or []

    info: dict[str, Any] = {
        "name": _extract_name(text),
        "email": _extract_email(text),
        "phone": _extract_phone(text),
        "linkedin": _extract_linkedin(text, links),
        "github": _extract_github(text, links),
        "education": _extract_section(text, "education"),
        "experience": _extract_section(text, "experience"),
        "projects": _extract_section(text, "projects"),
        "certifications": _extract_section(text, "certifications"),
        "summary": _extract_section(text, "summary"),
        "skills": extract_skills_from_text(text),
    }

    return info


def _extract_email(text: str) -> str:
    match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else "Not detected"


def _extract_phone(text: str) -> str:
    match = re.search(
        r"(\+?\d{1,3}[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}", text
    )
    return match.group(0).strip() if match else "Not detected"


def _extract_linkedin(
    text: str,
    links: list[str] | None = None,
) -> str:
    """Extract LinkedIn URL from text or embedded PDF links."""

    links = links or []

    # First check normal extracted text
    match = re.search(
        r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w\-]+/?",
        text,
        re.IGNORECASE,
    )

    if match:
        url = match.group(0)
        if not url.lower().startswith("http"):
            url = "https://" + url
        return url.rstrip("/")

    # Then check clickable PDF hyperlinks
    for link in links:
        if "linkedin.com/in/" in link.lower():
            return link.strip().rstrip("/")

    return "Not detected"


def _extract_github(
    text: str,
    links: list[str] | None = None,
) -> str:
    """Extract GitHub URL from text or embedded PDF links."""

    links = links or []

    # First check normal extracted text
    match = re.search(
        r"(?:https?://)?(?:www\.)?github\.com/[\w\-]+/?",
        text,
        re.IGNORECASE,
    )

    if match:
        url = match.group(0)
        if not url.lower().startswith("http"):
            url = "https://" + url
        return url.rstrip("/")

    # Then check clickable PDF hyperlinks
    for link in links:
        if "github.com/" in link.lower():
            return link.strip().rstrip("/")

    return "Not detected"


def _extract_name(text: str) -> str:
    """Heuristic: first non-empty line that looks like a name."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:5]:
        # Skip lines that look like emails, phones, URLs, or section headers
        if re.search(r"[@/\d]", line):
            continue
        if len(line.split()) in (2, 3) and all(
            w[0].isupper() for w in line.split() if w
        ):
            return line
    return "Not detected"


def _extract_section(text: str, section: str) -> str:
    """Extract the text block belonging to a named section."""
    headers = _SECTION_HEADERS.get(section, [])
    lines = text.split("\n")
    section_lines: list[str] = []
    in_section = False

    for i, line in enumerate(lines):
        lower = line.lower().strip()
        # Check if this line is a header for our target section
        if any(h in lower for h in headers) and len(lower) < 60:
            in_section = True
            continue
        # Stop at the next section header
        if in_section and any(
            any(h in lower for h in hlist)
            for sec, hlist in _SECTION_HEADERS.items()
            if sec != section and len(lower) < 60
        ):
            break
        if in_section and line.strip():
            section_lines.append(line.strip())

    result = "\n".join(section_lines).strip()
    return result if result else "Not detected"


# ---------------------------------------------------------------------------
# SKILL EXTRACTION
# ---------------------------------------------------------------------------

def extract_skills_from_text(text: str) -> list[str]:
    """Return deduplicated list of skills found in text."""
    _, all_skills, aliases = _get_skills_db()
    text_lower = text.lower()
    found: set[str] = set()

    for skill in all_skills:
        # Use word-boundary matching for short tokens (≤3 chars) to avoid false positives
        if len(skill) <= 3:
            pattern = r"\b" + re.escape(skill) + r"\b"
        else:
            pattern = re.escape(skill)
        if re.search(pattern, text_lower):
            found.add(skill)

    # Resolve aliases
    for alias, canonical in aliases.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", text_lower):
            found.add(canonical)

    return sorted(found)


# ---------------------------------------------------------------------------
# JOB DESCRIPTION ANALYSIS
# ---------------------------------------------------------------------------

_REQUIRED_MARKERS = ["required", "must have", "must-have", "you must", "requirements"]
_PREFERRED_MARKERS = [
    "preferred", "nice to have", "nice-to-have", "bonus", "plus", "desired", "ideally"
]


def analyze_job_description(jd_text: str) -> dict[str, Any]:
    """Extract structured information from a job description."""
    jd_lower = jd_text.lower()

    title = _extract_job_title(jd_text)
    all_skills = extract_skills_from_text(jd_text)
    required_skills, preferred_skills = _split_required_preferred(jd_text, all_skills)
    keywords = _extract_keywords(jd_text)
    experience_req = _extract_experience_requirement(jd_text)
    education_req = _extract_education_requirement(jd_text)

    return {
        "title": title,
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "all_skills": all_skills,
        "keywords": keywords,
        "experience_requirement": experience_req,
        "education_requirement": education_req,
        "raw_text": jd_text,
    }


def _extract_job_title(text: str) -> str:
    # Common title patterns at the start of JD
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    title_keywords = [
        "engineer", "developer", "scientist", "analyst", "manager", "designer",
        "architect", "consultant", "specialist", "lead", "intern", "associate",
        "director", "head", "officer", "coordinator", "administrator"
    ]
    for line in lines[:8]:
        lower = line.lower()
        if any(kw in lower for kw in title_keywords) and len(line) < 80:
            return line
    return lines[0] if lines else "Not specified"


def _split_required_preferred(
    jd_text: str, all_skills: list[str]
) -> tuple[list[str], list[str]]:
    """
    Attempt to assign skills to required vs preferred based on section context.
    Falls back to putting all in required if no markers found.
    """
    lines = jd_text.lower().split("\n")
    skill_set = set(all_skills)
    required: set[str] = set()
    preferred: set[str] = set()
    current_bucket: str = "required"

    for line in lines:
        if any(m in line for m in _REQUIRED_MARKERS):
            current_bucket = "required"
        elif any(m in line for m in _PREFERRED_MARKERS):
            current_bucket = "preferred"
        for skill in skill_set:
            if re.search(re.escape(skill), line):
                if current_bucket == "preferred":
                    preferred.add(skill)
                else:
                    required.add(skill)

    # Skills found only in required stay required
    only_required = required - preferred
    return sorted(only_required or skill_set), sorted(preferred)


def _extract_keywords(text: str) -> list[str]:
    """Extract top TF-IDF keywords from the job description."""
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english", max_features=30, ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in ranked[:20]]
    except Exception:
        return []


def _extract_experience_requirement(text: str) -> str:
    match = re.search(r"(\d+[\+\-]?\s*(?:to\s*\d+)?\s*years?[\s\w]*experience)", text, re.IGNORECASE)
    return match.group(0).strip() if match else "Not specified"


def _extract_education_requirement(text: str) -> str:
    for degree in ["Ph.D", "PhD", "Master", "Bachelor", "B.S", "M.S", "B.E", "M.E", "B.Tech", "M.Tech"]:
        if degree.lower() in text.lower():
            # Find surrounding context
            idx = text.lower().find(degree.lower())
            snippet = text[max(0, idx - 10): idx + 60].strip()
            return snippet
    return "Not specified"


# ---------------------------------------------------------------------------
# SEMANTIC SIMILARITY
# ---------------------------------------------------------------------------

def compute_semantic_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts using sentence-transformers."""
    embedder = _get_embedder()
    if embedder is not None:
        try:
            emb = embedder.encode([text1, text2], convert_to_numpy=True)
            sim = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
            return round(max(0.0, min(1.0, sim)), 4)
        except Exception:
            pass
    # Fallback: TF-IDF cosine similarity
    return _tfidf_similarity(text1, text2)


def _tfidf_similarity(text1: str, text2: str) -> float:
    try:
        vec = TfidfVectorizer(stop_words="english")
        tfidf = vec.fit_transform([text1, text2])
        sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        return round(max(0.0, min(1.0, sim)), 4)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

SCORE_WEIGHTS = {
    "skill_match": 0.40,
    "semantic_similarity": 0.30,
    "keyword_coverage": 0.20,
    "resume_quality": 0.10,
}


def compute_skill_match(resume_skills: list[str], job_skills: list[str]) -> float:
    if not job_skills:
        return 0.0
    resume_set = set(s.lower() for s in resume_skills)
    job_set = set(s.lower() for s in job_skills)
    matched = resume_set & job_set
    return round(len(matched) / len(job_set), 4)


def compute_keyword_coverage(resume_text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    resume_lower = resume_text.lower()
    covered = sum(1 for kw in keywords if kw.lower() in resume_lower)
    return round(covered / len(keywords), 4)


def compute_resume_quality_score(text: str, resume_info: dict) -> dict[str, Any]:
    """
    Analyze resume quality and return a score + breakdown.
    Score is purely heuristic — NOT an ATS score.
    """
    checks: dict[str, bool] = {}

    # Contact info
    checks["has_email"] = resume_info.get("email", "Not detected") != "Not detected"
    checks["has_phone"] = resume_info.get("phone", "Not detected") != "Not detected"
    checks["has_linkedin"] = resume_info.get("linkedin", "Not detected") != "Not detected"

    # Sections
    for section in ["education", "experience", "skills", "projects"]:
        val = resume_info.get(section, "Not detected")
        checks[f"has_{section}"] = val not in ("Not detected", "", [])

    # Skills count
    skills = resume_info.get("skills", [])
    checks["enough_skills"] = len(skills) >= 5

    # Action verbs usage
    text_lower = text.lower()
    verbs_found = sum(1 for v in _ACTION_VERBS if re.search(r"\b" + v + r"\b", text_lower))
    checks["uses_action_verbs"] = verbs_found >= 3

    # Quantifiable achievements (numbers in context of results)
    quant_pattern = r"\d+\s*%|\d+x|\$\d+|\d+\s*(users|customers|projects|models|apps|products)"
    checks["has_quantifiable_results"] = bool(re.search(quant_pattern, text, re.IGNORECASE))

    # Length sanity (200–800 lines considered normal)
    word_count = len(text.split())
    checks["reasonable_length"] = 100 <= word_count <= 2000

    # Repeated words (bad sign)
    words = re.findall(r"\b[a-z]{4,}\b", text_lower)
    from collections import Counter
    word_freq = Counter(words)
    most_common_count = word_freq.most_common(1)[0][1] if word_freq else 0
    checks["no_word_overuse"] = most_common_count < 15

    total = len(checks)
    passed = sum(checks.values())
    score = round(passed / total, 4)

    strengths = [k for k, v in checks.items() if v]
    improvements = [k for k, v in checks.items() if not v]

    return {
        "score": score,
        "checks": checks,
        "strengths": _humanize_checks(strengths),
        "improvements": _humanize_checks(improvements),
        "action_verbs_count": verbs_found,
        "word_count": word_count,
    }


_CHECK_LABELS = {
    "has_email": "Contact email present",
    "has_phone": "Contact phone present",
    "has_linkedin": "LinkedIn profile linked",
    "has_education": "Education section present",
    "has_experience": "Experience section present",
    "has_skills": "Skills section present",
    "has_projects": "Projects section present",
    "enough_skills": "5 or more skills listed",
    "uses_action_verbs": "Uses strong action verbs",
    "has_quantifiable_results": "Includes quantifiable results",
    "reasonable_length": "Resume has appropriate length",
    "no_word_overuse": "No significant word overuse",
}


def _humanize_checks(keys: list[str]) -> list[str]:
    return [_CHECK_LABELS.get(k, k.replace("_", " ").title()) for k in keys]


def compute_overall_compatibility(
    skill_match: float,
    semantic_similarity: float,
    keyword_coverage: float,
    resume_quality: float,
    weights: dict | None = None,
) -> float:
    w = weights or SCORE_WEIGHTS
    score = (
        skill_match * w["skill_match"]
        + semantic_similarity * w["semantic_similarity"]
        + keyword_coverage * w["keyword_coverage"]
        + resume_quality * w["resume_quality"]
    )
    return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# SKILL GAP ANALYSIS
# ---------------------------------------------------------------------------

def compute_skill_gap(
    resume_skills: list[str],
    required_skills: list[str],
    preferred_skills: list[str],
) -> dict[str, Any]:
    resume_set = set(s.lower() for s in resume_skills)
    req_set = set(s.lower() for s in required_skills)
    pref_set = set(s.lower() for s in preferred_skills)

    matched = sorted(resume_set & (req_set | pref_set))
    missing_required = sorted(req_set - resume_set)
    missing_preferred = sorted(pref_set - resume_set)

    return {
        "matched_skills": matched,
        "missing_required": missing_required,
        "missing_preferred": missing_preferred,
        "match_rate": compute_skill_match(resume_skills, required_skills),
    }


# ---------------------------------------------------------------------------
# RESUME IMPROVEMENT SUGGESTIONS
# ---------------------------------------------------------------------------

_WEAK_VERBS = {
    "worked on", "helped with", "was responsible for", "did", "made",
    "handled", "assisted", "involved in", "participated in", "was part of",
}

_STRONG_VERB_ALTERNATIVES = {
    "worked on": "developed / implemented / engineered",
    "helped with": "contributed to / supported / collaborated on",
    "was responsible for": "led / managed / owned",
    "did": "executed / performed / delivered",
    "made": "created / designed / built",
    "handled": "managed / directed / administered",
    "assisted": "supported / facilitated / enabled",
    "involved in": "contributed to / participated in",
    "participated in": "contributed to / collaborated on",
    "was part of": "contributed to / joined",
}


def generate_improvements(text: str, resume_info: dict) -> dict[str, Any]:
    """Generate actionable improvement suggestions from resume text."""
    suggestions: list[dict] = []
    weak_bullets: list[dict] = []

    lines = text.split("\n")
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 15:
            continue
        line_lower = stripped.lower()
        for wv in _WEAK_VERBS:
            if line_lower.startswith(wv) or f" {wv} " in line_lower:
                weak_bullets.append({
                    "original": stripped,
                    "issue": f"Weak opener: '{wv}'",
                    "suggestion": f"Replace '{wv}' with: {_STRONG_VERB_ALTERNATIVES.get(wv, 'a stronger action verb')}",
                })
                break

    # Long sentences
    long_sentences = [
        s.strip()
        for s in re.split(r"[.!?]", text)
        if len(s.split()) > 35
    ]
    if long_sentences:
        suggestions.append({
            "category": "Clarity",
            "suggestion": f"Shorten {len(long_sentences)} sentence(s) with more than 35 words for better readability.",
            "examples": long_sentences[:2],
        })

    # Missing quantifiable results
    quant_pattern = r"\d+\s*%|\d+x|\$\d+|\d+\s*(users|customers|projects)"
    if not re.search(quant_pattern, text, re.IGNORECASE):
        suggestions.append({
            "category": "Impact",
            "suggestion": "Add measurable results to your bullet points where you have evidence (e.g., reduced processing time by 30%, served 500+ users).",
            "examples": [],
        })

    # Missing sections
    for section in ["education", "projects", "certifications"]:
        val = resume_info.get(section, "Not detected")
        if val in ("Not detected", "", []):
            suggestions.append({
                "category": "Completeness",
                "suggestion": f"Consider adding a '{section.title()}' section to improve completeness.",
                "examples": [],
            })

    # Action verbs
    text_lower = text.lower()
    verbs_found = [v for v in _ACTION_VERBS if re.search(r"\b" + v + r"\b", text_lower)]
    if len(verbs_found) < 3:
        suggestions.append({
            "category": "Language",
            "suggestion": "Use stronger action verbs to start bullet points (e.g., Developed, Optimized, Architected, Deployed).",
            "examples": [],
        })

    return {
        "weak_bullets": weak_bullets[:8],
        "general_suggestions": suggestions,
    }


# ---------------------------------------------------------------------------
# INTERVIEW QUESTION GENERATION
# ---------------------------------------------------------------------------

_BEHAVIORAL_QUESTIONS = [
    "Tell me about yourself and your background.",
    "Describe a challenging problem you solved and how you approached it.",
    "Tell me about a time you worked effectively in a team.",
    "How do you handle tight deadlines or competing priorities?",
    "What is your greatest professional achievement so far?",
]

_HR_QUESTIONS = [
    "Why are you interested in this role?",
    "Where do you see yourself in 5 years?",
    "What are your salary expectations?",
    "Are you comfortable with remote/hybrid work?",
    "What motivates you in your work?",
]

_ML_QUESTION_TEMPLATES = {
    "random forest": [
        "Why did you choose Random Forest for your project?",
        "How does Random Forest handle overfitting compared to a single Decision Tree?",
        "How did you tune the hyperparameters (n_estimators, max_depth) in your Random Forest?",
        "What evaluation metrics did you use and why?",
        "What are the limitations of Random Forest for your use case?",
    ],
    "neural network": [
        "What architecture did you use and why?",
        "How did you handle overfitting (dropout, regularization, early stopping)?",
        "What optimizer and learning rate strategy did you use?",
        "How did you preprocess the data for your neural network?",
    ],
    "nlp": [
        "What NLP techniques did you apply in your project?",
        "How did you handle text preprocessing (tokenization, stop words, lemmatization)?",
        "What embedding strategy did you use (TF-IDF, Word2Vec, BERT)?",
        "How did you evaluate your NLP model?",
    ],
    "machine learning": [
        "Walk me through your machine learning pipeline end-to-end.",
        "How did you perform feature selection?",
        "How did you handle class imbalance in your dataset?",
        "What cross-validation strategy did you use?",
    ],
    "deep learning": [
        "How did you choose the number of layers and neurons in your architecture?",
        "How did you address the vanishing gradient problem?",
        "What data augmentation techniques did you apply?",
    ],
}


def generate_interview_questions(
    resume_info: dict, job_info: dict | None = None
) -> dict[str, list[str]]:
    """
    Generate interview questions based on actual resume content.
    Only generates project/technical questions for skills/projects present.
    """
    skills = [s.lower() for s in resume_info.get("skills", [])]
    projects_text = resume_info.get("projects", "").lower()
    experience_text = resume_info.get("experience", "").lower()

    technical: list[str] = []
    project_based: list[str] = []

    # Project-specific questions
    for keyword, questions in _ML_QUESTION_TEMPLATES.items():
        if keyword in projects_text or keyword in experience_text or keyword in skills:
            project_based.extend(questions[:3])

    # Technical questions based on skills
    if "python" in skills:
        technical.append("Explain the difference between a list and a generator in Python.")
        technical.append("What is the GIL in Python and when does it matter?")
    if "sql" in skills or "mysql" in skills or "postgresql" in skills:
        technical.append("Write a query to find the second highest salary in a table.")
        technical.append("Explain the difference between INNER JOIN and LEFT JOIN.")
    if "docker" in skills:
        technical.append("What is the difference between a Docker image and a container?")
        technical.append("How would you use Docker Compose for a multi-service application?")
    if "aws" in skills or "cloud" in skills:
        technical.append("Describe a cloud architecture you have worked with or designed.")
    if "react" in skills or "javascript" in skills:
        technical.append("Explain the virtual DOM and how React uses it.")
        technical.append("What is the difference between props and state in React?")
    if "fastapi" in skills or "flask" in skills or "django" in skills:
        technical.append("How do you handle authentication in a REST API?")
        technical.append("What is the difference between synchronous and asynchronous request handling?")

    # If job info available, add job-specific technical questions
    if job_info:
        for skill in job_info.get("required_skills", [])[:5]:
            skill_lower = skill.lower()
            if skill_lower not in skills:
                technical.append(f"Our role requires {skill} — how would you get up to speed with it?")

    # Deduplicate
    technical = list(dict.fromkeys(technical))[:8]
    project_based = list(dict.fromkeys(project_based))[:8]

    return {
        "technical": technical,
        "project_based": project_based,
        "behavioral": _BEHAVIORAL_QUESTIONS,
        "hr": _HR_QUESTIONS,
    }


# ---------------------------------------------------------------------------
# OPTIONAL LLM ENHANCEMENT
# ---------------------------------------------------------------------------

def enhance_with_llm(prompt: str, api_key: str | None = None) -> str | None:
    """
    Call an LLM API if a key is available.
    Currently supports OpenAI-compatible APIs.
    Returns None gracefully if unavailable.
    """
    key = api_key or os.getenv("LLM_API_KEY", "")
    if not key:
        return None
    try:
        import httpx
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.7,
        }
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return None
