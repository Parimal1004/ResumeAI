"""
Tests for ResumeAI backend services and API.
Run with: pytest tests/test_app.py -v
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RESUME_TEXT = """
John Doe
john.doe@email.com | +1-555-123-4567
linkedin.com/in/johndoe | github.com/johndoe

Summary
Experienced data scientist with 3 years of industry experience in machine learning and NLP.

Education
B.Tech in Computer Science — State University, 2021

Experience
Data Scientist — TechCorp (2021–Present)
- Developed machine learning models using Python and Scikit-learn to predict customer churn.
- Implemented NLP pipelines for sentiment analysis with spaCy and NLTK.
- Deployed models on AWS using Docker and Kubernetes.

Projects
Customer Churn Prediction
- Built a Random Forest classifier achieving high accuracy on a dataset of 50,000 records.
- Used pandas for data wrangling and matplotlib for visualization.

Skills
Python, Machine Learning, Deep Learning, NLP, Scikit-learn, TensorFlow, PyTorch,
pandas, NumPy, SQL, Docker, AWS, Git, Linux
"""

SAMPLE_JD_TEXT = """
Senior Data Scientist

We are looking for a skilled Data Scientist to join our team.

Requirements:
- 3+ years of experience in machine learning
- Proficient in Python, Scikit-learn, and TensorFlow
- Experience with SQL and cloud platforms (AWS or GCP)
- Strong knowledge of NLP and deep learning

Preferred:
- Experience with Docker and Kubernetes
- Familiarity with MLflow or similar MLOps tools
- PyTorch experience
"""


# ---------------------------------------------------------------------------
# Services unit tests (no HTTP, no PDF binary needed)
# ---------------------------------------------------------------------------

class TestSkillExtraction:
    def test_detects_python(self):
        from backend.services import extract_skills_from_text
        skills = extract_skills_from_text("I am proficient in Python and Java.")
        assert "python" in skills

    def test_detects_multiple_skills(self):
        from backend.services import extract_skills_from_text
        skills = extract_skills_from_text(SAMPLE_RESUME_TEXT)
        assert len(skills) >= 5

    def test_no_false_positives_on_empty(self):
        from backend.services import extract_skills_from_text
        skills = extract_skills_from_text("")
        assert skills == []

    def test_alias_resolution(self):
        from backend.services import extract_skills_from_text
        skills = extract_skills_from_text("Experienced in ML and NLP techniques.")
        # aliases should resolve to canonical names
        assert len(skills) >= 1


class TestResumeInfo:
    def test_extract_email(self):
        from backend.services import extract_resume_info
        info = extract_resume_info(SAMPLE_RESUME_TEXT)
        assert info["email"] == "john.doe@email.com"

    def test_extract_phone(self):
        from backend.services import extract_resume_info
        info = extract_resume_info(SAMPLE_RESUME_TEXT)
        assert info["phone"] != "Not detected"

    def test_extract_linkedin(self):
        from backend.services import extract_resume_info
        info = extract_resume_info(SAMPLE_RESUME_TEXT)
        assert "linkedin.com" in info["linkedin"]

    def test_extract_github(self):
        from backend.services import extract_resume_info
        info = extract_resume_info(SAMPLE_RESUME_TEXT)
        assert "github.com" in info["github"]

    def test_skills_are_list(self):
        from backend.services import extract_resume_info
        info = extract_resume_info(SAMPLE_RESUME_TEXT)
        assert isinstance(info["skills"], list)


class TestJobAnalysis:
    def test_extracts_title(self):
        from backend.services import analyze_job_description
        job = analyze_job_description(SAMPLE_JD_TEXT)
        assert job["title"] != ""

    def test_extracts_skills(self):
        from backend.services import analyze_job_description
        job = analyze_job_description(SAMPLE_JD_TEXT)
        assert len(job["all_skills"]) >= 3

    def test_extracts_keywords(self):
        from backend.services import analyze_job_description
        job = analyze_job_description(SAMPLE_JD_TEXT)
        assert len(job["keywords"]) > 0

    def test_experience_requirement(self):
        from backend.services import analyze_job_description
        job = analyze_job_description(SAMPLE_JD_TEXT)
        assert "3" in job["experience_requirement"] or job["experience_requirement"] == "Not specified"


class TestScoring:
    def test_skill_match_perfect(self):
        from backend.services import compute_skill_match
        score = compute_skill_match(["python", "sql"], ["python", "sql"])
        assert score == 1.0

    def test_skill_match_none(self):
        from backend.services import compute_skill_match
        score = compute_skill_match(["python"], ["java", "golang"])
        assert score == 0.0

    def test_skill_match_partial(self):
        from backend.services import compute_skill_match
        score = compute_skill_match(["python", "java"], ["python", "java", "go"])
        assert 0.0 < score < 1.0

    def test_keyword_coverage(self):
        from backend.services import compute_keyword_coverage
        score = compute_keyword_coverage("python machine learning data", ["python", "machine learning"])
        assert score > 0.0

    def test_overall_compatibility_bounds(self):
        from backend.services import compute_overall_compatibility
        score = compute_overall_compatibility(0.8, 0.7, 0.6, 0.9)
        assert 0.0 <= score <= 1.0

    def test_overall_compatibility_zeros(self):
        from backend.services import compute_overall_compatibility
        score = compute_overall_compatibility(0, 0, 0, 0)
        assert score == 0.0


class TestSemanticSimilarity:
    def test_identical_texts_high_similarity(self):
        from backend.services import compute_semantic_similarity
        text = "Python machine learning data science"
        score = compute_semantic_similarity(text, text)
        assert score > 0.9

    def test_unrelated_texts_lower_similarity(self):
        from backend.services import compute_semantic_similarity
        score = compute_semantic_similarity(
            "Python machine learning neural network",
            "cooking recipes baking bread flour"
        )
        # Should be meaningfully lower than identical text similarity
        assert score < 0.9

    def test_returns_float(self):
        from backend.services import compute_semantic_similarity
        score = compute_semantic_similarity("hello", "world")
        assert isinstance(score, float)


class TestQualityScore:
    def test_score_in_range(self):
        from backend.services import compute_resume_quality_score, extract_resume_info
        info = extract_resume_info(SAMPLE_RESUME_TEXT)
        quality = compute_resume_quality_score(SAMPLE_RESUME_TEXT, info)
        assert 0.0 <= quality["score"] <= 1.0

    def test_has_required_keys(self):
        from backend.services import compute_resume_quality_score, extract_resume_info
        info = extract_resume_info(SAMPLE_RESUME_TEXT)
        quality = compute_resume_quality_score(SAMPLE_RESUME_TEXT, info)
        assert "score" in quality
        assert "strengths" in quality
        assert "improvements" in quality
        assert "checks" in quality


class TestSkillGap:
    def test_skill_gap_structure(self):
        from backend.services import compute_skill_gap
        result = compute_skill_gap(
            ["python", "sql"],
            ["python", "sql", "docker"],
            ["aws", "kubernetes"]
        )
        assert "matched_skills" in result
        assert "missing_required" in result
        assert "missing_preferred" in result
        assert "docker" in result["missing_required"]
        assert "python" in result["matched_skills"]


class TestImprovements:
    def test_improvement_structure(self):
        from backend.services import generate_improvements, extract_resume_info
        info = extract_resume_info(SAMPLE_RESUME_TEXT)
        result = generate_improvements(SAMPLE_RESUME_TEXT, info)
        assert "weak_bullets" in result
        assert "general_suggestions" in result

    def test_weak_verb_detection(self):
        from backend.services import generate_improvements, extract_resume_info
        text = "Worked on a machine learning project at company XYZ for two years."
        info = extract_resume_info(text)
        result = generate_improvements(text, info)
        assert len(result["weak_bullets"]) >= 1


class TestInterviewQuestions:
    def test_generates_questions(self):
        from backend.services import generate_interview_questions, extract_resume_info
        info = extract_resume_info(SAMPLE_RESUME_TEXT)
        questions = generate_interview_questions(info)
        assert "technical" in questions
        assert "behavioral" in questions
        assert "hr" in questions

    def test_behavioral_questions_always_present(self):
        from backend.services import generate_interview_questions
        questions = generate_interview_questions({"skills": [], "projects": "", "experience": ""})
        assert len(questions["behavioral"]) > 0
        assert len(questions["hr"]) > 0


# ---------------------------------------------------------------------------
# FastAPI integration tests
# ---------------------------------------------------------------------------

class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.database import init_db
        from backend.main import app
        init_db()  # ensure tables exist before each API test
        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_analyze_job_valid(self, client):
        response = client.post(
            "/analyze-job",
            json={"job_description": SAMPLE_JD_TEXT}
        )
        assert response.status_code == 200
        data = response.json()
        assert "all_skills" in data
        assert "keywords" in data

    def test_analyze_job_too_short(self, client):
        response = client.post(
            "/analyze-job",
            json={"job_description": "Hi"}
        )
        assert response.status_code == 400

    def test_match_resume_valid(self, client):
        response = client.post(
            "/match-resume",
            json={
                "resume_text": SAMPLE_RESUME_TEXT,
                "job_description": SAMPLE_JD_TEXT,
                "resume_filename": "test_resume.pdf",
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "scores" in data
        assert "overall_compatibility" in data["scores"]
        score = data["scores"]["overall_compatibility"]
        assert 0 <= score <= 100

    def test_improve_resume_valid(self, client):
        response = client.post(
            "/improve-resume",
            json={"resume_text": SAMPLE_RESUME_TEXT}
        )
        assert response.status_code == 200
        data = response.json()
        assert "improvements" in data

    def test_generate_interview_valid(self, client):
        response = client.post(
            "/generate-interview",
            json={"resume_text": SAMPLE_RESUME_TEXT, "job_description": SAMPLE_JD_TEXT}
        )
        assert response.status_code == 200
        data = response.json()
        assert "questions" in data

    def test_history_endpoint(self, client):
        response = client.get("/history")
        assert response.status_code == 200
        data = response.json()
        assert "history" in data

    def test_analyze_resume_invalid_file(self, client):
        """Non-PDF upload should return 400."""
        response = client.post(
            "/analyze-resume",
            files={"file": ("resume.txt", b"this is not a pdf", "text/plain")}
        )
        assert response.status_code == 400

    def test_analyze_resume_empty_file(self, client):
        """Empty PDF payload should be rejected."""
        response = client.post(
            "/analyze-resume",
            files={"file": ("empty.pdf", b"", "application/pdf")}
        )
        assert response.status_code == 400
