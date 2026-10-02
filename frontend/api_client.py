"""
HTTP client for the ResumeAI FastAPI backend.
All Streamlit pages must use this module — no direct API calls in UI code.
"""
from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
TIMEOUT = 120  # seconds — embedding models can be slow on first load


class APIError(Exception):
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


def _handle_response(response: requests.Response) -> dict:
    if response.status_code == 200:
        return response.json()
    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text
    raise APIError(str(detail), response.status_code)


def health_check() -> dict:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return _handle_response(r)
    except requests.exceptions.ConnectionError:
        raise APIError("Cannot connect to the backend. Is it running?")
    except requests.exceptions.Timeout:
        raise APIError("Backend health check timed out.")


def analyze_resume(pdf_bytes: bytes, filename: str) -> dict:
    """Upload a PDF and get extracted resume information."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/analyze-resume",
            files={"file": (filename, pdf_bytes, "application/pdf")},
            timeout=TIMEOUT,
        )
        return _handle_response(r)
    except requests.exceptions.ConnectionError:
        raise APIError("Backend is not reachable. Please start the backend server.")
    except requests.exceptions.Timeout:
        raise APIError("Resume analysis timed out. Try a smaller PDF.")


def analyze_job(job_description: str) -> dict:
    """Analyze a job description."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/analyze-job",
            json={"job_description": job_description},
            timeout=TIMEOUT,
        )
        return _handle_response(r)
    except requests.exceptions.ConnectionError:
        raise APIError("Backend is not reachable.")
    except requests.exceptions.Timeout:
        raise APIError("Job analysis timed out.")


def match_resume(resume_text: str, job_description: str, filename: str = "resume.pdf") -> dict:
    """Run full resume ↔ job matching."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/match-resume",
            json={
                "resume_text": resume_text,
                "job_description": job_description,
                "resume_filename": filename,
            },
            timeout=TIMEOUT,
        )
        return _handle_response(r)
    except requests.exceptions.ConnectionError:
        raise APIError("Backend is not reachable.")
    except requests.exceptions.Timeout:
        raise APIError("Matching timed out.")


def improve_resume(resume_text: str) -> dict:
    """Get improvement suggestions for a resume."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/improve-resume",
            json={"resume_text": resume_text},
            timeout=TIMEOUT,
        )
        return _handle_response(r)
    except requests.exceptions.ConnectionError:
        raise APIError("Backend is not reachable.")
    except requests.exceptions.Timeout:
        raise APIError("Improvement generation timed out.")


def generate_interview(resume_text: str, job_description: str | None = None) -> dict:
    """Generate interview questions."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/generate-interview",
            json={"resume_text": resume_text, "job_description": job_description},
            timeout=TIMEOUT,
        )
        return _handle_response(r)
    except requests.exceptions.ConnectionError:
        raise APIError("Backend is not reachable.")
    except requests.exceptions.Timeout:
        raise APIError("Interview generation timed out.")


def get_history(limit: int = 50) -> dict:
    """Fetch analysis history."""
    try:
        r = requests.get(f"{BACKEND_URL}/history?limit={limit}", timeout=10)
        return _handle_response(r)
    except requests.exceptions.ConnectionError:
        raise APIError("Backend is not reachable.")
    except requests.exceptions.Timeout:
        raise APIError("History fetch timed out.")
