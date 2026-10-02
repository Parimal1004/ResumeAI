"""
FastAPI backend — ResumeAI
Endpoints:
  POST /analyze-resume
  POST /analyze-job
  POST /match-resume
  POST /improve-resume
  POST /generate-interview
  GET  /history
  GET  /health
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

load_dotenv()

from backend.database import get_db, init_db
from backend.models import AnalysisHistory
from backend import services

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app):
    init_db()
    yield


app = FastAPI(
    title="ResumeAI",
    description="AI-Powered Resume Analyzer & Job Matcher",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class JobRequest(BaseModel):
    job_description: str


class MatchRequest(BaseModel):
    resume_text: str
    job_description: str
    resume_filename: str = "resume.pdf"


class ImproveRequest(BaseModel):
    resume_text: str


class InterviewRequest(BaseModel):
    resume_text: str
    job_description: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_pdf_upload(file: UploadFile, content: bytes):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds {MAX_FILE_SIZE_MB} MB limit.",
        )
    if len(content) < 100:
        raise HTTPException(status_code=400, detail="File appears to be empty or corrupt.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "ResumeAI", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/analyze-resume")
async def analyze_resume(file: UploadFile = File(...)):
    """Extract text and structured information from a PDF resume."""
    content = await file.read()
    _validate_pdf_upload(file, content)

    try:
        text = services.extract_text_from_pdf(content)
        pdf_links = services.extract_links_from_pdf(content)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not extract PDF content: {str(e)}"
        )

    if not text or len(text.strip()) < 50:
        raise HTTPException(
            status_code=422,
            detail="No readable text found in the PDF. Please ensure the PDF is not scanned/image-only.",
        )

    resume_info = services.extract_resume_info(text, pdf_links)
    quality = services.compute_resume_quality_score(text, resume_info)

    return {
        "filename": file.filename,
        "text": text,
        "resume_info": resume_info,
        "quality": quality,
    }

@app.post("/analyze-job")
async def analyze_job(request: JobRequest):
    """Extract structured information from a job description."""
    jd = request.job_description.strip()
    if len(jd) < 30:
        raise HTTPException(status_code=400, detail="Job description is too short.")
    job_info = services.analyze_job_description(jd)
    return job_info


@app.post("/match-resume")
async def match_resume(request: MatchRequest, db: Session = Depends(get_db)):
    """Full resume ↔ job matching with compatibility scoring."""
    if len(request.resume_text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Resume text is too short.")
    if len(request.job_description.strip()) < 30:
        raise HTTPException(status_code=400, detail="Job description is too short.")

    resume_info = services.extract_resume_info(request.resume_text)
    job_info = services.analyze_job_description(request.job_description)
    quality = services.compute_resume_quality_score(request.resume_text, resume_info)

    skill_match = services.compute_skill_match(
        resume_info["skills"], job_info["all_skills"]
    )
    semantic_sim = services.compute_semantic_similarity(
        request.resume_text, request.job_description
    )
    keyword_cov = services.compute_keyword_coverage(
        request.resume_text, job_info["keywords"]
    )
    overall = services.compute_overall_compatibility(
        skill_match, semantic_sim, keyword_cov, quality["score"]
    )

    skill_gap = services.compute_skill_gap(
        resume_info["skills"],
        job_info["required_skills"],
        job_info["preferred_skills"],
    )

    # Persist to history
    try:
        record = AnalysisHistory(
            resume_filename=request.resume_filename,
            job_title=job_info.get("title", "Unknown"),
            compatibility_score=round(overall * 100, 1),
            skill_match_score=round(skill_match * 100, 1),
            semantic_similarity=round(semantic_sim * 100, 1),
            keyword_coverage=round(keyword_cov * 100, 1),
            resume_quality_score=round(quality["score"] * 100, 1),
            matched_skills=json.dumps(skill_gap["matched_skills"]),
            missing_skills=json.dumps(skill_gap["missing_required"]),
            resume_summary=resume_info.get("name", ""),
        )
        db.add(record)
        db.commit()
    except Exception:
        pass  # History storage failure should not break the analysis

    return {
        "scores": {
            "overall_compatibility": round(overall * 100, 1),
            "skill_match": round(skill_match * 100, 1),
            "semantic_similarity": round(semantic_sim * 100, 1),
            "keyword_coverage": round(keyword_cov * 100, 1),
            "resume_quality": round(quality["score"] * 100, 1),
        },
        "weights": services.SCORE_WEIGHTS,
        "skill_gap": skill_gap,
        "resume_info": resume_info,
        "job_info": job_info,
        "quality": quality,
    }


@app.post("/improve-resume")
async def improve_resume(request: ImproveRequest):
    """Generate resume improvement suggestions."""
    if len(request.resume_text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Resume text is too short.")

    resume_info = services.extract_resume_info(request.resume_text)
    improvements = services.generate_improvements(request.resume_text, resume_info)

    # Optional LLM enhancement
    llm_suggestions = None
    if os.getenv("LLM_API_KEY"):
        bullet_count = len(improvements.get("weak_bullets", []))
        if bullet_count:
            prompt = (
                "You are a professional resume coach. "
                "Rewrite the following weak resume bullet points to be stronger. "
                "Only use information provided — do NOT invent statistics or achievements.\n\n"
                + "\n".join(b["original"] for b in improvements["weak_bullets"][:3])
            )
            llm_suggestions = services.enhance_with_llm(prompt)

    return {
        "improvements": improvements,
        "llm_enhanced": llm_suggestions,
    }


@app.post("/generate-interview")
async def generate_interview(request: InterviewRequest):
    """Generate interview questions based on resume and optional job description."""
    if len(request.resume_text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Resume text is too short.")

    resume_info = services.extract_resume_info(request.resume_text)
    job_info = None
    if request.job_description and len(request.job_description.strip()) >= 30:
        job_info = services.analyze_job_description(request.job_description)

    questions = services.generate_interview_questions(resume_info, job_info)
    return {"questions": questions}


@app.get("/history")
async def get_history(db: Session = Depends(get_db), limit: int = 50):
    """Return the most recent analysis history entries."""
    records = (
        db.query(AnalysisHistory)
        .order_by(AnalysisHistory.created_at.desc())
        .limit(limit)
        .all()
    )
    result = []
    for r in records:
        result.append({
            "id": r.id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "resume_filename": r.resume_filename,
            "job_title": r.job_title,
            "compatibility_score": r.compatibility_score,
            "skill_match_score": r.skill_match_score,
            "semantic_similarity": r.semantic_similarity,
            "keyword_coverage": r.keyword_coverage,
            "resume_quality_score": r.resume_quality_score,
        })
    return {"history": result, "total": len(result)}
