"""
SQLAlchemy ORM models.
"""
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from backend.database import Base


def _utcnow():
    return datetime.now(timezone.utc)


class AnalysisHistory(Base):
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    resume_filename = Column(String(255), nullable=False)
    job_title = Column(String(255), nullable=True)
    compatibility_score = Column(Float, nullable=True)
    skill_match_score = Column(Float, nullable=True)
    semantic_similarity = Column(Float, nullable=True)
    keyword_coverage = Column(Float, nullable=True)
    resume_quality_score = Column(Float, nullable=True)
    matched_skills = Column(Text, nullable=True)   # JSON list stored as text
    missing_skills = Column(Text, nullable=True)   # JSON list stored as text
    resume_summary = Column(Text, nullable=True)
