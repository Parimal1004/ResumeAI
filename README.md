# ResumeAI — AI-Powered Resume Analyzer & Job Matcher

ResumeAI is a full-stack Python application that analyzes your PDF resume against a job description using NLP, semantic embeddings, and machine learning. It gives you a transparent compatibility score, a skill gap breakdown, resume improvement suggestions, and personalized interview questions.

---

## Features

| Feature | Description |
|---|---|
| **Resume Analysis** | Extracts name, contact info, skills, education, experience, projects from a PDF |
| **Job Matching** | Compares resume to a job description using skill matching, TF-IDF keywords, and sentence-transformer embeddings |
| **Skill Gap Analysis** | Shows matched, missing required, and missing preferred skills |
| **Resume Quality Score** | Transparent heuristic score (not a guaranteed ATS score) |
| **Improvement Suggestions** | Identifies weak bullet points and suggests stronger language |
| **Interview Preparation** | Generates technical, project-based, behavioral, and HR questions |
| **Analysis History** | Stores every analysis in SQLite for comparison over time |
| **Optional LLM Integration** | Set `LLM_API_KEY` to get AI-enhanced bullet rewrites |

---

## Project Structure

```
ResumeAI/
│
├── frontend/
│   ├── app.py          ← Streamlit UI (all pages)
│   ├── api_client.py   ← HTTP client for the backend
│   └── style.css       ← Custom CSS
│
├── backend/
│   ├── main.py         ← FastAPI app and all API routes
│   ├── database.py     ← SQLAlchemy engine and session
│   ├── models.py       ← ORM model for analysis history
│   └── services.py     ← All NLP/ML logic
│
├── data/
│   └── skills.json     ← Skills database with aliases
│
├── tests/
│   └── test_app.py     ← pytest test suite
│
├── .env.example
├── requirements.txt
├── run_backend.bat
└── run_frontend.bat
```

---

## Quick Start (Windows)

### 1. Create and activate a virtual environment

```cmd
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```cmd
pip install -r requirements.txt
```

### 3. Download the spaCy language model

```cmd
python -m spacy download en_core_web_sm
```

### 4. Configure environment variables

```cmd
copy .env.example .env
```

Open `.env` and fill in any optional values:

```
BACKEND_URL=http://127.0.0.1:8000
DATABASE_URL=sqlite:///./resumeai.db
LLM_API_KEY=          ← optional: OpenAI-compatible API key for enhanced suggestions
```

### 5. Start the backend

Double-click `run_backend.bat` or run:

```cmd
uvicorn backend.main:app --reload --port 8000
```

The backend starts at `http://127.0.0.1:8000`.  
API documentation is available at `http://127.0.0.1:8000/docs`.

### 6. Start the frontend (new terminal)

Double-click `run_frontend.bat` or run:

```cmd
streamlit run frontend/app.py
```

### 7. Open the application

Streamlit will open your browser automatically at `http://localhost:8501`.

---

## How to Use

1. Go to **Resume Analysis** in the sidebar.
2. Upload your PDF resume and click **Analyze Resume**.
3. Go to **Job Matching**, paste a job description, and click **Match Resume to Job**.
4. Browse **Skill Gap**, **Resume Improvement**, and **Interview Preparation** pages.
5. View **History** for a log of all previous analyses.

---

## Running Tests

```cmd
pytest tests/test_app.py -v
```

---

## Compatibility Score Formula

```
Overall Compatibility =
  Skill Match           × 40%
  Semantic Similarity   × 30%
  Keyword Coverage      × 20%
  Resume Quality Score  × 10%
```

This is a **Resume Quality Score** computed from NLP heuristics and semantic embeddings.  
It is **not** a guaranteed ATS pass/fail score.

---

## Optional LLM Integration

Set `LLM_API_KEY` in `.env` with an OpenAI-compatible API key.  
The application will use it to suggest AI-rewritten bullet points.  
The core analysis works fully without an API key.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit, Plotly |
| Backend | FastAPI, Uvicorn, Pydantic |
| NLP | spaCy, sentence-transformers, scikit-learn |
| PDF | PyMuPDF (fitz) |
| Database | SQLite, SQLAlchemy |
| Testing | pytest |

---

## Notes

- Raw resume text is not permanently stored in the database — only scores and metadata.
- Never commit your `.env` file. It is excluded by `.gitignore`.
- The sentence-transformer model (`all-MiniLM-L6-v2`) is downloaded automatically on first run (~80 MB).
