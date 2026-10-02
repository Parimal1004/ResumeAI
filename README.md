# ResumeAI

### AI-Powered Resume Analyzer & Job Matcher

ResumeAI is a web application that analyzes resumes and compares them with job descriptions to help users understand their job fit, identify skill gaps, and improve their resumes.

## Live Demo

**Frontend:** https://resumeai-analyzer-parimal.streamlit.app/

**Backend API:** https://resumeai-giu7.onrender.com/

## Features

* PDF resume analysis and information extraction
* Skill and keyword extraction
* Job description analysis
* Resume–job compatibility scoring
* Skill gap analysis
* Resume improvement suggestions
* AI-generated interview questions
* Analysis history and dashboard
* Optional LLM enhancement

## Tech Stack

**Frontend**

* Streamlit
* Plotly
* Custom CSS

**Backend**

* FastAPI
* Python
* SQLAlchemy
* SQLite

**NLP / ML**

* spaCy
* scikit-learn
* Sentence Transformers
* PyMuPDF
* Pandas / NumPy

## How Matching Works

The compatibility score combines:

| Component           | Weight |
| ------------------- | -----: |
| Skill Match         |    40% |
| Semantic Similarity |    30% |
| Keyword Coverage    |    20% |
| Resume Quality      |    10% |

The score is a project-generated compatibility measure and is not an official ATS score.

## Project Structure

```text
ResumeAI/
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   └── style.css
├── backend/
│   ├── main.py
│   ├── database.py
│   ├── models.py
│   └── services.py
├── data/
│   └── skills.json
├── tests/
│   └── test_app.py
├── requirements.txt
└── README.md
```

## Run Locally

```bash
git clone https://github.com/Parimal1004/ResumeAI.git
cd ResumeAI
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Start the backend:

```bash
.\run_backend.bat
```

Start the frontend in another terminal:

```bash
.\run_frontend.bat
```

Frontend: `http://localhost:8501`
Backend: `http://localhost:8000`


## Author

**Parimal Goud**
AI & Data Science Undergraduate, CBIT

GitHub: https://github.com/Parimal1004
