from __future__ import annotations

import os
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ResumeAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Resolve imports (works both as `streamlit run frontend/app.py` from root
# and as `streamlit run app.py` from inside frontend/)
# ---------------------------------------------------------------------------
import sys
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    import api_client
    from api_client import APIError
except ImportError:
    from frontend import api_client
    from frontend.api_client import APIError

load_dotenv()

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_CSS_PATH = Path(__file__).parent / "style.css"
if _CSS_PATH.exists():
    with open(_CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    "resume_data": None,
    "match_data": None,
    "improve_data": None,
    "interview_data": None,
    "job_description": "",
    "resume_text": "",
    "resume_filename": "",
    "backend_ok": None,
    "active_page": "Home",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_class(s: float) -> str:
    if s >= 75: return "excellent"
    if s >= 55: return "good"
    if s >= 35: return "fair"
    return "poor"

def _bar_class(s: float) -> str:
    if s >= 75: return "bar-excellent"
    if s >= 55: return "bar-good"
    if s >= 35: return "bar-fair"
    return "bar-poor"

def _metric_card(label: str, value: str, cls: str = "") -> str:
    return f"""<div class="metric-card {cls}">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>"""

def _score_bar(label: str, value: float) -> str:
    bc = _bar_class(value)
    return f"""<div class="score-bar-container">
        <div class="score-bar-label">
            <span>{label}</span>
            <span class="score-bar-value">{value:.1f}%</span>
        </div>
        <div class="score-bar-bg">
            <div class="score-bar-fill {bc}" style="width:{min(value,100)}%"></div>
        </div>
    </div>"""

def _skill_pill(skill: str, kind: str = "neutral") -> str:
    return f'<span class="skill-pill skill-{kind}">{skill}</span>'

def _page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"""<div class="page-header">
        <div class="page-title">{title}</div>
        {"<div class='page-subtitle'>" + subtitle + "</div>" if subtitle else ""}
    </div>""", unsafe_allow_html=True)

def _check_backend() -> None:
    try:
        api_client.health_check()
        st.session_state.backend_ok = True
    except APIError:
        st.session_state.backend_ok = False

# ---------------------------------------------------------------------------
# Plotly theme helpers
# ---------------------------------------------------------------------------
_PLOT_LAYOUT = dict(
    plot_bgcolor="#131929",
    paper_bgcolor="#131929",
    font=dict(family="Inter, -apple-system, sans-serif", color="#94a3b8", size=12),
    margin=dict(t=44, b=20, l=10, r=10),
    showlegend=False,
)

def _fig_bar(x, y, colors, title: str, ytitle: str = "Score (%)", yrange=None) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=x, y=y,
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.1f}%" if isinstance(v, float) else str(v) for v in y],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=12),
    ))
    layout = dict(**_PLOT_LAYOUT)
    layout["title"] = dict(text=title, font=dict(color="#e2e8f0", size=14), x=0, pad=dict(l=0))
    layout["yaxis"] = dict(
        title=ytitle,
        range=yrange or [0, max(y or [1]) * 1.35],
        gridcolor="#1e2535",
        zerolinecolor="#1e2535",
        color="#64748b",
    )
    layout["xaxis"] = dict(color="#64748b", tickfont=dict(color="#94a3b8"))
    fig.update_layout(**layout)
    return fig

def chart_score_breakdown(scores: dict) -> go.Figure:
    labels = ["Skill Match", "Semantic Sim.", "Keyword Cov.", "Resume Quality"]
    values = [
        scores.get("skill_match", 0),
        scores.get("semantic_similarity", 0),
        scores.get("keyword_coverage", 0),
        scores.get("resume_quality", 0),
    ]
    colors = ["#6366f1", "#8b5cf6", "#22c55e", "#f59e0b"]
    return _fig_bar(labels, values, colors, "Score Breakdown", "Score (%)", [0, 110])

def chart_skill_gap(matched, missing_req, missing_pref) -> go.Figure:
    categories = ["Matched", "Missing Required", "Missing Preferred"]
    counts = [len(matched), len(missing_req), len(missing_pref)]
    colors = ["#22c55e", "#ef4444", "#f59e0b"]
    return _fig_bar(categories, counts, colors, "Skill Gap Overview", "Skills", [0, max(counts or [1]) + 2])

def chart_completeness(checks: dict) -> go.Figure:
    human = [k.replace("_", " ").replace("has ", "").title() for k in checks.keys()]
    vals = [1 if v else 0 for v in checks.values()]
    colors = ["#22c55e" if v else "#ef4444" for v in vals]
    fig = go.Figure(go.Bar(
        x=human, y=vals,
        marker_color=colors,
        marker_line_width=0,
        text=["Pass" if v else "Fail" for v in vals],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=11),
    ))
    layout = dict(**_PLOT_LAYOUT)
    layout["title"] = dict(text="Resume Quality Checks", font=dict(color="#e2e8f0", size=14), x=0)
    layout["yaxis"] = dict(range=[0, 1.6], showticklabels=False, gridcolor="#1e2535", zerolinecolor="#1e2535")
    layout["xaxis"] = dict(tickangle=-38, color="#64748b", tickfont=dict(color="#94a3b8", size=10))
    layout["margin"] = dict(t=44, b=90, l=10, r=10)
    fig.update_layout(**layout)
    return fig

def chart_gauge(score: float) -> go.Figure:
    color = (
        "#22c55e" if score >= 75 else
        "#6366f1" if score >= 55 else
        "#f59e0b" if score >= 35 else
        "#ef4444"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Overall Compatibility", "font": {"size": 13, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#475569"}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#1e2535",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35],  "color": "rgba(239,68,68,0.08)"},
                {"range": [35, 55], "color": "rgba(245,158,11,0.08)"},
                {"range": [55, 75], "color": "rgba(99,102,241,0.08)"},
                {"range": [75, 100],"color": "rgba(34,197,94,0.08)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": score},
        },
        number={"suffix": "%", "font": {"size": 38, "color": "#f1f5f9"}},
    ))
    fig.update_layout(
        height=260,
        paper_bgcolor="#131929",
        font=dict(family="Inter, sans-serif"),
        margin=dict(t=30, b=10, l=20, r=20),
    )
    return fig

def chart_trend(scores: list) -> go.Figure:
    x = list(range(1, len(scores) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=scores,
        mode="lines+markers",
        line=dict(color="#6366f1", width=2.5),
        marker=dict(size=8, color="#6366f1", line=dict(color="#131929", width=2)),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.06)",
    ))
    layout = dict(**_PLOT_LAYOUT)
    layout["title"] = dict(text="Compatibility Score Trend", font=dict(color="#e2e8f0", size=14), x=0)
    layout["yaxis"] = dict(range=[0, 105], gridcolor="#1e2535", zerolinecolor="#1e2535", color="#64748b", title="Score (%)")
    layout["xaxis"] = dict(color="#64748b", title="Analysis #")
    fig.update_layout(**layout)
    return fig

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> str:
    with st.sidebar:
        # Brand
        st.markdown("""<div class="sidebar-brand">
            <div class="sidebar-brand-title">ResumeAI</div>
            <div class="sidebar-brand-sub">AI Resume Analyzer</div>
        </div>""", unsafe_allow_html=True)

        # Backend status
        if st.session_state.backend_ok is None:
            _check_backend()

        if st.session_state.backend_ok:
            st.markdown(
                '<div class="sidebar-status-row">'
                '<span class="sidebar-status-dot dot-ok"></span>'
                '<span class="sidebar-status-text">Backend connected</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="sidebar-status-row">'
                '<span class="sidebar-status-dot dot-err"></span>'
                '<span class="sidebar-status-text">Backend offline — start run_backend.bat</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Nav items — plain text, no emojis (avoids blank icon rendering issues)
        nav_items = [
            "Home",
            "Resume Analysis",
            "Job Matching",
            "Skill Gap",
            "Resume Improvement",
            "Interview Preparation",
            "History",
        ]

        if "sidebar_nav" not in st.session_state:
            st.session_state.sidebar_nav = st.session_state.get("active_page", "Home")
        if st.session_state.sidebar_nav not in nav_items:
            st.session_state.sidebar_nav = "Home"

        page = st.radio(
            "nav",
            nav_items,
            key="sidebar_nav",
            label_visibility="collapsed",
        )
        st.session_state.active_page = page

        # Loaded resume indicator
        if st.session_state.resume_filename:
            st.markdown(
                f'<div class="sidebar-file">&#128196; {st.session_state.resume_filename}</div>',
                unsafe_allow_html=True,
            )

        return page

# ---------------------------------------------------------------------------
# ── PAGE: HOME (with resume upload) ──
# ---------------------------------------------------------------------------

def page_home() -> None:
    # ── Hero ──
    st.markdown("""
    <div class="hero-section">
        <div class="hero-badge">AI-Powered</div>
        <div class="hero-title">
            Analyze your resume.<br>
            <span>Land your dream job.</span>
        </div>
        <div class="hero-subtitle">
            Upload your PDF resume to extract skills, score against any job
            description, find skill gaps, and get tailored interview
            preparation — all in seconds.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload + Job Description ──
    upload_col, jd_col = st.columns([1, 1], gap="large")

    # ── Step 1: Upload Resume ──
    with upload_col:
        st.markdown(
            """
            <div class="upload-card">
                <div class="upload-card-title">
                    STEP 1 — UPLOAD RESUME
                </div>
                <div class="upload-card-subtitle">
                    Upload your PDF resume to begin analysis
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="color:#64748b;font-size:0.82rem;'
            'margin:0.15rem 0 0.35rem 0;">'
            'PDF only · Maximum file size 5 MB'
            '</div>',
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            label_visibility="collapsed",
            key="home_uploader",
        )

        if uploaded:
            if uploaded.size > 5 * 1024 * 1024:
                st.error("File exceeds 5 MB limit.")
            else:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="info-label">Selected file</div>
                        <div class="info-value">{uploaded.name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Step 2: Job Description ──
    with jd_col:
        st.markdown(
            """
            <div class="upload-card">
                <div class="upload-card-title">
                    STEP 2 — JOB DESCRIPTION (OPTIONAL)
                </div>
                <div class="upload-card-subtitle">
                    Paste the job description for compatibility
                    and skill-gap analysis
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        jd = st.text_area(
            "Job Description",
            value=st.session_state.job_description,
            height=140,
            label_visibility="collapsed",
            placeholder=(
                "Paste the job description here to get a compatibility "
                "score and skill gap analysis..."
            ),
            key="home_jd",
        )

        st.session_state.job_description = jd

    # ── Analyze button ──
    st.markdown(
        "<div style='height:0.7rem'></div>",
        unsafe_allow_html=True,
    )

    analyze_clicked = st.button(
        "Analyze Resume",
        type="primary",
        key="home_analyze",
    )

    if analyze_clicked:
        if not uploaded:
            st.error("Please upload a PDF resume first.")
        elif uploaded.size > 5 * 1024 * 1024:
            st.error("File exceeds 5 MB limit.")
        else:
            _run_analysis(
                uploaded,
                jd if len(jd.strip()) >= 30 else None,
            )

    # ── Results ──
    match_data = st.session_state.match_data
    resume_data = st.session_state.resume_data

    if match_data:
        _render_dashboard_results(match_data)

    elif resume_data:
        _render_resume_only_results(resume_data)

    else:
        _render_feature_grid()


def _run_analysis(uploaded, jd_text: str | None) -> None:
    """Run resume analysis and optionally job matching."""
    with st.spinner("Extracting and analyzing your resume..."):
        try:
            pdf_bytes = uploaded.read()
            result = api_client.analyze_resume(pdf_bytes, uploaded.name)
            st.session_state.resume_data = result
            st.session_state.resume_text = result.get("text", "")
            st.session_state.resume_filename = uploaded.name
            st.session_state.match_data = None
        except APIError as e:
            st.error(f"Resume analysis failed: {e}")
            return

    if jd_text:
        with st.spinner("Running job match and semantic scoring..."):
            try:
                match = api_client.match_resume(
                    st.session_state.resume_text, jd_text, uploaded.name
                )
                st.session_state.match_data = match
            except APIError as e:
                st.warning(f"Job matching failed: {e}. Resume analysis still shown below.")

    st.success("Analysis complete. Results are shown below.")


def _render_dashboard_results(data: dict) -> None:
    scores = data["scores"]
    skill_gap = data.get("skill_gap", {})
    resume_info = data.get("resume_info", {})

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Metrics row ──
    mc = st.columns(5)
    metric_items = [
        ("Compatibility", f"{scores['overall_compatibility']:.1f}%", _score_class(scores["overall_compatibility"])),
        ("Skill Match",   f"{scores['skill_match']:.1f}%",           _score_class(scores["skill_match"])),
        ("Semantic Sim.", f"{scores['semantic_similarity']:.1f}%",   _score_class(scores["semantic_similarity"])),
        ("Keyword Cov.",  f"{scores['keyword_coverage']:.1f}%",      _score_class(scores["keyword_coverage"])),
        ("Resume Quality",f"{scores['resume_quality']:.1f}%",        _score_class(scores["resume_quality"])),
    ]
    for col, (label, value, cls) in zip(mc, metric_items):
        with col:
            st.markdown(_metric_card(label, value, cls), unsafe_allow_html=True)

    # ── Gauge + bars ──
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    g_col, b_col = st.columns([1, 2], gap="large")
    with g_col:
        st.plotly_chart(chart_gauge(scores["overall_compatibility"]), use_container_width=True)
        st.markdown('<div style="text-align:center;font-size:0.7rem;color:#475569;margin-top:-0.5rem">Heuristic score — not a guaranteed ATS score</div>', unsafe_allow_html=True)
    with b_col:
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        bars = "".join([
            _score_bar("Skill Match (40%)", scores["skill_match"]),
            _score_bar("Semantic Similarity (30%)", scores["semantic_similarity"]),
            _score_bar("Keyword Coverage (20%)", scores["keyword_coverage"]),
            _score_bar("Resume Quality (10%)", scores["resume_quality"]),
        ])
        st.markdown(bars, unsafe_allow_html=True)
        with st.expander("How is the score calculated?"):
            w = data.get("weights", {})
            st.markdown(
                f"**Compatibility** = Skill Match × {w.get('skill_match',0.4)*100:.0f}% + "
                f"Semantic Similarity × {w.get('semantic_similarity',0.3)*100:.0f}% + "
                f"Keyword Coverage × {w.get('keyword_coverage',0.2)*100:.0f}% + "
                f"Resume Quality × {w.get('resume_quality',0.1)*100:.0f}%"
            )

    # ── Charts row ──
    ch1, ch2 = st.columns(2, gap="large")
    with ch1:
        st.plotly_chart(chart_score_breakdown(scores), use_container_width=True)
    with ch2:
        st.plotly_chart(
            chart_skill_gap(
                skill_gap.get("matched_skills", []),
                skill_gap.get("missing_required", []),
                skill_gap.get("missing_preferred", []),
            ),
            use_container_width=True,
        )

    # ── Quick skill summary ──
    matched = skill_gap.get("matched_skills", [])
    missing = skill_gap.get("missing_required", [])
    if matched or missing:
        st.markdown('<div class="section-heading">Quick Skill Summary</div>', unsafe_allow_html=True)
        sk1, sk2 = st.columns(2)
        with sk1:
            st.markdown("**Matched**")
            st.markdown("".join(_skill_pill(s, "matched") for s in matched[:12]) or "<i style='color:#475569'>None</i>", unsafe_allow_html=True)
        with sk2:
            st.markdown("**Missing Required**")
            st.markdown("".join(_skill_pill(s, "missing") for s in missing[:12]) or "<i style='color:#475569'>None</i>", unsafe_allow_html=True)

    # ── Detected resume info ──
    _render_resume_identity(resume_info, data.get("quality", {}))


def _render_resume_only_results(data: dict) -> None:
    info = data.get("resume_info", {})
    quality = data.get("quality", {})
    score_pct = round(quality.get("score", 0) * 100, 1)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    mc = st.columns(4)
    with mc[0]:
        st.markdown(_metric_card("Resume Quality", f"{score_pct}%", _score_class(score_pct)), unsafe_allow_html=True)
    with mc[1]:
        skill_count = len(info.get("skills", []))
        st.markdown(_metric_card("Skills Detected", str(skill_count), "good" if skill_count >= 5 else "fair"), unsafe_allow_html=True)
    with mc[2]:
        word_count = quality.get("word_count", 0)
        st.markdown(_metric_card("Word Count", str(word_count), "good"), unsafe_allow_html=True)
    with mc[3]:
        verb_count = quality.get("action_verbs_count", 0)
        st.markdown(_metric_card("Action Verbs", str(verb_count), "excellent" if verb_count >= 5 else "fair"), unsafe_allow_html=True)

    if quality.get("checks"):
        st.plotly_chart(chart_completeness(quality["checks"]), use_container_width=True)

    _render_resume_identity(info, quality)


def _render_resume_identity(info: dict, quality: dict) -> None:
    """Render extracted contact info, skills, and quality breakdown."""
    st.markdown('<div class="section-heading">Extracted Information</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    pairs = [
        ("Name", info.get("name", "Not detected")),
        ("Email", info.get("email", "Not detected")),
        ("Phone", info.get("phone", "Not detected")),
        ("LinkedIn", info.get("linkedin", "Not detected")),
        ("GitHub", info.get("github", "Not detected")),
    ]
    cols = [c1, c2, c3]
    for i, (label, value) in enumerate(pairs):
        with cols[i % 3]:
            st.markdown(f"""<div class="info-card">
                <div class="info-label">{label}</div>
                <div class="info-value">{value}</div>
            </div>""", unsafe_allow_html=True)

    # Skills
    skills = info.get("skills", [])
    if skills:
        st.markdown('<div class="section-heading">Detected Skills</div>', unsafe_allow_html=True)
        st.markdown("".join(_skill_pill(s, "neutral") for s in skills), unsafe_allow_html=True)

    # Quality strengths / improvements
    strengths = quality.get("strengths", [])
    improvements = quality.get("improvements", [])
    if strengths or improvements:
        qa_col1, qa_col2 = st.columns(2)
        with qa_col1:
            if strengths:
                st.markdown('<div class="section-heading">Strengths</div>', unsafe_allow_html=True)
                for s in strengths:
                    st.markdown(f'<div class="strength-box">{s}</div>', unsafe_allow_html=True)
        with qa_col2:
            if improvements:
                st.markdown('<div class="section-heading">Areas for Improvement</div>', unsafe_allow_html=True)
                for s in improvements:
                    st.markdown(f'<div class="improvement-box">{s}</div>', unsafe_allow_html=True)


def _render_feature_grid() -> None:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">What you get</div>', unsafe_allow_html=True)
    features = [
        ("Resume Parsing", "Extracts name, contact info, skills, education, experience, and projects automatically."),
        ("Semantic Matching", "Uses sentence-transformer embeddings to measure how well your resume fits the role."),
        ("Skill Gap Analysis", "Pinpoints exactly which required and preferred skills are missing from your resume."),
        ("Quality Score", "Transparent heuristic checks: action verbs, quantifiable results, section completeness."),
        ("Improvement Tips", "Identifies weak bullet points and suggests stronger, impact-driven alternatives."),
        ("Interview Prep", "Generates technical, project, behavioral, and HR questions based on your actual resume."),
    ]
    cols = st.columns(3, gap="medium")
    for i, (title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""<div class="feature-card">
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ── PAGE: RESUME ANALYSIS ──
# ---------------------------------------------------------------------------

def page_resume_analysis() -> None:
    _page_header("Resume Analysis", "Full breakdown of your extracted resume content")

    if not st.session_state.resume_data:
        st.info("Upload and analyze a resume on the **Home** page first.")
        return

    data = st.session_state.resume_data
    info = data.get("resume_info", {})
    quality = data.get("quality", {})
    score_pct = round(quality.get("score", 0) * 100, 1)

    # ── Metrics ──
    mc = st.columns(4)
    with mc[0]:
        st.markdown(_metric_card("Quality Score", f"{score_pct}%", _score_class(score_pct)), unsafe_allow_html=True)
    with mc[1]:
        st.markdown(_metric_card("Skills Found", str(len(info.get("skills", []))), "good"), unsafe_allow_html=True)
    with mc[2]:
        st.markdown(_metric_card("Word Count", str(quality.get("word_count", 0)), "good"), unsafe_allow_html=True)
    with mc[3]:
        st.markdown(_metric_card("Action Verbs", str(quality.get("action_verbs_count", 0)), "excellent" if quality.get("action_verbs_count", 0) >= 5 else "fair"), unsafe_allow_html=True)

    # ── Contact info ──
    st.markdown('<div class="section-heading">Contact Information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    fields = [
        ("Name", info.get("name", "Not detected")),
        ("Email", info.get("email", "Not detected")),
        ("Phone", info.get("phone", "Not detected")),
        ("LinkedIn", info.get("linkedin", "Not detected")),
        ("GitHub", info.get("github", "Not detected")),
    ]
    for i, (lbl, val) in enumerate(fields):
        with [c1, c2, c3][i % 3]:
            st.markdown(f"""<div class="info-card">
                <div class="info-label">{lbl}</div>
                <div class="info-value">{val}</div>
            </div>""", unsafe_allow_html=True)

    # ── Skills ──
    skills = info.get("skills", [])
    st.markdown('<div class="section-heading">Detected Skills</div>', unsafe_allow_html=True)
    if skills:
        st.markdown("".join(_skill_pill(s, "neutral") for s in skills), unsafe_allow_html=True)
    else:
        st.caption("No skills detected.")

    # ── Resume sections ──
    st.markdown('<div class="section-heading">Resume Sections</div>', unsafe_allow_html=True)
    tabs = st.tabs(["Summary", "Education", "Experience", "Projects", "Certifications"])
    for tab, key in zip(tabs, ["summary", "education", "experience", "projects", "certifications"]):
        with tab:
            content = info.get(key, "Not detected")
            if content and content != "Not detected":
                st.text(content)
            else:
                st.caption("Section not detected in resume.")

    # ── Quality breakdown ──
    st.markdown('<div class="section-heading">Quality Analysis</div>', unsafe_allow_html=True)
    qa1, qa2 = st.columns(2, gap="large")
    with qa1:
        st.markdown("**Strengths**")
        for s in quality.get("strengths", []):
            st.markdown(f'<div class="strength-box">{s}</div>', unsafe_allow_html=True)
        if not quality.get("strengths"):
            st.caption("No strengths detected yet.")
    with qa2:
        st.markdown("**Areas for Improvement**")
        for s in quality.get("improvements", []):
            st.markdown(f'<div class="improvement-box">{s}</div>', unsafe_allow_html=True)
        if not quality.get("improvements"):
            st.caption("No issues detected.")

    if quality.get("checks"):
        with st.expander("View all quality checks"):
            st.plotly_chart(chart_completeness(quality["checks"]), use_container_width=True)

# ---------------------------------------------------------------------------
# ── PAGE: JOB MATCHING ──
# ---------------------------------------------------------------------------

def page_job_matching() -> None:
    _page_header("Job Matching", "Compare your resume against any job description")

    if not st.session_state.resume_text:
        st.warning("Upload a resume on the **Home** page first.")
        return

    st.markdown('<div class="section-heading">Job Description</div>', unsafe_allow_html=True)
    jd = st.text_area(
        "Job Description",
        value=st.session_state.job_description,
        height=200,
        label_visibility="collapsed",
        placeholder="Paste the full job description here...",
    )
    st.session_state.job_description = jd

    if st.button("Match Resume to Job", type="primary"):
        if len(jd.strip()) < 30:
            st.error("Job description is too short. Paste the full description.")
            return
        with st.spinner("Running semantic analysis — this may take 20–30 seconds on first run..."):
            try:
                result = api_client.match_resume(
                    st.session_state.resume_text, jd, st.session_state.resume_filename
                )
                st.session_state.match_data = result
                st.success("Matching complete.")
            except APIError as e:
                st.error(f"Matching failed: {e}")
                return

    data = st.session_state.match_data
    if not data:
        return

    scores = data["scores"]
    job_info = data.get("job_info", {})

    # ── Results ──
    st.markdown('<div class="section-heading">Results</div>', unsafe_allow_html=True)
    g_col, b_col = st.columns([1, 2], gap="large")
    with g_col:
        st.plotly_chart(chart_gauge(scores["overall_compatibility"]), use_container_width=True)
        st.markdown('<div style="text-align:center;font-size:0.7rem;color:#475569;margin-top:-0.5rem">Heuristic score — not a guaranteed ATS score</div>', unsafe_allow_html=True)
    with b_col:
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        bars = "".join([
            _score_bar("Skill Match (40%)", scores["skill_match"]),
            _score_bar("Semantic Similarity (30%)", scores["semantic_similarity"]),
            _score_bar("Keyword Coverage (20%)", scores["keyword_coverage"]),
            _score_bar("Resume Quality (10%)", scores["resume_quality"]),
        ])
        st.markdown(bars, unsafe_allow_html=True)

    # ── Job info ──
    st.markdown('<div class="section-heading">Job Details Extracted</div>', unsafe_allow_html=True)
    jc = st.columns(3)
    with jc[0]:
        st.markdown(f"""<div class="info-card">
            <div class="info-label">Job Title</div>
            <div class="info-value">{job_info.get("title", "Not specified")}</div>
        </div>""", unsafe_allow_html=True)
    with jc[1]:
        st.markdown(f"""<div class="info-card">
            <div class="info-label">Experience Required</div>
            <div class="info-value">{job_info.get("experience_requirement", "Not specified")}</div>
        </div>""", unsafe_allow_html=True)
    with jc[2]:
        st.markdown(f"""<div class="info-card">
            <div class="info-label">Education Required</div>
            <div class="info-value">{job_info.get("education_requirement", "Not specified")}</div>
        </div>""", unsafe_allow_html=True)

    keywords = job_info.get("keywords", [])
    if keywords:
        st.markdown('<div class="section-heading">Top Keywords</div>', unsafe_allow_html=True)
        st.markdown("".join(_skill_pill(kw, "neutral") for kw in keywords[:18]), unsafe_allow_html=True)

    st.plotly_chart(chart_score_breakdown(scores), use_container_width=True)

# ---------------------------------------------------------------------------
# ── PAGE: SKILL GAP ──
# ---------------------------------------------------------------------------

def page_skill_gap() -> None:
    _page_header("Skill Gap Analysis", "See exactly which skills you have and which you're missing")

    data = st.session_state.match_data
    if not data:
        st.warning("Run a job match first — go to **Job Matching**.")
        return

    skill_gap = data.get("skill_gap", {})
    job_info  = data.get("job_info", {})
    matched      = skill_gap.get("matched_skills", [])
    missing_req  = skill_gap.get("missing_required", [])
    missing_pref = skill_gap.get("missing_preferred", [])

    # ── Metrics ──
    mc = st.columns(3)
    with mc[0]:
        st.markdown(_metric_card("Matched Skills", str(len(matched)), "excellent"), unsafe_allow_html=True)
    with mc[1]:
        st.markdown(_metric_card("Missing Required", str(len(missing_req)), "poor" if missing_req else "excellent"), unsafe_allow_html=True)
    with mc[2]:
        st.markdown(_metric_card("Missing Preferred", str(len(missing_pref)), "fair" if missing_pref else "good"), unsafe_allow_html=True)

    st.plotly_chart(chart_skill_gap(matched, missing_req, missing_pref), use_container_width=True)

    # ── Skill columns ──
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        st.markdown('<div class="section-heading">Matched Skills</div>', unsafe_allow_html=True)
        if matched:
            st.markdown("".join(_skill_pill(s, "matched") for s in matched), unsafe_allow_html=True)
        else:
            st.caption("None matched.")

    with col2:
        st.markdown('<div class="section-heading">Missing Required</div>', unsafe_allow_html=True)
        if missing_req:
            st.markdown("".join(_skill_pill(s, "missing") for s in missing_req), unsafe_allow_html=True)
            st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)
            st.markdown("**Suggested learning resources:**")
            for skill in missing_req[:5]:
                q = skill.replace(" ", "+")
                st.markdown(f"- [{skill}](https://www.freecodecamp.org/news/search/?query={q})")
        else:
            st.success("All required skills covered!")

    with col3:
        st.markdown('<div class="section-heading">Missing Preferred</div>', unsafe_allow_html=True)
        if missing_pref:
            st.markdown("".join(_skill_pill(s, "preferred") for s in missing_pref), unsafe_allow_html=True)
        else:
            st.success("All preferred skills covered!")

    with st.expander("All skills in job description"):
        all_job_skills = job_info.get("all_skills", [])
        if all_job_skills:
            resume_set = {s.lower() for s in data.get("resume_info", {}).get("skills", [])}
            st.markdown(
                "".join(_skill_pill(s, "matched" if s.lower() in resume_set else "missing") for s in all_job_skills),
                unsafe_allow_html=True,
            )
        else:
            st.caption("No skills extracted from job description.")

# ---------------------------------------------------------------------------
# ── PAGE: RESUME IMPROVEMENT ──
# ---------------------------------------------------------------------------

def page_improvement() -> None:
    _page_header("Resume Improvement", "Identify weak language and get actionable suggestions")

    if not st.session_state.resume_text:
        st.warning("Upload a resume on the **Home** page first.")
        return

    if st.button("Generate Improvement Suggestions", type="primary"):
        with st.spinner("Scanning resume for improvement opportunities..."):
            try:
                result = api_client.improve_resume(st.session_state.resume_text)
                st.session_state.improve_data = result
            except APIError as e:
                st.error(f"Error: {e}")
                return

    data = st.session_state.improve_data
    if not data:
        st.info("Click the button above to generate suggestions.")
        return

    improvements = data.get("improvements", {})
    llm = data.get("llm_enhanced")

    # ── Weak bullets ──
    weak = improvements.get("weak_bullets", [])
    st.markdown('<div class="section-heading">Weak Bullet Points</div>', unsafe_allow_html=True)
    if weak:
        st.markdown(
            '<p style="font-size:0.82rem;color:#64748b;margin-bottom:0.8rem">'
            'These bullet points use weak or vague language. Rewrite them using your real experience.'
            '</p>',
            unsafe_allow_html=True,
        )
        for item in weak:
            st.markdown(f"""<div class="suggestion-box">
                <strong style="color:#fbbf24">Original:</strong> {item['original']}<br>
                <strong style="color:#f87171">Issue:</strong> {item['issue']}<br>
                <strong style="color:#4ade80">Suggestion:</strong> {item['suggestion']}
            </div>""", unsafe_allow_html=True)
    else:
        st.success("No weak bullet points detected.")

    # ── General suggestions ──
    general = improvements.get("general_suggestions", [])
    if general:
        st.markdown('<div class="section-heading">General Suggestions</div>', unsafe_allow_html=True)
        for item in general:
            ex = ""
            if item.get("examples"):
                ex = f"<br><em style='color:#64748b'>Example: {item['examples'][0][:120]}...</em>"
            st.markdown(f"""<div class="improvement-box">
                <strong style="color:#a5b4fc">[{item['category']}]</strong> {item['suggestion']}{ex}
            </div>""", unsafe_allow_html=True)

    # ── LLM rewrites ──
    if llm:
        st.markdown('<div class="section-heading">AI-Enhanced Rewrites</div>', unsafe_allow_html=True)
        st.info("Generated by LLM from your actual resume. Only use suggestions that accurately reflect your experience.")
        st.markdown(llm)
    else:
        st.markdown(
            '<p style="font-size:0.78rem;color:#475569;margin-top:0.8rem">'
            'Set <code>LLM_API_KEY</code> in <code>.env</code> for AI-enhanced bullet rewrites.'
            '</p>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# ── PAGE: INTERVIEW PREPARATION ──
# ---------------------------------------------------------------------------

def page_interview() -> None:
    _page_header("Interview Preparation", "Questions generated from your actual resume content")

    if not st.session_state.resume_text:
        st.warning("Upload a resume on the **Home** page first.")
        return

    if st.button("Generate Interview Questions", type="primary"):
        with st.spinner("Generating personalized interview questions..."):
            try:
                jd = st.session_state.job_description or None
                result = api_client.generate_interview(st.session_state.resume_text, jd)
                st.session_state.interview_data = result
            except APIError as e:
                st.error(f"Error: {e}")
                return

    data = st.session_state.interview_data
    if not data:
        st.info("Click the button above to generate questions.")
        return

    questions = data.get("questions", {})
    tabs = st.tabs(["Technical", "Project-Based", "Behavioral", "HR"])
    cats  = ["technical", "project_based", "behavioral", "hr"]

    for tab, cat in zip(tabs, cats):
        with tab:
            qs = questions.get(cat, [])
            if qs:
                for i, q in enumerate(qs, 1):
                    st.markdown(f'<div class="question-box"><strong style="color:#64748b">{i}.</strong> {q}</div>', unsafe_allow_html=True)
            else:
                st.caption("No questions generated for this category.")

    st.markdown(
        '<p style="font-size:0.78rem;color:#475569;margin-top:1rem">'
        'Questions are derived from your resume content only. Prepare answers using real experience.'
        '</p>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# ── PAGE: HISTORY ──
# ---------------------------------------------------------------------------

def page_history() -> None:
    _page_header("Analysis History", "All previous resume analyses stored locally")

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        st.button("Refresh", key="history_refresh")

    try:
        data = api_client.get_history()
    except APIError as e:
        st.error(f"Could not load history: {e}")
        return

    history = data.get("history", [])
    if not history:
        st.info("No analyses saved yet. Run a Job Match to start building history.")
        return

    st.markdown(f'<p style="font-size:0.8rem;color:#475569">{len(history)} record(s)</p>', unsafe_allow_html=True)

    import pandas as pd
    rows = []
    for r in history:
        rows.append({
            "Date":          r["created_at"][:19].replace("T", " ") if r.get("created_at") else "—",
            "File":          r.get("resume_filename", "—"),
            "Job Title":     r.get("job_title", "—"),
            "Compatibility": f"{r['compatibility_score']:.1f}%" if r.get("compatibility_score") is not None else "—",
            "Skill Match":   f"{r['skill_match_score']:.1f}%"   if r.get("skill_match_score")   is not None else "—",
            "Semantic Sim.": f"{r['semantic_similarity']:.1f}%"  if r.get("semantic_similarity")  is not None else "—",
            "Keyword Cov.":  f"{r['keyword_coverage']:.1f}%"     if r.get("keyword_coverage")     is not None else "—",
            "Quality":       f"{r['resume_quality_score']:.1f}%" if r.get("resume_quality_score") is not None else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    compat_scores = [r["compatibility_score"] for r in history if r.get("compatibility_score") is not None]
    if len(compat_scores) >= 2:
        st.plotly_chart(chart_trend(list(reversed(compat_scores))), use_container_width=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    page = _render_sidebar()

    dispatch = {
        "Home":                  page_home,
        "Resume Analysis":       page_resume_analysis,
        "Job Matching":          page_job_matching,
        "Skill Gap":             page_skill_gap,
        "Resume Improvement":    page_improvement,
        "Interview Preparation": page_interview,
        "History":               page_history,
    }
    dispatch.get(page, page_home)()


if __name__ == "__main__":
    main()
