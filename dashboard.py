# dashboard.py - VYOR AI Knowledge Intelligence Dashboard v2 Enhanced
# Requires: streamlit, requests, streamlit-lottie (optional), streamlit-shadcn-ui (optional)

import os
from pathlib import Path
import requests
import streamlit as st

# -- Optional: Lottie animations -------------------------------------------
st_lottie = lambda *a, **kw: None  # type: ignore[assignment]  # stub; real import below
try:
    from streamlit_lottie import st_lottie  # type: ignore[import-untyped]
    _LOTTIE_AVAIL = True
except ImportError:
    _LOTTIE_AVAIL = False

# -- Optional: Shadcn UI components ----------------------------------------
try:
    import streamlit_shadcn_ui as ui  # type: ignore[import-untyped]  # noqa: F401
    _SHADCN_AVAIL = True
except ImportError:
    _SHADCN_AVAIL = False

API = os.getenv("API_URL", "http://localhost:8000")

# -- Paths ------------------------------------------------------------------
_ASSETS = Path(__file__).parent / "assets"
_LOGO_PATH = _ASSETS / "logo.png"
_CSS_PATH  = _ASSETS / "dashboard.css"
_LOGO_EXISTS = _LOGO_PATH.exists()

# -- Page config ------------------------------------------------------------
st.set_page_config(
    page_title="VYOR AI - Knowledge Intelligence",
    page_icon=str(_LOGO_PATH) if _LOGO_EXISTS else "🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Load external CSS (Pyrefly never parses .css files) --------------------
if _CSS_PATH.exists():
    _css = _CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

# -- Lottie loader (cached) ------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_lottie(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


# -- Helper: confidence HTML -----------------------------------------------
def _conf_html(conf: float) -> str:
    pct  = conf * 100
    tier = "hi" if conf >= 0.80 else ("md" if conf >= 0.65 else "lo")
    return (
        '<div class="conf-row">'
        '  <span class="conf-label-left">Confidence</span>'
        '  <div class="conf-track">'
        f'    <div class="conf-bar {tier}" style="width:{pct:.1f}%"></div>'
        '  </div>'
        f'  <span class="conf-val {tier}">{pct:.0f}%</span>'
        '</div>'
    )


# -- Helper: citations HTML ------------------------------------------------
def _citations_html(citations: list) -> str:
    return "".join(
        '<div class="cite-item">'
        f'<span class="cite-idx">#{i+1}</span>'
        f'<span class="cite-text">{c}</span></div>'
        for i, c in enumerate(citations)
    )


# -- Neural-network SVG (hero fallback decoration) -------------------------
_NEURAL_SVG = (
    '<svg width="130" height="70" viewBox="0 0 130 70" fill="none" '
    'xmlns="http://www.w3.org/2000/svg">'
    '<line x1="18" y1="18" x2="55" y2="14" stroke="#6366f1" stroke-width="0.7" opacity="0.25"/>'
    '<line x1="18" y1="18" x2="55" y2="35" stroke="#6366f1" stroke-width="0.7" opacity="0.20"/>'
    '<line x1="18" y1="18" x2="55" y2="56" stroke="#6366f1" stroke-width="0.7" opacity="0.15"/>'
    '<line x1="18" y1="52" x2="55" y2="14" stroke="#6366f1" stroke-width="0.7" opacity="0.15"/>'
    '<line x1="18" y1="52" x2="55" y2="35" stroke="#6366f1" stroke-width="0.7" opacity="0.20"/>'
    '<line x1="18" y1="52" x2="55" y2="56" stroke="#6366f1" stroke-width="0.7" opacity="0.25"/>'
    '<line x1="62" y1="14" x2="112" y2="22" stroke="#8b5cf6" stroke-width="0.7" opacity="0.25"/>'
    '<line x1="62" y1="35" x2="112" y2="22" stroke="#8b5cf6" stroke-width="0.7" opacity="0.20"/>'
    '<line x1="62" y1="35" x2="112" y2="48" stroke="#8b5cf6" stroke-width="0.7" opacity="0.20"/>'
    '<line x1="62" y1="56" x2="112" y2="48" stroke="#8b5cf6" stroke-width="0.7" opacity="0.25"/>'
    '<circle cx="18" cy="18" r="5" fill="#6366f1" opacity="0.80"/>'
    '<circle cx="18" cy="52" r="5" fill="#6366f1" opacity="0.65"/>'
    '<circle cx="18" cy="18" r="5" fill="none" stroke="#6366f1" stroke-width="1">'
    '<animate attributeName="r" values="5;13;5" dur="2.8s" repeatCount="indefinite"/>'
    '<animate attributeName="opacity" values="0.7;0;0.7" dur="2.8s" repeatCount="indefinite"/>'
    '</circle>'
    '<circle cx="58" cy="14" r="4.5" fill="#8b5cf6" opacity="0.85"/>'
    '<circle cx="58" cy="35" r="4.5" fill="#8b5cf6" opacity="0.70"/>'
    '<circle cx="58" cy="56" r="4.5" fill="#8b5cf6" opacity="0.80"/>'
    '<circle cx="114" cy="22" r="5" fill="#22d3ee" opacity="0.90"/>'
    '<circle cx="114" cy="48" r="5" fill="#22d3ee" opacity="0.75"/>'
    '<circle r="2.5" fill="#22d3ee" opacity="0.9">'
    '<animateMotion dur="2.4s" repeatCount="indefinite" '
    'path="M18,18 C36,18 40,35 58,35 C76,35 94,22 114,22"/>'
    '<animate attributeName="opacity" values="0;1;1;0" dur="2.4s" repeatCount="indefinite"/>'
    '</circle>'
    '</svg>'
)

# -- Hero pills HTML --------------------------------------------------------
_PILLS_HTML = (
    '<div class="vyor-hero-pills">'
    '<span class="pill pill-ind">Qwen3-32B Backbone</span>'
    '<span class="pill pill-vio">Titans Neural Memory</span>'
    '<span class="pill pill-cyan">7-Mode RAG Stack</span>'
    '<span class="pill pill-grn">5-Layer Anti-Hallucination</span>'
    '</div>'
)

_HERO_BODY = (
    '<div class="vyor-hero"><div class="vyor-hero-inner">'
    '<div class="vyor-hero-text">'
    '<h1>VYOR AI - Knowledge Intelligence</h1>'
    + _PILLS_HTML +
    '</div>'
)


# -- Hero ------------------------------------------------------------------
lottie_data = None
if _LOTTIE_AVAIL:
    lottie_data = _load_lottie(
        "https://assets6.lottiefiles.com/packages/lf20_w51pcehl.json"
    )

if _LOGO_EXISTS:
    _hero_col, _logo_col = st.columns([3, 1])
    with _hero_col:
        st.markdown(_HERO_BODY + '</div></div>', unsafe_allow_html=True)
    with _logo_col:
        st.image(str(_LOGO_PATH), width=140)

elif _LOTTIE_AVAIL and lottie_data:
    _hero_col, _lottie_col = st.columns([4, 1])
    with _hero_col:
        st.markdown(_HERO_BODY + '</div></div>', unsafe_allow_html=True)
    with _lottie_col:
        st_lottie(lottie_data, height=90, key="hero_lottie")  # type: ignore[operator]

else:
    st.markdown(
        _HERO_BODY
        + f'<div style="flex-shrink:0;opacity:0.75">{_NEURAL_SVG}</div>'
        + '</div></div>',
        unsafe_allow_html=True,
    )


# -- Session state ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -- Sidebar ---------------------------------------------------------------
with st.sidebar:
    if _LOGO_EXISTS:
        _sb_logo, _sb_title = st.columns([1, 3])
        with _sb_logo:
            st.image(str(_LOGO_PATH), width=48)
        with _sb_title:
            st.markdown(
                '<div class="sidebar-logo-text">VYOR AI</div>'
                '<div class="sidebar-logo-sub">Knowledge Intelligence</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="sidebar-logo-text">VYOR AI</div>'
            '<div class="sidebar-logo-sub">Knowledge Intelligence</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.header("Upload Documents")
    st.caption("PDF  DOCX  PPTX  CSV  TXT  MD")

    uploaded = st.file_uploader(
        label="Choose a file",
        type=["pdf", "docx", "pptx", "csv", "txt", "md"],
        help="Upload any document to add it to the knowledge base.",
    )

    if uploaded and st.button("Upload and Process", type="primary", use_container_width=True):
        with st.spinner(f"Processing {uploaded.name} ..."):
            try:
                resp = requests.post(
                    f"{API}/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                st.success(f"Processed: {data['filename']}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Chunks",     data["total_chunks"])
                c2.metric("Qdrant",     data["saved_to_qdrant"])
                c3.metric("Neural mem", data["memory_updated"])
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API server. Is it running on port 8000?")
            except requests.exceptions.Timeout:
                st.error("Upload timed out. Try a smaller file.")
            except Exception as exc:
                st.error(f"Upload error: {exc}")

    st.divider()

    if st.button("System Health", use_container_width=True):
        try:
            hresp  = requests.get(f"{API}/health", timeout=5)
            health = hresp.json()
            status = health.get("status", "unknown")

            ov_cls = "badge badge-up" if status == "healthy" else "badge badge-down"
            ov_lbl = "HEALTHY" if status == "healthy" else "DEGRADED"
            rows = (
                '<div class="health-status-row">'
                '<span class="health-svc">Overall</span>'
                f'<span class="{ov_cls}">{ov_lbl}</span></div>'
            )
            for svc, state in health.get("services", {}).items():
                cls = (
                    "badge badge-up"   if state == "ok"      else
                    "badge badge-down" if state == "error"   else
                    "badge badge-ok"   if state == "standby" else
                    "badge badge-unk"
                )
                lbl = "UP" if state == "ok" else "DOWN" if state == "error" else state.upper()
                rows += (
                    '<div class="health-status-row">'
                    f'<span class="health-svc">{svc}</span>'
                    f'<span class="{cls}">{lbl}</span></div>'
                )
            st.markdown(f'<div class="health-panel">{rows}</div>', unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"API unreachable: {exc}")

    st.divider()
    st.caption("VYOR AI v2.0  |  NeurIPS 2025 Titans  |  Team 2")


# -- Main: Chat ------------------------------------------------------------
if not st.session_state.messages:
    st.info("Upload a document in the sidebar, then ask a question below to get started.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            conf = msg.get("confidence", 0.75)
            st.markdown(_conf_html(conf), unsafe_allow_html=True)
            if msg.get("citations"):
                with st.expander(f"Sources ({len(msg['citations'])})"):
                    st.markdown(_citations_html(msg["citations"]), unsafe_allow_html=True)
            if msg.get("uncertainty"):
                st.warning("Low confidence - verify against source documents.")


# -- Chat input ------------------------------------------------------------
if prompt := st.chat_input("Ask a question about your documents ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        typing_slot = st.empty()
        typing_slot.markdown(
            '<div class="typing-wrap">'
            '<span class="t-label">Thinking</span>'
            '<div class="typing-dots"><span></span><span></span><span></span></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        data: dict = {}  # always assigned in try; all except branches call st.stop()
        try:
            resp = requests.post(
                f"{API}/query",
                json={"query": prompt},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            typing_slot.empty()
            st.error("Request timed out.")
            st.stop()
        except requests.exceptions.ConnectionError:
            typing_slot.empty()
            st.error("Cannot reach API. Is uvicorn running on port 8000?")
            st.stop()
        except Exception as exc:
            typing_slot.empty()
            st.error(f"Error: {exc}")
            st.stop()

        typing_slot.empty()

        answer     = data["answer"]
        citations  = data.get("citations", [])
        confidence = float(data.get("confidence", 0.75))
        uncertain  = data.get("uncertainty", False)

        st.markdown(answer)
        st.markdown(_conf_html(confidence), unsafe_allow_html=True)

        if citations:
            with st.expander(f"Sources ({len(citations)})"):
                st.markdown(_citations_html(citations), unsafe_allow_html=True)

        if uncertain:
            st.warning("Low confidence - verify against source documents.")

    st.session_state.messages.append({
        "role":        "assistant",
        "content":     answer,
        "citations":   citations,
        "confidence":  confidence,
        "uncertainty": uncertain,
    })
