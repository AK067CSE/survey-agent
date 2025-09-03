import streamlit as st
from core.config import SURVEY_DOMAINS, INDIAN_REGIONS
from core.orchestrator import process_question, get_history
from core.db import db

st.set_page_config(page_title="AI Survey Agent", page_icon="🌾", layout="wide")

st.markdown(
    """
    <style>
    .main-header {font-size: 2.2rem; font-weight:700; margin-bottom: 0.5rem}
    .card {background:#fff;border-radius:10px;padding:1rem;box-shadow:0 1px 4px rgba(0,0,0,.08)}
    .response-box {background:#f8f9fa;padding:1rem;border-left:4px solid #1f77b4;border-radius:6px}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🌾 Advanced AI Multi-Agent Survey System</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Configuration")
    domain = st.selectbox("Domain", SURVEY_DOMAINS, index=0)
    region_category = st.selectbox("Region group", list(INDIAN_REGIONS.keys()))
    region = st.selectbox("Region", INDIAN_REGIONS[region_category])
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if st.button("New Session"):
        st.session_state.session_id = None
        st.success("Session reset")

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Ask a survey question")
    question = st.text_area("Question", height=120, placeholder="E.g., What are irrigation challenges during monsoon in Bihar?")
    extra_ctx = st.text_area("Optional context", height=80)
    go = st.button("Run", type="primary")

    if go and question.strip():
        ctx = {"additional": extra_ctx} if extra_ctx.strip() else None
        with st.spinner("Processing..."):
            res = process_question(domain, question, region, st.session_state.session_id, ctx)
            if res.get("session_id"):
                st.session_state.session_id = res["session_id"]
        if res.get("status") == "success":
            st.success("Done")
            st.markdown("**Response**")
            st.markdown(f"<div class='response-box'>{res['agent_response']}</div>", unsafe_allow_html=True)
            with st.expander("Raw JSON"):
                st.json(res)
        else:
            st.error(res.get("error", "Unknown error"))

with col2:
    st.subheader("History")
    if st.session_state.get("session_id"):
        h = get_history(st.session_state.session_id)
        for i, item in enumerate(reversed(h[-10:])):
            with st.expander(f"Q{i+1}"):
                st.write("Q:", item.get("q"))
                st.json(item.get("a"))
        if st.button("Finalize session to JSON"):
            from core.orchestrator import finalize_session
            final_json = finalize_session(st.session_state.session_id)
            st.success("Final JSON ready")
            st.json(final_json)

st.divider()

st.subheader("Recent responses (DB)")
rows = db.list_responses(25)
if rows:
    for r in rows[:5]:
        st.write(f"[{r.get('domain')}] {r.get('region')} - {r.get('question')}")
else:
    st.caption("No responses yet.")