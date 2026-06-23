import streamlit as st
from agents.rag_tool import get_vector_store
from agents.crew_builder import run_support_crew

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# Same pattern as all your previous projects.
# layout="centered" keeps content readable — no wide sprawl.
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechFlow AI Support",
    page_icon="🤖",
    layout="centered"
)

# ──────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE — LOAD ONCE AT STARTUP
#
# get_vector_store() builds the FAISS index from the hardcoded TechFlow FAQ.
# @st.cache_resource (defined inside rag_tool.py) ensures the embedding
# model and FAISS index are built ONCE and reused for every query.
# ──────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading TechFlow knowledge base..."):
    vector_store = get_vector_store()

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ──────────────────────────────────────────────────────────────────────────────

if "last_results" not in st.session_state:
    st.session_state.last_results = None

if "prefill_query" not in st.session_state:
    st.session_state.prefill_query = ""

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.title("🤖 TechFlow AI Support System")
st.markdown(
    "Four specialized AI agents work in sequence — "
    "**Classify → Research → Write → Quality Check** — "
    "to handle every customer query. Watch each agent's output appear live."
)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Query Input
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📨 Customer Query")

    st.markdown("**Quick test queries:**")

    samples = {
        "💳 Billing issue": (
            "I've been charged twice this month for my Professional plan. "
            "The duplicate charge appeared on my credit card statement yesterday. "
            "This is completely unacceptable and I need a full refund immediately."
        ),
        "🔧 Technical problem": (
            "My Gmail integration stopped syncing contacts 2 days ago. "
            "I've tried disconnecting and reconnecting but the status still shows Error. "
            "My whole team relies on this and it's blocking our sales work."
        ),
        "😤 Complaint": (
            "Your platform is absolutely terrible. I lost 3 months of pipeline data "
            "last week and nobody from your support team has helped me. "
            "I'm about to cancel and move to a competitor if this isn't resolved today."
        ),
        "❓ General question": (
            "I'm currently on the Professional plan with 8 users. "
            "Can you explain what additional features I would get on the Enterprise plan "
            "and whether there is a free trial available before committing?"
        ),
    }

    for label, sample_text in samples.items():
        if st.button(label, use_container_width=True):
            st.session_state.prefill_query = sample_text
            st.rerun()

    st.divider()

    query = st.text_area(
        "Or type your own query:",
        value=st.session_state.prefill_query,
        height=160,
        placeholder="Describe the customer's issue here...",
        help="The 4-agent crew will classify, research, write, and quality-check a response."
    )

    submitted = st.button(
        "🚀 Run Support Crew",
        type="primary",
        use_container_width=True,
    )

    st.divider()
    st.markdown("**📚 Knowledge Base**")
    st.success("✅ TechFlow CRM FAQ indexed")
    st.caption("Billing · Plans · Integrations · Troubleshooting · Policies")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN AREA — Agent Pipeline
# ──────────────────────────────────────────────────────────────────────────────

if submitted and query.strip():

    st.session_state.prefill_query = ""

    st.markdown("---")
    st.subheader("🔄 Agent Pipeline")
    st.caption("Each section below updates live as its agent finishes working.")
    st.markdown("")

    # ── Create placeholder containers for each agent ──────────────────────────
    # st.empty() creates a placeholder in the page layout.
    # crew_builder.py callbacks call container.markdown(...) to update them live.
    # Because callbacks run in the same thread as the main script,
    # Streamlit's websocket pushes each update to the browser immediately.

    st.markdown("#### 🔍 Agent 1 — Classifier")
    c1 = st.empty()
    c1.info("⏳ Analysing the query...")

    st.markdown("")
    st.markdown("#### 📚 Agent 2 — Researcher")
    c2 = st.empty()
    c2.info("⏳ Waiting for classification before searching knowledge base...")

    st.markdown("")
    st.markdown("#### ✍️ Agent 3 — Writer")
    c3 = st.empty()
    c3.info("⏳ Waiting for research findings before drafting response...")

    st.markdown("")
    st.markdown("#### ✅ Agent 4 — Quality Checker")
    c4 = st.empty()
    c4.info("⏳ Waiting for draft before quality review...")

    st.markdown("")
    st.markdown("---")

    final_header = st.empty()
    final_box    = st.empty()

    # ── Run the crew ──────────────────────────────────────────────────────────
    # run_support_crew() builds all 4 agents + tasks, attaches callbacks that
    # update the containers above, then calls crew.kickoff().
    # The function blocks until all 4 agents finish.
    try:
        with st.spinner("Crew is running... this takes 60-120 seconds total."):
            results = run_support_crew(
                query=query,
                containers={
                    "classifier": c1,
                    "researcher": c2,
                    "writer":     c3,
                    "qc":         c4,
                },
                groq_api_key=st.secrets["GROQ_API_KEY"],
                vector_store=vector_store,
            )

        # ── Final Response ────────────────────────────────────────────────────
        qc_output   = results.get("qc", "")
        qc_approved = "APPROVED" in qc_output.upper()

        final_header.markdown(
            "### ✅ Final Approved Response" if qc_approved
            else "### ⚠️ Final Response (QC Flagged — Review Recommended)"
        )
        final_box.success(results.get("writer", "No response generated."))

        st.session_state.last_results = results

    except Exception as e:
        c1.empty(); c2.empty(); c3.empty(); c4.empty()
        st.error(f"❌ Crew error: {str(e)}")
        st.info(
            "Common causes: Groq rate limit (wait 10 seconds and retry), "
            "or a CrewAI version conflict. Check the app logs for details."
        )

elif submitted and not query.strip():
    st.warning("⚠️ Please enter a customer query or click a sample query before running.")

elif not submitted and st.session_state.last_results:
    results = st.session_state.last_results
    st.markdown("---")
    st.subheader("📋 Last Run Results")
    st.caption("Submit a new query above to run the crew again.")

    with st.expander("🔍 Classifier Output",  expanded=False):
        st.markdown(results.get("classifier", ""))
    with st.expander("📚 Researcher Output", expanded=False):
        st.markdown(results.get("researcher", ""))
    with st.expander("✍️ Writer Output",     expanded=False):
        st.markdown(results.get("writer", ""))
    with st.expander("✅ QC Output",          expanded=False):
        st.markdown(results.get("qc", ""))

    st.markdown("---")
    qc_approved = "APPROVED" in results.get("qc", "").upper()
    st.markdown("### ✅ Final Approved Response" if qc_approved else "### ⚠️ Final Response (QC Flagged)")
    st.success(results.get("writer", ""))

else:
    st.markdown("")
    st.markdown("""
    ### 👈 How to use

    1. Click a **sample query** in the sidebar (or type your own)
    2. Click **Run Support Crew**
    3. Watch 4 agents work in sequence — each section lights up as it finishes
    4. The final approved response appears at the bottom

    ---

    **The 4 Agents:**

    | Agent | Job |
    |---|---|
    | 🔍 Classifier | Reads the query → assigns category (billing/technical/complaint/general) + urgency |
    | 📚 Researcher | Searches the TechFlow FAQ knowledge base using semantic search |
    | ✍️ Writer | Uses the classification + research to write a professional response |
    | ✅ Quality Checker | Reviews tone, accuracy, and completeness — flags any issues |
    """)
