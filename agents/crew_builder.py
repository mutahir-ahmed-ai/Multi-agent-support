from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.vectorstores import FAISS

# ──────────────────────────────────────────────────────────────────────────────
# WHY NO CREWAI?
#
# crewai requires Python <=3.13. Streamlit Cloud runs Python 3.14.
# No version of crewai installs on Python 3.14 — confirmed by pip.
#
# Solution: build the same 4-agent pipeline manually with LangChain.
# Each "agent" is a ChatGroq call with a unique system prompt (role + goal +
# backstory) and receives context from the previous agent's output.
#
# The behaviour is identical to CrewAI's sequential Process:
#   Classifier → Researcher → Writer → Quality Checker
# The UI, live updates, and context passing all work exactly the same way.
# ──────────────────────────────────────────────────────────────────────────────


def run_support_crew(
    query: str,
    containers: dict,
    groq_api_key: str,
    vector_store: FAISS,
) -> dict:
    """
    Run 4 agents in sequence. Each agent is a ChatGroq call with a distinct
    system prompt. Output of each feeds into the next as context.

    Args:
        query:        The customer's support message
        containers:   Dict of st.empty() placeholders keyed by agent name
        groq_api_key: From st.secrets["GROQ_API_KEY"]
        vector_store: FAISS index built from the TechFlow FAQ

    Returns:
        Dict with keys "classifier", "researcher", "writer", "qc"
    """

    # ── LLM instances ─────────────────────────────────────────────────────────
    # Two instances: factual (low temp) for classifier/researcher/QC,
    # slightly warmer for the writer so responses feel more natural.
    llm_factual = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.1,
    )
    llm_writer = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.4,
    )

    captured = {}

    # ──────────────────────────────────────────────────────────────────────────
    # AGENT 1 — CLASSIFIER
    # Reads the query, outputs: category + urgency + sentiment + tone guidance
    # ──────────────────────────────────────────────────────────────────────────
    containers["classifier"].info("⏳ Classifying the query...")

    classifier_system = """You are TechFlow's Senior Customer Support Classifier with 7 years of experience triaging support tickets.
Your job is to analyse a customer query and produce a structured classification.

Output format (use these exact labels):
CATEGORY: [billing / technical / complaint / general]
  - billing: charges, invoices, refunds, payments, plan changes
  - technical: features not working, integrations failing, sync issues, bugs
  - complaint: frustration, threats to cancel, escalation, dissatisfaction
  - general: product questions, plan comparisons, how-to requests

URGENCY: [high / medium / low]
  - high: financial impact, data loss, system down, cancellation threat
  - medium: important feature broken, integration issue, plan change needed
  - low: general questions, minor inconveniences, feature curiosity

KEY ISSUES: bullet list of the specific problems or questions raised
CUSTOMER SENTIMENT: one word (frustrated / angry / neutral / curious / confused)
TONE GUIDANCE: one sentence on how the Writer should approach this response"""

    result1 = llm_factual.invoke([
        SystemMessage(content=classifier_system),
        HumanMessage(content=f"Customer query to classify:\n\n{query}")
    ])
    captured["classifier"] = result1.content
    containers["classifier"].markdown(f"**Classification complete:**\n\n{result1.content}")

    # ──────────────────────────────────────────────────────────────────────────
    # AGENT 2 — RESEARCHER
    # Searches the FAISS knowledge base, synthesises relevant facts
    # No crewai tool needed — we call vector_store.similarity_search() directly
    # ──────────────────────────────────────────────────────────────────────────
    containers["researcher"].info("⏳ Searching knowledge base...")

    # Pull top 4 relevant chunks from the TechFlow FAQ
    search_docs = vector_store.similarity_search(query, k=4)
    kb_context = "\n\n---\n\n".join([doc.page_content for doc in search_docs])

    researcher_system = """You are TechFlow's Knowledge Base Specialist — the most thorough researcher on the support team.
You receive a customer query, its classification, and relevant sections from the TechFlow knowledge base.
Your job is to extract and organise only the facts that are directly useful for writing the support response.

Output format:
RELEVANT SECTIONS FOUND: which parts of the documentation apply
KEY FACTS: specific facts, prices, timeframes, or procedures relevant to this issue
RESOLUTION STEPS: exact numbered steps the customer should take (if technical issue)
APPLICABLE POLICIES: any refund, billing, cancellation, or support policies that apply
SUPPORT CONTACTS: relevant email addresses or links from the documentation"""

    result2 = llm_factual.invoke([
        SystemMessage(content=researcher_system),
        HumanMessage(content=(
            f"Customer query:\n{query}\n\n"
            f"Classification from previous agent:\n{result1.content}\n\n"
            f"Relevant knowledge base sections:\n{kb_context}"
        ))
    ])
    captured["researcher"] = result2.content
    containers["researcher"].markdown(f"**Research findings:**\n\n{result2.content}")

    # ──────────────────────────────────────────────────────────────────────────
    # AGENT 3 — WRITER
    # Uses classification (tone/urgency) + research (facts) to write the email
    # ──────────────────────────────────────────────────────────────────────────
    containers["writer"].info("⏳ Drafting the response...")

    writer_system = """You are TechFlow's Senior Customer Support Writer — famous for turning frustrated customers into loyal advocates.
You receive a customer query, its classification (tone/urgency guidance), and researched facts.
Write a complete professional customer support email response.

Requirements:
- Open with genuine empathy — acknowledge the specific issue by name, do not be generic
- Address EVERY issue raised in the customer's message
- Use the specific facts from the research (plan names, prices, timeframes, steps, email addresses)
- Provide clear numbered steps where the customer needs to take action
- Professional but warm tone — no corporate jargon, no hollow phrases like "We apologise for any inconvenience"
- Close with a reassuring, forward-looking statement
- Sign off as: TechFlow Customer Support Team

Start with:
SUBJECT: [email subject line]

Then write the full email body."""

    result3 = llm_writer.invoke([
        SystemMessage(content=writer_system),
        HumanMessage(content=(
            f"Customer query:\n{query}\n\n"
            f"Classification and tone guidance:\n{result1.content}\n\n"
            f"Research findings and facts to use:\n{result2.content}"
        ))
    ])
    captured["writer"] = result3.content
    containers["writer"].success(f"**Draft response:**\n\n{result3.content}")

    # ──────────────────────────────────────────────────────────────────────────
    # AGENT 4 — QUALITY CHECKER
    # Reviews the draft — flags issues or approves. Flag-only mode (no rewrite).
    # app.py reads for the word "APPROVED" to set the UI colour.
    # ──────────────────────────────────────────────────────────────────────────
    containers["qc"].info("⏳ Running quality check...")

    qc_system = """You are TechFlow's Customer Support QA Lead responsible for maintaining a 95% CSAT score.
Review the draft response against these criteria and give a clear assessment.

Output format:
TONE: PASS or NEEDS IMPROVEMENT — [brief note]
ACCURACY: PASS or NEEDS IMPROVEMENT — [brief note]
COMPLETENESS: PASS or NEEDS IMPROVEMENT — [brief note — did it address ALL issues?]
CLARITY: PASS or NEEDS IMPROVEMENT — [brief note — are next steps actionable?]
COMPLIANCE: PASS or NEEDS IMPROVEMENT — [any unrealistic promises or inappropriate statements?]
QUALITY SCORE: X/10
OVERALL VERDICT: APPROVED or NEEDS REVISION
QA NOTES: [any specific commendations or issues to address]"""

    result4 = llm_factual.invoke([
        SystemMessage(content=qc_system),
        HumanMessage(content=(
            f"Original customer query:\n{query}\n\n"
            f"Draft response to review:\n{result3.content}"
        ))
    ])
    captured["qc"] = result4.content

    if "APPROVED" in result4.content.upper():
        containers["qc"].success(f"**Quality Assessment:**\n\n{result4.content}")
    else:
        containers["qc"].warning(f"**Quality Assessment (Issues Flagged):**\n\n{result4.content}")

    return captured
