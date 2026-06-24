import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from agents.rag_tool import build_rag_tool


def run_support_crew(
    query: str,
    containers: dict,
    groq_api_key: str,
    vector_store: FAISS,
) -> dict:
    """
    Build and run the 4-agent CrewAI customer support crew.

    Uses Groq's OpenAI-compatible API endpoint via langchain-openai.
    This avoids langchain-groq version conflicts while keeping full
    temperature control and LangChain compatibility with crewai 0.28.8.

    CrewAI Process.sequential runs tasks in order:
        Classifier → Researcher → Writer → Quality Checker
    Each task receives context from previous tasks automatically.
    Task callbacks update Streamlit containers live as each agent finishes.
    """

    # ── Groq via OpenAI-compatible endpoint ──────────────────────────────────
    # Groq provides an OpenAI-compatible API at api.groq.com/openai/v1
    # langchain-openai works with langchain 0.1.x (no version conflict)
    # This gives us full temperature control per agent
    os.environ["GROQ_API_KEY"] = groq_api_key

    llm_factual = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        openai_api_key=groq_api_key,
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0.1,
    )
    llm_writer = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        openai_api_key=groq_api_key,
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0.4,
    )

    # ── RAG Tool ─────────────────────────────────────────────────────────────
    rag_tool = build_rag_tool(vector_store)

    # ── Captured outputs ─────────────────────────────────────────────────────
    captured = {"classifier": "", "researcher": "", "writer": "", "qc": ""}

    # ── Task outputs captured after kickoff ──────────────────────────────────
    # In crewai 0.28.8, the most reliable way to get each task's output is to
    # read task.output directly after crew.kickoff() completes.
    # Callbacks don't reliably update Streamlit containers because CrewAI runs
    # tasks in its own execution context separate from Streamlit's main thread.

    # ── Agents ────────────────────────────────────────────────────────────────
    classifier_agent = Agent(
        role="Senior Customer Support Classifier",
        goal=(
            "Accurately categorise incoming customer support queries by type "
            "and urgency so the support team can route and prioritise correctly."
        ),
        backstory=(
            "You are TechFlow's most experienced support analyst with 7 years "
            "of triaging customer tickets. You immediately identify category, "
            "urgency, and emotional state from any customer message."
        ),
        llm=llm_factual,
        allow_delegation=False,
        max_iter=3,
        verbose=False,
    )

    researcher_agent = Agent(
        role="TechFlow Knowledge Base Specialist",
        goal=(
            "Find the most accurate and complete information from TechFlow's "
            "documentation to help the Writer craft a factually correct response."
        ),
        backstory=(
            "You are TechFlow's resident expert on every policy, feature, and "
            "procedure. You search the knowledge base thoroughly and compile "
            "only the facts directly relevant to the issue."
        ),
        llm=llm_factual,
        tools=[rag_tool],
        allow_delegation=False,
        max_iter=5,
        verbose=False,
    )

    writer_agent = Agent(
        role="Senior Customer Support Writer",
        goal=(
            "Write professional, empathetic, solution-focused customer support "
            "responses that resolve issues clearly and leave customers feeling valued."
        ),
        backstory=(
            "You are TechFlow's head of customer communications — famous for "
            "turning frustrated customers into loyal advocates. You balance "
            "professional and genuinely human tone. No corporate jargon. "
            "You acknowledge problems directly and always close reassuringly."
        ),
        llm=llm_writer,
        allow_delegation=False,
        max_iter=3,
        verbose=False,
    )

    qc_agent = Agent(
        role="Customer Support Quality Assurance Lead",
        goal=(
            "Ensure every customer support response meets TechFlow's quality "
            "standards for tone, accuracy, completeness, and compliance."
        ),
        backstory=(
            "You are TechFlow's QA lead responsible for maintaining a 95% CSAT "
            "score. You review every response against five criteria and give "
            "a clear verdict. Thorough but fair — you approve good work quickly."
        ),
        llm=llm_factual,
        allow_delegation=False,
        max_iter=3,
        verbose=False,
    )

    # ── Tasks ─────────────────────────────────────────────────────────────────
    task_classify = Task(
        description=(
            f"Analyse this customer support query and produce a structured classification.\n\n"
            f"CUSTOMER QUERY:\n{query}\n\n"
            f"Output:\n"
            f"CATEGORY: billing / technical / complaint / general (choose one)\n"
            f"URGENCY: high / medium / low (choose one)\n"
            f"KEY ISSUES: bullet list of specific problems raised\n"
            f"CUSTOMER SENTIMENT: one word\n"
            f"TONE GUIDANCE: one sentence for the Writer\n"
        ),
        expected_output="Structured classification with CATEGORY, URGENCY, KEY ISSUES, CUSTOMER SENTIMENT, TONE GUIDANCE.",
        agent=classifier_agent,
    )

    task_research = Task(
        description=(
            f"Search the TechFlow knowledge base for all information relevant to this query.\n\n"
            f"ORIGINAL QUERY:\n{query}\n\n"
            f"Use the TechFlow Knowledge Base Search tool. Make multiple searches.\n\n"
            f"Output:\n"
            f"RELEVANT SECTIONS FOUND\n"
            f"KEY FACTS: specific facts, prices, timeframes, procedures\n"
            f"RESOLUTION STEPS: exact steps the customer should take\n"
            f"APPLICABLE POLICIES\n"
            f"SUPPORT CONTACTS\n"
        ),
        expected_output="Research summary with RELEVANT SECTIONS, KEY FACTS, RESOLUTION STEPS, APPLICABLE POLICIES, SUPPORT CONTACTS.",
        agent=researcher_agent,
        context=[task_classify],
    )

    task_write = Task(
        description=(
            f"Write a complete professional customer support email response.\n\n"
            f"ORIGINAL QUERY:\n{query}\n\n"
            f"Use the classification and research from previous agents (see context).\n\n"
            f"Requirements:\n"
            f"- Start with SUBJECT: [subject line]\n"
            f"- Open with genuine empathy — name the specific issue\n"
            f"- Address EVERY issue in the query\n"
            f"- Use specific facts from research (prices, timeframes, steps, contacts)\n"
            f"- Numbered steps where customer must take action\n"
            f"- Professional but warm tone — no jargon\n"
            f"- Sign off: TechFlow Customer Support Team\n"
        ),
        expected_output="Complete support email with SUBJECT LINE and full RESPONSE BODY.",
        agent=writer_agent,
        context=[task_classify, task_research],
    )

    task_qc = Task(
        description=(
            f"Quality assurance review of the draft support response.\n\n"
            f"ORIGINAL QUERY:\n{query}\n\n"
            f"Review the Writer's draft (see context) against:\n"
            f"1. TONE: empathetic, professional, non-defensive?\n"
            f"2. ACCURACY: facts match documentation?\n"
            f"3. COMPLETENESS: all issues in the query addressed?\n"
            f"4. CLARITY: next steps clear and actionable?\n"
            f"5. COMPLIANCE: any unrealistic promises?\n\n"
            f"Give PASS or NEEDS IMPROVEMENT for each.\n"
            f"Then OVERALL VERDICT: APPROVED or NEEDS REVISION\n"
            f"And QUALITY SCORE: X/10\n"
        ),
        expected_output="QA assessment with criterion scores, OVERALL VERDICT (APPROVED/NEEDS REVISION), QUALITY SCORE /10.",
        agent=qc_agent,
        context=[task_write],
    )

    # ── Crew ──────────────────────────────────────────────────────────────────
    crew = Crew(
        agents=[classifier_agent, researcher_agent, writer_agent, qc_agent],
        tasks=[task_classify, task_research, task_write, task_qc],
        process=Process.sequential,
        verbose=0,
    )

    # Run the crew — blocks until all 4 tasks complete
    crew.kickoff()

    # ── Read task outputs after kickoff ───────────────────────────────────────
    # task.output is a TaskOutput object in crewai 0.28.8
    # str() gives us the raw text output from each agent
    def get_output(task):
        if task.output is None:
            return "No output captured."
        if hasattr(task.output, "exported_output") and task.output.exported_output:
            return str(task.output.exported_output)
        if hasattr(task.output, "raw_output") and task.output.raw_output:
            return str(task.output.raw_output)
        return str(task.output)

    captured["classifier"] = get_output(task_classify)
    captured["researcher"] = get_output(task_research)
    captured["writer"]     = get_output(task_write)
    captured["qc"]         = get_output(task_qc)

    # ── Update UI containers now that all outputs are available ───────────────
    containers["classifier"].markdown(
        f"**Classification complete:**\n\n{captured['classifier']}"
    )
    containers["researcher"].markdown(
        f"**Research findings:**\n\n{captured['researcher']}"
    )
    containers["writer"].success(
        f"**Draft response:**\n\n{captured['writer']}"
    )
    if "APPROVED" in captured["qc"].upper():
        containers["qc"].success(
            f"**Quality Assessment:**\n\n{captured['qc']}"
        )
    else:
        containers["qc"].warning(
            f"**Quality Assessment (Issues Flagged):**\n\n{captured['qc']}"
        )

    return captured
