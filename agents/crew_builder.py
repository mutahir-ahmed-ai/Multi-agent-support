import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
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

    CrewAI coordinates the agents:
    - Each Agent has a role, goal, and backstory — defining its personality and expertise
    - Each Task has a description, expected_output, and context (previous task outputs)
    - The Crew runs them in Process.sequential order — one after another
    - Task callbacks update the Streamlit UI containers as each agent finishes

    Args:
        query:        The customer's support message
        containers:   Dict of st.empty() placeholders keyed by agent name
        groq_api_key: From st.secrets["GROQ_API_KEY"]
        vector_store: FAISS index built from the TechFlow FAQ

    Returns:
        Dict with keys "classifier", "researcher", "writer", "qc"
    """

    # ── LLM instances ────────────────────────────────────────────────────────
    # In CrewAI, each Agent can have its own LLM or share one.
    # We use two: low temperature for factual agents, higher for the writer.
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

    # ── RAG Tool ─────────────────────────────────────────────────────────────
    # Wraps the FAISS index in a LangChain Tool.
    # CrewAI passes this to the Researcher Agent's tools list.
    # The Agent decides when to call it and what query to pass.
    rag_tool = build_rag_tool(vector_store)

    # ── Captured outputs ─────────────────────────────────────────────────────
    captured = {"classifier": "", "researcher": "", "writer": "", "qc": ""}

    # ── Callback helpers ─────────────────────────────────────────────────────
    # CrewAI calls the Task callback with the task output when it completes.
    # In crewai 0.28.8, the output is a raw string.
    # We update the Streamlit container AND save to captured dict.

    def extract(output) -> str:
        if hasattr(output, "raw_output"): return str(output.raw_output)
        if hasattr(output, "result"):     return str(output.result)
        return str(output)

    def on_classifier_done(output):
        text = extract(output)
        captured["classifier"] = text
        containers["classifier"].markdown(f"**Classification complete:**\n\n{text}")

    def on_researcher_done(output):
        text = extract(output)
        captured["researcher"] = text
        containers["researcher"].markdown(f"**Research findings:**\n\n{text}")

    def on_writer_done(output):
        text = extract(output)
        captured["writer"] = text
        containers["writer"].success(f"**Draft response:**\n\n{text}")

    def on_qc_done(output):
        text = extract(output)
        captured["qc"] = text
        if "APPROVED" in text.upper():
            containers["qc"].success(f"**Quality Assessment:**\n\n{text}")
        else:
            containers["qc"].warning(f"**Quality Assessment (Issues Flagged):**\n\n{text}")

    # ── AGENT DEFINITIONS ────────────────────────────────────────────────────
    # role:             The agent's job title — shown in CrewAI's verbose output
    # goal:             What this agent is trying to achieve
    # backstory:        Shapes HOW the agent responds — its personality and expertise
    # llm:              The language model powering this agent
    # tools:            List of tools this agent can call (only Researcher gets RAG)
    # allow_delegation: If True, agents can hand tasks to each other — we control
    #                   flow explicitly via Task context, so this stays False
    # max_iter:         Maximum ReAct reasoning steps before the agent gives up
    # verbose:          We set False — callbacks handle the UI, not stdout logs

    classifier_agent = Agent(
        role="Senior Customer Support Classifier",
        goal=(
            "Accurately categorise incoming customer support queries by type and urgency "
            "so the support team can route and prioritise responses correctly."
        ),
        backstory=(
            "You are TechFlow's most experienced support analyst with 7 years of triaging "
            "customer tickets. You can immediately identify the category, urgency, and "
            "emotional state of any customer message. Your classifications directly determine "
            "how quickly and how the team responds."
        ),
        llm=llm_factual,
        allow_delegation=False,
        max_iter=3,
        verbose=False,
    )

    researcher_agent = Agent(
        role="TechFlow Knowledge Base Specialist",
        goal=(
            "Find the most accurate and complete information from TechFlow's documentation "
            "to help the Writer craft a factually correct response."
        ),
        backstory=(
            "You are TechFlow's resident expert on every policy, feature, and procedure. "
            "You know the exact refund policy, the precise steps to fix any integration "
            "issue, and every plan difference by heart. You search the knowledge base "
            "thoroughly and compile only the facts directly relevant to the issue."
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
            "Write professional, empathetic, and solution-focused customer support "
            "responses that resolve issues clearly and leave customers feeling valued."
        ),
        backstory=(
            "You are TechFlow's head of customer communications with a reputation for "
            "turning frustrated customers into loyal advocates. You balance professional "
            "and genuinely human tone. You never use corporate jargon. You acknowledge "
            "problems directly, provide clear steps, and always close on a reassuring note."
        ),
        llm=llm_writer,
        allow_delegation=False,
        max_iter=3,
        verbose=False,
    )

    qc_agent = Agent(
        role="Customer Support Quality Assurance Lead",
        goal=(
            "Ensure every customer support response meets TechFlow's quality standards "
            "for tone, accuracy, completeness, and compliance before it is sent."
        ),
        backstory=(
            "You are TechFlow's QA lead responsible for maintaining a 95% CSAT score. "
            "You review every response against five criteria: empathetic tone, factual "
            "accuracy, completeness, clarity of next steps, and compliance. "
            "You are thorough but fair — you approve good responses quickly."
        ),
        llm=llm_factual,
        allow_delegation=False,
        max_iter=3,
        verbose=False,
    )

    # ── TASK DEFINITIONS ─────────────────────────────────────────────────────
    # description:     Exact instructions for the agent
    # expected_output: The format/content the agent should return
    # agent:           Which CrewAI Agent executes this task
    # context:         List of previous Tasks whose outputs this agent can see
    #                  CrewAI automatically prepends context outputs to the prompt
    # callback:        Function called by CrewAI when the task completes

    task_classify = Task(
        description=(
            f"Analyse the following customer support query and produce a structured classification.\n\n"
            f"CUSTOMER QUERY:\n{query}\n\n"
            f"Your classification must include:\n"
            f"1. CATEGORY — choose exactly one: billing / technical / complaint / general\n"
            f"2. URGENCY — choose exactly one: high / medium / low\n"
            f"3. KEY ISSUES — bullet list of the specific problems raised\n"
            f"4. CUSTOMER SENTIMENT — one word (frustrated/angry/neutral/curious/confused)\n"
            f"5. TONE GUIDANCE — one sentence on how the Writer should approach this\n"
        ),
        expected_output=(
            "A structured classification with CATEGORY, URGENCY, KEY ISSUES, "
            "CUSTOMER SENTIMENT, and TONE GUIDANCE clearly labelled."
        ),
        agent=classifier_agent,
        callback=on_classifier_done,
    )

    task_research = Task(
        description=(
            f"Search the TechFlow knowledge base to find all information relevant to this query.\n\n"
            f"ORIGINAL CUSTOMER QUERY:\n{query}\n\n"
            f"You have the classification from the previous agent (see context).\n"
            f"Use the TechFlow Knowledge Base Search tool. Make multiple targeted searches.\n\n"
            f"Your output must include:\n"
            f"1. RELEVANT SECTIONS FOUND\n"
            f"2. KEY FACTS — specific facts, prices, timeframes, procedures\n"
            f"3. RESOLUTION STEPS — exact steps the customer should take (if technical)\n"
            f"4. APPLICABLE POLICIES — refund, billing, or support policies that apply\n"
            f"5. SUPPORT CONTACTS — relevant email addresses or links\n"
        ),
        expected_output=(
            "A research summary with RELEVANT SECTIONS FOUND, KEY FACTS, "
            "RESOLUTION STEPS, APPLICABLE POLICIES, and SUPPORT CONTACTS."
        ),
        agent=researcher_agent,
        context=[task_classify],
        callback=on_researcher_done,
    )

    task_write = Task(
        description=(
            f"Write a complete professional customer support response.\n\n"
            f"ORIGINAL CUSTOMER QUERY:\n{query}\n\n"
            f"You have the classification and research from the previous agents (see context).\n\n"
            f"Requirements:\n"
            f"- Open with genuine empathy — name the specific issue\n"
            f"- Address EVERY issue raised in the query\n"
            f"- Use specific facts from the research (prices, timeframes, steps, contacts)\n"
            f"- Provide clear numbered steps where the customer must take action\n"
            f"- Professional but warm — no corporate jargon\n"
            f"- Sign off as: TechFlow Customer Support Team\n"
            f"- Start with: SUBJECT: [email subject line]\n"
        ),
        expected_output=(
            "A complete support email with SUBJECT LINE and full RESPONSE BODY — "
            "greeting, empathetic opening, factual body, numbered steps, closing."
        ),
        agent=writer_agent,
        context=[task_classify, task_research],
        callback=on_writer_done,
    )

    task_qc = Task(
        description=(
            f"Perform quality assurance review of the draft customer support response.\n\n"
            f"ORIGINAL CUSTOMER QUERY:\n{query}\n\n"
            f"Review the Writer's draft (see context) against:\n"
            f"1. TONE: empathetic, professional, non-defensive?\n"
            f"2. ACCURACY: facts consistent with documentation?\n"
            f"3. COMPLETENESS: all issues in the original query addressed?\n"
            f"4. CLARITY: next steps clear and actionable?\n"
            f"5. COMPLIANCE: any unrealistic promises or inappropriate statements?\n\n"
            f"Give PASS or NEEDS IMPROVEMENT for each. Then OVERALL VERDICT and QUALITY SCORE /10.\n"
        ),
        expected_output=(
            "QA assessment with criterion scores (PASS/NEEDS IMPROVEMENT), "
            "OVERALL VERDICT (APPROVED or NEEDS REVISION), QUALITY SCORE /10, QA NOTES."
        ),
        agent=qc_agent,
        context=[task_write],
        callback=on_qc_done,
    )

    # ── CREW ASSEMBLY ────────────────────────────────────────────────────────
    # Process.sequential: tasks run in order, one after another
    # Each task waits for the previous to finish before starting
    # CrewAI handles the context injection automatically from context=[...]
    crew = Crew(
        agents=[classifier_agent, researcher_agent, writer_agent, qc_agent],
        tasks=[task_classify, task_research, task_write, task_qc],
        process=Process.sequential,
        verbose=0,
    )

    # kickoff() blocks until all 4 tasks complete.
    # Callbacks fire during execution and update the Streamlit containers live.
    crew.kickoff()

    return captured
