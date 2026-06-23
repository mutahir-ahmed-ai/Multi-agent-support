import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from agents.rag_tool import build_rag_tool

# ──────────────────────────────────────────────────────────────────────────────
# WHY THIS FILE EXISTS
#
# This file is the heart of the project. It defines:
#   - 4 CrewAI Agents (the workers)
#   - 4 CrewAI Tasks (the jobs assigned to each worker)
#   - 1 CrewAI Crew (the team that runs everything in sequence)
#
# HOW CREWAI DIFFERS FROM LANGCHAIN AGENTEXECUTOR (your Project 4):
#
#   LangChain AgentExecutor = ONE agent, ONE task, loop until done.
#     The agent reasons (Thought → Action → Observation) until it decides
#     it has enough information, then produces a final answer.
#
#   CrewAI Crew = MULTIPLE agents, each with a SPECIFIC role and task,
#     working in sequence (or hierarchy). Each agent is an expert at
#     one specific job. The output of one agent becomes the context
#     for the next — like passing a baton in a relay race.
#
# CREWAI CORE CONCEPTS:
#
#   Agent: A worker with a role, goal, and backstory. The role and goal
#     define what it does. The backstory shapes HOW it responds — its
#     personality and expertise level. Each agent has an LLM powering it.
#
#   Task: A specific job assigned to one agent. Has a description (what to do),
#     expected_output (what format to return), context (previous task outputs
#     it can see), and an optional callback (function called when done).
#
#   Crew: The team. Takes a list of agents and tasks, a process type
#     (sequential = one after another, hierarchical = manager delegates),
#     and runs them when you call crew.kickoff().
#
#   Process.sequential: Tasks run in order. Task 2 starts only after Task 1
#     finishes. Task 2 can see Task 1's output via context=[task1].
# ──────────────────────────────────────────────────────────────────────────────


def run_support_crew(
    query: str,
    containers: dict,
    groq_api_key: str,
    vector_store: FAISS,
) -> dict:
    """
    Build and run the 4-agent customer support crew.

    Sequence:
        Classifier → Researcher → Writer → Quality Checker

    Each task has a callback that updates the Streamlit container for that
    agent as soon as it finishes. Because callbacks run in the same thread
    as crew.kickoff(), Streamlit's websocket pushes the update to the browser
    immediately — this is the "live output" behaviour.

    Args:
        query:        The customer's support message
        containers:   Dict of st.empty() placeholders keyed by agent name
        groq_api_key: From st.secrets["GROQ_API_KEY"]
        vector_store: FAISS index built from the TechFlow FAQ

    Returns:
        Dict with keys "classifier", "researcher", "writer", "qc"
        containing the raw text output of each agent
    """

    # ── LLM Setup ─────────────────────────────────────────────────────────────
    # All 4 agents share the same model (Llama 3.3 70B via Groq) but can have
    # different temperature settings.
    #
    # temperature=0.1 for most agents: low temperature = factual, consistent.
    # The Writer gets slightly higher temperature for more natural prose.
    #
    # In CrewAI 0.28.8, you pass a LangChain chat model to Agent's llm= param.
    # CrewAI wraps it in its own reasoning loop internally.
    llm_factual = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.1,
    )
    llm_writer = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.4,   # Slightly more creative for empathetic writing
    )

    # ── RAG Tool ──────────────────────────────────────────────────────────────
    # Built from the FAISS index. Only the Researcher agent gets this tool.
    # Agents without tools just use their LLM directly (no tool-call loop).
    rag_tool = build_rag_tool(vector_store)

    # ── Captured Outputs ──────────────────────────────────────────────────────
    # Task callbacks update this dict AND the Streamlit containers.
    # After crew.kickoff() finishes, we return this dict to app.py.
    captured = {
        "classifier": "",
        "researcher": "",
        "writer":     "",
        "qc":         "",
    }

    # ──────────────────────────────────────────────────────────────────────────
    # AGENT 1: CLASSIFIER
    #
    # role: The job title. Shown in CrewAI's verbose output.
    # goal: What this agent is trying to achieve. Shapes its decisions.
    # backstory: Background and expertise. Makes the agent respond in character.
    #   Think of it like a system prompt for the agent's personality.
    # llm: The language model powering this agent.
    # allow_delegation: If True, this agent can hand tasks to other agents.
    #   We set False for all agents — we control the flow explicitly via Tasks.
    # max_iter: Maximum reasoning iterations before the agent gives up and
    #   returns its best answer. Prevents infinite loops on complex queries.
    # verbose: If True, prints every Thought/Action/Observation to stdout.
    #   We set False to keep Streamlit Cloud logs clean — we use callbacks instead.
    # ──────────────────────────────────────────────────────────────────────────
    classifier_agent = Agent(
        role="Senior Customer Support Classifier",
        goal=(
            "Accurately categorise incoming customer support queries by type and urgency "
            "so the support team can route and prioritise responses correctly."
        ),
        backstory=(
            "You are TechFlow's most experienced support analyst with 7 years of triaging "
            "customer tickets. You have seen every type of issue — from billing disputes "
            "to complex integration failures — and you can immediately identify the "
            "category, urgency, and emotional state of a customer from their message. "
            "Your classifications directly determine how quickly and how the team responds."
        ),
        llm=llm_factual,
        allow_delegation=False,
        max_iter=3,
        verbose=False,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # AGENT 2: RESEARCHER
    #
    # The only agent with a tool. It uses the TechFlow FAQ search tool to find
    # relevant documentation. CrewAI's ReAct loop gives it the ability to:
    #   - Decide what to search for
    #   - Call the tool with a query
    #   - Read the results
    #   - Decide if it needs another search (up to max_iter times)
    #   - Synthesise the findings
    # ──────────────────────────────────────────────────────────────────────────
    researcher_agent = Agent(
        role="TechFlow Knowledge Base Specialist",
        goal=(
            "Find the most accurate and complete information from TechFlow's documentation "
            "to help the Writer craft a factually correct response to the customer's issue."
        ),
        backstory=(
            "You are TechFlow's resident expert on every policy, feature, and procedure. "
            "You know the exact refund policy, the precise steps to fix any integration issue, "
            "and every plan difference by heart. When you receive a classified support ticket, "
            "you search the knowledge base thoroughly and compile only the facts that are "
            "directly relevant — no noise, no guessing."
        ),
        llm=llm_factual,
        tools=[rag_tool],     # <-- ONLY this agent has the search tool
        allow_delegation=False,
        max_iter=5,           # Higher because it may need multiple searches
        verbose=False,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # AGENT 3: WRITER
    #
    # Receives context from both the Classifier (tone guidance) and the
    # Researcher (facts to include). Writes the actual customer-facing response.
    # Higher temperature because support writing benefits from natural language.
    # ──────────────────────────────────────────────────────────────────────────
    writer_agent = Agent(
        role="Senior Customer Support Writer",
        goal=(
            "Write professional, empathetic, and solution-focused customer support "
            "responses that resolve issues clearly and leave customers feeling valued."
        ),
        backstory=(
            "You are TechFlow's head of customer communications with a reputation for "
            "turning frustrated customers into loyal advocates. You have a gift for "
            "striking the perfect balance between professional and genuinely human. "
            "You never use corporate jargon. You acknowledge problems directly, "
            "provide clear steps, and always close on a reassuring note. "
            "Your responses consistently achieve over 90% customer satisfaction scores."
        ),
        llm=llm_writer,
        allow_delegation=False,
        max_iter=3,
        verbose=False,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # AGENT 4: QUALITY CHECKER
    #
    # Reviews the Writer's draft. Does NOT rewrite — just assesses and flags.
    # This is the "flag only" approach chosen in the project brief.
    # app.py reads the QC output for the word "APPROVED" to set the UI colour.
    # ──────────────────────────────────────────────────────────────────────────
    qc_agent = Agent(
        role="Customer Support Quality Assurance Lead",
        goal=(
            "Ensure every customer support response meets TechFlow's quality standards "
            "for tone, accuracy, completeness, and compliance before it is sent."
        ),
        backstory=(
            "You are TechFlow's QA lead responsible for maintaining a 95% CSAT score. "
            "You review every response against four criteria: empathetic tone, factual "
            "accuracy, completeness (all issues addressed), and compliance (no promises "
            "that can't be kept, no inappropriate disclosures). You are thorough but fair — "
            "you approve good responses quickly and only flag genuine issues."
        ),
        llm=llm_factual,
        allow_delegation=False,
        max_iter=3,
        verbose=False,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # TASK CALLBACK HELPERS
    #
    # These functions are called by CrewAI when each task completes.
    # They do two things:
    #   1. Extract the text from the output (handle string or TaskOutput object)
    #   2. Update the Streamlit container for that agent
    #
    # Why closures?
    # Each callback captures its specific container from the containers dict.
    # This is the standard Python closure pattern — the function "closes over"
    # the variable from the enclosing scope.
    #
    # Why does updating containers work live?
    # crew.kickoff() runs in the main Streamlit thread. When a callback calls
    # container.markdown(), Streamlit sends a websocket delta to the browser
    # immediately — same mechanism as progress bars and spinners.
    # ──────────────────────────────────────────────────────────────────────────

    def extract_text(output) -> str:
        """
        Safely extract string text from CrewAI task output.
        In crewai 0.28.8, callback receives a raw string.
        In newer versions it may receive a TaskOutput object.
        This function handles both.
        """
        if hasattr(output, "raw_output"):
            return str(output.raw_output)
        if hasattr(output, "result"):
            return str(output.result)
        return str(output)

    def on_classifier_done(output):
        text = extract_text(output)
        captured["classifier"] = text
        containers["classifier"].markdown(
            f"**Classification complete:**\n\n{text}"
        )

    def on_researcher_done(output):
        text = extract_text(output)
        captured["researcher"] = text
        containers["researcher"].markdown(
            f"**Research findings:**\n\n{text}"
        )

    def on_writer_done(output):
        text = extract_text(output)
        captured["writer"] = text
        containers["writer"].success(
            f"**Draft response:**\n\n{text}"
        )

    def on_qc_done(output):
        text = extract_text(output)
        captured["qc"] = text
        # Colour the QC box based on verdict
        if "APPROVED" in text.upper():
            containers["qc"].success(f"**Quality Assessment:**\n\n{text}")
        else:
            containers["qc"].warning(f"**Quality Assessment (Issues Flagged):**\n\n{text}")

    # ──────────────────────────────────────────────────────────────────────────
    # TASK DEFINITIONS
    #
    # description: Exact instructions for the agent.
    # expected_output: The format the agent should return.
    #   CrewAI uses this to evaluate whether the task is complete.
    # agent: Which agent does this task.
    # context: List of previous tasks whose outputs this agent can see.
    #   CrewAI prepends context outputs to the task prompt automatically.
    # callback: Function called when the task finishes.
    # ──────────────────────────────────────────────────────────────────────────

    task_classify = Task(
        description=(
            f"Analyse the following customer support query and produce a structured classification.\n\n"
            f"CUSTOMER QUERY:\n{query}\n\n"
            f"Your classification must include:\n"
            f"1. CATEGORY — choose exactly one:\n"
            f"   - billing: charges, invoices, refunds, payments, plan changes\n"
            f"   - technical: features not working, integration failures, sync issues, bugs\n"
            f"   - complaint: expressed frustration, threats to cancel, escalations, dissatisfaction\n"
            f"   - general: product questions, plan comparisons, how-to, feature requests\n"
            f"2. URGENCY — choose exactly one:\n"
            f"   - high: financial impact (wrong charges), data loss, system down, cancellation threat\n"
            f"   - medium: important feature broken, integration issue, upgrade/downgrade request\n"
            f"   - low: general questions, minor inconveniences, feature curiosity\n"
            f"3. KEY ISSUES: bullet list of the specific problems or questions raised\n"
            f"4. CUSTOMER SENTIMENT: one word (frustrated / angry / neutral / curious / confused)\n"
            f"5. TONE GUIDANCE: one sentence on how the Writer should approach this response\n"
        ),
        expected_output=(
            "A structured classification with clearly labelled sections: "
            "CATEGORY, URGENCY, KEY ISSUES (bullet list), "
            "CUSTOMER SENTIMENT, and TONE GUIDANCE."
        ),
        agent=classifier_agent,
        callback=on_classifier_done,
    )

    task_research = Task(
        description=(
            f"Search the TechFlow knowledge base to find all information relevant to this customer query.\n\n"
            f"ORIGINAL CUSTOMER QUERY:\n{query}\n\n"
            f"You have access to the classification from the previous agent (see context).\n"
            f"Use the TechFlow Knowledge Base Search tool to find relevant documentation.\n"
            f"Make multiple targeted searches to cover all aspects of the issue.\n\n"
            f"Your research output must include:\n"
            f"1. RELEVANT SECTIONS FOUND: which parts of the documentation apply\n"
            f"2. KEY FACTS: specific facts, prices, timeframes, or procedures relevant to this issue\n"
            f"3. RESOLUTION STEPS: exact steps the customer should take (if technical issue)\n"
            f"4. APPLICABLE POLICIES: any refund, billing, or support policies that apply\n"
            f"5. SUPPORT CONTACTS: relevant email addresses or links from the documentation\n"
        ),
        expected_output=(
            "A research summary with: RELEVANT SECTIONS FOUND, KEY FACTS, "
            "RESOLUTION STEPS, APPLICABLE POLICIES, and SUPPORT CONTACTS."
        ),
        agent=researcher_agent,
        context=[task_classify],   # Researcher sees the Classifier's output
        callback=on_researcher_done,
    )

    task_write = Task(
        description=(
            f"Write a complete professional customer support response for this query.\n\n"
            f"ORIGINAL CUSTOMER QUERY:\n{query}\n\n"
            f"You have the classification (tone guidance, urgency) and research findings "
            f"from the previous agents (see context). Use both to craft your response.\n\n"
            f"Requirements:\n"
            f"- Open with genuine empathy — acknowledge the specific issue by name\n"
            f"- Address EVERY issue raised in the query\n"
            f"- Use the specific facts from the research (plan names, prices, timeframes, steps)\n"
            f"- Provide clear numbered steps where the customer needs to take action\n"
            f"- Professional but warm tone — no corporate jargon\n"
            f"- Include relevant contact details (email addresses) where appropriate\n"
            f"- Close with a reassuring, helpful statement\n"
            f"- Sign off as: TechFlow Customer Support Team\n"
        ),
        expected_output=(
            "A complete customer support response including a SUBJECT LINE and "
            "full RESPONSE BODY with greeting, issue acknowledgement, resolution "
            "or information, clear next steps, and professional closing."
        ),
        agent=writer_agent,
        context=[task_classify, task_research],  # Writer sees both previous outputs
        callback=on_writer_done,
    )

    task_qc = Task(
        description=(
            f"Perform a quality assurance review of the draft customer support response.\n\n"
            f"ORIGINAL CUSTOMER QUERY:\n{query}\n\n"
            f"Review the Writer's draft response (see context) against these criteria:\n\n"
            f"1. TONE: Is it empathetic, professional, and non-defensive?\n"
            f"2. ACCURACY: Are the facts correct and consistent with TechFlow's documentation?\n"
            f"3. COMPLETENESS: Does it address ALL issues raised in the original query?\n"
            f"4. CLARITY: Are next steps clear and actionable?\n"
            f"5. COMPLIANCE: Are there any promises that cannot realistically be kept?\n\n"
            f"For each criterion, write PASS or NEEDS IMPROVEMENT with a brief note.\n"
            f"Then give an OVERALL VERDICT: APPROVED or NEEDS REVISION.\n"
            f"Include a QUALITY SCORE out of 10.\n"
        ),
        expected_output=(
            "A QA assessment with individual criterion scores (PASS/NEEDS IMPROVEMENT), "
            "an OVERALL VERDICT (APPROVED or NEEDS REVISION), "
            "a QUALITY SCORE /10, and any specific QA notes."
        ),
        agent=qc_agent,
        context=[task_write],   # QC only needs to see the Writer's draft
        callback=on_qc_done,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # CREW ASSEMBLY
    #
    # Crew(agents, tasks, process, verbose)
    #
    # agents: All agents that are part of this crew.
    # tasks: All tasks in execution order (sequential = run in list order).
    # process: Process.sequential means task 1 → task 2 → task 3 → task 4.
    #   Each task waits for the previous one to finish before starting.
    # verbose: 0 = no stdout logging (we use callbacks for UI output instead).
    #   Setting this to 2 would print every thought/action to the terminal —
    #   useful for debugging locally but noisy on Streamlit Cloud.
    # ──────────────────────────────────────────────────────────────────────────
    crew = Crew(
        agents=[classifier_agent, researcher_agent, writer_agent, qc_agent],
        tasks=[task_classify,     task_research,    task_write,   task_qc],
        process=Process.sequential,
        verbose=0,
    )

    # ── Run the crew ──────────────────────────────────────────────────────────
    # crew.kickoff() starts the sequential execution.
    # It blocks until all 4 tasks complete.
    # During execution, task callbacks fire and update the Streamlit containers.
    # The return value is the final task's output (QC output in our case).
    crew.kickoff()

    return captured
