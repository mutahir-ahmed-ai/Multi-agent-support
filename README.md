# 🤖 TechFlow Multi-Agent Customer Support System

An AI-powered customer support pipeline where four specialized agents work in sequence — classify, research, write, and quality-check — to handle any customer query. Built with CrewAI and deployed on Streamlit Cloud.

🚀 Live Demo: https://multi-agent-support-zvple7yytkvntyrw2degzg.streamlit.app/

---

## What It Does

Type (or select) a customer support query → four agents process it in sequence → a professional, researched, quality-checked response is generated. Every agent's output is visible in the UI as it completes.

**Example query:** *"I've been charged twice this month for my Professional plan. I need a refund immediately."*

The crew processes it:
1. **Classifier** → `category: billing | urgency: high | sentiment: frustrated`
2. **Researcher** → finds duplicate charge policy, refund procedure, 3-5 day timeline, billing contact
3. **Writer** → drafts a full empathetic email acknowledging the issue, explaining next steps, providing billing@techflow.io
4. **Quality Checker** → scores 9/10, APPROVED — tone empathetic, all issues addressed, steps clear

---

## Architecture

```
Customer Query
      ↓
┌─────────────────┐
│  Agent 1        │  Role: Classifier
│  Classifier     │  Reads query → category + urgency + sentiment + tone guidance
└────────┬────────┘
         ↓ context passed
┌─────────────────┐
│  Agent 2        │  Role: Researcher
│  Researcher     │  Searches FAISS knowledge base (TechFlow FAQ) → key facts + steps
└────────┬────────┘
         ↓ context passed
┌─────────────────┐
│  Agent 3        │  Role: Writer
│  Writer         │  Classification + Research → professional support email
└────────┬────────┘
         ↓ context passed
┌─────────────────┐
│  Agent 4        │  Role: Quality Checker
│  QC Lead        │  Reviews draft → tone / accuracy / completeness → APPROVED or flagged
└─────────────────┘
         ↓
  Final Response displayed in UI
```

### How CrewAI Differs From a Single Agent

In Project 4 (Research Agent), one agent used a ReAct loop to search and synthesize. Here, each agent is a specialist:

| Approach | Pattern |
|---|---|
| Single Agent (Project 4) | One LLM, many tools, loop until done |
| Multi-Agent Crew (Project 6) | Four LLMs, each expert at one job, output passes to next |

The multi-agent approach produces better results because each agent's prompt is laser-focused on one task. The Writer doesn't also have to classify. The Classifier doesn't also have to write.

---

## Tech Stack

| Component | Tool |
|---|---|
| Agent Orchestration | CrewAI 0.28.8 |
| LLM | Llama 3.3 70B via Groq API |
| Knowledge Base | FAISS + HuggingFace all-MiniLM-L6-v2 |
| RAG Framework | LangChain (Document, TextSplitter, Tool) |
| UI + Deployment | Streamlit + Streamlit Cloud |

---

## Project Structure

```
multi-agent-support/
│
├── agents/
│     ├── __init__.py          ← makes agents/ importable as a package
│     ├── rag_tool.py          ← TechFlow FAQ + FAISS index + LangChain Tool
│     └── crew_builder.py      ← 4 Agents, 4 Tasks, 1 Crew
│
├── app.py                     ← Streamlit UI + session state + live containers
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Deployment (Streamlit Cloud)

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/mutahir-ahmed-ai/multi-agent-support.git
git push -u origin main
```

### 2. Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
2. Repository: `mutahir-ahmed-ai/multi-agent-support`
3. Main file: `app.py`
4. **Advanced settings → Secrets:**

```toml
GROQ_API_KEY = "your_groq_key_here"
```

5. Click **Deploy** — first build takes ~3-4 minutes

### Note on CrewAI Version

This project pins `crewai==0.28.8`. CrewAI changes its API between versions. If you see import errors after deployment, verify the pinned version is being used. Do not upgrade without testing.

---

## Run Locally

```bash
git clone https://github.com/mutahir-ahmed-ai/multi-agent-support
cd multi-agent-support
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_key_here"
```

```bash
streamlit run app.py
```

---

## Key Technical Decisions

**Why CrewAI instead of LangChain AgentExecutor?**
CrewAI is built for multi-agent workflows. Each agent has a distinct role, goal, and backstory — this produces more focused, higher-quality outputs than a single agent trying to do everything.

**Why hardcode the FAQ instead of user-uploaded PDFs?**
Customer support knowledge bases are typically static and company-controlled. A hardcoded FAQ ensures the agents always have consistent, reliable context without requiring upload steps.

**Why flag-only Quality Checker (no revision loop)?**
A revision loop adds significant latency and token cost. For a Streamlit demo, showing the QC assessment alongside the draft is more transparent and instructive — the user can see exactly what the QC flagged.

**Why pin crewai==0.28.8?**
CrewAI releases frequently and changes its API between versions. Pinning ensures deployment stability. The Task.callback parameter used for live UI updates exists in 0.28.8 and was removed/changed in later versions.


---

## Author

**Mutahir Ahmed** — AI Developer | Multi-Agent Systems & RAG
[LinkedIn](https://www.linkedin.com/in/mutahir-ahmed-8229341b5/) · [GitHub](https://github.com/mutahir-ahmed-ai)
