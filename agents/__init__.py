# Makes agents/ a Python package so app.py can do:
#   from agents.rag_tool import get_vector_store
#   from agents.crew_builder import run_support_crew
# Without this file Python treats agents/ as a plain folder, not an importable package.
