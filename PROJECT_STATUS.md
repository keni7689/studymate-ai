# PROJECT_STATUS

## Completed phases
1-13 complete (setup, loader, chunker, embeddings, ChromaDB, retrieval, local LLM code, RAG pipeline,
Streamlit UI, MCP server, tests, README, GitHub readiness). Phase 14 (deployment) investigated: local recommended.

## Current / remaining
Nothing remaining except the manual checks below (they need internet + your PC).

## Tests completed (in the build sandbox, offline embedder)
- `pytest`: 39 passed (loader, chunker, embeddings API, ChromaDB persistence, retrieval, RAG pipeline,
  fallbacks, duplicates, MCP tools, real MCP server over stdio).
- `python tests/ui_smoke.py`: headless Streamlit run (load samples, ask, MCP tab, duplicates) OK.
- `pip check`: no conflicts; pip resolved the full requirements.txt.

## NOT tested in the sandbox (no Hugging Face access / small disk)
- Real `all-MiniLM-L6-v2` download + embedding quality
- Real `Qwen2.5-0.5B-Instruct` download + generation
- Windows-specific commands
Manual test: `pip install -r requirements.txt`, `streamlit run app.py`, click "Load sample notes", ask
"Explain gradient descent simply". If answers say "couldn't find", lower MIN_SCORE (e.g. `set MIN_SCORE=0.1`).

## Known issues
- MIN_SCORE (0.2) was chosen from experience with MiniLM, not measured here. Tune if needed.
- First run downloads ~90 MB (embeddings) + ~1 GB (LLM).
