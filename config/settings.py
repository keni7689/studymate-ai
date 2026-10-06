"""All StudyMate AI settings live here.

Every value can be overridden with an environment variable, e.g. (Windows):
    set TOP_K=6
    streamlit run app.py
No API keys are needed anywhere.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Storage ---
DOCUMENTS_DIR = Path(os.getenv("STUDYMATE_DOCUMENTS_DIR", BASE_DIR / "data" / "documents"))
CHROMA_DIR = Path(os.getenv("STUDYMATE_CHROMA_DIR", BASE_DIR / "data" / "chroma"))
COLLECTION_NAME = "study_notes"
SAMPLE_NOTES_DIR = BASE_DIR / "sample_notes"
SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf")

# --- Chunking (measured in characters) ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "4"))
# Chunks less similar than this (0..1, cosine) are ignored -> "not found in notes"
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.2"))

# --- Models (downloaded once from Hugging Face, then cached locally) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Small instruct model, ~1 GB RAM on CPU. Alternative: "Qwen/Qwen2.5-1.5B-Instruct" (better, ~3-4 GB)
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "300"))

# "local"          -> generate answers with the local Transformers model
# "retrieval_only" -> no LLM; just show the best matching passages (very low RAM)
GENERATION_MODE = os.getenv("GENERATION_MODE", "local")
