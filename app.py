"""StudyMate AI - Streamlit UI.  Run with:  streamlit run app.py"""
from pathlib import Path

import streamlit as st

from config import settings
from mcp_server import tools
from src import rag_pipeline

st.set_page_config(page_title="StudyMate AI", page_icon="📚", layout="wide")
st.title("📚 StudyMate AI")
st.caption("Your personal RAG tutor - runs 100% locally, no API key.")

settings.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


def show_source_list(sources: list[dict]) -> None:
    if not sources:
        return
    st.markdown("**Sources:**")
    for source in sources:
        page = f" (page {source['page']})" if source["page"] else ""
        st.markdown(f"- {source['source']}{page} - similarity {source['score']:.2f}")


def index_files(paths: list[Path]) -> None:
    with st.spinner("Creating embeddings... (the first run downloads the embedding model once)"):
        for path in paths:
            result = rag_pipeline.add_document(path)
            {"added": st.success, "duplicate": st.info, "error": st.error}[result["status"]](result["message"])


# ---------------- Sidebar: knowledge base ----------------
with st.sidebar:
    st.header("1. Upload notes")
    uploads = st.file_uploader("PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Add to Knowledge Base", type="primary", disabled=not uploads):
        saved = []
        for upload in uploads:
            target = settings.DOCUMENTS_DIR / Path(upload.name).name  # Path().name blocks ../ tricks
            target.write_bytes(upload.getvalue())
            saved.append(target)
        index_files(saved)

    if st.button("Load sample notes"):
        index_files(sorted(settings.SAMPLE_NOTES_DIR.glob("*")))

    st.header("Knowledge base")
    try:
        documents = rag_pipeline.list_documents()
    except Exception as error:  # e.g. corrupted database folder
        documents = {}
        st.error(f"Database problem: {error}")
    if documents:
        for name, chunk_count in documents.items():
            st.write(f"📄 {name} - {chunk_count} chunks")
    else:
        st.write("Empty. Upload notes or load the samples.")

    confirm = st.checkbox("Yes, delete all indexed notes")
    if st.button("Clear knowledge base", disabled=not confirm):
        rag_pipeline.reset_knowledge_base()
        st.session_state.pop("messages", None)
        st.rerun()

    st.caption(f"Generation mode: `{settings.GENERATION_MODE}` | Model: `{settings.LLM_MODEL}`")

# ---------------- Chat ----------------
st.header("2. Ask a question")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_source_list(message.get("sources", []))

question = st.chat_input("e.g. Explain gradient descent in simple terms")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching your notes and thinking... (first answer loads the model)"):
            result = rag_pipeline.ask(question)
        st.markdown(result["answer"])
        show_source_list(result["sources"])
        if result["chunks"]:
            with st.expander("Retrieved text used as context"):
                for chunk in result["chunks"]:
                    st.caption(f"{chunk['source']} (similarity {chunk['score']:.2f})")
                    st.text(chunk["text"])
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )

# ---------------- MCP tools ----------------
with st.expander("🔧 MCP tools (the same functions the MCP server exposes)"):
    st.caption("Run the server for other apps with:  python -m mcp_server.server")
    tab_search, tab_topic, tab_quiz = st.tabs(["search_notes", "get_topic", "generate_quiz"])
    with tab_search:
        query = st.text_input("Query", key="mcp_query")
        if st.button("Run search_notes"):
            st.text(tools.search_notes(query))
    with tab_topic:
        topic = st.text_input("Topic", key="mcp_topic")
        if st.button("Run get_topic"):
            st.text(tools.get_topic(topic))
    with tab_quiz:
        quiz_topic = st.text_input("Quiz topic", key="mcp_quiz_topic")
        count = st.slider("Number of questions", 1, 10, 5)
        if st.button("Run generate_quiz"):
            with st.spinner("Writing quiz..."):
                st.text(tools.generate_quiz(quiz_topic, count))
