# 📚 StudyMate AI — Free Local RAG + MCP Learning Assistant

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Persistent_Vector_Store-purple?style=for-the-badge)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MCP Standard](https://img.shields.io/badge/MCP-Model_Context_Protocol-00ADEE?style=for-the-badge)
![Pytest](https://img.shields.io/badge/Pytest-39_Tests_Passed-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

---

## 1. Project Title & Description

**StudyMate AI** is a privacy-first, fully local Retrieval-Augmented Generation (RAG) educational assistant and Model Context Protocol (MCP) server. 

It allows students and researchers to ingest their personal study notes (PDF, TXT, Markdown), index them into a local vector database, and ask natural language questions (e.g., *"Explain gradient descent in simple terms"*). StudyMate AI retrieves the exact relevant excerpts from your notes and feeds them to a local AI model running on your computer. 

### Why StudyMate AI?
* 🔒 **100% Private & Local**: Nothing leaves your computer. No data is sent to online LLM APIs.
* 💸 **Zero Cost & No API Keys**: Built without requiring OpenAI, Anthropic, or paid subscriptions.
* ⚡ **Standalone Execution**: Runs without external local LLM orchestrators like Ollama.
* 🎯 **Zero Hallucination Citations**: Direct page-level and file-level citations derived strictly from retrieved document metadata.
* 🔌 **MCP Integration**: Wrapped with Model Context Protocol standards so external AI tools (such as Claude Desktop or IDE assistants) can seamlessly query your notes.

---

## 2. Tech Stack

| Domain | Technology / Library | Description |
| :--- | :--- | :--- |
| **Language** | `Python 3.10 - 3.12` | Core application development language |
| **User Interface** | `Streamlit` | Interactive web frontend dashboard for note ingestion and AI interaction |
| **Vector Database** | `ChromaDB` | Embedded, persistent vector database for document embeddings |
| **Embeddings** | `Sentence-Transformers` | Uses `all-MiniLM-L6-v2` (384-dimensional dense vectors) for semantic matching |
| **Local LLM** | `Hugging Face Transformers` | Runs lightweight Hugging Face Instruct models (default `Qwen/Qwen2.5-0.5B-Instruct`) via `PyTorch` |
| **Protocol Integration** | `FastMCP` (`mcp` SDK) | Exposes standard MCP tools over STDIO for external AI clients |
| **PDF Processing** | `pypdf` | Extracts clean text page-by-page preserving document metadata |
| **Testing** | `Pytest` | Complete automated test suite (39 tests including headless UI smoke tests) |

---

## 3. Features

Here is what users can do within StudyMate AI:

* 📄 **Multi-Format Document Ingestion**: Upload study material in `.pdf`, `.txt`, and `.md` formats.
* 📑 **Page-Level Source Citations**: Trace every answer back to exact file names and PDF page numbers.
* 🔍 **Semantic Vector Search**: Retrieve contextually relevant note chunks even when queries use different wording or synonyms.
* 🧠 **Offline RAG Ingestion & QA**: Ask questions and generate answers using an offline LLM lazy-loaded on CPU/GPU.
* 🚫 **Duplicate Prevention**: Ingested files are hashed (`SHA-256`) to automatically detect and ignore repeated uploads.
* 📊 **Pre-loaded Sample Notes**: Instant one-click sample note loading for rapid demonstration and testing.
* 🛠️ **Built-in MCP Tools Suite**:
  * `search_notes(query, top_k)`: Returns top matching excerpts with relevance scores.
  * `get_topic(topic)`: Aggregates comprehensive notes on specific topics with references.
  * `generate_quiz(topic, num_questions)`: Creates custom quiz questions grounded in your study notes.
* 🛡️ **Graceful Fallbacks**: If the local LLM cannot load due to memory limits, the system falls back to pure retrieval mode or fill-in-the-blank quiz generation.

---

## 4. Process

### How StudyMate AI Was Built

StudyMate AI was engineered in structured phases to ensure high reliability, modularity, and zero vendor lock-in:

#### Architecture Overview
```text
User Question ──► Streamlit UI ──► RAG Pipeline ──► Retriever ──► ChromaDB
                                                                      │
Answer ◄── Local LLM (Qwen2.5) ◄── Grounded Prompt ◄── Top-K Chunks ◄──┘

                               MCP Server
             ┌─────────────────────┼─────────────────────┐
      search_notes              get_topic           generate_quiz
```

1. **Document Processing & Chunking**:
   * Extracted text page-by-page from PDFs using `pypdf`.
   * Developed a custom recursive chunker (`src/chunker.py`) dividing text into ~800-character segments with 150-character overlap to preserve semantic context across chunk boundaries.

2. **Vector Store & Indexing**:
   * Implemented sentence embeddings using `all-MiniLM-L6-v2` (`src/embeddings.py`).
   * Configured `ChromaDB` (`src/vector_store.py`) with cosine similarity metrics for persistent local storage in `data/chroma/`.

3. **Retrieval & Grounded Prompting**:
   * Engineered `retriever.py` to convert user queries into vectors, search ChromaDB, and enforce a minimum relevance threshold (`MIN_SCORE = 0.2`).
   * Built strict system prompts (`src/prompts.py`) compelling the LLM to answer *only* using provided note chunks or explicitly state when information is missing.

4. **Lazy-Loaded Local Inference**:
   * Integrated `Qwen/Qwen2.5-0.5B-Instruct` using Hugging Face `transformers` pipeline (`src/generator.py`), configured to lazy-load on first question to optimize startup time and memory footprint.

5. **MCP Server Construction**:
   * Wrapped the RAG core into an open standard Model Context Protocol server (`mcp_server/server.py`), allowing tools like Claude Desktop to access notes.

6. **UI & Quality Assurance**:
   * Built a Streamlit interface (`app.py`) featuring document management, citation viewers, and MCP testing panels.
   * Created a 39-test `pytest` suite with headless UI check script (`tests/ui_smoke.py`) using lightweight mock embedders for offline verification.

### What We Learned

* **Local LLM Trade-offs**: Lightweight models (like `Qwen2.5-0.5B`) require precise prompt engineering to prevent straying from context, but provide remarkably fast inference on ordinary CPUs.
* **Context Overlap Matters**: 150-character overlaps prevent sentence truncation at chunk boundaries, dramatically improving answer coherence.
* **Metadata Integrity**: Preserving original file paths and page indices during chunking is essential for generating trustworthy citations.
* **Protocol Decoupling**: Separating core RAG logic from presentation layers enabled running the same RAG pipeline via both Streamlit and MCP servers simultaneously.

---

## 5. Setup Instructions

### Prerequisites
* **OS**: Windows, macOS, or Linux
* **Python**: `Python 3.10` to `3.12` (Check with `python --version`)
* **Disk Space**: ~3 GB free space (for virtual environment & cached Hugging Face models)

### 1. Clone the Repository
```bat
git clone https://github.com/<your-username>/studymate-ai.git
cd studymate-ai
```

### 2. Create and Activate Virtual Environment
* **Windows (PowerShell / CMD)**:
  ```bat
  python -m venv venv
  venv\Scripts\activate
  ```
  *(If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` first)*

* **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bat
pip install -r requirements.txt
```
*(For testing tools, run `pip install -r requirements-dev.txt`)*

### 4. Run the Streamlit Application
```bat
streamlit run app.py
```
Open your browser at `http://localhost:8501`. 

> 💡 **First Run Note**: The first run will download model weights (~90 MB for embeddings, ~1 GB for LLM). Subsequent runs execute offline from cache.

---

### Running the MCP Server

To expose your study notes to MCP-compatible AI clients (e.g., Claude Desktop):

1. **Start the MCP Server**:
   ```bat
   python -m mcp_server.server
   ```

2. **Test via Demo Client**:
   ```bat
   python -m mcp_server.client_demo "gradient descent"
   ```

3. **Configure Claude Desktop** (`claude_desktop_config.json`):
   ```json
   {
     "mcpServers": {
       "studymate": {
         "command": "C:\\path\\to\\studymate-ai\\venv\\Scripts\\python.exe",
         "args": ["-m", "mcp_server.server"],
         "cwd": "C:\\path\\to\\studymate-ai"
       }
     }
   }
   ```

---

### Running Tests
```bat
pytest
python tests/ui_smoke.py
```

---

## 6. Improvements

Here are potential future enhancements and optimizations planned for StudyMate AI:

* 🔄 **Cross-Encoder Reranking**: Incorporate a cross-encoder model (e.g., `ms-marco-MiniLM-L-6-v2`) to re-rank retrieved chunks for higher retrieval precision.
* 💬 **Conversational Memory**: Add multi-turn dialog memory to support follow-up questions while maintaining retrieved context history.
* 🔀 **Hybrid Search (BM25 + Dense Vectors)**: Combine keyword-based BM25 search with dense vector similarity to improve retrieval on specific technical terms and code snippets.
* 🔍 **Semantic Chunking**: Replace fixed-character chunking with semantic boundaries based on sentence similarity or header hierarchy.
* 📷 **OCR Ingestion**: Add `pytesseract` or `easyocr` to extract text from scanned image PDFs and handwritten diagrams.
* ⚙️ **Multi-Model Selector**: Provide a UI dropdown allowing users to toggle between lightweight CPU models (`Qwen2.5-0.5B`) and higher-capacity models (`Qwen2.5-1.5B` or `Llama-3.2-1B`) based on system hardware.
