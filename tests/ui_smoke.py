import os, tempfile
d = tempfile.mkdtemp()
os.environ.update(STUDYMATE_EMBEDDER="hashing", GENERATION_MODE="retrieval_only", MIN_SCORE="0.05",
                  STUDYMATE_CHROMA_DIR=d+"/c", STUDYMATE_DOCUMENTS_DIR=d+"/d")
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "app.py"), default_timeout=60).run()
assert not at.exception, at.exception
[b for b in at.sidebar.button if b.label=="Load sample notes"][0].click().run()
print("success msgs:", [s.value for s in at.sidebar.success])
at.chat_input[0].set_value("Explain gradient descent learning rate").run()
assert not at.exception, at.exception
md = " ".join(m.value for m in at.chat_message[1].markdown)
print(md[:400])
assert "machine_learning.md" in md
# MCP tab buttons
at.text_input(key="mcp_query").set_value("overfitting dropout").run()
[b for b in at.button if b.label=="Run search_notes"][0].click().run()
assert not at.exception and "machine_learning.md" in at.text[-1].value
# duplicate load
[b for b in at.sidebar.button if b.label=="Load sample notes"][0].click().run()
print("dup msgs:", [s.value for s in at.sidebar.info])
print("UI OK")
