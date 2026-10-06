"""Prompts sent to the local model."""

NOT_FOUND_MESSAGE = "I couldn't find enough information about this in your uploaded study material."

SYSTEM_PROMPT = f"""You are StudyMate, a friendly tutor.
Answer the student's question using ONLY the study notes in the CONTEXT.
Rules:
- Explain simply, in short clear sentences. Add a small example if it helps.
- Do not invent facts that are not in the context.
- If the context does not contain the answer, reply exactly: "{NOT_FOUND_MESSAGE}"
"""

QUIZ_SYSTEM_PROMPT = """You are StudyMate, a quiz maker.
Write quiz questions using ONLY the study notes in the CONTEXT.
Number the questions. After each question write "Answer:" with a short answer taken from the notes.
Do not use facts that are not in the context."""


def build_context(chunks: list[dict]) -> str:
    """Join chunks into one text block, labelled so the model can tell them apart."""
    parts = []
    for number, chunk in enumerate(chunks, start=1):
        page = f", page {chunk['page']}" if chunk["page"] else ""
        parts.append(f"[Note {number} - {chunk['source']}{page}]\n{chunk['text']}")
    return "\n\n".join(parts)


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    user_text = f"CONTEXT:\n{build_context(chunks)}\n\nQUESTION: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}]


def build_quiz_messages(topic: str, num_questions: int, chunks: list[dict]) -> list[dict]:
    user_text = (
        f"CONTEXT:\n{build_context(chunks)}\n\n"
        f"Write {num_questions} quiz questions about: {topic}"
    )
    return [{"role": "system", "content": QUIZ_SYSTEM_PROMPT}, {"role": "user", "content": user_text}]
