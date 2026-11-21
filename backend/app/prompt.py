from __future__ import annotations


def build_prompt(context: str, question: str, history: list[dict] | None = None) -> str:
    # Write the recent conversation as plain text (chat memory).
    conversation = ""
    for turn in history or []:
        conversation += f"User: {turn.get('question', '')}\n"
        conversation += f"Assistant: {turn.get('answer', '')}\n\n"

    if conversation:
        memory_part = f"""
Conversation so far (use it only to understand what the new question
refers to, e.g. "it", "that course", "what about the books?"):
{conversation}"""
    else:
        memory_part = ""

    return f"""
You are an intelligent assistant.

Answer the question strictly using the context below.
If the answer is not present in the context, say:
\"I do not know based on the provided documents.\"
{memory_part}
Context:
{context}

Question:
{question}

Answer:
"""
