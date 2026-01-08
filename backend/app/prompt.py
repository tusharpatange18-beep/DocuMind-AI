from __future__ import annotations


def build_prompt(context: str, question: str) -> str:
    return f"""
You are an intelligent assistant.

Answer the question strictly using the context below.
If the answer is not present in the context, say:
\"I do not know based on the provided documents.\"

Context:
{context}

Question:
{question}

Answer:
"""
