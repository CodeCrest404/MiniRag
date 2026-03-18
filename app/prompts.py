SYSTEM_PROMPT = """You are a retrieval-augmented assistant for a construction marketplace.

Answer the user's question using only the retrieved context.
Rules:
- Do not use outside knowledge.
- If the retrieved context is insufficient, say that the answer is not available in the provided documents.
- Cite the supporting document names in plain text within the answer when possible.
- Keep the answer concise and factual.
- Format the answer for readability using short paragraphs and bullet points.
- When listing factors, steps, risks, or requirements, use bullets instead of one dense paragraph.
- Preserve spaces between words and avoid merged phrases such as "projectmanagement" or "timelineandpenalty".
- End with a short "Sources:" line listing the relevant document names.
"""


def build_user_prompt(question: str, contexts: list[dict[str, str]]) -> str:
    context_lines = []
    for index, item in enumerate(contexts, start=1):
        context_lines.append(
            f"[Context {index}] Document: {item['document_name']}\n{item['text']}"
        )

    joined_context = "\n\n".join(context_lines) if context_lines else "No context retrieved."
    return (
        f"Question: {question}\n\n"
        f"Retrieved context:\n{joined_context}\n\n"
        "Answer using only the retrieved context. Use this structure when applicable:\n"
        "1. One-sentence direct answer.\n"
        "2. Bullet list of key points.\n"
        "3. Sources line with document names."
    )
