def build_prompt(query: str, context_docs: list[str]) -> str:
    context = "\n".join(context_docs)

    return f"""Eres un asistente que responde usando solo el contexto proporcionado.

Contexto:
{context}

Pregunta:
{query}

Respuesta:"""
