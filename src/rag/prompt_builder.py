def build_prompt(query: str, context_docs: list[str]) -> str:
    context = "\n---\n".join(context_docs)

    return f"""Responde a la pregunta usando SOLO el contexto proporcionado.
Si no encuentras la respuesta, di: "No tengo suficiente información".

Contexto:
{context}

Pregunta:
{query}

Respuesta clara y concisa:"""
