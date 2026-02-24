def build_prompt(query, docs):
    ctx = "\n---\n".join(docs)
    return f"""Responde usando SOLO el contexto.

Contexto:
{ctx}

Pregunta:
{query}

Respuesta:"""
