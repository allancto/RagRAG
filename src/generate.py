"""Response generation using Claude API."""

import anthropic
from config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_MAX_TOKENS


SYSTEM_PROMPT = """You are a helpful assistant that answers questions about building RAG (Retrieval-Augmented Generation) systems.

Use the provided context to answer questions accurately. When citing information:
- Reference the source numbers [1], [2], etc. when applicable
- Be specific and technical when the context supports it
- Acknowledge limitations in the provided context

If the context doesn't contain enough information to answer the question, say so clearly and explain what additional information would be needed."""


def build_prompt(query: str, context: str) -> str:
    """
    Build the user prompt with context and query.

    Args:
        query: User's question
        context: Formatted context from retrieval

    Returns:
        Complete prompt string
    """
    if not context:
        return f"""Question: {query}

Note: No relevant context was found in the knowledge base. Please answer based on general knowledge, but indicate that this is not from the RAG corpus."""

    return f"""Context from knowledge base:
{context}

---

Question: {query}

Please answer based on the context above. Cite sources using [1], [2], etc. when referencing specific information."""


def generate_response(query: str, context: str) -> str:
    """
    Generate a response using Claude API.

    Args:
        query: User's question
        context: Formatted context from retrieval

    Returns:
        Generated response text
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = build_prompt(query, context)

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text
