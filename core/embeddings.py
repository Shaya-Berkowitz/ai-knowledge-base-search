"""
Shared embedding utilities.

Used by both ingestion and retrieval
to ensure a consistent embedding model and vector space.
"""

from typing import List
from openai import OpenAI
from core.config import OPENAI_API_KEY, EMBEDDING_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts.

    Args:
        texts: List of text strings.

    Returns:
        List of embedding vectors, one per text.
    """
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    return [d.embedding for d in resp.data]


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.

    Args:
        query: The query text.

    Returns:
        The embedding vector for the query.
    """
    return embed_texts([query])[0]
