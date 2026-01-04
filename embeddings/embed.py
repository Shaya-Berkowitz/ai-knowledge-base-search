"""
Generates text embeddings using OpenAI's text-embedding-3-small model.
"""

from openai import OpenAI
from core.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text: str) -> list[float]:
    """
    Generate an embedding for the given text.

    Args:
        text (str): The input text to embed.

    Returns:
        list[float]: The 1536-embedding vector. 

        Note:
        Assumes the text has already been normalized by the ingestion
        pipeline
    """
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    # OpenAI returns a list of embeddings. Since we only sent one input,
    # get the first embedding vector.
    return resp.data[0].embedding
