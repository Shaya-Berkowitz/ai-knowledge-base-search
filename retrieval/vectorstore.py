"""
Vector store abstraction for Pinecone.

Provides a clean interface so the rest of the system
doesn't depend directly on Pinecone client specifics.
"""

from typing import Any, Dict, List, Optional
from pinecone import Pinecone
from core.config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from retrieval.models import Chunk


class VectorStoreClient:
    """Wrapper around Pinecone index."""

    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)

    def query(
        self,
        vector: List[float],
        top_k: int = 8,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Query Pinecone for nearest neighbors.

        Args:
            vector (List[float]): Embedding vector of the query.
            top_k (int): Number of results to return.
            metadata_filter ([Dict[str, Any]): Optional Pinecone metadata filter.

        Returns:
            List[Chunk]: List of Chunk objects.
        """

        # Perform the query using Pinecone client
        response = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=metadata_filter or {}
        )

        chunks: List[Chunk] = []

        # Process the response and convert to Chunk objects
        for match in response.matches:
            md = match.metadata or {}
            chunks.append(
                Chunk(
                    id=match.id,
                    text=md.get("text", ""),
                    metadata=md,
                    score=match.score
                )
            )

        return chunks
