"""
Semantic retrieval engine.

Responsible for embedding the query, retrieving
relevant chunks, and, then, filtering and ranking them.
"""

from typing import Dict, Optional, Any
import logging

from core.embeddings import embed_query
from retrieval.vectorstore import VectorStoreClient
from retrieval.models import Chunk, SearchResult

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Semantic retriever using a vector store."""

    def __init__(self, store: Optional[VectorStoreClient] = None):
        self.store = store or VectorStoreClient()

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """
        Perform semantic retrieval for the given query.

        Args:
            query (str): User query text.
            top_k (int): Desired number of final chunks.
            score_threshold (float, optional): Minimum similarity score.
            metadata_filter (Dict[str, Any], optional): Optional Pinecone metadata filter.

        Returns:
            SearchResult: SearchResult containing ranked chunks.
        """

        logger.info(f"Retrieving chunks for query='{query}'")

        # Embed query
        query_vec = embed_query(query)

        # Query Pinecone
        raw_chunks = self.store.query(
            vector=query_vec,
            top_k=top_k * 2,  # Get extra to allow for filtering
            metadata_filter=metadata_filter
        )

        # Optional score filtering
        filtered = []
        for c in raw_chunks:
            if score_threshold is None or c.score is None:
                filtered.append(c)
            elif c.score >= score_threshold:
                filtered.append(c)

        # Delete duplicates if present
        deduplicated = self.deduplicate(filtered)

        # Sort strongest to weakest
        deduplicated.sort(key=lambda c: c.score or 0, reverse=True)

        # Truncate to requested size
        final = deduplicated[:top_k]

        logger.info(
            f"Retrieval summary | query='{query}' |"
            f"Retrieved {len(final)} chunks "
            f"(raw={len(raw_chunks)}, filtered={len(filtered)})"
        )

        return SearchResult(query=query, chunks=final)

    def deduplicate(self, chunks: list[Chunk]) -> list[Chunk]:
        """Remove duplicate chunks based on metadata identity.

        Args:
            chunks (list[Chunk]): List of Chunk objects.

        Returns:
            list[Chunk]: Deduplicated list of chunks.
        """

        seen = set()
        result = []

        for c in chunks:

            key = (
                c.metadata.get("source"), #file name
                c.metadata.get("chunk_index"), #index within that file
                c.id #unique chunk id
            )

            if key not in seen:
                seen.add(key)
                result.append(c)

        return result
