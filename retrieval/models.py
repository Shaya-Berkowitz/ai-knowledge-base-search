"""
Data models for retrieval system.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    """
    Represents a chunk of text in the retrieval system.

    Attributes:
        id (str): Unique identifier for the chunk.
        text (str): The text content of the chunk.
        metadata (Dict[str, Any]): Additional metadata associated with the chunk.
        score (Optional[float]): Semantic similarity score returned during retrieval.
                                 Before retrieval, this is None.
    """
    id: str
    text: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


@dataclass
class SearchResult:
    """
    Represents the result of a search query.
    Attributes:
        query (str): The original search query.
        chunks (List[Chunk]): Ranked list of chunks returned as search results.
    """
    query: str
    chunks: List[Chunk]