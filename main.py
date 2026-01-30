"""
AI RAG System FastAPI Service

Provides:
- /health      : Health check endpoint
- /search      : Semantic retrieval only for debugging
- /rag-answer  : Full RAG with citations

This service retrieves knowledge chunks from Pinecone, constructs context,
and uses OpenAI to generate answers based on ingested documents.
"""

from fastapi import FastAPI, HTTPException, Request
from app.middleware.request_id import request_id_middleware
from app.errors import UpstreamServiceError
from fastapi.responses import JSONResponse

from pydantic import BaseModel
from typing import List, Optional
import logging

from retrieval.retriever import SemanticRetriever
from openai import OpenAI
from core.config import OPENAI_API_KEY


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app and components
app = FastAPI(
    title="AI RAG System API",
    description="""
A RAG service.

Features:
- Pinecone Vector Retrieval
- Metadata-aware semantic search
- RAG answering with citations
- OpenAI response generation
""",
    version="1.0.0"
)

#register middleware
app.middleware("http")(request_id_middleware)

# Exception handlers for upstream service errors
@app.exception_handler(UpstreamServiceError)
async def upstream_error_handler(request: Request, exc: UpstreamServiceError) -> JSONResponse:
    """
    Handle UpstreamServiceError exceptions.
    Args:
        request (Request): Incoming HTTP request.
        exc (UpstreamServiceError): The raised exception.
    Returns:
        JSONResponse: HTTP response with error details.
    """

    # Retrieve request ID from request state
    request_id = getattr(request.state, "request_id", None)

    return JSONResponse(
        status_code=503,
        content={
            "error": str(exc),
            "service": exc.service,
            "request_id": request_id,
        },
    )


retriever = SemanticRetriever()
client = OpenAI(api_key=OPENAI_API_KEY)


# Request Models
class SearchRequest(BaseModel):
    """
    Request body for /search endpoint.
    
    Attributes:
        query (str): User query text.
        top_k (int): Number of chunks to retrieve."""
    query: str
    top_k: int = 5


class RagRequest(BaseModel):
    """
    Request body for /rag-answer endpoint.

    Attributes:
        query: (str) User query text.
        top_k: (int) Number of chunks to retrieve.
        score_threshold: (float) Minimum similarity score for retrieval.
    """
    query: str
    top_k: int = 5
    score_threshold: float = 0.3

# Response Models
class ChunkResponse(BaseModel):
    """
    Response model for a retrieved chunk.

    Attributes:
        text (str): Chunk text content.
        source (str, optional): Source document identifier.
        score (float, optional): Similarity score.
        chunk_index (int, optional): Index of the chunk in the source document.
    """
    text: str
    source: Optional[str] = None
    score: Optional[float] = None
    chunk_index: Optional[int] = None


class SearchResponse(BaseModel):
    """
    Response model for /search endpoint.
    Attributes:
        query (str): Original user query.
        results (List[ChunkResponse]): List of retrieved chunks.
    """
    query: str
    results: List[ChunkResponse]


class RagAnswerResponse(BaseModel):
    """
    Response model for /rag-answer endpoint.
    Attributes:
        query (str): Original user query.
        answer (str): Generated answer from OpenAI.
        sources (List[ChunkResponse]): List of source chunks used for the answer.
    """
    query: str
    answer: str
    sources: List[ChunkResponse]


# API Endpoints

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    Returns a simple status message.
    """
    return {"status": "ok", "message": "API is healthy"}


@app.post("/search", response_model=SearchResponse)
def semantic_search(req: SearchRequest):
    """
    Semantic chunk retrieval only.
    Useful for debugging.
    """
    try:
        logger.info(f"/search request: '{req.query}'")

        # Perform retrieval
        result = retriever.retrieve(
            query=req.query,
            top_k=req.top_k
        )

        # Build response
        chunks = [
            ChunkResponse(
                text=c.text,
                source=c.metadata.get("source"),
                score=c.score,
                chunk_index=c.metadata.get("chunk_index")
            )
            for c in result.chunks
        ]

        return SearchResponse(query=req.query, results=chunks)
    
    except UpstreamServiceError:
        raise  # Let the exception handler deal with it
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag-answer", response_model=RagAnswerResponse)
def rag_answer(req: RagRequest):
    """
    Full RAG pipeline:
    1) retrieve best chunks
    2) build prompt
    3) call OpenAI
    4) respond with answer + citations
    """

    try:
        logger.info(f"/rag-answer request: '{req.query}'")

        # Retrieve relevant chunks
        result = retriever.retrieve(
            query=req.query,
            top_k=req.top_k,
            score_threshold=req.score_threshold
        )

        # If no chunks found, return a default message
        if not result.chunks:
            return RagAnswerResponse(
                query=req.query,
                answer="No relevant information found in given docs.",
                sources=[]
            )

        # Build context
        context_blocks = []
        sources = []

        for c in result.chunks:
            sources.append(
                ChunkResponse(
                    text=c.text,
                    source=c.metadata.get("source"),
                    score=c.score,
                    chunk_index=c.metadata.get("chunk_index")
                )
            )

            context_blocks.append(f"Source: {c.metadata.get('source')} | Chunk: {c.metadata.get('chunk_index')}\n{c.text}")

        # Construct prompt for OpenAI
        context_text = "\n\n---\n\n".join(context_blocks)

        prompt = f"""
You are a helpful AI assistant. Answer the user's question using ONLY the context provided.
If the answer is not in the context, say you do not have enough information.

QUESTION:
{req.query}

CONTEXT:
{context_text}

ANSWER:
"""
        # Call OpenAI to generate answer
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant answering based on provided documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        answer = completion.choices[0].message.content

        return RagAnswerResponse(
            query=req.query,
            answer=answer,
            sources=sources
        )
    except UpstreamServiceError:
        raise  # Let the exception handler deal with it
    except Exception as e:
        logger.exception("RAG answer failed")
        raise HTTPException(status_code=500, detail=str(e))
