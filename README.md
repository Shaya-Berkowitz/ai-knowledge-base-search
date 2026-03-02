# AI Knowledge Platform (RAG Backend)

A production-style AI Knowledge Platform that ingests documents, builds semantic embeddings, and provides intelligent search and question answering using Retrieval-Augmented Generation (RAG).

This system is designed as a scalable backend for knowledge assistants, analytics tools, and AI copilots.

## Architecture Decision

This project uses:

- Pinecone for embeddings storage and vector similarity search
- PostgreSQL for all non-vector relational data (documents, chunks, metadata)

## Features

- Document ingestion and chunking pipeline
- OpenAI embedding generation
- Pinecone vector retrieval with relevance thresholding and deduplication
- RAG answer generation with citations
- FastAPI service endpoints for health, retrieval, and RAG answering
- PostgreSQL persistence for document and chunk metadata
- Dockerized PostgreSQL setup for local development

## High-Level Flow

Ingestion -> Chunking -> Embeddings -> Pinecone Retrieval -> LLM (RAG) -> API Response

PostgreSQL stores relational metadata used by the application and ingestion pipeline.

## Tech Stack

| Layer | Technology |
|------|-----------|
| API | FastAPI |
| Vector Database | Pinecone |
| Relational Database | PostgreSQL |
| ORM | SQLAlchemy |
| LLM | OpenAI GPT |
| Embeddings | OpenAI embeddings |
| Containerization | Docker / Docker Compose |

## API Endpoints

- `GET /health`  
  Health check endpoint.

- `POST /search`  
  Semantic retrieval endpoint.  
  Request body: `{"query": "...", "top_k": 5}`

- `POST /rag-answer`  
  Full RAG pipeline endpoint (retrieve + generate + citations).  
  Request body: `{"query": "...", "top_k": 5, "score_threshold": 0.3}`

## Project Layout

- `main.py` - FastAPI application and endpoints
- `build_index.py` - Ingestion/indexing pipeline
- `core/` - configuration, DB setup, embeddings
- `retrieval/` - vector store client, retriever logic, response models
- `ingestion/` - document loading and chunking
- `app/` - middleware and app-level errors
- `docs/` - retrieval tuning notes
- `scripts/` - helper scripts and evaluation utilities

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Configure environment variables in `.env`.

```env
OPENAI_API_KEY=...
PINECONE_API_KEY=...
DATABASE_URL=postgresql://<user>:<password>@localhost:5432/<db_name>
```

4. Start PostgreSQL locally (optional for local relational metadata).

```bash
docker-compose up -d
```

5. Run the API.

```bash
uvicorn main:app --reload
```

## Notes

- Vector search is intentionally handled only by Pinecone.
- PostgreSQL is intentionally used for all non-vector application data.
