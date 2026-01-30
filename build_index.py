"""
Build the Pinecone vector index by:

1) Loading documents
2) Chunking text
3) Generating embeddings
4) Uploading vectors to Pinecone
"""

import os

import logging
from math import ceil

from ingestion.load_docs import load_text_files
from ingestion.chunk import chunk_text
from core.embeddings import embed_texts
from core.config import PINECONE_API_KEY
from pinecone import Pinecone


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Chunking parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# Pinecone index configuration
INDEX_NAME = "ai-knowledge-base"
EMBED_DIM = 1536


def ensure_index(pc: Pinecone ) -> None:
    """
    Ensure the Pinecone index exists with the expected configuration.
    Creates it if missing.

    Args:
        pc (Pinecone): Initialized Pinecone client.

    Returns:
        None
    """
    # Get list of existing indexes
    existing = [idx["name"] for idx in pc.list_indexes()]

    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine"
        )
        logger.info(f"Created new index: {INDEX_NAME}")
    else:
        logger.info(f"Using existing index: {INDEX_NAME}")


def main() -> None:
    """Run the full index build pipeline."""

    logger.info(f"Chunking config: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

    logger.info(f"Using Pinecone index: {INDEX_NAME}")
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Ensure index is ready
    ensure_index(pc)
    index = pc.Index(INDEX_NAME)

    # Load documents
    logger.info("Loading documents...")
    docs = load_text_files()
    logger.info(f"Documents loaded: {len(docs)}")

    # Chunk documents and prepare metadata
    logger.info("Chunking documents and preparing metadata...")
    chunks_with_meta = []
    chunk_id = 0

    for doc in docs:
        chunks = chunk_text(doc["text"], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        for idx, chunk in enumerate(chunks):
            chunks_with_meta.append({
                "id": str(chunk_id),
                "chunk": chunk,
                "source": doc["filename"],   
                "chunk_index": idx           #index within this document
            })
            chunk_id += 1

    logger.info(f"Total chunks generated: {len(chunks_with_meta)}")

    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = embed_texts([c["chunk"] for c in chunks_with_meta])
    logger.info("Embeddings created successfully.")

    # Prepare Pinecone vectors
    logger.info("Uploading to Pinecone...")

    vectors = [
        (
            c["id"],
            emb,
            {
                "text": c["chunk"],
                "source": c["source"],
                "chunk_index": c["chunk_index"]
            }
        )
        for c, emb in zip(chunks_with_meta, embeddings)
    ]

    # Upload in batches
    BATCH_SIZE = 100
    total_batches = ceil(len(vectors) / BATCH_SIZE)

    for i in range(total_batches):
        batch = vectors[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        index.upsert(vectors=batch)
        logger.info(f"Uploaded batch {i+1}/{total_batches}")

    logger.info("Pinecone index built and saved successfully.")


if __name__ == "__main__":
    main()
