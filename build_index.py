"""
Build the Pinecone vector index by:

1) Loading documents
2) Chunking text
3) Generating embeddings
4) Uploading vectors to Pinecone
"""

import logging
from math import ceil

from ingestion.load_docs import load_text_files
from ingestion.chunk import chunk_text
from embeddings.embed import get_embedding
from core.config import PINECONE_API_KEY
from pinecone import Pinecone


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


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
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Ensure index is ready
    ensure_index(pc)
    index = pc.Index(INDEX_NAME)

    # Load documents
    logger.info("Loading documents...")
    docs = load_text_files()
    logger.info(f"Documents loaded: {len(docs)}")

    # Chunk documents
    logger.info("Chunking documents...")
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc["text"]))
    logger.info(f"Total chunks generated: {len(all_chunks)}")

    # Create embeddings
    logger.info("Generating embeddings...")
    embeddings = [get_embedding(chunk) for chunk in all_chunks]
    logger.info("Embeddings created successfully.")

    # Prepare vectors for Pinecone
    logger.info("Uploading to Pinecone...")
    vectors = [
        (str(i), emb, {"text": chunk})
        for i, (emb, chunk) in enumerate(zip(embeddings, all_chunks))
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
