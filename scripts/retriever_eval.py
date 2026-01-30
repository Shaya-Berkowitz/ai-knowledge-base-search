from retrieval.retriever import SemanticRetriever

TEST_QUERIES = [
    "What are the three main stages of the RAG system?",
    "Why are documents split into overlapping chunks during ingestion?",
    "Which embedding model is used to generate document embeddings?",
    "Why must the same embedding model be used for ingestion and querying?",
    "What similarity metric is used by Pinecone during retrieval?",
    "What framework is used to implement the API?",
    "What is the purpose of assigning a request ID to each HTTP request?",
    "How does the API respond when an unexpected internal error occurs?",
    "Are internal stack traces returned to the client on errors?",
    "What does the /health endpoint indicate?"
]

retriever = SemanticRetriever()

for query in TEST_QUERIES:
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    result = retriever.retrieve(
        query=query,
        top_k=5,
    )

    if not result.chunks:
        print("NO RESULTS")
        continue

    for i, chunk in enumerate(result.chunks, start=1):
        print(f"\n{i}. score={chunk.score:.3f}")
        print(f"   source={chunk.metadata.get('source')}")
        print(f"   chunk_index={chunk.metadata.get('chunk_index')}")
        print("   text:")
        print("   " + chunk.text[:500])  # truncate for readability
