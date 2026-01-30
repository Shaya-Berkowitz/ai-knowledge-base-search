from retrieval.retriever import SemanticRetriever

r = SemanticRetriever()
result = r.retrieve("What does my ingestion pipeline do?", top_k=5)

print("running tests...")
print("len(result.chunks):", len(result.chunks))
for c in result.chunks:
    print("SCORE:", c.score)
    print("SOURCE:", c.metadata.get("source"))
    print("CHUNK:", c.text[:200])
    print("----")