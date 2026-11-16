
from app.services.embeddings import generate_embeddings
from app.services.reranking import rerank_chunks
from app.database import hybrid_search


question = "How can students avoid plagiarism?"

# Generate the query embedding.
query_embedding = generate_embeddings([question])[0]

# Retrieve 15 candidates using existing hybrid search.
chunks = hybrid_search(
    query=question,
    query_embedding=query_embedding,
    document_id=12,
    candidate_k=40,
    top_k=15
)

print(f"Retrieved {len(chunks)} candidates")

# Rerank the candidates.
results = rerank_chunks(
    question=question,
    chunks=chunks,
    top_k=8
)

for rank, chunk in enumerate(results, start=1):

    print(f"\n--- Reranked Result {rank} ---")
    print("Page:", chunk["page_number"])
    print("Rerank score:", chunk["rerank_score"])
    print("Original vector rank:", chunk["vector_rank"])
    print("Original FTS rank:", chunk["fts_rank"])
    print("Content:", chunk["content"])
