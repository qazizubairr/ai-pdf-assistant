from app.services.embeddings import generate_embeddings
from app.database import hybrid_search

question = "Why is paraphrasing sometimes considered plagiarism?"

document_id = 12

# Generate question embedding
query_embedding = generate_embeddings([question])[0]

# Perform hybrid search
results = hybrid_search(
    query=question,
    query_embedding=query_embedding,
    document_id=document_id,
    candidate_k=20,
    top_k=5
)

print("Number of results:", len(results))

for i, result in enumerate(results, start=1):

    print(f"\n--- Result {i} ---")
    print("Page:", result["page_number"])
    print("Chunk:", result["chunk_index"])
    print("RRF Score:", result["retrieval_score"])
    print("Vector Rank:", result["vector_rank"])
    print("FTS Rank:", result["fts_rank"])
    print("Similarity:", result.get("similarity"))
    print("Content:", result["content"])