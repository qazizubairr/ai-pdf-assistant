
from app.services.embeddings import generate_embeddings
from app.database import (
    search_similar_chunks,
    search_full_text
)

question = "How can students avoid plagiarism?"
document_id = 12

query_embedding = generate_embeddings([question])[0]

# Run vector search independently
vector_results = search_similar_chunks(
    query_embedding=query_embedding,
    document_id=document_id,
    top_k=20,
    similarity_threshold=0.35
)

# Run full-text search independently
fts_results = search_full_text(
    query=question,
    document_id=document_id,
    top_k=20
)


def display_results(title, results, score_key):

    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")

    for rank, result in enumerate(results, start=1):

        print(f"\nRank: {rank}")
        print("Page:", result["page_number"])
        print("Chunk:", result["chunk_index"])
        print("Score:", result.get(score_key))
        print("Content:", result["content"][:350])


display_results(
    "VECTOR SEARCH",
    vector_results,
    "similarity"
)

display_results(
    "FULL TEXT SEARCH",
    fts_results,
    "fts_score"
)
