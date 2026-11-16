from app.database import reciprocal_rank_fusion

vector_results = [
    {"page_number": 3, "chunk_index": 0, "content": "A"},
    {"page_number": 6, "chunk_index": 1, "content": "B"},
    {"page_number": 9, "chunk_index": 0, "content": "C"},
]

fts_results = [
    {"page_number": 9, "chunk_index": 0, "content": "C"},
    {"page_number": 3, "chunk_index": 0, "content": "A"},
    {"page_number": 10, "chunk_index": 1, "content": "D"},
]

results = reciprocal_rank_fusion(
    vector_results,
    fts_results,
    top_k=5
)

for result in results:
    print(
        result["content"],
        result["retrieval_score"],
        result["vector_rank"],
        result["fts_rank"]
    )