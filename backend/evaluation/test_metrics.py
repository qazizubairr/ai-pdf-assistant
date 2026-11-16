
from metrics import (
    recall_at_k,
    reciprocal_rank,
    hit_rate_at_k
)


results = [
    {"page_number": 4},
    {"page_number": 3},
    {"page_number": 6},
    {"page_number": 9},
    {"page_number": 12}
]

relevant_pages = [3, 9]

print("Recall@3:", recall_at_k(results, relevant_pages, 3))
print("Recall@5:", recall_at_k(results, relevant_pages, 5))

print("MRR:", reciprocal_rank(results, relevant_pages))

print("Hit Rate@3:", hit_rate_at_k(results, relevant_pages, 3))
print("Hit Rate@5:", hit_rate_at_k(results, relevant_pages, 5))
