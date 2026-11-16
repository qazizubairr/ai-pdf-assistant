from app.database import search_full_text

print("Starting full-text search test...")

question = "Why is paraphrasing sometimes considered plagiarism?"

results = search_full_text(
    query=question,
    document_id=12,
    top_k=10
)

print("Search function executed.")
print("Number of results:", len(results))

if not results:
    print("No matching chunks found.")
else:
    for i, result in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print("Page:", result["page_number"])
        print("FTS Score:", result["fts_score"])
        print("Content:", result["content"])