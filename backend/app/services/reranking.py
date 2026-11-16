
from sentence_transformers import CrossEncoder


# Load the model once when this module is imported.
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512
)


def rerank_chunks(question, chunks, top_k=5):

    if not chunks:
        return []

    # Create question-document pairs.
    pairs = [
        (question, chunk["content"])
        for chunk in chunks
    ]

    # Predict relevance scores.
    scores = reranker.predict(pairs)

    # Attach scores to the original chunks.
    reranked_results = []

    for chunk, score in zip(chunks, scores):
        result = chunk.copy()
        result["rerank_score"] = float(score)

        reranked_results.append(result)

    # Highest relevance score first.
    reranked_results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked_results[:top_k]
