import asyncio
from app.services.evidence_check import check_evidence
import json
from pathlib import Path

from app.database import (
    search_similar_chunks,
    hybrid_search
)
from app.services.embeddings import generate_embeddings
from app.services.reranking import rerank_chunks

from evaluation.metrics import (
    recall_at_k,
    reciprocal_rank,
    hit_rate_at_k
)


# Change this if you upload a new copy of the PDF.
DOCUMENT_ID = 12

# Number of candidates to retrieve and evaluate.
TOP_K = 15

K_VALUES = [5, 10]


def load_dataset():
    dataset_path = Path(__file__).parent / "dataset.json"

    with open(dataset_path, "r", encoding="utf-8") as file:
        return json.load(file)


def retrieve_results(question):
    # Generate the query embedding once.
    embeddings = generate_embeddings([question])
    query_embedding = embeddings[0]

    # 1. Vector search
    vector_results = search_similar_chunks(
        query_embedding=query_embedding,
        document_id=DOCUMENT_ID,
        top_k=TOP_K,
        similarity_threshold=0.35
    )

    # 2. Hybrid search
    hybrid_results = hybrid_search(
        query=question,
        query_embedding=query_embedding,
        document_id=DOCUMENT_ID,
        candidate_k=40,
        top_k=TOP_K
    )

    # 3. Hybrid search + reranking
    reranked_results = rerank_chunks(
        question=question,
        chunks=hybrid_results,
        top_k=TOP_K
    )

    return {
        "Vector": vector_results,
        "Hybrid": hybrid_results,
        "Hybrid + Reranker": reranked_results
    }


def evaluate_results(dataset):
    methods = [
        "Vector",
        "Hybrid",
        "Hybrid + Reranker"
    ]

    # Store metrics for each retrieval method.
    scores = {
        method: {
            "recall": {k: [] for k in K_VALUES},
            "mrr": [],
            "hit_rate": {k: [] for k in K_VALUES},
            "unanswerable_returns": 0
        }
        for method in methods
    }

    answerable_count = 0
    unanswerable_count = 0
        # Evidence checker evaluation
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for item in dataset:

        question = item["question"]
        relevant_pages = item["relevant_pages"]
        answerable = item["answerable"]

        print(f"\nEvaluating: {question}")

        results = retrieve_results(question)

        reranked = results["Hybrid + Reranker"]

        print(
            "  Rerank scores:",
            [round(chunk["rerank_score"], 4) for chunk in reranked]
        )

                # Evaluate evidence sufficiency
        # predicted_answerable = asyncio.run(
        #     check_evidence(
        #         question,
        #         reranked
        #     )
        # )

        # Compare prediction against dataset ground truth
        # if answerable and predicted_answerable:
        #     true_positive += 1

        # elif not answerable and not predicted_answerable:
        #     true_negative += 1

        # elif not answerable and predicted_answerable:
        #     false_positive += 1

        # else:
        #     false_negative += 1

        print(f"  Actual answerable: {answerable}")
        # print(f"  Predicted answerable: {predicted_answerable}")

        rerank_scores = [
            round(result["rerank_score"], 4)
            for result in reranked
        ]

        print(f"  Answerable: {answerable}")
        print(f"  Reranker scores: {rerank_scores[:5]}")

        if answerable:
            answerable_count += 1

            for method, retrieved in results.items():

                scores[method]["mrr"].append(
                    reciprocal_rank(
                        retrieved,
                        relevant_pages
                    )
                )

                for k in K_VALUES:

                    scores[method]["recall"][k].append(
                        recall_at_k(
                            retrieved,
                            relevant_pages,
                            k
                        )
                    )

                    scores[method]["hit_rate"][k].append(
                        hit_rate_at_k(
                            retrieved,
                            relevant_pages,
                            k
                        )
                    )

                print(
                    f"  {method}: "
                    f"{[r['page_number'] for r in retrieved[:5]]}"
                )

        else:
            unanswerable_count += 1

            for method, retrieved in results.items():

                if retrieved:
                    scores[method]["unanswerable_returns"] += 1

                print(
                    f"  {method}: "
                    f"{len(retrieved)} results returned"
                )

        answerability_metrics = {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative
        }

    return (
        scores,
        answerable_count,
        unanswerable_count,
        answerability_metrics
    )


def print_report(scores, answerable_count, unanswerable_count):

    print("\n" + "=" * 75)
    print("RAG RETRIEVAL EVALUATION REPORT")
    print("=" * 75)

    print(f"Answerable questions: {answerable_count}")
    print(f"Unanswerable questions: {unanswerable_count}")

    for method, data in scores.items():

        print(f"\n{method}")
        print("-" * 45)

        if answerable_count:

            for k in K_VALUES:

                avg_recall = sum(data["recall"][k]) / answerable_count
                avg_hit_rate = sum(data["hit_rate"][k]) / answerable_count

                print(f"Recall@{k}:    {avg_recall:.4f}")
                print(f"Hit Rate@{k}:  {avg_hit_rate:.4f}")

            avg_mrr = sum(data["mrr"]) / answerable_count

            print(f"MRR:           {avg_mrr:.4f}")

        if unanswerable_count:

            returned = data["unanswerable_returns"]

            print(
                f"Unanswerable queries returning results: "
                f"{returned}/{unanswerable_count}"
            )

def print_answerability_report(metrics):

    tp = metrics["true_positive"]
    tn = metrics["true_negative"]
    fp = metrics["false_positive"]
    fn = metrics["false_negative"]

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0

    precision = (
        tp / (tp + fp)
        if (tp + fp) else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) else 0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) else 0
    )

    print("\n" + "=" * 75)
    print("EVIDENCE SUFFICIENCY EVALUATION REPORT")
    print("=" * 75)

    print(f"Total questions:       {total}")
    print(f"True Positives:        {tp}")
    print(f"True Negatives:        {tn}")
    print(f"False Positives:       {fp}")
    print(f"False Negatives:       {fn}")

    print(f"\nAccuracy:              {accuracy:.4f}")
    print(f"Precision:             {precision:.4f}")
    print(f"Recall:                {recall:.4f}")
    print(f"False Positive Rate:   {false_positive_rate:.4f}")

def main():

    dataset = load_dataset()

    (
        scores,
        answerable_count,
        unanswerable_count,
        answerability_metrics
    ) = evaluate_results(dataset)

    print_report(
        scores,
        answerable_count,
        unanswerable_count
    )

    print_answerability_report(answerability_metrics)

if __name__ == "__main__":
    main()
