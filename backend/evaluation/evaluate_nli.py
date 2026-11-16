
import json
from pathlib import Path

from app.services.embeddings import generate_embeddings
from app.database import hybrid_search
from app.services.nli_verification import (
    verify_claim,
    aggregate_evidence
)

from evaluation.metrics import classification_metrics


DOCUMENT_ID = 12

LABELS = [
    "entailment",
    "contradiction",
    "neutral"
]

ENTAILMENT_THRESHOLD = 0.80
CONTRADICTION_THRESHOLD = 0.80


def load_dataset():

    dataset_path = (
        Path(__file__).parent / "nli_dataset.json"
    )

    with open(dataset_path, "r", encoding="utf-8") as file:
        return json.load(file)


def predict_document_label(results):

    if not results:
        return "neutral", False

    max_entailment = max(
        result["probabilities"]["entailment"]
        for result in results
    )

    max_contradiction = max(
        result["probabilities"]["contradiction"]
        for result in results
    )

    has_entailment = (
        max_entailment >= ENTAILMENT_THRESHOLD
    )

    has_contradiction = (
        max_contradiction >= CONTRADICTION_THRESHOLD
    )

    conflicting = has_entailment and has_contradiction

    if conflicting:

        if max_entailment >= max_contradiction:
            return "entailment", True

        return "contradiction", True

    if has_entailment:
        return "entailment", False

    if has_contradiction:
        return "contradiction", False

    return "neutral", False


def evaluate_nli():

    dataset = load_dataset()

    y_true = []
    y_pred = []

    correct = 0
    total = len(dataset)

    for index, item in enumerate(dataset, start=1):

        question = item["question"]
        hypothesis = item["hypothesis"]
        expected_label = item["expected_label"]

        print("\n" + "=" * 60)
        print(f"Test {index}: {question}")
        print("Hypothesis:", hypothesis)

        # Generate query embedding
        embeddings = generate_embeddings([question])
        query_embedding = embeddings[0]

        # Retrieve evidence
        chunks = hybrid_search(
            query=question,
            query_embedding=query_embedding,
            document_id=DOCUMENT_ID,
            candidate_k=40,
            top_k=15
        )

        # Keep the existing Test 4 diagnostic
        if index == 4:

            print("\n========== TEST 4 RETRIEVED EVIDENCE ==========")

            for i, chunk in enumerate(chunks, start=1):

                print(f"\n--- Chunk {i} ---")
                print("Page:", chunk["page_number"])
                print("Content:", chunk["content"][:600])

        # Verify hypothesis against retrieved chunks
        results = verify_claim(chunks, hypothesis)

        # Aggregate evidence for page-level diagnostics
        summary = aggregate_evidence(results)

        # Predict one document-level NLI label
        predicted_label, conflicting = predict_document_label(
            results
        )

        y_true.append(expected_label)
        y_pred.append(predicted_label)

        is_correct = predicted_label == expected_label
        correct += int(is_correct)

        print("\nRetrieved chunks:", len(chunks))
        print("Evidence status:", summary["status"])
        print("Supporting pages:", summary["supporting_pages"])
        print("Contradicting pages:", summary["contradicting_pages"])

        print("Expected label:", expected_label)
        print("Predicted label:", predicted_label)
        print("Conflicting evidence:", conflicting)
        print("Correct:", is_correct)

    # Calculate three-class classification metrics
    metrics = classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        labels=LABELS
    )

    print("\n" + "=" * 60)
    print("NLI CLASSIFICATION EVALUATION REPORT")
    print("=" * 60)

    print("Total tests:", total)
    print("Correct predictions:", correct)
    print("Accuracy:", round(metrics["accuracy"], 4))

    print("\nCONFUSION MATRIX")
    print("Rows = Actual | Columns = Predicted")

    print(f"\n{'Actual':<18}", end="")

    for label in LABELS:
        print(f"{label:<18}", end="")

    print()

    for actual in LABELS:

        print(f"{actual:<18}", end="")

        for predicted in LABELS:

            value = metrics["confusion_matrix"][actual][predicted]
            print(f"{value:<18}", end="")

        print()

    print("\nPER-CLASS METRICS")

    for label, scores in metrics["per_class"].items():

        print(f"\n{label.upper()}")

        print("Precision:", round(scores["precision"], 4))
        print("Recall:", round(scores["recall"], 4))
        print("F1-score:", round(scores["f1"], 4))


if __name__ == "__main__":
    evaluate_nli()
