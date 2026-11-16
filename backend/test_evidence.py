
import json
from pathlib import Path

from app.database import hybrid_search
from app.services.embeddings import generate_embeddings
from app.services.nli_verification import (
    check_question_evidence,
    verify_claim
)


DOCUMENT_ID = 12

dataset_path = (
    Path(__file__).parent
    / "evaluation"
    / "dataset.json"
)

with open(dataset_path, encoding="utf-8") as file:
    dataset = json.load(file)


# Specific hypotheses for diagnosing failed questions
diagnostic_hypotheses = {
    "What is the difference between collaboration and collusion?": [
        "Students can collaborate on group projects when the assignment requires joint work.",
        "Students working on individual assignments must complete their own work.",
        "Collusion occurs when students cooperate inappropriately on work that should be completed individually."
    ],

    "How can students avoid accidentally copying material when taking notes?": [
        "Students should avoid making notes while reading the original source.",
        "Students should read the source first, consider its meaning, and then make notes.",
        "Students should take care not to copy the original author's language when taking notes."
    ]
}


correct = 0
total = len(dataset)

for index, item in enumerate(dataset, start=1):

    question = item["question"]
    expected = item["answerable"]

    embedding = generate_embeddings([question])[0]

    chunks = hybrid_search(
        query=question,
        query_embedding=embedding,
        document_id=DOCUMENT_ID,
        candidate_k=40,
        top_k=15
    )

    result = check_question_evidence(chunks, question)

    # Run additional NLI diagnostics for the two known failures
    if question in diagnostic_hypotheses:

        print("\n--- HYPOTHESIS DIAGNOSTIC ---")
        print("Question:", question)
        print("Retrieved chunks:", len(chunks))

        hypotheses = diagnostic_hypotheses[question]

        for hypothesis in hypotheses:

            print("\nHypothesis:", hypothesis)

            verification = verify_claim(chunks, hypothesis)

            for chunk, evidence in zip(chunks, verification):

                probabilities = evidence["probabilities"]

                print(
                    f"Page {chunk['page_number']} | "
                    f"Entailment: {probabilities['entailment']:.4f} | "
                    f"Contradiction: {probabilities['contradiction']:.4f} | "
                    f"Neutral: {probabilities['neutral']:.4f}"
                )

    predicted = result["sufficient"]
    is_correct = predicted == expected

    correct += int(is_correct)

    print(f"\nTest {index}: {question}")
    print("Expected:", expected)
    print("Predicted:", predicted)
    print("Correct:", is_correct)
    print(
        "Supporting pages:",
        [
            chunk["page_number"]
            for chunk in result["supporting_chunks"]
        ]
    )

print("\n" + "=" * 40)
print(f"Accuracy: {correct}/{total}")
print(f"Percentage: {correct / total * 100:.2f}%")
