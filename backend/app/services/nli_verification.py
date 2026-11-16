import re
from sentence_transformers import CrossEncoder
import numpy as np
from app.database import hybrid_search
from app.services.embeddings import generate_embeddings

# Load the NLI model once.
nli_model = CrossEncoder(
    "cross-encoder/nli-deberta-v3-base",
    max_length=512
)

# Inspect the model's label mapping.
print("NLI labels:", nli_model.model.config.id2label)


def extract_candidate_claims(
    chunks,
    max_claims=30,
    min_length=20
):
    """
    Extract candidate factual sentences from retrieved chunks.

    Returns sentences with their original source information.
    """

    candidate_claims = []
    seen_claims = set()

    for chunk in chunks:

        content = chunk["content"]

        # Normalize whitespace, including PDF line breaks.
        content = re.sub(r"\s+", " ", content).strip()

        # Split text into sentences.
        sentences = re.split(
            r'(?<=[.!?])\s+(?=[A-Z0-9“"\'])',
            content
        )

        for sentence in sentences:

            sentence = sentence.strip()

            # Ignore short fragments.
            if len(sentence) < min_length:
                continue

            # Avoid duplicate sentences.
            normalized = sentence.casefold()

            if normalized in seen_claims:
                continue

            seen_claims.add(normalized)

            candidate_claims.append({
                "claim": sentence,
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"]
            })

            if len(candidate_claims) >= max_claims:
                return candidate_claims

    return candidate_claims


def classify_claim(premise: str, hypothesis: str):

    # Get raw logits from the NLI model
    logits = nli_model.predict([
        (premise, hypothesis)
    ])[0]

    # Convert logits into probabilities using softmax
    exp_scores = np.exp(logits - np.max(logits))
    probabilities = exp_scores / np.sum(exp_scores)

    # Read the model's actual label mapping
    labels = nli_model.model.config.id2label

    # Find the highest-scoring class
    predicted_index = int(np.argmax(probabilities))

    predicted_label = labels[predicted_index].lower()

    return {
        "label": predicted_label,
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            labels[i].lower(): float(probabilities[i])
            for i in range(len(probabilities))
        }
    }


def verify_claim(chunks, hypothesis: str):

    if not chunks:
        return []

    # Compare the hypothesis against every retrieved chunk.
    pairs = [
        (chunk["content"], hypothesis)
        for chunk in chunks
    ]

    # Run all comparisons in one batch.
    logits = nli_model.predict(pairs)

    # Convert logits into probabilities.
    exp_scores = np.exp(
        logits - np.max(logits, axis=1, keepdims=True)
    )

    probabilities = exp_scores / np.sum(
        exp_scores,
        axis=1,
        keepdims=True
    )

    labels = nli_model.model.config.id2label

    results = []

    for chunk, scores in zip(chunks, probabilities):

        predicted_index = int(np.argmax(scores))

        results.append({
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "label": labels[predicted_index].lower(),
            "confidence": float(scores[predicted_index]),
            "probabilities": {
                labels[i].lower(): float(scores[i])
                for i in range(len(scores))
            }
        })

    return results



def verify_candidate_claims(chunks, claims):

    if not chunks or not claims:
        return {
            "sufficient": False,
            "verified_claims": [],
            "unsupported_claims": [],
            "diagnostics": []
        }

    verified_claims = []
    unsupported_claims = []
    diagnostics = []

    for claim in claims:

        results = verify_claim(chunks, claim)

        paired_results = list(zip(results, chunks))

        paired_results.sort(
            key=lambda item: item[0]["probabilities"]["entailment"],
            reverse=True
        )

        if not paired_results:
            continue

        best_result, best_chunk = paired_results[0]

        top_evidence = [
            {
                "page": result["page_number"],
                "entailment": result["probabilities"]["entailment"],
                "content": chunk["content"]
            }
            for result, chunk in paired_results[:3]
        ]

        if best_result is None:
            continue

        best_entailment = (
            best_result["probabilities"]["entailment"]
        )

        best_contradiction = (
            best_result["probabilities"]["contradiction"]
        )

        best_neutral = (
            best_result["probabilities"]["neutral"]
        )

        is_supported = best_entailment >= 0.80

        diagnostics.append({
            "claim": claim,
            "best_page": best_result["page_number"],
            "entailment": best_entailment,
            "contradiction": best_contradiction,
            "neutral": best_neutral,
            "supported": is_supported,
            "top_evidence": top_evidence
        })

        if is_supported:

            verified_claims.append({
                "claim": claim,
                "supporting_pages": sorted({
                    result["page_number"]
                    for result in results
                    if result["probabilities"]["entailment"] >= 0.80
                })
            })

        else:
            unsupported_claims.append(claim)

    return {
        "sufficient": bool(verified_claims),
        "verified_claims": verified_claims,
        "unsupported_claims": unsupported_claims,
        "diagnostics": diagnostics
    }


def verify_extracted_claims(
    chunks,
    candidate_claims,
    threshold=0.80
):
    """
    Verify each extracted sentence against its original source chunk.
    """

    verified_claims = []
    unsupported_claims = []
    diagnostics = []

    for candidate in candidate_claims:

        claim = candidate["claim"]
        page_number = candidate["page_number"]
        chunk_index = candidate["chunk_index"]

        # Find the exact chunk from which this sentence was extracted.
        source_chunk = next(
            (
                chunk for chunk in chunks
                if chunk["page_number"] == page_number
                and chunk["chunk_index"] == chunk_index
            ),
            None
        )

        if source_chunk is None:
            continue

        # Verify against the original source only.
        result = verify_claim(
            [source_chunk],
            claim
        )[0]

        entailment = result["probabilities"]["entailment"]
        contradiction = result["probabilities"]["contradiction"]
        neutral = result["probabilities"]["neutral"]

        supported = entailment >= threshold

        diagnostics.append({
            "claim": claim,
            "page_number": page_number,
            "chunk_index": chunk_index,
            "entailment": entailment,
            "contradiction": contradiction,
            "neutral": neutral,
            "supported": supported
        })

        if supported:
            verified_claims.append({
                "claim": claim,
                "supporting_pages": [page_number]
            })
        else:
            unsupported_claims.append(claim)

    return {
        "sufficient": bool(verified_claims),
        "verified_claims": verified_claims,
        "unsupported_claims": unsupported_claims,
        "diagnostics": diagnostics
    }


def aggregate_evidence(
    results,
    entailment_threshold=0.80,
    contradiction_threshold=0.80
):

    if not results:
        return {
            "status": "insufficient",
            "supporting_pages": [],
            "contradicting_pages": []
        }

    supporting_pages = set()
    contradicting_pages = set()

    for result in results:

        probabilities = result["probabilities"]

        if probabilities["entailment"] >= entailment_threshold:
            supporting_pages.add(result["page_number"])

        if probabilities["contradiction"] >= contradiction_threshold:
            contradicting_pages.add(result["page_number"])

    if supporting_pages and contradicting_pages:
        status = "conflicting"

    elif supporting_pages:
        status = "supported"

    elif contradicting_pages:
        status = "contradicted"

    else:
        status = "insufficient"

    return {
        "status": status,
        "supporting_pages": sorted(supporting_pages),
        "contradicting_pages": sorted(contradicting_pages)
    }


def check_question_evidence(chunks, question):
    """
    Check whether retrieved chunks contain evidence
    relevant to answering the user's question.
    """

    if not chunks:
        return {
            "sufficient": False,
            "supporting_chunks": [],
            "reason": "No relevant chunks were retrieved."
        }

    # Turn the question into a simple candidate claim.
    # This is an experimental baseline, not a reliable
    # way to formulate every possible answer.
    hypotheses = [
        f"The document contains information that answers: {question}"
    ]

    results = verify_claim(chunks, hypotheses[0])

    supporting_chunks = [
        result
        for result in results
        if result["probabilities"]["entailment"] >= 0.80
    ]

    return {
        "sufficient": bool(supporting_chunks),
        "supporting_chunks": supporting_chunks,
        "reason": (
            "Supporting evidence found."
            if supporting_chunks
            else "No sufficiently strong supporting evidence found."
        )
    }



def test_nli():

    question = "What are the consequences of plagiarism?"

    hypothesis = (
        "Students who plagiarise are always expelled "
        "from the University."
    )

    document_id = 12

    # Generate the question embedding
    embeddings = generate_embeddings([question])
    query_embedding = embeddings[0]

    # Retrieve actual chunks from the HEC PDF
    chunks = hybrid_search(
        query=question,
        query_embedding=query_embedding,
        document_id=document_id,
        candidate_k=20,
        top_k=4
    )

    print(f"\nRetrieved {len(chunks)} chunks")

    # Display the retrieved evidence
    for i, chunk in enumerate(chunks, start=1):

        print(f"\n--- Chunk {i} ---")
        print("Page:", chunk["page_number"])
        print("Content:", chunk["content"][:500])

    # Verify the hypothesis against retrieved chunks
    results = verify_claim(chunks, hypothesis)

    summary = aggregate_evidence(results)

    print("\n========== AGGREGATED EVIDENCE ==========")
    print(summary)

    print("\n========== NLI VERIFICATION ==========")

    for result in results:

        print("\nPage:", result["page_number"])
        print("Prediction:", result["label"])
        print("Confidence:", round(result["confidence"], 4))
        print("Probabilities:", result["probabilities"])

if __name__ == "__main__":
    test_nli()
