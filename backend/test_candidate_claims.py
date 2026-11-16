
import asyncio
import httpx

from app.database import hybrid_search
from app.services.embeddings import generate_embeddings
from app.services.generation import generate_candidate_claims
from app.services.nli_verification import (
    extract_candidate_claims,
    verify_extracted_claims
)


DOCUMENT_ID = 12
CANDIDATE_K = 40
TOP_K = 15

QUESTION = "What are the consequences of plagiarism?"


async def main():

    print("=" * 60)
    print("SINGLE QUESTION: CLAIM GENERATION + NLI")
    print("=" * 60)

    print("\nQuestion:", QUESTION)

    try:

        # 1. Generate question embedding
        embedding = generate_embeddings([QUESTION])[0]

        # 2. Retrieve relevant chunks
        chunks = hybrid_search(
            query=QUESTION,
            query_embedding=embedding,
            document_id=DOCUMENT_ID,
            candidate_k=CANDIDATE_K,
            top_k=TOP_K
        )

        print("\nRetrieved chunks:", len(chunks))
        
        print("\n" + "=" * 60)
        print("FULL PAGE 12 EVIDENCE")
        print("=" * 60)

        for chunk in chunks:
            if chunk["page_number"] == 12:
                print("\nChunk index:", chunk["chunk_index"])
                print(chunk["content"])
                print("-" * 60)

        for index, chunk in enumerate(chunks, start=1):
            print(
                f"\nChunk {index} | Page {chunk['page_number']}"
            )
            print(chunk["content"][:300])

        if not chunks:
            print("\nNo chunks retrieved. Cannot verify evidence.")
            return

        # 3. Generate atomic claims
        # claims = await generate_candidate_claims(
        #     QUESTION,
        #     chunks
        # )

        # print("\n" + "=" * 60)
        # print("GENERATED ATOMIC CLAIMS")
        # print("=" * 60)

        # if not claims:
        #     print("No claims generated.")
        #     return

        # for index, claim in enumerate(claims, start=1):
        #     print(f"{index}. {claim}")

        

        # 4. Verify claims using local NLI
        candidate_claims = extract_candidate_claims(
            chunks,
            max_claims=30
        )

        print("\nExtracted candidate claims:")

        for index, item in enumerate(candidate_claims, start=1):
            print(
                f"{index}. [Page {item['page_number']}] "
                f"{item['claim']}"
            )

        test_claims = [
            item["claim"]
            for item in candidate_claims
        ]


        verification = verify_extracted_claims(
            chunks,
            candidate_claims
        )

        print("\n" + "=" * 60)
        print("NLI DIAGNOSTICS")
        print("=" * 60)

        for item in verification["diagnostics"]:

            print("\nClaim:", item["claim"])
            print("Page:", item["page_number"])
            print("Chunk:", item["chunk_index"])
            print("Entailment:", round(item["entailment"], 4))
            print("Contradiction:", round(item["contradiction"], 4))
            print("Neutral:", round(item["neutral"], 4))
            print("Supported:", item["supported"])

        # 5. Show verification results
        print("\n" + "=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)

        print("\nVerified claims:")

        for claim in verification["verified_claims"]:
            print("-", claim["claim"])
            print("  Supporting pages:", claim["supporting_pages"])

        print("\nUnsupported claims:")

        for claim in verification["unsupported_claims"]:
            print("-", claim)

        print("\nEvidence sufficient:", verification["sufficient"])

    except httpx.TimeoutException:
        print("\nERROR: OpenRouter request timed out.")

    except httpx.HTTPError as error:
        print("\nHTTP ERROR:", error)

    except (ValueError, KeyError, IndexError, TypeError) as error:
        print("\nPROCESSING ERROR:", error)


if __name__ == "__main__":
    asyncio.run(main())
