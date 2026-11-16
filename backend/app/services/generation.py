import asyncio
import json
import re
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

API_URL = "https://openrouter.ai/api/v1/chat/completions"


async def generate_answer(question, retrieved_chunks):

    if not API_KEY:
        raise ValueError("OpenRouter API key is missing")

    if not retrieved_chunks:
        return "I couldn't find relevant information in this document."

    context = "\n\n".join(
        [
            f"[Source: {chunk['filename']}, Page {chunk['page_number']}]\n"
            f"{chunk['content']}"
            for chunk in retrieved_chunks
        ]
    )

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant that answers questions "
                    "about uploaded PDF documents.\n\n"

                    "Follow these rules strictly:\n"
                    "1. Answer using only the provided context.\n"
                    "2. Do not invent facts or recommendations.\n"
                    "3. Distinguish explicitly stated information "
                    "from reasonable interpretations.\n"
                    "4. If you infer something, clearly label it "
                    "as an inference rather than a direct statement.\n"
                    "5. If the context does not contain enough "
                    "information, acknowledge the limitation.\n"
                    "6. Cite the source filename and page number "
                    "for every important claim.\n"
                    "7. Organize answers clearly using headings "
                    "or bullet points when appropriate.\n"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ],
        "temperature": 0.2
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "AI PDF Knowledge Assistant"
    }

    async with httpx.AsyncClient(timeout=60.0) as client:

        for attempt in range(3):

            print(f"\nOpenRouter request attempt: {attempt + 1}/3")

            response = await client.post(
                API_URL,
                headers=headers,
                json=payload
            )

            if response.status_code == 429:

                retry_after = response.headers.get("Retry-After")

                print("Rate limit response:", response.text)
                print("Retry-After:", retry_after)

                if attempt == 2:
                    response.raise_for_status()

                wait_time = (
                    int(retry_after)
                    if retry_after and retry_after.isdigit()
                    else 15 * (attempt + 1)
                )

                print(f"Waiting {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                continue

            response.raise_for_status()
            result = response.json()
            break

    return result["choices"][0]["message"]["content"]



async def generate_candidate_claims(question, retrieved_chunks):

    if not API_KEY:
        raise ValueError("OpenRouter API key is missing")

    if not retrieved_chunks:
        return []

    context = "\n\n".join(
        [
            f"[Page {chunk['page_number']}]\n{chunk['content']}"
            for chunk in retrieved_chunks
        ]
    )

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract factual claims from the document excerpts that directly "
                    "answer the user's question.\n\n"

                    "Rules:\n"
                    "1. Use ONLY information explicitly stated in the excerpts.\n"
                    "2. Generate claims relevant to the specific question, not general "
                    "facts that happen to appear in the document.\n"
                    "3. Each claim must express exactly ONE factual assertion.\n"
                    "4. Never combine assertions using 'and', 'but', or multiple clauses.\n"
                    "5. Split separate penalties, outcomes, and consequences into "
                    "individual claims.\n"
                    "6. Preserve important conditions. For example, retain "
                    "'if plagiarism is held to have taken place'.\n"
                    "7. Do not introduce assumptions or information from outside "
                    "the excerpts.\n"
                    "8. Do not repeat the same information in different wording.\n"
                    "9. Prefer short, simple, self-contained sentences.\n"
                    "10. Do not include unrelated definitions, explanations, or "
                    "background information.\n"
                    "11. If the excerpts contain no information relevant to the question, "
                    "return an empty list.\n"
                    "12. Return at most 10 claims.\n\n"

                    "Return ONLY valid JSON:\n"
                    '{"claims": ["Atomic claim one.", "Atomic claim two."]}'
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Document excerpts:\n{context}"
                )
            }
        ],
        "temperature": 0
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "AI PDF Knowledge Assistant"
    }

    async with httpx.AsyncClient(timeout=60.0) as client:

        response = await client.post(
            API_URL,
            headers=headers,
            json=payload
        )

        response.raise_for_status()

        result = response.json()

    # ---------------------------------------
    # Debug OpenRouter response
    # ---------------------------------------

    print("\n--- OPENROUTER DEBUG ---")

    print("Question:", question)
    print("Model:", result.get("model"))

    choices = result.get("choices", [])

    if not choices:
        raise ValueError(
            f"OpenRouter returned no choices: {result}"
        )

    choice = choices[0]

    print("Finish reason:", choice.get("finish_reason"))

    message = choice.get("message", {})
    content = message.get("content")

    print("Raw content:", repr(content))

    # ---------------------------------------
    # Validate generated content
    # ---------------------------------------

    if not content or not isinstance(content, str):
        print("Model returned empty or invalid content.")
        return []

    content = content.strip()

    if not content:
        print("Model returned whitespace-only content.")
        return []

    # ---------------------------------------
    # Remove Markdown JSON code fences
    # ---------------------------------------

    content = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        content,
        flags=re.IGNORECASE
    ).strip()

    # ---------------------------------------
    # Parse JSON
    # ---------------------------------------

    try:
        parsed = json.loads(content)

    except json.JSONDecodeError as error:

        print("\nJSON parsing failed:", error)
        print("Unparsed content:", repr(content))

        raise ValueError(
            "OpenRouter returned invalid JSON"
        ) from error

    if not isinstance(parsed, dict):
        raise ValueError(
            "OpenRouter response must be a JSON object"
        )

    claims = parsed.get("claims")

    if not isinstance(claims, list):
        raise ValueError(
            "OpenRouter response does not contain a valid claims list"
        )

    # ---------------------------------------
    # Clean generated claims
    # ---------------------------------------

    cleaned_claims = [
        claim.strip()
        for claim in claims
        if isinstance(claim, str) and claim.strip()
    ]

    print("Parsed claims:", len(cleaned_claims))

    return cleaned_claims[:15]


async def split_into_atomic_claims(claims):

    if not API_KEY:
        raise ValueError("OpenRouter API key is missing")

    if not claims:
        return []

    claims_text = "\n".join(
        f"- {claim}" for claim in claims
    )

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You break factual statements into atomic claims "
                    "for Natural Language Inference verification.\n\n"

                    "Rules:\n"
                    "1. Each atomic claim must express one "
                    "independently verifiable fact.\n"
                    "2. Preserve the original meaning exactly.\n"
                    "3. Do not introduce new information.\n"
                    "4. Do not remove important conditions or "
                    "qualifications.\n"
                    "5. Keep claims self-contained and declarative.\n"
                    "6. Do not create duplicate claims.\n"
                    "7. Simple claims should remain unchanged.\n"
                    "8. Return between 1 and 15 claims.\n\n"

                    "Return ONLY valid JSON:\n"
                    '{"claims": ["Atomic claim one.", "Atomic claim two."]}'
                )
            },
            {
                "role": "user",
                "content": (
                    "Break these statements into atomic claims:\n\n"
                    f"{claims_text}"
                )
            }
        ],
        "temperature": 0
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "AI PDF Knowledge Assistant"
    }

    async with httpx.AsyncClient(timeout=60.0) as client:

        response = await client.post(
            API_URL,
            headers=headers,
            json=payload
        )

        response.raise_for_status()

        result = response.json()

        content = result["choices"][0]["message"]["content"]

        if not content:
            return []

        content = content.strip()

        content = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            content,
            flags=re.IGNORECASE
        )

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return []

        atomic_claims = parsed.get("claims", [])

        if not isinstance(atomic_claims, list):
            return []

        return [
            claim.strip()
            for claim in atomic_claims
            if isinstance(claim, str) and claim.strip()
        ][:15]
