
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

API_URL = "https://openrouter.ai/api/v1/chat/completions"


async def check_evidence(question, retrieved_chunks):

    if not API_KEY:
        raise ValueError("OpenRouter API key is missing")

    if not retrieved_chunks:
        return False

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
                    "You classify whether document excerpts contain enough "
                    "information to answer a question.\n\n"
                    "Return SUFFICIENT if the excerpts contain relevant "
                    "facts that answer the question.\n"
                    "Return INSUFFICIENT if they do not.\n\n"
                    "Do not explain your decision. "
                    "Return exactly one word."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Document excerpts:\n{context}"
                )
            }
        ],
        "temperature": 0,
        "max_tokens": 512
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

        if response.status_code != 200:
            print("OpenRouter error:", response.status_code)
            print("Response body:", response.text)

        response.raise_for_status()

        result = response.json()

        content = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )

        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"Evidence checker returned empty content. "
                f"Response: {result}"
            )

        classification = content.strip().upper().strip(" .\n\t")

        if classification == "SUFFICIENT":
            return True

        if classification == "INSUFFICIENT":
            return False

        raise ValueError(
            f"Unexpected evidence classification: {classification}"
        )
