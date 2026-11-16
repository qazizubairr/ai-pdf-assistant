from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
import os
from app.routes.documents import router as documents_router
from app.database import init_db
from app.database import hybrid_search
from app.services.embeddings import generate_embeddings
from app.services.generation import generate_answer
from fastapi.concurrency import run_in_threadpool
from app.services.reranking import rerank_chunks
from app.services.evidence_check import check_evidence

load_dotenv()

app = FastAPI(title="AI PDF Knowledge Assistant")

app.include_router(documents_router)

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

API_URL = "https://openrouter.ai/api/v1/chat/completions"

@app.on_event("startup")
def startup():
    init_db()

class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"message": "AI PDF Assistant API is running"}


@app.post("/chat")
async def chat(request: ChatRequest):

    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OpenRouter API key is missing"
        )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "AI PDF Knowledge Assistant"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": request.message
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                API_URL,
                headers=headers,
                json=payload
            )

            response.raise_for_status()

            result = response.json()

            return {
                "answer": result["choices"][0]["message"]["content"]
            }

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="AI provider timed out"
        )

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=e.response.text
        )

    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(
            status_code=500,
            detail="Something went wrong"
        )


class AskRequest(BaseModel):
    question: str

    

@app.post("/documents/{document_id}/ask")
async def ask_document(document_id: int, request: AskRequest):

    try:
        # 1. Generate embedding for the user's question
        embeddings = await run_in_threadpool(
            generate_embeddings,
            [request.question]
        )

        query_embedding = embeddings[0]

        # 2. Retrieve relevant chunks using hybrid search
        chunks = await run_in_threadpool(
            hybrid_search,
            request.question,
            query_embedding,
            document_id,
            40,  # candidate_k
            15    # final top_k
        )

        chunks = await run_in_threadpool(
            rerank_chunks,
            request.question,
            chunks,
            8
        )

        # Check whether retrieved context contains sufficient evidence
        evidence_sufficient = await check_evidence(
            request.question,
            chunks
        )

        if not evidence_sufficient:
            return {
                "document_id": document_id,
                "question": request.question,
                "answer": (
                    "I couldn't find sufficient information in "
                    "this document to answer your question reliably."
                ),
                "sources": [],
                "evidence_sufficient": False
            }

        answer = await generate_answer(
            request.question,
            chunks
        )

        # 4. Return answer and sources
        sources = [
            {
                "filename": chunk["filename"],
                "page_number": chunk["page_number"],
                "retrieval_score": round(
                    chunk["retrieval_score"], 4
                ),
                "rerank_score": round(
                    chunk["rerank_score"], 4
                ),
                "vector_rank": chunk["vector_rank"],
                "fts_rank": chunk["fts_rank"]
            }
            for chunk in chunks
        ]

        return {
            "document_id": document_id,
            "question": request.question,
            "answer": answer,
            "sources": sources,
            "evidence_sufficient": True
        }

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="AI provider timed out"
        )

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter error: {e.response.text}"
        )

    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    except Exception as e:
        print("Document question error:", str(e))
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while processing your question"
        )
