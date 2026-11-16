import fitz  # PyMuPDF

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.chunking import chunk_text
from app.services.embeddings import (
    generate_embeddings,
    get_embedding_dimension
)
from starlette.concurrency import run_in_threadpool
from app.database import save_document, save_chunks

router = APIRouter(prefix="/documents", tags=["Documents"])

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Basic file validation
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="The file is empty")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File size must not exceed 10 MB"
        )

    try:
        pdf = fitz.open(stream=content, filetype="pdf")

        pages = []

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            pages.append({
                "page_number": page_number,
                "text": text
            })

        pdf.close()

        extracted_text = "\n".join(
            page["text"] for page in pages
        )

        if not extracted_text.strip():
            raise HTTPException(
                status_code=422,
                detail="No readable text found. This may be a scanned PDF."
            )
        
        all_chunks = []

        for page in pages:
            page_chunks = chunk_text(page["text"])

            for index, chunk in enumerate(page_chunks):
                all_chunks.append({
                    "page_number": page["page_number"],
                    "chunk_index": index,
                    "text": chunk
                })
        
        texts = [chunk["text"] for chunk in all_chunks]

        embeddings = await run_in_threadpool(
            generate_embeddings,
            texts
        )

        for chunk, embedding in zip(all_chunks, embeddings):
            chunk["embedding"] = embedding

        
        document_id = save_document(
            filename=file.filename,
            total_pages=len(pages)
        )

        save_chunks(
            document_id=document_id,
            chunks=all_chunks
        )

        return {
            "document_id": document_id,
            "filename": file.filename,
            "total_pages": len(pages),
            "total_chunks": len(all_chunks),
            "message": "PDF processed and saved successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read this PDF file: {e}"
        )