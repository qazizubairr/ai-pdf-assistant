
import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from pgvector import Vector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is missing")

    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id BIGSERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                total_pages INTEGER NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id BIGSERIAL PRIMARY KEY,
                document_id BIGINT NOT NULL
                    REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                page_number INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR(384) NOT NULL
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS
            document_chunks_embedding_idx
            ON document_chunks
            USING hnsw (embedding vector_cosine_ops)
        """)


def save_document(filename, total_pages):
    with get_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO documents (filename, total_pages)
            VALUES (%s, %s)
            RETURNING id
            """,
            (filename, total_pages)
        ).fetchone()

        return row[0]



def save_chunks(document_id, chunks):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO document_chunks
                (document_id, chunk_index, page_number, content, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                [
                    (
                        document_id,
                        chunk["chunk_index"],
                        chunk["page_number"],
                        chunk["text"],
                        chunk["embedding"]
                    )
                    for chunk in chunks
                ]
            )


def search_similar_chunks(
    query_embedding,
    document_id,
    top_k=20,
    similarity_threshold=0.35
):
    query_vector = Vector(query_embedding)

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                dc.content,
                dc.page_number,
                dc.chunk_index,
                d.filename,
                dc.embedding <=> %s AS distance
            FROM document_chunks dc
            JOIN documents d
                ON dc.document_id = d.id
            WHERE dc.document_id = %s
            ORDER BY dc.embedding <=> %s
            LIMIT %s
            """,
            (query_vector, document_id, query_vector, top_k)
        ).fetchall()

        results = [
            {
                "content": row[0],
                "page_number": row[1],
                "chunk_index": row[2],
                "filename": row[3],
                "similarity": 1 - row[4]
            }
            for row in rows
        ]

        return [
            result
            for result in results
            if result["similarity"] >= similarity_threshold
        ]

def search_full_text(
    query,
    document_id,
    top_k=20
):
    with get_connection() as conn:

        rows = conn.execute(
            """
            SELECT
                dc.content,
                dc.page_number,
                dc.chunk_index,
                d.filename,
                ts_rank(
                    dc.search_vector,
                    to_tsquery('english', %s)
                ) AS rank
            FROM document_chunks dc
            JOIN documents d
                ON dc.document_id = d.id
            WHERE
                dc.document_id = %s
                AND dc.search_vector @@
                    to_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
            """,
            (
                " | ".join(query.split()),
                document_id,
                " | ".join(query.split()),
                top_k
            )
        ).fetchall()

        return [
            {
                "content": row[0],
                "page_number": row[1],
                "chunk_index": row[2],
                "filename": row[3],
                "fts_score": row[4]
            }
            for row in rows
        ]


def reciprocal_rank_fusion(
    vector_results,
    fts_results,
    top_k=5,
    rrf_k=60
):
    fused_results = {}

    result_lists = [
        ("vector", vector_results),
        ("fts", fts_results)
    ]

    for search_type, results in result_lists:

        for rank, result in enumerate(results, start=1):

            key = (
                result["page_number"],
                result["chunk_index"]
            )

            if key not in fused_results:
                fused_results[key] = {
                    **result,
                    "retrieval_score": 0,
                    "vector_rank": None,
                    "fts_rank": None
                }

            fused_results[key]["retrieval_score"] += (
                1 / (rrf_k + rank)
            )

            fused_results[key][f"{search_type}_rank"] = rank

    ranked_results = sorted(
        fused_results.values(),
        key=lambda x: x["retrieval_score"],
        reverse=True
    )

    return ranked_results[:top_k]


def hybrid_search(
    query,
    query_embedding,
    document_id,
    candidate_k=20,
    top_k=5,
    similarity_threshold=0.35,
    rrf_k=60
):
    # 1. Retrieve candidates using vector search
    vector_results = search_similar_chunks(
        query_embedding=query_embedding,
        document_id=document_id,
        top_k=candidate_k,
        similarity_threshold=similarity_threshold
    )

    # 2. Retrieve candidates using full-text search
    fts_results = search_full_text(
        query=query,
        document_id=document_id,
        top_k=candidate_k
    )

    # 3. Combine both result lists using RRF
    results = reciprocal_rank_fusion(
        vector_results=vector_results,
        fts_results=fts_results,
        top_k=top_k,
        rrf_k=rrf_k
    )

    return results