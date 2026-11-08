
from langchain_core.documents import Document

def get_chunks(vector_store, document_id):
    result = vector_store.get(
        where={"document_id": document_id},
        include=["documents", "metadatas"],
    )

    chunks = [
        Document(page_content=text, metadata={**(meta or {}), "chunk_id": cid})
        for cid, text, meta in zip(result["ids"], result["documents"], result["metadatas"])
    ]

    # restore original order if you stored chunk_index at ingestion
    chunks.sort(key=lambda c: c.metadata.get("chunk_index", 0))
    return chunks