import asyncio
import os
import hashlib
from unified_parser import read_units
from recursive_chunking import make_chunks
from embedding_splitter import create_embedding_vectorstore_splitter
from chunk_load import load_chunks , bm25_index
from retrival import get_chunks
from check_document import is_already_loaded
from hybrid_retrival_rrf import rrf_results
from re_rank import rerank_results
from  fastapi import FastAPI , File , HTTPException , UploadFile
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager 
import tempfile
from langchain_core.documents import Document
from pydantic import BaseModel , Field
from tokenizer import tokenize


class RetrieveRequest( BaseModel ) :
    document_id: str
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once when the server starts
    print("Loading embedding model and vector store...")
    app.state.vectorstore = create_embedding_vectorstore_splitter()
    build_index()
    print("Vector store ready & index is ready")
    app.state.indexes = {}
    yield


app = FastAPI(lifespan=lifespan)




def build_index():
    vs = app.state.vectorstore
    data = vs.get(include=["documents", "metadatas"])

    chunks = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(data["documents"], data["metadatas"])
    ]
    chunks.sort(key=lambda c: (
        c.metadata["document_id"],
        c.metadata["page"],
        c.metadata["chunk_index"],
    ))

    if chunks:
        app.state.index = {"chunks": chunks, "bm25": bm25_index(chunks)}
    else:
        app.state.index = None

def add_to_index(doc_id):
    vs = app.state.vectorstore
    new_chunks = get_chunks(vs, doc_id)      # read from Chroma so text and IDs match dense search

    old = app.state.index
    chunks = (old["chunks"] if old else []) + new_chunks

    # build first, then swap, so a running search never sees a half-built index
    app.state.index = {"chunks": chunks, "bm25": bm25_index(chunks)}





@app.get("/")
def home():
    return {"message": "API is running"}


@app.get("/status")
def status():
    vs = app.state.vectorstore
    count = vs._collection.count()
    return {"vector_store": "ready", "total_chunks": count}


@app.post("/load")
async def load_file( file: UploadFile = File(...) ):

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    doc_id = hashlib.sha256(content).hexdigest()[:12]
    vs = app.state.vectorstore

    if is_already_loaded(doc_id , vs):
        return {"status": "already_loaded", "document_id": doc_id}

    ext = os.path.splitext(file.filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete = False , suffix=ext ) as tmp:
        tmp.write(content)
        tmp_path  = tmp.name

    try :

        units = await read_units(tmp_path)
        chunks = make_chunks(units , doc_id)
        await load_chunks(chunks , vs)
        add_to_index(doc_id)
    except Exception as e:
        raise HTTPException(status_code = 500 , detail  = f"Load failed {e}")
    finally:
        os.remove(tmp_path)
    return {"status": "loading done", "document_id": doc_id, "chunks": len(chunks) }





@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    vs = app.state.vectorstore
    index = app.state.index

    # 1. Check the document exists
    if index is None or not is_already_loaded(req.document_id, vs):
        raise HTTPException(status_code=404, detail="Document not loaded")

    # 2. Hybrid search over the shared index
    results = rrf_results(req.query, vs, index["bm25"], index["chunks"])

    # 3. Keep only chunks from the requested document
    results = [
        r for r in results
        if r["document"].metadata.get("document_id") == req.document_id
    ]
    if not results:
        raise HTTPException(status_code=404, detail="No matches in that document")

    # 4. Rerank and return
    ranked = rerank_results(req.query, results, req.top_k)

    return {
        "query": req.query,
        "document_id": req.document_id,
        "results": [
            {
                "chunk_id": r["document"].metadata.get("chunk_id"),
                "page": r["document"].metadata.get("page"),
                "rerank_score": float(r["rerank_score"]),
                "text": r["document"].page_content,
            }
            for r in ranked
        ],
    }




def format_results(query, ranked, document_id=None):
    return {
        "query": query,
        "document_id": document_id,
        "results": [
            {
                "document_id": r["document"].metadata.get("document_id"),
                "chunk_id": r["document"].metadata.get("chunk_id"),
                "page": r["document"].metadata.get("page"),
                "rerank_score": float(r["rerank_score"]),
                "text": r["document"].page_content,
            }
            for r in ranked
        ],
    }





@app.post("/search")
def search(req: SearchRequest):
    index = app.state.index
    if index is None:
        raise HTTPException(status_code=404, detail="No documents loaded")

    results = rrf_results(req.query, app.state.vectorstore, index["bm25"], index["chunks"])
    ranked = rerank_results(req.query, results, req.top_k)
    return format_results(req.query, ranked)


