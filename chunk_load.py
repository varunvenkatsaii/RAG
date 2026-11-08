from langchain_chroma import Chroma
from recursive_chunking import make_chunks
import asyncio
import re 
from tokenizer import tokenize
from rank_bm25 import BM25Okapi


async def load_chunks(chunks , vector_store  , batch_size=100 , concurrency = 4, ):
    sem = asyncio.Semaphore(concurrency)
    async def add_batch(batch):
        async with sem:
            await vector_store.aadd_documents(batch , ids=[c.metadata["chunk_id"] for c in batch] )

    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    await asyncio.gather(*(add_batch(b) for b in batches))


    print("Successfully stored semantic chunks!")



def bm25_index( chunks ):
    tokenized_corpus = [
    tokenize(doc.page_content)
    for doc in chunks
    ]
    bm25_index = BM25Okapi(tokenized_corpus)
    return bm25_index
