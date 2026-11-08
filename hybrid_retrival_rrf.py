
import numpy as np
from recursive_chunking import make_chunks
from unified_parser import read_units
import asyncio
from tokenizer import tokenize

def hybrid_retrieval(
    query,
    vectorstore,
    bm25,
    semantic_chunks,
    dense_k=10,
    sparse_k=10,
    final_k=20,
    rrf_k=60,
    dense_weight=1.0,
    sparse_weight=1.0
):

    # --------------------------------
    # 1. Dense retrieval using ChromaDB
    # --------------------------------

    dense_results = vectorstore.similarity_search(
        query,
        k=dense_k
    )

    # --------------------------------
    # 2. Sparse retrieval using BM25
    # --------------------------------

    query_tokens = tokenize(query)

    bm25_scores = bm25.get_scores(query_tokens)

    sparse_indices = np.argsort(bm25_scores)[::-1][:sparse_k]

    sparse_results = [
        semantic_chunks[idx]
        for idx in sparse_indices
        if bm25_scores[idx] > 0
    ]

    # --------------------------------
    # 3. Reciprocal Rank Fusion
    # --------------------------------

    fused_results = {}

    def add_results(results, weight):

        for rank, doc in enumerate(results, start=1):

            # Stable chunk ID is preferred
            chunk_id = doc.metadata["chunk_id"]
            
            if chunk_id is None:
                raise ValueError("Document is missing chunk_id")


            if chunk_id not in fused_results:

                fused_results[chunk_id] = {
                    "document": doc,
                    "rrf_score": 0.0
                }

            fused_results[chunk_id]["rrf_score"] += (
                weight / (rrf_k + rank)
            )

    add_results(dense_results, dense_weight)
    add_results(sparse_results, sparse_weight)

    # --------------------------------
    # 4. Sort by fused RRF score
    # --------------------------------

    ranked_results = sorted(
        fused_results.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return ranked_results[:final_k]


# --------------------------------
# 5. Execute hybrid retrieval
# --------------------------------

def rrf_results( qry , vec_store , bm25_index  , chunks ):
    
    results = hybrid_retrieval( 
        query=qry,
        vectorstore=vec_store,
        bm25=bm25_index,
        semantic_chunks=chunks,
        dense_k=10,
        sparse_k=10,
        final_k=10
        )
    
    return results






'''
for rank, result in enumerate(results, start=1):

    doc = result["document"]

    print(f"\nRank: {rank}")
    print(f"RRF Score: {result['rrf_score']:.5f}")
    print(f"Page: {doc.metadata.get('page')}")
    print(f"Content: {doc.page_content[:300]}")
'''