from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu",
    max_length=512
)

def rerank_results(query, fused_results, top_k=5):

    docs = [item["document"] for item in fused_results]

    pairs = [
        (query, doc.page_content)
        for doc in docs
    ]

    scores = reranker.predict(pairs, batch_size=8)

    ranked_results = sorted(
        zip(scores, docs),
        key=lambda x: float(x[0]),
        reverse=True
    )

    return [
        {
            "document": doc,
            "rerank_score": float(score)
        }
        for score, doc in ranked_results[:top_k]
    ]



