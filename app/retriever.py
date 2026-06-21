from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from app.embeddings import get_embedder, get_reranker
from app.ingest import CHROMA_DIR

_store = None
_bm25 = None


def _init():
    global _store, _bm25
    if _store is None:
        _store = Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embedder())
        stored = _store.get()                       # pull chunks back to seed BM25
        _bm25 = BM25Retriever.from_texts(stored["documents"], metadatas=stored["metadatas"])
        _bm25.k = 10
    return _store, _bm25


def hybrid_retrieve(query: str, top_k: int = 4) -> list[dict]:
    store, bm25 = _init()
    candidates = store.similarity_search(query, k=10) + bm25.invoke(query)   # dense + keyword
    print(f"candidates: {len(candidates)}")
    seen, unique = set(), []
    for d in candidates:                            
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique.append(d)

    scores = get_reranker().predict([(query, d.page_content) for d in unique])
    ranked = sorted(zip(scores, unique), key=lambda x: x[0], reverse=True)
    return [
        {"content": d.page_content, "source": d.metadata.get("source", "?"),
        "page": d.metadata.get("page"), "score": float(s)}
        for s, d in ranked[:top_k]
    ]


if __name__ == "__main__":
    import sys
    hits = hybrid_retrieve(sys.argv[1] if len(sys.argv) > 1 else "what is this about?")
    assert hits, "retrieval returned nothing"  
    for h in hits:
        print(f"[{h['score']:.2f}] {h['source']} p{h['page']}: {h['content'][:80]}...")
