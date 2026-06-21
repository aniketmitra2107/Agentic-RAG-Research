from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

_embedder = None
_reranker = None

def get_embedder():
    """all-MiniLM-L6-v2 - small, CPU-fine, no API key. Cached after the first load."""
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embedder

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker