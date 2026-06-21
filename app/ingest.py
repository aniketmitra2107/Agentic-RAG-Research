from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from app.embeddings import get_embedder

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"


def _load(path: Path) -> list[Document]:
    if path.suffix.lower() == ".pdf":
        pages = PyPDFLoader(str(path)).load()
        return pages
    text = path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": path.name})]


def ingest():
    files = [p for p in DATA_DIR.glob("**/*") if p.is_file()]
    docs = [d for p in files for d in _load(p)]
    if not docs:
        raise SystemExit(f"No documents found in {DATA_DIR}/ — drop some .pdf/.md/.txt in there first.")

    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(docs)
    for c in chunks:
        c.metadata["source"] = Path(c.metadata.get("source", "?")).name
    sample = chunks[0]

    vec = get_embedder().embed_query(sample.page_content)

    store = Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embedder())
    store.add_documents(chunks)
    return store


if __name__ == "__main__":
    store = ingest()
    assert store._collection.count() > 0, "ingest produced an empty collection"