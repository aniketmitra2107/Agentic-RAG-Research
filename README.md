# Agentic RAG

A local, fully self-hosted **agentic Retrieval-Augmented Generation** pipeline built with [LangGraph](https://github.com/langchain-ai/langgraph). It doesn't just retrieve-then-answer — it **grades** its own retrieved context and **rewrites the query and retries** when the context isn't good enough, before generating a cited answer.

Everything runs on your machine: embeddings and reranking via local Sentence-Transformers, generation via [Ollama](https://ollama.com/). No API keys, no data leaving the box.

---

## How it works

```
            ┌──────────┐      grade: good?
  question ─►│ retrieve │──────────┬──────────► generate ─► answer
            └──────────┘          │
                 ▲                │ grade: weak?
                 │                ▼
                 └──────────── rewrite  (bounded by MAX_RETRIES)
```

1. **retrieve** — hybrid search: dense (Chroma + MiniLM embeddings) **+** keyword (BM25), deduped, then re-ranked by a cross-encoder.
2. **grade** — an LLM judges whether the retrieved chunks can actually answer the question.
3. **rewrite** — if not, the LLM rephrases the query and we loop back to retrieve (up to `MAX_RETRIES`).
4. **generate** — synthesize an answer from the context, with inline `[source pN]` citations.

## Stack

| Concern | Choice |
|---|---|
| Orchestration | LangGraph `StateGraph` |
| Vector store | Chroma (persisted to `chroma_db/`) |
| Embeddings | `all-MiniLM-L6-v2` (local, CPU-fine) |
| Keyword search | BM25 (`rank-bm25`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | Ollama (`qwen2.5:7b-instruct` by default) |
| UI | Streamlit |

## Project layout

```
app/
  embeddings.py   # cached embedder + cross-encoder reranker
  ingest.py       # load docs → chunk → embed → Chroma
  retriever.py    # hybrid_retrieve: dense + BM25 + rerank
  state.py        # LangGraph State (question/documents/answer/retries)
  nodes.py        # retrieve / grade / rewrite / generate
  graph.py        # wires the nodes into a StateGraph
  llm_utils.py    # shared Ollama client
ui/
  streamlit_app.py
data/             # drop your .pdf / .md / .txt here
```

## Setup

**1. Install Ollama and pull a model**
```bash
ollama pull qwen2.5:7b-instruct
```

**2. Python deps** (a virtualenv is recommended)
```bash
pip install -r requirements.txt
```

**3. Config** — copy the example env file
```bash
cp .env.example .env
```
```ini
OLLAMA_BASE_URL=http://localhost:11434/v1
GEN_MODEL=qwen2.5:7b-instruct
```

## Usage

**1. Add documents** to `data/` (PDF, Markdown, or text), then ingest:
```bash
python -m app.ingest
```

**2. Run the UI:**
```bash
streamlit run ui/streamlit_app.py
```
> If you hit `No module named app`, run from the project root with `PYTHONPATH=.` prefixed, or add the project root to `sys.path` at the top of the script.

**Or query from the command line:**
```bash
python -m app.graph "what is this about?"
```

**Test retrieval alone:**
```bash
python -m app.retriever "your query"
```

## Notes

- First run downloads the embedding + reranker models (a few hundred MB), cached thereafter.
- The chunker uses `chunk_size=800`, `chunk_overlap=120` — tune in `app/ingest.py`.
- The rewrite loop is bounded by `MAX_RETRIES = 2` in `app/nodes.py`.
- On some Windows + NVIDIA setups Ollama's CUDA backend crashes on init; running Ollama with the GPU hidden (`CUDA_VISIBLE_DEVICES=""`) forces a stable CPU fallback.
