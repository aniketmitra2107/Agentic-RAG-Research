from app.state import State
from app.retriever import hybrid_retrieve
from app.llm_utils import make_llm, as_text

MAX_RETRIES = 2

def retrieve(state: State) -> dict:
    return {
        "documents": hybrid_retrieve(state["question"], top_k=10),
    }

def grade(state: State) -> str:
    """
        Routing edge: are the retrieved docs good enough to answer from?
    """
    if state.get("retries", 0) >= MAX_RETRIES:
        return "generate"
    if not state.get("documents"):
        return "rewrite"
    context = "\n\n".join(d["content"] for d in state["documents"])
    verdict = as_text(make_llm().invoke("Can the question below be answered using ONLY the context? "
                                        f"Reply yes or no. \n\nQuestion: {state['question']}\n\nContext:\n{context}").content).strip().lower()
    return "generate" if verdict.startswith("y") else "rewrite"


def rewrite(state: State) -> dict:
    better = as_text(make_llm().invoke(
        "Rewrite this search query to retrieve more relevant documents. "
        f"Return only the rewritten query.\n\n{state['question']}"
    ).content).strip()
    return {"question": better, "retries": state.get("retries", 0) + 1}

def generate(state: State) -> dict:
    docs = state.get("documents", [])
    context = "\n\n".join(
        f"[{d.get("source", "?")} p{d.get("page")}]{d['content']}" for d in docs
    ) or "(no documents retireved)"
    answer = as_text(make_llm().invoke(
        "Answer the question using the context. Cite sources as [source pN]. "
        "If the context is insufficient, say so.\n\n"
        f"Question: {state["question"]}\n\nContext:\n{context}"
    ).content).strip()
    return {"answer": answer}                                    
                                        
    