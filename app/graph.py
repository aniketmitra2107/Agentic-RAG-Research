from langgraph.graph import StateGraph, START, END
from app.state import State
from app.nodes import retrieve, rewrite, generate, grade

_app = None

def build():
    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)

    g.add_edge(START, "retrieve")
    g.add_conditional_edges(
        "retrieve",
        grade,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    return g.compile()

def answer(question: str) -> str:
    global _app
    if _app is None:
        _app = build()
    return _app.invoke({"question": question})["answer"]


if __name__ == "__main__":
    import sys
    print(answer(sys.argv[1] if len(sys.argv) > 1 else "What is this about?"))