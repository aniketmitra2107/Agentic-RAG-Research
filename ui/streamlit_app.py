import streamlit as st
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from app.graph import build

st.title("Agentic RAG")

app = st.cache_resource(build)()

question = st.text_input("Ask a question")
if question: 
    with st.spinner("Thinking..."):
        result = app.invoke({"question": question})
    st.markdown(result["answer"])
    with st.expander(f"Soruces ({len(result.get('documents', []))})"):
        for d in result.get("documents", []):
            st.markdown(f"**{d.get('source', '?')}** p{d.get('page')} . score {d.get('score', 0):.2f}")
            st.caption(d["content"][:300] + " ... ")