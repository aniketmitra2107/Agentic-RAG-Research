import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

def make_llm(**kwargs):
    """Shared LLM client - Ollama via its OpenAI-compatible /v1 endpoint."""
    return ChatOpenAI(
        model=os.environ.get("GEN_MODEL", "qwen2.5:7b-instruct"),
        temperature=0,
        api_key="ollama",
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        **kwargs
    )

def as_text(content):
    """
        Normalize a Langchain message .content to str (some providers return a list of parts).
    """
    if isinstance(content, list):
        return "".join(p if isinstance(p, str) else p.get("text", "") for p in content)
    return content if isinstance(content, str) else str(content)