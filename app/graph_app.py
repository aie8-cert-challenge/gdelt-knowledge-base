# app/graph_app.py
from src.graph import build_all_graphs
from src.retrievers import create_retrievers
from src.config import create_vector_store, get_llm
from src.utils import load_documents_from_huggingface

def get_app():
    """
    LangGraph Server entrypoint.
    Returns a CompiledGraph (or LangGraphApp) to serve via LangGraph Server.
    """
    docs = load_documents_from_huggingface()
    vs = create_vector_store(docs, recreate_collection=False)
    rets = create_retrievers(docs, vs, k=5)
    llm = get_llm()
    graphs = build_all_graphs(rets, llm=llm)
    return graphs["cohere_rerank"]  # default; can switch later by state
