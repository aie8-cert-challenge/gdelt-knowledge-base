# graph.py

# LangChain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# LangGraph
from langgraph.graph import START, StateGraph

# import src/prompts.py and src/retrievers.py
from src.prompts import BASELINE_PROMPT
from src.retrievers import baseline_retriever, bm25_retriever, compression_retriever, ensemble_retriever
from src.state import State

# Configuration
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
rag_prompt = ChatPromptTemplate.from_template(BASELINE_PROMPT)

# Modular retriever functions (following session08 pattern)
def retrieve_baseline(state):
    """Naive dense vector search"""
    retrieved_docs = baseline_retriever.invoke(state["question"])
    return {"context": retrieved_docs}

def retrieve_bm25(state):
    """BM25 sparse keyword matching"""
    retrieved_docs = bm25_retriever.invoke(state["question"])
    return {"context": retrieved_docs}

def retrieve_reranked(state):
    """Cohere contextual compression with reranking"""
    retrieved_docs = compression_retriever.invoke(state["question"])
    return {"context": retrieved_docs}

def retrieve_ensemble(state):
    """Ensemble hybrid search (dense + sparse)"""
    retrieved_docs = ensemble_retriever.invoke(state["question"])
    return {"context": retrieved_docs}

# Shared generate function
def generate(state):
    """Generate answer from context"""
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = rag_prompt.format_messages(question=state["question"], context=docs_content)
    response = llm.invoke(messages)
    return {"response": response.content}

# Create LangGraphs for each retriever
print("   Creating LangGraphs...")
baseline_graph_builder = StateGraph(State).add_sequence([retrieve_baseline, generate])
baseline_graph_builder.add_edge(START, "retrieve_baseline")
baseline_graph = baseline_graph_builder.compile()

bm25_graph_builder = StateGraph(State).add_sequence([retrieve_bm25, generate])
bm25_graph_builder.add_edge(START, "retrieve_bm25")
bm25_graph = bm25_graph_builder.compile()

ensemble_graph_builder = StateGraph(State).add_sequence([retrieve_ensemble, generate])
ensemble_graph_builder.add_edge(START, "retrieve_ensemble")
ensemble_graph = ensemble_graph_builder.compile()

rerank_graph_builder = StateGraph(State).add_sequence([retrieve_reranked, generate])
rerank_graph_builder.add_edge(START, "retrieve_reranked")
rerank_graph = rerank_graph_builder.compile()
# Configure retrievers
retrievers_config = {
    "naive": baseline_graph,
    "bm25": bm25_graph,
    "ensemble": ensemble_graph,
    "cohere_rerank": rerank_graph,
}
