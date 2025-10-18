# retrievers.py

# LangChain
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever

# LangChain Integrations
from langchain_cohere import CohereRerank
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

# Qdrant
from qdrant_client import QdrantClient

# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "gdelt_comparative_eval"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Check if collection exists, recreate if needed
collections = qdrant_client.get_collections().collections
collection_names = [c.name for c in collections]

# Create vector store
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# Baseline: Dense vector search (k=5)
baseline_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# BM25: Sparse keyword matching
bm25_retriever = BM25Retriever.from_documents(documents, k=5)

# Cohere Rerank: Contextual compression (retrieve 20, rerank to 5)
baseline_retriever_20 = vector_store.as_retriever(search_kwargs={"k": 20})

# Ensemble: Hybrid search (dense + sparse)
ensemble_retriever = EnsembleRetriever(
    retrievers=[baseline_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

# Cohere Rerank: Contextual compression (retrieve 20, rerank to 5)
baseline_retriever_20 = vector_store.as_retriever(search_kwargs={"k": 20})
compressor = CohereRerank(model="rerank-v3.5")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=baseline_retriever_20,
    search_kwargs={"k": 5}
)