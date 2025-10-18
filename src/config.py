# config.py

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "gdelt_comparative_eval"

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")