# Architecture Diagrams

## Overview

This document presents comprehensive architecture diagrams for the GDELT RAG (Retrieval-Augmented Generation) system. The system implements a production-grade question-answering pipeline that retrieves relevant GDELT documents and generates grounded responses using multiple retrieval strategies.

The architecture follows a clean, layered design with clear separation of concerns:
- **Configuration Layer**: Manages external service connections (LLM, embeddings, vector store)
- **Data Layer**: Handles document loading and persistence
- **Retrieval Layer**: Implements multiple retrieval strategies (naive, BM25, ensemble, reranking)
- **Orchestration Layer**: LangGraph workflows coordinate retrieval and generation
- **Execution Layer**: Scripts and applications that orchestrate the entire pipeline

## System Architecture

The GDELT RAG system follows a modular, factory-pattern architecture with four primary layers. Each layer has specific responsibilities and dependencies flow downward through the stack.

**Key architectural decisions:**
1. **Factory Pattern**: Retrievers and graphs are created via factory functions, not instantiated at module level
2. **Lazy Initialization**: LRU-cached getters prevent premature instantiation
3. **Dependency Injection**: Graphs accept retrievers and LLMs as parameters
4. **State Management**: TypedDict-based state flows through LangGraph nodes
5. **Configuration Centralization**: All external dependencies configured in `config.py`

```mermaid
graph TB
    subgraph "Execution Layer"
        APP[app/graph_app.py<br/>LangGraph Server]
        EVAL[scripts/run_eval_harness.py<br/>RAGAS Evaluation]
        INGEST[scripts/ingest_raw_pdfs.py<br/>Data Ingestion]
    end

    subgraph "Orchestration Layer"
        GRAPH[src/graph.py<br/>build_graph<br/>build_all_graphs]
        STATE[src/state.py<br/>State TypedDict]
    end

    subgraph "Retrieval Layer"
        RET[src/retrievers.py<br/>create_retrievers]
        NAIVE[Naive Retriever<br/>Dense Vector Search]
        BM25[BM25 Retriever<br/>Sparse Keyword]
        ENS[Ensemble Retriever<br/>Dense + Sparse Hybrid]
        RERANK[Cohere Rerank<br/>Contextual Compression]
    end

    subgraph "Data Layer"
        LOAD[src/utils/loaders.py<br/>load_documents_from_huggingface<br/>load_golden_testset_from_huggingface]
        MANIFEST[src/utils/manifest.py<br/>generate_manifest]
    end

    subgraph "Configuration Layer"
        CONFIG[src/config.py]
        LLM[get_llm<br/>ChatOpenAI]
        EMB[get_embeddings<br/>OpenAIEmbeddings]
        QDRANT[get_qdrant<br/>QdrantClient]
        VS[create_vector_store<br/>QdrantVectorStore]
    end

    subgraph "External Services"
        OPENAI[OpenAI API<br/>gpt-4.1-mini<br/>text-embedding-3-small]
        QDRANT_SVC[Qdrant Vector DB<br/>localhost:6333]
        HF[HuggingFace Datasets<br/>dwb2023/gdelt-rag-sources<br/>dwb2023/gdelt-rag-golden-testset]
        COHERE[Cohere API<br/>rerank-v3.5]
    end

    subgraph "Prompts & Schema"
        PROMPTS[src/prompts.py<br/>BASELINE_PROMPT]
    end

    %% Execution dependencies
    APP --> GRAPH
    APP --> RET
    APP --> CONFIG
    APP --> LOAD

    EVAL --> GRAPH
    EVAL --> RET
    EVAL --> CONFIG
    EVAL --> LOAD
    EVAL --> MANIFEST

    INGEST --> LOAD
    INGEST --> CONFIG

    %% Orchestration dependencies
    GRAPH --> STATE
    GRAPH --> PROMPTS
    GRAPH --> LLM
    GRAPH -.uses.-> RET

    %% Retrieval dependencies
    RET --> NAIVE
    RET --> BM25
    RET --> ENS
    RET --> RERANK

    NAIVE --> VS
    BM25 -.in-memory docs.-> LOAD
    ENS --> NAIVE
    ENS --> BM25
    RERANK --> VS
    RERANK --> COHERE

    %% Data dependencies
    LOAD --> HF

    %% Config dependencies
    CONFIG --> LLM
    CONFIG --> EMB
    CONFIG --> QDRANT
    CONFIG --> VS

    LLM --> OPENAI
    EMB --> OPENAI
    QDRANT --> QDRANT_SVC
    VS --> QDRANT
    VS --> EMB

    classDef execution fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef orchestration fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef retrieval fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef data fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef config fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef external fill:#f5f5f5,stroke:#424242,stroke-width:2px
    classDef schema fill:#e0f2f1,stroke:#004d40,stroke-width:2px

    class APP,EVAL,INGEST execution
    class GRAPH,STATE orchestration
    class RET,NAIVE,BM25,ENS,RERANK retrieval
    class LOAD,MANIFEST data
    class CONFIG,LLM,EMB,QDRANT,VS config
    class OPENAI,QDRANT_SVC,HF,COHERE external
    class PROMPTS schema
```

## Component Relationships

This diagram shows how the core components interact at runtime. The system follows a clear initialization-then-execution pattern where components are bootstrapped in dependency order.

**Key interaction patterns:**
1. **Bootstrap Sequence**: Documents → Vector Store → Retrievers → Graphs
2. **Runtime Execution**: Question → Graph → Retriever → Documents → LLM → Response
3. **Caching Strategy**: LLM, embeddings, and Qdrant clients are cached via `@lru_cache`
4. **State Flow**: State objects flow through graph nodes, accumulating context and responses

```mermaid
graph LR
    subgraph "Initialization Phase"
        direction TB
        A[Load Documents<br/>from HuggingFace] --> B[Create Vector Store<br/>with Embeddings]
        B --> C[Create Retrievers<br/>4 strategies]
        C --> D[Build Graphs<br/>1 per retriever]
    end

    subgraph "Runtime Phase"
        direction TB
        E[User Question] --> F[Graph.invoke]
        F --> G[Retrieve Node<br/>Get relevant docs]
        G --> H[Generate Node<br/>LLM generates answer]
        H --> I[Response + Context]
    end

    subgraph "Configuration Services"
        direction TB
        J[get_llm<br/>Cached]
        K[get_embeddings<br/>Cached]
        L[get_qdrant<br/>Cached]
    end

    D --> F
    C --> G
    J --> H
    K --> B
    L --> B

    classDef init fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef runtime fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef config fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px

    class A,B,C,D init
    class E,F,G,H,I runtime
    class J,K,L config
```

## Data Flow Architecture

This diagram illustrates the complete data pipeline from raw PDFs through evaluation. The system implements a three-phase workflow: ingestion, retrieval, and evaluation.

**Pipeline stages:**
1. **Ingestion**: PDFs → Documents → Vector Store + HuggingFace datasets
2. **Retrieval**: Questions → Retrievers → Relevant Documents → Responses
3. **Evaluation**: Test Questions → 4 Retrievers → RAGAS Metrics → Comparative Analysis

```mermaid
flowchart TD
    subgraph "Phase 1: Data Ingestion"
        PDF[Raw PDFs<br/>data/raw/] --> LOADER[PyMuPDFLoader<br/>DirectoryLoader]
        LOADER --> DOCS[LangChain Documents<br/>List of Document]
        DOCS --> INTERIM[Interim Storage<br/>data/interim/]

        INTERIM --> JSONL[sources.docs.jsonl]
        INTERIM --> PARQUET[sources.docs.parquet]
        INTERIM --> HFDS[sources.hfds<br/>HuggingFace Dataset]

        DOCS --> RAGAS_GEN[RAGAS TestsetGenerator<br/>gpt-4.1-mini]
        RAGAS_GEN --> GOLDEN[Golden Testset<br/>Questions + Answers + Contexts]

        GOLDEN --> GT_JSONL[golden_testset.jsonl]
        GOLDEN --> GT_PARQUET[golden_testset.parquet]
        GOLDEN --> GT_HFDS[golden_testset.hfds]

        INTERIM --> MAN1[manifest.json<br/>Checksums & Provenance]
    end

    subgraph "Phase 2: RAG Retrieval Pipeline"
        HFDS2[HuggingFace Dataset<br/>dwb2023/gdelt-rag-sources] --> LOAD2[load_documents_from_huggingface]
        LOAD2 --> DOCS2[Documents]

        DOCS2 --> VS[create_vector_store<br/>QdrantVectorStore]
        DOCS2 --> RET[create_retrievers]
        VS --> RET

        RET --> R1[Naive<br/>Dense Vector]
        RET --> R2[BM25<br/>Sparse Keyword]
        RET --> R3[Ensemble<br/>Hybrid 50/50]
        RET --> R4[Cohere Rerank<br/>Contextual Compression]

        R1 --> G1[Graph: Naive]
        R2 --> G2[Graph: BM25]
        R3 --> G3[Graph: Ensemble]
        R4 --> G4[Graph: Cohere Rerank]
    end

    subgraph "Phase 3: Evaluation"
        GT_HFDS2[Golden Testset<br/>dwb2023/gdelt-rag-golden-testset] --> EVAL[RAGAS Evaluation Harness]

        G1 --> EVAL
        G2 --> EVAL
        G3 --> EVAL
        G4 --> EVAL

        EVAL --> FAITH[Faithfulness<br/>Answer grounded in context]
        EVAL --> REL[Answer Relevancy<br/>Addresses question]
        EVAL --> PREC[Context Precision<br/>Relevant docs ranked higher]
        EVAL --> REC[Context Recall<br/>Ground truth coverage]

        FAITH --> COMP[Comparative Analysis<br/>comparative_ragas_results.csv]
        REL --> COMP
        PREC --> COMP
        REC --> COMP

        COMP --> MAN2[RUN_MANIFEST.json<br/>Reproducibility Config]
    end

    classDef ingestion fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef retrieval fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef evaluation fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef artifact fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px

    class PDF,LOADER,DOCS,RAGAS_GEN,GOLDEN ingestion
    class HFDS2,LOAD2,DOCS2,VS,RET,R1,R2,R3,R4,G1,G2,G3,G4 retrieval
    class GT_HFDS2,EVAL,FAITH,REL,PREC,REC,COMP evaluation
    class JSONL,PARQUET,HFDS,GT_JSONL,GT_PARQUET,GT_HFDS,MAN1,MAN2 artifact
```

## Class Hierarchies

This diagram shows the key class structures and their relationships. The system uses composition over inheritance, with factory functions producing configured instances.

**Design patterns:**
1. **TypedDict State**: Immutable state schema with type hints
2. **Strategy Pattern**: Multiple retriever implementations with common interface
3. **Composition**: Retrievers composed into graphs, not inherited
4. **Factory Methods**: `create_retrievers()`, `build_graph()`, `create_vector_store()`

```mermaid
classDiagram
    class State {
        <<TypedDict>>
        +str question
        +List~Document~ context
        +str response
    }

    class Document {
        <<LangChain>>
        +str page_content
        +dict metadata
    }

    class BaseRetriever {
        <<Abstract Interface>>
        +invoke(query) List~Document~
    }

    class VectorStoreRetriever {
        +QdrantVectorStore vector_store
        +int k
        +invoke(query) List~Document~
    }

    class BM25Retriever {
        +List~Document~ documents
        +int k
        +invoke(query) List~Document~
    }

    class EnsembleRetriever {
        +List~BaseRetriever~ retrievers
        +List~float~ weights
        +invoke(query) List~Document~
    }

    class ContextualCompressionRetriever {
        +BaseRetriever base_retriever
        +CohereRerank compressor
        +invoke(query) List~Document~
    }

    class QdrantVectorStore {
        +QdrantClient client
        +str collection_name
        +Embeddings embedding
        +add_documents(docs)
        +as_retriever(search_kwargs) Retriever
    }

    class ChatOpenAI {
        <<LangChain LLM>>
        +str model
        +int temperature
        +invoke(messages) AIMessage
    }

    class OpenAIEmbeddings {
        <<LangChain Embeddings>>
        +str model
        +embed_query(text) List~float~
        +embed_documents(texts) List~List~float~~
    }

    class CompiledGraph {
        <<LangGraph>>
        +StateGraph graph
        +invoke(state) State
    }

    class TestsetGenerator {
        <<RAGAS>>
        +LLM llm
        +EmbeddingModel embedding_model
        +generate_with_langchain_docs(docs, testset_size)
    }

    State ..> Document : contains

    BaseRetriever <|-- VectorStoreRetriever : implements
    BaseRetriever <|-- BM25Retriever : implements
    BaseRetriever <|-- EnsembleRetriever : implements
    BaseRetriever <|-- ContextualCompressionRetriever : implements

    VectorStoreRetriever --> QdrantVectorStore : uses
    EnsembleRetriever --> VectorStoreRetriever : composes
    EnsembleRetriever --> BM25Retriever : composes
    ContextualCompressionRetriever --> VectorStoreRetriever : wraps

    QdrantVectorStore --> OpenAIEmbeddings : uses
    CompiledGraph --> BaseRetriever : uses
    CompiledGraph --> ChatOpenAI : uses
    CompiledGraph --> State : manages

    TestsetGenerator --> ChatOpenAI : uses
    TestsetGenerator --> OpenAIEmbeddings : uses
    TestsetGenerator --> Document : generates from

    classDef schema fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    classDef retriever fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef vectorstore fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef llm fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef graph fill:#fce4ec,stroke:#880e4f,stroke-width:2px

    class State,Document schema
    class BaseRetriever,VectorStoreRetriever,BM25Retriever,EnsembleRetriever,ContextualCompressionRetriever retriever
    class QdrantVectorStore vectorstore
    class ChatOpenAI,OpenAIEmbeddings,TestsetGenerator llm
    class CompiledGraph graph
```

## Module Dependencies

This diagram shows the import relationships between modules. The dependency graph is acyclic and follows clear layering principles.

**Dependency rules:**
1. **No Circular Imports**: All imports flow downward/outward
2. **External at Edges**: LangChain, OpenAI, Qdrant, RAGAS at boundaries
3. **Utils as Foundation**: Lowest-level module with no internal dependencies
4. **Config as Service Layer**: Provides cached instances to all layers
5. **Scripts Depend on Everything**: Orchestration scripts import from all modules

```mermaid
graph TB
    subgraph "Scripts Layer"
        EVAL_SCRIPT[scripts/run_eval_harness.py]
        INGEST_SCRIPT[scripts/ingest_raw_pdfs.py]
        APP_SCRIPT[app/graph_app.py]
    end

    subgraph "Core Modules"
        GRAPH_MOD[src/graph.py]
        RET_MOD[src/retrievers.py]
        CONFIG_MOD[src/config.py]
        STATE_MOD[src/state.py]
        PROMPTS_MOD[src/prompts.py]
    end

    subgraph "Utility Modules"
        UTILS_LOADERS[src/utils/loaders.py]
        UTILS_MANIFEST[src/utils/manifest.py]
    end

    subgraph "External Dependencies"
        LANGCHAIN[langchain_core<br/>langchain_openai<br/>langchain_qdrant<br/>langchain_community]
        RAGAS[ragas]
        DATASETS[datasets]
        QDRANT[qdrant_client]
        OPENAI[openai]
        COHERE[langchain_cohere]
    end

    %% Script dependencies
    EVAL_SCRIPT --> GRAPH_MOD
    EVAL_SCRIPT --> RET_MOD
    EVAL_SCRIPT --> CONFIG_MOD
    EVAL_SCRIPT --> UTILS_LOADERS
    EVAL_SCRIPT --> UTILS_MANIFEST
    EVAL_SCRIPT --> RAGAS

    INGEST_SCRIPT --> UTILS_LOADERS
    INGEST_SCRIPT --> CONFIG_MOD
    INGEST_SCRIPT --> RAGAS

    APP_SCRIPT --> GRAPH_MOD
    APP_SCRIPT --> RET_MOD
    APP_SCRIPT --> CONFIG_MOD
    APP_SCRIPT --> UTILS_LOADERS

    %% Core module dependencies
    GRAPH_MOD --> STATE_MOD
    GRAPH_MOD --> PROMPTS_MOD
    GRAPH_MOD --> CONFIG_MOD
    GRAPH_MOD --> LANGCHAIN

    RET_MOD --> CONFIG_MOD
    RET_MOD --> LANGCHAIN
    RET_MOD --> COHERE

    CONFIG_MOD --> LANGCHAIN
    CONFIG_MOD --> QDRANT
    CONFIG_MOD --> OPENAI

    STATE_MOD --> LANGCHAIN

    %% Utility dependencies
    UTILS_LOADERS --> DATASETS
    UTILS_LOADERS --> LANGCHAIN

    UTILS_MANIFEST -.no imports.-> UTILS_LOADERS

    classDef scripts fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef core fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef utils fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef external fill:#f5f5f5,stroke:#424242,stroke-width:2px

    class EVAL_SCRIPT,INGEST_SCRIPT,APP_SCRIPT scripts
    class GRAPH_MOD,RET_MOD,CONFIG_MOD,STATE_MOD,PROMPTS_MOD core
    class UTILS_LOADERS,UTILS_MANIFEST utils
    class LANGCHAIN,RAGAS,DATASETS,QDRANT,OPENAI,COHERE external
```

## Retriever Strategy Pattern

This diagram shows the retriever factory pattern and how different retrieval strategies are instantiated and used. The factory function `create_retrievers()` returns a dictionary of retriever instances, each implementing a different strategy.

**Retriever strategies:**
1. **Naive (Dense)**: Baseline semantic search using OpenAI embeddings (k=5)
2. **BM25 (Sparse)**: Lexical keyword matching over in-memory documents (k=5)
3. **Ensemble (Hybrid)**: 50/50 weighted combination of dense + sparse (k=5 each)
4. **Cohere Rerank**: Dense retrieval (k=20) → Contextual reranking → Top 3-5

**Key design decisions:**
- Factory returns dict of retrievers for easy iteration and comparison
- All strategies share common `invoke(query)` interface
- BM25 operates on in-memory docs (fast but memory-intensive)
- Reranker retrieves wide (20 docs) then compresses to top results
- Each retriever gets its own compiled graph in `build_all_graphs()`

```mermaid
graph TB
    subgraph "Factory Function"
        CREATE[create_retrievers<br/>documents, vector_store, k=5]
    end

    subgraph "Input Dependencies"
        DOCS[List of Documents<br/>from HuggingFace]
        VS[QdrantVectorStore<br/>Populated collection]
    end

    subgraph "Retriever Strategies"
        NAIVE_RET[Naive Retriever<br/>Dense Vector Search]
        BM25_RET[BM25 Retriever<br/>Sparse Keyword Match]
        ENS_RET[Ensemble Retriever<br/>Hybrid Dense + Sparse]
        RERANK_RET[Cohere Rerank<br/>Contextual Compression]
    end

    subgraph "Naive Strategy Details"
        NAIVE_IMPL[vector_store.as_retriever<br/>search_kwargs: k=5]
        NAIVE_EMB[OpenAI Embeddings<br/>text-embedding-3-small]
        NAIVE_SEARCH[Cosine Similarity<br/>in Qdrant]
    end

    subgraph "BM25 Strategy Details"
        BM25_IMPL[BM25Retriever.from_documents<br/>documents, k=5]
        BM25_INDEX[In-Memory BM25 Index<br/>TF-IDF scoring]
        BM25_KEYWORD[Keyword Matching<br/>Lexical search]
    end

    subgraph "Ensemble Strategy Details"
        ENS_IMPL[EnsembleRetriever<br/>retrievers=[naive, bm25]<br/>weights=[0.5, 0.5]]
        ENS_MERGE[Result Merging<br/>Weighted reciprocal rank fusion]
    end

    subgraph "Rerank Strategy Details"
        RERANK_WIDE[Wide Retriever<br/>k=20 docs from vector_store]
        RERANK_MODEL[CohereRerank<br/>model=rerank-v3.5]
        RERANK_COMPRESS[ContextualCompressionRetriever<br/>Rerank to top k results]
    end

    subgraph "Output Dictionary"
        OUTPUT["{\n  'naive': naive_retriever,\n  'bm25': bm25_retriever,\n  'ensemble': ensemble_retriever,\n  'cohere_rerank': compression_retriever\n}"]
    end

    subgraph "Usage in Graphs"
        BUILD_GRAPHS[build_all_graphs<br/>retrievers dict]
        GRAPH_NAIVE[Graph: Naive]
        GRAPH_BM25[Graph: BM25]
        GRAPH_ENS[Graph: Ensemble]
        GRAPH_RERANK[Graph: Cohere Rerank]
    end

    %% Factory inputs
    DOCS --> CREATE
    VS --> CREATE

    %% Factory produces strategies
    CREATE --> NAIVE_RET
    CREATE --> BM25_RET
    CREATE --> ENS_RET
    CREATE --> RERANK_RET

    %% Strategy implementations
    NAIVE_RET --> NAIVE_IMPL
    NAIVE_IMPL --> NAIVE_EMB
    NAIVE_IMPL --> NAIVE_SEARCH
    NAIVE_SEARCH --> VS

    BM25_RET --> BM25_IMPL
    BM25_IMPL --> BM25_INDEX
    BM25_INDEX --> BM25_KEYWORD
    BM25_KEYWORD --> DOCS

    ENS_RET --> ENS_IMPL
    ENS_IMPL --> NAIVE_RET
    ENS_IMPL --> BM25_RET
    ENS_IMPL --> ENS_MERGE

    RERANK_RET --> RERANK_COMPRESS
    RERANK_COMPRESS --> RERANK_WIDE
    RERANK_COMPRESS --> RERANK_MODEL
    RERANK_WIDE --> VS

    %% Output to graphs
    NAIVE_RET --> OUTPUT
    BM25_RET --> OUTPUT
    ENS_RET --> OUTPUT
    RERANK_RET --> OUTPUT

    OUTPUT --> BUILD_GRAPHS
    BUILD_GRAPHS --> GRAPH_NAIVE
    BUILD_GRAPHS --> GRAPH_BM25
    BUILD_GRAPHS --> GRAPH_ENS
    BUILD_GRAPHS --> GRAPH_RERANK

    classDef factory fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef input fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef strategy fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef impl fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef output fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef graph fill:#e0f2f1,stroke:#004d40,stroke-width:2px

    class CREATE factory
    class DOCS,VS input
    class NAIVE_RET,BM25_RET,ENS_RET,RERANK_RET strategy
    class NAIVE_IMPL,NAIVE_EMB,NAIVE_SEARCH,BM25_IMPL,BM25_INDEX,BM25_KEYWORD,ENS_IMPL,ENS_MERGE,RERANK_WIDE,RERANK_MODEL,RERANK_COMPRESS impl
    class OUTPUT output
    class BUILD_GRAPHS,GRAPH_NAIVE,GRAPH_BM25,GRAPH_ENS,GRAPH_RERANK graph
```

## LangGraph Workflow Architecture

This diagram shows the internal structure of the LangGraph workflow that orchestrates retrieval and generation. Each compiled graph follows the same two-node pattern.

**Workflow structure:**
1. **START**: Initial state with question
2. **Retrieve Node**: Invokes retriever, returns context (List[Document])
3. **Generate Node**: Formats prompt with context, invokes LLM, returns response
4. **END**: Final state with question, context, and response

**State management:**
- Nodes return partial state dicts (e.g., `{"context": docs}`)
- LangGraph automatically merges updates into state
- State accumulates: question → question+context → question+context+response
- Final state contains full provenance chain

```mermaid
graph LR
    subgraph "Graph Factory"
        BUILD[build_graph<br/>retriever, llm, prompt_template]
    end

    subgraph "LangGraph Workflow"
        START([START]) --> RETRIEVE[Retrieve Node<br/>retriever.invoke]
        RETRIEVE --> GENERATE[Generate Node<br/>llm.invoke with prompt]
        GENERATE --> END([END])
    end

    subgraph "Retrieve Node Logic"
        R1[state: question] --> R2[retriever.invoke<br/>state.question]
        R2 --> R3[docs: List of Document]
        R3 --> R4["return {context: docs}"]
    end

    subgraph "Generate Node Logic"
        G1["state: question + context"] --> G2[Join context docs<br/>page_content]
        G2 --> G3[Format RAG prompt<br/>question + context]
        G3 --> G4[llm.invoke<br/>prompt_messages]
        G4 --> G5[response: AIMessage]
        G5 --> G6["return {response: response.content}"]
    end

    subgraph "State Evolution"
        S1["Initial State<br/>{question: str}"]
        S2["After Retrieve<br/>{question: str,<br/>context: List of Document}"]
        S3["After Generate<br/>{question: str,<br/>context: List of Document,<br/>response: str}"]

        S1 --> S2
        S2 --> S3
    end

    subgraph "Dependencies"
        RET_DEP[Retriever Instance<br/>naive/bm25/ensemble/rerank]
        LLM_DEP[ChatOpenAI<br/>gpt-4.1-mini, temp=0]
        PROMPT_DEP[BASELINE_PROMPT<br/>Template string]
    end

    RET_DEP --> BUILD
    LLM_DEP --> BUILD
    PROMPT_DEP --> BUILD

    BUILD --> RETRIEVE
    BUILD --> GENERATE

    RETRIEVE --> R1
    GENERATE --> G1

    START --> S1
    RETRIEVE --> S2
    GENERATE --> S3
    S3 --> END

    classDef factory fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef node fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef logic fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef state fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef dep fill:#fce4ec,stroke:#880e4f,stroke-width:2px

    class BUILD factory
    class START,RETRIEVE,GENERATE,END node
    class R1,R2,R3,R4,G1,G2,G3,G4,G5,G6 logic
    class S1,S2,S3 state
    class RET_DEP,LLM_DEP,PROMPT_DEP dep
```

## Evaluation Pipeline Architecture

This diagram shows the complete RAGAS evaluation pipeline implemented in `run_eval_harness.py`. The pipeline evaluates all four retriever strategies against a golden testset using four RAGAS metrics.

**Evaluation workflow:**
1. **Data Loading**: Load source documents + golden testset from HuggingFace
2. **RAG Stack Setup**: Build vector store → retrievers → graphs
3. **Inference**: Run 12 questions × 4 retrievers = 48 Q&A pairs
4. **RAGAS Evaluation**: Compute 4 metrics per retriever (48 total metric scores)
5. **Comparative Analysis**: Aggregate scores, rank retrievers, identify winner

**RAGAS metrics:**
- **Faithfulness**: Answer grounded in retrieved context (hallucination detection)
- **Answer Relevancy**: Answer addresses the question asked
- **Context Precision**: Relevant contexts ranked higher than irrelevant
- **Context Recall**: Retrieved contexts cover ground truth reference contexts

```mermaid
graph TB
    subgraph "Phase 1: Data Loading"
        HF_SRC[HuggingFace Dataset<br/>dwb2023/gdelt-rag-sources] --> LOAD_SRC[load_documents_from_huggingface]
        HF_GOLD[HuggingFace Dataset<br/>dwb2023/gdelt-rag-golden-testset] --> LOAD_GOLD[load_golden_testset_from_huggingface]

        LOAD_SRC --> DOCS[38 Source Documents]
        LOAD_GOLD --> GOLDEN[12 Test Questions<br/>+ Reference Answers<br/>+ Reference Contexts]
    end

    subgraph "Phase 2: RAG Stack Build"
        DOCS --> VS_BUILD[create_vector_store<br/>recreate_collection=False]
        VS_BUILD --> VS[QdrantVectorStore<br/>gdelt_comparative_eval]

        DOCS --> RET_BUILD[create_retrievers<br/>k=5]
        VS --> RET_BUILD

        RET_BUILD --> R1[naive]
        RET_BUILD --> R2[bm25]
        RET_BUILD --> R3[ensemble]
        RET_BUILD --> R4[cohere_rerank]

        R1 --> G1[build_graph<br/>Graph: Naive]
        R2 --> G2[build_graph<br/>Graph: BM25]
        R3 --> G3[build_graph<br/>Graph: Ensemble]
        R4 --> G4[build_graph<br/>Graph: Cohere Rerank]
    end

    subgraph "Phase 3: Inference (12 questions × 4 retrievers)"
        GOLDEN --> Q1[Question 1]
        GOLDEN --> Q2[Question 2]
        GOLDEN --> Q_DOTS[...]
        GOLDEN --> Q12[Question 12]

        Q1 --> G1
        Q1 --> G2
        Q1 --> G3
        Q1 --> G4

        G1 --> RES1[Response + Contexts]
        G2 --> RES2[Response + Contexts]
        G3 --> RES3[Response + Contexts]
        G4 --> RES4[Response + Contexts]
    end

    subgraph "Phase 4: RAGAS Evaluation"
        RES1 --> EVAL1[evaluate<br/>Naive Dataset]
        RES2 --> EVAL2[evaluate<br/>BM25 Dataset]
        RES3 --> EVAL3[evaluate<br/>Ensemble Dataset]
        RES4 --> EVAL4[evaluate<br/>Cohere Rerank Dataset]

        EVAL1 --> M1[Faithfulness<br/>Answer Relevancy<br/>Context Precision<br/>Context Recall]
        EVAL2 --> M2[Faithfulness<br/>Answer Relevancy<br/>Context Precision<br/>Context Recall]
        EVAL3 --> M3[Faithfulness<br/>Answer Relevancy<br/>Context Precision<br/>Context Recall]
        EVAL4 --> M4[Faithfulness<br/>Answer Relevancy<br/>Context Precision<br/>Context Recall]
    end

    subgraph "Phase 5: Comparative Analysis"
        M1 --> AGG[Aggregate Scores]
        M2 --> AGG
        M3 --> AGG
        M4 --> AGG

        AGG --> COMP[Comparative Table<br/>Rank by Average Score]
        COMP --> WINNER[Identify Winner<br/>vs Baseline]
    end

    subgraph "Phase 6: Artifacts"
        RES1 --> RAW1[naive_raw_dataset.parquet]
        RES2 --> RAW2[bm25_raw_dataset.parquet]
        RES3 --> RAW3[ensemble_raw_dataset.parquet]
        RES4 --> RAW4[cohere_rerank_raw_dataset.parquet]

        M1 --> DET1[naive_detailed_results.csv]
        M2 --> DET2[bm25_detailed_results.csv]
        M3 --> DET3[ensemble_detailed_results.csv]
        M4 --> DET4[cohere_rerank_detailed_results.csv]

        COMP --> COMP_CSV[comparative_ragas_results.csv]
        WINNER --> MANIFEST[RUN_MANIFEST.json<br/>Full reproducibility config]
    end

    classDef loading fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef build fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef inference fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef evaluation fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef analysis fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef artifacts fill:#e0f2f1,stroke:#004d40,stroke-width:2px

    class HF_SRC,HF_GOLD,LOAD_SRC,LOAD_GOLD,DOCS,GOLDEN loading
    class VS_BUILD,VS,RET_BUILD,R1,R2,R3,R4,G1,G2,G3,G4 build
    class Q1,Q2,Q_DOTS,Q12,RES1,RES2,RES3,RES4 inference
    class EVAL1,EVAL2,EVAL3,EVAL4,M1,M2,M3,M4 evaluation
    class AGG,COMP,WINNER analysis
    class RAW1,RAW2,RAW3,RAW4,DET1,DET2,DET3,DET4,COMP_CSV,MANIFEST artifacts
```

---

## Summary

These diagrams comprehensively document the GDELT RAG system architecture:

1. **System Architecture**: Shows the 5-layer stack from execution down to external services
2. **Component Relationships**: Illustrates initialization vs runtime phases and component interactions
3. **Data Flow Architecture**: Traces the complete pipeline from PDFs to evaluation results
4. **Class Hierarchies**: Documents the object model and design patterns
5. **Module Dependencies**: Maps import relationships and dependency flow
6. **Retriever Strategy Pattern**: Details the factory pattern and retriever implementations
7. **LangGraph Workflow**: Shows the internal graph structure and state management
8. **Evaluation Pipeline**: Documents the RAGAS evaluation workflow end-to-end

Key architectural principles:
- **Factory Pattern**: Deferred instantiation via factory functions
- **Dependency Injection**: Components accept dependencies as parameters
- **Caching**: LRU-cached singletons for expensive resources
- **Separation of Concerns**: Clear boundaries between layers
- **Reproducibility**: Manifest generation for full provenance tracking
