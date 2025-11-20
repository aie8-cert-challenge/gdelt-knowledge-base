# LangGraph API Collection - GDELT Knowledge Base

A comprehensive Bruno API collection for testing and exploring the GDELT Knowledge Base LangGraph server. This collection provides complete lifecycle testing from health checks to HITL workflows.

> **✅ Validated Against**: langgraph-api 0.5.20, LangGraph 1.0.1
> **📅 Last Updated**: 2025-11-20
> **🎯 Target**: Self-hosted LangGraph Server (local development)
> **📊 Working Endpoints**: 13/16 (3 marked as educational/not available)

## 📋 Table of Contents

- [Overview](#overview)
- [Version Compatibility](#version-compatibility)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Collection Structure](#collection-structure)
- [Execution Sequences](#execution-sequences)
- [Environment Variables](#environment-variables)
- [Performance & Cost](#performance--cost)
- [Common Workflows](#common-workflows)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Overview

This Bruno collection tests the **GDELT Knowledge Base** LangGraph server, which implements a production RAG (Retrieval-Augmented Generation) system for GDELT documentation.

### Architecture

```
Question → [retrieve node] → [generate node] → Answer
              ↓                    ↓
         Cohere Rerank        GPT-4.1-mini
         (k=20 → k=5)         (temperature=0)
```

**Key Components**:
- **Retriever**: Cohere Rerank v3.5 (best performing, 94.43% avg score)
- **Vector Store**: Qdrant (local, collection: `gdelt_comparative_eval`)
- **Embeddings**: OpenAI text-embedding-3-small (1536 dimensions)
- **LLM**: GPT-4.1-mini (deterministic, temperature=0)
- **Documents**: 38 GDELT documentation PDFs

## Version Compatibility

**Validated Against**:
- **langgraph-api**: 0.5.20
- **LangGraph**: 1.0.1
- **Host**: Self-hosted (local development)

**Endpoint Status**:

| Category | Working | Not Available | Notes |
|----------|---------|---------------|-------|
| Health & Info | 1/1 | - | Use `/ok` not `/health` |
| Introspection | 3/3 | - | Schemas, graph, subgraphs |
| Threads | 2/2 | - | Create, get |
| Runs | 4/4 | - | Wait, stream, status, history |
| State | 1/1 | - | Get thread state |
| Debug | 2/3 | 1 | Config endpoint N/A in 0.5.20 |
| HITL | 0/2 | 2 | Use interrupt pattern instead |
| **Total** | **13/16** | **3/16** | **81% functional** |

**Not Available in 0.5.20**:
- ❌ `/assistants/{id}/config` - Use `/assistants/{id}/schemas` instead
- ❌ `/threads/{id}/events` (GET/POST) - Use interrupt pattern in `/runs/wait`

**See**: [VERSION_VALIDATION.md](VERSION_VALIDATION.md) for detailed validation report

## Prerequisites

### 1. Start the LangGraph Server

```bash
# Navigate to project root
cd /home/donbr/don-aie-cohort8/gdelt-knowledge-base

# Activate virtual environment
source .venv/bin/activate

# Start Qdrant (if not already running)
make qdrant-up

# Start LangGraph server
uv run langgraph dev --allow-blocking
```

**Server URLs**:
- API: `http://127.0.0.1:2024`
- Studio UI: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`
- Docs: `http://127.0.0.1:2024/docs`

### 2. Environment Variables

Ensure these are set:

```bash
export OPENAI_API_KEY="sk-..."
export COHERE_API_KEY="..."  # For reranking
export QDRANT_URL="http://localhost:6333"
```

### 3. Install Bruno

Download from [usebruno.com](https://www.usebruno.com/)

## Quick Start

### 1. Import Collection

1. Open Bruno
2. Click "Open Collection"
3. Navigate to: `/home/donbr/don-aie-cohort8/gdelt-knowledge-base/bruno/langgraph-api/`
4. Select the folder

### 2. Select Environment

- **local-dev**: Pre-configured for local development (default)
- **production**: Template for LangSmith deployment

### 3. Run First Request

```
01-introspection → get-schemas.bru
```

Click **Send** → You should see input/output schemas for the graph

### 4. Execute Full Workflow

Run requests in order:
1. `02-threads/create-thread.bru` → Creates thread, captures `thread_id`
2. `03-runs-non-streaming/run-wait.bru` → Executes RAG query
3. `05-results/get-state.bru` → Retrieves complete answer

## Collection Structure

```
bruno/langgraph-api/
├── README.md                    # This file
├── bruno.json                   # Collection configuration
├── environments/
│   ├── local-dev.bru           # Local development (http://127.0.0.1:2024)
│   └── production.bru          # Production template
└── collections/
    ├── 00-health/              # Health checks
    │   └── health-check.bru
    ├── 01-introspection/       # Schema & graph structure
    │   ├── get-schemas.bru
    │   ├── get-graph.bru
    │   └── get-subgraphs.bru
    ├── 02-threads/             # Thread lifecycle
    │   ├── create-thread.bru
    │   └── get-thread.bru
    ├── 03-runs-non-streaming/  # Synchronous execution
    │   └── run-wait.bru
    ├── 04-runs-streaming/      # Asynchronous execution
    │   └── run-stream.bru
    ├── 05-results/             # State & history
    │   ├── get-run.bru
    │   ├── get-state.bru
    │   └── get-history.bru
    ├── 06-hitl/                # Human-in-the-loop
    │   ├── get-events.bru
    │   └── post-event.bru
    └── 07-debug/               # Utilities
        ├── list-assistants.bru
        ├── get-assistant.bru
        └── get-config.bru
```

## Execution Sequences

### Sequence 1: Simple RAG Query

**Purpose**: Execute a single question → answer workflow

```
1. 02-threads/create-thread.bru       [Captures thread_id]
2. 03-runs-non-streaming/run-wait.bru [Executes graph]
   → Response includes complete state (question + context + response)
```

**Time**: ~10-12 seconds
**Cost**: ~$0.0021

### Sequence 2: Multi-Turn Conversation

**Purpose**: Multiple queries on same thread (maintains context)

```
1. 02-threads/create-thread.bru       [Captures thread_id]
2. 03-runs-non-streaming/run-wait.bru [First query]
3. 05-results/get-state.bru           [See accumulated state]
4. 03-runs-non-streaming/run-wait.bru [Second query, same thread_id]
5. 05-results/get-state.bru           [See full conversation]
```

**Time**: ~20-24 seconds (2 queries)
**Cost**: ~$0.0042

### Sequence 3: Debug Graph Execution

**Purpose**: Inspect node-by-node state evolution

```
1. 02-threads/create-thread.bru
2. 03-runs-non-streaming/run-wait.bru [Captures run_id]
3. 05-results/get-history.bru         [See all checkpoints]
   → Checkpoint 0: __start__ → {question}
   → Checkpoint 1: retrieve → {question, context}
   → Checkpoint 2: generate → {question, context, response}
```

**Use Case**: Time-travel debugging, performance analysis

### Sequence 4: Streaming Response

**Purpose**: Get real-time updates during execution

```
1. 02-threads/create-thread.bru
2. 04-runs-streaming/run-stream.bru
   → Stream event 1: {question}
   → Stream event 2: {question, context}
   → Stream event 3: {question, context, response}
```

**Note**: Bruno may not render SSE streams well. Use `curl` or LangGraph Studio for better streaming visualization.

### Sequence 5: HITL Workflow (Educational)

**Purpose**: Learn HITL patterns (not applicable to GDELT graph)

```
1. 02-threads/create-thread.bru
2. 03-runs-non-streaming/run-wait.bru [Would interrupt if configured]
3. 06-hitl/get-events.bru             [Check for interrupts]
4. 06-hitl/post-event.bru             [Provide human input]
5. 05-results/get-run.bru             [Verify run resumed]
```

**Note**: Current GDELT graph has no interrupt nodes. This sequence is for learning purposes.

## Environment Variables

### Local Development (local-dev.bru)

```javascript
{
  base_url: "http://127.0.0.1:2024",
  assistant_id: "e45a51c7-e110-55eb-b8c6-c2b51f0c2c8f",
  graph_id: "gdelt",
  thread_id: "",  // Auto-captured from create-thread.bru
  run_id: ""      // Auto-captured from run-wait.bru
}
```

**Auto-Capture Scripts**:
- `create-thread.bru` → Sets `thread_id` environment variable
- `run-wait.bru` → Sets `run_id` environment variable

### Production (production.bru)

```javascript
{
  base_url: "https://your-langsmith-deployment-url.com",
  assistant_id: "",  // Get from LangSmith deployment
  graph_id: "gdelt",
  thread_id: "",
  run_id: ""
}
vars:secret [
  LANGCHAIN_API_KEY
]
```

## Performance & Cost

### Per-Query Breakdown

| Phase | Duration | External API Calls | Cost |
|-------|----------|-------------------|------|
| **Initialization** | ~2s | HuggingFace (5×), Qdrant (2×), OpenAI (1×) | ~$0.0001 |
| **Retrieve Node** | ~2.5s | OpenAI embed (1×), Qdrant query (1×), Cohere rerank (1×) | ~$0.001 |
| **Generate Node** | ~6.6s | OpenAI chat (1×) | ~$0.001 |
| **Total** | **~10.5s** | **14 calls** | **~$0.0021** |

### Latency Details

- **OpenAI Embeddings**: 398ms
- **Qdrant Vector Search**: 20ms (local)
- **Cohere Rerank**: 552ms
- **OpenAI Chat Completion**: 6.6s

### Optimization Notes

⚠️ **Initialization Overhead**: Each schema introspection call (Studio UI loads) triggers full retriever initialization:
- 8 introspection calls × 8 external APIs = 64 unnecessary calls
- Known issue in `langgraph-api 0.4.48`
- Consider caching schemas client-side

## Common Workflows

### Test Different Questions

Edit `run-wait.bru` body:

```json
{
  "assistant_id": "{{assistant_id}}",
  "input": {
    "question": "How does GDELT monitor news?"  // Change this
  },
  "stream_mode": ["values"]
}
```

**Sample Questions**:
- "What is GDELT?"
- "How does GDELT monitor news?"
- "What data formats does GDELT support?"
- "Explain GDELT's event database structure"
- "What is the CAMEO event classification?"
- "How does GDELT handle multilingual data?"

### Test Different Configurations

Add `config` to run request:

```json
{
  "assistant_id": "{{assistant_id}}",
  "input": {"question": "..."},
  "config": {
    "configurable": {
      "temperature": 0.5,     // More creative (default: 0)
      "k_documents": 10       // More context (default: 5)
    }
  }
}
```

### Measure Response Times

1. Run `03-runs-non-streaming/run-wait.bru`
2. Check Bruno's response time in bottom status bar
3. Compare with server logs for node-by-node breakdown

### Export Conversation

1. Run multi-turn sequence
2. `05-results/get-history.bru`
3. Copy response JSON
4. Process checkpoints to extract conversation flow

## Troubleshooting

### Server Connection Issues

**Error**: `Failed to connect to http://127.0.0.1:2024`

**Solutions**:
```bash
# Check if server is running
curl http://127.0.0.1:2024/health

# Start server if not running
uv run langgraph dev --allow-blocking

# Check for port conflicts
lsof -i :2024
```

### Thread Not Found

**Error**: `404 Thread not found`

**Solutions**:
1. Run `02-threads/create-thread.bru` first
2. Check that `thread_id` variable is set (bottom panel → Vars)
3. Server may have restarted (threads are in-memory)

### Timeout Errors

**Error**: Request timeout after 30s

**Solutions**:
1. Increase Bruno timeout: Settings → Request → Timeout → 60s
2. Check Qdrant is running: `curl http://localhost:6333`
3. Verify API keys are set: `echo $OPENAI_API_KEY`
4. Check server logs for errors

### Invalid Assistant ID

**Error**: `404 Assistant not found`

**Solutions**:
1. Run `07-debug/list-assistants.bru` to get valid IDs
2. Update `local-dev.bru` environment with correct `assistant_id`
3. Verify graph is registered in `langgraph.json`

### Streaming Not Working

**Issue**: Streaming request shows incomplete response

**Note**: Bruno's HTTP client doesn't fully support Server-Sent Events (SSE).

**Alternatives**:
```bash
# Use curl instead
curl -X POST http://127.0.0.1:2024/threads/{thread_id}/runs/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  --no-buffer \
  -d '{"assistant_id": "...", "input": {"question": "..."}}'

# Or use LangGraph Studio UI
# https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### Empty Events (HITL)

**Behavior**: `06-hitl/get-events.bru` returns `{"events": []}`

**Expected**: Current GDELT graph has no interrupt nodes.

**To Test HITL**:
1. Modify graph to include interrupts
2. Or use a different multi-agent graph with approval gates

## API Reference

### Base URL

```
http://127.0.0.1:2024
```

### Authentication

None required (local development)

### Endpoints Summary

| Category | Endpoint | Method | Purpose |
|----------|----------|--------|---------|
| **Health** | `/health` | GET | Server health check |
| **Introspection** | `/assistants/{id}/schemas` | GET | Input/output schemas |
| | `/assistants/{id}/graph` | GET | Graph structure |
| | `/assistants/{id}/subgraphs` | GET | Nested graphs |
| **Threads** | `/threads` | POST | Create thread |
| | `/threads/{id}` | GET | Get thread metadata |
| **Runs** | `/threads/{id}/runs/wait` | POST | Execute graph (blocking) |
| | `/threads/{id}/runs/stream` | POST | Execute graph (streaming) |
| **Results** | `/threads/{id}/runs/{run_id}` | GET | Run status |
| | `/threads/{id}/state` | GET | Current state |
| | `/threads/{id}/history` | GET | Checkpoint history |
| **HITL** | `/threads/{id}/events` | GET | Pending interrupts |
| | `/threads/{id}/events` | POST | Respond to interrupt |
| **Debug** | `/assistants` | GET | List all assistants |
| | `/assistants/{id}` | GET | Assistant details |
| | `/assistants/{id}/config` | GET | Configuration schema |

### Request/Response Formats

All requests use `Content-Type: application/json`
All responses are JSON except streaming (`text/event-stream`)

## Additional Resources

### Documentation

- **Project README**: `../../README.md`
- **CLAUDE.md**: `../../CLAUDE.md` (architecture guide)
- **Makefile Guide**: `../../README_MAKEFILE.md`
- **Scripts Guide**: `../../scripts/README.md`

### LangGraph Resources

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangGraph Studio](https://smith.langchain.com/studio/)
- [LangSmith](https://smith.langchain.com/)

### External APIs

- [OpenAI API Docs](https://platform.openai.com/docs)
- [Cohere Rerank Docs](https://docs.cohere.com/docs/rerank-2)
- [Qdrant API Docs](https://qdrant.tech/documentation/)

## Contributing

Found an issue or want to add more requests?

1. Create new `.bru` file in appropriate folder
2. Follow existing naming conventions
3. Add comprehensive `docs {}` block
4. Test thoroughly
5. Submit PR

## License

MIT (same as parent project)

---

**Built with** ❤️ **using [Bruno](https://www.usebruno.com/)**

**For**: AI Engineering Bootcamp Cohort 8 - Certification Challenge
