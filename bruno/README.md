# Bruno API Collections

This directory contains two Bruno API collections for testing different LangGraph deployments.

## Collections Overview

### 1. `langgraph-api/` - **Self-Hosted Development** ⭐ PRIMARY

**Target**: Local LangGraph Server (self-hosted)
**Version**: langgraph-api 0.5.20, LangGraph 1.0.1
**Endpoints**: 13 working, 3 educational/not available
**Purpose**: Testing the GDELT Knowledge Base RAG system locally

**When to Use**:
- ✅ Local development and testing
- ✅ Self-hosted deployments
- ✅ Understanding core LangGraph Server API
- ✅ Learning RAG workflows and performance profiling

**Features**:
- Pre-configured for GDELT project
- Detailed inline documentation
- 5 documented workflow sequences
- Performance and cost breakdowns
- Auto-capture of thread_id and run_id
- 477-line comprehensive README

**Start Here**: [langgraph-api/README.md](langgraph-api/README.md)

---

### 2. `LangGraphTags/` - **LangSmith Platform Reference**

**Target**: LangSmith Platform API (cloud, enterprise)
**Version**: Platform API (multi-version support)
**Endpoints**: 68 endpoints across all platform features
**Purpose**: Reference collection for LangSmith Platform capabilities

**When to Use**:
- ✅ Testing LangSmith Platform deployments
- ✅ Enterprise features (Store, Crons, A2A, MCP)
- ✅ Multi-tenant cloud deployments
- ✅ Learning platform-specific APIs

**Features**:
- Comprehensive platform API coverage
- Store API (persistent key-value)
- Cron scheduling (scheduled execution)
- A2A (agent-to-agent communication)
- MCP (Model Context Protocol)
- 68 endpoints organized by domain

**Note**: This appears to be a reference/template collection from LangChain AI team, not specific to GDELT project.

---

## Quick Comparison

| Feature | `langgraph-api/` | `LangGraphTags/` |
|---------|------------------|------------------|
| **Target** | Self-hosted local dev | LangSmith Platform cloud |
| **API Version** | 0.5.20 | Platform API (multi-version) |
| **Endpoints** | 16 (13 working) | 68 (all platform features) |
| **Health Check** | `/ok` | `/ok?check_db=` |
| **Thread Creation** | Simple (metadata only) | Complex (if_exists, supersteps, ttl) |
| **Store API** | ❌ Not available | ✅ Available |
| **Crons** | ❌ Not available | ✅ Available (Plus tier) |
| **A2A** | ❌ Not available | ✅ Available |
| **MCP** | ❌ Not available | ✅ Available |
| **Documentation** | 477-line README + inline docs | Basic collection structure |
| **GDELT-Specific** | ✅ Yes (pre-configured) | ❌ No (generic template) |
| **Workflow Examples** | ✅ 5 sequences documented | ❌ None |
| **Performance Data** | ✅ Cost/latency per query | ❌ None |

---

## Which Collection Should I Use?

### Use `langgraph-api/` if you're:
- Developing locally with `uv run langgraph dev`
- Testing the GDELT Knowledge Base project
- Learning core LangGraph Server concepts
- Need detailed workflow documentation
- Want performance profiling and cost analysis

### Use `LangGraphTags/` if you're:
- Deploying to LangSmith Platform
- Need enterprise features (Store, Crons, A2A)
- Working with multi-tenant cloud deployments
- Exploring full platform capabilities
- Need comprehensive platform API reference

### Use Both if you're:
- Developing locally, deploying to platform
- Learning differences between self-hosted and cloud
- Migrating from local to platform deployment
- Need complete API coverage

---

## Getting Started

### For Local Development (Recommended)

1. **Start the server**:
   ```bash
   cd /path/to/gdelt-knowledge-base
   source .venv/bin/activate
   make qdrant-up
   uv run langgraph dev --allow-blocking
   ```

2. **Open Bruno**:
   - Import collection: `bruno/langgraph-api/`
   - Select environment: "local-dev"
   - Run: `00-health/health-check.bru`

3. **Read the docs**:
   - Collection README: [langgraph-api/README.md](langgraph-api/README.md)
   - API Guide: [../docs/calling-langgraph-as-an-api.md](../docs/calling-langgraph-as-an-api.md)
   - Version Validation: [langgraph-api/VERSION_VALIDATION.md](langgraph-api/VERSION_VALIDATION.md)

### For LangSmith Platform

1. **Configure deployment**:
   - Get your deployment URL from LangSmith
   - Get your API key
   - Update environment variables in collection

2. **Open Bruno**:
   - Import collection: `bruno/LangGraphTags/LangSmith Deployment/`
   - Select appropriate environment
   - Configure authentication headers

---

## Environment Variables

### `langgraph-api/` (Local Dev)

```javascript
{
  base_url: "http://127.0.0.1:2024",
  assistant_id: "e45a51c7-e110-55eb-b8c6-c2b51f0c2c8f",
  graph_id: "gdelt",
  thread_id: "",  // Auto-captured
  run_id: ""      // Auto-captured
}
```

### `LangGraphTags/` (Platform)

```javascript
{
  baseUrl: "https://your-deployment.langsmith.com",
  assistant_id: "your-assistant-id",
  graph_id: "gdelt",
  thread_id: "",
  run_id: ""
}
```

Plus authentication headers in environment settings.

---

## API Endpoints Availability

### Core Endpoints (Both Collections)

✅ Available in both self-hosted and platform:
- `/ok` - Health check
- `/info` - Server information
- `/assistants` - List/get assistants
- `/assistants/{id}/schemas` - Input/output schemas
- `/assistants/{id}/graph` - Graph structure
- `/threads` - Create/get threads
- `/threads/{id}/runs/wait` - Execute graph (blocking)
- `/threads/{id}/runs/stream` - Execute graph (streaming)
- `/threads/{id}/state` - Get thread state
- `/threads/{id}/history` - Get checkpoint history

### Platform-Only Endpoints

❌ NOT available in self-hosted (use LangGraphTags collection):
- `/store/items` - Persistent key-value storage
- `/store/namespaces` - Store namespaces
- `/crons/{id}` - Scheduled execution (Plus tier)
- `/assistants/{id}/a2a` - Agent-to-agent communication
- `/mcp` - Model Context Protocol integrations

### Self-Hosted Notes

⚠️ Some endpoints documented in `langgraph-api/` are marked as "not available in 0.5.20":
- `/assistants/{id}/config` - Use `/schemas` instead
- `/threads/{id}/events` - Use interrupt pattern in `/runs/wait`

See [langgraph-api/VERSION_VALIDATION.md](langgraph-api/VERSION_VALIDATION.md) for details.

---

## Documentation Resources

- **Primary API Guide**: [docs/calling-langgraph-as-an-api.md](../docs/calling-langgraph-as-an-api.md)
- **Local Collection README**: [langgraph-api/README.md](langgraph-api/README.md)
- **Version Validation**: [langgraph-api/VERSION_VALIDATION.md](langgraph-api/VERSION_VALIDATION.md)
- **Official API Docs**: http://127.0.0.1:2024/docs (when server running)
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **LangSmith**: https://smith.langchain.com/

---

## Contributing

Found an issue or want to add requests?

1. For self-hosted endpoints: Edit `langgraph-api/`
2. For platform endpoints: Edit `LangGraphTags/`
3. Follow existing naming conventions
4. Add comprehensive `docs {}` blocks
5. Test thoroughly before committing

---

## License

MIT (same as parent project)

---

**Built with** ❤️ **using [Bruno](https://www.usebruno.com/)**
