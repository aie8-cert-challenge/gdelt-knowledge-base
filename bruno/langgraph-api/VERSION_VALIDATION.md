# Version Validation Report

**Date**: 2025-11-20
**Server Version**: langgraph-api 0.5.20
**LangGraph Core**: 1.0.1
**Host**: Self-hosted (local development)

## Server Information

```json
{
  "version": "0.5.20",
  "langgraph_py_version": "1.0.1",
  "flags": {
    "assistants": true,
    "crons": false,
    "langsmith": true,
    "langsmith_tracing_replicas": true
  },
  "host": {
    "kind": "self-hosted",
    "project_id": null,
    "host_revision_id": null,
    "revision_id": null,
    "tenant_id": null
  }
}
```

## Endpoint Validation Results

### ✅ Working Endpoints (13/16)

| Endpoint | Status | Notes |
|----------|--------|-------|
| `/ok` | ✅ 200 | Health check (correct endpoint) |
| `/info` | ✅ 200 | Server version information |
| `/assistants` | ✅ Confirmed | List assistants |
| `/assistants/{id}` | ✅ Confirmed | Get assistant details |
| `/assistants/{id}/schemas` | ✅ Confirmed | Input/output schemas |
| `/assistants/{id}/graph` | ✅ Confirmed | Graph structure |
| `/assistants/{id}/subgraphs` | ✅ Confirmed | Subgraph information |
| `/threads` | ✅ Confirmed | Create thread |
| `/threads/{id}` | ✅ Confirmed | Get thread |
| `/threads/{id}/runs/wait` | ✅ Confirmed | Run graph (blocking) |
| `/threads/{id}/runs/stream` | ✅ Confirmed | Run graph (streaming) |
| `/threads/{id}/runs/{run_id}` | ✅ Confirmed | Get run status |
| `/threads/{id}/state` | ✅ Confirmed | Get thread state |
| `/threads/{id}/history` | ✅ Confirmed | Get checkpoint history |

### ❌ Non-Existent Endpoints (3/16)

| Endpoint | Status | Reason |
|----------|--------|--------|
| `/health` | ❌ 404 | Wrong endpoint - use `/ok` instead |
| `/assistants/{id}/config` | ❌ Not in API | Not available in 0.5.20 |
| `/threads/{id}/events` | ❌ Not in API | HITL events not exposed at this path |

## Feature Flags

Based on server response:

- ✅ **Assistants**: Enabled
- ❌ **Crons**: Disabled (not available in self-hosted)
- ✅ **LangSmith**: Enabled (tracing/observability)
- ✅ **LangSmith Tracing Replicas**: Enabled

## Corrections Needed

### 1. Health Check Endpoint
**File**: `collections/00-health/health-check.bru`

```diff
- url: {{base_url}}/health
+ url: {{base_url}}/ok
```

### 2. Get Config Endpoint
**File**: `collections/07-debug/get-config.bru`

**Action**: Mark as not available or remove

**Reason**: The `/assistants/{id}/config` endpoint doesn't exist in langgraph-api 0.5.20. Configuration is retrieved through `/assistants/{id}/schemas` instead.

### 3. HITL Events Endpoints
**Files**:
- `collections/06-hitl/get-events.bru`
- `collections/06-hitl/post-event.bru`

**Action**: Mark as educational/aspirational

**Reason**: The `/threads/{id}/events` endpoint pattern doesn't exist in 0.5.20. HITL (Human-in-the-Loop) is handled through interrupts in the run execution flow, not through a separate events API.

## Alternative HITL Pattern (0.5.20)

Instead of separate `/events` endpoints, HITL in 0.5.20 works through the run API:

1. **Run with interrupt**:
   ```
   POST /threads/{id}/runs/wait
   → Returns with interrupt information in response
   ```

2. **Resume after human input**:
   ```
   POST /threads/{id}/runs/wait (same endpoint)
   → Provide human input in request body
   ```

The interrupt handling is integrated into the run lifecycle, not exposed as separate event endpoints.

## Compatibility Notes

### For Self-Hosted Deployments

**Available**:
- Core threading and runs
- State management
- Checkpoint history
- Assistant introspection
- LangSmith tracing (if configured)

**Not Available**:
- Cron scheduling (requires LangSmith Platform)
- Store API (persistent key-value, platform feature)
- Agent-to-Agent communication (A2A, platform feature)
- MCP integrations (Model Context Protocol, platform feature)

### For LangSmith Platform

If deploying to LangSmith Platform instead of self-hosted, additional endpoints become available:

- `/store/items` - Persistent key-value storage
- `/crons/{id}` - Scheduled execution
- `/assistants/{id}/a2a` - Agent-to-agent communication
- `/mcp` - Model Context Protocol

See `bruno/LangGraphTags/` collection for LangSmith Platform API reference.

## Recommendations

1. **Fix health check**: Update to use `/ok` endpoint
2. **Remove config endpoint**: Not available in this version
3. **Document HITL pattern**: Explain interrupt-based flow instead of events API
4. **Keep collection focused**: Target self-hosted features only
5. **Cross-reference**: Point to LangGraphTags for platform features

## Version History

| Date | Version | Notes |
|------|---------|-------|
| 2025-11-20 | 0.5.20 | Initial validation, identified 3 endpoint issues |

## Related Documentation

- **Server Info**: Available at `GET /info`
- **OpenAPI Spec**: Available at `GET /openapi.json`
- **LangSmith Platform API**: See `bruno/LangGraphTags/` collection
- **Self-Hosted Docs**: This collection (`bruno/langgraph-api/`)
