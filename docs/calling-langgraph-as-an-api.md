# Calling LangGraph 1.0 as an API (Threads & Runs)

*A concise, practical reference for AI engineers using LangGraph Server dev (Nov 2025).*

This guide assumes you're running:

```bash
uv run langgraph dev --allow-blocking
```

Which prints something like:

* API: `http://127.0.0.1:2024`
* Studio UI: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`
* API Docs: `http://127.0.0.1:2024/docs`

Your compiled graph is registered as an **assistant** with an ID (for you: `gdelt`), and you interact with it via **threads** and **runs**.

---

## 1. Mental Model

When deployed behind LangGraph Server / Agent Server, your graph behaves like a **stateful, resumable workflow**:

* **Assistant (graph)** – your compiled LangGraph graph, referred to by `assistant_id` (e.g. `"gdelt"` or `"agent"`).
* **Thread** – a long-lived conversation / state container (`thread_id`).
* **Run** – an invocation of an assistant on a thread (may pause on interrupts, then resume).
* **State** – JSON document that your graph reads/writes.
* **Interrupts** – points where the graph pauses and asks for human or external input.

In code (Python / JS SDK), the core interaction is:

```python
result = await client.runs.wait(
    thread_id,
    assistant_id,
    input={...},      # initial or updated state
    # optional: interrupt_before / interrupt_after
)
```

The HTTP API your API client (and Bruno) will hit is a thin wrapper around that.

---

## 2. Core HTTP Flow (Dev Server)

For a locally running dev server (`http://127.0.0.1:2024`), an end-to-end flow looks like this:

1. **Create a thread**

   ```http
   POST /threads
   ```

   ```json
   {}
   ```

   Response (shape simplified):

   ```json
   {
     "thread_id": "46301aad-faba-4ec0-b2ee-143ffbc8ff4e",
     "created_at": "...",
     "metadata": {}
   }
   ```

2. **Run the graph on that thread (wait until done / interrupted)**

   ```http
   POST /threads/{thread_id}/runs/wait
   ```

   Example body (adapt field names to your graph input schema):

   ```json
   {
     "assistant_id": "gdelt",
     "input": {
       "query": "how was llamaindex used in the paper?"
     }
   }
   ```

   The server will:

   * Embed your query via OpenAI
   * Query Qdrant (`gdelt_comparative_eval` collection)
   * Rerank with Cohere
   * Call the LLM to generate an answer

   Response (conceptual):

   ```json
   {
     "thread_id": "...",
     "run_id": "019aa2ce-3151-735d-b915-45e449fc5fcf",
     "values": {
       "answer": "…",
       "documents": [ ... ],
       "latency_ms": 11993
     },
     "__interrupt__": null
   }
   ```

   If your graph uses human-in-the-loop, you may instead see:

   ```json
   {
     "values": { ...partial state... },
     "__interrupt__": [
       {
         "value": { "text_to_revise": "original text" },
         "resumable": true,
         "ns": ["human_node:..."],
         "when": "during"
       }
     ]
   }
   ```

3. **Inspect final thread state**

   ```http
   GET /threads/{thread_id}
   ```

   This returns the latest committed state for that thread (useful to debug what your graph actually wrote).

---

## 3. Other Useful Endpoints

Exact details are visible at `http://127.0.0.1:2024/docs`, but the common ones are:

| Method | Path                                        | Purpose                         |
| ------ | ------------------------------------------- | ------------------------------- |
| GET    | `/assistants/{assistant_id}/schemas`        | Input/output schema & nodes     |
| GET    | `/assistants/{assistant_id}/graph`          | Graph structure                 |
| GET    | `/assistants/{assistant_id}/subgraphs`      | Subgraph info                   |
| POST   | `/threads`                                  | Create a new thread             |
| GET    | `/threads/{thread_id}`                      | Get thread state / metadata     |
| POST   | `/threads/{thread_id}/runs/wait`            | Start/resume run and wait       |
| GET    | `/threads/{thread_id}/runs/{run_id}`        | Get run info                    |
| GET    | `/threads/{thread_id}/runs/{run_id}/events` | (When enabled) stream events    |
| POST   | `/threads/{thread_id}/runs/{run_id}/events` | (When enabled) send HITL events |

For most application use-cases you can stay on the **`runs/wait`** path and let LangGraph manage interrupts for you.

---

## 4. Authentication

In **local dev**, auth is typically disabled:

* No `x-api-key` needed
* Base URL: `http://127.0.0.1:2024`

In **LangSmith Cloud / production Agent Server**, you will usually see:

* `x-api-key: <LANGSMITH_API_KEY>`
  or
* `Authorization: Bearer <token>`

Prefer:

* HTTPS
* Short-lived tokens for UI-exposed flows
* Scoping keys to a deployment / environment

---

## 5. curl Examples (Local Dev, Single Graph `gdelt`)

### 5.1 Create a Thread

```bash
curl -X POST http://127.0.0.1:2024/threads \
  -H "Content-Type: application/json" \
  -d '{}'
```

Save the `thread_id` from the response.

### 5.2 Run the Graph (Blocking)

```bash
THREAD_ID="46301aad-faba-4ec0-b2ee-143ffbc8ff4e"

curl -X POST \
  "http://127.0.0.1:2024/threads/${THREAD_ID}/runs/wait" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "gdelt",
    "input": {
      "query": "how was llamaindex used in the paper?"
    }
  }'
```

### 5.3 Resume After Interrupt (HITL)

If the previous call returned a `__interrupt__` payload:

```bash
curl -X POST \
  "http://127.0.0.1:2024/threads/${THREAD_ID}/runs/wait" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "gdelt",
    "command": {
      "resume": "Edited text from human."
    }
  }'
```

For static breakpoints you can also pass `input: null` to "just continue" to the next breakpoint.

---

## 6. Using Bruno to Call LangGraph

Bruno collections are just `.bru` files. Recommended environment variables:

* `{{base_url}}` – e.g. `http://127.0.0.1:2024`
* `{{assistant_id}}` – e.g. `gdelt`
* `{{thread_id}}` – captured from the first request
* `{{run_id}}` – optional, if you use run-level endpoints

### 6.1 Create Thread (Bruno Request)

```text
# @name create-thread
POST {{base_url}}/threads
Content-Type: application/json

{}
```

From the response, copy `thread.thread_id` into `{{thread_id}}`.

### 6.2 Run Graph with Question

```text
# @name run-gdelt-wait
POST {{base_url}}/threads/{{thread_id}}/runs/wait
Content-Type: application/json

{
  "assistant_id": "{{assistant_id}}",
  "input": {
    "query": "how was llamaindex used in the paper?"
  }
}
```

### 6.3 Inspect Thread State

```text
# @name get-thread
GET {{base_url}}/threads/{{thread_id}}
```

### 6.4 Resume After Interrupt (Optional)

```text
# @name resume-after-interrupt
POST {{base_url}}/threads/{{thread_id}}/runs/wait
Content-Type: application/json

{
  "assistant_id": "{{assistant_id}}",
  "command": {
    "resume": "Human-approved revision or answer."
  }
}
```

---

## 7. HITL Appendix (Interrupt-Based Pattern)

**Dynamic interrupts** (via `interrupt(...)` in your graph) behave like:

1. First `runs/wait` call:

   * Runs until an interrupt is hit.
   * Returns `values` (partial state) + `__interrupt__` with payload and metadata.

2. You display the interrupt payload to the user, or process it in a tool.

3. Second `runs/wait` call:

   * Same `thread_id` and `assistant_id`.
   * Provide a `command` (e.g. `Command(resume=...)` in SDK, or `{"resume": ...}` in JSON).
   * Graph continues from that point using the human/tool input, and eventually returns updated `values`.

This pattern is sufficient for:

* "Approve / reject" actions
* "Edit this text and continue"
* Tool-result review workflows
* Simple multi-step human interactions

If you ever need more granular control (events, node-level streaming), you can drop down to the lower-level `/runs/{run_id}/events` endpoints, but for now `runs/wait` is the cleanest abstraction.

---

## Related Resources

- **Bruno Collection**: `bruno/langgraph-api/` (local development endpoints)
- **LangSmith Platform Collection**: `bruno/LangGraphTags/` (enterprise features)
- **Version Validation**: `bruno/langgraph-api/VERSION_VALIDATION.md`
- **API Docs**: http://127.0.0.1:2024/docs (when server running)
- **LangGraph Studio**: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
