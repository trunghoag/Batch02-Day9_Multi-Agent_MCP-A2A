# DrugLaw Day 09 Hands-on Submission

Day 09 submission nay nam tron trong repo Day 09. Day 08 artifact da duoc
nhung vao folder `day08_artifact/` de khi nop repo nay khong can nop them repo
Day 08 rieng.

## Folder Layout

```text
Batch02-Day9_Multi-Agent_MCP-A2A/
├── day08_artifact/
│   ├── data/                  # embedded Day 08 data/index/docs
│   └── src/                   # Day 08 Task 4-10 RAG pipeline modules
├── druglaw_day09/
│   ├── multi_agent.py         # public import wrapper
│   └── __main__.py            # CLI demo: python -m druglaw_day09
├── tests/
│   └── test_druglaw_day09.py  # rubric-focused tests
└── DRUGLAW_DAY09_HANDS_ON.md
```

## What Was Built

Muc tieu lab: bien mot agent RAG don thanh he multi-agent nho co route ro,
capability ro, va trace ro.

System pieces:

| Piece | Role | Writes |
|---|---|---|
| `supervisor` | route decision, plan, final status review | `plan`, `route_decision`, `status` |
| `retrieval_worker` | calls Day 08 hybrid retrieval | `retrieval_result` |
| `mcp_tool_worker` | calls external capability via MCP-style adapter | `tool_result` |
| `synthesis_worker` | generates final answer with citation | `synthesis_draft`, `final_answer`, `sources` |

## Shared State

State schema is implemented in `day08_artifact/src/day09_multi_agent.py` and
exported via `druglaw_day09.multi_agent`.

Important fields:

```python
task: str
user_context: dict
plan: list[str]
route_decision: dict
retrieval_result: list[dict]
tool_result: dict
synthesis_draft: str
final_answer: str
sources: list[dict]
status: str
trace: list[TraceEvent]
error: list[str]
```

Ownership is documented in `DAY09_FIELD_OWNERS`. Workers only write their own
fields. `trace` and `error` are append-only.

## Message Contract

Supervisor-to-worker message:

```python
class WorkerMessage(TypedDict):
    trace_id: str
    request_id: str
    task: str
    payload: dict[str, Any]
    options: dict[str, Any]
```

Worker result:

```python
class WorkerResult(TypedDict):
    worker: str
    status: str
    output: dict[str, Any]
    error: str | None
```

## MCP Capability

Chosen lab option: mock MCP interface with a real PageIndex-backed capability.

`MCPStyleClient` demonstrates:

1. `discover("vectorless_search")`
2. `call_tool("pageindex_vectorless_search", {"query": ..., "top_k": ...})`

The tool calls Day 08 `pageindex_search()`. For this worker to succeed, set a
real `PAGEINDEX_API_KEY`, install the `pageindex` package, and upload documents
once so `day08_artifact/data/index/pageindex_documents.json` contains real
document IDs. If PageIndex is not configured, the worker fails gracefully and the
trace records the setup error.

Trace events:

- `mcp.discover`
- `mcp.call`

## Trace Evidence

Each trace event includes:

- `trace_id`
- `span_id`
- `parent_span_id`
- `agent`
- `event`
- `capability`
- `status`
- `latency_ms`
- `input`
- `output`
- `error`

This is enough to answer: which agent did what, which capability was used, and
where the failure happened.

## Demo Commands

Run rubric tests:

```powershell
python -m unittest tests.test_druglaw_day09 -v
```

Run normal demo:

```powershell
python -m druglaw_day09
```

This route calls the MCP worker and OpenRouter synthesis. For status `completed`:

- set `OPENROUTER_API_KEY` for final answer synthesis
- set `PAGEINDEX_API_KEY`
- install the `pageindex` package
- upload PageIndex documents once so document IDs exist

Without PageIndex configuration, the run will be `degraded` and the trace will
show the MCP setup error. Without `OPENROUTER_API_KEY`, synthesis will fail and
the trace will show the missing-key error.

Run demo without MCP worker:

```powershell
python -m druglaw_day09 --no-mcp
```

Run graceful failure demo:

```powershell
python -m druglaw_day09 --fail-mcp
```

Expected graceful failure result:

- `status = "degraded"`
- answer still exists
- `errors` contains MCP failure
- trace contains `mcp.call` with `status = "error"`

## Rubric Mapping

| Rubric item | Evidence |
|---|---|
| 1 supervisor + 2-3 workers | `supervisor`, `retrieval_worker`, `mcp_tool_worker`, `synthesis_worker` |
| MCP connected capability | `MCPStyleClient.discover()` + `call_tool()` |
| Shared state | `Day09State`, `DAY09_FIELD_OWNERS` |
| Trace quality | `TraceEvent`, `summarize_trace()` |
| Routing logic | `supervisor()` writes `route_decision.conditional_edges` |
| Worker fail gracefully | `python -m druglaw_day09 --fail-mcp` |

## Short Presentation Script

Day 08 used one RAG flow for retrieve, fallback, generation, and source
formatting. Day 09 splits ownership:

- Retrieval worker owns evidence retrieval.
- MCP tool worker owns external capability.
- Synthesis worker owns final answer.
- Supervisor owns route and final state review.

The trade-off is more moving parts, but trace makes debugging easier: we can see
exactly which worker ran, what it used, whether it failed, and whether the final
answer is still reliable enough to show.
