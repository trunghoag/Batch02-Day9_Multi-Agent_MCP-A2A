"""Day 09 multi-agent wrapper for the Day 08 DrugLaw RAG artifact.

This module keeps the Day 08 retrieval/generation pipeline intact, then adds a
small supervisor-worker layer with shared state, message contracts, trace events,
and one MCP-style external capability adapter.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Literal, NotRequired, TypedDict
from uuid import uuid4

from .task8_pageindex_vectorless import pageindex_search
from .task9_retrieval_pipeline import retrieve
from .task10_generation import format_context


AgentName = Literal[
    "supervisor",
    "retrieval_worker",
    "mcp_tool_worker",
    "synthesis_worker",
]


class TraceEvent(TypedDict):
    trace_id: str
    span_id: str
    parent_span_id: str | None
    agent: str
    event: str
    capability: str
    status: str
    started_at: str
    ended_at: str
    latency_ms: int
    input: dict[str, Any]
    output: dict[str, Any]
    error: str | None


class Day09State(TypedDict):
    task: str
    user_context: dict[str, Any]
    plan: list[str]
    route_decision: dict[str, Any]
    retrieval_result: list[dict[str, Any]]
    tool_result: dict[str, Any]
    synthesis_draft: str
    final_answer: str
    sources: list[dict[str, Any]]
    status: str
    trace: list[TraceEvent]
    error: list[str]


class WorkerMessage(TypedDict):
    trace_id: str
    request_id: str
    task: str
    payload: dict[str, Any]
    options: dict[str, Any]


class WorkerResult(TypedDict):
    worker: str
    status: str
    output: dict[str, Any]
    error: NotRequired[str | None]


DAY09_FIELD_OWNERS: dict[str, str] = {
    "task": "supervisor",
    "user_context": "supervisor",
    "plan": "supervisor",
    "route_decision": "supervisor",
    "retrieval_result": "retrieval_worker",
    "tool_result": "mcp_tool_worker",
    "synthesis_draft": "synthesis_worker",
    "final_answer": "synthesis_worker",
    "sources": "synthesis_worker",
    "status": "supervisor",
    "trace": "all_agents_append_only",
    "error": "all_agents_append_only",
}


WORKER_CONTRACTS: dict[str, dict[str, Any]] = {
    "supervisor": {
        "input": ["task", "user_context"],
        "writes": ["plan", "route_decision", "status"],
        "does_not_write": ["retrieval_result", "tool_result", "final_answer"],
    },
    "retrieval_worker": {
        "input": ["task", "top_k", "use_reranking"],
        "writes": ["retrieval_result"],
        "capability": "Day 08 hybrid retrieve: dense + BM25 + RRF/rerank",
    },
    "mcp_tool_worker": {
        "input": ["task", "top_k", "force_tool_failure"],
        "writes": ["tool_result"],
        "capability": "MCP-style discovery + PageIndex vectorless search adapter",
    },
    "synthesis_worker": {
        "input": ["task", "retrieval_result", "tool_result"],
        "writes": ["synthesis_draft", "final_answer", "sources"],
        "capability": "Day 08 citation generation",
    },
}


MCP_TOOL_CARD: dict[str, Any] = {
    "server": "day09-mock-mcp",
    "tool_name": "pageindex_vectorless_search",
    "description": (
        "External retrieval capability exposed through an MCP-style adapter. "
        "It requires PageIndex configuration and reports errors through trace."
    ),
    "input_schema": {"query": "str", "top_k": "int"},
    "output_schema": {
        "status": "ok|failed",
        "chunks": "list[dict]",
        "source": "pageindex_adapter",
    },
}


def new_state(
    question: str,
    *,
    top_k: int = 5,
    use_reranking: bool = True,
    use_external_tool: bool = True,
    force_tool_failure: bool = False,
) -> Day09State:
    """Create the shared state object for one Day 09 run."""
    return {
        "task": question,
        "user_context": {
            "trace_id": str(uuid4()),
            "request_id": str(uuid4()),
            "top_k": top_k,
            "use_reranking": use_reranking,
            "use_external_tool": use_external_tool,
            "force_tool_failure": force_tool_failure,
        },
        "plan": [],
        "route_decision": {},
        "retrieval_result": [],
        "tool_result": {},
        "synthesis_draft": "",
        "final_answer": "",
        "sources": [],
        "status": "created",
        "trace": [],
        "error": [],
    }


def trace_id(state: Day09State) -> str:
    return str(state["user_context"]["trace_id"])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def add_trace(
    state: Day09State,
    *,
    agent: AgentName | str,
    event: str,
    capability: str,
    status: str,
    started_at: str,
    input_data: dict[str, Any],
    output_data: dict[str, Any] | None = None,
    error: str | None = None,
    parent_span_id: str | None = None,
) -> str:
    """Append one trace event. Trace is append-only by design."""
    ended_at = utc_now()
    span_id = str(uuid4())
    latency_ms = int(
        (
            datetime.fromisoformat(ended_at)
            - datetime.fromisoformat(started_at)
        ).total_seconds()
        * 1000
    )
    state["trace"].append(
        {
            "trace_id": trace_id(state),
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "agent": str(agent),
            "event": event,
            "capability": capability,
            "status": status,
            "started_at": started_at,
            "ended_at": ended_at,
            "latency_ms": latency_ms,
            "input": input_data,
            "output": output_data or {},
            "error": error,
        }
    )
    return span_id


def build_worker_message(
    state: Day09State,
    task: str,
    payload: dict[str, Any],
    options: dict[str, Any] | None = None,
) -> WorkerMessage:
    """Minimal message contract passed from supervisor to workers."""
    return {
        "trace_id": trace_id(state),
        "request_id": str(state["user_context"]["request_id"]),
        "task": task,
        "payload": payload,
        "options": options or {},
    }


def supervisor(state: Day09State) -> WorkerResult:
    """Decide the conditional route and write only supervisor-owned fields."""
    started_at = utc_now()
    use_external_tool = bool(state["user_context"].get("use_external_tool", True))
    force_tool_failure = bool(state["user_context"].get("force_tool_failure", False))

    conditional_edges = ["retrieval_worker"]
    if use_external_tool:
        conditional_edges.append("mcp_tool_worker")
    conditional_edges.append("synthesis_worker")

    reason = (
        "External capability enabled for Day 09 evidence audit."
        if use_external_tool
        else "External capability disabled; route uses only Day 08 RAG."
    )
    if force_tool_failure:
        reason += " The MCP worker will simulate failure to demonstrate graceful handling."

    state["plan"] = conditional_edges
    state["route_decision"] = {
        "routing_style": "LangGraph-style conditional edges",
        "conditional_edges": conditional_edges,
        "use_external_tool": use_external_tool,
        "force_tool_failure": force_tool_failure,
        "reason": reason,
    }
    state["status"] = "running"

    add_trace(
        state,
        agent="supervisor",
        event="route_decision",
        capability="conditional_routing",
        status="ok",
        started_at=started_at,
        input_data={
            "task_preview": state["task"][:120],
            "use_external_tool": use_external_tool,
            "force_tool_failure": force_tool_failure,
        },
        output_data=state["route_decision"],
    )
    return {
        "worker": "supervisor",
        "status": "ok",
        "output": {"plan": state["plan"], "route_decision": state["route_decision"]},
    }


def retrieval_worker(state: Day09State) -> WorkerResult:
    """Run the Day 08 hybrid retrieval pipeline."""
    started_at = utc_now()
    top_k = int(state["user_context"].get("top_k", 5))
    use_reranking = bool(state["user_context"].get("use_reranking", True))
    message = build_worker_message(
        state,
        "retrieve_relevant_context",
        {"query": state["task"]},
        {"top_k": top_k, "use_reranking": use_reranking},
    )
    try:
        chunks = retrieve(state["task"], top_k=top_k, use_reranking=use_reranking)
        state["retrieval_result"] = chunks
        if not chunks:
            state["error"].append("retrieval_worker returned no chunks")
        add_trace(
            state,
            agent="retrieval_worker",
            event="worker.complete",
            capability="hybrid_retrieval",
            status="ok" if chunks else "empty",
            started_at=started_at,
            input_data=message,
            output_data={
                "chunk_count": len(chunks),
                "best_score": round(float(chunks[0].get("score", 0)), 4) if chunks else 0,
                "top_source": chunks[0].get("metadata", {}).get("source", "") if chunks else "",
            },
        )
        return {
            "worker": "retrieval_worker",
            "status": "ok",
            "output": {"retrieval_result": chunks},
        }
    except Exception as exc:
        message_text = f"retrieval_worker failed: {exc}"
        state["error"].append(message_text)
        add_trace(
            state,
            agent="retrieval_worker",
            event="worker.failed",
            capability="hybrid_retrieval",
            status="error",
            started_at=started_at,
            input_data=message,
            error=str(exc),
        )
        return {
            "worker": "retrieval_worker",
            "status": "failed",
            "output": {"retrieval_result": []},
            "error": str(exc),
        }


class MCPStyleClient:
    """Tiny MCP-style adapter with discovery and tool call phases."""

    def __init__(self, *, force_failure: bool = False):
        self.force_failure = force_failure

    def discover(self, capability: str) -> dict[str, Any]:
        if capability != "vectorless_search":
            raise ValueError(f"No MCP capability named {capability}")
        return MCP_TOOL_CARD.copy()

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.force_failure:
            raise RuntimeError("Simulated MCP transport failure")
        if tool_name != MCP_TOOL_CARD["tool_name"]:
            raise ValueError(f"Unknown MCP tool {tool_name}")
        query = str(arguments.get("query", ""))
        top_k = max(1, min(int(arguments.get("top_k", 3)), 5))
        chunks = pageindex_search(query, top_k=top_k)
        return {
            "status": "ok",
            "tool_name": tool_name,
            "source": "pageindex_adapter",
            "chunks": chunks,
            "count": len(chunks),
        }


def mcp_tool_worker(state: Day09State) -> WorkerResult:
    """Use the MCP-style external capability and write tool_result."""
    force_failure = bool(state["user_context"].get("force_tool_failure", False))
    client = MCPStyleClient(force_failure=force_failure)
    top_k = int(state["user_context"].get("top_k", 5))

    discover_started = utc_now()
    message = build_worker_message(
        state,
        "call_external_vectorless_capability",
        {"query": state["task"]},
        {"top_k": min(top_k, 3), "force_tool_failure": force_failure},
    )

    try:
        tool_card = client.discover("vectorless_search")
        discover_span = add_trace(
            state,
            agent="mcp_tool_worker",
            event="mcp.discover",
            capability="mcp_tool_discovery",
            status="ok",
            started_at=discover_started,
            input_data={"capability": "vectorless_search"},
            output_data={"tool_name": tool_card["tool_name"], "server": tool_card["server"]},
        )

        call_started = utc_now()
        result = client.call_tool(
            tool_card["tool_name"],
            {"query": state["task"], "top_k": min(top_k, 3)},
        )
        state["tool_result"] = result
        add_trace(
            state,
            agent="mcp_tool_worker",
            event="mcp.call",
            capability=tool_card["tool_name"],
            status="ok",
            started_at=call_started,
            input_data=message,
            output_data={
                "count": result["count"],
                "source": result["source"],
                "first_source": (
                    result["chunks"][0].get("metadata", {}).get("source", "")
                    if result["chunks"]
                    else ""
                ),
            },
            parent_span_id=discover_span,
        )
        return {
            "worker": "mcp_tool_worker",
            "status": "ok",
            "output": {"tool_result": result},
        }
    except Exception as exc:
        result = {
            "status": "failed",
            "tool_name": MCP_TOOL_CARD["tool_name"],
            "source": "pageindex_adapter",
            "chunks": [],
            "count": 0,
            "error": str(exc),
        }
        state["tool_result"] = result
        state["error"].append(f"mcp_tool_worker failed gracefully: {exc}")
        add_trace(
            state,
            agent="mcp_tool_worker",
            event="mcp.call",
            capability=MCP_TOOL_CARD["tool_name"],
            status="error",
            started_at=discover_started,
            input_data=message,
            output_data={"count": 0, "source": "pageindex_adapter"},
            error=str(exc),
        )
        return {
            "worker": "mcp_tool_worker",
            "status": "failed",
            "output": {"tool_result": result},
            "error": str(exc),
        }


def merge_context_chunks(state: Day09State) -> list[dict[str, Any]]:
    """Merge retrieval and MCP chunks while preserving deterministic order."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_chunk(chunk: dict[str, Any]) -> None:
        metadata = chunk.get("metadata", {})
        key = (
            f"{metadata.get('source', '')}:"
            f"{metadata.get('chunk_index', metadata.get('doc_id', ''))}:"
            f"{chunk.get('content', '')[:120]}"
        )
        if key in seen:
            return
        seen.add(key)
        merged.append(chunk)

    for chunk in state.get("retrieval_result", []):
        add_chunk(chunk)
    for chunk in state.get("tool_result", {}).get("chunks", []):
        add_chunk(chunk)
    return merged


def synthesize_with_citations(query: str, context_chunks: list[dict[str, Any]]) -> str:
    """Generate a citation-focused answer with the repo's OpenRouter LLM."""
    if not context_chunks:
        raise RuntimeError("No context chunks available for synthesis.")

    from langchain_core.messages import HumanMessage, SystemMessage

    from common.llm import get_llm

    context = format_context(context_chunks)
    messages = [
        SystemMessage(
            content=(
                "Answer in Vietnamese using only the provided context. Cite factual claims "
                "with the source labels shown in the context. If evidence is insufficient, "
                "state exactly what is missing instead of guessing."
            )
        ),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ]
    response = get_llm().invoke(messages)
    return response.content.strip()


def synthesis_worker(state: Day09State) -> WorkerResult:
    """Generate the final answer from retrieved and external context."""
    started_at = utc_now()
    context_chunks = merge_context_chunks(state)
    message = build_worker_message(
        state,
        "synthesize_answer_with_citations",
        {
            "query": state["task"],
            "retrieval_chunks": len(state["retrieval_result"]),
            "mcp_chunks": len(state.get("tool_result", {}).get("chunks", [])),
        },
        {"top_k": int(state["user_context"].get("top_k", 5))},
    )
    try:
        state["synthesis_draft"] = (
            f"Use {len(state['retrieval_result'])} RAG chunks and "
            f"{len(state.get('tool_result', {}).get('chunks', []))} MCP chunks."
        )
        answer = synthesize_with_citations(state["task"], context_chunks)
        state["final_answer"] = answer
        state["sources"] = context_chunks
        add_trace(
            state,
            agent="synthesis_worker",
            event="worker.complete",
            capability="citation_generation",
            status="ok" if state["final_answer"] else "empty",
            started_at=started_at,
            input_data=message,
            output_data={
                "answer_chars": len(state["final_answer"]),
                "source_count": len(state["sources"]),
                "retrieval_source": context_chunks[0].get("source", "none") if context_chunks else "none",
            },
        )
        return {
            "worker": "synthesis_worker",
            "status": "ok",
            "output": {
                "final_answer": state["final_answer"],
                "sources": state["sources"],
            },
        }
    except Exception as exc:
        message_text = f"synthesis_worker failed: {exc}"
        state["error"].append(message_text)
        add_trace(
            state,
            agent="synthesis_worker",
            event="worker.failed",
            capability="citation_generation",
            status="error",
            started_at=started_at,
            input_data=message,
            error=str(exc),
        )
        return {
            "worker": "synthesis_worker",
            "status": "failed",
            "output": {"final_answer": ""},
            "error": str(exc),
        }


def run_day09_multi_agent(
    question: str,
    *,
    top_k: int = 5,
    use_reranking: bool = True,
    use_external_tool: bool = True,
    force_tool_failure: bool = False,
) -> Day09State:
    """Run the complete Day 09 supervisor + workers flow."""
    state = new_state(
        question,
        top_k=top_k,
        use_reranking=use_reranking,
        use_external_tool=use_external_tool,
        force_tool_failure=force_tool_failure,
    )
    supervisor(state)

    if "retrieval_worker" in state["plan"]:
        retrieval_worker(state)
    if "mcp_tool_worker" in state["plan"]:
        mcp_tool_worker(state)
    if "synthesis_worker" in state["plan"]:
        synthesis_worker(state)

    if state["final_answer"]:
        state["status"] = "degraded" if state["error"] else "completed"
    else:
        state["status"] = "failed"

    final_started = utc_now()
    add_trace(
        state,
        agent="supervisor",
        event="run.finalize",
        capability="state_review",
        status=state["status"],
        started_at=final_started,
        input_data={"plan": state["plan"], "errors": len(state["error"])},
        output_data={
            "status": state["status"],
            "answer_chars": len(state["final_answer"]),
            "trace_events": len(state["trace"]) + 1,
        },
    )
    return state


def summarize_trace(trace: list[TraceEvent]) -> list[dict[str, Any]]:
    """Return a compact trace table for console/API demos."""
    return [
        {
            "agent": item["agent"],
            "event": item["event"],
            "status": item["status"],
            "capability": item["capability"],
            "latency_ms": item["latency_ms"],
            "error": item["error"],
        }
        for item in trace
    ]


def blueprint() -> dict[str, Any]:
    """Return the Day 09 design evidence in one inspectable object."""
    return {
        "system_pieces": {
            "supervisor": "conditional routing + final state review",
            "workers": ["retrieval_worker", "mcp_tool_worker", "synthesis_worker"],
            "external_capability": MCP_TOOL_CARD,
        },
        "shared_state_fields": DAY09_FIELD_OWNERS,
        "worker_contracts": WORKER_CONTRACTS,
        "trace_required_fields": [
            "trace_id",
            "span_id",
            "agent",
            "event",
            "capability",
            "status",
            "latency_ms",
            "input",
            "output",
            "error",
        ],
    }


if __name__ == "__main__":
    demo_question = (
        "Hinh phat cho toi tang tru trai phep chat ma tuy theo phap luat Viet Nam?"
    )
    started = time.time()
    demo_state = run_day09_multi_agent(demo_question, top_k=3)
    print(json.dumps(
        {
            "status": demo_state["status"],
            "plan": demo_state["plan"],
            "route_decision": demo_state["route_decision"],
            "answer_preview": demo_state["final_answer"][:500],
            "errors": demo_state["error"],
            "trace": summarize_trace(demo_state["trace"]),
            "elapsed_ms": int((time.time() - started) * 1000),
        },
        ensure_ascii=True,
        indent=2,
    ))
