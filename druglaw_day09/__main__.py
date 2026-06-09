"""CLI demo for the Day 09 DrugLaw multi-agent wrapper."""

from __future__ import annotations

import argparse
import json
import time

from .multi_agent import run_day09_multi_agent, summarize_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Day 09 DrugLaw multi-agent demo.")
    parser.add_argument(
        "question",
        nargs="?",
        default="Hinh phat cho toi tang tru trai phep chat ma tuy theo phap luat Viet Nam?",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-mcp", action="store_true", help="Skip the MCP-style worker.")
    parser.add_argument(
        "--fail-mcp",
        action="store_true",
        help="Simulate MCP transport failure to demonstrate graceful degradation.",
    )
    args = parser.parse_args()

    started = time.time()
    state = run_day09_multi_agent(
        args.question,
        top_k=args.top_k,
        use_external_tool=not args.no_mcp,
        force_tool_failure=args.fail_mcp,
    )
    print(
        json.dumps(
            {
                "status": state["status"],
                "plan": state["plan"],
                "route_decision": state["route_decision"],
                "answer_preview": state["final_answer"][:600],
                "errors": state["error"],
                "trace": summarize_trace(state["trace"]),
                "elapsed_ms": int((time.time() - started) * 1000),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
