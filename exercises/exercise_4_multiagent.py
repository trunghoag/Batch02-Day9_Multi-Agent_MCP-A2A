"""Exercise 4: add a privacy agent to the in-process multi-agent system."""

import asyncio
import os
import sys
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from common.llm import get_llm


def _last_wins(left: str | None, right: str | None) -> str:
    """Reducer: newest non-null value wins."""
    return right if right is not None else (left or "")


class State(TypedDict):
    question: str
    law_analysis: Annotated[str, _last_wins]
    tax_analysis: Annotated[str, _last_wins]
    compliance_analysis: Annotated[str, _last_wins]
    privacy_analysis: Annotated[str, _last_wins]
    final_response: str


def law_agent(state: State) -> dict:
    """General legal analysis agent."""
    prompt = f"""Bạn là chuyên gia pháp lý. Phân tích câu hỏi sau:

{state['question']}

Tập trung vào: hợp đồng, trách nhiệm dân sự, quyền và nghĩa vụ pháp lý."""
    response = get_llm().invoke([HumanMessage(content=prompt)])
    return {"law_analysis": response.content}


def check_routing(state: State) -> list[Send]:
    """Route to specialist agents based on keywords."""
    question_lower = state["question"].lower()
    tasks: list[Send] = []

    if any(kw in question_lower for kw in ["tax", "irs", "thuế", "fbar", "fatca"]):
        tasks.append(Send("tax_agent", state))

    if any(kw in question_lower for kw in ["compliance", "sec", "regulation", "sox", "aml", "tuân thủ"]):
        tasks.append(Send("compliance_agent", state))

    if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "ccpa", "dữ liệu", "rò rỉ"]):
        tasks.append(Send("privacy_agent", state))

    return tasks if tasks else [Send("aggregate_results", state)]


def tax_agent(state: State) -> dict:
    """Tax specialist."""
    prompt = f"""Bạn là chuyên gia thuế. Phân tích khía cạnh thuế trong câu hỏi:

Câu hỏi: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Tập trung: IRS, tax evasion, penalties, FBAR, FATCA."""
    response = get_llm().invoke([HumanMessage(content=prompt)])
    return {"tax_analysis": response.content}


def compliance_agent(state: State) -> dict:
    """Compliance specialist."""
    prompt = f"""Bạn là chuyên gia compliance. Phân tích khía cạnh tuân thủ:

Câu hỏi: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Tập trung: SEC, SOX, FCPA, AML, regulatory violations."""
    response = get_llm().invoke([HumanMessage(content=prompt)])
    return {"compliance_analysis": response.content}


def privacy_agent(state: State) -> dict:
    """Privacy/data-protection specialist."""
    prompt = f"""Bạn là chuyên gia bảo vệ dữ liệu cá nhân và quyền riêng tư.

Câu hỏi: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Tập trung: GDPR, CCPA, data breach notification, consent, data minimisation,
quyền của chủ thể dữ liệu, và rủi ro khi rò rỉ dữ liệu khách hàng."""
    response = get_llm().invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}


def aggregate_results(state: State) -> dict:
    """Synthesize all specialist outputs."""
    sections = []
    if state.get("law_analysis"):
        sections.append(f"PHÂN TÍCH PHÁP LÝ:\n{state['law_analysis']}")
    if state.get("tax_analysis"):
        sections.append(f"PHÂN TÍCH THUẾ:\n{state['tax_analysis']}")
    if state.get("compliance_analysis"):
        sections.append(f"PHÂN TÍCH TUÂN THỦ:\n{state['compliance_analysis']}")
    if state.get("privacy_analysis"):
        sections.append(f"PHÂN TÍCH PRIVACY / DỮ LIỆU:\n{state['privacy_analysis']}")

    combined = "\n\n".join(sections)
    prompt = f"""Tổng hợp các phân tích sau thành một báo cáo pháp lý hoàn chỉnh:

{combined}

Câu hỏi gốc: {state['question']}

Hãy tạo một báo cáo ngắn gọn, có cấu trúc rõ ràng."""
    response = get_llm().invoke([HumanMessage(content=prompt)])
    return {"final_response": response.content}


def build_graph():
    """Build the multi-agent graph with privacy routing."""
    graph = StateGraph(State)
    graph.add_node("law_agent", law_agent)
    graph.add_node("check_routing", check_routing)
    graph.add_node("tax_agent", tax_agent)
    graph.add_node("compliance_agent", compliance_agent)
    graph.add_node("privacy_agent", privacy_agent)
    graph.add_node("aggregate_results", aggregate_results)

    graph.add_edge(START, "law_agent")
    graph.add_edge("law_agent", "check_routing")
    graph.add_conditional_edges("check_routing", lambda x: x)
    graph.add_edge("tax_agent", "aggregate_results")
    graph.add_edge("compliance_agent", "aggregate_results")
    graph.add_edge("privacy_agent", "aggregate_results")
    graph.add_edge("aggregate_results", END)
    return graph.compile()


async def main():
    load_dotenv()
    question = "Nếu công ty bị rò rỉ dữ liệu khách hàng, hậu quả pháp lý, privacy và thuế là gì?"

    print("=" * 70)
    print("MULTI-AGENT SYSTEM với Privacy Agent")
    print("=" * 70)
    print(f"\nCâu hỏi: {question}\n")

    graph = build_graph()
    result = await graph.ainvoke(
        {
            "question": question,
            "law_analysis": "",
            "tax_analysis": "",
            "compliance_analysis": "",
            "privacy_analysis": "",
            "final_response": "",
        }
    )

    print("\n" + "=" * 70)
    print("KẾT QUẢ CUỐI CÙNG")
    print("=" * 70)
    print(result["final_response"])
    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
