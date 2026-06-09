"""Exercise 2: add tools and a knowledge-base entry.

This file is the completed hands-on version referenced by exercises/README.md.
It still uses the same manual tool-call loop as Stage 2 so students can inspect
exactly how function calling is wired.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from common.llm import get_llm


LEGAL_KNOWLEDGE = [
    {
        "id": "ucc_breach",
        "keywords": ["breach", "contract", "remedies", "damages", "ucc"],
        "text": (
            "Under the Uniform Commercial Code (UCC) Article 2, remedies for breach of contract "
            "include expectation damages, consequential damages, specific performance, and cover "
            "damages. Statute of limitations is typically 4 years (UCC § 2-725)."
        ),
    },
    {
        "id": "labor_law",
        "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
        "text": (
            "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể đơn phương "
            "chấm dứt hợp đồng trong các trường hợp như người lao động thường xuyên không hoàn "
            "thành công việc, ốm đau kéo dài, thiên tai/hỏa hoạn, hoặc người lao động đủ tuổi "
            "nghỉ hưu. Việc chấm dứt phải có căn cứ và tuân thủ thời hạn báo trước."
        ),
    },
]


@tool
def search_legal_knowledge(query: str) -> str:
    """Search the legal knowledge base."""
    query_lower = query.lower()
    for entry in LEGAL_KNOWLEDGE:
        if any(kw in query_lower for kw in entry["keywords"]):
            return f"[{entry['id']}] {entry['text']}"
    return "Không tìm thấy thông tin liên quan."


@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Check the statute of limitations for a case type.

    Args:
        case_type: Case type: contract, tort, property, labor.
    """
    limits = {
        "contract": "4 năm đối với hợp đồng mua bán hàng hóa theo UCC § 2-725.",
        "tort": "Thường 2-3 năm tùy bang/thẩm quyền.",
        "property": "Thường khoảng 5 năm tùy loại tranh chấp.",
        "labor": "Tranh chấp lao động có thời hiệu riêng; cần xác định loại tranh chấp cụ thể theo Bộ luật Lao động Việt Nam.",
    }
    return limits.get(case_type.lower(), "Không xác định. Hãy dùng contract, tort, property hoặc labor.")


async def main():
    load_dotenv()
    llm = get_llm()

    tools = [search_legal_knowledge, check_statute_of_limitations]
    tool_map = {tool_item.name: tool_item for tool_item in tools}
    llm_with_tools = llm.bind_tools(tools)

    question = "Thời hiệu khởi kiện vụ vi phạm hợp đồng là bao lâu?"
    messages = [
        SystemMessage(content="Bạn là chuyên gia pháp lý. Sử dụng tools để tra cứu thông tin."),
        HumanMessage(content=question),
    ]

    print(f"Câu hỏi: {question}\n")
    response = await llm_with_tools.ainvoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"Tool call: {tool_call['name']} {tool_call['args']}")
            tool_result = await tool_map[tool_call["name"]].ainvoke(tool_call["args"])
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))

        final_response = await llm_with_tools.ainvoke(messages)
        print(f"\nKết quả:\n{final_response.content}")
    else:
        print(f"\nKết quả:\n{response.content}")


if __name__ == "__main__":
    asyncio.run(main())
