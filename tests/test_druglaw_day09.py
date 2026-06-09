import unittest
from unittest.mock import patch


SAMPLE_CHUNK = {
    "content": "Dieu 249 quy dinh ve toi tang tru trai phep chat ma tuy.",
    "score": 0.91,
    "source": "hybrid",
    "metadata": {
        "source": "bo-luat-hinh-su-2015.md",
        "type": "legal",
        "chunk_index": 1,
    },
}


SAMPLE_MCP_CHUNK = {
    "content": "PageIndex tim thay noi dung lien quan den chat ma tuy.",
    "score": 0.72,
    "source": "pageindex",
    "metadata": {
        "source": "luat-phong-chong-ma-tuy-2021.pdf",
        "type": "pageindex",
        "doc_id": "demo-doc",
    },
}


def fake_synthesis(query, context_chunks):
    return "Cau tra loi demo co citation [bo-luat-hinh-su-2015.md, 2015]."


class TestDrugLawDay09(unittest.TestCase):
    @patch("day08_artifact.src.day09_multi_agent.synthesize_with_citations", side_effect=fake_synthesis)
    @patch("day08_artifact.src.day09_multi_agent.pageindex_search", return_value=[SAMPLE_MCP_CHUNK])
    @patch("day08_artifact.src.day09_multi_agent.retrieve", return_value=[SAMPLE_CHUNK])
    def test_supervisor_runs_workers_and_trace(self, *_):
        from druglaw_day09 import run_day09_multi_agent

        state = run_day09_multi_agent("Hoi ve hinh phat ma tuy", top_k=2)

        self.assertEqual(state["status"], "completed")
        self.assertEqual(
            state["plan"],
            ["retrieval_worker", "mcp_tool_worker", "synthesis_worker"],
        )
        self.assertTrue(state["final_answer"])
        self.assertGreaterEqual(len(state["trace"]), 5)
        self.assertIn("mcp.call", [event["event"] for event in state["trace"]])

    @patch("day08_artifact.src.day09_multi_agent.synthesize_with_citations", side_effect=fake_synthesis)
    @patch("day08_artifact.src.day09_multi_agent.retrieve", return_value=[SAMPLE_CHUNK])
    def test_route_can_skip_external_tool(self, *_):
        from druglaw_day09 import run_day09_multi_agent

        state = run_day09_multi_agent(
            "Hoi ve hinh phat ma tuy",
            top_k=2,
            use_external_tool=False,
        )

        self.assertEqual(state["status"], "completed")
        self.assertNotIn("mcp_tool_worker", state["plan"])
        self.assertEqual(state["tool_result"], {})

    @patch("day08_artifact.src.day09_multi_agent.synthesize_with_citations", side_effect=fake_synthesis)
    @patch("day08_artifact.src.day09_multi_agent.retrieve", return_value=[SAMPLE_CHUNK])
    def test_mcp_worker_failure_is_graceful(self, *_):
        from druglaw_day09 import run_day09_multi_agent

        state = run_day09_multi_agent(
            "Hoi ve hinh phat ma tuy",
            top_k=2,
            force_tool_failure=True,
        )

        self.assertEqual(state["status"], "degraded")
        self.assertTrue(state["final_answer"])
        self.assertEqual(state["tool_result"]["status"], "failed")
        self.assertTrue(state["error"])
        self.assertIn("error", [event["status"] for event in state["trace"]])

    def test_blueprint_documents_rubric_evidence(self):
        from druglaw_day09.multi_agent import DAY09_FIELD_OWNERS, WORKER_CONTRACTS, blueprint

        expected_fields = {
            "task",
            "user_context",
            "plan",
            "route_decision",
            "retrieval_result",
            "tool_result",
            "synthesis_draft",
            "final_answer",
            "status",
            "trace",
            "error",
        }
        self.assertTrue(expected_fields.issubset(DAY09_FIELD_OWNERS))
        self.assertIn("supervisor", WORKER_CONTRACTS)
        self.assertIn("mcp_tool_worker", WORKER_CONTRACTS)
        self.assertIn("trace_required_fields", blueprint())


if __name__ == "__main__":
    unittest.main()
