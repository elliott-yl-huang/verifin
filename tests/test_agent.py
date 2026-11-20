import json

import pytest

from verifin.agent import Agent, parse_periods
from verifin.data import build
from verifin.evals import run


class FakeLLM:
    """Scripted model: calls a tool, hallucinates once, then repairs using the tool output it was given."""
    model = "fake"

    def __init__(self):
        self.turn = 0

    def chat(self, messages, tools=None):
        self.turn += 1
        if self.turn == 1:
            return {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {
                "name": "budget_variance", "arguments": '{"period": "2025"}'}}]}
        if self.turn == 2:
            return {"role": "assistant", "content": "Actual revenue was $99.99B [F17]."}
        tool_out = json.loads(next(m["content"] for m in messages if m["role"] == "tool"))
        assert "Verifier rejected" in messages[-1]["content"]
        return {"role": "assistant", "content": f"Actual revenue was {tool_out['facts']['actual']['display']}."}


@pytest.fixture(scope="module")
def con():
    return build()


def test_parse_periods():
    assert parse_periods("Q2 2025 vs 2025-Q1 and 2024-03, FY2023") == ["2025-Q2", "2025-Q1", "2024-03", "2023"]


def test_offline_answers_are_fully_grounded(con):
    a = Agent(con)
    for q in ("Why did Gaming revenue change in 2025?", "Forecast revenue next 4 months", "Audit journal entries"):
        ans = a.ask(q)
        assert ans.verification.ok and ans.verification.grounding_rate == 1.0
        assert ans.audit_pack()["facts"]


def test_llm_repair_loop(con):
    ans = Agent(con, FakeLLM()).ask("How did 2025 revenue compare to budget?")
    assert ans.repairs == 1 and ans.verification.ok and "$99.99B" not in ans.text


def test_eval_gates_pass():
    assert run()["metrics"]["passed"]
