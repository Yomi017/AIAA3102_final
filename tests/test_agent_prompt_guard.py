"""Integration tests for Agent prompt guard behavior."""
from agent import Agent
from security.prompt_guard import ATTACK_TEST_PROMPTS


class _DummyModel:
    """Lightweight stand-in model that satisfies Agent.chat signature."""

    def chat(self, response: str, history, meta_instruction: str):
        return "Final Answer: ok", history


def _make_agent():
    return Agent(
        model=_DummyModel(),
        security_config={"enable_prompt_guard": True, "lock_on_violation": True},
    )


def test_guard_locks_session_on_attack():
    agent = _make_agent()
    result = agent._apply_prompt_guard(ATTACK_TEST_PROMPTS["Prompt Injection"])
    assert result is not None
    assert agent.prompt_guard_blocked is True

    # Once locked, the guard rejects follow-up prompts immediately.
    locked_message = agent._apply_prompt_guard("这是一条正常请求。")
    assert "会话因检测到提示词攻击" in locked_message


def test_reset_security_clears_lock():
    agent = _make_agent()
    agent.prompt_guard_blocked = True
    was_blocked = agent.reset_security()
    assert was_blocked is True
    assert agent.prompt_guard_blocked is False

    # Reset allows safe prompts to pass through.
    follow_up = agent._apply_prompt_guard("请解释注意力机制。")
    assert follow_up is None
