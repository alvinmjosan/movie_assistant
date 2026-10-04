def test_agent_rejects_uncited_final_answer(monkeypatch):
    """Retrieval returns irrelevant blocks; LLM answers without citations; must be rejected."""
    from agent import core_agent
    from unittest.mock import MagicMock

    # 1. Stub the retrieval to return *some* blocks (non-empty).
    fake_blocks = [{
        "dialogue": "Wake up, Neo.",
        "movie_title": "The Matrix 1999",
        "start_time": "00:00:01,000",
        "end_time": "00:00:03,000",
        "speakers": "NEO",
    }]
    monkeypatch.setattr(core_agent, "_run_movie_query", lambda q: {
        "context_str": "some matrix context",
        "blocks": fake_blocks,
        "empty": False,
    })

    # 2. First LLM turn: tool call. Second LLM turn: uncited Titanic answer.
    tool_call = MagicMock()
    tool_call.id = "t1"
    tool_call.function.name = "query_movie_database"
    tool_call.function.arguments = '{"query": "Titanic"}'

    first_msg = MagicMock()
    first_msg.tool_calls = [tool_call]
    first_msg.content = None
    first_choice = MagicMock()
    first_choice.message = first_msg
    first_resp = MagicMock()
    first_resp.choices = [first_choice]

    second_msg = MagicMock()
    second_msg.tool_calls = None
    second_msg.content = "Titanic is a 1997 film by James Cameron starring Leonardo DiCaprio."
    second_choice = MagicMock()
    second_choice.message = second_msg
    second_resp = MagicMock()
    second_resp.choices = [second_choice]

    monkeypatch.setattr(
        core_agent.client.chat.completions,
        "create",
        lambda **k: first_resp if k["messages"][-1].get("role") == "user" else second_resp,
    )
    # Simpler: use side_effect list
    monkeypatch.setattr(
        core_agent.client.chat.completions, "create",
        MagicMock(side_effect=[first_resp, second_resp]),
    )

    result = core_agent.get_agent_response("What is Titanic about?", auto_send_email=False)
    assert "couldn't ground" in result["answer"].lower()
    assert result["verification"]["valid"] is False