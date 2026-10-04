from unittest.mock import patch

from rag import qa_engine


def test_answer_query_refuses_when_no_context():
    with patch.object(qa_engine, "retrieve", return_value={"documents": [[]], "metadatas": [[]]}):
        result = qa_engine.answer_query("What is Titanic about?")
    assert "don't have" in result["answer"].lower()
    assert result["verification"]["valid"] is False


def test_answer_query_rejects_uncited_answer(monkeypatch):
    fake_results = {
        "documents": [["Hello, Neo."]],
        "metadatas": [[{
            "movie_title": "The Matrix",
            "start_time": "00:00:01,000",
            "end_time": "00:00:03,000",
            "speakers": "NEO",
        }]],
    }
    monkeypatch.setattr(qa_engine, "retrieve", lambda *a, **k: fake_results)
    monkeypatch.setattr(
        qa_engine,
        "generate_grounded_answer",
        lambda *a, **k: "Titanic is a 1997 film by James Cameron.",
    )
    result = qa_engine.answer_query("What is Titanic about?")
    assert "couldn't ground" in result["answer"].lower()
    assert result["verification"]["valid"] is False


def test_agent_refuses_when_retrieval_empty(monkeypatch):
    from agent import core_agent
    monkeypatch.setattr(core_agent, "retrieve", lambda *a, **k: {"documents": [[]], "metadatas": [[]]})

    # Force the model to call query_movie_database
    from unittest.mock import MagicMock
    tc = MagicMock()
    tc.id = "t1"
    tc.function.name = "query_movie_database"
    tc.function.arguments = '{"query": "What is Titanic about?"}'
    msg = MagicMock()
    msg.tool_calls = [tc]
    msg.content = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]

    monkeypatch.setattr(core_agent.client.chat.completions, "create", lambda **k: resp)

    result = core_agent.get_agent_response("What is Titanic about?", auto_send_email=False)
    assert "don't have" in result["answer"].lower()