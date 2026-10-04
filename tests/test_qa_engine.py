from rag import qa_engine


def test_build_context_and_format():
    results = {
        "documents": [["Hello Neo.", "Wake up."]],
        "metadatas": [[
            {"movie_title": "The Matrix", "start_time": "00:00:01,000", "end_time": "00:00:03,000", "speakers": "NEO"},
            {"movie_title": "The Matrix", "start_time": "00:00:04,000", "end_time": "00:00:06,000", "speakers": "TRINITY"},
        ]],
    }
    blocks = qa_engine.build_context(results)
    ctx = qa_engine.format_context(blocks)
    assert "The Matrix" in ctx
    assert "00:00:01,000 - 00:00:03,000" in ctx


def test_verify_citations_matches_and_flags():
    blocks = [{
        "dialogue": "x", "movie_title": "The Matrix",
        "start_time": "00:00:01,000", "end_time": "00:00:03,000", "speakers": "NEO",
    }]
    good = "Neo wakes up [The Matrix, 00:00:01,000 - 00:00:03,000]"
    bad = "Neo wakes up [The Matrix, 00:99:99,000 - 00:99:99,000]"
    assert qa_engine.verify_citations(good, blocks)["valid"] is True
    assert qa_engine.verify_citations(bad, blocks)["valid"] is False