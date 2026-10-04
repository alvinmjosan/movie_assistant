from unittest.mock import patch

from ingestion import vector_store


def test_populate_uses_upsert(monkeypatch):
    upserts = []

    class FakeCollection:
        def upsert(self, documents, metadatas, ids):
            upserts.append({"docs": documents, "metas": metadatas, "ids": ids})

    monkeypatch.setattr(vector_store, "_collection", lambda *a, **k: FakeCollection())

    chunks = [{
        "text": "Hello",
        "metadata": {"movie_title": "M", "start_time": "00:00:01,000",
                     "end_time": "00:00:02,000", "chunk_id": "M_chunk_0"},
    }]
    vector_store.populate_vector_store(chunks)
    assert len(upserts) == 1
    assert upserts[0]["ids"] == ["M_chunk_0"]


def test_query_signature_intact():
    # ensure backwards-compatible signature still exists
    assert hasattr(vector_store, "query_vector_store")