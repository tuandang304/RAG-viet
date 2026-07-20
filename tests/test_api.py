from fastapi.testclient import TestClient

from rag_vie.api.app import app
from rag_vie.api.service import manager

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_indexes_empty_when_no_data(tmp_path, monkeypatch):
    monkeypatch.setattr("rag_vie.api.service.settings.index_dir", str(tmp_path / "missing"))
    monkeypatch.setattr("rag_vie.api.service.settings.checkpoint_dir", str(tmp_path / "missing"))
    res = client.get("/api/indexes")
    assert res.status_code == 200
    body = res.json()
    assert body["index_dirs"] == []
    assert body["mlp_checkpoints"] == []


def test_query_missing_index_returns_404(tmp_path, monkeypatch):
    manager._cache.clear()
    res = client.post(
        "/api/query",
        json={"query": "Thủ đô của Việt Nam là gì?", "index_dir": str(tmp_path / "nope")},
    )
    assert res.status_code == 404
    assert "index" in res.json()["detail"].lower()


def test_query_rejects_empty_string():
    res = client.post("/api/query", json={"query": ""})
    assert res.status_code == 422


def test_static_ui_served_at_root():
    res = client.get("/")
    assert res.status_code == 200
    assert "RAG_vie" in res.text


def test_compare_missing_index_returns_404(tmp_path):
    manager._cache.clear()
    res = client.post(
        "/api/compare",
        json={"query": "Thủ đô của Việt Nam là gì?", "index_dir": str(tmp_path / "nope")},
    )
    assert res.status_code == 404
    assert "index" in res.json()["detail"].lower()


def test_compare_rejects_empty_query():
    res = client.post("/api/compare", json={"query": ""})
    assert res.status_code == 422
