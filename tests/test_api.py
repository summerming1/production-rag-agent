import json

from fastapi.testclient import TestClient

from production_rag.api import create_app


def _client(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"id": "a", "text": "alpha evidence", "source": "a.md"}) + "\n",
        encoding="utf-8",
    )
    return TestClient(create_app(str(corpus)))


def test_openapi_and_normal_json_requests(tmp_path):
    client = _client(tmp_path)
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/health").status_code == 200
    response = client.post("/retrieve", json={"question": "alpha", "top_k": 1})
    assert response.status_code == 200
    assert response.json()[0]["id"] == "a"


def test_request_validation(tmp_path):
    client = _client(tmp_path)
    assert client.post("/retrieve", json={"question": "", "top_k": 1}).status_code == 422
    assert client.post("/retrieve", json={"question": "alpha", "top_k": 0}).status_code == 422
    assert client.post("/retrieve", json={"question": "alpha", "top_k": 51}).status_code == 422
