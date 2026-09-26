from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "documentation" in data
    assert data["status"] == "operational"


def test_liveness_probe():
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_prediction_validation_error():
    # Text too short (< 2 characters)
    response = client.post("/api/v1/predict", json={"text": "a"})
    assert response.status_code == 422


@patch("app.api.routes.get_cached_prediction")
@patch("app.api.routes.analyze_text_task.delay")
def test_prediction_cache_miss_dispatches_task(mock_delay, mock_cache):
    # Mock cache miss
    mock_cache.return_value = None

    # Mock Celery task dispatch
    mock_task = MagicMock()
    mock_task.id = "mock-task-id-12345"
    mock_delay.return_value = mock_task

    payload = {"text": "The distributed system handles async inference reliably."}
    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 202
    data = response.json()
    assert data["task_id"] == "mock-task-id-12345"
    assert data["status"] == "PENDING"
    assert data["cached"] is False
    assert "mock-task-id-12345" in data["check_status_url"]


@patch("app.api.routes.get_cached_prediction")
def test_prediction_cache_hit_returns_cached_result(mock_cache):
    # Mock cache HIT
    mock_cache.return_value = {
        "input_hash": "abc1234",
        "sentiment": {"label": "POSITIVE", "score": 0.85, "confidence": 0.92},
        "risk": {"risk_level": "SAFE", "risk_score": 0.05, "flagged_categories": []},
        "extracted_keywords": ["distributed", "system"],
        "word_count": 6,
        "char_count": 45,
        "model_version": "v1.2.0-tfidf-logreg",
        "processing_time_ms": 1.25,
    }

    payload = {"text": "Identical query that was previously cached."}
    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["cached"] is True
    assert data["status"] == "SUCCESS"
    assert data["result"]["sentiment"]["label"] == "POSITIVE"


@patch("app.api.routes.AsyncResult")
def test_get_task_status_pending(mock_async_result):
    mock_res = MagicMock()
    mock_res.state = "PENDING"
    mock_res.info = None
    mock_async_result.return_value = mock_res

    response = client.get("/api/v1/tasks/task-abc")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "PENDING"
    assert data["progress"] == 0


@patch("app.api.routes.AsyncResult")
def test_get_task_status_progress(mock_async_result):
    mock_res = MagicMock()
    mock_res.state = "PROGRESS"
    mock_res.info = {"progress": 60, "stage": "Executing dual NLP inference pipelines"}
    mock_async_result.return_value = mock_res

    response = client.get("/api/v1/tasks/task-abc")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "PROGRESS"
    assert data["progress"] == 60
    assert "Executing dual NLP" in data["stage"]
