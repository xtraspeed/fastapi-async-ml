from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.core.cache import compute_cache_key
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


SAMPLE_PREDICTION = {
    "input_hash": "h1",
    "sentiment": {"label": "POSITIVE", "score": 0.85, "confidence": 0.92},
    "risk": {"risk_level": "SAFE", "risk_score": 0.05, "flagged_categories": []},
    "extracted_keywords": ["release"],
    "word_count": 6,
    "char_count": 45,
    "model_version": "v1.2.0-tfidf-logreg",
    "processing_time_ms": 1.25,
}


@patch("app.api.routes.AsyncResult")
def test_get_task_status_success_returns_prediction(mock_async_result):
    mock_res = MagicMock()
    mock_res.state = "SUCCESS"
    mock_res.info = None
    mock_res.result = SAMPLE_PREDICTION
    mock_async_result.return_value = mock_res

    response = client.get("/api/v1/tasks/task-abc")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["progress"] == 100
    assert data["result"]["sentiment"]["label"] == "POSITIVE"
    assert data["result"]["input_hash"] == "h1"


@patch("app.api.routes.AsyncResult")
def test_get_task_status_success_returns_batch_result(mock_async_result):
    """A completed batch task must expose its aggregate result, not a null result."""
    mock_res = MagicMock()
    mock_res.state = "SUCCESS"
    mock_res.info = None
    mock_res.result = {
        "batch_id": "batch-1",
        "total_processed": 2,
        "cached_items": 1,
        "results": [SAMPLE_PREDICTION, {**SAMPLE_PREDICTION, "input_hash": "h2"}],
    }
    mock_async_result.return_value = mock_res

    response = client.get("/api/v1/tasks/batch-1")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["result"]["batch_id"] == "batch-1"
    assert data["result"]["total_processed"] == 2
    assert data["result"]["cached_items"] == 1
    assert len(data["result"]["results"]) == 2


@patch("app.api.routes.AsyncResult")
def test_get_task_status_failure(mock_async_result):
    mock_res = MagicMock()
    mock_res.state = "FAILURE"
    mock_res.info = "ValueError: boom"
    mock_async_result.return_value = mock_res

    response = client.get("/api/v1/tasks/task-abc")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "FAILURE"
    assert "boom" in data["error"]


@patch("app.api.routes.AsyncResult")
def test_get_task_status_retry_state(mock_async_result):
    mock_res = MagicMock()
    mock_res.state = "RETRY"
    mock_res.info = None
    mock_async_result.return_value = mock_res

    response = client.get("/api/v1/tasks/task-abc")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "RETRY"
    assert data["progress"] > 0
    assert "retry" in data["stage"].lower()


@patch("app.api.routes.AsyncResult")
def test_get_task_status_revoked(mock_async_result):
    mock_res = MagicMock()
    mock_res.state = "REVOKED"
    mock_res.info = None
    mock_res.result = None
    mock_async_result.return_value = mock_res

    response = client.get("/api/v1/tasks/task-abc")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "REVOKED"
    assert data["result"] is None


def test_prediction_rejects_whitespace_only_text():
    response = client.post("/api/v1/predict", json={"text": "    "})
    assert response.status_code == 422


def test_prediction_text_is_normalized():
    """Surrounding whitespace is stripped before hashing / dispatching."""
    with patch("app.api.routes.get_cached_prediction", return_value=None) as mock_cache:
        with patch("app.api.routes.analyze_text_task.delay") as mock_delay:
            mock_task = MagicMock()
            mock_task.id = "task-norm"
            mock_delay.return_value = mock_task

            response = client.post("/api/v1/predict", json={"text": "  padded text  "})

    assert response.status_code == 202
    assert mock_delay.call_args[0][0] == "padded text"
    assert mock_cache.call_args[0][0] == compute_cache_key("padded text")


@patch("app.api.routes.batch_analyze_text_task.delay")
def test_batch_prediction_dispatch(mock_delay):
    mock_task = MagicMock()
    mock_task.id = "batch-task-1"
    mock_delay.return_value = mock_task

    payload = {"items": [{"text": "great product"}, {"text": "terrible, a complete waste"}]}
    response = client.post("/api/v1/batch-predict", json=payload)

    assert response.status_code == 202
    data = response.json()
    assert data["total_items"] == 2
    assert len(data["tasks"]) == 1
    assert data["tasks"][0]["task_id"] == "batch-task-1"
    assert data["tasks"][0]["status"] == "PENDING"

    sent_items = mock_delay.call_args[0][0]
    assert len(sent_items) == 2
    assert all("input_hash" in item for item in sent_items)


def test_batch_prediction_rejects_empty_items():
    response = client.post("/api/v1/batch-predict", json={"items": []})
    assert response.status_code == 422


@patch("app.api.routes.celery_app.control.revoke")
def test_cancel_task(mock_revoke):
    response = client.delete("/api/v1/tasks/task-abc")
    assert response.status_code == 200
    assert response.json()["task_id"] == "task-abc"
    mock_revoke.assert_called_once()


@patch("app.api.routes.celery_app.control.revoke", side_effect=ConnectionError("broker down"))
def test_cancel_task_returns_503_when_broker_down(mock_revoke):
    """A broker outage is a dependency failure, not a 500."""
    response = client.delete("/api/v1/tasks/task-abc")
    assert response.status_code == 503
    assert "RabbitMQ" in response.json()["detail"]


@patch("app.api.routes.AsyncResult")
def test_get_task_status_returns_503_when_result_store_down(mock_async_result):
    """Polling is the hottest endpoint; a Redis outage must not surface as a 500."""
    mock_res = MagicMock()
    type(mock_res).state = property(
        lambda self: (_ for _ in ()).throw(ConnectionError("redis down"))
    )
    mock_async_result.return_value = mock_res

    response = client.get("/api/v1/tasks/task-abc")
    assert response.status_code == 503
    assert "Redis" in response.json()["detail"]


@patch("app.api.routes.analyze_text_task.delay", side_effect=ConnectionError("broker down"))
@patch("app.api.routes.get_cached_prediction", return_value=None)
def test_predict_returns_503_when_broker_down(mock_cache, mock_delay):
    response = client.post("/api/v1/predict", json={"text": "broker is down right now"})
    assert response.status_code == 503
    assert "RabbitMQ" in response.json()["detail"]


@patch("app.api.routes.batch_analyze_text_task.delay", side_effect=ConnectionError("broker down"))
def test_batch_predict_returns_503_when_broker_down(mock_delay):
    response = client.post("/api/v1/batch-predict", json={"items": [{"text": "some text here"}]})
    assert response.status_code == 503
    assert "RabbitMQ" in response.json()["detail"]


def test_readiness_probe_reports_degraded_without_infra():
    """No broker/cache running in unit tests -> probe must report degraded, not crash."""
    response = client.get("/health/ready")
    assert response.status_code in (200, 503)
    data = response.json()
    assert "redis" in data["services"]
    assert "rabbitmq" in data["services"]
