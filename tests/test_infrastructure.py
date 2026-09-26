"""Tests for infrastructure configuration and health reporting.

These lock in the production guarantees documented in the README (fair dispatch,
late acknowledgment, fail-fast infrastructure handling) so a config regression
cannot silently reintroduce a hang or a lost task.
"""

from unittest.mock import patch

import pytest

from app.config import settings
from app.core.celery_app import celery_app, check_rabbitmq_health
from app.core.logger import logger

# Points the broker at a port with nothing listening so the connection is
# refused immediately rather than after a DNS/handshake timeout.
UNREACHABLE_BROKER = "amqp://guest:guest@127.0.0.1:5673//"


def test_documented_worker_guarantees():
    """README: prefetch_multiplier=1 for fair dispatch, acks_late for durability."""
    assert celery_app.conf.worker_prefetch_multiplier == 1
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_track_started is True
    assert celery_app.conf.task_time_limit == settings.CELERY_TASK_TIMEOUT


def test_routes_single_and_batch_to_distinct_queues():
    routes = celery_app.conf.task_routes
    assert routes["app.workers.tasks.analyze_text_task"]["queue"] == "ml_inference_queue"
    assert routes["app.workers.tasks.batch_analyze_text_task"]["queue"] == "ml_batch_queue"


def test_infrastructure_failures_do_not_block_requests():
    """A broker/backend outage must surface an exception promptly, not hang.

    Celery's defaults (broker retry loop, 20 backend retries) block the HTTP
    request for over a minute before the route can return 503.
    """
    conf = celery_app.conf
    assert conf.broker_transport_options["max_retries"] == 0
    assert conf.broker_connection_max_retries == 0
    assert conf.result_backend_always_retry is False
    assert conf.result_backend_transport_options["retry_policy"]["max_retries"] == 0
    assert conf.redis_socket_connect_timeout == settings.REDIS_SOCKET_TIMEOUT
    assert conf.redis_socket_timeout == settings.REDIS_SOCKET_TIMEOUT
    assert conf.redis_retry_on_timeout is False


def test_broker_transport_options_bound_socket_waits():
    options = celery_app.conf.broker_transport_options
    assert options["socket_connect_timeout"] == settings.BROKER_CONNECTION_TIMEOUT
    assert options["socket_timeout"] == settings.BROKER_CONNECTION_TIMEOUT


def test_serialization_is_locked_to_json():
    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.result_serializer == "json"
    assert celery_app.conf.accept_content == ["json"]


def test_result_expiry_matches_cache_ttl():
    assert celery_app.conf.result_expires == settings.CACHE_TTL_SECONDS


def test_rabbitmq_health_check_reports_false_without_raising():
    """An unreachable broker is a reported condition, never a crash."""
    with patch.object(settings, "RABBITMQ_URL", UNREACHABLE_BROKER):
        assert check_rabbitmq_health() is False


def test_redis_health_check_reports_false_without_raising():
    from app.core import cache as cache_module

    unreachable = "redis://127.0.0.1:6399/0"
    with patch.object(settings, "REDIS_URL", unreachable):
        with patch.object(cache_module, "_redis_client", None):
            assert cache_module.check_redis_health() is False


def test_cache_operations_degrade_gracefully_without_redis():
    """Cache failures must degrade to a miss, never propagate into the request."""
    from app.core import cache as cache_module

    unreachable = "redis://127.0.0.1:6399/0"
    with patch.object(settings, "REDIS_URL", unreachable):
        with patch.object(cache_module, "_redis_client", None):
            assert cache_module.get_cached_prediction("cache:prediction:abc") is None
            assert cache_module.set_cached_prediction("cache:prediction:abc", {"a": 1}) is False


def test_logger_is_configured_with_a_stream_handler():
    assert logger.name == "fastapi-celery-ml"
    assert len(logger.handlers) >= 1
