"""Shared pytest fixtures and Celery test-harness configuration.

Unit tests must never require a live RabbitMQ or Redis instance. Celery's
result backend is therefore swapped for the in-process ``cache+memory://``
backend so ``update_state()`` progress reporting still works while the
application's Redis client is mocked per-test.
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("CACHE_TTL_SECONDS", "3600")


@pytest.fixture(scope="session", autouse=True)
def _celery_test_backend():
    """Configures Celery for eager, in-memory execution before any task is imported."""
    from app.core.celery_app import celery_app

    celery_app.conf.update(
        result_backend="cache+memory://",
        task_always_eager=True,
        task_store_eager_result=True,
        task_eager_propagates=True,
        broker_url="memory://",
    )
    yield
    celery_app.conf.update(
        result_backend=celery_app.conf.result_backend,
    )


@pytest.fixture
def fake_task_self():
    """Builds a stub Celery task ``self`` for driving retry/backoff logic directly.

    Using the raw undecorated function lets a test control ``request.retries``
    and assert on whether the task requested a retry, without depending on
    Celery's eager-mode retry re-dispatch behaviour. ``retry()`` raises
    ``Retry`` exactly as the real implementation does.
    """
    from celery.exceptions import Retry

    def _build(retries: int = 0, max_retries: int = 3) -> MagicMock:
        stub = MagicMock()
        stub.request.id = "unit-task-id"
        stub.request.retries = retries
        stub.max_retries = max_retries
        stub.update_state.return_value = None
        stub.retry.side_effect = Retry("task requested retry", None)
        return stub

    return _build


@pytest.fixture
def in_memory_cache():
    """Dict-backed stand-in for the Redis prediction cache used inside worker tasks.

    Exposes ``patch()`` so a test can redirect ``app.workers.tasks`` cache calls
    at the in-memory store instead of hitting a real Redis instance.
    """
    from contextlib import ExitStack
    from unittest.mock import patch

    store: dict[str, dict] = {}

    def _get(cache_key: str):
        return store.get(cache_key)

    def _set(cache_key: str, data: dict, ttl=None) -> bool:
        store[cache_key] = data
        return True

    class _Cache:
        data = store
        get = staticmethod(_get)
        set = staticmethod(_set)

        @staticmethod
        @contextmanager
        def patch():
            with ExitStack() as stack:
                stack.enter_context(
                    patch("app.workers.tasks.get_cached_prediction", side_effect=_get)
                )
                stack.enter_context(
                    patch("app.workers.tasks.set_cached_prediction", side_effect=_set)
                )
                yield _Cache

    return _Cache
