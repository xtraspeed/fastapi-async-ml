from app.config import settings
from app.core.logger import logger

try:
    from celery import Celery
    from kombu import Connection

    celery_app = Celery(
        "ml_worker",
        broker=settings.RABBITMQ_URL,
        backend=settings.REDIS_URL,
        include=["app.workers.tasks"],
    )

    celery_app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_time_limit=settings.CELERY_TASK_TIMEOUT,
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        result_expires=settings.CACHE_TTL_SECONDS,
        task_routes={
            "app.workers.tasks.analyze_text_task": {"queue": "ml_inference_queue"},
            "app.workers.tasks.batch_analyze_text_task": {"queue": "ml_batch_queue"},
        },
        # Fail fast on unavailable infrastructure so a publish attempt surfaces an
        # exception (and therefore a 503) instead of blocking the request inside
        # Celery's default broker/backend retry loops.
        broker_transport_options={
            "max_retries": 0,
            "socket_timeout": settings.BROKER_CONNECTION_TIMEOUT,
            "socket_connect_timeout": settings.BROKER_CONNECTION_TIMEOUT,
        },
        broker_connection_retry_on_startup=True,
        broker_connection_max_retries=0,
        result_backend_always_retry=False,
        # The Redis backend retries failed store/get operations 20 times by
        # default (~100s of blocking). Publishing must fail fast so the API can
        # return 503 instead of holding the HTTP request open.
        result_backend_transport_options={
            "retry_policy": {"max_retries": 0},
        },
        redis_socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
        redis_socket_connect_timeout=settings.REDIS_SOCKET_TIMEOUT,
        redis_retry_on_timeout=False,
    )

    def check_rabbitmq_health() -> bool:
        """Verifies that the RabbitMQ broker is reachable."""
        try:
            conn = Connection(settings.RABBITMQ_URL)
            conn.connect()
            conn.release()
            return True
        except Exception as e:
            logger.warning(f"RabbitMQ health check failed: {e}")
            return False

except ImportError:
    logger.warning("Celery or Kombu not found. Operating in mock task mode for local testing.")

    class _MockControl:
        def revoke(self, *args, **kwargs):
            pass

    class _MockCelery:
        def __init__(self, *args, **kwargs):
            self.conf = {}
            self.control = _MockControl()

        def task(self, *args, **kwargs):
            def decorator(func):
                class _TaskWrapper:
                    def __init__(self, f):
                        self.f = f
                        self.max_retries = kwargs.get("max_retries", 0)
                        self.request = type("Req", (), {"id": "mock-task-id", "retries": 0})()

                    def delay(self, *a, **k):
                        mock_async = type("AsyncRes", (), {"id": "mock-task-id"})()
                        return mock_async

                    def __call__(self, *a, **k):
                        return self.f(self, *a, **k)

                    def update_state(self, *a, **k):
                        pass

                    def retry(self, *a, **k):
                        pass

                return _TaskWrapper(func)

            return decorator

    celery_app = _MockCelery()

    def check_rabbitmq_health() -> bool:
        return False
