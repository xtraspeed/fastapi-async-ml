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
