import time
from typing import Any

try:
    from celery.exceptions import MaxRetriesExceededError
except ImportError:
    class MaxRetriesExceededError(Exception):
        pass

from app.core.cache import compute_cache_key, set_cached_prediction
from app.core.celery_app import celery_app
from app.core.logger import logger
from app.models.ml_model import get_ml_model


@celery_app.task(
    bind=True,
    name="app.workers.tasks.analyze_text_task",
    max_retries=3,
    default_retry_delay=5,
)
def analyze_text_task(self, text: str, input_hash: str) -> dict[str, Any]:
    """
    Asynchronous Celery task that executes ML text analysis,
    emits granular progress status updates, and caches the result.
    """
    task_id = self.request.id
    logger.info(f"Task {task_id}: Started processing for hash {input_hash}")

    try:
        # Step 1: Preprocessing & Validation
        self.update_state(
            state="PROGRESS",
            meta={"progress": 20, "stage": "Validating and normalizing text input"},
        )
        time.sleep(0.1)  # Simulated micro-delay for realistic pipeline stages

        # Step 2: ML Inference
        self.update_state(
            state="PROGRESS",
            meta={"progress": 60, "stage": "Executing dual NLP inference pipelines"},
        )
        model = get_ml_model()
        prediction_result = model.predict(text=text, input_hash=input_hash)

        # Step 3: Cache storage in Redis
        self.update_state(
            state="PROGRESS",
            meta={"progress": 85, "stage": "Persisting result in Redis cache"},
        )
        cache_key = compute_cache_key(text)
        set_cached_prediction(cache_key, prediction_result)

        # Step 4: Finalize
        self.update_state(
            state="PROGRESS",
            meta={"progress": 100, "stage": "Analysis complete"},
        )
        logger.info(f"Task {task_id}: Successfully completed inference for hash {input_hash}")
        return prediction_result

    except Exception as exc:
        logger.error(f"Task {task_id} encountered an error: {exc}", exc_info=True)
        try:
            # Exponential backoff retry
            countdown = 2 ** self.request.retries
            raise self.retry(exc=exc, countdown=countdown)
        except MaxRetriesExceededError:
            logger.critical(f"Task {task_id} exceeded max retries. Failing permanently.")
            raise


@celery_app.task(
    bind=True,
    name="app.workers.tasks.batch_analyze_text_task",
)
def batch_analyze_text_task(self, items: list[dict[str, Any]], batch_id: str) -> dict[str, Any]:
    """Processes a batch of prediction requests iteratively with overall batch progress."""
    total = len(items)
    results = []
    model = get_ml_model()

    for idx, item in enumerate(items, start=1):
        text = item["text"]
        input_hash = item["input_hash"]
        cache_key = compute_cache_key(text)

        # Predict
        res = model.predict(text=text, input_hash=input_hash)
        set_cached_prediction(cache_key, res)
        results.append(res)

        # Report batch progress
        pct = int((idx / total) * 100)
        self.update_state(
            state="PROGRESS",
            meta={
                "progress": pct,
                "stage": f"Processed {idx}/{total} items",
                "current_item_index": idx,
                "total_items": total,
            },
        )

    return {
        "batch_id": batch_id,
        "total_processed": total,
        "results": results,
    }
