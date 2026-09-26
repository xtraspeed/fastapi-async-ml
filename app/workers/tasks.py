import time
from typing import Any

from app.core.cache import (
    compute_cache_key,
    get_cached_prediction,
    set_cached_prediction,
)
from app.core.celery_app import celery_app
from app.core.logger import logger
from app.models.ml_model import get_ml_model


def _exponential_backoff(retries: int) -> int:
    """Returns an exponential backoff delay in seconds (1s, 2s, 4s, ...)."""
    return 2 ** retries


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
    cache_key = compute_cache_key(text)
    logger.info(f"Task {task_id}: Started processing for hash {input_hash}")

    try:
        # Step 1: Preprocessing & Validation
        self.update_state(
            state="PROGRESS",
            meta={"progress": 20, "stage": "Validating and normalizing text input"},
        )
        time.sleep(0.1)  # Simulated micro-delay for realistic pipeline stages

        # Step 2: ML Inference (short-circuited when a prior run already cached the payload)
        self.update_state(
            state="PROGRESS",
            meta={"progress": 60, "stage": "Executing dual NLP inference pipelines"},
        )
        cached = get_cached_prediction(cache_key)
        if cached is not None:
            logger.info(f"Task {task_id}: cache hit, skipping duplicate inference")
            prediction_result = cached
        else:
            model = get_ml_model()
            prediction_result = model.predict(text=text, input_hash=input_hash)

        # Step 3: Cache storage in Redis
        self.update_state(
            state="PROGRESS",
            meta={"progress": 85, "stage": "Persisting result in Redis cache"},
        )
        set_cached_prediction(cache_key, prediction_result)

        # Step 4: Finalize
        self.update_state(
            state="PROGRESS",
            meta={"progress": 100, "stage": "Analysis complete"},
        )
        logger.info(f"Task {task_id}: Successfully completed inference for hash {input_hash}")
        return prediction_result

    except Exception as exc:
        retries = self.request.retries
        if retries >= self.max_retries:
            logger.critical(
                f"Task {task_id} exhausted max retries ({self.max_retries}); failing permanently."
            )
            raise

        countdown = _exponential_backoff(retries)
        logger.warning(
            f"Task {task_id} failed ({exc}); retry {retries + 1}/{self.max_retries} in {countdown}s"
        )
        raise self.retry(exc=exc, countdown=countdown)


@celery_app.task(
    bind=True,
    name="app.workers.tasks.batch_analyze_text_task",
    max_retries=3,
    default_retry_delay=5,
)
def batch_analyze_text_task(self, items: list[dict[str, Any]], batch_id: str) -> dict[str, Any]:
    """Processes a batch of prediction requests iteratively with overall batch progress."""
    total = len(items)
    results: list[dict[str, Any]] = []
    cached_items = 0
    model = None

    logger.info(f"Task {self.request.id}: Started batch {batch_id} with {total} items")

    try:
        for idx, item in enumerate(items, start=1):
            text = item["text"]
            input_hash = item["input_hash"]
            cache_key = compute_cache_key(text)

            self.update_state(
                state="PROGRESS",
                meta={
                    "progress": int(((idx - 1) / total) * 100),
                    "stage": f"Processing item {idx}/{total}",
                    "current_item_index": idx,
                    "total_items": total,
                },
            )

            cached = get_cached_prediction(cache_key)
            if cached is not None:
                cached_items += 1
                res = cached
            else:
                if model is None:
                    model = get_ml_model()
                res = model.predict(text=text, input_hash=input_hash)
                set_cached_prediction(cache_key, res)

            results.append(res)

        self.update_state(
            state="PROGRESS",
            meta={
                "progress": 100,
                "stage": f"Completed {total}/{total} items ({cached_items} cached)",
                "total_items": total,
            },
        )

        logger.info(
            f"Task {self.request.id}: Batch {batch_id} completed, {cached_items}/{total} served from cache"
        )
        return {
            "batch_id": batch_id,
            "total_processed": total,
            "cached_items": cached_items,
            "results": results,
        }

    except Exception as exc:
        retries = self.request.retries
        if retries >= self.max_retries:
            logger.critical(
                f"Batch {batch_id} exhausted max retries ({self.max_retries}); failing permanently."
            )
            raise

        countdown = _exponential_backoff(retries)
        logger.warning(
            f"Batch {batch_id} failed ({exc}); retry {retries + 1}/{self.max_retries} in {countdown}s"
        )
        raise self.retry(exc=exc, countdown=countdown)
