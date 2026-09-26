import hashlib
import uuid
from fastapi import APIRouter, HTTPException, Request, Response, status

try:
    from celery.result import AsyncResult
except ImportError:
    class AsyncResult:
        def __init__(self, task_id, app=None):
            self.id = task_id
            self.state = "PENDING"
            self.info = None
            self.result = None

from app.core.cache import (
    compute_cache_key,
    get_cached_prediction,
)
from app.core.celery_app import celery_app
from app.core.logger import logger
from app.models.schemas import (
    BatchPredictionRequest,
    BatchResult,
    BatchTaskEntry,
    BatchTaskResponse,
    PredictionRequest,
    PredictionResult,
    TaskResponse,
    TaskStatusResponse,
)
from app.workers.tasks import analyze_text_task, batch_analyze_text_task

router = APIRouter(prefix="/api/v1", tags=["ML Predictions & Tasks"])


@router.post(
    "/predict",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit Text for Async ML Analysis",
    description="Checks Redis cache first. If a cache miss occurs, dispatches a Celery task through RabbitMQ.",
)
async def submit_prediction(
    request: PredictionRequest, response: Response, req: Request
) -> TaskResponse:
    # 1. Compute deterministic hash for deduplication
    clean_text = request.text.strip()
    input_hash = hashlib.sha256(clean_text.lower().encode("utf-8")).hexdigest()
    cache_key = compute_cache_key(clean_text)

    # 2. Check Cache
    cached_data = get_cached_prediction(cache_key)
    if cached_data:
        response.status_code = status.HTTP_200_OK
        return TaskResponse(
            task_id=f"cached-{input_hash[:12]}",
            status="SUCCESS",
            message="Result fetched directly from Redis cache (idempotent lookup).",
            check_status_url="",
            cached=True,
            result=PredictionResult(**cached_data),
        )

    # 3. Cache Miss: Dispatch task asynchronously to RabbitMQ
    try:
        celery_task = analyze_text_task.delay(clean_text, input_hash)
        logger.info(f"Dispatched Celery task ID: {celery_task.id} for input hash {input_hash}")
    except Exception as e:
        logger.error(f"Failed to publish task to RabbitMQ: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Message broker (RabbitMQ) is currently unavailable to accept tasks.",
        )

    status_url = str(req.url_for("get_task_status", task_id=celery_task.id))

    return TaskResponse(
        task_id=celery_task.id,
        status="PENDING",
        message="Inference job dispatched to RabbitMQ queue. Poll check_status_url for progress.",
        check_status_url=status_url,
        cached=False,
        result=None,
    )


@router.post(
    "/batch-predict",
    response_model=BatchTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit Batch Items for Async Analysis",
)
async def submit_batch_predictions(
    request: BatchPredictionRequest, req: Request
) -> BatchTaskResponse:
    batch_id = str(uuid.uuid4())
    items_payload = []

    for item in request.items:
        clean_text = item.text.strip()
        h = hashlib.sha256(clean_text.lower().encode("utf-8")).hexdigest()
        items_payload.append({"text": clean_text, "input_hash": h})

    try:
        batch_task = batch_analyze_text_task.delay(items_payload, batch_id)
        logger.info(f"Dispatched batch task ID: {batch_task.id} with {len(items_payload)} items")
    except Exception as e:
        logger.error(f"Failed to publish batch task to RabbitMQ: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Message broker (RabbitMQ) is currently unavailable.",
        )

    status_url = str(req.url_for("get_task_status", task_id=batch_task.id))

    return BatchTaskResponse(
        batch_id=batch_id,
        total_items=len(items_payload),
        tasks=[
            BatchTaskEntry(
                task_id=batch_task.id,
                status="PENDING",
                check_status_url=status_url,
            )
        ],
    )


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    name="get_task_status",
    summary="Query Task Execution Status & Result",
    description="Inspects Celery state (PENDING, STARTED, PROGRESS, RETRY, SUCCESS, REVOKED, FAILURE) and returns progress percentage.",
)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    async_res = AsyncResult(task_id, app=celery_app)

    # Reading task state requires the Redis result store. An outage there is a
    # dependency failure, not an application fault, so report 503 rather than
    # letting it surface as a 500 from the global handler.
    try:
        state = async_res.state
    except Exception as exc:
        logger.error(f"Result store unavailable while polling task {task_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Result store (Redis) is currently unavailable to read task status.",
        )

    response = TaskStatusResponse(task_id=task_id, status=state)

    if state == "PENDING":
        response.progress = 0
        response.stage = "Queued in RabbitMQ, awaiting available worker"

    elif state == "STARTED":
        response.progress = 5
        response.stage = "Picked up by worker, initializing pipeline"

    elif state in ("PROGRESS", "RETRY"):
        info = async_res.info
        if isinstance(info, dict):
            response.progress = info.get("progress", 50)
            response.stage = info.get("stage", "Processing ML pipeline")
        elif state == "RETRY":
            response.progress = 10
            response.stage = "Transient failure, task scheduled for retry with backoff"
        else:
            response.progress = 10
            response.stage = "Processing ML pipeline"

    elif state == "SUCCESS":
        response.progress = 100
        response.stage = "Completed"
        data = async_res.result
        if isinstance(data, dict):
            if "sentiment" in data:
                response.result = PredictionResult(**data)
            elif "results" in data:
                response.result = BatchResult(**data)

    elif state == "REVOKED":
        response.progress = 100
        response.stage = "Revoked before execution"

    elif state == "FAILURE":
        response.progress = 100
        response.stage = "Failed"
        response.error = str(async_res.info or "Unknown task execution error")

    else:
        response.stage = f"State: {state}"

    return response


@router.delete(
    "/tasks/{task_id}",
    summary="Cancel / Revoke a Queued Task",
)
async def cancel_task(task_id: str) -> dict[str, str]:
    """Revokes a task from the queue or terminates it if already executing."""
    try:
        celery_app.control.revoke(task_id, terminate=True)
    except Exception as exc:
        logger.error(f"Failed to dispatch revocation for task {task_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Message broker (RabbitMQ) is currently unavailable to accept revocations.",
        )
    logger.info(f"Sent revocation signal for task ID: {task_id}")
    return {"task_id": task_id, "action": "revocation_signal_dispatched"}
