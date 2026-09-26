from datetime import datetime, timezone
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.core.cache import check_redis_health
from app.core.celery_app import check_rabbitmq_health
from app.models.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["Health & Observability"])


@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_probe() -> dict[str, str]:
    """Kubernetes / Docker liveness probe: returns 200 if API process is running."""
    return {"status": "alive"}


@router.get("/ready", response_model=HealthResponse)
async def readiness_probe():
    """
    Readiness probe: validates availability of dependent infrastructure
    (RabbitMQ broker and Redis result store/cache).
    """
    redis_ok = check_redis_health()
    rabbitmq_ok = check_rabbitmq_health()

    services_status = {
        "redis": "healthy" if redis_ok else "unreachable",
        "rabbitmq": "healthy" if rabbitmq_ok else "unreachable",
    }

    all_healthy = redis_ok and rabbitmq_ok
    payload = {
        "status": "healthy" if all_healthy else "degraded",
        "services": services_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    status_code = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=status_code, content=payload)
