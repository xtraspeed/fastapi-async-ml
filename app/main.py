from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.health import router as health_router
from app.api.routes import router as api_router
from app.config import settings
from app.core.logger import logger
from app.models.ml_model import get_ml_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for application startup and shutdown events."""
    logger.info(f"Starting {settings.PROJECT_NAME} in [{settings.ENVIRONMENT}] mode")
    # Pre-warm ML model pipeline so initial inferences are instant
    try:
        model = get_ml_model()
        logger.info(f"ML Model warmed up successfully: {model.MODEL_VERSION}")
    except Exception as e:
        logger.error(f"Failed to initialize ML model on startup: {e}")

    yield

    logger.info("Shutting down application...")


def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=(
            "Production-grade Asynchronous ML Inference Platform powered by "
            "FastAPI, Celery, RabbitMQ, and Redis."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Cross-Origin Resource Sharing
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach Routers
    app.include_router(health_router)
    app.include_router(api_router)

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "service": settings.PROJECT_NAME,
            "version": "1.0.0",
            "documentation": "/docs",
            "health_check": "/health/ready",
            "flower_dashboard": "http://localhost:5555",
            "status": "operational",
        }

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled server error on {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "InternalServerError", "message": "An unexpected error occurred."},
        )

    return app


app = create_application()
