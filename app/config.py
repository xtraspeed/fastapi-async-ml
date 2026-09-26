import os

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        PROJECT_NAME: str = "Async ML Risk Intelligence Platform"
        ENVIRONMENT: str = "development"
        DEBUG: bool = True
        API_V1_STR: str = "/api/v1"

        # Message Broker (RabbitMQ)
        RABBITMQ_URL: str = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672//")

        # Result Backend & Cache (Redis)
        REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        # Caching & Worker Settings
        CACHE_TTL_SECONDS: int = 3600
        CELERY_TASK_TIMEOUT: int = 300
        MODEL_ARTIFACT_PATH: str = os.getenv("MODEL_ARTIFACT_PATH", "app/models/model_artifacts.joblib")

        # Fail-fast tuning: the API must return 503 quickly when the broker or
        # result store is unreachable, instead of blocking the HTTP request
        # inside Celery's default connection/retry loops.
        BROKER_CONNECTION_TIMEOUT: float = 3.0
        REDIS_SOCKET_TIMEOUT: float = 3.0

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

except ImportError:
    from pydantic import BaseModel

    class Settings(BaseModel):
        PROJECT_NAME: str = "Async ML Risk Intelligence Platform"
        ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
        DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")
        API_V1_STR: str = "/api/v1"

        RABBITMQ_URL: str = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672//")
        REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
        CELERY_TASK_TIMEOUT: int = int(os.getenv("CELERY_TASK_TIMEOUT", "300"))
        MODEL_ARTIFACT_PATH: str = os.getenv("MODEL_ARTIFACT_PATH", "app/models/model_artifacts.joblib")
        BROKER_CONNECTION_TIMEOUT: float = float(os.getenv("BROKER_CONNECTION_TIMEOUT", "3.0"))
        REDIS_SOCKET_TIMEOUT: float = float(os.getenv("REDIS_SOCKET_TIMEOUT", "3.0"))


settings = Settings()
