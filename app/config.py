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


settings = Settings()
