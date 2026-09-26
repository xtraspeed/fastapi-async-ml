from typing import Any, Optional, Union
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=2,
        max_length=10000,
        description="Text content to analyze for sentiment, risk, and entities.",
        examples=["This product is exceptional! The customer service was smooth and responsive."],
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional client metadata (e.g., user_id, source, document_id).",
    )

    @field_validator("text")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) < 2:
            raise ValueError("text must contain at least 2 non-whitespace characters")
        return stripped


class BatchPredictionRequest(BaseModel):
    items: list[PredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Batch of text items to process concurrently.",
    )


class SentimentOutput(BaseModel):
    label: str = Field(..., description="Sentiment classification: POSITIVE, NEGATIVE, or NEUTRAL.")
    score: float = Field(..., description="Polarity score ranging from -1.0 to 1.0.")
    confidence: float = Field(..., description="Model confidence score between 0.0 and 1.0.")


class RiskAnalysisOutput(BaseModel):
    risk_level: str = Field(..., description="Risk tier: SAFE, LOW_RISK, MEDIUM_RISK, or HIGH_RISK.")
    risk_score: float = Field(..., description="Risk probability score between 0.0 and 1.0.")
    flagged_categories: list[str] = Field(
        default_factory=list,
        description="Identified risk categories (e.g., toxicity, spam, aggressive language).",
    )


class PredictionResult(BaseModel):
    input_hash: str
    sentiment: SentimentOutput
    risk: RiskAnalysisOutput
    extracted_keywords: list[str] = Field(default_factory=list)
    word_count: int
    char_count: int
    model_version: str
    processing_time_ms: float


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str
    check_status_url: str
    cached: bool = False
    result: Optional[PredictionResult] = None


class BatchResult(BaseModel):
    batch_id: str
    total_processed: int = Field(..., ge=0)
    cached_items: int = Field(default=0, ge=0, description="Items served from the Redis idempotency cache.")
    results: list[PredictionResult] = Field(default_factory=list)


class BatchTaskEntry(BaseModel):
    task_id: str
    status: str
    check_status_url: str


class BatchTaskResponse(BaseModel):
    batch_id: str
    total_items: int
    tasks: list[BatchTaskEntry]


TaskResult = Union[PredictionResult, BatchResult]


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int = Field(default=0, ge=0, le=100)
    stage: Optional[str] = None
    result: Optional[TaskResult] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    services: dict[str, str]
    timestamp: str
