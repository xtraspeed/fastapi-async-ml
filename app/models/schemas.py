from typing import Any, Optional
from pydantic import BaseModel, Field


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


class BatchTaskResponse(BaseModel):
    batch_id: str
    total_items: int
    tasks: list[dict[str, Any]]


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int = Field(default=0, ge=0, le=100)
    stage: Optional[str] = None
    result: Optional[PredictionResult] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    services: dict[str, str]
    timestamp: str
