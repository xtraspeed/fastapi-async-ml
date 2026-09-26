import hashlib
import pytest
from app.core.cache import compute_cache_key
from app.models.ml_model import MLRiskIntelligenceModel, get_ml_model


def test_model_initialization_and_prediction():
    model = get_ml_model()
    assert model is not None
    assert model.sentiment_pipeline is not None
    assert model.risk_pipeline is not None

    # Positive sample test
    pos_text = "The new user interface is remarkably clean and intuitive!"
    pos_hash = hashlib.sha256(pos_text.lower().encode("utf-8")).hexdigest()
    res = model.predict(pos_text, pos_hash)

    assert res["input_hash"] == pos_hash
    assert res["sentiment"]["label"] in ["POSITIVE", "NEUTRAL", "NEGATIVE"]
    assert -1.0 <= res["sentiment"]["score"] <= 1.0
    assert 0.0 <= res["sentiment"]["confidence"] <= 1.0
    assert res["risk"]["risk_level"] in ["SAFE", "LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"]
    assert res["word_count"] > 0
    assert res["processing_time_ms"] > 0


def test_high_risk_flagging():
    model = get_ml_model()
    threat_text = "Urgent wire transfer required! Click to claim your prize or we will hack you."
    threat_hash = hashlib.sha256(threat_text.lower().encode("utf-8")).hexdigest()
    res = model.predict(threat_text, threat_hash)

    assert res["risk"]["risk_level"] in ["MEDIUM_RISK", "HIGH_RISK"]
    assert len(res["risk"]["flagged_categories"]) > 0


def test_cache_key_generation():
    key1 = compute_cache_key("  Hello World  ")
    key2 = compute_cache_key("hello world")
    assert key1 == key2
    assert key1.startswith("cache:prediction:")
