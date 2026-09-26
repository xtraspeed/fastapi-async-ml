import hashlib
from collections import Counter
from unittest.mock import patch
import pytest
from celery.exceptions import Retry

from app.core.cache import compute_cache_key
from app.models.ml_model import MLRiskIntelligenceModel, get_ml_model
from app.workers.tasks import (
    _exponential_backoff,
    analyze_text_task,
    batch_analyze_text_task,
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.lower().encode("utf-8")).hexdigest()


def test_model_initialization_and_prediction():
    model = get_ml_model()
    assert model is not None
    assert model.sentiment_pipeline is not None
    assert model.risk_pipeline is not None

    # Positive sample test
    pos_text = "The new user interface is remarkably clean and intuitive!"
    pos_hash = _hash(pos_text)
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
    res = model.predict(threat_text, _hash(threat_text))

    assert res["risk"]["risk_level"] in ["MEDIUM_RISK", "HIGH_RISK"]
    assert len(res["risk"]["flagged_categories"]) > 0


def test_safe_text_is_not_flagged():
    model = get_ml_model()
    safe_text = "The meeting is scheduled at 3 PM in conference room B."
    res = model.predict(safe_text, _hash(safe_text))

    assert res["risk"]["flagged_categories"] == []
    assert res["risk"]["risk_level"] == "SAFE"
    assert res["risk"]["risk_score"] < 0.25


def test_model_is_a_singleton():
    assert get_ml_model() is get_ml_model()


def test_corpus_is_class_balanced():
    """An imbalanced corpus collapses a linear classifier onto the majority prior."""
    corpus = get_ml_model()._get_training_corpus()
    counts = Counter(label for _, label, _ in corpus)
    risk_counts = Counter(risk for _, _, risk in corpus)

    assert len(corpus) >= 100
    assert min(counts.values()) > 0
    # No class may dominate the sentiment label distribution.
    assert max(counts.values()) / len(corpus) < 0.55
    assert risk_counts[1] >= 20


@pytest.mark.parametrize(
    "text,expected_label",
    [
        ("The new release resolved our pipeline bottlenecks! Super smooth experience.", "POSITIVE"),
        ("This product is exceptional! The customer service was smooth and responsive.", "POSITIVE"),
        ("The meeting is scheduled at 3 PM in conference room B.", "NEUTRAL"),
        ("The application crashed three times today, very frustrating bug.", "NEGATIVE"),
    ],
)
def test_sentiment_labels_match_readme_examples(text, expected_label):
    """Locks in the sentiment output documented in the README."""
    model = get_ml_model()
    assert model.predict(text, _hash(text))["sentiment"]["label"] == expected_label


@pytest.mark.parametrize(
    "text,expected_tier",
    [
        ("The new release resolved our pipeline bottlenecks! Super smooth experience.", "SAFE"),
        ("This product is exceptional! The customer service was smooth and responsive.", "SAFE"),
        ("URGENT: Click here to verify your bank password immediately!", "HIGH_RISK"),
        ("Urgent wire transfer required! Click to claim your prize or we will hack you.", "HIGH_RISK"),
    ],
)
def test_risk_tiers_match_readme_examples(text, expected_tier):
    """Locks in the risk output documented in the README."""
    model = get_ml_model()
    res = model.predict(text, _hash(text))
    assert res["risk"]["risk_level"] == expected_tier


@pytest.mark.parametrize(
    "text,expected_label,expected_tier",
    [
        ("This is fantastic, we love the new dashboard.", "POSITIVE", "SAFE"),
        ("Superb quality and quick delivery, thank you!", "POSITIVE", "SAFE"),
        ("This is broken and the quality is awful.", "NEGATIVE", "SAFE"),
        ("Terrible support and a very slow response.", "NEGATIVE", "SAFE"),
        ("The standup was moved to 10am tomorrow.", "NEUTRAL", "SAFE"),
        ("The release notes are in the wiki.", "NEUTRAL", "SAFE"),
        ("Send me your password or I will destroy your site.", "NEGATIVE", "HIGH_RISK"),
        ("I hate you, you idiot, I will attack you.", "NEGATIVE", "HIGH_RISK"),
        ("Urgent! Claim your free prize money now.", "NEGATIVE", "HIGH_RISK"),
        ("Verify your bank credentials at this link immediately.", "NEGATIVE", "HIGH_RISK"),
    ],
)
def test_model_generalizes_to_unseen_text(text, expected_label, expected_tier):
    """Held-out phrasings must classify correctly, proving the model learned
    signal rather than memorising the training corpus."""
    model = get_ml_model()
    trained = {t.lower() for t, _, _ in model._get_training_corpus()}
    assert text.lower() not in trained, "held-out sample leaked into the training corpus"

    res = model.predict(text, _hash(text))
    assert res["sentiment"]["label"] == expected_label
    assert res["risk"]["risk_level"] == expected_tier


def test_risk_score_is_bounded_and_ordered():
    """Benign text must score far below risky text for the tiers to be meaningful."""
    model = get_ml_model()
    safe = model.predict("Please review the attached report.", "a")["risk"]["risk_score"]
    risky = model.predict("Claim your free prize, click now!", "b")["risk"]["risk_score"]

    for score in (safe, risky):
        assert 0.0 <= score <= 1.0
    assert safe < 0.25
    assert risky > 0.85
    assert risky - safe > 0.5


def test_keyword_evidence_escalates_tier():
    """Deterministic keyword hits act as a floor, never ignored."""
    model = get_ml_model()
    # Unambiguously benign to the classifier, but contains a hard phishing term.
    res = model.predict("Please forward my bank account details today.", "h")
    assert "credential_theft" in res["risk"]["flagged_categories"]
    assert res["risk"]["risk_level"] != "SAFE"


def test_artifact_roundtrip_preserves_calibration(tmp_path):
    """A saved artifact must restore the derived risk band, not reset it."""
    from app.models.ml_model import MLRiskIntelligenceModel

    artifact = tmp_path / "model.joblib"
    trained = MLRiskIntelligenceModel(artifact_path=str(artifact))
    assert artifact.exists()

    reloaded = MLRiskIntelligenceModel(artifact_path=str(artifact))
    assert reloaded.risk_center == trained.risk_center
    assert reloaded.risk_scale == trained.risk_scale
    assert reloaded.MODEL_VERSION == trained.MODEL_VERSION

    probe = "Urgent! Claim your free prize money now."
    before = trained.predict(probe, "h")
    after = reloaded.predict(probe, "h")
    assert {k: v for k, v in before.items() if k != "processing_time_ms"} == {
        k: v for k, v in after.items() if k != "processing_time_ms"
    }


def test_cache_key_generation():
    key1 = compute_cache_key("  Hello World  ")
    key2 = compute_cache_key("hello world")
    assert key1 == key2
    assert key1.startswith("cache:prediction:")


def test_cache_key_differs_for_different_text():
    assert compute_cache_key("alpha") != compute_cache_key("beta")


def test_exponential_backoff_doubles():
    assert [_exponential_backoff(i) for i in range(4)] == [1, 2, 4, 8]


def test_analyze_task_runs_pipeline_and_caches(in_memory_cache):
    text = "Absolutely brilliant, the support team fixed everything instantly."
    input_hash = _hash(text)

    with in_memory_cache.patch() as cache:
        result = analyze_text_task.apply(args=[text, input_hash]).get()

    assert result["input_hash"] == input_hash
    assert result["sentiment"]["label"] in ("POSITIVE", "NEUTRAL", "NEGATIVE")
    assert 0.0 <= result["risk"]["risk_score"] <= 1.0
    assert compute_cache_key(text) in cache.data


def test_analyze_task_is_idempotent_on_redelivery(in_memory_cache):
    """A redelivered message must not re-run inference when the result is already cached."""
    text = "Duplicate delivery of the exact same payload."
    input_hash = _hash(text)
    sentinel = {**get_ml_model().predict(text, input_hash), "sentinel": "from-cache"}

    with in_memory_cache.patch() as cache:
        cache.data[compute_cache_key(text)] = sentinel

        with patch("app.workers.tasks.get_ml_model") as mock_model:
            result = analyze_text_task.apply(args=[text, input_hash]).get()

    assert result == sentinel
    mock_model.assert_not_called()


def test_analyze_task_reports_monotonic_progress(in_memory_cache):
    text = "Progress reporting should advance through the pipeline stages."
    reported: list[int] = []

    with in_memory_cache.patch():
        with patch.object(analyze_text_task, "update_state", side_effect=lambda **kw: reported.append(kw["meta"]["progress"])):
            analyze_text_task.apply(args=[text, _hash(text)]).get()

    assert reported == sorted(reported)
    assert reported[0] == 20
    assert reported[-1] == 100
    assert reported[-2] == 85


def test_analyze_task_requests_retry_below_ceiling(fake_task_self):
    text = "Payload that will always blow up."
    stub = fake_task_self(retries=0, max_retries=analyze_text_task.max_retries)

    with patch("app.workers.tasks.get_cached_prediction", return_value=None):
        with patch("app.workers.tasks.get_ml_model") as mock_model:
            mock_model.return_value.predict.side_effect = RuntimeError("inference blew up")

            with pytest.raises(Retry) as exc_info:
                analyze_text_task.__wrapped__.__func__(stub, text, _hash(text))

    stub.retry.assert_called_once()
    assert stub.retry.call_args.kwargs["countdown"] == 1
    assert "inference blew up" in str(stub.retry.call_args.kwargs["exc"])


def test_analyze_task_backoff_grows_with_retry_count(fake_task_self):
    text = "Still blowing up."
    stub = fake_task_self(retries=2, max_retries=analyze_text_task.max_retries)

    with patch("app.workers.tasks.get_cached_prediction", return_value=None):
        with patch("app.workers.tasks.get_ml_model") as mock_model:
            mock_model.return_value.predict.side_effect = RuntimeError("inference blew up")
            with pytest.raises(Retry):
                analyze_text_task.__wrapped__.__func__(stub, text, _hash(text))

    stub.retry.assert_called_once()
    assert stub.retry.call_args.kwargs["countdown"] == 4


def test_analyze_task_fails_permanently_at_retry_ceiling(fake_task_self):
    text = "Payload that will always blow up."
    stub = fake_task_self(retries=3, max_retries=analyze_text_task.max_retries)

    with patch("app.workers.tasks.get_cached_prediction", return_value=None):
        with patch("app.workers.tasks.get_ml_model") as mock_model:
            mock_model.return_value.predict.side_effect = RuntimeError("inference blew up")

            with pytest.raises(RuntimeError, match="inference blew up"):
                analyze_text_task.__wrapped__.__func__(stub, text, _hash(text))

    stub.retry.assert_not_called()


def test_batch_task_processes_all_items(in_memory_cache):
    texts = ["Fantastic service, truly delightful.", "Awful. A complete waste of money."]
    items = [{"text": t, "input_hash": _hash(t)} for t in texts]

    with in_memory_cache.patch() as cache:
        result = batch_analyze_text_task.apply(args=[items, "batch-1"]).get()

    assert result["batch_id"] == "batch-1"
    assert result["total_processed"] == 2
    assert result["cached_items"] == 0
    assert [r["input_hash"] for r in result["results"]] == [i["input_hash"] for i in items]
    assert all(compute_cache_key(t) in cache.data for t in texts)


def test_batch_task_reuses_cached_items(in_memory_cache):
    texts = ["already processed one", "brand new item"]
    items = [{"text": t, "input_hash": _hash(t)} for t in texts]

    with in_memory_cache.patch() as cache:
        cache.data[compute_cache_key(texts[0])] = {**items[0], "sentinel": "cached"}

        with patch("app.workers.tasks.get_ml_model") as mock_model:
            mock_model.return_value.predict.return_value = {**items[1], "sentinel": "fresh"}
            result = batch_analyze_text_task.apply(args=[items, "batch-2"]).get()

    assert result["cached_items"] == 1
    assert result["results"][0]["sentinel"] == "cached"
    assert result["results"][1]["sentinel"] == "fresh"
    # The cached item must not have been re-inferred.
    assert mock_model.return_value.predict.call_count == 1


def test_batch_task_reports_progress_to_100(in_memory_cache):
    items = [{"text": f"item number {i}", "input_hash": _hash(f"item number {i}")} for i in range(3)]
    reported: list[int] = []

    with in_memory_cache.patch():
        with patch.object(
            batch_analyze_text_task, "update_state", side_effect=lambda **kw: reported.append(kw["meta"]["progress"])
        ):
            batch_analyze_text_task.apply(args=[items, "batch-3"]).get()

    assert reported == sorted(reported)
    assert reported[-1] == 100


def test_batch_task_requests_retry_below_ceiling(fake_task_self):
    items = [{"text": "explode", "input_hash": "h1"}]
    stub = fake_task_self(retries=0, max_retries=batch_analyze_text_task.max_retries)

    with patch("app.workers.tasks.get_cached_prediction", return_value=None):
        with patch("app.workers.tasks.get_ml_model") as mock_model:
            mock_model.return_value.predict.side_effect = RuntimeError("batch blew up")

            with pytest.raises(Retry):
                batch_analyze_text_task.__wrapped__.__func__(stub, items, "batch-4")

    stub.retry.assert_called_once()
    assert stub.retry.call_args.kwargs["countdown"] == 1


def test_batch_task_fails_permanently_at_retry_ceiling(fake_task_self):
    items = [{"text": "explode", "input_hash": "h1"}]
    stub = fake_task_self(retries=3, max_retries=batch_analyze_text_task.max_retries)

    with patch("app.workers.tasks.get_cached_prediction", return_value=None):
        with patch("app.workers.tasks.get_ml_model") as mock_model:
            mock_model.return_value.predict.side_effect = RuntimeError("batch blew up")

            with pytest.raises(RuntimeError, match="batch blew up"):
                batch_analyze_text_task.__wrapped__.__func__(stub, items, "batch-5")

    stub.retry.assert_not_called()
