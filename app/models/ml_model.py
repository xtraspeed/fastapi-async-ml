import os
import re
import time
from typing import Any, Optional
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.config import settings
from app.core.logger import logger


class MLRiskIntelligenceModel:
    MODEL_VERSION = "v1.2.0-tfidf-logreg"

    def __init__(self, artifact_path: Optional[str] = None):
        self.artifact_path = artifact_path or settings.MODEL_ARTIFACT_PATH
        self.sentiment_pipeline: Optional[Pipeline] = None
        self.risk_pipeline: Optional[Pipeline] = None
        self.load_or_train()

    def _get_training_corpus(self):
        """Curated corpus for sentiment and risk scoring initialization."""
        texts = [
            # Positive
            ("This is an outstanding product and wonderful experience!", "POSITIVE", 0),
            ("I absolutely loved the customer support team, super helpful.", "POSITIVE", 0),
            ("Great speed, clean code, highly recommend to anyone.", "POSITIVE", 0),
            ("Brilliant solution! Fixed our production bottlenecks seamlessly.", "POSITIVE", 0),
            ("Delightful experience, very fast and dependable service.", "POSITIVE", 0),
            ("Exceeded all expectations. Very polite and efficient team.", "POSITIVE", 0),

            # Neutral
            ("The package arrived on Tuesday as per the tracking schedule.", "NEUTRAL", 0),
            ("We received the invoice and forwarded it to the finance department.", "NEUTRAL", 0),
            ("The meeting is scheduled at 3 PM in conference room B.", "NEUTRAL", 0),
            ("This documentation explains how to configure environment variables.", "NEUTRAL", 0),
            ("The server restarted at midnight during standard maintenance.", "NEUTRAL", 0),

            # Negative (Low/Medium Risk)
            ("The application crashed three times today, very frustrating bug.", "NEGATIVE", 0),
            ("Terrible delay in shipment and the packaging was damaged.", "NEGATIVE", 0),
            ("Poor communication and confusing instructions in the manual.", "NEGATIVE", 0),
            ("Disappointed with the performance under heavy traffic loads.", "NEGATIVE", 0),

            # High Risk / Toxic / Suspicious
            ("Click this urgent link immediately to claim your free reward prize!", "NEGATIVE", 1),
            ("You idiot, I will hack your database and destroy your servers.", "NEGATIVE", 1),
            ("Bank verification failed! Submit your password and credentials now.", "NEGATIVE", 1),
            ("Disgusting scam! I hate you and will personally attack your family.", "NEGATIVE", 1),
            ("Urgent wire transfer required to unblock suspended bank account.", "NEGATIVE", 1),
            ("Vile hate speech and offensive slurs targeted at community members.", "NEGATIVE", 1),
        ]
        return texts

    def load_or_train(self) -> None:
        """Loads saved model artifacts or trains and persists them if not found."""
        if os.path.exists(self.artifact_path):
            try:
                artifacts = joblib.load(self.artifact_path)
                self.sentiment_pipeline = artifacts["sentiment_pipeline"]
                self.risk_pipeline = artifacts["risk_pipeline"]
                logger.info(f"Loaded ML model artifacts from {self.artifact_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load {self.artifact_path}, retraining: {e}")

        logger.info("Initializing and training baseline ML pipelines...")
        corpus = self._get_training_corpus()
        texts = [item[0] for item in corpus]
        sentiment_labels = [item[1] for item in corpus]
        risk_labels = [item[2] for item in corpus]

        # 1. Sentiment Classifier Pipeline
        sentiment_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(random_state=42, max_iter=200)),
        ])
        sentiment_pipe.fit(texts, sentiment_labels)

        # 2. Risk / Toxicity Classifier Pipeline
        risk_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(random_state=42, max_iter=200)),
        ])
        risk_pipe.fit(texts, risk_labels)

        self.sentiment_pipeline = sentiment_pipe
        self.risk_pipeline = risk_pipe

        # Persist artifacts
        try:
            os.makedirs(os.path.dirname(self.artifact_path), exist_ok=True)
            joblib.dump(
                {"sentiment_pipeline": sentiment_pipe, "risk_pipeline": risk_pipe},
                self.artifact_path,
            )
            logger.info(f"Saved trained ML models to {self.artifact_path}")
        except Exception as e:
            logger.warning(f"Could not persist model artifacts: {e}")

    def _extract_keywords(self, text: str, top_k: int = 5) -> list[str]:
        """Extracts top significant words using TF-IDF scores."""
        words = re.findall(r"\b[A-Za-z]{3,}\b", text.lower())
        stopwords = {
            "the", "and", "is", "in", "to", "of", "for", "with", "a", "an",
            "this", "that", "it", "on", "at", "as", "by", "was", "are", "be"
        }
        filtered = [w for w in words if w not in stopwords]
        if not filtered:
            return []
        # Return unique words ordered by frequency
        from collections import Counter
        counts = Counter(filtered)
        return [word for word, _ in counts.most_common(top_k)]

    def predict(self, text: str, input_hash: str) -> dict[str, Any]:
        """Runs inference pipeline and returns structured prediction results."""
        start_time = time.perf_counter()

        # Clean text
        clean_text = text.strip()
        word_count = len(clean_text.split())
        char_count = len(clean_text)

        # 1. Sentiment Inference
        sentiment_probs = self.sentiment_pipeline.predict_proba([clean_text])[0]
        sentiment_classes = self.sentiment_pipeline.classes_
        class_prob_map = dict(zip(sentiment_classes, sentiment_probs))

        pred_sentiment = self.sentiment_pipeline.predict([clean_text])[0]
        confidence = float(class_prob_map[pred_sentiment])

        # Compute polarity score (-1.0 to +1.0)
        pos_p = float(class_prob_map.get("POSITIVE", 0.0))
        neg_p = float(class_prob_map.get("NEGATIVE", 0.0))
        polarity = round(pos_p - neg_p, 4)

        # 2. Risk Inference
        risk_probs = self.risk_pipeline.predict_proba([clean_text])[0]
        risk_score = round(float(risk_probs[1]), 4) if len(risk_probs) > 1 else 0.0

        flagged_categories = []
        lower_text = clean_text.lower()
        if any(term in lower_text for term in ["urgent", "password", "reward", "prize", "wire transfer", "bank"]):
            flagged_categories.append("phishing_or_spam_pattern")
        if any(term in lower_text for term in ["hack", "destroy", "attack", "idiot", "hate"]):
            flagged_categories.append("hostility_or_threat")

        if risk_score >= 0.70 or len(flagged_categories) >= 2:
            risk_level = "HIGH_RISK"
        elif risk_score >= 0.40 or len(flagged_categories) == 1:
            risk_level = "MEDIUM_RISK"
        elif risk_score >= 0.20:
            risk_level = "LOW_RISK"
        else:
            risk_level = "SAFE"

        # 3. Entity/Keywords Extraction
        keywords = self._extract_keywords(clean_text)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "input_hash": input_hash,
            "sentiment": {
                "label": pred_sentiment,
                "score": polarity,
                "confidence": round(confidence, 4),
            },
            "risk": {
                "risk_level": risk_level,
                "risk_score": risk_score,
                "flagged_categories": flagged_categories,
            },
            "extracted_keywords": keywords,
            "word_count": word_count,
            "char_count": char_count,
            "model_version": self.MODEL_VERSION,
            "processing_time_ms": elapsed_ms,
        }


# Global lazy singleton instance
_model_instance: Optional[MLRiskIntelligenceModel] = None


def get_ml_model() -> MLRiskIntelligenceModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = MLRiskIntelligenceModel()
    return _model_instance
