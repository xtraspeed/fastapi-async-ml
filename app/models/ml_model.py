import math
import os
import re
import time
from collections import Counter
from typing import Any, Optional
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.config import settings
from app.core.logger import logger

# Temperature applied to sentiment decision margins before the softmax. Chosen so
# the observed margin scale (~1.0 top-1/top-2 gap) maps onto a confidence range
# that stays informative instead of collapsing toward 1/3.
SENTIMENT_TEMPERATURE = 3.0

# Slope applied to the normalized risk margin. 2.0 places the training-set
# separation gap at roughly 0.12 / 0.88 on the risk score scale.
RISK_SCORE_SLOPE = 2.0

# Floor for the derived risk band width, guarding against a degenerate scale.
MIN_RISK_SCALE = 0.25

RISK_TIERS = ("SAFE", "LOW_RISK", "MEDIUM_RISK", "HIGH_RISK")

# Upper bounds (exclusive) for each tier, ordered from safest to riskiest.
RISK_TIER_THRESHOLDS = (
    (0.25, "SAFE"),
    (0.55, "LOW_RISK"),
    (0.85, "MEDIUM_RISK"),
)

RISK_KEYWORDS = {
    "phishing_or_spam_pattern": (
        "urgent", "reward", "prize", "wire transfer", "click this link",
        "claim your", "free money", "lottery", "gift card", "suspended account",
    ),
    "hostility_or_threat": (
        "hack", "destroy", "attack you", "idiot", "i hate", "hate you",
        "kill you", "worthless", "personal attack", "disgusting",
    ),
    "credential_theft": (
        "password", "credentials", "credit card", "bank account",
        "verify your", "login credentials", "card number",
    ),
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - float(np.max(scores))
    exponentials = np.exp(shifted)
    return exponentials / float(exponentials.sum())


class MLRiskIntelligenceModel:
    MODEL_VERSION = "v2.0.0-tfidf-logreg"

    def __init__(self, artifact_path: Optional[str] = None):
        self.artifact_path = artifact_path or settings.MODEL_ARTIFACT_PATH
        self.sentiment_pipeline: Optional[Pipeline] = None
        self.risk_pipeline: Optional[Pipeline] = None
        self.risk_center: float = 0.0
        self.risk_scale: float = 1.0
        self.load_or_train()

    def _get_training_corpus(self) -> list[tuple[str, str, int]]:
        """Balanced, lexically separable corpus for sentiment and risk scoring.

        Each row is ``(text, sentiment_label, risk_label)`` where ``risk_label``
        is ``1`` for toxic / phishing-style content and ``0`` for benign text.
        Class balance matters: an imbalanced corpus makes a linear classifier
        collapse onto the majority-class prior instead of learning the signal.
        """
        corpus: list[tuple[str, str, int]] = []

        positive = [
            "This is an outstanding product and a wonderful experience.",
            "I absolutely loved the customer support team, they were super helpful.",
            "Great speed, clean code, and I highly recommend it to anyone.",
            "Brilliant solution! It fixed our production bottlenecks seamlessly.",
            "A delightful experience, very fast and dependable service.",
            "Exceeded all my expectations with a very polite and efficient team.",
            "The new release resolved our pipeline bottlenecks. Super smooth experience.",
            "This product is exceptional and the customer service was smooth and responsive.",
            "Fantastic quality and the best support I have ever received.",
            "Everything worked perfectly on the first try. Highly recommended.",
            "A wonderful, polished interface that my team loves.",
            "Superb performance and incredibly reliable uptime.",
            "The onboarding was effortless and the documentation is excellent.",
            "I am so happy with this purchase. Great value for the money.",
            "Friendly, helpful, and professional from start to finish.",
            "A fantastic update that made our workflow dramatically faster.",
            "Excellent uptime and lightning fast responses all week.",
            "The team went above and beyond. Truly wonderful people.",
            "Beautifully designed, intuitive, and a pleasure to use.",
            "Great product, great price, and delightful packaging.",
            "Resolved my issue in minutes. Outstanding support.",
            "We love the new dashboard. Clean, fast, and reliable.",
            "A smooth rollout with zero downtime. Highly recommended.",
            "Impressive quality and outstanding value for money.",
            "The best decision we made this quarter. Fantastic results.",
            "A very pleasant experience from purchase to delivery.",
            "Excellent documentation and a wonderfully helpful support team.",
            "Fast, dependable, and delightfully simple to configure.",
            "Our customers consistently praise the smooth experience.",
            "A brilliant product that keeps getting better. Highly recommend.",
            "Great service, thank you!",
            "Awesome product, works perfectly.",
            "Love it, exactly what we needed.",
            "Super fast delivery, thank you.",
            "Excellent support, thanks so much.",
            "Fantastic quality, highly recommend.",
            "Wonderful experience from start to finish.",
            "Very happy with this purchase.",
        ]

        negative = [
            "The application crashed three times today. Very frustrating bug.",
            "A terrible delay in shipment and the packaging was completely damaged.",
            "Poor communication and confusing instructions in the manual.",
            "I am disappointed with the performance under heavy traffic.",
            "Awful experience. The product broke within a week.",
            "Horrible support. I have been waiting for a reply for days.",
            "The worst purchase I have made this year. Completely useless.",
            "Terrible bugs, slow performance, and an angry customer base.",
            "Frustrated and disappointed. The software is broken and unusable.",
            "A dreadful billing error that nobody has been able to fix.",
            "Slow, unreliable, and deeply frustrating to work with.",
            "The worst support experience I have ever had.",
            "Poor build quality. It failed during our first deployment.",
            "This is a garbage product and I want a refund.",
            "Disappointing release with countless regressions and failures.",
            "The interface is horribly cluttered and painfully slow.",
            "Angry and frustrated after another pointless failure.",
            "Terrible value. It broke faster than we could repair it.",
            "The service was unreliable and the outage was unacceptable.",
            "Awful quality control. The unit arrived completely broken.",
            "Poorly designed and frustratingly difficult to configure.",
            "The worst downtime we have experienced. Completely unacceptable.",
            "This release is a disaster. Many critical failures remain.",
            "Disagreeable and unhelpful staff ruined an otherwise fine day.",
            "The update is painfully slow and full of terrible regressions.",
            "A broken promise and a disappointing product experience.",
            "Frustrating bugs that crash the worker every few minutes.",
            "Horrible. I regret buying this and want my money back.",
            "The worst reliability of any tool we have evaluated.",
            "Dreadful error messages and a confusing broken workflow.",
            "Terrible, a complete waste of money.",
            "Awful support, nobody replied.",
            "Broken on arrival, very disappointing.",
            "Very frustrating and poorly built.",
            "Horrible experience, do not buy.",
            "Refund requested, the item is broken.",
        ]

        neutral = [
            "The package arrived on Tuesday as per the tracking schedule.",
            "We received the invoice and forwarded it to the finance department.",
            "The meeting is scheduled at 3 PM in conference room B.",
            "This documentation explains how to configure environment variables.",
            "The server restarted at midnight during standard maintenance.",
            "The quarterly report is attached for your review.",
            "Please find the updated deployment schedule for the next sprint.",
            "The specification was revised by the architecture group.",
            "Our office moves to the third floor on Monday.",
            "The build pipeline finished at 04:12 UTC.",
            "Version 2.4 introduces a new configuration option.",
            "The shipment is scheduled to arrive on the eighteenth.",
            "I have forwarded the request to the accounts department.",
            "The dashboard now displays the quarterly figures.",
            "Please submit the expense form before the deadline.",
            "The API reference lists each available endpoint.",
            "Our team reviewed the draft and added comments.",
            "The database migration completed without incident.",
            "The office will be closed on the public holiday.",
            "This release adds support for the newer protocol.",
            "The training session is available on the internal portal.",
            "Please review the attached architecture diagram.",
            "The data warehouse refresh runs every night at two.",
            "We scheduled a follow up call for next Wednesday.",
            "The changelog for the previous release is attached.",
            "The compliance audit is scheduled for the following month.",
            "Support hours are nine to five on business days.",
            "The repository has been updated with the new module.",
            "Kindly confirm your attendance for the annual review.",
            "The certificate expires at the end of the fiscal year.",
            "Meeting at 3 PM in room B.",
            "Invoice attached for your records.",
            "Please review the attached report.",
            "Deployment is scheduled for Monday.",
            "Acknowledged, thanks.",
        ]

        risky = [
            "Click this urgent link immediately to claim your free reward prize!",
            "You idiot, I will hack your database and destroy your servers.",
            "Bank verification failed! Submit your password and credentials now.",
            "Disgusting scam! I hate you and will personally attack your family.",
            "Urgent wire transfer required to unblock your suspended bank account.",
            "Vile hate speech and offensive slurs targeted at our community members.",
            "Verify your account password at this link or lose access today.",
            "You are an idiot and I will destroy your entire career.",
            "Claim your cash reward now, just confirm your credit card details.",
            "Send the wire transfer immediately or your account will be suspended.",
            "I hate all of you and I will personally attack your offices tomorrow.",
            "Free prize winner! Enter your bank credentials to claim the reward.",
            "Urgent notice: your password expires, click to verify your login.",
            "I will hack your network and destroy every file you own.",
            "Scam artists like you deserve to be hacked and robbed.",
            "Submit your credit card number to unlock the free reward.",
            "Your account is suspended. Urgent bank verification required.",
            "Offensive slur filled hate message directed at a colleague.",
            "Click here to claim the lottery prize before it expires tonight.",
            "Transfer the funds urgently or legal action will be taken.",
            "Stupid idiot, I will personally attack you at your house.",
            "Confirm your password and login credentials by email today.",
            "Exclusive reward: wire transfer required to release the funds.",
            "Malware warning, download and run this exploit tool immediately.",
            "Hateful abusive content attacking a protected group.",
            "Claim your bonus reward by entering your bank account number.",
            "I will destroy your servers and hack your databases tonight.",
            "Urgent scam alert: verify your credentials or lose everything.",
            "Send money now, this is an emergency wire transfer request.",
            "You are worthless, I hate you and I will attack you.",
            "Free money prize, click the link and enter your password.",
            "Suspended account, urgent bank verification to restore access.",
            "Threatening message promising to hack and destroy the company.",
            "Enter your login credentials on this page to avoid suspension.",
            "Congratulations, claim your reward and free gift card now.",
            "Abusive rant attacking and threatening a coworker by name.",
            "Send your password now!",
            "Claim your free prize!",
            "I will hack you!",
            "Urgent: verify your bank account now.",
            "You are an idiot and I hate you.",
        ]

        safe = [
            "The package arrived on Tuesday as per the tracking schedule.",
            "We received the invoice and forwarded it to the finance department.",
            "The meeting is scheduled at 3 PM in conference room B.",
            "This documentation explains how to configure environment variables.",
            "The server restarted at midnight during standard maintenance.",
            "The quarterly report is attached for your review.",
            "Please find the updated deployment schedule for the next sprint.",
            "The specification was revised by the architecture group.",
            "The build pipeline finished at 04:12 UTC.",
            "Version 2.4 introduces a new configuration option.",
            "The shipment is scheduled to arrive on the eighteenth.",
            "The dashboard now displays the quarterly figures.",
            "The API reference lists each available endpoint.",
            "The database migration completed without incident.",
            "This release adds support for the newer protocol.",
            "The training session is available on the internal portal.",
            "Please review the attached architecture diagram.",
            "The data warehouse refresh runs every night at two.",
            "The changelog for the previous release is attached.",
            "The compliance audit is scheduled for the following month.",
            "Support hours are nine to five on business days.",
            "The repository has been updated with the new module.",
            "Kindly confirm your attendance for the annual review.",
            "The certificate expires at the end of the fiscal year.",
            "The new release resolved our pipeline bottlenecks smoothly.",
            "This product is exceptional and the customer service was responsive.",
            "The deployment finished without any errors or incidents.",
            "Our team published the sprint retrospective summary.",
            "The quarterly numbers have been added to the report.",
            "Please update your contact details in the internal portal.",
            "The library was upgraded to the latest stable version.",
            "The onboarding checklist is available on the wiki.",
            "Our office hours remain unchanged during the holiday.",
            "The feature flag was enabled for ten percent of traffic.",
            "A new monitoring dashboard was added for the API service.",
            "The design document covers the rollout and rollback plan.",
            "Thanks for the update.",
            "Report attached for review.",
            "Build passed successfully.",
            "Meeting notes are in the wiki.",
            "Acknowledged, will review.",
        ]

        for text in positive:
            corpus.append((text, "POSITIVE", 0))
        for text in negative:
            corpus.append((text, "NEGATIVE", 0))
        for text in neutral:
            corpus.append((text, "NEUTRAL", 0))
        for text in risky:
            corpus.append((text, "NEGATIVE", 1))
        for text in safe:
            corpus.append((text, "NEUTRAL", 0))

        return corpus

    def load_or_train(self) -> None:
        """Loads saved model artifacts or trains and persists them if not found."""
        if os.path.exists(self.artifact_path):
            try:
                artifacts = joblib.load(self.artifact_path)
                if artifacts.get("model_version") != self.MODEL_VERSION:
                    logger.info(
                        f"Model artifact version mismatch "
                        f"({artifacts.get('model_version')} != {self.MODEL_VERSION}), retraining"
                    )
                else:
                    self.sentiment_pipeline = artifacts["sentiment_pipeline"]
                    self.risk_pipeline = artifacts["risk_pipeline"]
                    self.risk_center = float(artifacts.get("risk_center", 0.0))
                    self.risk_scale = float(artifacts.get("risk_scale", 1.0))
                    logger.info(
                        f"Loaded ML model artifacts from {self.artifact_path} "
                        f"(risk_center={self.risk_center:.3f}, risk_scale={self.risk_scale:.3f})"
                    )
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
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")),
        ])
        sentiment_pipe.fit(texts, sentiment_labels)

        # 2. Risk / Toxicity Classifier Pipeline
        risk_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")),
        ])
        risk_pipe.fit(texts, risk_labels)

        self.sentiment_pipeline = sentiment_pipe
        self.risk_pipeline = risk_pipe

        # 3. Derive the risk decision band from the training separation gap.
        #    L2-regularised LogisticRegression squashes predict_proba() toward the
        #    class prior, so absolute probability thresholds are meaningless. The
        #    signed decision margin is used instead, and the empty band between the
        #    two classes is located empirically to place the tier boundaries.
        margins = risk_pipe.decision_function(texts)
        labels_arr = np.asarray(risk_labels)
        benign_max = float(margins[labels_arr == 0].max())
        risky_min = float(margins[labels_arr == 1].min())

        if benign_max < risky_min:
            self.risk_center = (benign_max + risky_min) / 2.0
            self.risk_scale = max((risky_min - benign_max) / 2.0, MIN_RISK_SCALE)
            logger.info(
                f"Risk band: benign<={benign_max:.3f} risky>={risky_min:.3f} "
                f"-> center={self.risk_center:.3f} scale={self.risk_scale:.3f}"
            )
        else:
            # Classes overlap on the training set; fall back to the raw boundary.
            self.risk_center = 0.0
            self.risk_scale = 1.0
            logger.warning(
                "Risk classes are not linearly separable; falling back to the raw decision boundary"
            )

        # Persist artifacts
        try:
            parent = os.path.dirname(self.artifact_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            joblib.dump(
                {
                    "model_version": self.MODEL_VERSION,
                    "sentiment_pipeline": sentiment_pipe,
                    "risk_pipeline": risk_pipe,
                    "risk_center": self.risk_center,
                    "risk_scale": self.risk_scale,
                },
                self.artifact_path,
            )
            logger.info(f"Saved trained ML models to {self.artifact_path}")
        except Exception as e:
            logger.warning(f"Could not persist model artifacts: {e}")

    def _extract_keywords(self, text: str, top_k: int = 5) -> list[str]:
        """Extracts top significant words using frequency ranking."""
        words = re.findall(r"\b[A-Za-z]{3,}\b", text.lower())
        stopwords = {
            "the", "and", "is", "in", "to", "of", "for", "with", "a", "an",
            "this", "that", "it", "on", "at", "as", "by", "was", "are", "be"
        }
        filtered = [w for w in words if w not in stopwords]
        if not filtered:
            return []
        counts = Counter(filtered)
        return [word for word, _ in counts.most_common(top_k)]

    def predict(self, text: str, input_hash: str) -> dict[str, Any]:
        """Runs inference pipeline and returns structured prediction results."""
        start_time = time.perf_counter()

        # Clean text
        clean_text = text.strip()
        word_count = len(clean_text.split())
        char_count = len(clean_text)

        # 1. Sentiment Inference (single decision_function call feeds label, score and confidence)
        sentiment_margins = self.sentiment_pipeline.decision_function([clean_text])[0]
        classes = list(self.sentiment_pipeline.classes_)
        probs = _softmax(sentiment_margins * SENTIMENT_TEMPERATURE)
        class_prob_map = dict(zip(classes, probs))

        pred_index = int(np.argmax(sentiment_margins))
        pred_sentiment = classes[pred_index]
        confidence = float(class_prob_map[pred_sentiment])

        # Compute polarity score (-1.0 to +1.0)
        pos_p = float(class_prob_map.get("POSITIVE", 0.0))
        neg_p = float(class_prob_map.get("NEGATIVE", 0.0))
        polarity = round(pos_p - neg_p, 4)

        # 2. Risk Inference (centered decision margin -> calibrated score)
        margin = float(self.risk_pipeline.decision_function([clean_text])[0])
        normalized = (margin - self.risk_center) / self.risk_scale
        risk_score = round(_sigmoid(RISK_SCORE_SLOPE * normalized), 4)

        flagged_categories = self._flag_categories(clean_text)
        risk_level = self._risk_level(risk_score, flagged_categories)

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

    def _flag_categories(self, text: str) -> list[str]:
        """Deterministic keyword safety net layered on top of the statistical model."""
        lower_text = text.lower()
        flagged: list[str] = []
        if any(term in lower_text for term in RISK_KEYWORDS["phishing_or_spam_pattern"]):
            flagged.append("phishing_or_spam_pattern")
        if any(term in lower_text for term in RISK_KEYWORDS["hostility_or_threat"]):
            flagged.append("hostility_or_threat")
        if any(term in lower_text for term in RISK_KEYWORDS["credential_theft"]):
            flagged.append("credential_theft")
        return flagged

    def _risk_level(self, risk_score: float, flagged_categories: list[str]) -> str:
        """Maps a calibrated risk score to a tier, escalating on keyword evidence.

        Keyword hits act as a floor: each flagged category escalates the tier by
        one step, so deterministic signals can never be silently ignored.
        """
        tier = RISK_TIERS[-1]
        for threshold, name in RISK_TIER_THRESHOLDS:
            if risk_score < threshold:
                tier = name
                break

        index = RISK_TIERS.index(tier)
        index = min(index + len(flagged_categories), len(RISK_TIERS) - 1)
        return RISK_TIERS[index]


# Global lazy singleton instance
_model_instance: Optional[MLRiskIntelligenceModel] = None


def get_ml_model() -> MLRiskIntelligenceModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = MLRiskIntelligenceModel()
    return _model_instance
