from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import joblib

from dataset import TRAINING_DATA

MODEL_VERSION = "v1"
MODEL_PATH = Path(__file__).resolve().parent / "intent_model.pkl"

_pipeline: Optional[object] = None


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation noise, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"['\"]", "", text)
    text = re.sub(r"[^a-z0-9\s\-\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_pipeline() -> object:
    """Train TF-IDF + Logistic Regression on the embedded dataset."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline as SkPipeline
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for ML intent classification.\n"
            "Install it with:  pip install scikit-learn"
        ) from exc

    texts = [_normalise(t) for t, _ in TRAINING_DATA]
    labels = [label for _, label in TRAINING_DATA]

    pipeline = SkPipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 3),
                    min_df=1,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    C=4.0,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(texts, labels)
    joblib.dump((MODEL_VERSION, pipeline), MODEL_PATH)
    return pipeline


def _get_or_build_pipeline() -> object:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    if MODEL_PATH.exists():
        try:
            version, loaded = joblib.load(MODEL_PATH)
            if version == MODEL_VERSION:
                _pipeline = loaded
                return _pipeline
        except Exception:
            pass

    _pipeline = _build_pipeline()
    return _pipeline


def _warm_up() -> None:
    _get_or_build_pipeline()


def predict_intent(text: str) -> tuple[str, float]:
    """
    ML-only prediction helper.
    Returns raw best intent + confidence.
    """
    source = (text or "").strip()
    if not source:
        return "unknown", 0.0

    pipeline = _get_or_build_pipeline()
    normalised = _normalise(source)
    proba_array = pipeline.predict_proba([normalised])[0]
    best_idx = int(proba_array.argmax())
    confidence = float(proba_array[best_idx])
    ml_intent = pipeline.classes_[best_idx]
    return ml_intent, round(confidence, 4)


_warm_up()

