"""
Laya AI Chatbot Evaluation Framework

A high-performance evaluation framework for chatbot responses using Laya AI decision models.
Supports multiple evaluation metrics with >80% accuracy target.
"""

__version__ = "0.1.0"

# Core exports
from laya_eval.core.evaluator import ChatbotEvaluator, quick_evaluate
from laya_eval.core.metrics import BaseMetric, MetricResult, MetricRegistry
from laya_eval.preprocessing.data_loader import DataLoader, load_evaluation_data
from laya_eval.models.laya_client import LayaClient, create_client

# Metrics
from laya_eval.metrics.response_match import ResponseMatchMetric
from laya_eval.metrics.relevancy import RelevancyMetric
from laya_eval.metrics.faithfulness import FaithfulnessMetric
from laya_eval.metrics.hallucination import HallucinationMetric
from laya_eval.metrics.contextual_recall import ContextualRecallMetric
from laya_eval.metrics.toxicity import ToxicityMetric

__all__ = [
    "__version__",
    # Core classes
    "ChatbotEvaluator",
    "quick_evaluate",
    "BaseMetric",
    "MetricResult",
    "MetricRegistry",
    "DataLoader",
    "load_evaluation_data",
    "LayaClient",
    "create_client",
    # Metrics
    "ResponseMatchMetric",
    "RelevancyMetric",
    "FaithfulnessMetric",
    "HallucinationMetric",
    "ContextualRecallMetric",
    "ToxicityMetric",
]
