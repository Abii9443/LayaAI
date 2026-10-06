"""
Answer Relevancy Metric

Score-based metric (0-100) that evaluates how relevant the bot's response
is to the user's query. Uses Laya's 'score' decision for ordinal rating.
"""

from typing import Dict, Any
import logging

from laya_eval.core.metrics import (
    BaseMetric,
    LayaDecision,
    DecisionType,
    MetricResult
)

logger = logging.getLogger(__name__)


class RelevancyMetric(BaseMetric):
    """
    Evaluates how relevant the bot response is to the user query

    Decision Type: SCORE (0-100 scale)
    Output: Relevancy score and normalized confidence

    Evaluation criteria:
    - 90-100: Perfectly addresses the query with complete, relevant information
    - 70-89: Mostly relevant with minor tangential information
    - 50-69: Partially relevant but missing key points or has irrelevant content
    - 30-49: Somewhat related but largely off-topic
    - 0-29: Not relevant or completely off-topic
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_metric_name(self) -> str:
        return "answer_relevancy"

    def prepare_decision(self, row: Dict[str, Any]) -> LayaDecision:
        """
        Formulate relevancy evaluation as a scoring question

        Args:
            row: Data row with 'user_query' and 'actual_response'

        Returns:
            LayaDecision with score-type question (0-100)
        """
        user_query = row.get('user_query', '')
        actual = row.get('actual_response', '')
        test_id = row.get('test_id', row.get('id', 'unknown'))

        question = f"""Rate the relevancy of the bot's response to the user's query on a scale from 0 to 100.

User Query: "{user_query}"

Bot Response: "{actual}"

Evaluation criteria:
- 90-100: Perfectly addresses the query with complete, relevant information
- 70-89: Mostly relevant with minor tangential information
- 50-69: Partially relevant but missing key points or has irrelevant content
- 30-49: Somewhat related but largely off-topic
- 0-29: Not relevant or completely off-topic

What is the relevancy score (0-100)?"""

        return LayaDecision(
            question=question,
            decision_type=DecisionType.SCORE,
            decision_id=f"relevancy_{test_id}",
            scale_min=0,
            scale_max=100,
            metadata={'metric': 'relevancy', 'test_id': test_id}
        )

    def process_result(self, laya_result: Any, row: Dict[str, Any]) -> MetricResult:
        """Convert Laya score result into relevancy metric result"""
        score = laya_result.score
        confidence = laya_result.confidence

        # Normalize to 0-1
        normalized_score = score / 100.0

        # Generate reasoning
        if score >= 90:
            level = "excellent"
            reasoning = "The response perfectly addresses the user's query with complete relevant information."
        elif score >= 70:
            level = "good"
            reasoning = "The response is mostly relevant with only minor tangential information."
        elif score >= 50:
            level = "fair"
            reasoning = "The response is partially relevant but missing key points or contains irrelevant content."
        elif score >= 30:
            level = "poor"
            reasoning = "The response is somewhat related but largely off-topic."
        else:
            level = "very poor"
            reasoning = "The response is not relevant or completely off-topic."

        return MetricResult(
            metric_name=self.get_metric_name(),
            value=score,
            score=normalized_score,
            confidence=confidence,
            raw_value=laya_result,
            reasoning=f"Relevancy score: {score}/100 ({level}). {reasoning}",
            metadata={'relevancy_level': level, 'test_id': row.get('test_id', row.get('id'))}
        )


# Register the metric
from laya_eval.core.metrics import MetricRegistry
MetricRegistry.register(RelevancyMetric)
