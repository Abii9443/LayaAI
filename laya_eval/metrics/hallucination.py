"""
Hallucination Detection Metric

Score-based metric (0-100) that evaluates the degree of hallucination
(fabricated content) in the bot's response. Uses Laya's 'score' decision.
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


class HallucinationMetric(BaseMetric):
    """
    Evaluates the degree of hallucination in the bot response

    Decision Type: SCORE (0-100 scale)
    Output: Hallucination score (lower is better - 0 means no hallucination)

    Requires: 'context' or 'knowledge_base' field in data

    Hallucination scoring (inverse of faithfulness):
    - 0-10: No hallucination, all facts grounded in context
    - 11-30: Minor unsupported embellishments but core facts correct
    - 31-50: Moderate hallucination with some fabricated details
    - 51-70: Significant hallucination with major unsupported claims
    - 71-100: Severe hallucination, mostly fabricated information
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_metric_name(self) -> str:
        return "hallucination"

    def requires_context(self) -> bool:
        """This metric requires context/knowledge base"""
        return True

    def prepare_decision(self, row: Dict[str, Any]) -> LayaDecision:
        """Formulate hallucination detection as a scoring question"""
        user_query = row.get('user_query', '')
        actual = row.get('actual_response', '')
        context = row.get('context', row.get('knowledge_base', ''))
        test_id = row.get('test_id', row.get('id', 'unknown'))

        question = f"""Rate the degree of hallucination in the bot's response on a scale from 0 to 100.

Hallucination means fabricated or unsupported information that is not grounded in the provided context.

User Query: "{user_query}"

Context/Knowledge Base: "{context}"

Bot Response: "{actual}"

Hallucination scoring:
- 0-10: No hallucination, all facts grounded in context
- 11-30: Minor unsupported embellishments but core facts correct
- 31-50: Moderate hallucination with some fabricated details
- 51-70: Significant hallucination with major unsupported claims
- 71-100: Severe hallucination, mostly fabricated information

What is the hallucination score (0-100)?"""

        return LayaDecision(
            question=question,
            decision_type=DecisionType.SCORE,
            decision_id=f"hallucination_{test_id}",
            scale_min=0,
            scale_max=100,
            metadata={'metric': 'hallucination', 'test_id': test_id}
        )

    def process_result(self, laya_result: Any, row: Dict[str, Any]) -> MetricResult:
        """Convert Laya score result into hallucination metric result"""
        score = laya_result.score
        confidence = laya_result.confidence

        # For hallucination, lower is better, so we invert for normalized score
        # A score of 0 (no hallucination) = 1.0 normalized
        # A score of 100 (severe hallucination) = 0.0 normalized
        normalized_score = (100 - score) / 100.0

        if score <= 10:
            level = "none"
            reasoning = "No hallucination detected. All facts are grounded in the context."
        elif score <= 30:
            level = "minor"
            reasoning = "Minor unsupported embellishments, but core facts are correct."
        elif score <= 50:
            level = "moderate"
            reasoning = "Moderate hallucination with some fabricated details."
        elif score <= 70:
            level = "significant"
            reasoning = "Significant hallucination with major unsupported claims."
        else:
            level = "severe"
            reasoning = "Severe hallucination. Response contains mostly fabricated information."

        return MetricResult(
            metric_name=self.get_metric_name(),
            value=score,
            score=normalized_score,  # Inverted: higher normalized score = less hallucination
            confidence=confidence,
            raw_value=laya_result,
            reasoning=f"Hallucination score: {score}/100 ({level}). {reasoning}",
            metadata={
                'hallucination_level': level,
                'test_id': row.get('test_id', row.get('id')),
                'interpretation': 'lower_is_better'
            }
        )


# Register the metric
from laya_eval.core.metrics import MetricRegistry
MetricRegistry.register(HallucinationMetric)
