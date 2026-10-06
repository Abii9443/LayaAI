"""
Contextual Recall Metric

Score-based metric (0-100) that evaluates how much relevant context
information is recalled and used in the bot's response.
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


class ContextualRecallMetric(BaseMetric):
    """
    Evaluates how completely the bot retrieved and used relevant context

    Decision Type: SCORE (0-100 scale)
    Output: Contextual recall score (higher = more complete retrieval)

    Requires: 'context' or 'knowledge_base' field in data
    Optional: 'expected_response' for comparison

    Measures completeness of response relative to available context:
    - 90-100: All relevant context points retrieved and used
    - 70-89: Most relevant points retrieved, minor omissions
    - 50-69: Partial retrieval, missing important context
    - 30-49: Minimal context retrieval, major gaps
    - 0-29: Little to no relevant context retrieved
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_metric_name(self) -> str:
        return "contextual_recall"

    def requires_context(self) -> bool:
        """This metric requires context/knowledge base"""
        return True

    def prepare_decision(self, row: Dict[str, Any]) -> LayaDecision:
        """Formulate contextual recall evaluation as a scoring question"""
        user_query = row.get('user_query', '')
        actual = row.get('actual_response', '')
        context = row.get('context', row.get('knowledge_base', ''))
        expected = row.get('expected_response', '')
        test_id = row.get('test_id', row.get('id', 'unknown'))

        # Include expected response if available for comparison
        expected_section = f'\n\nExpected Key Points (for reference): "{expected}"' if expected else ''

        question = f"""Rate the contextual recall of the bot's response on a scale from 0 to 100.

Contextual recall measures how completely the bot retrieved and used relevant information from the available context.

User Query: "{user_query}"

Available Context: "{context}"{expected_section}

Actual Bot Response: "{actual}"

Contextual recall criteria:
- 90-100: All relevant context points retrieved and used appropriately
- 70-89: Most relevant points retrieved, only minor omissions
- 50-69: Partial retrieval, missing some important context
- 30-49: Minimal context retrieval, major information gaps
- 0-29: Little to no relevant context retrieved

What is the contextual recall score (0-100)?"""

        return LayaDecision(
            question=question,
            decision_type=DecisionType.SCORE,
            decision_id=f"contextual_recall_{test_id}",
            scale_min=0,
            scale_max=100,
            metadata={'metric': 'contextual_recall', 'test_id': test_id}
        )

    def process_result(self, laya_result: Any, row: Dict[str, Any]) -> MetricResult:
        """Convert Laya score result into contextual recall metric result"""
        score = laya_result.score
        confidence = laya_result.confidence
        normalized_score = score / 100.0

        if score >= 90:
            level = "excellent"
            reasoning = "All relevant context points were retrieved and used appropriately."
        elif score >= 70:
            level = "good"
            reasoning = "Most relevant points retrieved with only minor omissions."
        elif score >= 50:
            level = "fair"
            reasoning = "Partial retrieval with some important context missing."
        elif score >= 30:
            level = "poor"
            reasoning = "Minimal context retrieval with major information gaps."
        else:
            level = "very poor"
            reasoning = "Little to no relevant context was retrieved."

        return MetricResult(
            metric_name=self.get_metric_name(),
            value=score,
            score=normalized_score,
            confidence=confidence,
            raw_value=laya_result,
            reasoning=f"Contextual recall score: {score}/100 ({level}). {reasoning}",
            metadata={'recall_level': level, 'test_id': row.get('test_id', row.get('id'))}
        )


# Register the metric
from laya_eval.core.metrics import MetricRegistry
MetricRegistry.register(ContextualRecallMetric)
