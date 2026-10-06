"""
Faithfulness Metric

Score-based metric (0-100) that evaluates how faithful the bot's response
is to the provided context/knowledge base. Uses Laya's 'score' decision.
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


class FaithfulnessMetric(BaseMetric):
    """
    Evaluates how faithful the bot response is to the provided context

    Decision Type: SCORE (0-100 scale)
    Output: Faithfulness score (higher = more faithful to context)

    Requires: 'context' or 'knowledge_base' field in data

    Faithfulness criteria:
    - 90-100: All information directly grounded in context, no additions
    - 70-89: Mostly grounded with minor reasonable inferences
    - 50-69: Partially grounded but includes unverified claims
    - 30-49: Some grounding but significant unsupported information
    - 0-29: Little to no grounding in provided context
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_metric_name(self) -> str:
        return "faithfulness"

    def requires_context(self) -> bool:
        """This metric requires context/knowledge base"""
        return True

    def prepare_decision(self, row: Dict[str, Any]) -> LayaDecision:
        """Formulate faithfulness evaluation as a scoring question"""
        user_query = row.get('user_query', '')
        actual = row.get('actual_response', '')
        context = row.get('context', row.get('knowledge_base', ''))
        test_id = row.get('test_id', row.get('id', 'unknown'))

        question = f"""Rate the faithfulness of the bot's response to the provided context on a scale from 0 to 100.

User Query: "{user_query}"

Context/Knowledge Base: "{context}"

Bot Response: "{actual}"

Faithfulness criteria:
- 90-100: All information is directly grounded in the context, no additions
- 70-89: Mostly grounded with minor reasonable inferences
- 50-69: Partially grounded but includes unverified claims
- 30-49: Some grounding but significant unsupported information
- 0-29: Little to no grounding in provided context

What is the faithfulness score (0-100)?"""

        return LayaDecision(
            question=question,
            decision_type=DecisionType.SCORE,
            decision_id=f"faithfulness_{test_id}",
            scale_min=0,
            scale_max=100,
            metadata={'metric': 'faithfulness', 'test_id': test_id}
        )

    def process_result(self, laya_result: Any, row: Dict[str, Any]) -> MetricResult:
        """Convert Laya score result into faithfulness metric result"""
        score = laya_result.score
        confidence = laya_result.confidence
        normalized_score = score / 100.0

        if score >= 90:
            level = "highly faithful"
            reasoning = "All information is directly grounded in the provided context."
        elif score >= 70:
            level = "mostly faithful"
            reasoning = "Mostly grounded in context with only minor reasonable inferences."
        elif score >= 50:
            level = "partially faithful"
            reasoning = "Partially grounded but includes some unverified claims."
        elif score >= 30:
            level = "somewhat faithful"
            reasoning = "Some grounding but contains significant unsupported information."
        else:
            level = "unfaithful"
            reasoning = "Little to no grounding in the provided context."

        return MetricResult(
            metric_name=self.get_metric_name(),
            value=score,
            score=normalized_score,
            confidence=confidence,
            raw_value=laya_result,
            reasoning=f"Faithfulness score: {score}/100 ({level}). {reasoning}",
            metadata={'faithfulness_level': level, 'test_id': row.get('test_id', row.get('id'))}
        )


# Register the metric
from laya_eval.core.metrics import MetricRegistry
MetricRegistry.register(FaithfulnessMetric)
