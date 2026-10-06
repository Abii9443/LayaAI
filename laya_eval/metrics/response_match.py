"""
Response Match Metric

Binary classification metric that determines if the actual bot response
matches the expected response semantically. Uses Laya's 'noul' decision
for yes/no probability assessment.
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


class ResponseMatchMetric(BaseMetric):
    """
    Evaluates whether the actual bot response matches the expected response

    Decision Type: NOUL (yes/no probability)
    Output: PASS/FAIL with confidence score

    This metric performs semantic matching rather than exact string comparison.
    It considers:
    - Core information accuracy
    - Intent fulfillment
    - Key facts preservation
    - Overall message equivalence

    Example:
        Expected: "Your order will arrive in 3-5 business days"
        Actual: "Delivery takes 3 to 5 working days"
        Result: PASS (semantically equivalent)
    """

    def __init__(self, threshold: float = 0.70, **kwargs):
        """
        Initialize Response Match metric

        Args:
            threshold: Probability threshold for PASS classification (default: 0.70)
                      Values >= threshold are PASS, values < threshold are FAIL
        """
        super().__init__(threshold=threshold, **kwargs)

    def get_metric_name(self) -> str:
        """Return metric identifier"""
        return "response_match"

    def requires_expected_response(self) -> bool:
        """This metric requires an expected response"""
        return True

    def prepare_decision(self, row: Dict[str, Any]) -> LayaDecision:
        """
        Formulate response matching as a yes/no question for Laya

        Args:
            row: Data row with 'user_query', 'expected_response', 'actual_response'

        Returns:
            LayaDecision with noul-type question
        """
        user_query = row.get('user_query', '')
        expected = row.get('expected_response', '')
        actual = row.get('actual_response', '')
        test_id = row.get('test_id', row.get('id', 'unknown'))

        # Formulate the evaluation question
        question = f"""Does the actual bot response match the expected response for this user query?

Consider semantic equivalence, not exact wording. The actual response should:
- Convey the same core information as the expected response
- Address the same intent and purpose
- Preserve key facts and details
- Maintain equivalent meaning even if phrased differently

User Query: "{user_query}"

Expected Response: "{expected}"

Actual Response: "{actual}"

Question: Does the actual response match the expected response semantically?"""

        return LayaDecision(
            question=question,
            decision_type=DecisionType.NOUL,
            decision_id=f"response_match_{test_id}",
            threshold=self.threshold,
            metadata={
                'metric': 'response_match',
                'test_id': test_id
            }
        )

    def process_result(self, laya_result: Any, row: Dict[str, Any]) -> MetricResult:
        """
        Convert Laya's noul result into a PASS/FAIL metric result

        Args:
            laya_result: NoulResult from Laya AI
            row: Original data row

        Returns:
            MetricResult with PASS/FAIL value and normalized score
        """
        # Extract probability from Laya result
        probability = laya_result.probability
        confidence = laya_result.confidence

        # Determine PASS/FAIL based on threshold
        is_match = probability >= self.threshold
        value = "PASS" if is_match else "FAIL"

        # Generate reasoning
        if is_match:
            reasoning = (
                f"The actual response matches the expected response with "
                f"{probability:.1%} probability (threshold: {self.threshold:.1%}). "
                f"The responses are semantically equivalent."
            )
        else:
            reasoning = (
                f"The actual response does NOT match the expected response. "
                f"Match probability is {probability:.1%} (threshold: {self.threshold:.1%}). "
                f"The responses have significant semantic differences."
            )

        return MetricResult(
            metric_name=self.get_metric_name(),
            value=value,
            score=probability,  # Normalized 0-1 score
            confidence=confidence,
            raw_value=laya_result,
            reasoning=reasoning,
            metadata={
                'threshold': self.threshold,
                'is_match': is_match,
                'test_id': row.get('test_id', row.get('id'))
            }
        )


# Register the metric
from laya_eval.core.metrics import MetricRegistry
MetricRegistry.register(ResponseMatchMetric)
