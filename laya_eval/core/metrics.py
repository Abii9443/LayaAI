"""
Base Metric Classes for Chatbot Evaluation

Defines the interface and data structures for all evaluation metrics.
Each metric formulates evaluation tasks as Laya AI decision questions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of Laya AI decisions"""
    CHOICE = "choice"  # Select one option from a list
    SCORE = "score"    # Rate on ordinal scale
    NOUL = "noul"      # Yes/no probability


@dataclass
class LayaDecision:
    """
    Represents a decision to be made by Laya AI

    This is the standardized format for all evaluation questions.
    """
    question: str
    decision_type: DecisionType
    decision_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Type-specific fields
    options: Optional[List[str]] = None  # For CHOICE
    scale_min: Optional[int] = None      # For SCORE
    scale_max: Optional[int] = None      # For SCORE
    threshold: Optional[float] = 0.5     # For NOUL


@dataclass
class MetricResult:
    """
    Result from evaluating a single metric on one sample

    This is the standardized output format for all metrics.
    """
    metric_name: str
    value: Union[str, float, bool]  # The main result (PASS/FAIL, score, etc.)
    score: float                     # Normalized score (0-1)
    confidence: float                # Laya confidence score
    raw_value: Any = None            # Original Laya response
    reasoning: Optional[str] = None  # Optional explanation
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'metric_name': self.metric_name,
            'value': self.value,
            'score': self.score,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'metadata': self.metadata
        }


@dataclass
class BatchMetricResults:
    """Results from evaluating a metric on multiple samples"""
    metric_name: str
    results: List[MetricResult]
    aggregate_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.results)

    def get_scores(self) -> List[float]:
        """Get list of all scores"""
        return [r.score for r in self.results]

    def get_mean_score(self) -> float:
        """Calculate mean score"""
        scores = self.get_scores()
        return sum(scores) / len(scores) if scores else 0.0

    def get_pass_rate(self) -> float:
        """
        Calculate pass rate (for binary metrics)

        For binary metrics (PASS/FAIL), this returns the percentage of passing samples.
        For continuous metrics, returns percentage with score >= 0.5.
        """
        if not self.results:
            return 0.0

        passing = sum(1 for r in self.results if r.score >= 0.5)
        return passing / len(self.results)


class BaseMetric(ABC):
    """
    Abstract base class for all evaluation metrics

    Each metric must implement:
    1. prepare_decision() - Convert a data row into a Laya decision question
    2. process_result() - Convert Laya's response into a MetricResult
    3. get_metric_name() - Return the metric's identifier

    Subclasses define how to formulate evaluation tasks as Laya AI questions.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        **kwargs
    ):
        """
        Initialize metric

        Args:
            threshold: Decision threshold for binary classifications
            **kwargs: Additional metric-specific parameters
        """
        self.threshold = threshold
        self.config = kwargs

    @abstractmethod
    def prepare_decision(self, row: Dict[str, Any]) -> LayaDecision:
        """
        Convert a data row into a Laya AI decision question

        This method formulates the evaluation task as a question that Laya can answer.

        Args:
            row: Dictionary with keys like 'user_query', 'actual_response', etc.

        Returns:
            LayaDecision object ready for Laya AI

        Example:
            For Response Match metric, this might return:
            LayaDecision(
                question="Does the actual response match the expected response?",
                decision_type=DecisionType.NOUL,
                decision_id=f"response_match_{row['test_id']}"
            )
        """
        pass

    @abstractmethod
    def process_result(self, laya_result: Any, row: Dict[str, Any]) -> MetricResult:
        """
        Convert Laya's prediction result into a standardized MetricResult

        Args:
            laya_result: Raw result from Laya AI (ChoiceResult, ScoreResult, or NoulResult)
            row: Original data row (for context/metadata)

        Returns:
            MetricResult with normalized scores and values

        Example:
            For Response Match metric:
            MetricResult(
                metric_name='response_match',
                value='PASS' if laya_result.probability >= threshold else 'FAIL',
                score=laya_result.probability,
                confidence=laya_result.confidence
            )
        """
        pass

    @abstractmethod
    def get_metric_name(self) -> str:
        """
        Return the metric's identifier

        Returns:
            Metric name as string (e.g., 'response_match', 'relevancy')
        """
        pass

    def requires_context(self) -> bool:
        """
        Check if this metric requires a 'context' field

        Returns:
            True if context is required, False otherwise
        """
        return False

    def requires_expected_response(self) -> bool:
        """
        Check if this metric requires an 'expected_response' field

        Returns:
            True if expected response is required, False otherwise
        """
        return False

    def validate_row(self, row: Dict[str, Any]) -> bool:
        """
        Validate that a row has all required fields for this metric

        Args:
            row: Data row to validate

        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        required = ['user_query', 'actual_response']

        if self.requires_context():
            required.append('context')

        if self.requires_expected_response():
            required.append('expected_response')

        for field in required:
            if field not in row or not row[field] or str(row[field]).strip() == '':
                logger.warning(
                    f"Row missing required field '{field}' for metric {self.get_metric_name()}"
                )
                return False

        return True

    def evaluate(self, row: Dict[str, Any], laya_client) -> Optional[MetricResult]:
        """
        Evaluate a single row (convenience method)

        Args:
            row: Data row to evaluate
            laya_client: LayaClient instance

        Returns:
            MetricResult or None if evaluation fails
        """
        if not self.validate_row(row):
            return None

        try:
            # Prepare decision
            decision = self.prepare_decision(row)

            # Execute with Laya
            if decision.decision_type == DecisionType.CHOICE:
                laya_result = laya_client.predict_choice(
                    state=self._build_state(row),
                    question=decision.question,
                    options=decision.options or [],
                    decision_id=decision.decision_id
                )
            elif decision.decision_type == DecisionType.SCORE:
                laya_result = laya_client.predict_score(
                    state=self._build_state(row),
                    question=decision.question,
                    scale_min=decision.scale_min or 0,
                    scale_max=decision.scale_max or 100,
                    decision_id=decision.decision_id
                )
            elif decision.decision_type == DecisionType.NOUL:
                laya_result = laya_client.predict_noul(
                    state=self._build_state(row),
                    question=decision.question,
                    threshold=decision.threshold or 0.5,
                    decision_id=decision.decision_id
                )
            else:
                raise ValueError(f"Unknown decision type: {decision.decision_type}")

            # Process result
            metric_result = self.process_result(laya_result, row)
            return metric_result

        except Exception as e:
            logger.error(
                f"Error evaluating {self.get_metric_name()} for row: {e}"
            )
            return None

    def evaluate_batch(
        self,
        rows: List[Dict[str, Any]],
        laya_client
    ) -> BatchMetricResults:
        """
        Evaluate multiple rows (batch evaluation)

        Args:
            rows: List of data rows
            laya_client: LayaClient instance

        Returns:
            BatchMetricResults containing all individual results
        """
        results = []

        for row in rows:
            result = self.evaluate(row, laya_client)
            if result:
                results.append(result)

        batch_results = BatchMetricResults(
            metric_name=self.get_metric_name(),
            results=results,
            aggregate_score=sum(r.score for r in results) / len(results) if results else 0.0,
            metadata={'total_evaluated': len(results), 'total_rows': len(rows)}
        )

        return batch_results

    def _build_state(self, row: Dict[str, Any]) -> str:
        """
        Build the state/context string for Laya from a data row

        This combines all relevant information into a single context string.

        Args:
            row: Data row

        Returns:
            Formatted state string
        """
        parts = []

        if 'user_query' in row and row['user_query']:
            parts.append(f"User Query: {row['user_query']}")

        if 'actual_response' in row and row['actual_response']:
            parts.append(f"Bot Response: {row['actual_response']}")

        if self.requires_expected_response() and 'expected_response' in row:
            if row['expected_response']:
                parts.append(f"Expected Response: {row['expected_response']}")

        if self.requires_context() and 'context' in row:
            if row['context']:
                parts.append(f"Context: {row['context']}")

        return "\n\n".join(parts)


class MetricRegistry:
    """
    Registry for managing available metrics

    Allows dynamic registration and retrieval of metric classes.
    """

    _metrics: Dict[str, type] = {}

    @classmethod
    def register(cls, metric_class: type) -> None:
        """Register a metric class"""
        if not issubclass(metric_class, BaseMetric):
            raise TypeError(f"{metric_class} must inherit from BaseMetric")

        # Get metric name from an instance
        instance = metric_class()
        name = instance.get_metric_name()
        cls._metrics[name] = metric_class
        logger.debug(f"Registered metric: {name}")

    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """Get a metric class by name"""
        return cls._metrics.get(name)

    @classmethod
    def get_all(cls) -> Dict[str, type]:
        """Get all registered metrics"""
        return cls._metrics.copy()

    @classmethod
    def list_metrics(cls) -> List[str]:
        """List all registered metric names"""
        return list(cls._metrics.keys())
