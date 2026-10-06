"""
Chatbot Evaluator

Main orchestrator for chatbot response evaluation using Laya AI.
Coordinates data loading, metric evaluation, and result aggregation.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime
from tqdm import tqdm

from laya_eval.preprocessing.data_loader import DataLoader, load_evaluation_data
from laya_eval.models.laya_client import LayaClient, create_client
from laya_eval.core.metrics import (
    BaseMetric,
    MetricResult,
    BatchMetricResults,
    MetricRegistry
)

# Import all metrics to register them
from laya_eval.metrics.response_match import ResponseMatchMetric
from laya_eval.metrics.relevancy import RelevancyMetric
from laya_eval.metrics.faithfulness import FaithfulnessMetric
from laya_eval.metrics.hallucination import HallucinationMetric
from laya_eval.metrics.contextual_recall import ContextualRecallMetric
from laya_eval.metrics.toxicity import ToxicityMetric

logger = logging.getLogger(__name__)


@dataclass
class RowEvaluation:
    """Complete evaluation for one data row"""
    row_id: Union[int, str]
    user_query: str
    actual_response: str
    expected_response: Optional[str]
    metrics: Dict[str, MetricResult]
    overall_score: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'row_id': self.row_id,
            'user_query': self.user_query,
            'actual_response': self.actual_response,
            'expected_response': self.expected_response,
            'overall_score': self.overall_score,
            'timestamp': self.timestamp.isoformat(),
            **{
                f"{metric_name}_{key}": value
                for metric_name, result in self.metrics.items()
                for key, value in [
                    ('value', result.value),
                    ('score', result.score),
                    ('confidence', result.confidence)
                ]
            }
        }


@dataclass
class EvaluationResults:
    """Complete evaluation results for a dataset"""
    rows: List[RowEvaluation]
    overall_accuracy: float
    metric_summaries: Dict[str, Dict[str, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rows)

    def to_dataframe(self) -> pd.DataFrame:
        """Export as pandas DataFrame"""
        return pd.DataFrame([row.to_dict() for row in self.rows])

    def to_csv(self, path: Union[str, Path]) -> None:
        """Export as CSV"""
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info(f"Results exported to {path}")

    def to_json(self, path: Union[str, Path]) -> None:
        """Export as JSON"""
        df = self.to_dataframe()
        df.to_json(path, orient='records', indent=2)
        logger.info(f"Results exported to {path}")

    def summary(self) -> str:
        """Generate text summary report"""
        lines = [
            "=" * 60,
            "EVALUATION SUMMARY",
            "=" * 60,
            f"Total Samples Evaluated: {len(self.rows)}",
            f"Overall Accuracy: {self.overall_accuracy:.1%}",
            "",
            "Metric Performance:",
            "-" * 60,
        ]

        for metric_name, summary in self.metric_summaries.items():
            lines.append(f"\n{metric_name.upper()}:")
            for key, value in summary.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.1%}" if value <= 1.0 else f"  {key}: {value:.2f}")
                else:
                    lines.append(f"  {key}: {value}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


class ChatbotEvaluator:
    """
    Main orchestrator for chatbot evaluation

    Coordinates:
    - Data loading and validation
    - Metric selection and initialization
    - Batch evaluation with Laya AI
    - Result aggregation and reporting

    Example usage:
        evaluator = ChatbotEvaluator(
            model='laya',
            metrics=['all']
        )
        results = evaluator.evaluate_file('responses.csv')
        print(results.summary())
        results.to_csv('evaluation_results.csv')
    """

    # Default metric weights for overall accuracy calculation
    DEFAULT_METRIC_WEIGHTS = {
        'response_match': 0.25,
        'answer_relevancy': 0.20,
        'faithfulness': 0.20,
        'hallucination': 0.15,  # Note: inverted (lower is better)
        'contextual_recall': 0.15,
        'toxicity': 0.05,       # Note: inverted (lower is better)
    }

    def __init__(
        self,
        model: str = "auto",
        metrics: Union[str, List[str]] = "all",
        batch_size: int = 32,
        metric_weights: Optional[Dict[str, float]] = None,
        strict_mode: bool = False,
        show_progress: bool = True,
    ):
        """
        Initialize Chatbot Evaluator

        Args:
            model: Laya model to use ('laya', 'laya-multilingual', 'auto')
            metrics: List of metrics to evaluate or 'all' for all available metrics
            batch_size: Number of samples to process in each batch
            metric_weights: Custom weights for overall accuracy calculation
            strict_mode: If True, raise errors on validation failures
            show_progress: If True, show progress bars during evaluation
        """
        self.model = model
        self.batch_size = batch_size
        self.strict_mode = strict_mode
        self.show_progress = show_progress

        # Initialize Laya client
        logger.info(f"Initializing Laya client with model: {model}")
        self.laya_client = create_client(model=model)

        # Health check
        if not self.laya_client.health_check():
            logger.warning("Laya client health check failed")

        # Initialize metrics
        self.metrics = self._initialize_metrics(metrics)
        logger.info(f"Initialized {len(self.metrics)} metrics: {[m.get_metric_name() for m in self.metrics]}")

        # Set metric weights
        self.metric_weights = metric_weights or self.DEFAULT_METRIC_WEIGHTS

        # Initialize data loader
        self.data_loader = DataLoader(strict_mode=strict_mode)

    def _initialize_metrics(self, metrics: Union[str, List[str]]) -> List[BaseMetric]:
        """Initialize metric instances"""
        if metrics == "all":
            metric_names = MetricRegistry.list_metrics()
        elif isinstance(metrics, str):
            metric_names = [metrics]
        else:
            metric_names = metrics

        initialized = []
        for name in metric_names:
            metric_class = MetricRegistry.get(name)
            if metric_class:
                initialized.append(metric_class())
            else:
                logger.warning(f"Metric '{name}' not found in registry")

        if not initialized:
            raise ValueError("No valid metrics specified")

        return initialized

    def evaluate_file(self, file_path: Union[str, Path]) -> EvaluationResults:
        """
        Evaluate chatbot responses from a CSV/Excel file

        Args:
            file_path: Path to data file

        Returns:
            EvaluationResults with all metrics and scores
        """
        logger.info(f"Loading evaluation data from: {file_path}")

        # Load data
        df = load_evaluation_data(file_path, strict_mode=self.strict_mode, clean=True)

        # Evaluate
        return self.evaluate(df)

    def evaluate(self, data: pd.DataFrame) -> EvaluationResults:
        """
        Evaluate chatbot responses from a DataFrame

        Args:
            data: DataFrame with evaluation data

        Returns:
            EvaluationResults with all metrics and scores
        """
        logger.info(f"Evaluating {len(data)} samples with {len(self.metrics)} metrics")

        # Convert DataFrame to list of dicts
        rows = data.to_dict('records')

        # Evaluate each row
        row_evaluations = []

        iterator = tqdm(rows, desc="Evaluating", disable=not self.show_progress)

        for idx, row in enumerate(iterator):
            row_eval = self._evaluate_row(row, idx)
            if row_eval:
                row_evaluations.append(row_eval)

        # Calculate metric summaries
        metric_summaries = self._calculate_metric_summaries(row_evaluations)

        # Calculate overall accuracy
        overall_accuracy = self._calculate_overall_accuracy(row_evaluations)

        results = EvaluationResults(
            rows=row_evaluations,
            overall_accuracy=overall_accuracy,
            metric_summaries=metric_summaries,
            metadata={
                'total_rows': len(data),
                'evaluated_rows': len(row_evaluations),
                'metrics_used': [m.get_metric_name() for m in self.metrics],
                'model': self.model,
                'timestamp': datetime.now().isoformat()
            }
        )

        logger.info(f"Evaluation complete. Overall accuracy: {overall_accuracy:.1%}")

        return results

    def _evaluate_row(self, row: Dict[str, Any], idx: int) -> Optional[RowEvaluation]:
        """Evaluate a single row"""
        metric_results = {}

        # Evaluate each metric
        for metric in self.metrics:
            try:
                result = metric.evaluate(row, self.laya_client)
                if result:
                    metric_results[metric.get_metric_name()] = result
            except Exception as e:
                logger.error(f"Error evaluating {metric.get_metric_name()} for row {idx}: {e}")

        if not metric_results:
            logger.warning(f"No metrics could be evaluated for row {idx}")
            return None

        # Calculate overall score for this row
        overall_score = self._calculate_row_score(metric_results)

        return RowEvaluation(
            row_id=row.get('test_id', row.get('id', idx)),
            user_query=row.get('user_query', ''),
            actual_response=row.get('actual_response', ''),
            expected_response=row.get('expected_response'),
            metrics=metric_results,
            overall_score=overall_score
        )

    def _calculate_row_score(self, metric_results: Dict[str, MetricResult]) -> float:
        """Calculate weighted overall score for a single row"""
        weighted_sum = 0.0
        total_weight = 0.0

        for metric_name, result in metric_results.items():
            weight = self.metric_weights.get(metric_name, 0.0)
            if weight > 0:
                weighted_sum += result.score * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _calculate_metric_summaries(
        self,
        row_evaluations: List[RowEvaluation]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics for each metric"""
        summaries = {}

        for metric in self.metrics:
            metric_name = metric.get_metric_name()
            scores = [
                row.metrics[metric_name].score
                for row in row_evaluations
                if metric_name in row.metrics
            ]

            if scores:
                summaries[metric_name] = {
                    'mean_score': sum(scores) / len(scores),
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'pass_rate': sum(1 for s in scores if s >= 0.5) / len(scores),
                    'samples_evaluated': len(scores)
                }

        return summaries

    def _calculate_overall_accuracy(
        self,
        row_evaluations: List[RowEvaluation]
    ) -> float:
        """Calculate overall accuracy across all rows and metrics"""
        if not row_evaluations:
            return 0.0

        total_score = sum(row.overall_score for row in row_evaluations)
        return total_score / len(row_evaluations)


def quick_evaluate(
    file_path: Union[str, Path],
    model: str = "auto",
    metrics: Union[str, List[str]] = "all"
) -> EvaluationResults:
    """
    Convenience function for quick evaluation

    Args:
        file_path: Path to CSV/Excel file
        model: Laya model to use
        metrics: Metrics to evaluate

    Returns:
        EvaluationResults

    Example:
        results = quick_evaluate('responses.csv')
        print(results.summary())
    """
    evaluator = ChatbotEvaluator(model=model, metrics=metrics)
    return evaluator.evaluate_file(file_path)
