"""
Laya AI Client Wrapper

Handles all interactions with Laya AI decision models including:
- Choice decisions (select one label from options)
- Score decisions (rank on ordinal scales)
- Noul decisions (yes/no probability assessments)
- Batch processing for efficiency
"""

import os
import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import logging

try:
    import laya
    from laya import Router
except ImportError:
    raise ImportError(
        "Laya AI is not installed. Please install it with: pip install laya"
    )


logger = logging.getLogger(__name__)


@dataclass
class DecisionResult:
    """Base class for decision results"""
    decision_id: str
    confidence: float
    raw_response: Any


@dataclass
class ChoiceResult(DecisionResult):
    """Result from a choice decision"""
    selected_option: str
    all_probabilities: Dict[str, float]


@dataclass
class ScoreResult(DecisionResult):
    """Result from a score decision"""
    score: float
    scale_min: int
    scale_max: int


@dataclass
class NoulResult(DecisionResult):
    """Result from a noul (yes/no) decision"""
    probability: float  # Probability of "yes"
    is_yes: bool  # True if probability >= threshold


class LayaClient:
    """
    Wrapper for Laya AI API interactions

    Supports three decision types:
    - choice: Select one label from provided options
    - score: Rank responses on ordinal scales
    - noul: Yes/no probability assessments

    Features:
    - Automatic model routing
    - Batch processing
    - Error handling and retries
    - Rate limiting
    """

    def __init__(
        self,
        model: str = "auto",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 30,
        use_router: bool = True,
    ):
        """
        Initialize Laya AI client

        Args:
            model: Model to use ('laya', 'laya-multilingual', 'laya-typed-decisions', 'auto')
            max_retries: Maximum number of retry attempts on failure
            retry_delay: Delay between retries in seconds
            timeout: Request timeout in seconds
            use_router: Use Laya Router for automatic model selection (recommended)
        """
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.use_router = use_router

        # Initialize Laya client or router
        if use_router:
            logger.info("Initializing Laya Router with automatic model selection")
            self.agent = Router(preload=True)
        else:
            logger.info(f"Initializing Laya agent with model: {model}")
            if model == "auto":
                model = "convaiinnovations/laya"
            self.agent = laya.load(model)

    def predict_choice(
        self,
        state: str,
        question: str,
        options: List[str],
        decision_id: Optional[str] = None,
    ) -> ChoiceResult:
        """
        Make a choice decision (select one option from a list)

        Args:
            state: Context or text to evaluate
            question: Question to answer
            options: List of possible choices
            decision_id: Optional identifier for this decision

        Returns:
            ChoiceResult with selected option and probabilities
        """
        question_dict = {
            "type": "choice",
            "text": question,
            "options": options
        }

        result = self._execute_with_retry(state, [question_dict])

        # Extract choice result
        choice_result = result[0]
        selected = choice_result.get("answer", options[0])
        probs = choice_result.get("probabilities", {})
        confidence = choice_result.get("confidence", 0.0)

        return ChoiceResult(
            decision_id=decision_id or "choice",
            selected_option=selected,
            all_probabilities=probs,
            confidence=confidence,
            raw_response=choice_result
        )

    def predict_score(
        self,
        state: str,
        question: str,
        scale_min: int = 0,
        scale_max: int = 100,
        decision_id: Optional[str] = None,
    ) -> ScoreResult:
        """
        Make a score decision (rate on ordinal scale)

        Args:
            state: Context or text to evaluate
            question: Question to answer
            scale_min: Minimum score value
            scale_max: Maximum score value
            decision_id: Optional identifier for this decision

        Returns:
            ScoreResult with numeric score
        """
        question_dict = {
            "type": "score",
            "text": question,
            "scale": {
                "min": scale_min,
                "max": scale_max
            }
        }

        result = self._execute_with_retry(state, [question_dict])

        # Extract score result
        score_result = result[0]
        score = score_result.get("answer", scale_min)
        confidence = score_result.get("confidence", 0.0)

        return ScoreResult(
            decision_id=decision_id or "score",
            score=float(score),
            scale_min=scale_min,
            scale_max=scale_max,
            confidence=confidence,
            raw_response=score_result
        )

    def predict_noul(
        self,
        state: str,
        question: str,
        threshold: float = 0.5,
        decision_id: Optional[str] = None,
    ) -> NoulResult:
        """
        Make a noul decision (yes/no probability assessment)

        Args:
            state: Context or text to evaluate
            question: Yes/no question to answer
            threshold: Probability threshold for yes/no classification
            decision_id: Optional identifier for this decision

        Returns:
            NoulResult with probability and boolean classification
        """
        question_dict = {
            "type": "noul",
            "text": question
        }

        result = self._execute_with_retry(state, [question_dict])

        # Extract noul result
        noul_result = result[0]
        probability = noul_result.get("answer", 0.5)
        confidence = noul_result.get("confidence", 0.0)

        return NoulResult(
            decision_id=decision_id or "noul",
            probability=float(probability),
            is_yes=probability >= threshold,
            confidence=confidence,
            raw_response=noul_result
        )

    def predict_batch(
        self,
        states: List[str],
        questions: List[List[Dict[str, Any]]],
    ) -> List[List[Dict[str, Any]]]:
        """
        Process multiple decisions in batch for efficiency

        Args:
            states: List of context strings
            questions: List of question lists (one per state)

        Returns:
            List of results (one per state)
        """
        if len(states) != len(questions):
            raise ValueError("Number of states must match number of question lists")

        results = []
        for state, qs in zip(states, questions):
            result = self._execute_with_retry(state, qs)
            results.append(result)

        return results

    def _execute_with_retry(
        self,
        state: str,
        questions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Execute Laya prediction with retry logic

        Args:
            state: Context string
            questions: List of question dictionaries

        Returns:
            List of prediction results
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                # Call Laya predict method
                result = self.agent.predict(state=state, questions=questions)
                return result

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Laya prediction attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff

        # All retries failed
        logger.error(f"All {self.max_retries} retry attempts failed")
        raise RuntimeError(
            f"Laya prediction failed after {self.max_retries} attempts"
        ) from last_exception

    def health_check(self) -> bool:
        """
        Check if Laya client is healthy and can make predictions

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple test prediction
            result = self.predict_noul(
                state="This is a test",
                question="Is this a test?",
                decision_id="health_check"
            )
            return result.confidence > 0
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


def create_client(
    model: str = "auto",
    **kwargs
) -> LayaClient:
    """
    Factory function to create a LayaClient instance

    Args:
        model: Model to use ('laya', 'laya-multilingual', 'laya-typed-decisions', 'auto')
        **kwargs: Additional arguments passed to LayaClient constructor

    Returns:
        Configured LayaClient instance
    """
    return LayaClient(model=model, **kwargs)
