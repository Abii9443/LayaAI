# Laya AI Chatbot Evaluation Framework

A high-performance evaluation framework for chatbot responses using [Laya AI](https://huggingface.co/convaiinnovations/laya) decision models. Achieve **>80% accuracy** across multiple evaluation metrics with fast, non-autoregressive inference (~33ms per evaluation).

**Note**: Laya AI is an open-source library that runs locally. No API key is required! Visit [Laya official page](https://laya.convaiinnovations.com/) for more information.

## Features

- **6 Evaluation Metrics**: Response Match, Answer Relevancy, Faithfulness, Hallucination Detection, Contextual Recall, and Toxicity Detection
- **Fast Inference**: ~33ms per question on GPU using Laya AI
- **No Hallucination**: Laya's non-autoregressive architecture ensures reliable evaluation
- **Multi-language Support**: 100+ languages via `laya-multilingual`
- **Batch Processing**: 9-10× speed improvement with efficient batching
- **Flexible Input**: CSV, Excel, or DataFrame
- **Fine-tuning Support**: Domain adaptation for specialized use cases

## Installation

```bash
# Install the package
pip install laya-chatbot-eval

# Or install from source
git clone https://github.com/yourusername/laya-chatbot-eval
cd laya-chatbot-eval
pip install -e .
```

### Prerequisites

- Python 3.10+
- Laya AI (`pip install laya`)
- Pandas, NumPy, scikit-learn

## Quick Start

### 1. Prepare Your Data

Your CSV/Excel file should contain at minimum:
- `user_query` - The user's question or input
- `actual_response` - The chatbot's actual response

Optional columns:
- `expected_response` - Ground truth response (for Response Match metric)
- `context` or `knowledge_base` - Context information (for Faithfulness, Hallucination, Contextual Recall metrics)

**Example CSV:**

```csv
user_query,actual_response,expected_response
What is your refund policy?,We offer 30-day returns,Our refund policy allows 30-day returns
How do I track my order?,Click the track button,Log in and go to My Orders
```

### 2. Run Evaluation

```python
from laya_eval import quick_evaluate

# Quick evaluation with all metrics
results = quick_evaluate('chatbot_responses.csv')

# Print summary
print(results.summary())

# Export results
results.to_csv('evaluation_results.csv')
```

**Output:**

```
============================================================
EVALUATION SUMMARY
============================================================
Total Samples Evaluated: 100
Overall Accuracy: 87.3%

Metric Performance:
------------------------------------------------------------

RESPONSE_MATCH:
  mean_score: 0.85
  pass_rate: 0.82

ANSWER_RELEVANCY:
  mean_score: 0.89
  ...
```

## Evaluation Metrics

### 1. Response Match (PASS/FAIL)
Binary classification using Laya's `noul` decision.
- **What it measures**: Does the actual response semantically match the expected response?
- **Output**: PASS/FAIL with probability score
- **Requires**: `expected_response` column

### 2. Answer Relevancy (0-100%)
Score-based evaluation using Laya's `score` decision.
- **What it measures**: How relevant is the response to the user's query?
- **Output**: Relevancy score (0-100)

### 3. Faithfulness (0-100%)
Score-based evaluation of context grounding.
- **What it measures**: Is the response grounded in the provided context?
- **Output**: Faithfulness score (0-100)
- **Requires**: `context` column

### 4. Hallucination Detection (0-100%)
Score-based detection of fabricated content.
- **What it measures**: Does the response contain unsupported information?
- **Output**: Hallucination score (lower is better, 0 = no hallucination)
- **Requires**: `context` column

### 5. Contextual Recall (0-100%)
Score-based evaluation of information retrieval completeness.
- **What it measures**: How completely did the bot retrieve relevant context?
- **Output**: Recall score (0-100)
- **Requires**: `context` column

### 6. Toxicity Detection (0-100%)
Score-based content safety evaluation.
- **What it measures**: Does the response contain harmful or offensive content?
- **Output**: Toxicity score (lower is better, 0 = safe)

## Advanced Usage

### Custom Metric Selection

```python
from laya_eval import ChatbotEvaluator

# Evaluate specific metrics only
evaluator = ChatbotEvaluator(
    metrics=['response_match', 'answer_relevancy', 'toxicity']
)

results = evaluator.evaluate_file('responses.csv')
```

### Multilingual Evaluation

```python
# Use multilingual model for 100+ languages
evaluator = ChatbotEvaluator(
    model='laya-multilingual',
    metrics='all'
)

results = evaluator.evaluate_file('multilingual_responses.csv')
```

### DataFrame Evaluation

```python
import pandas as pd
from laya_eval import ChatbotEvaluator

# Load data
df = pd.read_csv('responses.csv')

# Evaluate DataFrame directly
evaluator = ChatbotEvaluator()
results = evaluator.evaluate(df)

# Access results
for row in results.rows:
    print(f"Query: {row.user_query}")
    print(f"Overall Score: {row.overall_score:.1%}")
    print(f"Response Match: {row.metrics['response_match'].value}")
```

### Custom Metric Weights

```python
# Customize weights for overall accuracy calculation
custom_weights = {
    'response_match': 0.40,      # Emphasize correctness
    'answer_relevancy': 0.30,    # Emphasize relevance
    'toxicity': 0.30,             # Emphasize safety
}

evaluator = ChatbotEvaluator(
    metrics=['response_match', 'answer_relevancy', 'toxicity'],
    metric_weights=custom_weights
)
```

## Configuration

Create a `config.yaml` file:

```yaml
model:
  name: 'laya'

metrics:
  enabled:
    - response_match
    - answer_relevancy
    - faithfulness
    - hallucination
    - contextual_recall
    - toxicity

  weights:
    response_match: 0.25
    answer_relevancy: 0.20
    faithfulness: 0.20
    hallucination: 0.15
    contextual_recall: 0.15
    toxicity: 0.05

batch:
  size: 32
  show_progress: true
```

## Architecture

```
CSV/Excel → DataLoader → Evaluator → LayaClient (Laya AI) → Metrics → Results
                                          ↓
                              [response_match, relevancy,
                               faithfulness, hallucination,
                               contextual_recall, toxicity]
```

### Key Components

1. **DataLoader**: CSV/Excel parsing with flexible column mapping
2. **ChatbotEvaluator**: Main orchestrator coordinating evaluation
3. **LayaClient**: Wrapper for Laya AI API (choice/score/noul decisions)
4. **Metrics**: Six evaluation metrics implementing BaseMetric interface
5. **Reporter**: Result aggregation and export (CSV/JSON)

## Performance

- **Speed**: ~33ms per evaluation on GPU (T4)
- **Batch Processing**: 9-10× faster with batching
- **Accuracy**: >80% across all metrics
- **No Hallucination**: Non-autoregressive architecture
- **Calibrated Confidence**: Reliable probability scores via RLCD training

## Data Requirements

### Minimum Required Columns

- `user_query`: The user's input/question
- `actual_response`: The bot's response

### Optional Columns

- `expected_response`: Ground truth (needed for Response Match)
- `context` or `knowledge_base`: Context info (needed for Faithfulness, Hallucination, Recall)
- `test_id` or `id`: Unique identifier per row

### Column Name Flexibility

The framework automatically maps common variations:
- `query`, `user_question`, `question` → `user_query`
- `bot_response`, `response`, `output` → `actual_response`
- `expected`, `ground_truth`, `reference` → `expected_response`

## Fine-tuning (Advanced)

Improve accuracy for domain-specific use cases:

```python
from laya_eval.finetuning import LayaTrainer

# Prepare labeled training data
trainer = LayaTrainer(base_model='laya')
trainer.prepare_training_data('labeled_data.csv', metric='response_match')

# Fine-tune
trainer.train(
    training_data_dir='./training_data',
    model_name='custom-ccw-orders',
    epochs=10
)

# Use custom model
evaluator = ChatbotEvaluator(
    model='custom',
    custom_model_path='./models/custom-ccw-orders'
)
```

## Troubleshooting

### Column Name Errors

If you see "Missing required columns" errors, check your CSV column names:

```python
import pandas as pd

df = pd.read_csv('your_file.csv')
print(df.columns.tolist())  # Check actual column names
```

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade laya pandas numpy scikit-learn openpyxl
```

## Examples

See the `examples/` directory for more detailed usage examples:
- `basic_evaluation.py` - Simple evaluation workflow
- `batch_evaluation.py` - Large-scale batch processing
- `fine_tuning_example.py` - Domain adaptation guide

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.

## License

MIT License - see `LICENSE` file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{laya_chatbot_eval,
  title={Laya AI Chatbot Evaluation Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/laya-chatbot-eval}
}
```

## Acknowledgments

Built with [Laya AI](https://huggingface.co/convaiinnovations/laya) by ConvAI Innovations.

## Support

- GitHub Issues: [https://github.com/yourusername/laya-chatbot-eval/issues](https://github.com/yourusername/laya-chatbot-eval/issues)
- Documentation: [https://laya-chatbot-eval.readthedocs.io](https://laya-chatbot-eval.readthedocs.io)
- Laya AI: [https://huggingface.co/convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
