"""
Basic Evaluation Example

Demonstrates how to use the Laya Chatbot Evaluation Framework
to evaluate chatbot responses from a CSV file.
"""

import logging
from pathlib import Path
from laya_eval import ChatbotEvaluator, quick_evaluate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """
    Example 1: Quick evaluation with default settings
    """
    print("=" * 60)
    print("Example 1: Quick Evaluation")
    print("=" * 60)

    # Path to your data file
    data_file = "path/to/your/chatbot_responses.csv"

    # Quick evaluation with all metrics
    try:
        results = quick_evaluate(data_file, model='auto', metrics='all')

        # Print summary
        print(results.summary())

        # Export results
        results.to_csv('evaluation_results.csv')
        results.to_json('evaluation_results.json')

        print(f"\nResults exported to evaluation_results.csv and evaluation_results.json")

    except FileNotFoundError:
        print(f"Error: File not found at {data_file}")
        print("Please update the data_file path to point to your CSV file")
        return

    """
    Example 2: Customized evaluation
    """
    print("\n" + "=" * 60)
    print("Example 2: Customized Evaluation")
    print("=" * 60)

    # Create evaluator with custom settings
    evaluator = ChatbotEvaluator(
        model='laya',  # Use English model
        metrics=['response_match', 'answer_relevancy', 'toxicity'],  # Specific metrics
        batch_size=32,
        show_progress=True
    )

    try:
        # Evaluate from file
        results = evaluator.evaluate_file(data_file)

        # Access individual metrics
        print(f"\nMetric Performance:")
        for metric_name, summary in results.metric_summaries.items():
            print(f"\n{metric_name}:")
            print(f"  Mean Score: {summary['mean_score']:.1%}")
            print(f"  Pass Rate: {summary['pass_rate']:.1%}")
            print(f"  Samples: {summary['samples_evaluated']}")

        # Access individual row results
        print(f"\n\nSample Row Results:")
        for i, row in enumerate(results.rows[:3]):  # First 3 rows
            print(f"\nRow {i+1}:")
            print(f"  Query: {row.user_query[:50]}...")
            print(f"  Overall Score: {row.overall_score:.1%}")
            for metric_name, metric_result in row.metrics.items():
                print(f"  {metric_name}: {metric_result.value} (score: {metric_result.score:.2f})")

    except FileNotFoundError:
        print(f"Error: File not found at {data_file}")

    """
    Example 3: DataFrame evaluation
    """
    print("\n" + "=" * 60)
    print("Example 3: DataFrame Evaluation")
    print("=" * 60)

    import pandas as pd

    # Create sample data
    sample_data = pd.DataFrame([
        {
            'user_query': 'What is your refund policy?',
            'actual_response': 'We offer 30-day returns on all products.',
            'expected_response': 'Our refund policy allows 30-day returns.',
        },
        {
            'user_query': 'How do I track my order?',
            'actual_response': 'You can track your order in the My Orders section.',
            'expected_response': 'Log in and go to My Orders to track your shipment.',
        },
    ])

    # Evaluate DataFrame directly
    evaluator = ChatbotEvaluator(metrics='all')
    results = evaluator.evaluate(sample_data)

    print(f"Evaluated {len(results)} samples")
    print(f"Overall Accuracy: {results.overall_accuracy:.1%}")


if __name__ == '__main__':
    # Laya AI runs locally and does not require an API key
    main()
