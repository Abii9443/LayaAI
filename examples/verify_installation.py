"""
Verify Laya Chatbot Evaluation Framework Installation

Run this script to check if all components are properly installed.
"""

import sys

def check_imports():
    """Check if all required modules can be imported"""
    print("Checking imports...")
    errors = []

    try:
        import laya_eval
        print(f"✓ laya_eval v{laya_eval.__version__}")
    except ImportError as e:
        errors.append(f"✗ laya_eval: {e}")

    try:
        from laya_eval import ChatbotEvaluator
        print("✓ ChatbotEvaluator")
    except ImportError as e:
        errors.append(f"✗ ChatbotEvaluator: {e}")

    try:
        from laya_eval import DataLoader
        print("✓ DataLoader")
    except ImportError as e:
        errors.append(f"✗ DataLoader: {e}")

    try:
        from laya_eval import LayaClient
        print("✓ LayaClient")
    except ImportError as e:
        errors.append(f"✗ LayaClient: {e}")

    try:
        from laya_eval import MetricRegistry
        print(f"✓ MetricRegistry")
        metrics = MetricRegistry.list_metrics()
        print(f"  Registered metrics: {', '.join(metrics)}")
    except ImportError as e:
        errors.append(f"✗ MetricRegistry: {e}")

    return errors


def check_dependencies():
    """Check if required dependencies are installed"""
    print("\nChecking dependencies...")
    errors = []

    deps = [
        'laya',
        'pandas',
        'numpy',
        'sklearn',
        'yaml',
        'tqdm',
    ]

    for dep in deps:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            errors.append(f"✗ {dep} not installed")
            print(f"✗ {dep}")

    return errors


def check_environment():
    """Check environment configuration"""
    print("\nChecking environment...")

    # Laya runs locally and doesn't require an API key
    print("✓ Laya runs locally (no API key required)")

    return []


def run_simple_test():
    """Run a simple test with sample data"""
    print("\nRunning simple test...")

    try:
        import pandas as pd
        from laya_eval import ChatbotEvaluator

        # Create sample data
        sample_data = pd.DataFrame([
            {
                'user_query': 'What is 2+2?',
                'actual_response': 'The answer is 4.',
                'expected_response': '2+2 equals 4.',
            }
        ])

        print("✓ Sample data created")

        # Run a simple evaluation test
        evaluator = ChatbotEvaluator(metrics=['response_match'])
        results = evaluator.evaluate(sample_data)
        print(f"✓ Evaluation successful: {results.overall_accuracy:.1%} accuracy")

        return []

    except Exception as e:
        return [f"Test failed: {e}"]


def main():
    """Run all checks"""
    print("=" * 60)
    print("Laya Chatbot Evaluation Framework - Installation Verification")
    print("=" * 60)

    all_errors = []

    all_errors.extend(check_imports())
    all_errors.extend(check_dependencies())
    all_errors.extend(check_environment())
    all_errors.extend(run_simple_test())

    print("\n" + "=" * 60)
    if not all_errors:
        print("✓ All checks passed! Framework is ready to use.")
        print("\nNext steps:")
        print("1. Prepare your evaluation data (CSV/Excel)")
        print("2. Run: python examples/basic_evaluation.py")
    else:
        print("✗ Some checks failed:")
        for error in all_errors:
            print(f"  - {error}")
        sys.exit(1)

    print("=" * 60)


if __name__ == '__main__':
    main()
