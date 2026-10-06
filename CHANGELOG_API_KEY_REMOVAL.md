# API Key Removal Update

## Summary
Updated the Laya AI evaluation framework to reflect that **Laya AI does not require an API key** as confirmed by the official page at https://laya.convaiinnovations.com/. Laya is an open-source library that runs locally.

## Changes Made

### 1. Core Library Files

#### `laya_eval/models/laya_client.py`
- **Removed**: `api_key` parameter from `LayaClient.__init__()`
- **Removed**: `self.api_key` attribute
- **Updated**: Docstring to remove API key documentation

#### `laya_eval/core/evaluator.py`
- **Removed**: `api_key` parameter from `ChatbotEvaluator.__init__()`
- **Updated**: `create_client()` call to not pass `api_key`
- **Updated**: Docstring to remove API key documentation

### 2. Documentation

#### `README.md`
- **Added**: Note at the top clarifying that Laya runs locally and doesn't require an API key
- **Removed**: "Set up API Key" section from Quick Start
- **Removed**: API key troubleshooting section
- **Removed**: `api_key_env` reference from configuration example
- **Updated**: Quick Start section numbering (1→1, 2→2, 3→2)

### 3. Configuration Files

#### `configs/default_config.yaml`
- **Removed**: `api_key_env: 'LAYA_API_KEY'` line from model configuration

### 4. Example Files

#### `examples/basic_evaluation.py`
- **Removed**: API key warning checks in `__main__`
- **Replaced**: Multi-line warning with simple comment: "Laya AI runs locally and does not require an API key"

#### `examples/verify_installation.py`
- **Updated**: `check_environment()` function to show "✓ Laya runs locally (no API key required)" instead of checking for API key
- **Updated**: `run_simple_test()` to actually run evaluation instead of skipping it
- **Updated**: Success message to remove API key setup step

## Impact

### Breaking Changes
- If anyone was passing `api_key` parameter to `LayaClient` or `ChatbotEvaluator`, they'll need to remove it
- Environment variable `LAYA_API_KEY` is no longer referenced

### Benefits
- **Simplified setup**: Users no longer need to find or set an API key
- **Accurate documentation**: Reflects the actual behavior of Laya AI
- **Reduced confusion**: Eliminates misleading API key references
- **Easier onboarding**: New users can start using the framework immediately

## Migration Guide

If you have existing code that passes an API key:

### Before:
```python
from laya_eval import ChatbotEvaluator

evaluator = ChatbotEvaluator(
    model='auto',
    api_key='your-api-key',  # Remove this line
    metrics='all'
)
```

### After:
```python
from laya_eval import ChatbotEvaluator

evaluator = ChatbotEvaluator(
    model='auto',
    metrics='all'
)
```

## Verification

Run the installation verification script to confirm everything works:

```bash
python3 examples/verify_installation.py
```

The script will now show:
```
✓ Laya runs locally (no API key required)
```

Instead of requesting an API key.

## References

- Laya Official Site: https://laya.convaiinnovations.com/
- Laya HuggingFace: https://huggingface.co/convaiinnovations/laya
