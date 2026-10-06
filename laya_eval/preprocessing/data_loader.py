"""
Data Loader for Chatbot Evaluation Data

Handles loading and standardizing CSV/Excel files with various column naming conventions.
Supports flexible schema with automatic column mapping.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    row_count: int
    valid_row_count: int


class DataLoader:
    """
    Load and standardize chatbot evaluation data from CSV/Excel files

    Features:
    - Support CSV, XLSX, XLS formats
    - Flexible column name mapping
    - Missing value detection
    - Schema validation
    """

    # Standard column names (internal representation)
    STANDARD_COLUMNS = {
        'test_id': 'test_id',
        'user_query': 'user_query',
        'actual_response': 'actual_response',
        'expected_response': 'expected_response',
        'context': 'context',
        'session_id': 'session_id',
    }

    # Column name mappings (various conventions → standard names)
    COLUMN_MAPPINGS = {
        # User Query variations
        'user_query': 'user_query',
        'query': 'user_query',
        'user question': 'user_query',
        'user_question': 'user_query',
        'question': 'user_query',
        'input': 'user_query',
        'user input': 'user_query',
        'user_input': 'user_query',

        # Actual Response variations
        'actual_response': 'actual_response',
        'actual bot response': 'actual_response',
        'actual_bot_response': 'actual_response',
        'bot_response': 'actual_response',
        'bot response': 'actual_response',
        'response': 'actual_response',
        'actual': 'actual_response',
        'output': 'actual_response',
        'bot_output': 'actual_response',

        # Expected Response variations
        'expected_response': 'expected_response',
        'expected bot response': 'expected_response',
        'expected_bot_response': 'expected_response',
        'expected': 'expected_response',
        'ground_truth': 'expected_response',
        'ground truth': 'expected_response',
        'reference': 'expected_response',
        'reference_response': 'expected_response',

        # Context variations
        'context': 'context',
        'knowledge_base': 'context',
        'knowledge base': 'context',
        'background': 'context',
        'information': 'context',

        # Test ID variations
        'test_id': 'test_id',
        'test id': 'test_id',
        'id': 'test_id',
        'row_id': 'test_id',
        'case_id': 'test_id',

        # Session ID variations
        'session_id': 'session_id',
        'session id': 'session_id',
        'conversation_id': 'session_id',
    }

    # Required columns (must be present after mapping)
    REQUIRED_COLUMNS = ['user_query', 'actual_response']

    # Optional columns (nice to have)
    OPTIONAL_COLUMNS = ['expected_response', 'context', 'test_id', 'session_id']

    def __init__(self, strict_mode: bool = False):
        """
        Initialize DataLoader

        Args:
            strict_mode: If True, raise errors on validation failures.
                        If False, log warnings and continue with valid rows.
        """
        self.strict_mode = strict_mode

    def load(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load and standardize data from file

        Args:
            file_path: Path to CSV or Excel file

        Returns:
            Standardized DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported or data is invalid
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load based on file extension
        logger.info(f"Loading data from: {file_path}")

        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                "Supported formats: .csv, .xlsx, .xls"
            )

        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

        # Standardize column names
        df = self._standardize_columns(df)

        # Validate schema
        validation = self._validate_schema(df)

        if not validation.is_valid:
            error_msg = f"Data validation failed:\n" + "\n".join(validation.errors)
            if self.strict_mode:
                raise ValueError(error_msg)
            else:
                logger.warning(error_msg)

        if validation.warnings:
            for warning in validation.warnings:
                logger.warning(warning)

        logger.info(
            f"Data loaded successfully: {validation.valid_row_count}/{validation.row_count} valid rows"
        )

        return df

    def load_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Load and standardize existing DataFrame

        Args:
            df: Input DataFrame

        Returns:
            Standardized DataFrame
        """
        logger.info(f"Standardizing DataFrame with {len(df)} rows")

        # Standardize column names
        df = self._standardize_columns(df)

        # Validate schema
        validation = self._validate_schema(df)

        if not validation.is_valid and self.strict_mode:
            raise ValueError(
                f"Data validation failed:\n" + "\n".join(validation.errors)
            )

        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map column variations to standard names

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with standardized column names
        """
        # Create a copy to avoid modifying original
        df = df.copy()

        # Normalize column names (lowercase, strip spaces)
        column_map = {}
        for col in df.columns:
            normalized = col.lower().strip()
            if normalized in self.COLUMN_MAPPINGS:
                standard_name = self.COLUMN_MAPPINGS[normalized]
                column_map[col] = standard_name
                logger.debug(f"Mapping column '{col}' -> '{standard_name}'")

        # Rename columns
        if column_map:
            df = df.rename(columns=column_map)
            logger.info(f"Standardized {len(column_map)} column names")

        return df

    def _validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate DataFrame schema and data quality

        Args:
            df: DataFrame to validate

        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []
        valid_row_count = len(df)

        # Check required columns
        missing_required = [
            col for col in self.REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing_required:
            errors.append(
                f"Missing required columns: {', '.join(missing_required)}"
            )

        # Check for missing optional columns
        missing_optional = [
            col for col in self.OPTIONAL_COLUMNS
            if col not in df.columns
        ]

        if missing_optional:
            warnings.append(
                f"Missing optional columns (may limit some evaluations): "
                f"{', '.join(missing_optional)}"
            )

        # Check for missing values in required columns
        if not errors:  # Only check if required columns exist
            for col in self.REQUIRED_COLUMNS:
                if col in df.columns:
                    missing_count = df[col].isna().sum()
                    if missing_count > 0:
                        warnings.append(
                            f"Column '{col}' has {missing_count} missing values "
                            f"({missing_count/len(df)*100:.1f}%)"
                        )
                        valid_row_count -= missing_count

        # Check for empty strings
        if 'actual_response' in df.columns:
            empty_responses = (df['actual_response'].astype(str).str.strip() == '').sum()
            if empty_responses > 0:
                warnings.append(
                    f"Found {empty_responses} empty actual responses"
                )

        # Check text length (very long texts might cause issues)
        max_length = 10000  # characters
        if 'actual_response' in df.columns:
            long_responses = (df['actual_response'].astype(str).str.len() > max_length).sum()
            if long_responses > 0:
                warnings.append(
                    f"Found {long_responses} responses longer than {max_length} characters"
                )

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            row_count=len(df),
            valid_row_count=max(0, valid_row_count)
        )

    def get_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with missing required data

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with only complete rows
        """
        clean_df = df.dropna(subset=self.REQUIRED_COLUMNS, how='any')

        # Also remove rows with empty strings
        for col in self.REQUIRED_COLUMNS:
            if col in clean_df.columns:
                clean_df = clean_df[clean_df[col].astype(str).str.strip() != '']

        removed = len(df) - len(clean_df)
        if removed > 0:
            logger.info(f"Removed {removed} rows with missing required data")

        return clean_df

    def preview(self, df: pd.DataFrame, n: int = 5) -> Dict:
        """
        Generate a preview summary of the data

        Args:
            df: DataFrame to preview
            n: Number of sample rows to include

        Returns:
            Dictionary with preview information
        """
        preview = {
            'total_rows': len(df),
            'columns': list(df.columns),
            'column_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_values': {
                col: int(df[col].isna().sum())
                for col in df.columns
            },
            'sample_rows': df.head(n).to_dict('records')
        }

        return preview


def load_evaluation_data(
    file_path: Union[str, Path],
    strict_mode: bool = False,
    clean: bool = True
) -> pd.DataFrame:
    """
    Convenience function to load evaluation data

    Args:
        file_path: Path to data file
        strict_mode: If True, raise errors on validation failures
        clean: If True, remove rows with missing data

    Returns:
        Standardized DataFrame ready for evaluation
    """
    loader = DataLoader(strict_mode=strict_mode)
    df = loader.load(file_path)

    if clean:
        df = loader.get_clean_data(df)

    return df
