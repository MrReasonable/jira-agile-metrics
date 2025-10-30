"""Configuration exceptions for Jira Agile Metrics.

This module provides custom exception classes for configuration-related errors.
"""


class ConfigError(Exception):
    """
    Exception raised for errors in the configuration.
    """


class ChartGenerationError(Exception):
    """
    Exception raised for errors during chart generation.

    This exception is used to wrap errors from external libraries
    (matplotlib, pandas, numpy) during chart generation, while allowing
    programming errors (like AttributeError from typos) to propagate.
    """
