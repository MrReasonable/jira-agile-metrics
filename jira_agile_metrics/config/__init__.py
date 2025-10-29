"""Configuration module for Jira Agile Metrics.

This module provides configuration loading and error handling utilities.
"""

from .exceptions import ConfigError
from .loader import config_to_options

__all__ = ["config_to_options", "ConfigError"]
