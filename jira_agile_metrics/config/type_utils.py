"""Type utilities for configuration processing.

This module provides utilities for type conversion and validation in
configuration files.
"""

import datetime

from .exceptions import ConfigError


def force_list(val) -> list:
    """
    Ensure the value is a list.
    """
    return list(val) if isinstance(val, (list, tuple)) else [val]


def force_int(key, value) -> int:
    """
    Convert value to int, raise ConfigError on failure.
    """
    try:
        return int(value)
    except ValueError:
        raise ConfigError(
            f"Could not convert value `{value}` for key `{expand_key(key)}` to integer"
        ) from None


def force_float(key, value) -> float:
    """
    Convert value to float, raise ConfigError on failure.
    """
    try:
        return float(value)
    except ValueError:
        raise ConfigError(
            f"Could not convert value `{value}` for key `{expand_key(key)}` to decimal"
        ) from None


def force_date(key, value) -> datetime.date:
    """
    Ensure value is a datetime.date, raise ConfigError otherwise.
    """
    if not isinstance(value, datetime.date):
        raise ConfigError(f"Value `{value}` for key `{expand_key(key)}` is not a date")
    return value


def expand_key(key) -> str:
    """
    Expand config key for display.
    """
    return str(key).replace("_", " ").lower()
