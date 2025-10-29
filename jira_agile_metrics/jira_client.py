"""JIRA client utilities for Jira Agile Metrics.

This module provides common utilities for creating and configuring JIRA clients
to reduce code duplication between CLI and webapp modules.
"""

import logging
import os

from jira import JIRA

logger = logging.getLogger(__name__)


def get_jira_connection_params(connection):
    """Extract JIRA connection parameters from connection configuration."""
    # Extract parameters with consistent fallback pattern
    url = connection.get("domain") or os.environ.get("JIRA_URL")
    username = connection.get("username") or os.environ.get("JIRA_USERNAME")
    password = connection.get("password") or os.environ.get("JIRA_PASSWORD")

    # Normalize/strip values
    url = url.strip() if url else None
    username = username.strip() if username else None
    password = password.strip() if password else None

    # Validate that all required parameters are present and non-empty
    missing_params = []
    if not url:
        missing_params.append("url")
    if not username:
        missing_params.append("username")
    if not password:
        missing_params.append("password")

    if missing_params:
        raise ValueError(
            f"Missing required JIRA connection parameters: "
            f"{', '.join(missing_params)}. "
            f"Provide them via connection config or environment variables "
            f"(JIRA_URL, JIRA_USERNAME, JIRA_PASSWORD)."
        )

    return url, username, password


def create_jira_client(connection):
    """Create a JIRA client with the given connection options."""
    url, username, password = get_jira_connection_params(connection)

    jira_client_options = connection["jira_client_options"]

    jira_options = {"server": url, "rest_api_version": 3}
    jira_options.update(jira_client_options)

    try:
        return JIRA(
            options=jira_options,
            basic_auth=(username, password),
        )
    except Exception as e:
        logger.error("Failed to create JIRA client: %s", e)
        raise
