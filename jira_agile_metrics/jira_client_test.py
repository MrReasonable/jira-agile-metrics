"""Tests for JIRA client utilities."""

import os
from unittest.mock import Mock, patch

import pytest

from .jira_client import create_jira_client, get_jira_connection_params


class TestGetJiraConnectionParams:
    """Test cases for get_jira_connection_params function."""

    def test_get_params_from_config(self):
        """Test extracting parameters from connection config."""
        connection = {
            "domain": "https://jira.example.com",
            "username": "testuser",
            "password": "testpass",
            "jira_client_options": {},
        }

        url, username, password = get_jira_connection_params(connection)

        assert url == "https://jira.example.com"
        assert username == "testuser"
        assert password == "testpass"

    def test_get_params_from_env_vars(self):
        """Test extracting parameters from environment variables."""
        connection = {
            "domain": None,
            "username": None,
            "password": None,
            "jira_client_options": {},
        }

        with patch.dict(
            os.environ,
            {
                "JIRA_URL": "https://jira.example.com",
                "JIRA_USERNAME": "envuser",
                "JIRA_PASSWORD": "envpass",
            },
        ):
            url, username, password = get_jira_connection_params(connection)

        assert url == "https://jira.example.com"
        assert username == "envuser"
        assert password == "envpass"

    def test_get_params_mixed_config_and_env(self):
        """Test extracting parameters from both config and env vars."""
        connection = {
            "domain": "https://jira.example.com",
            "username": None,
            "password": None,
            "jira_client_options": {},
        }

        with patch.dict(
            os.environ,
            {
                "JIRA_USERNAME": "envuser",
                "JIRA_PASSWORD": "envpass",
            },
        ):
            url, username, password = get_jira_connection_params(connection)

        assert url == "https://jira.example.com"
        assert username == "envuser"
        assert password == "envpass"

    def test_get_params_strips_whitespace(self):
        """Test that parameters are stripped of whitespace."""
        connection = {
            "domain": "  https://jira.example.com  ",
            "username": "  testuser  ",
            "password": "  testpass  ",
            "jira_client_options": {},
        }

        url, username, password = get_jira_connection_params(connection)

        assert url == "https://jira.example.com"
        assert username == "testuser"
        assert password == "testpass"

    def test_get_params_missing_url(self):
        """Test error when URL is missing."""
        connection = {
            "domain": None,
            "username": "testuser",
            "password": "testpass",
            "jira_client_options": {},
        }

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError,
                match="Missing required JIRA connection parameters: url",
            ):
                get_jira_connection_params(connection)

    def test_get_params_missing_username(self):
        """Test error when username is missing."""
        connection = {
            "domain": "https://jira.example.com",
            "username": None,
            "password": "testpass",
            "jira_client_options": {},
        }

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError,
                match="Missing required JIRA connection parameters: username",
            ):
                get_jira_connection_params(connection)

    def test_get_params_missing_password(self):
        """Test error when password is missing."""
        connection = {
            "domain": "https://jira.example.com",
            "username": "testuser",
            "password": None,
            "jira_client_options": {},
        }

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError,
                match="Missing required JIRA connection parameters: password",
            ):
                get_jira_connection_params(connection)

    def test_get_params_multiple_missing(self):
        """Test error when multiple parameters are missing."""
        connection = {
            "domain": None,
            "username": None,
            "password": None,
            "jira_client_options": {},
        }

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError,
                match=(
                    "Missing required JIRA connection parameters: "
                    "url, username, password"
                ),
            ):
                get_jira_connection_params(connection)

    def test_get_params_empty_strings(self):
        """Test error when parameters are empty strings."""
        connection = {
            "domain": "",
            "username": "",
            "password": "",
            "jira_client_options": {},
        }

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError,
                match=(
                    "Missing required JIRA connection parameters: "
                    "url, username, password"
                ),
            ):
                get_jira_connection_params(connection)

    def test_get_params_whitespace_only_strings(self):
        """Test error when parameters are whitespace only."""
        connection = {
            "domain": "   ",
            "username": "   ",
            "password": "   ",
            "jira_client_options": {},
        }

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError,
                match=(
                    "Missing required JIRA connection parameters: "
                    "url, username, password"
                ),
            ):
                get_jira_connection_params(connection)


class TestCreateJiraClient:
    """Test cases for create_jira_client function."""

    @patch("jira_agile_metrics.jira_client.JIRA")
    def test_create_client_success(self, mock_jira_class):
        """Test successful JIRA client creation."""
        mock_jira_instance = Mock()
        mock_jira_class.return_value = mock_jira_instance

        connection = {
            "domain": "https://jira.example.com",
            "username": "testuser",
            "password": "testpass",
            "jira_client_options": {"verify": True},
        }

        result = create_jira_client(connection)

        mock_jira_class.assert_called_once()
        call_kwargs = mock_jira_class.call_args[1]
        call_options = call_kwargs["options"]
        assert call_kwargs["basic_auth"] == ("testuser", "testpass")
        assert call_options["server"] == "https://jira.example.com"
        assert call_options["rest_api_version"] == 3
        assert call_options["verify"] is True
        assert result == mock_jira_instance

    @patch("jira_agile_metrics.jira_client.JIRA")
    def test_create_client_with_custom_options(self, mock_jira_class):
        """Test JIRA client creation with custom options."""
        mock_jira_instance = Mock()
        mock_jira_class.return_value = mock_jira_instance

        connection = {
            "domain": "https://jira.example.com",
            "username": "testuser",
            "password": "testpass",
            "jira_client_options": {
                "verify": False,
                "timeout": 30,
                "max_retries": 3,
            },
        }

        result = create_jira_client(connection)

        call_kwargs = mock_jira_class.call_args[1]
        call_options = call_kwargs["options"]
        assert call_options["verify"] is False
        assert call_options["timeout"] == 30
        assert call_options["max_retries"] == 3
        assert result == mock_jira_instance

    @patch("jira_agile_metrics.jira_client.JIRA")
    def test_create_client_overrides_api_version(self, mock_jira_class):
        """Test that rest_api_version can be overridden by custom options."""
        mock_jira_instance = Mock()
        mock_jira_class.return_value = mock_jira_instance

        connection = {
            "domain": "https://jira.example.com",
            "username": "testuser",
            "password": "testpass",
            "jira_client_options": {"rest_api_version": 2},
        }

        result = create_jira_client(connection)

        call_kwargs = mock_jira_class.call_args[1]
        call_options = call_kwargs["options"]
        # Custom option should override default
        assert call_options["rest_api_version"] == 2
        assert result == mock_jira_instance

    @patch("jira_agile_metrics.jira_client.JIRA")
    @patch("jira_agile_metrics.jira_client.logger")
    def test_create_client_exception_handling(self, mock_logger, mock_jira_class):
        """Test exception handling during client creation."""
        error = Exception("Connection failed")
        mock_jira_class.side_effect = error

        connection = {
            "domain": "https://jira.example.com",
            "username": "testuser",
            "password": "testpass",
            "jira_client_options": {},
        }

        with pytest.raises(Exception, match="Connection failed"):
            create_jira_client(connection)

        mock_logger.error.assert_called_once()
        assert "Failed to create JIRA client" in str(mock_logger.error.call_args[0][0])

    @patch("jira_agile_metrics.jira_client.get_jira_connection_params")
    @patch("jira_agile_metrics.jira_client.JIRA")
    def test_create_client_calls_get_params(self, mock_jira_class, mock_get_params):
        """Test that create_jira_client calls get_jira_connection_params."""
        mock_jira_instance = Mock()
        mock_jira_class.return_value = mock_jira_instance
        mock_get_params.return_value = (
            "https://jira.example.com",
            "testuser",
            "testpass",
        )

        connection = {
            "domain": "https://jira.example.com",
            "username": "testuser",
            "password": "testpass",
            "jira_client_options": {},
        }

        create_jira_client(connection)

        mock_get_params.assert_called_once_with(connection)
