"""Tests for CLI functionality in Jira Agile Metrics.

This module contains unit tests for the command line interface.
"""

import json
import os
import tempfile

import pytest

from jira_agile_metrics.config import ConfigError

from .cli import (
    _should_use_colors,
    configure_argument_parser,
    get_jira_client,
    get_trello_client,
    override_options,
    run_command_line,
    run_server,
)


def test_override_options():
    """Test override_options functionality."""

    class FauxArgs:
        """Mock arguments class for testing."""

        def __init__(self, opts):
            self.__dict__.update(opts)
            for k, v in opts.items():
                setattr(self, k, v)

        def __repr__(self):
            """String representation."""
            return f"FauxArgs({self.__dict__})"

        def get(self, key, default=None):
            """Get attribute value."""
            return getattr(self, key, default)

    options = {"one": 1, "two": 2}
    override_options(options, FauxArgs({}))
    assert json.dumps(options) == json.dumps({"one": 1, "two": 2})

    options = {"one": 1, "two": 2}
    override_options(options, FauxArgs({"one": 11}))
    assert json.dumps(options) == json.dumps({"one": 11, "two": 2})

    options = {"one": 1, "two": 2}
    override_options(options, FauxArgs({"three": 3}))
    assert json.dumps(options) == json.dumps({"one": 1, "two": 2})


def test_run_command_line_with_trello_client(mocker):
    """Test run_command_line with Trello client."""
    config = """
Connection:
  Type: trello

Query: project = "JLF"

Workflow:
  Backlog: Open
  In Progress:
    - In Progress
    - Reopened
  Done:
    - Resolved
    - Closed

Output:

    # CSV files with raw data for input to other
    # tools or further analysis in a spreadsheet
    # If you use .json or .xlsx as the extension,
    # you can get JSON data files or Excel
    # spreadsheets instead

    Cycle time data:
        - cycletime.csv
        - cycletime.json
    CFD data: cfd.csv
"""
    mock_get_trello_client = mocker.patch("jira_agile_metrics.cli.get_trello_client")
    mocker.patch("jira_agile_metrics.cli.QueryManager")
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as config_file:
        config_file.write(config)
        config_file.flush()
        parser = configure_argument_parser()
        # Ensure CSV outputs go to a temporary directory
        # (avoid writing to repo root)
        with tempfile.TemporaryDirectory() as outdir:
            args = parser.parse_args([config_file.name, "--output-directory", outdir])
            run_command_line(parser, args)
    mock_get_trello_client.assert_called_once()


def test_get_trello_client(mocker):
    """Test get_trello_client functionality."""
    mock_trello = mocker.patch("jira_agile_metrics.cli.TrelloClient")

    get_trello_client({"username": "me", "key": "my_key", "token": "my_token"}, {})

    mock_trello.assert_called_once()


class TestServerMode:
    """Test server mode functionality."""

    def test_run_server_with_host_and_port(self, mocker):
        """Test run_server with both host and port specified."""
        mock_set_chart_context = mocker.patch(
            "jira_agile_metrics.cli.set_chart_context"
        )
        mock_webapp_run = mocker.patch("jira_agile_metrics.cli.webapp.run")

        parser = configure_argument_parser()
        args = parser.parse_args(["--server", "127.0.0.1:8080"])

        run_server(parser, args)

        mock_set_chart_context.assert_called_once_with("paper")
        mock_webapp_run.assert_called_once_with(host="127.0.0.1", port=8080)

    def test_run_server_with_port_only(self, mocker):
        """Test run_server with only port specified."""
        mock_set_chart_context = mocker.patch(
            "jira_agile_metrics.cli.set_chart_context"
        )
        mock_webapp_run = mocker.patch("jira_agile_metrics.cli.webapp.run")

        parser = configure_argument_parser()
        args = parser.parse_args(["--server", "5000"])

        run_server(parser, args)

        mock_set_chart_context.assert_called_once_with("paper")
        mock_webapp_run.assert_called_once_with(host=None, port=5000)


class TestErrorHandling:
    """Test error handling in CLI."""

    def test_run_command_line_with_missing_config_file(self):
        """Test run_command_line handles FileNotFoundError gracefully."""
        parser = configure_argument_parser()
        args = parser.parse_args(["nonexistent.yml"])

        # Should not raise exception, but return early
        run_command_line(parser, args)

    def test_run_command_line_with_config_error(self, mocker):
        """Test run_command_line handles ConfigError."""
        parser = configure_argument_parser()
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as config_file:
            config_file.write("Invalid: YAML: [")
            config_file.flush()
            # Use temporary output directory to avoid writing to project root
            with tempfile.TemporaryDirectory() as outdir:
                args = parser.parse_args(
                    [
                        config_file.name,
                        "--output-directory",
                        outdir,
                    ]
                )

                # Mock config_to_options to raise ConfigError
                mocker.patch(
                    "jira_agile_metrics.cli.config_to_options",
                    side_effect=ConfigError("Invalid config"),
                )

                # Should handle error gracefully
                with pytest.raises(ConfigError):
                    run_command_line(parser, args)

            os.unlink(config_file.name)

    def test_run_command_line_with_unknown_source(self, mocker):
        """Test run_command_line handles unknown source type."""
        config = """
Connection:
  Type: unknown

Query: project = "TEST"
"""
        parser = configure_argument_parser()
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as config_file:
            config_file.write(config)
            config_file.flush()
            # Use temporary output directory to avoid writing to project root
            with tempfile.TemporaryDirectory() as outdir:
                args = parser.parse_args(
                    [
                        config_file.name,
                        "--output-directory",
                        outdir,
                    ]
                )

                mocker.patch("jira_agile_metrics.cli.config_to_options")
                mocker.patch("jira_agile_metrics.cli.override_options")

                # Should raise ConfigError for unknown source
                with pytest.raises(ConfigError, match="Unknown source"):
                    run_command_line(parser, args)

            os.unlink(config_file.name)


class TestColorLogging:
    """Test color logging functionality."""

    def test_should_use_colors_without_colorlog(self, mocker):
        """Test _should_use_colors returns False when colorlog unavailable."""
        mocker.patch("jira_agile_metrics.cli.COLORLOG_AVAILABLE", False)
        assert _should_use_colors() is False

    def test_should_use_colors_with_force_color(self, mocker):
        """Test _should_use_colors respects FORCE_COLOR environment variable."""
        mocker.patch("jira_agile_metrics.cli.COLORLOG_AVAILABLE", True)
        mocker.patch.dict(os.environ, {"FORCE_COLOR": "1"})
        assert _should_use_colors() is True

        mocker.patch.dict(os.environ, {"FORCE_COLOR": "true"})
        assert _should_use_colors() is True

        mocker.patch.dict(os.environ, {"FORCE_COLOR": "yes"})
        assert _should_use_colors() is True

    def test_should_use_colors_with_tty(self, mocker):
        """Test _should_use_colors detects TTY."""
        mocker.patch("jira_agile_metrics.cli.COLORLOG_AVAILABLE", True)
        mocker.patch.dict(os.environ, {}, clear=True)

        # Mock stdout.isatty to return True
        mock_stdout = mocker.Mock()
        mock_stdout.isatty.return_value = True
        mocker.patch("sys.stdout", mock_stdout)

        assert _should_use_colors() is True

    def test_should_use_colors_without_tty(self, mocker):
        """Test _should_use_colors returns False when not TTY."""
        mocker.patch("jira_agile_metrics.cli.COLORLOG_AVAILABLE", True)
        mocker.patch.dict(os.environ, {}, clear=True)

        # Mock stdout.isatty to return False
        mock_stdout = mocker.Mock()
        mock_stdout.isatty.return_value = False
        mocker.patch("sys.stdout", mock_stdout)

        assert _should_use_colors() is False


class TestJiraClientCreation:
    """Test JIRA client creation."""

    def test_get_jira_client_with_proxies(self, mocker):
        """Test get_jira_client creates client with proxy configuration."""
        mock_jira = mocker.patch("jira_agile_metrics.cli.JIRA")
        _mock_get_connection_params = mocker.patch(
            "jira_agile_metrics.cli.get_jira_connection_params",
            return_value=("https://test.jira.com", "user", "pass"),
        )

        connection = {
            "http_proxy": "http://proxy.local:8080",
            "https_proxy": "https://proxy.local:8080",
            "jira_server_version_check": True,
            "jira_client_options": {},
        }

        get_jira_client(connection)

        # Verify JIRA was called with proxies
        mock_jira.assert_called_once()
        call_kwargs = mock_jira.call_args[1]
        assert "proxies" in call_kwargs
        assert call_kwargs["proxies"]["http"] == "http://proxy.local:8080"
        assert call_kwargs["proxies"]["https"] == "https://proxy.local:8080"

    def test_get_jira_client_prompts_for_credentials(self, mocker):
        """Test get_jira_client prompts for missing credentials."""
        mock_jira = mocker.patch("jira_agile_metrics.cli.JIRA")
        _mock_get_connection_params = mocker.patch(
            "jira_agile_metrics.cli.get_jira_connection_params",
            return_value=("https://test.jira.com", None, None),
        )
        mock_input = mocker.patch("builtins.input", return_value="testuser")
        mock_getpass = mocker.patch(
            "jira_agile_metrics.cli.getpass.getpass", return_value="testpass"
        )

        connection = {
            "http_proxy": None,
            "https_proxy": None,
            "jira_server_version_check": False,
            "jira_client_options": {},
        }

        get_jira_client(connection)

        # Verify prompts were called
        mock_input.assert_called_once()
        mock_getpass.assert_called_once()
        mock_jira.assert_called_once()

    def test_get_jira_client_with_jira_client_options(self, mocker):
        """Test get_jira_client uses jira_client_options."""
        mock_jira = mocker.patch("jira_agile_metrics.cli.JIRA")
        _mock_get_connection_params = mocker.patch(
            "jira_agile_metrics.cli.get_jira_connection_params",
            return_value=("https://test.jira.com", "user", "pass"),
        )

        connection = {
            "http_proxy": None,
            "https_proxy": None,
            "jira_server_version_check": True,
            "jira_client_options": {"timeout": 30, "max_retries": 3},
        }

        get_jira_client(connection)

        # Verify JIRA was called with options
        mock_jira.assert_called_once()
        call_args = mock_jira.call_args[0]
        assert call_args[0]["timeout"] == 30
        assert call_args[0]["max_retries"] == 3


class TestTrelloClientCreation:
    """Test Trello client creation."""

    def test_get_trello_client_prompts_for_credentials(self, mocker):
        """Test get_trello_client prompts for missing credentials."""
        mock_trello = mocker.patch("jira_agile_metrics.cli.TrelloClient")
        mock_input = mocker.patch("builtins.input", return_value="testuser")
        mock_getpass = mocker.patch("jira_agile_metrics.cli.getpass.getpass")
        mock_getpass.side_effect = ["testkey", "testtoken"]

        connection = {
            "username": None,
            "key": None,
            "token": None,
        }

        get_trello_client(connection, {})

        # Verify prompts were called
        assert mock_input.call_count == 1
        assert mock_getpass.call_count == 2
        mock_trello.assert_called_once()

    def test_get_trello_client_with_credentials(self, mocker):
        """Test get_trello_client with provided credentials."""
        mock_trello = mocker.patch("jira_agile_metrics.cli.TrelloClient")

        connection = {
            "username": "testuser",
            "key": "testkey",
            "token": "testtoken",
        }

        get_trello_client(connection, {})

        # Verify client was created without prompting
        mock_trello.assert_called_once()


class TestCommandLineArguments:
    """Test command line argument parsing and handling."""

    def test_configure_argument_parser_creates_parser(self):
        """Test that configure_argument_parser creates a valid parser."""
        parser = configure_argument_parser()
        assert parser is not None

    def test_parser_accepts_config_file(self):
        """Test parser accepts config file argument."""
        parser = configure_argument_parser()
        args = parser.parse_args(["config.yml"])
        assert args.config == "config.yml"

    def test_parser_accepts_verbose_flags(self):
        """Test parser accepts verbose flags."""
        parser = configure_argument_parser()
        args = parser.parse_args(["-v", "config.yml"])
        assert args.verbose is True
        assert args.very_verbose is False

        args = parser.parse_args(["-vv", "config.yml"])
        assert args.very_verbose is True

    def test_parser_accepts_max_results(self):
        """Test parser accepts max_results option."""
        parser = configure_argument_parser()
        args = parser.parse_args(["-n", "20", "config.yml"])
        assert args.max_results == 20

    def test_parser_accepts_output_directory(self):
        """Test parser accepts output directory option."""
        parser = configure_argument_parser()
        args = parser.parse_args(["--output-directory", "/tmp/output", "config.yml"])
        assert args.output_directory == "/tmp/output"

    def test_parser_accepts_connection_options(self):
        """Test parser accepts connection options."""
        parser = configure_argument_parser()
        args = parser.parse_args(
            [
                "--domain",
                "https://test.jira.com",
                "--username",
                "user",
                "--password",
                "pass",
                "config.yml",
            ]
        )
        assert args.domain == "https://test.jira.com"
        assert args.username == "user"
        assert args.password == "pass"

    def test_parser_accepts_proxy_options(self):
        """Test parser accepts proxy options."""
        parser = configure_argument_parser()
        args = parser.parse_args(
            [
                "--http-proxy",
                "http://proxy.local:8080",
                "--https-proxy",
                "https://proxy.local:8080",
                "config.yml",
            ]
        )
        assert args.http_proxy == "http://proxy.local:8080"
        assert args.https_proxy == "https://proxy.local:8080"
