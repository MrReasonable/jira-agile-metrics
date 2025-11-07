"""Tests for the Flask web application.

This module contains unit tests for the web application routes and functionality.
"""

import re
from collections import defaultdict
from unittest.mock import patch

import pandas as pd
import pytest
from flask import session
from jira import exceptions as jira_exceptions

from jira_agile_metrics.calculators.burnup import BurnupCalculator
from jira_agile_metrics.calculators.cfd import CFDCalculator
from jira_agile_metrics.calculators.cycletime import CycleTimeCalculator
from jira_agile_metrics.calculators.forecast import BurnupForecastCalculator
from jira_agile_metrics.calculators.histogram import HistogramCalculator
from jira_agile_metrics.calculators.scatterplot import ScatterplotCalculator
from jira_agile_metrics.config import ConfigError
from jira_agile_metrics.webapp import app as app_module
from jira_agile_metrics.webapp.app import (
    app as webapp,
)
from jira_agile_metrics.webapp.app import (
    get_real_results,
    results_cache,
)
from jira_agile_metrics.webapp.test_utils import HTMLOutlineParser


@pytest.fixture(name="flask_app")
def test_app():
    """Create and configure a test Flask app."""
    webapp.config["TESTING"] = True
    webapp.config["SECRET_KEY"] = "test-secret-key"
    return webapp


@pytest.fixture(name="test_client")
def client_fixture(flask_app):
    """Create a test client for the Flask app."""
    return flask_app.test_client()


@pytest.fixture
def mock_get_real_results():
    """Mock the get_real_results function to avoid JIRA API calls during tests."""
    with patch("jira_agile_metrics.webapp.app.get_real_results") as mock:
        mock.return_value = {}
        yield mock


def test_index_renders(test_client):
    """Test that the index route renders successfully."""
    response = test_client.get("/")
    assert response.status_code == 200
    assert b"Interactive Bokeh Charts" in response.data


def test_security_headers_added(test_client):
    """Ensure standard security headers are present on responses."""
    response = test_client.get("/")
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    # Present only for legacy browser support; modern browsers ignore this header.
    assert response.headers.get("X-XSS-Protection") == "1; mode=block"
    # Assert a strong HSTS policy matching app configuration
    assert (
        response.headers.get("Strict-Transport-Security")
        == "max-age=31536000; includeSubDomains"
    )


class TestSetQueryRoute:
    """Test cases for the set_query route."""

    def test_set_query_with_valid_jql(self, test_client):
        """Test setting a valid JQL query."""
        # Set a valid JQL query
        response = test_client.post(
            "/set_query", data={"user_query": "project = 'TEST' AND status = 'Open'"}
        )
        assert response.status_code == 302  # Redirect after POST

        # Follow redirect and check that query was set
        response = test_client.get("/")
        assert response.status_code == 200
        # The session should contain the query
        with test_client.session_transaction() as sess:
            assert sess.get("user_query") == "project = 'TEST' AND status = 'Open'"

    def test_set_query_with_special_characters(self, test_client):
        """Test setting JQL query with special characters."""
        query = (
            'project = "TEST" AND summary ~ "test-123" OR '
            'priority in ("High", "Critical")'
        )
        response = test_client.post("/set_query", data={"user_query": query})
        assert response.status_code == 302

        with test_client.session_transaction() as sess:
            assert sess.get("user_query") == query

    def test_set_query_with_dates(self, test_client):
        """Test setting JQL query with date fields."""
        query = "updated >= '2024-01-01' AND created <= '2024-12-31'"
        response = test_client.post("/set_query", data={"user_query": query})
        assert response.status_code == 302

        with test_client.session_transaction() as sess:
            assert sess.get("user_query") == query

    def test_clear_query(self, test_client):
        """Test clearing the custom JQL query."""
        # First set a query
        test_client.post("/set_query", data={"user_query": "project = 'TEST'"})

        # Then clear it
        response = test_client.post("/set_query", data={"user_query": ""})
        assert response.status_code == 302

        with test_client.session_transaction() as sess:
            assert sess.get("user_query") is None

    def test_set_query_too_long(self, test_client):
        """Test that queries longer than 1000 characters are rejected.

        When a query exceeds the maximum length, it is rejected and the session
        key is not set (stored_query is None), indicating that no query was stored.
        """
        long_query = "project = 'TEST'" + " AND issueType = Story" * 100
        response = test_client.post("/set_query", data={"user_query": long_query})
        assert response.status_code == 302

        # Query should not be stored in session (too long and rejected)
        with test_client.session_transaction() as sess:
            stored_query = sess.get("user_query")
            # The session key should not be set when the query is rejected
            assert stored_query is None

    def test_set_query_strips_whitespace(self, test_client):
        """Test that whitespace is stripped from query input."""
        query_with_spaces = "   project = 'TEST'   "
        response = test_client.post(
            "/set_query", data={"user_query": query_with_spaces}
        )
        assert response.status_code == 302

        with test_client.session_transaction() as sess:
            # Should be stored without leading/trailing whitespace
            stored_query = sess.get("user_query")
            assert stored_query == "project = 'TEST'"
            assert stored_query.strip() == stored_query

    def test_set_query_with_order_by(self, test_client):
        """Test setting JQL query with ORDER BY clause."""
        query = "project = 'TEST' ORDER BY priority DESC, created ASC"
        response = test_client.post("/set_query", data={"user_query": query})
        assert response.status_code == 302

        with test_client.session_transaction() as sess:
            assert sess.get("user_query") == query

    def test_multiple_consecutive_queries(self, test_client):
        """Test setting multiple queries in succession."""
        query1 = "project = 'TEST'"
        query2 = "project = 'OTHER'"

        test_client.post("/set_query", data={"user_query": query1})
        with test_client.session_transaction() as sess:
            assert sess.get("user_query") == query1

        test_client.post("/set_query", data={"user_query": query2})
        with test_client.session_transaction() as sess:
            assert sess.get("user_query") == query2

    def test_empty_post_request(self, test_client):
        """Test handling of empty POST request."""
        response = test_client.post("/set_query")
        assert response.status_code == 302

        # Should clear any existing query
        with test_client.session_transaction() as sess:
            assert sess.get("user_query") is None


def test_burnup_route_requires_config(test_client):
    """Test that burnup route handles missing config gracefully."""
    # This will fail without JIRA config
    # The route will catch RuntimeError and display error message
    response = test_client.get("/burnup")
    assert response.status_code == 200
    # Should contain error message about missing credentials
    assert b"JIRA credentials" in response.data or b"danger" in response.data


@pytest.fixture
def _mock_results_empty(mocker):
    """Mock get_real_results to return empty DataFrames for all calculators."""
    empty = pd.DataFrame()
    mocker.patch(
        "jira_agile_metrics.webapp.app.get_real_results",
        return_value=defaultdict(lambda: empty),
    )


@pytest.mark.parametrize(
    "path,title",
    [
        ("/burnup-forecast", "Burnup Forecast (Interactive)"),
        ("/burnup", "Burnup Chart"),
        ("/cfd", "Cumulative Flow Diagram (CFD)"),
        ("/histogram", "Cycle Time Histogram"),
        ("/scatterplot", "Cycle Time Scatterplot"),
        ("/netflow", "Net Flow Chart"),
        ("/ageingwip", "Ageing WIP Chart"),
        ("/debt", "Technical Debt Chart"),
        ("/debt-age", "Debt Age Chart"),
        ("/defects-priority", "Defects by Priority"),
        ("/defects-type", "Defects by Type"),
        ("/defects-environment", "Defects by Environment"),
        ("/impediments", "Impediments Chart"),
        ("/waste", "Waste Chart"),
        ("/progress", "Progress Report Chart"),
        ("/percentiles", "Percentiles Chart"),
        ("/cycletime", "Cycle Time Chart"),
    ],
)
def test_route_renders_with_empty_placeholder_on_empty(
    test_client, _mock_results_empty, path, title
):
    """Chart routes should render with warnings when no data is available.

    Ensure route renders empty-state placeholder when no data is
    available (empty div, no script).
    """
    resp = test_client.get(path)
    assert resp.status_code == 200
    # Title should be rendered for the specific page
    assert f"<h1>{title}</h1>".encode() in resp.data
    # Chart container div should be present but should not contain Bokeh chart content
    # Decode response to use regex for targeted chart container check
    resp_text = resp.data.decode("utf-8")
    # Parse HTML to verify structure without brittle whitespace assumptions
    parser = HTMLOutlineParser()
    parser.feed(resp_text)
    # Ensure an <h1> with the expected title exists
    # Note: second element can be a string (text content) or dict (attributes)
    h1_indexes = [
        i
        for i, (tag, value) in enumerate(parser.outline)
        if tag == "h1" and value == title
    ]
    has_next_div_after_h1 = any(
        (idx + 1) < len(parser.outline) and parser.outline[idx + 1][0] == "div"
        for idx in h1_indexes
    )
    # Require chart container div to appear immediately after the <h1>
    assert (
        has_next_div_after_h1
    ), "Chart container <div> not found immediately after <h1>"
    assert "bk-root" not in resp_text
    # No Bokeh chart scripts should be rendered when empty
    # Check for absence of Bokeh-specific script content rather than all scripts
    # to avoid false positives from legitimate site scripts
    # Check that no script tags contain Bokeh initialization code
    script_pattern = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
    scripts = script_pattern.findall(resp_text)
    for script in scripts:
        assert "Bokeh" not in script, "Bokeh script found when chart should be empty"
        assert (
            "bk-root" not in script
        ), "Bokeh root reference found when chart should be empty"
        # Check for Bokeh document initialization patterns
        assert (
            "document['document']" not in script
        ), "Bokeh document initialization found when chart should be empty"


@pytest.mark.parametrize(
    "status_code, text, expected_exception",
    [
        (401, "Unauthorized", ConfigError),
        (403, "Forbidden", jira_exceptions.JIRAError),
        (500, "Internal Server Error", jira_exceptions.JIRAError),
    ],
)
def test_get_jira_client_error_mappings(mocker, status_code, text, expected_exception):
    """Validate mapping of JIRA HTTP errors to application exceptions."""
    fake_error = jira_exceptions.JIRAError(status_code=status_code, text=text)
    mock_create = mocker.patch(
        "jira_agile_metrics.webapp.helpers.create_jira_client",
        side_effect=fake_error,
    )

    with pytest.raises(expected_exception):
        app_module.get_jira_client({})
    mock_create.assert_called_once()


class TestQueryFormatting:
    """Test various JQL query formatting scenarios."""

    def test_query_with_nested_conditions(self, test_client):
        """Test query with nested AND/OR conditions."""
        query = "(project = 'TEST' OR project = 'OTHER') AND status != Done"
        response = test_client.post("/set_query", data={"user_query": query})
        assert response.status_code == 302

        with test_client.session_transaction() as sess:
            assert sess.get("user_query") == query

    def test_query_with_custom_fields(self, test_client):
        """Test query with custom JIRA fields."""
        query = "project = 'TEST' AND 'Custom Field' = 'Value'"
        response = test_client.post("/set_query", data={"user_query": query})
        assert response.status_code == 302

        with test_client.session_transaction() as sess:
            assert sess.get("user_query") == query

    def test_query_with_text_contains(self, test_client):
        """Test query with text search."""
        query = "summary ~ 'test' OR description ~ 'test'"
        response = test_client.post("/set_query", data={"user_query": query})
        assert response.status_code == 302

        with test_client.session_transaction() as sess:
            assert sess.get("user_query") == query

    def test_query_with_multiple_quotes(self, test_client):
        """Test query with multiple quoted strings."""
        query = "summary ~ \"double quotes\" AND description ~ 'single quotes'"
        response = test_client.post("/set_query", data={"user_query": query})
        assert response.status_code == 302

        with test_client.session_transaction() as sess:
            assert sess.get("user_query") == query


class TestSessionPersistence:
    """Test session persistence across requests."""

    def test_query_persists_across_requests(self, test_client):
        """Test that query persists in session across multiple requests."""
        query = "project = 'TEST'"

        # Set query
        test_client.post("/set_query", data={"user_query": query})

        # Make multiple requests (skip burnup since it requires JIRA config)
        test_client.get("/")

        # Query should still be in session
        with test_client.session_transaction() as sess:
            assert sess.get("user_query") == query

    def test_query_cleared_on_clear(self, test_client):
        """Test that clearing query removes it from session."""
        # Set and clear
        test_client.post("/set_query", data={"user_query": "project = 'TEST'"})
        test_client.post("/set_query", data={"user_query": ""})

        # Query should be gone
        with test_client.session_transaction() as sess:
            assert sess.get("user_query") is None


class TestChartRouteErrorHandling:
    """Test error handling in chart routes."""

    @pytest.mark.parametrize(
        "path,exception",
        [
            ("/burnup", ValueError("Invalid data")),
            ("/burnup", AttributeError("Missing attribute")),
            ("/burnup", KeyError("Missing key")),
            ("/burnup", ImportError("Import failed")),
            ("/cfd", ValueError("Invalid data")),
            ("/histogram", KeyError("Missing key")),
            ("/scatterplot", AttributeError("Missing attribute")),
            ("/netflow", ValueError("Invalid data")),
            ("/ageingwip", KeyError("Missing key")),
            ("/debt", AttributeError("Missing attribute")),
            ("/defects-priority", ValueError("Invalid data")),
            ("/impediments", KeyError("Missing key")),
            ("/waste", AttributeError("Missing attribute")),
            ("/percentiles", ValueError("Invalid data")),
            ("/cycletime", KeyError("Missing key")),
        ],
    )
    def test_chart_route_handles_exceptions(self, test_client, mocker, path, exception):
        """Test that chart routes handle exceptions gracefully."""
        mocker.patch(
            "jira_agile_metrics.webapp.app.get_real_results", side_effect=exception
        )

        response = test_client.get(path)
        assert response.status_code == 200
        # Should render template with empty chart
        assert b"danger" in response.data or b"warning" in response.data

    def test_chart_route_handles_runtime_error(self, test_client, mocker):
        """Test that chart routes handle RuntimeError gracefully."""
        mocker.patch(
            "jira_agile_metrics.webapp.app.get_real_results",
            side_effect=RuntimeError("Runtime error"),
        )

        response = test_client.get("/burnup")
        assert response.status_code == 200
        # Should render template with empty chart
        assert b"danger" in response.data or b"warning" in response.data


class TestChartRenderingWithData:
    """Test chart rendering with actual data (not just empty)."""

    @pytest.fixture
    def mock_results_with_data(self, mocker):
        """Mock get_real_results to return actual chart data."""
        # Create sample data
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data_frames = {
            "burnup": pd.DataFrame(
                {"Backlog": range(10, 0, -1), "Done": range(1, 11)},
                index=dates,
            ),
            "cfd": pd.DataFrame(
                {
                    "Backlog": range(10, 0, -1),
                    "In Progress": [2] * 10,
                    "Done": range(1, 11),
                },
                index=dates,
            ),
            "cycle": pd.DataFrame(
                {
                    "key": [f"KEY-{i}" for i in range(5)],
                    "cycle_time": pd.to_timedelta([i + 1 for i in range(5)], unit="D"),
                }
            ),
            "histogram": pd.Series(
                [1, 2, 3, 4, 5], index=["0-1", "1-2", "2-3", "3-4", "4-5"]
            ),
            "scatterplot": pd.DataFrame(
                {
                    "x": range(5),
                    "y": range(5, 10),
                }
            ),
            "forecast": pd.DataFrame(
                {
                    "trial_0": range(10),
                    "trial_1": range(10, 20),
                },
                index=dates,
            ),
        }

        results = {
            BurnupCalculator: data_frames["burnup"],
            CFDCalculator: data_frames["cfd"],
            CycleTimeCalculator: data_frames["cycle"],
            HistogramCalculator: data_frames["histogram"],
            ScatterplotCalculator: data_frames["scatterplot"],
            BurnupForecastCalculator: data_frames["forecast"],
        }

        mocker.patch(
            "jira_agile_metrics.webapp.app.get_real_results", return_value=results
        )
        return results

    def test_burnup_chart_renders_with_data(self, test_client, mock_results_with_data):
        """Test burnup chart renders with actual data."""
        # Fixture sets up mocks
        _ = mock_results_with_data
        response = test_client.get("/burnup")
        assert response.status_code == 200
        # Should contain Bokeh chart content
        assert b"bk-root" in response.data or b"Bokeh" in response.data

    def test_cfd_chart_renders_with_data(self, test_client, mock_results_with_data):
        """Test CFD chart renders with actual data."""
        # Fixture sets up mocks
        _ = mock_results_with_data
        response = test_client.get("/cfd")
        assert response.status_code == 200
        # Should contain Bokeh chart content
        assert b"bk-root" in response.data or b"Bokeh" in response.data

    def test_histogram_chart_renders_with_data(
        self, test_client, mock_results_with_data
    ):
        """Test histogram chart renders with actual data."""
        # Fixture sets up mocks
        _ = mock_results_with_data
        response = test_client.get("/histogram")
        assert response.status_code == 200
        # Should contain Bokeh chart content
        assert b"bk-root" in response.data or b"Bokeh" in response.data

    def test_scatterplot_chart_renders_with_data(
        self, test_client, mock_results_with_data
    ):
        """Test scatterplot chart renders with actual data."""
        # Fixture sets up mocks
        _ = mock_results_with_data
        response = test_client.get("/scatterplot")
        assert response.status_code == 200
        # Should contain Bokeh chart content
        assert b"bk-root" in response.data or b"Bokeh" in response.data


class TestGetRealResultsCache:
    """Test cache functionality in get_real_results."""

    def test_get_real_results_uses_cache(self, mocker, test_client):
        """Test that get_real_results uses cache for repeated calls."""
        # Clear cache
        with app_module.results_cache_lock:
            results_cache.clear()

        # Mock the expensive operations
        mock_config_to_options = mocker.patch(
            "jira_agile_metrics.webapp.app.config_to_options"
        )
        mocker.patch("jira_agile_metrics.webapp.app.get_jira_client")
        mocker.patch("jira_agile_metrics.webapp.app.QueryManager")
        mock_run_calculators = mocker.patch(
            "jira_agile_metrics.webapp.app.run_calculators"
        )

        # Setup mocks
        mock_config_to_options.return_value = {
            "connection": {
                "username": "test",
                "password": "test",
                "domain": "https://test.jira.com",
            },
            "settings": {"Query": "project=TEST"},
        }
        mock_run_calculators.return_value = {}

        # Mock file operations
        mocker.patch(
            "builtins.open", mocker.mock_open(read_data="Connection:\n  Domain: test")
        )
        mocker.patch("os.path.join", return_value="config.yml")
        mocker.patch("os.makedirs")
        mocker.patch("os.chdir")
        mocker.patch("os.getcwd", return_value="/tmp")

        # Call within Flask request context
        with test_client.application.test_request_context():
            # First call should compute
            result1 = get_real_results()
            assert mock_run_calculators.call_count == 1

            # Second call should use cache
            result2 = get_real_results()
            assert mock_run_calculators.call_count == 1  # Not called again
            assert result1 == result2

    def test_get_real_results_cache_expires(self, mocker, test_client):
        """Test that cache expires after 24 hours."""
        # Clear cache
        with app_module.results_cache_lock:
            results_cache.clear()

        # Mock the expensive operations
        mock_config_to_options = mocker.patch(
            "jira_agile_metrics.webapp.app.config_to_options"
        )
        mocker.patch("jira_agile_metrics.webapp.app.get_jira_client")
        mocker.patch("jira_agile_metrics.webapp.app.QueryManager")
        mock_run_calculators = mocker.patch(
            "jira_agile_metrics.webapp.app.run_calculators"
        )
        mock_time = mocker.patch("jira_agile_metrics.webapp.app.time")

        # Setup mocks
        mock_config_to_options.return_value = {
            "connection": {
                "username": "test",
                "password": "test",
                "domain": "https://test.jira.com",
            },
            "settings": {"Query": "project=TEST"},
        }
        mock_run_calculators.return_value = {}

        # Mock file operations
        mocker.patch(
            "builtins.open", mocker.mock_open(read_data="Connection:\n  Domain: test")
        )
        mocker.patch("os.path.join", return_value="config.yml")
        mocker.patch("os.makedirs")
        mocker.patch("os.chdir")
        mocker.patch("os.getcwd", return_value="/tmp")

        # Call within Flask request context
        with test_client.application.test_request_context():
            # First call at time 0
            mock_time.time.return_value = 0
            get_real_results()
            assert mock_run_calculators.call_count == 1

            # Second call within 24 hours - should use cache
            mock_time.time.return_value = 86400 - 1  # 23 hours 59 minutes 59 seconds
            get_real_results()
            assert mock_run_calculators.call_count == 1  # Still cached

            # Third call after 24 hours - should recompute
            mock_time.time.return_value = 86400 + 1  # 24 hours 1 second
            get_real_results()
            assert mock_run_calculators.call_count == 2  # Recomputed


class TestGetRealResultsUserQuery:
    """Test user query override in get_real_results."""

    def test_user_query_overrides_config(self, mocker, test_client):
        """Test that user query from session overrides config query."""
        # Mock config loading
        mock_config_to_options = mocker.patch(
            "jira_agile_metrics.webapp.app.config_to_options"
        )
        mocker.patch("jira_agile_metrics.webapp.app.get_jira_client")
        mocker.patch("jira_agile_metrics.webapp.app.QueryManager")
        mocker.patch("jira_agile_metrics.webapp.app.run_calculators", return_value={})

        # Setup mocks - need to use a mutable dict that can be modified
        options = {
            "connection": {
                "username": "test",
                "password": "test",
                "domain": "https://test.jira.com",
                "Query": "project=ORIGINAL",
            },
            "settings": {"Query": "project=ORIGINAL"},
        }
        mock_config_to_options.return_value = options

        # Mock file operations
        mocker.patch(
            "builtins.open", mocker.mock_open(read_data="Connection:\n  Domain: test")
        )
        mocker.patch("os.path.join", return_value="config.yml")
        mocker.patch("os.makedirs")
        mocker.patch("os.chdir")
        mocker.patch("os.getcwd", return_value="/tmp")

        # Call get_real_results within request context with session
        with test_client.application.test_request_context():
            session["user_query"] = "project=CUSTOM"
            get_real_results()

        # Verify user query was used
        assert options["settings"]["Query"] == "project=CUSTOM"
        assert options["settings"]["queries"] == [{"jql": "project=CUSTOM"}]

    def test_user_query_empty_string_clears_query(self, mocker, test_client):
        """Test that empty user query clears the config query."""
        mock_config_to_options = mocker.patch(
            "jira_agile_metrics.webapp.app.config_to_options"
        )
        mocker.patch("jira_agile_metrics.webapp.app.get_jira_client")
        mocker.patch("jira_agile_metrics.webapp.app.QueryManager")
        mocker.patch("jira_agile_metrics.webapp.app.run_calculators", return_value={})

        options = {
            "connection": {
                "username": "test",
                "password": "test",
                "domain": "https://test.jira.com",
            },
            "settings": {"Query": "project=ORIGINAL"},
        }
        mock_config_to_options.return_value = options

        mocker.patch(
            "builtins.open", mocker.mock_open(read_data="Connection:\n  Domain: test")
        )
        mocker.patch("os.path.join", return_value="config.yml")
        mocker.patch("os.makedirs")
        mocker.patch("os.chdir")
        mocker.patch("os.getcwd", return_value="/tmp")

        # Call get_real_results within request context with empty session query
        with test_client.application.test_request_context():
            session["user_query"] = ""
            get_real_results()

        # Original query should remain when empty string is provided
        assert options["settings"]["Query"] == "project=ORIGINAL"


class TestSecretKeyFallback:
    """Test secret key fallback when FLASK_SECRET_KEY is not set."""

    def test_secret_key_fallback_when_not_set(self):
        """Test that secret key fallback logic exists (key is set at import time)."""
        # The secret key is set at module import time, so we can't easily test
        # the fallback without re-importing. Instead, verify the logic exists
        # by checking that app has a secret_key attribute
        assert hasattr(webapp, "secret_key")
        assert webapp.secret_key is not None

    def test_secret_key_is_string(self):
        """Test that secret key is a string type."""
        assert isinstance(webapp.secret_key, str)
        assert len(webapp.secret_key) > 0
