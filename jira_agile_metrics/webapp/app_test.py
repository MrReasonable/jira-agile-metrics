"""Tests for the Flask web application.

This module contains unit tests for the web application routes and functionality.
"""

import re
from collections import defaultdict
from unittest.mock import patch

import pandas as pd
import pytest
from jira import exceptions as jira_exceptions

from jira_agile_metrics.config import ConfigError
from jira_agile_metrics.webapp import app as app_module
from jira_agile_metrics.webapp.app import app as webapp
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
    """Test that burnup route tries to load config."""
    # This will fail without JIRA config
    # The route will raise RuntimeError for missing credentials
    with pytest.raises(RuntimeError, match=r"JIRA.*credentials"):
        test_client.get("/burnup")


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
