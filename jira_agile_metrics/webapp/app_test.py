"""Tests for the Flask web application.

This module contains unit tests for the web application routes and functionality.
"""

from unittest.mock import patch

import pytest

from jira_agile_metrics.webapp.app import app as webapp


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
