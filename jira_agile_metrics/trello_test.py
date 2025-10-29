"""Tests for Trello integration functionality in Jira Agile Metrics.

This module contains unit tests for Trello API integration and JIRA-like interfaces.
"""

# -*- coding: utf-8 -*-

import pytest

from .trello import (
    JiraLikeFields,
    JiraLikeHistory,
    JiraLikeHistoryItem,
    JiraLikeIssue,
    TrelloClient,
    TrelloConfig,
)

MEMBER = "chrisyoung277"
KEY = "key"
TOKEN = "token"


def test_create(mock_trello_api):
    """Test TrelloClient creation and connection.

    Create a client and connect to Trello
    """

    _ = mock_trello_api
    my_trello = TrelloClient({"member": MEMBER, "key": KEY, "token": TOKEN})
    assert isinstance(my_trello, TrelloClient)


def test_search_issues(mock_trello_api):
    """Test search_issues functionality.

    Get back a jira-like set of issues

    To get the complete history of a Trello card you have to use the Board
    actions rather than the Card Actions. This is because Trello re-writes
    the card history when it transitions from one Board to another for
    security reasons.
    """

    _ = mock_trello_api
    my_trello = TrelloClient({"member": MEMBER, "key": KEY, "token": TOKEN})
    issues = my_trello.search_issues("my_board")
    assert len(issues) == 2


def test_fields(mock_trello_api):
    """
    Get back a list of jira-like fields requied by the calculators
    """
    _ = mock_trello_api
    my_trello = TrelloClient({"member": MEMBER, "key": KEY, "token": TOKEN})
    fields = my_trello.fields()
    assert fields == [
        {"id": "status", "name": "status"},
        {"id": "Flagged", "name": "Flagged"},
    ]


def test_jira_like_history_item():
    """
    The events in the history of a trello card - JIRA style
    """

    my_item = JiraLikeHistoryItem(
        field="status", from_string="Open", to_string="Closed"
    )
    assert isinstance(my_item, JiraLikeHistoryItem)


def test_jira_like_history():
    """
    The history of a trello card - JIRA style
    """

    my_history = JiraLikeHistory(
        created="",
        item=JiraLikeHistoryItem(
            field="status", from_string="Open", to_string="Closed"
        ),
    )
    assert isinstance(my_history, JiraLikeHistory)


def test_jira_like_issue():
    """
    The trello wrapper returns objects which look to the
    query manager like JIRA issues
    """
    my_jira_like_issue = JiraLikeIssue(
        key="some number",
        url="https://some.where",
        fields=None,
        history=JiraLikeHistory(
            created="",
            item=JiraLikeHistoryItem(
                field="status", from_string="Open", to_string="Closed"
            ),
        ),
    )
    assert isinstance(my_jira_like_issue, JiraLikeIssue)


def test_jira_like_fields():
    """
    Wrap up trello card attributes to look like JIRA fields
    """
    my_jira_fields = JiraLikeFields(
        {
            "labels": ["this", "that", "the other"],
            "summary": "Fair to middling",
            "status": "In Progress",
            "created": "foo",
            "issuetype": "hmmm",
        }
    )

    assert isinstance(my_jira_fields, JiraLikeFields)


def test_set_type_from_label(mock_trello_api):
    """
    Allow us to identify a type of work - e.g. Failure demand - from
    a specific label
    """

    _ = mock_trello_api
    my_trello = TrelloClient(
        {
            "member": MEMBER,
            "key": KEY,
            "token": TOKEN,
            "type_mapping": {"defect": ["bug"]},
        }
    )
    issues = my_trello.search_issues("my_board")
    assert issues[1].fields.issuetype.name == "defect"


def test_missing_required_fields():
    """Test that missing required fields raise ValueError."""
    with pytest.raises(
        ValueError, match="Missing required Trello configuration fields"
    ):
        TrelloClient({})


def test_missing_member_field():
    """Test that missing member field raises ValueError."""
    with pytest.raises(
        ValueError, match="Missing required Trello configuration fields: member"
    ):
        TrelloClient({"key": KEY, "token": TOKEN})


def test_missing_key_field():
    """Test that missing key field raises ValueError."""
    with pytest.raises(
        ValueError, match="Missing required Trello configuration fields: key"
    ):
        TrelloClient({"member": MEMBER, "token": TOKEN})


def test_missing_token_field():
    """Test that missing token field raises ValueError."""
    with pytest.raises(
        ValueError, match="Missing required Trello configuration fields: token"
    ):
        TrelloClient({"member": MEMBER, "key": KEY})


def test_empty_string_fields():
    """Test that empty string fields are detected as missing."""
    with pytest.raises(
        ValueError,
        match="Missing required Trello configuration fields: member, key, token",
    ):
        TrelloClient({"member": "", "key": "", "token": ""})


def test_optional_fields(mock_trello_api):
    """Test that type_mapping and flagged_mapping remain optional."""
    _ = mock_trello_api
    my_trello = TrelloClient(
        {
            "member": MEMBER,
            "key": KEY,
            "token": TOKEN,
        }
    )
    assert my_trello.config.type_mapping is None
    assert my_trello.config.flagged_mapping is None


def test_trello_config_from_dict():
    """Test TrelloConfig creation from dictionary."""
    config = TrelloConfig.from_dict(
        {
            "member": MEMBER,
            "key": KEY,
            "token": TOKEN,
            "type_mapping": {"defect": ["bug"]},
        }
    )
    assert config.member == MEMBER
    assert config.key == KEY
    assert config.token == TOKEN
    assert config.type_mapping == {"defect": ["bug"]}
    assert config.flagged_mapping is None


def test_trello_config_validation():
    """Test TrelloConfig validation."""
    with pytest.raises(
        ValueError, match="Missing required Trello configuration fields"
    ):
        TrelloConfig.from_dict({})


def test_trello_config_direct():
    """Test creating TrelloConfig directly."""
    config = TrelloConfig(
        member=MEMBER, key=KEY, token=TOKEN, type_mapping={"defect": ["bug"]}
    )
    assert config.member == MEMBER
    assert config.type_mapping == {"defect": ["bug"]}
