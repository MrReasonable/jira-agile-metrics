"""Test configuration and fixtures for Jira Agile Metrics.

This module provides test fixtures and mock objects for testing the metrics
calculations.
"""

import pytest
import trello
from mock import Mock
from pandas import DataFrame, Timestamp

from .calculators.cfd import CFDCalculator
from .querymanager import QueryManager
from .test_classes import (
    FauxJIRA,
)
from .test_data import (
    COMMON_CFD_COLUMNS,
    COMMON_CFD_DATA,
    COMMON_CYCLE_CONFIG,
)
from .test_data_factory import (
    create_large_cycle_time_results,
    create_minimal_cycle_time_results,
)
from .test_utils import (
    create_common_cycle_status_list,
    create_common_cycle_time_columns,
)
from .utils import extend_dict

# Fake a portion of the JIRA API

# Fixtures


@pytest.fixture(name="base_minimal_settings")
def minimal_settings():
    """The smallest `settings` required to build a query manager and cycle time
    calculation.
    """
    return {
        "attributes": {},
        "known_values": {"Release": ["R1", "R3"]},
        "max_results": None,
        "verbose": False,
        "cycle": COMMON_CYCLE_CONFIG,
        "query_attribute": None,
        "queries": [{"jql": "(filter=123)", "value": None}],
        "backlog_column": "Backlog",
        "committed_column": "Committed",
        "done_column": "Done",
    }


@pytest.fixture(name="base_custom_settings")
def custom_settings(base_minimal_settings):
    """A `settings` dict that uses custom fields and attributes."""
    return extend_dict(
        base_minimal_settings,
        {
            "attributes": {
                "Release": "Releases",
                "Team": "Team",
                "Estimate": "Size",
            },
            "known_values": {"Release": ["R1", "R3"]},
            "progress_report": True,
            "progress_report_outcomes": {},
            "progress_report_outcome_query": None,
            "progress_report_fields": {},
            "progress_report_epic_query_template": "project = TEST AND type = Epic",
            "progress_report_story_query_template": "project = TEST AND type = Story",
            "progress_report_epic_deadline_field": None,
            "progress_report_epic_min_stories_field": None,
            "progress_report_epic_max_stories_field": None,
            "progress_report_epic_team_field": None,
            "progress_report_teams": [],
            "progress_report_outcome_deadline_field": None,
            "progress_report_quantiles": [0.5, 0.85, 0.95],
            "progress_report_cfd_chart": None,
            "progress_report_cfd_chart_title": None,
            "progress_report_burnup_chart": None,
            "progress_report_burnup_chart_title": None,
            "progress_report_burnup_forecast_chart": None,
            "progress_report_burnup_forecast_chart_title": None,
            "progress_report_scatterplot_chart": None,
            "progress_report_scatterplot_chart_title": None,
            "progress_report_histogram_chart": None,
            "progress_report_histogram_chart_title": None,
            "progress_report_throughput_chart": None,
            "progress_report_throughput_chart_title": None,
            "progress_report_wip_chart": None,
            "progress_report_wip_chart_title": None,
            "progress_report_ageing_wip_chart": None,
            "progress_report_ageing_wip_chart_title": None,
            "progress_report_net_flow_chart": None,
            "progress_report_net_flow_chart_title": None,
            "progress_report_impediments_chart": None,
            "progress_report_impediments_chart_title": None,
            "progress_report_defects_chart": None,
            "progress_report_defects_chart_title": None,
            "progress_report_debt_chart": None,
            "progress_report_debt_chart_title": None,
            "progress_report_waste_chart": None,
            "progress_report_waste_chart_title": None,
        },
    )


# Fields + corresponding columns


@pytest.fixture(name="base_minimal_fields")
def minimal_fields():
    """A `fields` list for all basic fields, but no custom fields."""
    return [
        {"id": "summary", "name": "Summary"},
        {"id": "issuetype", "name": "Issue type"},
        {"id": "status", "name": "Status"},
        {"id": "resolution", "name": "Resolution"},
        {"id": "created", "name": "Created date"},
        {"id": "updated", "name": "Updated date"},
        {"id": "project", "name": "Project"},
        {"id": "reporter", "name": "Reporter"},
        {"id": "assignee", "name": "Assignee"},
        {"id": "priority", "name": "Priority"},
        {"id": "type", "name": "Type"},
        {"id": "labels", "name": "Labels"},
        {"id": "components", "name": "Component/s"},
        {"id": "fixVersions", "name": "Fix version/s"},
        {"id": "resolutiondate", "name": "Resolution date"},
        {"id": "customfield_100", "name": "Flagged"},
    ]


@pytest.fixture(name="base_custom_fields")
def custom_fields(base_minimal_fields):
    """A `fields` list with the three custom fields used
    by `custom_settings`"""
    return base_minimal_fields + [
        {"id": "customfield_001", "name": "Team"},
        {"id": "customfield_002", "name": "Size"},
        {"id": "customfield_003", "name": "Releases"},
    ]


@pytest.fixture(name="base_minimal_cycle_time_columns")
def minimal_cycle_time_columns():
    """A columns list for the results of CycleTimeCalculator without any
    custom fields.
    """
    return create_common_cycle_time_columns() + COMMON_CFD_COLUMNS


@pytest.fixture
def custom_cycle_time_columns(_base_minimal_fields):
    """A columns list for the results of CycleTimeCalculator with the three
    custom fields from `custom_settings`.
    """
    return (
        create_common_cycle_time_columns()
        + [
            "Estimate",
            "Release",
            "Team",
        ]
        + create_common_cycle_status_list()
    )


@pytest.fixture(name="base_cfd_columns")
def cfd_columns():
    """A columns list for the results of the CFDCalculator."""
    return COMMON_CFD_COLUMNS


# Query manager


@pytest.fixture
def minimal_query_manager(base_minimal_fields, base_minimal_settings):
    """A minimal query manager (no custom fields)"""
    jira = FauxJIRA(fields=base_minimal_fields, issues=[])
    return QueryManager(jira, base_minimal_settings)


@pytest.fixture
def custom_query_manager(base_custom_fields, base_custom_settings):
    """A query manager capable of returning values for custom fields"""
    jira = FauxJIRA(fields=base_custom_fields, issues=[])
    return QueryManager(jira, base_custom_settings)


# Results object with rich cycle time data


def _ts(datestring, timestring="00:00:00"):
    return Timestamp(f"{datestring} {timestring}")


@pytest.fixture(name="base_minimal_cycle_time_results")
def minimal_cycle_time_results(
    base_minimal_cycle_time_columns,
):
    """A results dict mimicing a minimal
    result from the CycleTimeCalculator."""
    return create_minimal_cycle_time_results(base_minimal_cycle_time_columns)


@pytest.fixture
def large_cycle_time_results(base_minimal_cycle_time_columns):
    """A results dict mimicing a larger result
    from the CycleTimeCalculator."""
    return create_large_cycle_time_results(base_minimal_cycle_time_columns)


@pytest.fixture
def minimal_cfd_results(base_minimal_cycle_time_results, base_cfd_columns):
    """A results dict mimicing a minimal
    result from the CycleTimeCalculator."""
    return extend_dict(
        base_minimal_cycle_time_results,
        {
            CFDCalculator: DataFrame(
                COMMON_CFD_DATA,
                columns=base_cfd_columns,
                index=[
                    _ts("2018-01-01", "00:00:00"),
                    _ts("2018-01-02", "00:00:00"),
                    _ts("2018-01-03", "00:00:00"),
                    _ts("2018-01-04", "00:00:00"),
                    _ts("2018-01-05", "00:00:00"),
                    _ts("2018-01-06", "00:00:00"),
                    _ts("2018-01-07", "00:00:00"),
                ],
            )
        },
    )


@pytest.fixture
def mock_trello_api(mocker):
    """Mock Trello API for testing."""
    trello_api = mocker.patch("jira_agile_metrics.trello.TrelloApi")

    mock_api = Mock(spec=trello.TrelloApi)
    mock_members = Mock(spec=trello.members)
    mock_members.get_board = Mock(return_value=[{"name": "my_board", "id": "my_id"}])
    mock_api.members = mock_members

    mock_cards = Mock(spec=trello.cards)
    mock_cards.get = Mock()
    mock_cards.get.side_effect = [
        {
            "labels": [],
            "pos": 16384,
            "manualCoverAttachment": False,
            "id": "56ae35346b23ea1d6843a67f",
            "badges": {
                "votes": 0,
                "attachments": 0,
                "subscribed": False,
                "due": None,
                "comments": 0,
                "checkItemsChecked": 0,
                "fogbugz": "",
                "viewingMemberVoted": False,
                "checkItems": 0,
                "description": False,
            },
            "idBoard": "56ae35260b361ede7bfbb1ba",
            "idShort": 1,
            "due": None,
            "shortUrl": "https://trello.com/c/J6st5pG8",
            "closed": False,
            "email": "worldofchris+7327776194338bb7c60@boards.trello.com",
            "dateLastActivity": "2016-01-31T16:24:36.264Z",
            "idList": "56ae352bee563becb21b3b82",
            "idLabels": [],
            "idMembers": [],
            "checkItemStates": [],
            "desc": "",
            "descData": None,
            "name": "Card One",
            "url": "https://trello.com/c/J6st5pG8/1-card-one",
            "idAttachmentCover": None,
            "idChecklists": [],
        },
        {
            "labels": [{"name": "bug"}],
            "pos": 16384,
            "manualCoverAttachment": False,
            "id": "56ae35346b23ea1d6843a67a",
            "badges": {
                "votes": 0,
                "attachments": 0,
                "subscribed": False,
                "due": None,
                "comments": 0,
                "checkItemsChecked": 0,
                "fogbugz": "",
                "viewingMemberVoted": False,
                "checkItems": 0,
                "description": False,
            },
            "idBoard": "56ae35260b361ede7bfbb1ba",
            "idShort": 1,
            "due": None,
            "shortUrl": "https://trello.com/c/J6st5pG8",
            "closed": False,
            "email": "worldofchris+559248f3a0cca5aeb0@boards.trello.com",
            "dateLastActivity": "2016-01-31T16:24:36.264Z",
            "idList": "56ae352bee563becb21b3b82",
            "idLabels": [],
            "idMembers": [],
            "checkItemStates": [],
            "desc": "",
            "descData": None,
            "name": "Card One",
            "url": "https://trello.com/c/J6st5pG8/1-card-one",
            "idAttachmentCover": None,
            "idChecklists": [],
        },
    ]

    mock_api.cards = mock_cards

    mock_boards = Mock(spec=trello.boards)
    mock_boards.get_action = Mock()

    mock_boards.get_action.side_effect = [
        [
            {
                "type": "updateCard",
                "idMemberCreator": "559248f3a0cca5aeb0277db6",
                "memberCreator": {
                    "username": "worldofchris",
                    "fullName": "Chris Young",
                    "initials": "CY",
                    "id": "559248f3a0cca5aeb0277db6",
                    "avatarHash": "1171b29b10de82b6a77187b79d8b9a41",
                },
                "date": "2016-01-31T16:24:36.269Z",
                "data": {
                    "listBefore": {
                        "name": "Three",
                        "id": "56ae35296061372e997c0321",
                    },
                    "old": {"idList": "56ae35296061372e997c0321"},
                    "board": {
                        "id": "56ae35260b361ede7bfbb1ba",
                        "name": "API Test 001",
                        "shortLink": "l4YiX1fv",
                    },
                    "card": {
                        "idShort": 1,
                        "id": "56ae35346b23ea1d6843a67f",
                        "name": "Card One",
                        "idList": "56ae352bee563becb21b3b82",
                        "shortLink": "J6st5pG8",
                    },
                    "listAfter": {
                        "name": "Four",
                        "id": "56ae352bee563becb21b3b82",
                    },
                },
                "id": "56ae35444acfaca041099908",
            },
            {
                "type": "moveCardToBoard",
                "idMemberCreator": "559248f3a0cca5aeb0277db6",
                "memberCreator": {
                    "username": "worldofchris",
                    "fullName": "Chris Young",
                    "initials": "CY",
                    "id": "559248f3a0cca5aeb0277db6",
                    "avatarHash": "1171b29b10de82b6a77187b79d8b9a41",
                },
                "date": "2016-01-31T16:24:29.768Z",
                "data": {
                    "boardSource": {
                        "name": "API Test 000",
                        "id": "56ae351097460cd456a5f323",
                    },
                    "list": {
                        "name": "Three",
                        "id": "56ae35296061372e997c0321",
                    },
                    "board": {
                        "id": "56ae35260b361ede7bfbb1ba",
                        "name": "API Test 001",
                        "shortLink": "l4YiX1fv",
                    },
                    "card": {
                        "idShort": 1,
                        "id": "56ae35346b23ea1d6843a67f",
                        "name": "Card One",
                        "shortLink": "J6st5pG8",
                    },
                },
                "id": "56ae353d1fd6686e1baa1d93",
            },
            {
                "type": "createCard",
                "idMemberCreator": "559248f3a0cca5aeb0277db6",
                "memberCreator": {
                    "username": "worldofchris",
                    "fullName": "Chris Young",
                    "initials": "CY",
                    "id": "559248f3a0cca5aeb0277db6",
                    "avatarHash": "1171b29b10de82b6a77187b79d8b9a41",
                },
                "date": "2016-01-31T16:24:20.398Z",
                "data": {
                    "list": {
                        "name": "List One",
                        "id": "56ae3514326fd4436da31bbf",
                    },
                    "board": {
                        "name": "API Test 001",
                        "id": "56ae35260b361ede7bfbb1ba",
                    },
                    "card": {
                        "idShort": 1,
                        "id": "56ae35346b23ea1d6843a67f",
                        "name": "Card One",
                        "shortLink": "J6st5pG8",
                    },
                },
                "id": "56ae35346b23ea1d6843a680",
            },
            {
                "type": "createList",
                "idMemberCreator": "559248f3a0cca5aeb0277db6",
                "memberCreator": {
                    "username": "worldofchris",
                    "fullName": "Chris Young",
                    "initials": "CY",
                    "id": "559248f3a0cca5aeb0277db6",
                    "avatarHash": "1171b29b10de82b6a77187b79d8b9a41",
                },
                "date": "2016-01-31T16:24:11.845Z",
                "data": {
                    "list": {
                        "name": "Four",
                        "id": "56ae352bee563becb21b3b82",
                    },
                    "board": {
                        "id": "56ae35260b361ede7bfbb1ba",
                        "name": "API Test 001",
                        "shortLink": "l4YiX1fv",
                    },
                },
                "id": "56ae352bee563becb21b3b83",
            },
            {
                "type": "createList",
                "idMemberCreator": "559248f3a0cca5aeb0277db6",
                "memberCreator": {
                    "username": "worldofchris",
                    "fullName": "Chris Young",
                    "initials": "CY",
                    "id": "559248f3a0cca5aeb0277db6",
                    "avatarHash": "1171b29b10de82b6a77187b79d8b9a41",
                },
                "date": "2016-01-31T16:24:09.766Z",
                "data": {
                    "list": {
                        "name": "Three",
                        "id": "56ae35296061372e997c0321",
                    },
                    "board": {
                        "id": "56ae35260b361ede7bfbb1ba",
                        "name": "API Test 001",
                        "shortLink": "l4YiX1fv",
                    },
                },
                "id": "56ae35296061372e997c0322",
            },
            {
                "type": "createBoard",
                "idMemberCreator": "559248f3a0cca5aeb0277db6",
                "memberCreator": {
                    "username": "worldofchris",
                    "fullName": "Chris Young",
                    "initials": "CY",
                    "id": "559248f3a0cca5aeb0277db6",
                    "avatarHash": "1171b29b10de82b6a77187b79d8b9a41",
                },
                "date": "2016-01-31T16:24:06.359Z",
                "data": {
                    "board": {
                        "id": "56ae35260b361ede7bfbb1ba",
                        "name": "API Test 001",
                        "shortLink": "l4YiX1fv",
                    }
                },
                "id": "56ae35260b361ede7bfbb1bc",
            },
            {
                "type": "createCard",
                "idMemberCreator": "559248f3a0cca5aeb0277db6",
                "memberCreator": {
                    "username": "worldofchris",
                    "fullName": "Chris Young",
                    "initials": "CY",
                    "id": "559248f3a0cca5aeb0277db6",
                    "avatarHash": "1171b29b10de82b6a77187b79d8b9a41",
                },
                "date": "2016-01-31T16:24:20.398Z",
                "data": {
                    "list": {
                        "name": "List One",
                        "id": "56ae3514326fd4436da31bbf",
                    },
                    "board": {
                        "name": "API Test 001",
                        "id": "56ae35260b361ede7bfbb1ba",
                    },
                    "card": {
                        "idShort": 1,
                        "id": "56ae35346b23ea1d6843a67a",
                        "name": "Card Two",
                        "shortLink": "J6st5pG8",
                    },
                },
                "id": "56ae35346b23ea1d6843a680",
            },
        ],
        [],
    ]

    mock_api.boards = mock_boards

    mock_lists = Mock(spec=trello.lists)
    mock_lists.get = Mock(
        return_value={
            "pos": 131071,
            "idBoard": "56ae35260b361ede7bfbb1ba",
            "id": "56ae352bee563becb21b3b82",
            "closed": False,
            "name": "Four",
        }
    )
    mock_api.lists = mock_lists

    trello_api.return_value = mock_api
