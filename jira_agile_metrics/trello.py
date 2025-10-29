"""Trello integration module for Jira Agile Metrics.

This module provides Trello API integration and JIRA-like interfaces for
processing Trello data in the metrics calculations.
"""

import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional

import requests
from trello import TrelloApi

from jira_agile_metrics.utils import retry_with_backoff

logger = logging.getLogger(__name__)


class JiraLikeIssue:
    """JIRA-like issue interface for Trello cards."""

    def __init__(self, key, url, fields, history):
        self.key = key
        self.url = url
        self.id = key
        self.fields = fields
        self.changelog = Changelog(history)

    def get_field_value(self, field_name):
        """Get a field value by name."""
        return getattr(self.fields, field_name, None)

    def __str__(self):
        """Return string representation."""
        return f"JiraLikeIssue(key={self.key})"


class Changelog:
    """JIRA-like changelog interface for Trello card history."""

    def __init__(self, history):
        self.histories = []
        if history is not None:
            self.histories.append(history)

    def append(self, history):
        """Append a history entry to the changelog."""
        self.histories.append(history)

    def sort(self):
        """Sort history entries by creation date."""
        self.histories.sort(key=lambda x: x.created)

    def get_latest_change(self):
        """Get the most recent change."""
        if not self.histories:
            return None
        return max(self.histories, key=lambda x: x.created)

    def __len__(self):
        """Return number of history entries."""
        return len(self.histories)


class JiraLikeHistory:
    """JIRA-like history interface for Trello card changes."""

    def __init__(self, created, item):
        self.items = [item]
        self.created = created

    def add_item(self, item):
        """Add a history item."""
        self.items.append(item)

    def __str__(self):
        """Return string representation."""
        return f"JiraLikeHistory(created={self.created}, items={len(self.items)})"


class JiraLikeHistoryItem:
    """JIRA-like history item interface for Trello card changes."""

    def __init__(self, field, from_string, to_string):
        self.field = field
        self.from_string = from_string
        self.to_string = to_string

    def is_status_change(self):
        """Check if this is a status change."""
        return self.field == "status"

    def __str__(self):
        """Return string representation."""
        return (
            f"JiraLikeHistoryItem(field={self.field}, "
            f"from={self.from_string}, to={self.to_string})"
        )


class JiraLikeField:
    """JIRA-like field interface for Trello card properties."""

    def __init__(self, name):
        self.name = name

    def __str__(self):
        """Return string representation of the field."""
        return str(self.name)

    def __repr__(self):
        """Return detailed string representation."""
        return f"JiraLikeField(name={self.name})"


class JiraLikeFields:
    """JIRA-like fields interface for Trello card properties."""

    def __init__(self, fields_dict):
        """Initialize with a dictionary of fields."""
        self.labels = fields_dict.get("labels", [])
        self.summary = fields_dict.get("summary", "")
        self.status = fields_dict.get("status")
        self.created = fields_dict.get("created", "")
        self.issuetype = fields_dict.get("issuetype", JiraLikeField("card"))
        self.resolution = fields_dict.get("resolution", JiraLikeField("na"))

    def get_field_names(self):
        """Get list of available field names."""
        return ["labels", "summary", "status", "created", "issuetype", "resolution"]

    def __str__(self):
        """Return string representation."""
        return f"JiraLikeFields(summary={self.summary})"


@dataclass
class TrelloConfig:
    """Configuration object for Trello API connection.

    Attributes:
        member: Trello member identifier
        key: Trello API key
        token: Trello API token
        type_mapping: Optional mapping of labels to issue types
        flagged_mapping: Optional mapping for flagged items
    """

    member: str
    key: str
    token: str
    type_mapping: Optional[Dict[str, List[str]]] = None
    flagged_mapping: Optional[Dict] = None

    def __post_init__(self):
        """Validate required fields after initialization."""
        missing_fields = []

        # Validate each required field is non-empty
        for field in ["member", "key", "token"]:
            value = getattr(self, field)
            if not value or (isinstance(value, str) and not value.strip()):
                missing_fields.append(field)

        if missing_fields:
            fields_str = ", ".join(missing_fields)
            raise ValueError(
                f"Missing required Trello configuration fields: {fields_str}"
            )

    @classmethod
    def from_dict(cls, config_dict):
        """Create TrelloConfig from a dictionary.

        Args:
            config_dict: Dictionary containing configuration values

        Returns:
            TrelloConfig instance

        Raises:
            ValueError: If required fields are missing
        """
        return cls(
            member=config_dict.get("member"),
            key=config_dict.get("key"),
            token=config_dict.get("token"),
            type_mapping=config_dict.get("type_mapping"),
            flagged_mapping=config_dict.get("flagged_mapping"),
        )


class TrelloClient:
    """Wrapper around the Trello API.

    Exposes methods that match those provided by the JIRA client.
    """

    def __init__(self, config_dict):
        """Initialize with a configuration dictionary."""
        # Create configuration object with validation
        self.config = TrelloConfig.from_dict(config_dict)

        # Initialize Trello API connection
        self.trello = TrelloApi(self.config.key, token=self.config.token)
        self.boards = self.trello.members.get_board(self.config.member)
        self.from_date = datetime.strptime("2010-01-01", "%Y-%m-%d")

    def fields(self):
        """Return available fields for Trello cards."""
        return [
            {"id": "status", "name": "status"},
            {"id": "Flagged", "name": "Flagged"},
        ]

    def search_issues(self, board_name, _expand=False, _max_results=None):
        """Search for issues (cards) in a Trello board."""
        issues = None
        for board in self.boards:
            if board["name"] == board_name:
                issues = self.issues_from_board_actions(board)
                break
        return issues

    def issues_from_board_actions(self, board):
        """
        Get all the work items in a boards history of actions
        """
        actions = self._fetch_board_actions(board)
        logger.info("%s has %s actions.", board["name"], str(len(actions)))

        work_items = []
        for index, action in enumerate(actions):
            self._process_action(action, work_items)
            logger.info("processed action %s of %s", index, len(actions))

        return work_items

    def _fetch_board_actions(self, board):
        """Fetch all actions for a board."""
        actions = []
        limit = 1000
        filter_list = [
            "createCard",
            "updateCard",
            "moveCardToBoard",
            "moveCardFromBoard",
            "copyCard",
            "copyCommentCard",
        ]
        before = None

        while (before is None) or (before > self.from_date.date()):
            batch = self._get_action_batch(board["id"], limit, filter_list, before)
            actions.extend(batch)
            if len(batch) > 0:
                id_time = int(batch[-1]["id"][0:8], 16)
                before = date.fromtimestamp(id_time)
            else:
                break

        return actions

    @retry_with_backoff(
        max_attempts=3, base_delay=2.0, exceptions=(requests.exceptions.HTTPError,)
    )
    def _get_action_batch(self, board_id, limit, filter_list, before):
        """Get a batch of actions from Trello.

        Args:
            board_id: The Trello board ID
            limit: Maximum number of actions to fetch
            filter_list: List of action types to filter
            before: Fetch actions before this date

        Raises:
            requests.exceptions.HTTPError: If all retry attempts fail
        """
        return self.trello.boards.get_action(
            board_id,
            limit=limit,
            filter=filter_list,
            before=before,
        )

    def _process_action(self, action, work_items):
        """Process a single action."""
        try:
            card_id = action["data"]["card"]["id"]
        except KeyError:
            return

        existing_item = self._find_existing_work_item(work_items, card_id)
        state_transition = self.state_transition(action)

        if existing_item is not None:
            self._update_existing_item(existing_item, state_transition)
        else:
            self._create_new_work_item(card_id, state_transition, work_items)

    def _find_existing_work_item(self, work_items, card_id):
        """Find existing work item by card ID."""
        return next(
            (work_item for work_item in work_items if work_item.id == card_id),
            None,
        )

    def _update_existing_item(self, work_item, state_transition):
        """Update existing work item with state transition."""
        if state_transition is not None:
            work_item.changelog.append(state_transition)
            work_item.changelog.sort()

    def _create_new_work_item(self, card_id, state_transition, work_items):
        """Create a new work item from card data."""
        card = self._get_card(card_id)
        if card is not None:
            work_item = self._build_work_item(card, card_id, state_transition)
            work_items.append(work_item)

    def _get_card(self, card_id):
        """Get card data from Trello with retry logic.

        Args:
            card_id: The Trello card ID to fetch

        Returns:
            Card data dict or None if card not found (404) or max attempts exceeded
        """
        try:
            # Try once to check for 404 before retries
            result = self.trello.cards.get(card_id)
            return result
        except requests.exceptions.HTTPError as exception:
            # Handle 404 errors immediately without retry
            if exception.response.status_code == 404:
                sys.stdout.write("_")
                sys.stdout.flush()
                return None
            # For other errors, retry with exponential backoff
            return self._get_card_with_retry(card_id)

    @retry_with_backoff(
        max_attempts=5,
        base_delay=1.0,
        exceptions=(requests.exceptions.HTTPError,),
        return_on_failure=lambda: None,
    )
    def _get_card_with_retry(self, card_id):
        """Get card data from Trello (internal method with retry)."""
        return self.trello.cards.get(card_id)

    def _build_work_item(self, card, card_id, state_transition):
        """Build a work item from card data."""
        date_created = datetime.fromtimestamp(int(card["id"][0:8], 16)).strftime(
            "%m/%d/%Y, %H:%M:%S"
        )

        card_list = self._get_card_list(card["idList"])
        labels, issuetype = self._process_card_labels(card["labels"])

        return JiraLikeIssue(
            key=card_id,
            url=card["url"],
            fields=JiraLikeFields(
                {
                    "labels": labels,
                    "summary": card["name"],
                    "status": JiraLikeField(card_list["name"]),
                    "created": date_created,
                    "issuetype": JiraLikeField(issuetype),
                }
            ),
            history=state_transition,
        )

    @retry_with_backoff(
        max_attempts=5, base_delay=1.0, exceptions=(requests.exceptions.HTTPError,)
    )
    def _get_card_list(self, list_id):
        """Get card list data from Trello with retry logic.

        Args:
            list_id: The Trello list ID to fetch

        Returns:
            List data dict

        Raises:
            requests.exceptions.HTTPError: If all retry attempts fail
        """
        return self.trello.lists.get(list_id)

    def _process_card_labels(self, labels):
        """Process card labels and determine issue type."""
        processed_labels = []
        issuetype = "card"

        for label in labels:
            clean_label = label["name"].lower().strip()
            if self.config.type_mapping is not None:
                for mapping in self.config.type_mapping:
                    if clean_label in self.config.type_mapping[mapping]:
                        issuetype = mapping
            processed_labels.append(clean_label)

        return processed_labels, issuetype

    def state_transition(self, action):
        """
        Get a state transition from an action
        """
        if action["type"] == "updateCard":
            if "listAfter" in action["data"]:
                to_state = action["data"]["listAfter"]["name"]
            else:
                return None
            from_state = action["data"]["listBefore"]["name"]
        elif action["type"] == "moveCardToBoard":
            list_details = self._get_card_list(action["data"]["list"]["id"])
            to_state = list_details["name"]
            from_state = "undefined"
        elif action["type"] == "moveCardFromBoard":
            to_state = "undefined"
            list_details = self._get_card_list(action["data"]["list"]["id"])
            from_state = list_details["name"]
        elif action["type"] == "createCard":
            from_state = "CREATED"
            list_details = self._get_card_list(action["data"]["list"]["id"])
            to_state = list_details["name"]
        elif action["type"] in [
            "addAttachmentToCard",
            "commentCard",
            "addMemberToCard",
            "updateCheckItemStateOnCard",
            "addChecklistToCard",
            "removeMemberFromCard",
            "deleteCard",
            "deleteAttachmentFromCard",
            "removeChecklistFromCard",
        ]:
            # Do we want to do something different with deleteCard?
            return None
        elif action["type"] in ["copyCard", "copyCommentCard"]:
            # Grab history from previous card and add it to this one?
            return None
        else:
            logger.info("Found Action Type: %s", action["type"])
            return None

        state_transition = JiraLikeHistory(
            action["date"],
            JiraLikeHistoryItem(
                field="status", from_string=from_state, to_string=to_state
            ),
        )

        return state_transition
