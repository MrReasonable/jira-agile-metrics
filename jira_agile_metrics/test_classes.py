"""Test classes for Jira Agile Metrics tests.

This module provides mock classes for testing without circular imports.
"""

import logging
from datetime import datetime

import dateutil.parser

logger = logging.getLogger(__name__)


class FauxFieldValue:
    """A complex field value, with a name and a typed value"""

    def __init__(self, name, value, display_name=None, email_address=None):
        self.name = name
        self.value = value
        self.displayName = (  # pylint: disable=invalid-name
            display_name or name
        )  # Add displayName for JIRA API compatibility
        if email_address:
            self.emailAddress = email_address  # pylint: disable=invalid-name

    def __str__(self):
        """Return string representation."""
        return str(self.value)

    def get_name(self):
        """Get the field name."""
        return self.name

    def get_value(self):
        """Get the field value."""
        return self.value


class FauxChangeItem:
    """An item in a changelog change"""

    def __init__(self, field, from_string, to_string):
        self.field = field
        self.fromString = from_string  # pylint: disable=invalid-name
        self.toString = to_string  # pylint: disable=invalid-name
        self.from_ = self.from_string = from_string  # Keep backward compatibility
        self.to = self.to_string = to_string  # Keep backward compatibility

    def is_status_change(self):
        """Check if this is a status change."""
        return self.field == "status"

    def get_field(self):
        """Get the field name."""
        return self.field

    def get_from_string(self):
        """Get the from string value."""
        return self.from_string


class FauxChange:
    """A change in a changelog. Contains a list of change items."""

    def __init__(self, created, items):
        self.created = created
        self.items = [FauxChangeItem(*i) for i in items]

    def add_item(self, item):
        """Add a change item."""
        self.items.append(item)

    def get_created(self):
        """Get the creation timestamp."""
        return self.created

    def get_items(self):
        """Get the list of change items."""
        return self.items


class FauxChangelog:
    """Mock changelog for testing."""

    def __init__(self, histories):
        self.histories = histories

    def __repr__(self):
        return f"FauxChangelog(histories={len(self.histories)})"

    @staticmethod
    def _normalize_timestamp(timestamp):
        """Normalize a timestamp to a datetime.datetime object.

        Args:
            timestamp: A timestamp value (datetime, ISO string, or other)

        Returns:
            datetime.datetime: Normalized datetime object

        Raises:
            ValueError: If timestamp cannot be converted to datetime
        """
        if isinstance(timestamp, datetime):
            return timestamp
        if isinstance(timestamp, str):
            try:
                return dateutil.parser.parse(timestamp)
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Failed to parse timestamp string '%s': %s", timestamp, e
                )
                raise ValueError(f"Invalid timestamp string: {timestamp}") from e
        raise ValueError(
            f"Invalid timestamp type: {type(timestamp).__name__}, value: {timestamp}"
        )

    def get_latest_change(self):
        """Get the most recent change."""
        if not self.histories:
            return None

        # Normalize and validate all timestamps before comparison
        valid_histories = []
        for history in self.histories:
            try:
                # Validate and cache the normalized timestamp
                normalized_ts = self._normalize_timestamp(history.created)
                valid_histories.append((history, normalized_ts))
            except ValueError as e:
                logger.warning("Skipping history entry with invalid timestamp: %s", e)
                continue

        if not valid_histories:
            logger.warning("No valid history entries found")
            return None

        latest_history, _ = max(valid_histories, key=lambda x: x[1])
        return latest_history

    def get_histories(self):
        """Get the list of histories."""
        return self.histories

    def add_history(self, history):
        """Add a history entry."""
        self.histories.append(history)


class FauxFields:
    """Container for `issue.fields`"""

    def __init__(self, fields):
        self.__dict__.update(fields)

    def get_field_names(self):
        """Get list of field names."""
        return list(self.__dict__.keys())

    def get_field(self, field_name):
        """Get a field value by name."""
        return getattr(self, field_name, None)

    def set_field(self, field_name, value):
        """Set a field value."""
        setattr(self, field_name, value)


class FauxIssue:
    """An issue, with a key, change log, and set of fields"""

    def __init__(self, key, changes, **fields):
        self.key = key
        self.id = f"{key}-id"  # Add missing ID field
        self.self = f"https://example.org/rest/api/2/issue/{key}"  # Add self link
        self.expand = ""  # Add expand field
        self.fields = FauxFields(fields)
        self.changelog = FauxChangelog(changes)

    def get_field_value(self, field_name):
        """Get a field value by name."""
        return getattr(self.fields, field_name, None)

    def get_key(self):
        """Get the issue key."""
        return self.key

    def get_changelog(self):
        """Get the changelog."""
        return self.changelog


class FauxJIRA:
    """JIRA interface. Initialised with a set of issues, which will be returned
    by `search_issues()`.
    """

    def __init__(
        self,
        fields,
        issues,
        options=None,
        filter_func=None,
    ):
        if options is None:
            options = {"server": "https://example.org"}
        self._options = options
        self._fields = fields  # [{ id, name }]
        self._issues = issues
        self._filter = filter_func

    def fields(self):
        """Return the fields configuration."""
        return self._fields

    def search_issues(self, jql, *_args, **_kwargs):
        """Search for issues using JQL query."""
        return (
            self._issues
            if self._filter is None
            else [i for i in self._issues if self._filter(i, jql)]
        )

    def server_url(self):
        """Return the server URL."""
        return self._options.get("server", "https://example.org")
