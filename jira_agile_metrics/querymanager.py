"""Query management module for Jira Agile Metrics.

This module handles querying JIRA for issue data and managing field mappings
and attribute resolution.
"""

import itertools
import json
import logging

import dateutil.parser
from jira.exceptions import JIRAError

from .config import ConfigError

logger = logging.getLogger(__name__)


def multi_getattr(obj, attr, **kw):
    """Get nested attribute from object using dot notation.

    Args:
        obj: The object to get attributes from
        attr: Dot-separated attribute path (e.g., 'field.subfield')
        **kw: Keyword arguments, including 'default' for fallback value

    Returns:
        The attribute value or default if specified

    Raises:
        AttributeError: If attribute doesn't exist and no default provided
    """
    attributes = attr.split(".")
    for i in attributes:
        try:
            obj = getattr(obj, i)
            if callable(obj):
                obj = obj()
        except AttributeError:
            logger.info("Not able to get data")

            if "default" in kw:
                return kw["default"]
            raise
    return obj


class IssueSnapshot:
    """A snapshot of the key fields of an issue
    at a point in its change history"""

    def __init__(self, change, transition_data):
        """Initialize ChangeItem with change and transition data.

        Args:
            change: The change object
            transition_data: Dictionary containing key, date, from_string, to_string
        """
        self.change = change
        self.key = transition_data["key"]
        self.date = transition_data["date"]
        self.from_string = transition_data["from_string"]
        self.to_string = transition_data["to_string"]

    def __eq__(self, other):
        return all(
            (
                self.change == other.change,
                self.key == other.key,
                self.date.isoformat() == other.date.isoformat(),
                self.from_string == other.from_string,
                self.to_string == other.to_string,
            )
        )

    def __repr__(self):
        return (
            f"<IssueSnapshot change={self.change} key={self.key} "
            f"date={self.date.isoformat()} from={self.from_string} "
            f"to={self.to_string}>"
        )


class QueryManager:
    """Manage and execute queries"""

    settings = {
        "attributes": {},
        "known_values": {},
        "max_results": False,
    }

    def __init__(self, jira, settings):
        self.jira = jira
        self.settings = self.settings.copy()
        self.settings.update(settings)

        self.attributes_to_fields = {}
        self.fields_to_attributes = {}

        # Look up fields in JIRA and resolve attributes to fields
        logger.debug("Resolving JIRA fields")
        self.jira_fields = self.jira.fields()

        if len(self.jira_fields) == 0:
            raise ConfigError(
                (
                    "No field data retrieved from JIRA. "
                    "This likely means a problem with the JIRA API."
                )
            ) from None

        self.jira_fields_to_names = {
            field["id"]: field["name"] for field in self.jira_fields
        }
        field_id = None

        for name, field in self.settings["attributes"].items():
            field_id = self.field_name_to_id(field)
            self.attributes_to_fields[name] = field_id
            self.fields_to_attributes[field_id] = name

    def _get_tolerant_attr(self, obj, camel_case_name, snake_case_name, default=None):
        """Get attribute from object, trying camelCase first, then snake_case.

        This helper supports both real JIRA PropertyHolder attributes (camelCase)
        and test aliases (snake_case).

        Args:
            obj: The object to get attributes from
            camel_case_name: The camelCase attribute name (e.g., "fromString")
            snake_case_name: The snake_case attribute name (e.g., "from_string")
            default: Default value if neither attribute exists (default: None)

        Returns:
            The attribute value or default if neither exists
        """
        return getattr(obj, camel_case_name, getattr(obj, snake_case_name, default))

    def field_name_to_id(self, name):
        """Convert field name to JIRA field ID.

        Args:
            name: The field name to convert

        Returns:
            The JIRA field ID

        Raises:
            ConfigError: If field name doesn't exist in JIRA
        """
        arr_name = name.split(".")
        append_text = ("." + ".".join(arr_name[1:])) if len(arr_name) > 1 else ""
        try:
            return (
                next(
                    (
                        f["id"]
                        for f in self.jira_fields
                        if f["name"].lower() == name.lower()
                    )
                )
                + append_text
            )
        except StopIteration:
            # Note: Field lookup may fail if JIRA field names don't match exactly
            # This can happen due to case sensitivity or field name variations
            logger.debug(
                "Failed to look up %s in JIRA fields: %s",
                name,
                json.dumps(self.jira_fields),
            )

            raise ConfigError(
                f"JIRA field with name `{name}` does not exist "
                f"(did you try to use the field id instead?)"
            ) from None

    def resolve_attribute_value(self, issue, attribute_name):
        """Given an attribute name (i.e. one named in the config file and
        mapped to a field in JIRA), return its value from the given issue.
        Respects the `Known Values` settings and tries to resolve complex
        data types.
        """
        field_id = self.attributes_to_fields[attribute_name]
        return self.resolve_field_value(issue, field_id)

    def resolve_field_value(self, issue, field_id):
        """Given a JIRA internal field id, return its value from the given
        issue. Respects the `Known Values` settings and tries to resolve
        complex data types.
        """

        try:
            field_value = multi_getattr(issue.fields, field_id)

        except AttributeError:
            field_name = self.jira_fields_to_names.get(field_id, "Unknown name")
            logger.debug(
                "Could not get field value for field %s. "
                "Probably this is a wrong workflow field mapping",
                field_name,
            )
            field_value = None

        if field_value is None:
            return None

        value = getattr(field_value, "value", field_value)

        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                value = None
            else:
                values = [getattr(v, "name", v) for v in value]

                # is this a `Known Values` attribute?
                attribute_name = self.fields_to_attributes.get(field_id, None)
                if attribute_name not in self.settings["known_values"]:
                    value = values[0]
                else:
                    try:
                        value = next(
                            filter(
                                lambda v: v in values,
                                self.settings["known_values"][attribute_name],
                            )
                        )
                    except StopIteration:
                        value = None

        if not isinstance(value, (int, float, bool, str, bytes)):
            try:
                value = str(value)
            except TypeError:
                pass

        return value

    def iter_changes(self, issue, fields):
        """Yield an IssueSnapshot for each time the issue changed, including an
        initial value. `fields` is a list of fields to monitor, e.g.
        `['status']`.
        """

        for field in fields:
            try:
                field_id = self.field_name_to_id(field)
                initial_value = self.resolve_field_value(issue, field_id)
            except ConfigError:
                # Field not present in this JIRA instance/fixture;
                # proceed without initial value
                logger.debug(
                    "Unknown JIRA field '%s' in iter_changes; no initial value", field
                )
                initial_value = None
            try:
                found_item = next(
                    filter(
                        lambda h, f=field: h.field == f,
                        itertools.chain.from_iterable(
                            (
                                c.items
                                for c in sorted(
                                    issue.changelog.histories,
                                    key=lambda c: dateutil.parser.parse(c.created),
                                )
                            )
                        ),
                    )
                )
                # Support both real JIRA PropertyHolder attributes and test aliases
                initial_value = self._get_tolerant_attr(
                    found_item, "fromString", "from_string"
                )
            except StopIteration:
                pass

            yield IssueSnapshot(
                change=field,
                transition_data={
                    "key": issue.key,
                    "date": dateutil.parser.parse(issue.fields.created),
                    "from_string": None,
                    "to_string": initial_value,
                },
            )

        for change in sorted(
            issue.changelog.histories,
            key=lambda c: dateutil.parser.parse(c.created),
        ):
            change_date = dateutil.parser.parse(change.created, ignoretz=True)

            for item in change.items:
                if item.field in fields:
                    yield IssueSnapshot(
                        change=item.field,
                        transition_data={
                            "key": issue.key,
                            "date": change_date,
                            # Support both camelCase from JIRA and snake_case in tests
                            "from_string": self._get_tolerant_attr(
                                item, "fromString", "from_string"
                            ),
                            "to_string": self._get_tolerant_attr(
                                item, "toString", "to_string"
                            ),
                        },
                    )

    # Basic queries

    def find_issues(self, jql, expand="changelog", max_results=None):
        """Return a list of issues with changelog metadata for the given JQL.

        Args:
            jql: JQL query string
            expand: Fields to expand (default: "changelog")
            max_results: Optional limit on number of results. If None, uses
                settings["max_results"]. If False, no limit.

        Returns:
            List of issues
        """
        if max_results is None:
            max_results = self.settings["max_results"]

        logger.info("Fetching issues with query `%s`", jql)
        if max_results:
            logger.info("Limiting to %d results", max_results)

        try:
            # Convert False to None for jira library - False means "no limit"
            max_results_param = None if max_results is False else max_results
            issues = self.jira.search_issues(
                jql, expand=expand, maxResults=max_results_param
            )
            issues_count = len(issues)
            logger.info("Fetched %d issues", issues_count)
            if issues_count == 0:
                logger.warning(
                    "Query returned 0 issues. This may indicate: "
                    "1. The JQL query doesn't match any issues "
                    "2. Authentication/authorization issues "
                    "3. The query needs adjustment"
                )
            return issues
        except JIRAError as e:
            logger.error(
                "JIRA API error while fetching issues with query `%s`: %s (Status: %s)",
                jql,
                getattr(e, "text", str(e)),
                getattr(e, "status_code", "Unknown"),
            )
            raise
        except Exception as e:
            logger.error(
                "Unexpected error while fetching issues with query `%s`: %s",
                jql,
                str(e),
            )
            raise

    def has_issues_for_jql(self, jql):
        """Check if there are any issues matching the JQL query.

        This is a lightweight check that queries for only one result to
        determine data availability without mutating global settings.

        Args:
            jql: JQL query string to check

        Returns:
            True if at least one issue exists, False otherwise

        Note:
            Returns False on expected runtime errors (connection, JIRA API errors).
            Other exceptions (AttributeError, ValueError, TypeError, etc.) will
            propagate to surface programming bugs during development/testing.
        """
        try:
            issues = self.find_issues(jql, expand="", max_results=1)
            return len(issues) > 0
        except (ConnectionError, JIRAError) as e:
            # Return False on expected runtime errors (connection, JIRA API errors)
            # ConnectionError covers network connectivity issues
            # JIRAError covers JIRA-specific errors like authentication failures,
            # invalid queries, and API errors
            # Note: AttributeError, ValueError, TypeError and other exceptions
            # are allowed to propagate to surface programming bugs during development
            logger.debug(
                "Error checking data availability with JQL '%s': %s",
                jql,
                e,
            )
            return False
