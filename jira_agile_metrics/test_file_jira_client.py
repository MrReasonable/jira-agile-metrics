"""File-backed JIRA client for functional tests.

This minimal client loads realistic-looking data from JSON fixtures and exposes
the subset of the `jira.JIRA` interface used by our code: `fields()` and
`search_issues()`.
"""

import json
import os
from typing import Any, Dict, List, Optional

from .test_classes import FauxChange, FauxIssue


class FileJiraClient:
    """A simple file-backed JIRA client used in functional tests.

    It reads two JSON fixture files inside a directory:
    - fields.json: list of {"id": str, "name": str}
    - search_issues.json: list of JIRA issue-like dicts
    """

    def __init__(self, fixtures_dir: str, server_url: str = "https://example.org"):
        self._fixtures_dir = fixtures_dir
        self._options = {"server": server_url}
        self._fields_cache: Optional[List[Dict[str, Any]]] = None
        self._issues_cache: Optional[List[FauxIssue]] = None

    def _load_fields(self) -> List[Dict[str, Any]]:
        if self._fields_cache is None:
            path = os.path.join(self._fixtures_dir, "fields.json")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._fields_cache = json.load(f)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Fixture file not found: {path}. Underlying error: {str(e)}"
                ) from e
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse JSON from fixture file: {path}. "
                    f"Error at line {e.lineno}, column {e.colno}: {e.msg}"
                ) from e
        return self._fields_cache

    def _load_issues(self) -> List[FauxIssue]:
        if self._issues_cache is None:
            path = os.path.join(self._fixtures_dir, "search_issues.json")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._issues_cache = [self._to_faux_issue(i) for i in raw]
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Fixture file not found: {path}. Underlying error: {str(e)}"
                ) from e
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse JSON from fixture file: {path}. "
                    f"Error at line {e.lineno}, column {e.colno}: {e.msg}"
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Unexpected error while reading fixture file: {path}. {e}"
                ) from e
        return self._issues_cache

    def _to_faux_issue(self, data: Dict[str, Any]) -> FauxIssue:
        fields = data["fields"]
        changes = [
            FauxChange(
                ch["created"],
                [
                    (it["field"], it.get("fromString"), it.get("toString"))
                    for it in ch["items"]
                ],
            )
            for ch in data.get("changelog", {}).get("histories", [])
        ]
        return FauxIssue(
            data["key"],
            changes,
            summary=fields.get("summary"),
            issuetype=fields.get("issuetype"),
            status=fields.get("status"),
            resolution=fields.get("resolution"),
            created=fields.get("created"),
            updated=fields.get("updated", fields.get("created")),
        )

    # Public API used by QueryManager
    def fields(self) -> List[Dict[str, Any]]:  # noqa: D401 - minimal test stub
        """Return field metadata list."""
        return self._load_fields()

    def search_issues(  # noqa: D401
        self, _jql: str, *_args, **_kwargs
    ) -> List[FauxIssue]:
        """Return FauxIssue list for the given JQL (fixtures are static)."""
        return self._load_issues()
