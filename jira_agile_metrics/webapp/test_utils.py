"""Test utilities for webapp tests.

This module provides reusable test utilities for HTML parsing and validation
in webapp tests.
"""

from html.parser import HTMLParser


class HTMLOutlineParser(HTMLParser):
    """Parser for extracting HTML structure outline.

    Parses HTML and builds an outline of tags and their attributes/text content
    in document order. Useful for testing HTML structure without brittle whitespace
    assumptions.

    Attributes:
        outline: List of (tag, text_or_attrs) tuples in document order.
            For div tags, the second element is a dict of attributes.
            For h1 tags, the second element is the text content string.
    """

    def __init__(self):
        """Initialize the parser."""
        super().__init__()
        self._collecting_h1 = False
        self._h1_buffer = []
        self.outline = []  # list of (tag, text_or_attrs) in document order

    def handle_starttag(self, tag, attrs):
        """Handle start tags."""
        if tag.lower() == "h1":
            self._collecting_h1 = True
            self._h1_buffer = []
        elif tag.lower() == "div":
            # Convert attrs list of (name, value) tuples to dict
            attrs_dict = dict(attrs) if attrs else {}
            self.outline.append(("div", attrs_dict))

    def handle_endtag(self, tag):
        """Handle end tags."""
        if tag.lower() == "h1" and self._collecting_h1:
            text = "".join(self._h1_buffer).strip()
            self.outline.append(("h1", text))
            self._collecting_h1 = False
            self._h1_buffer = []

    def handle_data(self, data):
        """Handle text data."""
        if self._collecting_h1:
            self._h1_buffer.append(data)


def has_chart_container_attrs(attrs_dict):
    """Check if div attributes indicate a chart container.

    Args:
        attrs_dict: Dictionary of HTML attributes.

    Returns:
        bool: True if the attributes indicate a chart container,
            False otherwise.
    """
    if not attrs_dict or not isinstance(attrs_dict, dict):
        return False
    # Check for class="chart-container" or id="chart"
    class_attr = attrs_dict.get("class", "")
    id_attr = attrs_dict.get("id", "")
    has_chart_container_class = "chart-container" in class_attr.split()
    has_chart_id = id_attr == "chart"
    return has_chart_container_class or has_chart_id
