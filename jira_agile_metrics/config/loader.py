"""Configuration loader for Jira Agile Metrics."""

import logging
import os.path

import yaml

from ..common_constants import (
    BOTTLENECK_CHART_SETTINGS,
    CHART_FILENAME_KEYS,
    DATA_FILENAME_KEYS,
)
from .exceptions import ConfigError
from .progress_report_utils import (
    to_progress_report_outcomes_list,
    to_progress_report_teams_list,
)
from .type_utils import (
    expand_key,
    force_date,
    force_float,
    force_int,
    force_list,
)
from .yaml_utils import ordered_load


def _create_default_options():
    """Create default options dictionary."""
    return {
        "connection": {
            "domain": None,
            "type": "jira",
            "username": None,
            "password": None,
            "key": None,
            "token": None,
            "http_proxy": None,
            "https_proxy": None,
            "jira_server_version_check": True,
            "jira_client_options": {},
        },
        "settings": {
            "queries": [],
            "query_attribute": None,
            "attributes": {},
            "known_values": {},
            "type_mapping": {},
            "cycle": [],
            "max_results": None,
            "verbose": False,
            "quantiles": [0.5, 0.85, 0.95],
            "date_format": "%d/%m/%Y",
            "backlog_column": None,
            "committed_column": None,
            "done_column": None,
            "cycle_time_data": None,
            "percentiles_data": None,
            "scatterplot_window": None,
            "scatterplot_data": None,
            "scatterplot_chart": None,
            "scatterplot_chart_title": None,
            "histogram_window": None,
            "histogram_data": None,
            "histogram_chart": None,
            "histogram_chart_title": None,
            "cfd_window": None,
            "cfd_data": None,
            "cfd_chart": None,
            "cfd_chart_title": None,
            "throughput_frequency": "1W-MON",
            "throughput_window": None,
            "throughput_data": None,
            "throughput_chart": None,
            "throughput_chart_title": None,
            "burnup_window": None,
            "burnup_chart": None,
            "burnup_chart_title": None,
            "burnup_forecast_window": None,
            "burnup_forecast_chart": None,
            "burnup_forecast_chart_title": None,
            "burnup_forecast_chart_data": None,
            "burnup_forecast_chart_target": None,
            "burnup_forecast_chart_deadline": None,
            "burnup_forecast_chart_deadline_confidence": None,
            "burnup_forecast_chart_trials": 1000,
            "burnup_forecast_chart_confidence": 0.8,
            "burnup_forecast_chart_random_seed": None,
            "burnup_forecast_chart_max_iterations": 9999,
            "burnup_forecast_chart_throughput_window": 60,
            "burnup_forecast_chart_throughput_window_end": None,
            "burnup_forecast_chart_backlog_growth_window": None,
            "wip_frequency": "1W-MON",
            "wip_window": None,
            "wip_chart": None,
            "wip_chart_title": None,
            "ageing_wip_chart": None,
            "ageing_wip_chart_title": None,
            "net_flow_frequency": "1W-MON",
            "net_flow_window": None,
            "net_flow_chart": None,
            "net_flow_chart_title": None,
            "impediments_data": None,
            "impediments_window": None,
            "impediments_chart": None,
            "impediments_chart_title": None,
            "impediments_days_chart": None,
            "impediments_days_chart_title": None,
            "impediments_status_chart": None,
            "impediments_status_chart_title": None,
            "impediments_status_days_chart": None,
            "impediments_status_days_chart_title": None,
            "impediment_flags": ["Impediment", "Awaiting input"],
            "defects_query": None,
            "defects_window": None,
            "defects_priority_field": None,
            "defects_priority_values": None,
            "defects_type_field": None,
            "defects_type_values": None,
            "defects_environment_field": None,
            "defects_environment_values": None,
            "defects_by_priority_chart": None,
            "defects_by_priority_chart_title": None,
            "defects_by_type_chart": None,
            "defects_by_type_chart_title": None,
            "defects_by_environment_chart": None,
            "defects_by_environment_chart_title": None,
            "debt_query": None,
            "debt_window": None,
            "debt_priority_field": None,
            "debt_priority_values": None,
            "debt_chart": None,
            "debt_chart_title": None,
            "debt_age_chart": None,
            "debt_age_chart_title": None,
            "debt_age_chart_bins": [30, 60, 90],
            "waste_query": None,
            "waste_window": None,
            "waste_frequency": "MS",
            "waste_chart": None,
            "waste_chart_title": None,
            "progress_report": None,
            "progress_report_title": None,
            "progress_report_epic_query_template": None,
            "progress_report_story_query_template": None,
            "progress_report_epic_deadline_field": None,
            "progress_report_epic_min_stories_field": None,
            "progress_report_epic_max_stories_field": None,
            "progress_report_epic_team_field": None,
            "progress_report_teams": None,
            "progress_report_outcomes": None,
            "progress_report_outcome_query": None,
            "progress_report_outcome_deadline_field": None,
            "lead_time_histogram_chart_title": None,
            "negative_duration_handling": None,
        },
    }


def _parse_connection_config(config, options):
    """Parse connection configuration."""
    if "connection" not in config:
        return

    conn_config = config["connection"]
    conn_options = options["connection"]

    # Map connection fields
    field_mappings = [
        ("domain", "domain"),
        ("type", "type"),
        ("username", "username"),
        ("password", "password"),
        ("key", "key"),
        ("token", "token"),
        ("http proxy", "http_proxy"),
        ("https proxy", "https_proxy"),
        ("jira client options", "jira_client_options"),
        ("jira server version check", "jira_server_version_check"),
    ]

    for config_key, option_key in field_mappings:
        if config_key in conn_config:
            conn_options[option_key] = conn_config[config_key]


def _parse_output_config(config, options):
    """Parse output configuration."""
    if "output" not in config:
        return

    output_config = config["output"]
    settings = options["settings"]

    # Output directory support
    if expand_key("output_directory") in output_config:
        options["output_directory"] = output_config[expand_key("output_directory")]

    # Quantiles
    if "quantiles" in output_config:
        try:
            settings["quantiles"] = list(map(float, output_config["quantiles"]))
        except ValueError:
            raise ConfigError(
                f"Could not convert value `{output_config['quantiles']}` "
                f"for key `quantiles` to a list of decimals"
            ) from None

    # Parse different types of values
    _parse_int_values(output_config, settings)
    _parse_float_values(output_config, settings)
    _parse_date_values(output_config, settings)
    _parse_filename_values(output_config, settings)
    _parse_filename_list_values(output_config, settings)
    _parse_list_values(output_config, settings)
    _parse_string_values(output_config, settings)
    _parse_boolean_values(output_config, settings)
    _parse_progress_report_values(output_config, settings)


def _parse_int_values(output_config, settings):
    """Parse integer values from output config."""
    int_keys = [
        "scatterplot_window",
        "histogram_window",
        "wip_window",
        "net_flow_window",
        "throughput_window",
        "cfd_window",
        "burnup_window",
        "burnup_forecast_window",
        "burnup_forecast_chart_throughput_window",
        "burnup_forecast_chart_target",
        "burnup_forecast_chart_trials",
        "burnup_forecast_chart_max_iterations",
        "burnup_forecast_chart_random_seed",
        "impediments_window",
        "defects_window",
        "debt_window",
        "waste_window",
        "burnup_forecast_chart_backlog_growth_window",
    ]

    for key in int_keys:
        if expand_key(key) in output_config:
            value = output_config[expand_key(key)]
            # Special handling for random_seed which can be None
            if key == "burnup_forecast_chart_random_seed" and value is None:
                settings[key] = None
            else:
                settings[key] = force_int(key, value)


def _parse_float_values(output_config, settings):
    """Parse float values from output config."""
    float_keys = [
        "burnup_forecast_chart_deadline_confidence",
        "burnup_forecast_chart_confidence",
    ]

    for key in float_keys:
        if expand_key(key) in output_config:
            settings[key] = force_float(key, output_config[expand_key(key)])


def _parse_date_values(output_config, settings):
    """Parse date values from output config."""
    date_keys = [
        "burnup_forecast_chart_throughput_window_end",
        "burnup_forecast_chart_deadline",
    ]

    for key in date_keys:
        if expand_key(key) in output_config:
            settings[key] = force_date(key, output_config[expand_key(key)])


def _parse_filename_values(output_config, settings):
    """Parse filename values from output config."""
    filename_keys = CHART_FILENAME_KEYS + BOTTLENECK_CHART_SETTINGS

    for key in filename_keys:
        if expand_key(key) in output_config:
            settings[key] = os.path.basename(output_config[expand_key(key)])


def _parse_filename_list_values(output_config, settings):
    """Parse filename list values from output config."""
    filename_list_keys = DATA_FILENAME_KEYS

    for key in filename_list_keys:
        if expand_key(key) in output_config:
            settings[key] = list(
                map(
                    os.path.basename,
                    force_list(output_config[expand_key(key)]),
                )
            )


def _parse_list_values(output_config, settings):
    """Parse list values from output config."""
    list_keys = [
        "defects_priority_values",
        "defects_type_values",
        "defects_environment_values",
        "debt_priority_values",
        "debt_age_chart_bins",
        "impediment_flags",
    ]

    for key in list_keys:
        if expand_key(key) in output_config:
            settings[key] = force_list(output_config[expand_key(key)])


def _parse_string_values(output_config, settings):
    """Parse string values from output config."""
    string_keys = [
        "date_format",
        "backlog_column",
        "committed_column",
        "done_column",
        "throughput_frequency",
        "burnup_forecast_chart_throughput_frequency",
        "scatterplot_chart_title",
        "histogram_chart_title",
        "cfd_chart_title",
        "throughput_chart_title",
        "burnup_chart_title",
        "burnup_forecast_chart_title",
        "wip_chart_title",
        "wip_frequency",
        "ageing_wip_chart_title",
        "net_flow_chart_title",
        "net_flow_frequency",
        "impediments_chart_title",
        "impediments_days_chart_title",
        "impediments_status_chart_title",
        "impediments_status_days_chart_title",
        "defects_query",
        "defects_by_priority_chart_title",
        "defects_priority_field",
        "defects_by_type_chart_title",
        "defects_type_field",
        "defects_by_environment_chart_title",
        "defects_environment_field",
        "debt_query",
        "debt_priority_field",
        "debt_chart_title",
        "debt_age_chart_title",
        "waste_query",
        "waste_frequency",
        "waste_chart_title",
        "progress_report_title",
        "progress_report_epic_query_template",
        "progress_report_story_query_template",
        "progress_report_epic_deadline_field",
        "progress_report_epic_min_stories_field",
        "progress_report_epic_max_stories_field",
        "progress_report_epic_team_field",
        "progress_report_outcome_query",
        "progress_report_outcome_deadline_field",
        "lead_time_histogram_chart_title",
        "negative_duration_handling",
    ]

    for key in string_keys:
        if expand_key(key) in output_config:
            settings[key] = str(output_config[expand_key(key)])


def _parse_boolean_values(output_config, settings):
    """Parse boolean values from output config."""
    boolean_keys = [
        "use_cache",
        "reset_on_backwards",
        "keep_first_entry_time",
        "burnup_forecast_chart_smart_window",
    ]

    for key in boolean_keys:
        if expand_key(key) in output_config:
            settings[key] = bool(output_config[expand_key(key)])


def _parse_progress_report_values(output_config, settings):
    """Parse progress report specific values."""
    if expand_key("progress_report_teams") in output_config:
        settings["progress_report_teams"] = to_progress_report_teams_list(
            output_config[expand_key("progress_report_teams")]
        )

    if expand_key("progress_report_outcomes") in output_config:
        settings["progress_report_outcomes"] = to_progress_report_outcomes_list(
            output_config[expand_key("progress_report_outcomes")]
        )


def _has_nested_progress_report(settings_dict):
    pr = settings_dict.get("progress_report")
    return isinstance(pr, dict) and "templates" in pr


def _build_progress_report_structure(settings_dict):
    progress_report_filename = settings_dict.get("progress_report")
    settings_dict["progress_report"] = {
        "enabled": progress_report_filename is not None,
        "title": settings_dict.get("progress_report_title"),
        "outcomes": settings_dict.get("progress_report_outcomes") or {},
        "outcome_query": settings_dict.get("progress_report_outcome_query"),
        "fields": settings_dict.get("progress_report_fields") or {},
        "templates": {
            "epic": settings_dict.get("progress_report_epic_query_template"),
            "story": settings_dict.get("progress_report_story_query_template"),
        },
        "epic_fields": {
            "deadline": settings_dict.get("progress_report_epic_deadline_field"),
            "min_stories": settings_dict.get("progress_report_epic_min_stories_field"),
            "max_stories": settings_dict.get("progress_report_epic_max_stories_field"),
            "team": settings_dict.get("progress_report_epic_team_field"),
        },
        "outcome_fields": {
            "deadline": settings_dict.get("progress_report_outcome_deadline_field"),
        },
        "teams": settings_dict.get("progress_report_teams") or [],
        "quantiles": settings_dict.get("progress_report_quantiles")
        or [0.5, 0.85, 0.95],
        "charts": {},
    }
    # Build charts dict in a data-driven way to reduce repetition
    chart_types = [
        "cfd",
        "burnup",
        "burnup_forecast",
        "scatterplot",
        "histogram",
        "throughput",
        "wip",
        "ageing_wip",
        "net_flow",
        "impediments",
        "defects",
        "debt",
        "waste",
    ]
    charts = {}
    for chart_type in chart_types:
        charts[chart_type] = {
            "filename": settings_dict.get(f"progress_report_{chart_type}_chart"),
            "title": settings_dict.get(f"progress_report_{chart_type}_chart_title"),
        }
    settings_dict["progress_report"]["charts"] = charts
    if progress_report_filename:
        settings_dict["progress_report"]["filename"] = progress_report_filename


def _remove_flat_progress_report_keys(settings_dict):
    flat_keys_to_remove = [
        "progress_report_title",
        "progress_report_epic_query_template",
        "progress_report_story_query_template",
        "progress_report_epic_deadline_field",
        "progress_report_epic_min_stories_field",
        "progress_report_epic_max_stories_field",
        "progress_report_epic_team_field",
        "progress_report_teams",
        "progress_report_outcomes",
        "progress_report_outcome_query",
        "progress_report_outcome_deadline_field",
        "progress_report_quantiles",
        "progress_report_cfd_chart",
        "progress_report_cfd_chart_title",
        "progress_report_burnup_chart",
        "progress_report_burnup_chart_title",
        "progress_report_burnup_forecast_chart",
        "progress_report_burnup_forecast_chart_title",
        "progress_report_scatterplot_chart",
        "progress_report_scatterplot_chart_title",
        "progress_report_histogram_chart",
        "progress_report_histogram_chart_title",
        "progress_report_throughput_chart",
        "progress_report_throughput_chart_title",
        "progress_report_wip_chart",
        "progress_report_wip_chart_title",
        "progress_report_ageing_wip_chart",
        "progress_report_ageing_wip_chart_title",
        "progress_report_net_flow_chart",
        "progress_report_net_flow_chart_title",
        "progress_report_impediments_chart",
        "progress_report_impediments_chart_title",
        "progress_report_defects_chart",
        "progress_report_defects_chart_title",
        "progress_report_debt_chart",
        "progress_report_debt_chart_title",
        "progress_report_waste_chart",
        "progress_report_waste_chart_title",
        "progress_report_fields",
    ]
    for key in flat_keys_to_remove:
        settings_dict.pop(key, None)


def _transform_progress_report_to_nested(settings):
    """Transform flat progress_report_* keys into nested progress_report dict."""
    # Check if any progress_report_* keys exist (flat keys)
    has_flat_keys = any(
        key.startswith("progress_report_") and key != "progress_report"
        for key in settings
    )

    # Already nested: just clean up leftover flat keys and return
    if _has_nested_progress_report(settings):
        _remove_flat_progress_report_keys(settings)
        return

    # Nothing to do if neither flat nor nested exists
    if not has_flat_keys and "progress_report" not in settings:
        return

    # Build nested structure if needed
    if not isinstance(settings.get("progress_report"), dict):
        _build_progress_report_structure(settings)

    # Remove stale flat keys for clarity
    _remove_flat_progress_report_keys(settings)


def _parse_queries_config(config, options):
    """Parse queries configuration."""
    if "queries" in config:
        options["settings"]["query_attribute"] = config["queries"].get(
            "attribute", None
        )
        options["settings"]["queries"] = [
            {
                "value": q.get("value", None),
                "jql": q.get("jql", None),
            }
            for q in config["queries"]["criteria"]
        ]

    if "query" in config:
        options["settings"]["queries"] = [
            {
                "value": None,
                "jql": config["query"],
            }
        ]


def _parse_workflow_config(config, options, extended):
    """Parse workflow configuration."""
    if "workflow" not in config:
        if not extended and len(options["settings"]["cycle"]) == 0:
            raise ConfigError("`Workflow` section not found")
        return

    workflow = config["workflow"]
    if len(workflow.keys()) < 3:
        raise ConfigError("`Workflow` section must contain at least three statuses")

    column_names = []
    for name, statuses in workflow.items():
        statuses = force_list(statuses)
        options["settings"]["cycle"].append({"name": name, "statuses": statuses})
        column_names.append(name)

    _validate_and_set_columns(options, column_names)


def _validate_and_set_columns(options, column_names):
    """Validate and set backlog, committed, and done columns."""
    settings = options["settings"]

    # Handle backlog column
    if settings["backlog_column"] is None:
        if settings["committed_column"] is None:
            settings["backlog_column"] = column_names[0]
            logging.getLogger(__name__).info(
                "`Backlog column` automatically set to `%s`",
                settings["backlog_column"],
            )
            settings["committed_column"] = column_names[1]
            logging.getLogger(__name__).info(
                "`Committed column` automatically set to `%s`",
                settings["committed_column"],
            )
        else:
            _validate_committed_column(settings, column_names)
    else:
        _validate_backlog_column(settings, column_names)

    # Handle done column
    if settings["done_column"] is None:
        settings["done_column"] = column_names[-1]
        logging.getLogger(__name__).info(
            "`Done column` automatically set to `%s`",
            settings["done_column"],
        )
    elif settings["done_column"] not in column_names:
        raise ConfigError(
            f"`Done column` ({settings['done_column']}) "
            f"must exist in `Workflow`: {column_names}"
        )

    _validate_column_order(settings, column_names)


def _validate_committed_column(settings, column_names):
    """Validate committed column configuration."""
    if settings["committed_column"] not in column_names:
        raise ConfigError(
            f"`Committed column` "
            f"({settings['committed_column']}) "
            f"must exist in `Workflow`: {column_names}"
        )

    if column_names.index(settings["committed_column"]) > 0:
        settings["backlog_column"] = column_names[
            column_names.index(settings["committed_column"]) - 1
        ]
        logging.getLogger(__name__).info(
            "`Backlog column` automatically set to `%s`",
            settings["backlog_column"],
        )
    else:
        raise ConfigError(
            f"There must be at least 1 column before "
            f"`Committed column` "
            f"({settings['committed_column']}) "
            f"in `Workflow`: {column_names}"
        )


def _validate_backlog_column(settings, column_names):
    """Validate backlog column configuration."""
    if settings["backlog_column"] not in column_names:
        raise ConfigError(
            f"`Backlog column` ({settings['backlog_column']}) "
            f"must exist in `Workflow`: {column_names}"
        )

    if column_names.index(settings["backlog_column"]) < (len(column_names) - 2):
        settings["committed_column"] = column_names[
            column_names.index(settings["backlog_column"]) + 1
        ]
        logging.getLogger(__name__).info(
            "`Committed column` automatically set to `%s`",
            settings["committed_column"],
        )
    else:
        raise ConfigError(
            f"There must be at least 2 columns after "
            f"`Backlog column` ({settings['backlog_column']}) "
            f"in `Workflow`: {column_names}"
        )


def _validate_column_order(settings, column_names):
    """Validate the order of columns."""
    # Backlog column must come before committed column
    if not (column_names.index(settings["backlog_column"]) + 1) == column_names.index(
        settings["committed_column"]
    ):
        raise ConfigError(
            f"`Backlog column` ({settings['backlog_column']}) "
            f"must come immediately before `Committed column` "
            f"({settings['committed_column']}) in `Workflow`"
        )

    # Committed column must come before done column
    if not column_names.index(settings["committed_column"]) < column_names.index(
        settings["done_column"]
    ):
        raise ConfigError(
            f"`Committed column` ({settings['committed_column']}) "
            f"must come before `Done column` "
            f"({settings['done_column']}) "
            f"in `Workflow`: {column_names}"
        )


def config_to_options(data, cwd=None, extended=False, _visited_files=None):
    """
    Parse YAML config data and return options dict.
    """
    if _visited_files is None:
        _visited_files = set()

    try:
        config = ordered_load(data, yaml.SafeLoader)
    except Exception as e:
        raise ConfigError("Unable to parse YAML configuration file.") from e

    if config is None:
        raise ConfigError("Configuration file is empty") from None

    options = _create_default_options()

    # Handle extends configuration
    if "extends" in config:
        if cwd is None:
            raise ConfigError("`extends` is not supported here.")

        extends_filename = os.path.abspath(
            os.path.normpath(
                os.path.join(cwd, config["extends"].replace("/", os.path.sep))
            )
        )

        if not os.path.exists(extends_filename):
            raise ConfigError(
                f"File `{extends_filename}` referenced in `extends` not found."
            ) from None

        if extends_filename in _visited_files:
            raise ConfigError(
                f"Circular extends reference detected: {extends_filename}"
            ) from None

        _visited_files.add(extends_filename)

        logging.getLogger(__name__).debug("Extending file %s", extends_filename)
        with open(extends_filename, encoding="utf-8") as extends_file:
            options = config_to_options(
                extends_file.read(),
                cwd=os.path.dirname(extends_filename),
                extended=True,
                _visited_files=_visited_files,
            )

    # Parse configuration sections
    _parse_connection_config(config, options)
    _parse_output_config(config, options)
    _parse_queries_config(config, options)
    _parse_workflow_config(config, options, extended)

    # Parse additional configuration sections
    _parse_additional_config(config, options, extended)

    # Transform flat progress_report_* keys to nested structure
    _transform_progress_report_to_nested(options["settings"])

    return options


def _parse_additional_config(config, options, extended):
    """Parse additional configuration sections like attributes, known values, etc."""
    # Parse attributes (fields) - merge from extended file if needed
    if "attributes" in config and config["attributes"] is not None:
        options["settings"]["attributes"].update(dict(config["attributes"]))

    if "known values" in config:
        for name, values in config["known values"].items():
            options["settings"]["known_values"][name] = force_list(values)

    # Trello label to type mapping
    if "type mapping" in config:
        for name, values in config["type mapping"].items():
            options["settings"]["type_mapping"][name] = force_list(values)

    # Validate queries if not extended
    if not extended and len(options["settings"]["queries"]) == 0:
        logging.getLogger(__name__).warning(
            "No `Query` value or `Queries` section found. "
            "Many calculators rely on one of these."
        )
