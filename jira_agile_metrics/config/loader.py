"""Configuration loader for Jira Agile Metrics."""

import logging
import os.path

import yaml

from .exceptions import ConfigError
from .progress_report_utils import (to_progress_report_outcomes_list,
                                    to_progress_report_teams_list)
from .type_utils import (expand_key, force_date, force_float, force_int,
                         force_list)
from .yaml_utils import ordered_load


def config_to_options(data, cwd=None, extended=False):
    """
    Parse YAML config data and return options dict.
    """
    try:
        config = ordered_load(data, yaml.SafeLoader)
    except Exception as e:
        raise ConfigError("Unable to parse YAML configuration file.") from e

    if config is None:
        raise ConfigError("Configuration file is empty") from None

    options = {
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
            "burnup_forecast_chart_target": None,
            "burnup_forecast_chart_deadline": None,
            "burnup_forecast_chart_deadline_confidence": None,
            "burnup_forecast_chart_trials": 100,
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

    # Recursively parse an `extends` file but only if a base path is given,
    # otherwise we can plausible leak files in server mode.
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

        logging.getLogger(__name__).debug(
            "Extending file %s", extends_filename
        )
        with open(extends_filename, encoding="utf-8") as extends_file:
            options = config_to_options(
                extends_file.read(),
                cwd=os.path.dirname(extends_filename),
                extended=True,
            )

    # Parse and validate Connection

    if "connection" in config:
        if "domain" in config["connection"]:
            options["connection"]["domain"] = config["connection"]["domain"]

        if "type" in config["connection"]:
            options["connection"]["type"] = config["connection"]["type"]

        if "username" in config["connection"]:
            options["connection"]["username"] = config["connection"][
                "username"
            ]

        if "password" in config["connection"]:
            options["connection"]["password"] = config["connection"][
                "password"
            ]

        if "key" in config["connection"]:
            options["connection"]["key"] = config["connection"]["key"]

        if "token" in config["connection"]:
            options["connection"]["token"] = config["connection"]["token"]

        if "http proxy" in config["connection"]:
            options["connection"]["http_proxy"] = config["connection"][
                "http proxy"
            ]

        if "https proxy" in config["connection"]:
            options["connection"]["https_proxy"] = config["connection"][
                "https proxy"
            ]

        if "jira client options" in config["connection"]:
            options["connection"]["jira_client_options"] = config[
                "connection"
            ]["jira client options"]

        if "jira server version check" in config["connection"]:
            options["connection"]["jira_server_version_check"] = config[
                "connection"
            ]["jira server version check"]

    # Parse and validate output options
    if "output" in config:
        # Output directory support
        if expand_key("output_directory") in config["output"]:
            options["output_directory"] = config["output"][
                expand_key("output_directory")
            ]
        if "quantiles" in config["output"]:
            try:
                options["settings"]["quantiles"] = list(
                    map(float, config["output"]["quantiles"])
                )
            except ValueError:
                raise ConfigError(
                    f"Could not convert value `{config['output']['quantiles']}` for key `quantiles` to a list of decimals"
                ) from None

        # int values
        for key in [
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
            "impediments_window",
            "defects_window",
            "debt_window",
            "waste_window",
            "burnup_forecast_chart_backlog_growth_window",
        ]:
            if expand_key(key) in config["output"]:
                options["settings"][key] = force_int(
                    key, config["output"][expand_key(key)]
                )

        # float values
        for key in [
            "burnup_forecast_chart_deadline_confidence",
        ]:
            if expand_key(key) in config["output"]:
                options["settings"][key] = force_float(
                    key, config["output"][expand_key(key)]
                )

        # date values
        for key in [
            "burnup_forecast_chart_throughput_window_end",
            "burnup_forecast_chart_deadline",
        ]:
            if expand_key(key) in config["output"]:
                options["settings"][key] = force_date(
                    key, config["output"][expand_key(key)]
                )

        # file name values
        for key in [
            "scatterplot_chart",
            "histogram_chart",
            "cfd_chart",
            "throughput_chart",
            "burnup_chart",
            "burnup_forecast_chart",
            "wip_chart",
            "ageing_wip_chart",
            "net_flow_chart",
            "impediments_chart",
            "impediments_days_chart",
            "impediments_status_chart",
            "impediments_status_days_chart",
            "defects_by_priority_chart",
            "defects_by_type_chart",
            "defects_by_environment_chart",
            "debt_chart",
            "debt_age_chart",
            "waste_chart",
            "progress_report",
            "bottleneck_stacked_per_issue_chart",
            "bottleneck_stacked_aggregate_mean_chart",
            "bottleneck_stacked_aggregate_median_chart",
            "bottleneck_boxplot_chart",
            "bottleneck_violin_chart",
        ]:
            if expand_key(key) in config["output"]:
                options["settings"][key] = os.path.basename(
                    config["output"][expand_key(key)]
                )

        # file name list values
        for key in [
            "cycle_time_data",
            "cfd_data",
            "scatterplot_data",
            "histogram_data",
            "throughput_data",
            "percentiles_data",
            "impediments_data",
            "lead_time_histogram_data",
            "lead_time_histogram_chart",
        ]:
            if expand_key(key) in config["output"]:
                options["settings"][key] = list(
                    map(
                        os.path.basename,
                        force_list(config["output"][expand_key(key)]),
                    )
                )

        # list values
        for key in [
            "defects_priority_values",
            "defects_type_values",
            "defects_environment_values",
            "debt_priority_values",
            "debt_age_chart_bins",
        ]:
            if expand_key(key) in config["output"]:
                options["settings"][key] = force_list(
                    config["output"][expand_key(key)]
                )

        # string values
        for key in [
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
        ]:
            if expand_key(key) in config["output"]:
                options["settings"][key] = str(
                    config["output"][expand_key(key)]
                )

        # Special objects for progress reports
        if expand_key("progress_report_teams") in config["output"]:
            options["settings"]["progress_report_teams"] = (
                to_progress_report_teams_list(
                    config["output"][expand_key("progress_report_teams")]
                )
            )
        if expand_key("progress_report_outcomes") in config["output"]:
            options["settings"]["progress_report_outcomes"] = (
                to_progress_report_outcomes_list(
                    config["output"][expand_key("progress_report_outcomes")]
                )
            )

        # boolean values
        for key in [
            "use_cache",
            "reset_on_backwards",
            "burnup_forecast_chart_smart_window",
        ]:
            if expand_key(key) in config["output"]:
                options["settings"][key] = bool(
                    config["output"][expand_key(key)]
                )

    # Parse Queries and/or a single Query

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

    if not extended and len(options["settings"]["queries"]) == 0:
        logging.getLogger(__name__).warning(
            ("No `Query` value or `Queries` section found. Many calculators rely on one of these."))

    # Parse Workflow. Assume first status is backlog
    # and last status is complete.

    if "workflow" in config:
        if len(config["workflow"].keys()) < 3:
            raise ConfigError(
                "`Workflow` section must contain at least three statuses"
            )

        column_names = []
        for name, statuses in config["workflow"].items():
            statuses = force_list(statuses)
            options["settings"]["cycle"].append(
                {"name": name, "statuses": statuses}
            )
            column_names.append(name)

        if options["settings"]["backlog_column"] is None:
            if options["settings"]["committed_column"] is None:
                options["settings"]["backlog_column"] = column_names[0]
                logging.getLogger(__name__).info(
                    "`Backlog column` automatically set to `%s`",
                    options["settings"]["backlog_column"],
                )
                options["settings"]["committed_column"] = column_names[1]
                logging.getLogger(__name__).info(
                    "`Committed column` automatically set to `%s`",
                    options["settings"]["committed_column"],
                )
            else:
                if options["settings"]["committed_column"] not in column_names:
                    raise ConfigError(
                        f"`Committed column` ({options['settings']['committed_column']}) must exist in `Workflow`: {column_names}"
                    )
                elif (
                    column_names.index(options["settings"]["committed_column"])
                    > 0
                ):
                    options["settings"]["backlog_column"] = column_names[
                        column_names.index(
                            options["settings"]["committed_column"]
                        )
                        - 1
                    ]
                    logging.getLogger(__name__).info(
                        "`Backlog column` automatically set to `%s`",
                        options["settings"]["backlog_column"],
                    )
                else:
                    raise ConfigError(
                        f"There must be at least 1 column before "
                        f"`Committed column` ({options['settings']['committed_column']}) in `Workflow`: {column_names}")
        else:
            if options["settings"]["backlog_column"] not in column_names:
                raise ConfigError(
                    f"`Backlog column` ({options['settings']['backlog_column']}) must exist in `Workflow`: {column_names}"
                )
            elif column_names.index(options["settings"]["backlog_column"]) < (
                len(column_names) - 2
            ):
                options["settings"]["committed_column"] = column_names[
                    column_names.index(options["settings"]["backlog_column"])
                    + 1
                ]
                logging.getLogger(__name__).info(
                    "`Committed column` automatically set to `%s`",
                    options["settings"]["committed_column"],
                )
            else:
                raise ConfigError(
                    f"There must be at least 2 columns after "
                    f"`Backlog column` ({options['settings']['committed_column']}) in `Workflow`: {column_names}")

        if options["settings"]["done_column"] is None:
            options["settings"]["done_column"] = column_names[-1]
            logging.getLogger(__name__).info(
                "`Done column` automatically set to `%s`",
                options["settings"]["done_column"],
            )
        elif options["settings"]["done_column"] not in column_names:
            raise ConfigError(
                f"`Done column` ({options['settings']['done_column']}) must exist in `Workflow`: {column_names}"
            )

        # backlog column must come before committed column
        if not (
            column_names.index(options["settings"]["backlog_column"]) + 1
        ) == column_names.index(options["settings"]["committed_column"]):
            raise ConfigError(
                f"`Backlog column` ({options['settings']['backlog_column']}) must come immediately "
                f"before `Committed column` ({options['settings']['committed_column']}) in `Workflow`")

        # committed column must come before done column
        if not column_names.index(
            options["settings"]["committed_column"]
        ) < column_names.index(options["settings"]["done_column"]):
            raise ConfigError(
                f"`Committed column` ({options['settings']['committed_column']}) must come before `Done column` ({options['settings']['done_column']}) in `Workflow`: {column_names}"
            )

    # Make sure we have workflow
    # (but only if this file is not being extended by another)
    if not extended and len(options["settings"]["cycle"]) == 0:
        raise ConfigError("`Workflow` section not found")

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
    return options
