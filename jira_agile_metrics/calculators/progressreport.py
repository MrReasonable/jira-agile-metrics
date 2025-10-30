"""Progress report calculator for Jira Agile Metrics.

This module provides functionality to generate comprehensive progress reports
with forecasting, team analysis, and outcome tracking.
"""

import base64
import datetime
import io
import logging

import jinja2
import matplotlib.pyplot as plt

from ..calculator import Calculator
from ..chart_styling_utils import set_chart_style
from ..config.exceptions import ConfigError
from .cfd import calculate_cfd_data
from .cycletime import CycleTimeCalculator
from .progressreport_models import (
    EpicProcessingParams,
    Outcome,
    Team,
)
from .progressreport_utils import (
    calculate_epic_target,
    find_epics,
    find_outcomes,
    update_story_counts,
)
from .scatterplot import calculate_scatterplot_data

logger = logging.getLogger(__name__)

jinja_env = jinja2.Environment(
    loader=jinja2.PackageLoader("jira_agile_metrics", "calculators"),
    autoescape=jinja2.select_autoescape(["html", "xml"]),
)


class ProgressReportCalculator(Calculator):
    """Output a progress report based on Monte Carlo forecast to completion"""

    def _validate_configuration(self):
        """Validate progress report configuration settings."""
        pr_config = self.settings.get("progress_report")
        if pr_config is None or not isinstance(pr_config, dict):
            return None

        epic_query_template = pr_config["templates"]["epic"]
        if not epic_query_template:
            if (
                pr_config["outcome_query"] is not None
                or pr_config["outcomes"] is None
                or len(pr_config["outcomes"]) == 0
                or any(
                    map(
                        lambda o: o["epic_query"] is None,
                        pr_config["outcomes"],
                    )
                )
            ):
                logger.error(
                    (
                        "`Progress report epic query template`"
                        "is required unless all outcomes have "
                        "`Epic query` set."
                    )
                )
                return None

        story_query_template = pr_config["templates"]["story"]
        if not story_query_template:
            logger.error("`Progress report story query template` is required")
            return None

        return epic_query_template, story_query_template

    def _resolve_field_ids(self):
        """Resolve field names to field IDs for epic configuration."""
        pr_config = self.settings.get("progress_report", {})
        epic_fields = pr_config.get("epic_fields", {})

        epic_deadline_field = epic_fields.get("deadline")
        if (
            epic_deadline_field
            and epic_deadline_field not in self.query_manager.jira_fields_to_names
        ):
            epic_deadline_field = self.query_manager.field_name_to_id(
                epic_deadline_field
            )

        epic_min_stories_field = epic_fields.get("min_stories")
        if (
            epic_min_stories_field
            and epic_min_stories_field not in self.query_manager.jira_fields_to_names
        ):
            epic_min_stories_field = self.query_manager.field_name_to_id(
                epic_min_stories_field
            )

        epic_max_stories_field = epic_fields.get("max_stories")
        if not epic_max_stories_field:
            epic_max_stories_field = epic_min_stories_field
        elif epic_max_stories_field not in self.query_manager.jira_fields_to_names:
            epic_max_stories_field = self.query_manager.field_name_to_id(
                epic_max_stories_field
            )

        epic_team_field = epic_fields.get("team")
        if (
            epic_team_field
            and epic_team_field not in self.query_manager.jira_fields_to_names
        ):
            epic_team_field = self.query_manager.field_name_to_id(epic_team_field)

        return (
            epic_deadline_field,
            epic_min_stories_field,
            epic_max_stories_field,
            epic_team_field,
        )

    def _validate_teams(self):
        """Validate team configuration settings."""
        pr_config = self.settings.get("progress_report", {})
        teams = pr_config.get("teams") or []

        # Validate each team
        for team in teams:
            try:
                validation_error = self.validate_single_team(team)
                if validation_error:
                    logger.error(validation_error)
                    return None
            except ConfigError as e:
                logger.error(str(e))
                return None

        return teams

    def validate_single_team(self, team):
        """Validate a single team configuration."""
        # Check for missing required keys and raise ConfigError with helpful message
        team_name = team.get("name", "Unknown")
        config_snippet = f" (team: {team_name})" if team_name != "Unknown" else ""

        try:
            team["name"]
        except KeyError:
            raise ConfigError(
                f"Team configuration missing required field: 'name'{config_snippet}"
            ) from None

        try:
            team["wip"]
        except KeyError:
            raise ConfigError(
                f"Team configuration missing required field: 'wip'{config_snippet}"
            ) from None

        error_message = None

        # Validate team name
        if not team["name"]:
            error_message = "Teams must have a name."
        # Validate WIP
        elif not team["wip"] or team["wip"] < 1:
            error_message = "Team WIP must be >= 1"
        # Validate throughput configuration
        elif team.get("min_throughput") or team.get("max_throughput"):
            if not (team.get("min_throughput") and team.get("max_throughput")):
                error_message = (
                    "If `Min throughput` is set, "
                    "`Max throughput` must also be set, "
                    "and vice versa."
                )
            elif team.get("min_throughput") > team.get("max_throughput"):
                error_message = "`Min throughput` must be <= `Max throughput`."
            elif team.get("throughput_samples"):
                error_message = (
                    "Cannot set both `Min/Max throughput` and `Throughput samples`."
                )
        # Validate throughput samples
        elif team.get("throughput_samples") and not team.get(
            "throughput_samples_window"
        ):
            error_message = (
                "If `Throughput samples` is set, "
                "`Throughput samples window` must also be set."
            )

        return error_message

    def setup_teams_and_outcomes(self, teams, epic_query_template):
        """Setup teams and outcomes for the progress report."""
        # Convert team configs to Team objects
        team_objects = [
            Team(
                name=team["name"],
                wip=team["wip"],
                min_throughput=team["min_throughput"],
                max_throughput=team["max_throughput"],
                throughput_samples=(
                    team["throughput_samples"].format(
                        team=f'"{team["name"]}"',
                    )
                    if team["throughput_samples"]
                    else None
                ),
                throughput_samples_window=team["throughput_samples_window"],
            )
            for team in teams
        ]

        team_lookup = {team.name.lower(): team for team in team_objects}
        team_epics = {team.name.lower(): [] for team in team_objects}

        # Find outcomes
        pr_config = self.settings.get("progress_report", {})
        outcomes = self._process_outcomes(epic_query_template)

        # Only create default outcome if outcomes is not explicitly configured
        # and no outcome_query. If outcomes={} explicitly, we should keep it empty
        has_explicit_outcomes = (
            "outcomes" in pr_config and pr_config["outcomes"] is not None
        )

        if (
            not outcomes
            and not pr_config.get("outcome_query")
            and not has_explicit_outcomes
        ):
            outcomes = [
                Outcome(
                    {
                        "name": None,
                        "key": None,
                        "deadline": None,
                        "epic_query": epic_query_template,
                    }
                )
            ]

        if pr_config.get("outcome_query"):
            if len(outcomes) > 0:
                if not all(outcome.name for outcome in outcomes):
                    logger.error("Outcomes must have a name.")
                    return None

        return team_objects, team_lookup, team_epics, outcomes

    def _process_outcomes(self, epic_query_template):
        """Process and validate outcomes configuration."""
        # Find outcomes, either in the config file or by querying JIRA (or both).
        # If none set, we use a single epic query and don't group by outcomes
        pr_config = self.settings.get("progress_report", {})
        outcomes = (
            [
                Outcome(
                    {
                        "name": o["name"],
                        "key": o["key"] if o["key"] else o["name"],
                        "deadline": (
                            datetime.datetime.combine(
                                o["deadline"], datetime.datetime.min.time()
                            )
                            if o["deadline"]
                            else None
                        ),
                        "epic_query": (
                            o["epic_query"]
                            if o["epic_query"]
                            else epic_query_template.format(
                                outcome=f'"{o["key"] if o["key"] else o["name"]}"'
                            )
                        ),
                    }
                )
                for o in pr_config["outcomes"]
            ]
            if pr_config.get("outcomes") is not None
            and len(pr_config.get("outcomes", {})) > 0
            else []
        )

        outcome_query = pr_config.get("outcome_query")
        if outcome_query:
            outcome_deadline_field = pr_config["outcome_fields"]["deadline"]
            if (
                outcome_deadline_field
                and outcome_deadline_field
                not in self.query_manager.jira_fields_to_names
            ):
                outcome_deadline_field = self.query_manager.field_name_to_id(
                    outcome_deadline_field
                )

            outcomes.extend(
                find_outcomes(
                    self.query_manager,
                    outcome_query,
                    outcome_deadline_field,
                    epic_query_template,
                )
            )

        # Check if outcomes was explicitly set to empty dict/list
        has_explicit_empty_outcomes = (
            "outcomes" in pr_config
            and pr_config["outcomes"] is not None
            and len(pr_config.get("outcomes", [])) == 0
        )

        if len(outcomes) > 0:
            if not all(outcome.name for outcome in outcomes):
                logger.error("Outcomes must have a name.")
                return None
        elif not has_explicit_empty_outcomes:
            # Only create default outcome if outcomes wasn't explicitly set to empty
            outcomes = [
                Outcome(
                    {
                        "name": None,
                        "key": None,
                        "deadline": None,
                        "epic_query": epic_query_template,
                    }
                )
            ]

        return outcomes

    def _setup_team_objects(self, teams):
        """Convert team configurations to Team objects."""
        return [
            Team(
                name=team["name"],
                wip=team["wip"],
                min_throughput=team["min_throughput"],
                max_throughput=team["max_throughput"],
                throughput_samples=(
                    team["throughput_samples"].format(
                        team=f'"{team["name"]}"',
                    )
                    if team["throughput_samples"]
                    else None
                ),
                throughput_samples_window=team["throughput_samples_window"],
            )
            for team in teams
        ]

    def _process_epics(self, params):
        """Process epics for each outcome."""
        for outcome in params.params["outcomes"]:
            for epic in find_epics(
                query_manager=self.query_manager,
                epic_config=params.params["epic_config"],
                outcome=outcome,
            ):
                self._assign_epic_team(epic, params)
                outcome.epics.append(epic)

                if epic.team is not None:
                    params.params["team_epics"][epic.team.name.lower()].append(epic)

                epic.data["story_query"] = params.params["story_query_template"].format(
                    epic=f'"{epic.key}"',
                    team=(f'"{epic.team.name}"' if epic.team is not None else None),
                    outcome=f'"{outcome.key}"',
                )

                # Calculate story counts for this epic
                update_story_counts(
                    epic=epic,
                    query_manager=self.query_manager,
                    cycle=params.params["cycle"],
                    backlog_column=params.params["backlog_column"],
                    done_column=params.params["done_column"],
                )

                # Calculate target completion date for this epic
                calculate_epic_target(epic)

    def _assign_epic_team(self, epic, params):
        """Assign team to epic."""
        if not params.params["epic_team_field"]:
            epic.team = params.params["default_team"]  # single defined team, or None
        else:
            epic_team_name = (
                epic.data["team_name"].strip() if epic.data["team_name"] else ""
            )
            epic.team = params.params["team_lookup"].get(epic_team_name.lower(), None)

            if epic.team is None:
                logger.warning(
                    (
                        "Cannot find team `%s` for epic `%s`."
                        "Dynamically adding a non-forecasted team."
                    ),
                    epic_team_name,
                    epic.key,
                )
                epic.team = Team(name=epic_team_name)
                epic.team.throughput_config["sampler"] = None
                params.params["team_lookup"][epic_team_name.lower()] = epic.team
                params.params["team_epics"][epic_team_name.lower()] = []

    def _prepare_configuration(self):
        """Prepare and validate configuration for progress report."""
        # Validate configuration
        config_result = self._validate_configuration()
        if config_result is None:
            return None
        epic_query_template, story_query_template = config_result

        # Resolve field IDs
        field_ids = self._resolve_field_ids()
        (
            epic_deadline_field,
            epic_min_stories_field,
            epic_max_stories_field,
            epic_team_field,
        ) = field_ids

        # Validate teams
        teams = self._validate_teams()
        if teams is None:
            return None

        # Additional validation for epic team field
        if not epic_team_field and len(teams) > 1:
            logger.error(
                (
                    "`Progress report epic team field` is required "
                    "if there is more than one team under "
                    "`Progress report teams`."
                )
            )
            return None

        return {
            "epic_query_template": epic_query_template,
            "story_query_template": story_query_template,
            "epic_deadline_field": epic_deadline_field,
            "epic_min_stories_field": epic_min_stories_field,
            "epic_max_stories_field": epic_max_stories_field,
            "epic_team_field": epic_team_field,
            "teams": teams,
        }

    def _setup_epic_processing(self, config):
        """Setup epic processing parameters."""
        # Setup teams and outcomes
        setup_result = self.setup_teams_and_outcomes(
            config["teams"], config["epic_query_template"]
        )
        if setup_result is None:
            return None
        team_objects, team_lookup, team_epics, outcomes = setup_result

        # Process outcomes
        self._process_outcomes(config["epic_query_template"])

        # Get cycle configuration
        cycle = self.query_manager.settings["cycle"]
        backlog_column = self.query_manager.settings["backlog_column"]
        done_column = self.query_manager.settings["done_column"]

        # Determine default team
        default_team = (
            team_objects[0]
            if not config["epic_team_field"] and len(team_objects) == 1
            else None
        )

        # Process epics for each outcome
        epic_config = {
            "min_stories_field": config["epic_min_stories_field"],
            "max_stories_field": config["epic_max_stories_field"],
            "team_field": config["epic_team_field"],
            "deadline_field": config["epic_deadline_field"],
        }

        epic_params = EpicProcessingParams(
            {
                "outcomes": outcomes,
                "epic_config": epic_config,
                "team_lookup": team_lookup,
                "team_epics": team_epics,
                "default_team": default_team,
                "epic_team_field": config["epic_team_field"],
                "story_query_template": config["story_query_template"],
                "cycle": cycle,
                "backlog_column": backlog_column,
                "done_column": done_column,
            }
        )
        self._process_epics(epic_params)

        return outcomes, team_objects

    def run(self, now=None, trials=1000):
        """Run the progress report calculator."""
        # Prepare configuration
        config = self._prepare_configuration()
        if config is None:
            return None

        # Setup epic processing
        processing_result = self._setup_epic_processing(config)
        if processing_result is None:
            return None
        outcomes, team_objects = processing_result

        # Run Monte Carlo simulation to complete
        pr_config = self.settings.get("progress_report", {})
        quantiles = pr_config.get("quantiles", [0.5, 0.85, 0.95])

        team_objects.sort(key=lambda t: t.name)

        for team in team_objects:
            if team.throughput_config["sampler"] is not None:
                forecast_config = {
                    "quantiles": quantiles,
                    "now": now,
                    "trials": trials,
                }
                team.run_forecast(forecast_config)

        # Generate charts
        charts = self._generate_charts()

        # Generate report data
        report_data = self._generate_report_data(outcomes, team_objects, charts, now)

        return report_data

    def _generate_charts(self):
        """Generate charts for the progress report."""
        charts = {}
        pr_config = self.settings.get("progress_report", {})
        chart_config = pr_config.get("charts", {})

        # Get cycle time data
        cycle_data = self.get_result(CycleTimeCalculator)
        if cycle_data is not None and len(cycle_data) > 0:
            # Generate CFD chart
            if chart_config.get("cfd", {}).get("filename"):
                charts["cfd"] = self._generate_cfd_chart(cycle_data)

            # Generate scatter plot
            if chart_config.get("scatterplot", {}).get("filename"):
                charts["scatterplot"] = self._generate_scatterplot_chart(cycle_data)

        return charts

    def _generate_cfd_chart(self, cycle_data):
        """Generate CFD chart."""
        cfd_data = calculate_cfd_data(
            cycle_data,
            [s["name"] for s in self.query_manager.settings["cycle"]],
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        cfd_data.plot(ax=ax)
        ax.set_title("Cumulative Flow Diagram")
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of items")
        set_chart_style()

        # Convert to base64 for embedding
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)

        return chart_data

    def _generate_scatterplot_chart(self, cycle_data):
        """Generate scatter plot chart."""
        scatterplot_data = calculate_scatterplot_data(cycle_data)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(
            scatterplot_data["completed_timestamp"], scatterplot_data["cycle_time"]
        )
        ax.set_title("Cycle Time Scatter Plot")
        ax.set_xlabel("Completed Date")
        ax.set_ylabel("Cycle Time (days)")
        set_chart_style()

        # Convert to base64 for embedding
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)

        return chart_data

    def _generate_report_data(self, outcomes, team_objects, charts, now):
        """Generate report data."""
        return {
            "outcomes": outcomes,
            "teams": team_objects,
            "charts": charts,
            "generated_at": now,
        }
