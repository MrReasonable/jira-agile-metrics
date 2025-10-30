"""Data models for progress report calculator.

This module contains the data classes used by the progress report calculator.
"""


class EpicProcessingParams:
    """Parameters for epic processing."""

    def __init__(self, params_dict):
        """Initialize with a dictionary of parameters.

        Args:
            params_dict: Dictionary containing parameters with required keys:
                outcomes: List of Outcome objects (required)
                epic_config: Dictionary of epic configuration fields (required)
                team_lookup: Dictionary mapping team names to Team objects (required)
                story_query_template: String template for story queries (required)
                cycle: Dictionary or object representing cycle configuration (required)
                backlog_column: String name of backlog column (required)
                done_column: String name of done column (required)
                team_epics: Dictionary mapping team names to lists of epics
                    (required)
                default_team: Team object or None (optional, defaults to None)
                epic_team_field: String field name for epic team or None/empty
                    (optional, defaults to None)

        Raises:
            ValueError: If required fields are missing or have invalid types.
        """
        # Validate required fields
        required_fields = {
            "outcomes": (list, tuple),
            "epic_config": dict,
            "team_lookup": dict,
            "team_epics": dict,
            "story_query_template": str,
            "backlog_column": str,
            "done_column": str,
        }

        for field_name, expected_types in required_fields.items():
            value = params_dict.get(field_name)
            if value is None:
                raise ValueError(
                    f"EpicProcessingParams: required field '{field_name}' "
                    f"is missing or None"
                )
            # Check type for fields
            if isinstance(expected_types, tuple):
                if not isinstance(value, expected_types):
                    raise ValueError(
                        f"EpicProcessingParams: field '{field_name}' "
                        f"must be one of {expected_types}, "
                        f"got {type(value).__name__}"
                    )
            elif not isinstance(value, expected_types):
                raise ValueError(
                    f"EpicProcessingParams: field '{field_name}' "
                    f"must be {expected_types.__name__}, "
                    f"got {type(value).__name__}"
                )

        # Validate cycle separately (required but no type check -
        # can be dict or custom object)
        if "cycle" not in params_dict or params_dict["cycle"] is None:
            raise ValueError(
                "EpicProcessingParams: required field 'cycle' is missing or None"
            )

        # Validate optional fields with defaults
        default_team = params_dict.get("default_team")
        epic_team_field = params_dict.get("epic_team_field")

        self.params = {
            "outcomes": params_dict["outcomes"],
            "epic_config": params_dict["epic_config"],
            "team_lookup": params_dict["team_lookup"],
            "team_epics": params_dict["team_epics"],
            "default_team": default_team,
            "epic_team_field": epic_team_field,
            "story_query_template": params_dict["story_query_template"],
            "cycle": params_dict["cycle"],
            "backlog_column": params_dict["backlog_column"],
            "done_column": params_dict["done_column"],
        }

    def get_params_dict(self):
        """Get parameters as a dictionary.

        Returns:
            Dictionary containing all parameters, with None values preserved
            for optional fields (default_team, epic_team_field).
        """
        return self.params.copy()

    def __str__(self):
        """Return string representation.

        Safely handles None/missing values to prevent exceptions.
        """
        outcomes = self.params.get("outcomes") or []
        outcomes_count = len(outcomes)
        return f"EpicProcessingParams(outcomes={outcomes_count})"


class Outcome:
    """Represents an outcome with associated epics and metadata."""

    def __init__(self, outcome_data):
        """Initialize an Outcome.

        Args:
            outcome_data: Dictionary containing outcome information with keys:
                key: Unique key for the outcome
                name: Name of the outcome
                deadline: Deadline for the outcome
                epic_query: Query for finding epics
                epics: List of Epic objects
                forecast: Forecast object
        """
        self.key = outcome_data["key"]
        self.name = outcome_data["name"]
        self.deadline = outcome_data["deadline"]
        self.epic_query = outcome_data["epic_query"]
        self.epics = outcome_data.get("epics", [])
        self.forecast = outcome_data.get("forecast")

    def get_key(self):
        """Get the outcome key."""
        return self.key

    def get_name(self):
        """Get the outcome name."""
        return self.name

    def get_deadline(self):
        """Get the outcome deadline."""
        return self.deadline

    def get_epics(self):
        """Get the list of epics."""
        return self.epics

    def __repr__(self):
        """Return string representation of Outcome."""
        return f"Outcome(key={self.key!r}, name={self.name!r})"


class Team:
    """Represents a team with throughput and work-in-progress settings."""

    def __init__(self, name, **kwargs):
        """Initialize a Team.

        Args:
            name: Name of the team
            **kwargs: Additional keyword arguments including:
                wip: Work in progress limit (default: 1)
                min_throughput: Minimum throughput value
                max_throughput: Maximum throughput value
                throughput_samples: Query for throughput samples
                throughput_samples_window: Window for throughput samples
                throughput_samples_cycle_times: Cycle times for throughput samples
                sampler: Throughput sampler function
        """
        self.name = name
        self.wip = kwargs.get("wip", 1)
        self.throughput_config = {
            "min_throughput": kwargs.get("min_throughput"),
            "max_throughput": kwargs.get("max_throughput"),
            "throughput_samples": kwargs.get("throughput_samples"),
            "throughput_samples_window": kwargs.get("throughput_samples_window"),
            "throughput_samples_cycle_times": kwargs.get(
                "throughput_samples_cycle_times"
            ),
            "sampler": kwargs.get("sampler"),
        }

    def get_name(self):
        """Get the team name."""
        return self.name

    def get_wip(self):
        """Get the work in progress limit."""
        return self.wip

    def set_wip(self, wip):
        """Set the work in progress limit."""
        self.wip = wip

    def get_throughput_range(self):
        """Get the throughput range."""
        return (
            self.throughput_config["min_throughput"],
            self.throughput_config["max_throughput"],
        )

    # Direct attribute access for test compatibility
    @property
    def throughput_samples_cycle_times(self):
        """Get throughput samples cycle times."""
        return self.throughput_config["throughput_samples_cycle_times"]

    @property
    def sampler(self):
        """Get sampler."""
        return self.throughput_config["sampler"]

    def __repr__(self):
        """Return string representation of Team."""
        return f"Team(name={self.name!r}, wip={self.wip})"


class Epic:
    """Represents an epic with associated stories and forecasting data."""

    def __init__(self, epic_data):
        """Initialize an Epic.

        Args:
            epic_data: Dictionary containing epic information with keys:
                key: Unique key for the epic (required)
                summary: Summary description of the epic (required)
                status: Current status of the epic (required)
                resolution: Resolution status (required)
                resolution_date: Date when epic was resolved (required)
                min_stories: Minimum number of stories (required)
                max_stories: Maximum number of stories (required)
                team_name: Name of the team working on the epic (required)
                deadline: Deadline for the epic (required)
                story_query: Query for finding stories (optional)
                story_cycle_times: Cycle times for stories (optional)
                stories_raised: Number of stories raised (optional)
                stories_in_backlog: Number of stories in backlog (optional)
                stories_in_progress: Number of stories in progress (optional)
                stories_done: Number of stories done (optional)
                first_story_started: Date first story started (optional)
                last_story_finished: Date last story finished (optional)
                team: Team object (optional)
                outcome: Outcome object (optional)
                forecast: Forecast object (optional)

        Raises:
            ValueError: If required fields are missing.
        """
        # Validate required fields
        required_fields = [
            "key",
            "summary",
            "status",
            "resolution",
            "resolution_date",
            "min_stories",
            "max_stories",
            "team_name",
            "deadline",
        ]
        missing_fields = [field for field in required_fields if field not in epic_data]
        if missing_fields:
            raise ValueError(
                f"Epic: required fields missing: {', '.join(missing_fields)}"
            )

        # Core epic data
        self.key = epic_data["key"]
        self.summary = epic_data["summary"]

        # All other data grouped together
        self.data = {
            "status": epic_data["status"],
            "resolution": epic_data["resolution"],
            "resolution_date": epic_data["resolution_date"],
            "min_stories": epic_data["min_stories"],
            "max_stories": epic_data["max_stories"],
            "story_query": epic_data.get("story_query"),
            "story_cycle_times": epic_data.get("story_cycle_times"),
            "stories_raised": epic_data.get("stories_raised"),
            "stories_in_backlog": epic_data.get("stories_in_backlog"),
            "stories_in_progress": epic_data.get("stories_in_progress"),
            "stories_done": epic_data.get("stories_done"),
            "first_story_started": epic_data.get("first_story_started"),
            "last_story_finished": epic_data.get("last_story_finished"),
            "team_name": epic_data["team_name"],
            "deadline": epic_data["deadline"],
        }

        # Related objects
        self.team = epic_data.get("team")
        self.outcome = epic_data.get("outcome")
        self.forecast = epic_data.get("forecast")

    def get_key(self):
        """Get the epic key."""
        return self.key

    def get_summary(self):
        """Get the epic summary."""
        return self.summary

    def get_status(self):
        """Get the epic status."""
        return self.data["status"]

    def get_team(self):
        """Get the epic team."""
        return self.team

    # Direct attribute access for test compatibility
    @property
    def max_stories(self):
        """Get max stories."""
        return self.data["max_stories"]

    @property
    def story_cycle_times(self):
        """Get story cycle times."""
        return self.data["story_cycle_times"]

    @property
    def stories_raised(self):
        """Get stories raised."""
        return self.data["stories_raised"]

    @property
    def stories_in_backlog(self):
        """Get stories in backlog."""
        return self.data["stories_in_backlog"]

    @property
    def stories_in_progress(self):
        """Get stories in progress."""
        return self.data["stories_in_progress"]

    @property
    def stories_done(self):
        """Get stories done."""
        return self.data["stories_done"]

    @property
    def first_story_started(self):
        """Get first story started."""
        return self.data["first_story_started"]

    @property
    def last_story_finished(self):
        """Get last story finished."""
        return self.data["last_story_finished"]

    @property
    def min_stories(self):
        """Get min stories."""
        return self.data["min_stories"]

    def __repr__(self):
        """Return string representation of Epic."""
        return f"Epic(key={self.key!r}, summary={self.summary!r})"


class Forecast:
    """Represents a forecast with quantiles and deadline information."""

    def __init__(self, forecast_data):
        """Initialize a Forecast.

        Args:
            forecast_data: Dictionary containing forecast information with keys:
                quantiles: Dictionary of quantile values
                deadline: Deadline for the forecast
                target: Target completion date
                confidence: Confidence level
        """
        self.quantiles = forecast_data.get("quantiles", {})
        self.deadline = forecast_data.get("deadline")
        self.target = forecast_data.get("target")
        self.confidence = forecast_data.get("confidence")

    def get_quantiles(self):
        """Get the quantiles."""
        return self.quantiles

    def get_deadline(self):
        """Get the deadline."""
        return self.deadline

    def get_target(self):
        """Get the target."""
        return self.target

    def get_confidence(self):
        """Get the confidence."""
        return self.confidence

    def __repr__(self):
        """Return string representation of Forecast."""
        return f"Forecast(deadline={self.deadline!r}, target={self.target!r})"
