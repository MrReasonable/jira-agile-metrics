# API Reference

## Quick Reference

| Concept | Location |
|---------|----------|
| Calculator pattern design | [Architecture: Calculator Pattern](architecture.md#calculator-pattern) |
| Calculator code examples | [Patterns: Calculator Patterns](patterns.md#calculator-patterns) |
| Creating new calculators | [Development: Creating a New Calculator](development.md#creating-a-new-calculator) |
| Query Manager architecture | [Architecture: Query Manager Pattern](architecture.md#query-manager-pattern) |
| Configuration system design | [Architecture: Configuration System](architecture.md#configuration-system) |
| Configuration patterns | [Patterns: Configuration Patterns](patterns.md#configuration-patterns) |
| Chart generation architecture | [Architecture: Chart Generation](architecture.md#chart-generation) |
| Chart generation patterns | [Patterns: Chart Generation Patterns](patterns.md#chart-generation-patterns) |
| Error handling architecture | [Architecture: Error Handling Patterns](architecture.md#error-handling-patterns) |
| Error handling patterns | [Patterns: Error Handling Patterns](patterns.md#error-handling-patterns) |
| Testing calculators | [Testing: Calculator Testing](testing.md#calculator-testing) |

## Calculator Base Class

**Related Documentation:**

- [Calculator Pattern Architecture](architecture.md#calculator-pattern) -
  System design and execution order
- [Calculator Patterns](patterns.md#calculator-patterns) - Code examples
  and usage patterns
- [Creating New Calculators](development.md#creating-a-new-calculator) -
  Step-by-step guide

```python
class Calculator:
    """Base class for all metric calculators."""
    
    def __init__(self, query_manager, settings, results):
        """Initialize calculator with query manager and settings.
        
        Args:
            query_manager: QueryManager instance for data access
            settings: Dictionary of configuration settings
            results: Shared results dictionary for inter-calculator
communication
        """
        
    def run(self, now=None):
        """Main calculation method. Must be implemented by subclasses.
        
        Args:
            now: Optional datetime for time-based calculations
            
        Returns:
            Calculation results (typically DataFrame or dict)
        """
        
    def write(self):
        """Write output files. Called after run() phase."""
        
    def get_result(self, calculator=None, default=None):
        """Get results from a previous calculator.
        
        Args:
            calculator: Calculator class to get results from. If omitted or 
                explicitly set to None, defaults to the current calculator's 
                class (self.__class__). Both `get_result()`
                and `get_result(None)` 
                behave identically and return the current calculator's result.
            default: Default value if result not found
            
        Returns:
            Results from specified calculator or default value if not found
            
        Examples:
            # Get current calculator's result (both forms are equivalent)
            result = self.get_result()          # Returns self's result
            result = self.get_result(None)      # Returns self's result (same
            as above)
            
            # Get another calculator's result
            cycle_data = self.get_result(CycleTimeCalculator)
        """
```

## Configuration API

**Related Documentation:**

- [Configuration System Architecture](architecture.md#configuration-system) -
  Configuration flow and structure
- [Configuration Patterns](patterns.md#configuration-patterns) - Validation
  and inheritance examples

### Loading Configuration

```python
from jira_agile_metrics.config import config_to_options, ConfigError

def config_to_options(data, cwd=None, extended=False):
    """Parse YAML config data and return options dict.
    
    Args:
        data: YAML configuration data (dict or file path)
        cwd: Current working directory for relative paths
        extended: Whether to process Extends directives
        
    Returns:
        Dictionary of configuration options
        
    Raises:
        ConfigError: If configuration is invalid
    """
```

### Configuration Exceptions

```python
class ConfigError(Exception):
    """Configuration parsing errors."""
    pass
```

## Query Manager API

**Related Documentation:**

- [Query Manager Pattern Architecture](architecture.md#query-manager-pattern) -
  Design and responsibilities

```python
class QueryManager:
    """Manages JIRA/Trello queries and data fetching."""
    
    def __init__(self, jira_client, settings):
        """Initialize with JIRA client and settings.
        
        Args:
            jira_client: JIRA or Trello client instance
            settings: Dictionary of query settings
        """
        
    def get_issues(self):
        """Fetch issues from JIRA/Trello.
        
        Returns:
            List of issue dictionaries
        """
        
    def get_cycle_data(self):
        """Get processed cycle time data.
        
        Returns:
            DataFrame with cycle time data
        """
```

## JIRA Client API

```python
class JiraClient:
    """JIRA API client wrapper."""
    
    def __init__(self, domain, username, password, **kwargs):
        """Initialize JIRA client.
        
        Args:
            domain: JIRA domain URL
            username: JIRA username
            password: JIRA password or API token
            **kwargs: Additional connection options
        """
        
    def search_issues(self, jql, max_results=None):
        """Search for issues using JQL.
        
        Args:
            jql: JIRA Query Language query string
            max_results: Maximum number of results to return
            
        Returns:
            List of issue objects
        """
        
    def get_issue_changelog(self, issue_key):
        """Get changelog for an issue.
        
        Args:
            issue_key: JIRA issue key (e.g., "PROJ-123")
            
        Returns:
            List of changelog entries
        """
```

## Calculator Runner

```python
from jira_agile_metrics.calculator import run_calculators

def run_calculators(calculators, query_manager, settings):
    """Run all calculators passed in, in the order listed.
    
    Args:
        calculators: List of calculator classes to run
        query_manager: QueryManager instance
        settings: Configuration settings dictionary
        
    Returns:
        Dictionary mapping calculator classes to their results
    """
```

## Web Application API

### Flask Routes

**Dashboard:**

- `GET /` - Main dashboard with chart links

**Chart Routes:**

- `GET /burnup` - Interactive burnup chart
- `GET /burnup-forecast` - Burnup forecast with Monte Carlo
- `GET /cfd` - Cumulative Flow Diagram
- `GET /scatterplot` - Cycle time scatter plot
- `GET /histogram` - Cycle time histogram
- `GET /throughput` - Throughput chart
- `GET /wip` - Work in Progress chart
- `GET /ageingwip` - Ageing WIP chart
- `GET /netflow` - Net flow chart
- `GET /debt` - Technical debt chart
- `GET /defects-priority` - Defects by priority
- `GET /impediments` - Impediments chart
- `GET /waste` - Waste chart
- `GET /progress` - Progress report

**Configuration:**

- `POST /set_query` - Set custom JQL query

### Web Application Usage

```python
from jira_agile_metrics.webapp.app import app

# Run Flask development server
app.run(host='0.0.0.0', port=5000, debug=True)

# Or use production WSGI server
# gunicorn jira_agile_metrics.webapp.app:app
```

## Command Line Interface

### CLI Entry Point

```python
from jira_agile_metrics.cli import main

if __name__ == "__main__":
    main()
```

### CLI Arguments

```bash
jira-agile-metrics [OPTIONS] [config.yml]

Options:
  -v, --verbose          Verbose output (INFO level)
  -vv                    Very verbose output (DEBUG level)
  -n N                   Limit to N most recently updated issues
  --server HOST:PORT     Run as web server
  --output-directory DIR Write output files to directory
  --domain URL           JIRA domain URL
  --username USER        JIRA/Trello username
  --password PASS        JIRA password or API token
  --key KEY              Trello API key
  --token TOKEN          Trello API token
  --http-proxy URL       HTTP proxy URL
  --https-proxy URL      HTTPS proxy URL
```

## Utility Functions

**Related Documentation:**

- [Chart Generation Architecture](architecture.md#chart-generation) - Static
  and interactive charts
- [Chart Generation Patterns](patterns.md#chart-generation-patterns) - Code
  examples

### Chart Styling

```python
from jira_agile_metrics.chart_styling_utils import apply_chart_style

def apply_chart_style(fig, ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent styling to matplotlib chart.
    
    Args:
        fig: Matplotlib figure object
        ax: Matplotlib axes object
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
```

### Data Utilities

```python
from jira_agile_metrics.utils import breakdown_by_month,
create_monthly_bar_chart

def breakdown_by_month(data, date_field, group_field=None):
    """Break down data by month.
    
    Args:
        data: DataFrame with date field
        date_field: Name of date column
        group_field: Optional grouping field
        
    Returns:
        DataFrame with monthly breakdown
    """
```

### Type Utilities

```python
from jira_agile_metrics.config.type_utils import force_int, force_float,
force_date

def force_int(value, default=None):
    """Convert value to integer with default fallback."""
    
def force_float(value, default=None):
    """Convert value to float with default fallback."""
    
def force_date(value, default=None):
    """Convert value to date with default fallback."""
```

## Calculator Classes

### CycleTimeCalculator

```python
class CycleTimeCalculator(Calculator):
    """Calculate cycle time data for issues."""
    
    def run(self, now=None):
        """Calculate cycle time and return DataFrame."""
```

### CFDCalculator

```python
class CFDCalculator(Calculator):
    """Calculate Cumulative Flow Diagram data."""
    
    def run(self, now=None):
        """Calculate CFD data and return DataFrame."""
```

### BurnupCalculator

```python
class BurnupCalculator(Calculator):
    """Calculate burn-up chart data."""
    
    def run(self, now=None):
        """Calculate burnup data and return DataFrame."""
```

### BurnupForecastCalculator

```python
class BurnupForecastCalculator(Calculator):
    """Calculate burn-up forecast using Monte Carlo simulation."""
    
    def run(self, now=None):
        """Run Monte Carlo simulation and return forecast DataFrame."""
```

## Error Handling

**Related Documentation:**

- [Error Handling Architecture](architecture.md#error-handling-patterns) -
  Configuration, data, and API error patterns
- [Error Handling Patterns](patterns.md#error-handling-patterns) - Code examples

### Common Exceptions

```python
from jira_agile_metrics.config.exceptions import ConfigError

# Configuration errors
raise ConfigError("Missing required field: workflow")

# Data validation
if data is None or len(data) == 0:
    logger.warning("Cannot process empty data")
    return None

# API errors
try:
    issues = jira_client.search_issues(jql)
except JIRAError as e:
    logger.error(f"JIRA API error: {e}")
    raise
```

## Usage Examples

### Basic Calculator Usage

```python
from jira_agile_metrics.calculator import run_calculators
from jira_agile_metrics.calculators.cycletime import CycleTimeCalculator
from jira_agile_metrics.querymanager import QueryManager
from jira_agile_metrics.jira_client import get_jira_connection_params

# Setup
jira_params = get_jira_connection_params(domain, username, password)
jira_client = JIRA(**jira_params)
query_manager = QueryManager(jira_client, settings)

# Run calculator
results = run_calculators(
    [CycleTimeCalculator],
    query_manager,
    settings
)

# Access results
cycle_data = results[CycleTimeCalculator]
```

### Configuration Loading

```python
from jira_agile_metrics.config import config_to_options
import yaml

# Load from file
with open("config.yml") as f:
    config_data = yaml.safe_load(f)

options = config_to_options(config_data)

# Or load from dict
config_data = {
    "Connection": {"Domain": "https://my.jira.com"},
    "Query": "project=ABC",
    "Workflow": {"Backlog": "Backlog", "Done": "Done"},
}
options = config_to_options(config_data)
```

### Custom Calculator

```python
from jira_agile_metrics.calculator import Calculator
import pandas as pd

class MyCustomCalculator(Calculator):
    """Custom metric calculator."""
    
    def run(self, now=None):
        """Calculate custom metric."""
        # Get data from query manager or other calculators
        cycle_data = self.get_result(CycleTimeCalculator)
        
        # Perform calculation
        result = pd.DataFrame({
            "metric": [1, 2, 3],
            "value": [10, 20, 30]
        })
        
        return result
    
    def write(self):
        """Write output files."""
        result = self.get_result()
        if result is None:
            return
        
        if self.settings.get("my_metric_data"):
            self.write_data_file(result, self.settings["my_metric_data"])
```
