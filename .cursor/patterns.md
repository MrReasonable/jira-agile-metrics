# Common Patterns & Best Practices

## Quick Reference

| Concept | Location |
|---------|----------|
| Calculator base class | [API: Calculator Base Class](api.md#calculator-base-class) |
| Calculator pattern design | [Architecture: Calculator Pattern](architecture.md#calculator-pattern) |
| Creating new calculators | [Development: Creating a New Calculator](development.md#creating-a-new-calculator) |
| Calculator execution order | [Architecture: Calculator Dependencies](architecture.md#calculator-dependencies) |
| Query Manager API | [API: Query Manager API](api.md#query-manager-api) |
| Configuration system | [Architecture: Configuration System](architecture.md#configuration-system) |
| Chart styling utilities | [API: Chart Styling](api.md#chart-styling) |
| Test fixtures | [Testing: Test Fixtures](testing.md#test-fixtures) |
| Calculator testing | [Testing: Calculator Testing](testing.md#calculator-testing) |

## Calculator Patterns

**Related Documentation:**

- [Calculator Base Class API](api.md#calculator-base-class) - Complete API
  reference
- [Calculator Pattern Architecture](architecture.md#calculator-pattern) -
  System design details
- [Creating New Calculators](development.md#creating-a-new-calculator) -
  Step-by-step guide

### Basic Calculator Pattern

```python
from jira_agile_metrics.calculator import Calculator
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class MyCalculator(Calculator):
    """Calculate my metric."""
    
    def run(self, now=None):
        """Calculate and return results."""
        # Get prerequisite data
        cycle_data = self.get_result(CycleTimeCalculator)
        if cycle_data is None or len(cycle_data) == 0:
            logger.warning("No cycle data available")
            return None
        
        # Perform calculation
        result = self._calculate(cycle_data)
        return result
    
    def write(self):
        """Write output files."""
        result = self.get_result()
        if result is None:
            return
        
        # Write data file
        if self.settings.get("my_data"):
            self.write_data_file(result, self.settings["my_data"])
        
        # Write chart
        if self.settings.get("my_chart"):
            self.write_chart_file(result, self.settings["my_chart"])
    
    def _calculate(self, cycle_data):
        """Internal calculation logic."""
        # Implementation
        pass
```

### Calculator with Dependencies

```python
class DependentCalculator(Calculator):
    """Calculator that depends on multiple previous calculators."""
    
    def run(self, now=None):
        """Calculate using multiple data sources."""
        # Get data from multiple calculators
        cycle_data = self.get_result(CycleTimeCalculator)
        cfd_data = self.get_result(CFDCalculator)
        burnup_data = self.get_result(BurnupCalculator)
        
        # Validate all dependencies
        if any(data is None for data in [cycle_data, cfd_data, burnup_data]):
            logger.warning("Missing required calculator results")
            return None
        
        # Combine and process
        result = self._combine_data(cycle_data, cfd_data, burnup_data)
        return result
```

### Empty Data Handling Pattern

```python
def run(self, now=None):
    """Handle empty data gracefully."""
    cycle_data = self.get_result(CycleTimeCalculator)
    
    # Check for empty data
    if cycle_data is None or len(cycle_data) == 0:
        logger.warning("No data available for calculation")
        return None
    
    # Check for required columns
    required_columns = ["key", "start", "end"]
    if not all(col in cycle_data.columns for col in required_columns):
        logger.error("Missing required columns in cycle data")
        return None
    
    # Proceed with calculation
    return self._calculate(cycle_data)
```

## Chart Generation Patterns

**Related Documentation:**

- [Chart Generation Architecture](architecture.md#chart-generation) - Static
  and interactive charts
- [Chart Styling API](api.md#chart-styling) - `apply_chart_style` function
  reference

### Matplotlib Static Chart

```python
import matplotlib.pyplot as plt
from jira_agile_metrics.chart_styling_utils import apply_chart_style

def create_chart(data, output_file, title=None):
    """Create static chart with matplotlib."""
    # Validate data
    if data is None or len(data) == 0:
        logger.warning("Cannot draw chart with zero items")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot data
    ax.plot(data.index, data.values, marker='o')
    
    # Apply styling
    apply_chart_style(fig, ax, title=title or "My Chart")
    
    # Save
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()  # Important: close to free memory
```

### Bokeh Interactive Chart

```python
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import HoverTool

def create_interactive_chart(data):
    """Create interactive Bokeh chart."""
    # Create figure
    p = figure(
        title="My Chart",
        x_axis_label="Date",
        y_axis_label="Value",
        width=800,
        height=400
    )
    
    # Add hover tool
    hover = HoverTool(tooltips=[("Date", "@x"), ("Value", "@y")])
    p.add_tools(hover)
    
    # Plot data
    p.line(data.index, data.values, line_width=2)
    
    return p
```

### Chart with Windowing

```python
def create_windowed_chart(data, output_file, window=30):
    """Create chart showing only recent data."""
    if data is None or len(data) == 0:
        return
    
    # Apply window if specified
    if window > 0:
        data = data.tail(window)
    
    # Create chart
    fig, ax = plt.subplots()
    ax.plot(data.index, data.values)
    apply_chart_style(fig, ax, title=f"Last {window} days")
    plt.savefig(output_file)
    plt.close()
```

## Data Processing Patterns

**Related Documentation:**

- [Data Structures](architecture.md#data-structures) - Cycle Time
  and CFD data formats
- [Performance Optimizations](architecture.md#performance-optimizations) -
  Pandas operations and memory management

### Pandas DataFrame Operations

```python
import pandas as pd

def process_dataframe(df):
    """Process DataFrame efficiently."""
    # Use vectorized operations
    df["new_column"] = df["column1"] + df["column2"]
    
    # Filter efficiently
    filtered = df[df["status"] == "Done"]
    
    # Group and aggregate
    grouped = df.groupby("team")["cycle_time"].mean()
    
    # Avoid row-by-row iteration
    # Bad: for idx, row in df.iterrows():
    # Good: df.apply(lambda row: process_row(row), axis=1)
    
    return processed_df
```

### Date Handling Pattern

```python
import pandas as pd
from datetime import datetime

def process_dates(df):
    """Handle date columns properly."""
    # Convert to datetime if needed
    df["date_column"] = pd.to_datetime(df["date_column"])
    
    # Set timezone if needed
    df["date_column"] = df["date_column"].dt.tz_localize("UTC")
    
    # Extract date components
    df["year"] = df["date_column"].dt.year
    df["month"] = df["date_column"].dt.month
    
    return df
```

### Monthly Breakdown Pattern

```python
from jira_agile_metrics.utils import breakdown_by_month

def create_monthly_analysis(data, date_field, value_field):
    """Create monthly breakdown analysis."""
    # Use utility function
    monthly = breakdown_by_month(data, date_field)
    
    # Or manual grouping
    data["month"] = pd.to_datetime(data[date_field]).dt.to_period("M")
    monthly = data.groupby("month")[value_field].sum()
    
    return monthly
```

## Configuration Patterns

**Related Documentation:**

- [Configuration System Architecture](architecture.md#configuration-system) -
  Configuration flow and structure
- [Configuration API](api.md#configuration-api) - `config_to_options`
  and `ConfigError`

### Configuration Validation

```python
from jira_agile_metrics.config.exceptions import ConfigError

def validate_config(config):
    """Validate configuration settings."""
    # Check required fields
    if "Workflow" not in config:
        raise ConfigError("Missing required field: Workflow")
    
    # Validate workflow structure
    workflow = config["Workflow"]
    if not isinstance(workflow, dict):
        raise ConfigError("Workflow must be a dictionary")
    
    if len(workflow) < 2:
        raise ConfigError("Workflow must have at least 2 stages")
    
    return True
```

### Configuration Inheritance

```python
def load_config_with_extends(config_path):
    """Load configuration with Extends support."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Handle Extends
    if "Extends" in config:
        base_path = resolve_path(config["Extends"], config_path)
        base_config = load_config_with_extends(base_path)
        
        # Merge: base config first, then override with current
        merged = {**base_config, **config}
        # Remove Extends key
        merged.pop("Extends", None)
        return merged
    
    return config
```

## Error Handling Patterns

**Related Documentation:**

- [Error Handling Architecture](architecture.md#error-handling-patterns) -
  Configuration, data, and API error patterns
- [Error Handling API](api.md#error-handling) - Common exceptions and usage

### Graceful Degradation

```python
def calculate_with_fallback(data):
    """Calculate with fallback on error."""
    try:
        result = complex_calculation(data)
        return result
    except ValueError as e:
        logger.warning(f"Calculation failed: {e}, using fallback")
        return simple_fallback(data)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None
```

### Validation Before Processing

```python
def process_with_validation(data, settings):
    """Validate before processing."""
    # Validate inputs
    if data is None:
        raise ValueError("Data cannot be None")
    
    if len(data) == 0:
        logger.warning("Empty dataset, returning None")
        return None
    
    # Validate settings
    required_settings = ["field1", "field2"]
    missing = [s for s in required_settings if s not in settings]
    if missing:
        raise ConfigError(f"Missing required settings: {missing}")
    
    # Proceed with processing
    return process(data, settings)
```

## Logging Patterns

### Structured Logging

```python
import logging

logger = logging.getLogger(__name__)

def process_data(data):
    """Process data with logging."""
    logger.info("Starting data processing")
    logger.debug(f"Data shape: {data.shape}")
    
    try:
        result = calculate(data)
        logger.info(f"Processing complete: {len(result)} items")
        return result
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise
```

### Progress Logging

```python
def process_large_dataset(data):
    """Log progress for long-running operations."""
    total = len(data)
    logger.info(f"Processing {total} items")
    
    for i, item in enumerate(data, 1):
        if i % 100 == 0:
            logger.info(f"Progress: {i}/{total} ({100*i/total:.1f}%)")
        process_item(item)
    
    logger.info("Processing complete")
```

## Testing Patterns

**Related Documentation:**

- [Testing Guidelines](testing.md) - Complete testing documentation
- [Test Fixtures](testing.md#test-fixtures) - Detailed fixture documentation
- [Calculator Testing](testing.md#calculator-testing) - Calculator test patterns

**Note on Fixture Names:** The canonical fixture names used in tests are:

- `query_manager` - QueryManager instance for data access
- `simple_cycle_settings` - Returns tuple of `(settings_dict, output_file)` with basic cycle time settings

For complete fixture documentation including all available fixtures, their
types, purposes, and usage examples, see the [Test Fixtures](testing.md#test-fixtures) section in the testing guidelines.

### Testing Fixtures Reference

The repository provides pytest fixtures in two locations:

- **Unit test fixtures**: `jira_agile_metrics/conftest.py` - For tests
  alongside source files (`*_test.py`)
- **Functional test fixtures**: `jira_agile_metrics/tests/functional/conftest.py` - For tests in `tests/functional/`

#### Functional Test Fixtures

These fixtures are available for functional tests (in `tests/functional/`):

- **`jira_client`** → `FileJiraClient` instance that reads from JSON fixture
  files (`tests/fixtures/jira/`)
- **`query_manager`** → `QueryManager` instance configured with minimal
  settings, uses `jira_client` fixture
- **`simple_cycle_settings`** → Returns `tuple[dict, Path]` of `(settings_dict, output_file_path)` with:
  - Settings dict: cycle config, committed/done columns, queries,
    cycle_time_data
  - Output file: Temporary CSV path for cycle time data

**Example usage:**

```python
def test_calculator(query_manager, simple_cycle_settings):
    settings, output_file = simple_cycle_settings
    # Modify settings as needed
    settings["my_data"] = str(output_file)
    calculator = MyCalculator(query_manager, settings, {})
    calculator.write()
    assert output_file.exists()
```

**See examples:**

- [`test_cfd_functional.py:12`](../../jira_agile_metrics/tests/functional/test_cfd_functional.py#L12) - CFD calculator test
- [`test_burnup_functional.py:10`](../../jira_agile_metrics/tests/functional/test_burnup_functional.py#L10) - Burnup calculator test
- [`test_burnup_forecast_functional.py:213`](../../jira_agile_metrics/tests/functional/test_burnup_forecast_functional.py#L213) - Forecast output format tests

#### Unit Test Fixtures

These fixtures are available for unit tests (files named `*_test.py` alongside
source files):

- **`base_minimal_settings`** → Minimal settings dict with cycle config,
  queries, column names
- **`base_custom_settings`** → Extended settings with custom fields (Release,
  Team, Estimate) and progress_report config
- **`base_minimal_fields`** → List of basic JIRA field definitions (no custom
  fields)
- **`base_custom_fields`** → Field definitions including custom fields (Team,
  Size, Releases)
- **`minimal_query_manager`** → `QueryManager` with minimal setup (uses
  `FauxJIRA` mock, no custom fields)
- **`custom_query_manager`** → `QueryManager` capable of handling custom fields
- **`base_minimal_cycle_time_results`** → Minimal cycle time results dict
  (mimics `CycleTimeCalculator` output)
- **`large_cycle_time_results`** → Larger cycle time results
  for testing with more data
- **`base_minimal_cycle_time_columns`** → Column names
  for cycle time results without custom fields
- **`base_cfd_columns`** → Column names for CFD calculator results
- **`minimal_cfd_results`** → Results dict with CFD data included
- **`mock_trello_api`** → Mock Trello API for testing Trello integrations

**Example usage:**

```python
def test_calculator_with_custom_fields(custom_query_manager,
base_custom_settings):
    calculator = MyCalculator(custom_query_manager, base_custom_settings, {})
    result = calculator.run()
    assert result is not None
```

#### Creating Custom Fixtures

**For functional tests**, add fixtures to
`jira_agile_metrics/tests/functional/conftest.py`:

```python
@pytest.fixture()
def my_custom_settings(tmp_path, simple_cycle_settings):
    """Create custom settings for specific test needs."""
    settings_dict, _ = simple_cycle_settings
    custom_output = tmp_path / "my_output.csv"
    
    return {
        **settings_dict,
        "my_custom_data": str(custom_output),
        "my_custom_setting": "value",
    }, custom_output
```

**For unit tests**, add fixtures to `jira_agile_metrics/conftest.py`
or in your test file:

```python
# In your test file (e.g., my_calculator_test.py)
@pytest.fixture()
def my_custom_settings(base_minimal_settings):
    """Extend base settings for my calculator."""
    return {
        **base_minimal_settings,
        "my_calculator_data": "output.csv",
    }
```

**For fixtures specific to a single test file**, define them in that file:

```python
# In my_calculator_test.py
@pytest.fixture()
def calculator_instance(minimal_query_manager, base_minimal_settings):
    """Create a calculator instance for testing."""
    return MyCalculator(minimal_query_manager, base_minimal_settings, {})
```

### Calculator Test Pattern

```python
import pytest
from jira_agile_metrics.calculators.my_calculator import MyCalculator

def test_calculator_basic(query_manager, simple_cycle_settings):
    """Test calculator with basic settings."""
    settings, output_file = simple_cycle_settings
    settings["my_data"] = str(output_file)
    
    calculator = MyCalculator(query_manager, settings, {})
    result = calculator.run()
    
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    
    calculator.write()
    assert output_file.exists()
```

### Mock API Pattern

```python
from unittest.mock import Mock, patch

def test_with_mocked_api():
    """Test with mocked external API."""
    mock_client = Mock()
    mock_client.search_issues.return_value = [
        Mock(key="TEST-1", summary="Test Issue")
    ]
    
    with patch("jira_agile_metrics.jira_client.JIRA", return_value=mock_client):
        # Test code here
        pass
```

## Performance Patterns

**Related Documentation:**

- [Performance Optimizations](architecture.md#performance-optimizations) -
  Caching, data processing, and memory management

### Caching Pattern

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_calculation(param1, param2):
    """Cache expensive calculations."""
    # Expensive operation
    return result
```

### Batch Processing Pattern

```python
def process_in_chunks(data, chunk_size=1000):
    """Process large datasets in chunks."""
    results = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        result = process_chunk(chunk)
        results.append(result)
    
    return pd.concat(results)
```

### Memory-Efficient Pattern

```python
def process_large_file(file_path):
    """Process large file without loading all into memory."""
    # Use chunking for CSV
    for chunk in pd.read_csv(file_path, chunksize=1000):
        process_chunk(chunk)
        # Chunk is automatically garbage collected
```

## Best Practices Summary

1. **[Critical]** **Always validate inputs** before processing - prevents
   runtime errors and improves error messages; see [Validation Before Processing pattern](#validation-before-processing)
1. **[Critical]** **Handle empty data gracefully** - return None
   or empty DataFrame to avoid crashes; see [Empty Data Handling Pattern](#empty-data-handling-pattern)
1. **[Recommended]** **Use vectorized operations** in pandas - significantly
   faster than row-by-row iteration; see [Pandas DataFrame Operations](#pandas-dataframe-operations)
1. **[Critical]** **Close matplotlib figures** to free memory - prevents memory
   leaks in long-running processes; see [Matplotlib Static Chart](#matplotlib-static-chart)
1. **[Recommended]** **Log important events** at appropriate levels - enables
   debugging and monitoring; see [Structured Logging](#structured-logging)
1. **[Critical]** **Use type hints** for better code clarity - improves IDE
   support and documentation; see development guidelines
1. **[Critical]** **Write tests** for all new functionality - ensures
   correctness and prevents regressions; see [Testing Patterns](#testing-patterns)
1. **[Critical]** **Document public APIs** with docstrings - enables
   maintainability and usage; see [Basic Calculator Pattern](#basic-calculator-pattern)
1. **[Critical]** **Follow calculator pattern** for consistency - ensures
   calculators work together correctly; see [Basic Calculator Pattern](#basic-calculator-pattern)
1. **[Optional]** **Cache expensive operations** when appropriate - improves
   performance for repeated calculations; see [Caching Pattern](#caching-pattern)
