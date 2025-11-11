# Testing Guidelines

## Quick Reference

| Concept | Location |
|---------|----------|
| Calculator base class API | [API Reference: Calculator Base Class](api.md#calculator-base-class) |
| Calculator pattern design | [Architecture: Calculator Pattern](architecture.md#calculator-pattern) |
| Calculator code examples | [Patterns: Calculator Patterns](patterns.md#calculator-patterns) |
| Calculator testing patterns | [Patterns: Testing Patterns](patterns.md#testing-patterns) |
| Creating new calculators | [Development: Creating a New Calculator](development.md#creating-a-new-calculator) |
| Development workflow | [Development: Development Workflow](development.md#development-workflow) |

## Test Organization

### Test Directory Structure

```text
jira_agile_metrics/
├── calculators/
│   ├── cycletime.py
│   ├── cycletime_test.py          # Unit tests
│   └── ...
├── tests/
│   ├── functional/                # Functional tests
│   │   ├── conftest.py
│   │   ├── test_cycletime_functional.py
│   │   └── ...
│   ├── e2e/                       # End-to-end tests
│   │   ├── e2e_config.py
│   │   ├── test_cli_e2e.py
│   │   └── ...
│   ├── fixtures/                  # Test data
│   │   ├── jira/                  # JIRA API responses
│   │   └── expected/               # Expected outputs
│   └── helpers/                   # Test utilities
│       ├── assertions.py
│       ├── csv_utils.py
│       └── dataframe_utils.py
```

### Test Types

**Unit Tests:**

- Test individual functions/classes in isolation
- Mock external dependencies
- Fast execution
- Located alongside source files: `{module}_test.py`

**Functional Tests:**

- Test calculators with real data fixtures
- Verify output file generation
- Check data correctness
- Located in `tests/functional/`

**E2E Tests:**

- Test full CLI workflow
- Verify end-to-end data flow
- Test configuration loading
- Located in `tests/e2e/`

## Test Fixtures

### Fixture Locations

Fixtures are defined in `conftest.py` files that pytest automatically discovers:

- **Functional tests**: `jira_agile_metrics/tests/functional/conftest.py`

  - Provides fixtures for testing calculators with real data fixtures
  - Uses `FileJiraClient` with JSON fixture data from `tests/fixtures/jira/`

- **Unit tests**: `jira_agile_metrics/conftest.py`

  - Provides fixtures for testing individual components in isolation
  - Uses `FauxJIRA` mock objects for fast, isolated tests

### Functional Test Fixtures

These fixtures are available for tests in `tests/functional/`:

#### `jira_client`

- **Type**: `FileJiraClient`
- **Purpose**: Provides a JIRA client that reads from JSON fixture files
- **Location**: `jira_agile_metrics/tests/functional/conftest.py:49`
- **Usage**:

```python
def test_something(jira_client):
    # jira_client is a FileJiraClient instance
    issues = jira_client.search_issues("project=TEST")
```

#### `query_manager`

- **Type**: `QueryManager`
- **Purpose**: Provides a QueryManager instance configured for functional tests
- **Location**: `jira_agile_metrics/tests/functional/conftest.py:68`
- **Dependencies**: Requires `jira_client` fixture
- **Settings**: Minimal QueryManager settings (attributes, known_values,
  max_results)
- **Usage**:

```python
def test_calculator(query_manager):
    # query_manager is ready to use with test data
    data = query_manager.get_cycle_time_data()
```

#### `simple_cycle_settings`

- **Type**: `tuple[dict, Path]`
- **Purpose**: Returns a tuple of `(settings_dict, output_file_path)` with
  basic cycle time configuration
- **Location**: `jira_agile_metrics/tests/functional/conftest.py:102`
- **Returns**:
  - First element: Settings dictionary with cycle config, columns, queries,
    and cycle_time_data
  - Second element: Path object for the output CSV file (in temporary directory)
- **Usage**:

```python
def test_calculator(query_manager, simple_cycle_settings):
    settings_dict, output_file = simple_cycle_settings
    # Modify settings if needed
    settings = {**settings_dict, "my_setting": "value"}
    calculator = MyCalculator(query_manager, settings, {})
    calculator.write()
    assert output_file.exists()
```

**Settings Dictionary Contents:**

- `cycle`: Standard cycle configuration (from `e2e_config._get_standard_cycle_config()`)
- `committed_column`: "Committed"
- `done_column`: "Done"
- `attributes`: Empty dict
- `queries`: `[{"jql": "project=TEST"}]`
- `query_attribute`: None
- `cycle_time_data`: List containing the output CSV path

### Unit Test Fixtures

These fixtures are available for unit tests (files named `*_test.py` alongside
source files):

#### `base_minimal_settings`

- **Type**: `dict`
- **Purpose**: Minimal settings dictionary required for QueryManager
  and cycle time calculations
- **Location**: `jira_agile_metrics/conftest.py:54`
- **Contents**:
  - `attributes`: `{}`
  - `known_values`: `{"Release": ["R1", "R3"]}`
  - `max_results`: `None`
  - `verbose`: `False`
  - `cycle`: Common cycle configuration
  - `query_attribute`: `None`
  - `queries`: `[{"jql": "(filter=123)", "value": None}]`
  - `backlog_column`: "Backlog"
  - `committed_column`: "Committed"
  - `done_column`: "Done"

#### `base_custom_settings`

- **Type**: `dict`
- **Purpose**: Settings with custom fields and attributes (extends
  `base_minimal_settings`)
- **Location**: `jira_agile_metrics/conftest.py:73`
- **Additional fields**: Includes attributes for Release, Team, Estimate,
  and progress_report configuration

#### `base_minimal_fields`

- **Type**: `list[dict]`
- **Purpose**: List of basic JIRA field definitions (no custom fields)
- **Location**: `jira_agile_metrics/conftest.py:115`
- **Fields**: summary, issuetype, status, resolution, created, updated,
  project, reporter, assignee, priority, type, labels, components, fixVersions, resolutiondate, customfield_100

#### `base_custom_fields`

- **Type**: `list[dict]`
- **Purpose**: Field definitions including custom fields (extends
  `base_minimal_fields`)
- **Location**: `jira_agile_metrics/conftest.py:138`
- **Additional fields**: customfield_001 (Team), customfield_002 (Size),
  customfield_003 (Releases)

#### `minimal_query_manager`

- **Type**: `QueryManager`
- **Purpose**: QueryManager with minimal setup (no custom fields)
- **Location**: `jira_agile_metrics/conftest.py:183`
- **Dependencies**: `base_minimal_fields`, `base_minimal_settings`
- **Uses**: `FauxJIRA` mock with empty issues list

#### `custom_query_manager`

- **Type**: `QueryManager`
- **Purpose**: QueryManager capable of handling custom fields
- **Location**: `jira_agile_metrics/conftest.py:190`
- **Dependencies**: `base_custom_fields`, `base_custom_settings`

#### `base_minimal_cycle_time_results`

- **Type**: `dict[type, pd.DataFrame]`
- **Purpose**: Minimal cycle time results dictionary (mimics
  CycleTimeCalculator output)
- **Location**: `jira_agile_metrics/conftest.py:203`

#### `large_cycle_time_results`

- **Type**: `dict[type, pd.DataFrame]`
- **Purpose**: Larger cycle time results for testing with more data
- **Location**: `jira_agile_metrics/conftest.py:213`

#### `base_minimal_cycle_time_columns`

- **Type**: `list[str]`
- **Purpose**: Column names for cycle time results without custom fields
- **Location**: `jira_agile_metrics/conftest.py:149`

### Creating Custom Fixtures

#### For Functional Tests

Add fixtures to `jira_agile_metrics/tests/functional/conftest.py`:

```python
@pytest.fixture()
def my_custom_settings(tmp_path, simple_cycle_settings):
    """Create custom settings for my specific test needs."""
    settings_dict, _ = simple_cycle_settings
    custom_output = tmp_path / "my_output.csv"
    
    return {
        **settings_dict,
        "my_custom_data": str(custom_output),
        "my_custom_setting": "value",
    }, custom_output
```

#### For Unit Tests

Add fixtures to `jira_agile_metrics/conftest.py` or in your test file:

```python
# In your test file
@pytest.fixture()
def my_custom_settings(base_minimal_settings):
    """Extend base settings for my calculator."""
    return {
        **base_minimal_settings,
        "my_calculator_data": "output.csv",
    }
```

#### In-Test Fixtures

For fixtures specific to a single test file, define them in that file:

```python
# In my_calculator_test.py
@pytest.fixture()
def calculator_instance(query_manager, settings):
    """Create a calculator instance for testing."""
    return MyCalculator(query_manager, settings, {})
```

### Fixture Usage Examples

**Functional Test Example:**
See `jira_agile_metrics/tests/functional/test_cycletime_functional.py`:

```python
def test_cycletime_generates_expected_csv(query_manager, simple_cycle_settings):
    settings, output_csv = simple_cycle_settings
    calculator = CycleTimeCalculator(query_manager, settings, {})
    calculator.run()
    calculator.write()
    assert output_csv.exists()
```

**Unit Test Example:**
See `jira_agile_metrics/calculators/cycletime_test.py`:

```python
def test_empty(base_custom_fields, test_settings):
    jira = FauxJIRA(fields=base_custom_fields, issues=[])
    query_manager = QueryManager(jira, test_settings)
    calculator = CycleTimeCalculator(query_manager, test_settings, {})
    result = calculator.run()
    assert result is None
```

### Helper Functions

The functional test conftest also provides helper functions (not fixtures):

- `get_burnup_base_settings(base_settings)`: Get settings for burnup-related tests
- `get_default_forecast_settings()`: Get default forecast test settings
- `run_forecast_calculators(query_mgr, settings)`: Run forecast calculator chain
- `validate_forecast_result_structure(forecast_result)`: Validate forecast
  DataFrame structure

## Test Patterns

### Calculator Testing

**Related Documentation:**

- [Calculator Patterns](patterns.md#calculator-patterns) - Calculator
  implementation examples
- [Calculator Pattern Architecture](architecture.md#calculator-pattern) -
  System design
- [Calculator Base Class API](api.md#calculator-base-class) - API reference
- [Creating New Calculators](development.md#creating-a-new-calculator) -
  Development guide

**Basic Pattern:**

```python
def test_calculator_basic(query_manager, simple_cycle_settings):
    """Test calculator with basic settings."""
    settings, output_file = simple_cycle_settings
    calculator = MyCalculator(query_manager, settings, {})
    
    result = calculator.run()
    assert result is not None
    
    calculator.write()
    assert output_file.exists()
```

**With Dependencies:**

```python
def test_calculator_with_dependencies(query_manager, simple_cycle_settings):
    """Test calculator that depends on other calculators."""
    # Run prerequisite calculator first
    settings_dict, _ = simple_cycle_settings
    cycle_calc = CycleTimeCalculator(query_manager, settings_dict, {})
    cycle_data = cycle_calc.run()
    
    # Now run dependent calculator
    settings = {**settings_dict, "my_setting": "value"}
    my_calc = MyCalculator(query_manager, settings, {CycleTimeCalculator:
    cycle_data})
    result = my_calc.run()
    
    assert result is not None
```

**Empty Data Handling:**

```python
def test_calculator_with_empty_data(query_manager, simple_cycle_settings):
    """Test calculator handles empty data gracefully."""
    # Create settings with no matching issues
    settings_dict, _ = simple_cycle_settings
    settings = {**settings_dict, "queries": [{"jql": "project=NONEXISTENT"}]}
    calculator = MyCalculator(query_manager, settings, {})
    
    result = calculator.run()
    # Should not crash, may return None or empty DataFrame
    assert result is None or len(result) == 0
```

### Data Validation Testing

**CSV Output Validation:**

```python
def test_csv_output_format(tmp_path, query_manager, settings):
    """Test CSV output has correct format."""
    output_file = tmp_path / "output.csv"
    settings["my_data"] = str(output_file)
    
    calculator = MyCalculator(query_manager, settings, {})
    calculator.run()
    calculator.write()
    
    # Read and validate
    df = pd.read_csv(output_file)
    assert "expected_column" in df.columns
    assert len(df) > 0
    assert df["expected_column"].dtype == "expected_type"
```

**DataFrame Content Validation:**

```python
def test_dataframe_content(query_manager, settings):
    """Test DataFrame has correct content."""
    calculator = MyCalculator(query_manager, settings, {})
    result = calculator.run()
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == expected_count
    assert (result["column"] >= 0).all()  # All values non-negative
    assert result["date_column"].dtype == "datetime64[ns]"
```

### Chart Testing

**Chart File Generation:**

```python
def test_chart_generation(tmp_path, query_manager, settings):
    """Test chart file is generated."""
    chart_file = tmp_path / "chart.png"
    settings["my_chart"] = str(chart_file)
    
    calculator = MyCalculator(query_manager, settings, {})
    calculator.run()
    calculator.write()
    
    assert chart_file.exists()
    assert chart_file.stat().st_size > 0  # Non-empty file
```

## Running Tests

### Basic Commands

```bash
# All tests
pytest

# Specific test file
pytest jira_agile_metrics/calculators/cycletime_test.py

# Specific test
pytest jira_agile_metrics/calculators/cycletime_test.py::test_specific_case

# With markers
pytest -m "not functional"  # Skip functional tests
pytest -m "e2e"              # Only e2e tests
```

### Test Markers

**Available Markers:**

- `@pytest.mark.functional` - Functional tests
- `@pytest.mark.e2e` - End-to-end tests

**Using Markers:**

```python
@pytest.mark.functional
def test_functional_case():
    """This is a functional test."""
    pass
```

### Coverage

```bash
# Generate coverage report
pytest --cov=jira_agile_metrics --cov-report=html

# View coverage
open htmlcov/index.html

# Coverage with specific threshold
pytest --cov=jira_agile_metrics --cov-report=term-missing --cov-fail-under=80
```

## Test Best Practices

### Do's

✅ **Test behavior, not implementation:**

```python
# Good: Test the result
assert result["cycle_time"] == 10

# Bad: Test internal state
assert calculator._internal_counter == 5
```

✅ **Use descriptive test names:**

```python
# Good
def test_calculator_handles_empty_data_gracefully():

# Bad
def test_calculator():
```

✅ **Test edge cases:**

```python
def test_calculator_with_zero_issues():
def test_calculator_with_missing_dates():
def test_calculator_with_invalid_workflow():
```

✅ **Keep tests independent:**

```python
# Each test should be able to run in isolation
# Don't rely on test execution order
```

✅ **Use fixtures for common setup:**

```python
# Good: Reusable fixture
@pytest.fixture()
def calculator_settings():
    return create_settings()

# Bad: Duplicated setup in each test
```

### Don'ts

❌ **Don't test external libraries:**

```python
# Bad: Testing pandas functionality
def test_pandas_dataframe():
    df = pd.DataFrame({"a": [1, 2]})
    assert len(df) == 2  # This is testing pandas, not our code
```

❌ **Don't make tests too complex:**

```python
# Bad: Complex test with many assertions
def test_everything():
    # 50 lines of setup
    # 20 assertions
    # Hard to understand what's being tested
```

❌ **Don't ignore test failures:**

```python
# Bad: Ignoring known failures
@pytest.mark.xfail
def test_broken_feature():
    # Should fix the feature, not ignore the test
```

❌ **Don't use real API calls in unit tests:**

```python
# Bad: Real API call
def test_with_real_jira():
    jira = JIRA("https://real.jira.com", auth=("user", "pass"))
    # This will fail without network, credentials, etc.
```

## Debugging Tests

### Using pytest Debugger

```python
def test_with_debugger():
    """Test with debugger breakpoint."""
    result = calculate()
    import pdb; pdb.set_trace()  # Breakpoint here
    assert result is not None
```

### Print Debugging

```python
def test_with_print():
    """Test with print statements."""
    result = calculate()
    print(f"Result: {result}")  # Use -s flag to see output
    print(f"Shape: {result.shape}")
    assert result is not None
```

### Logging in Tests

```python
import logging

def test_with_logging():
    """Test with logging."""
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    result = calculate()
    logger.debug("Result: %s", result)
    assert result is not None
```

## Continuous Integration

### GitHub Actions

**Test Workflow:**

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest
```

### Pre-commit Hooks

**Running Tests Before Commit:**

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Manual run
pre-commit run --all-files
```

## Test Maintenance

### Updating Test Data

When API responses change:

1. Update fixture files in `tests/fixtures/jira/`
1. Regenerate expected outputs if needed
1. Run tests to verify

### Refactoring Tests

When refactoring code:

1. Run all tests first
1. Refactor incrementally
1. Run tests after each change
1. Update tests if behavior changes

### Test Performance

**Slow Tests:**

- Mark with `@pytest.mark.slow`
- Run separately: `pytest -m "not slow"`

**Parallel Execution:**

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest -n auto
```
