# Testing Guidelines

## Test Organization

### Test Directory Structure

```
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

### Common Fixtures (conftest.py)

**JIRA Client:**
```python
@pytest.fixture()
def jira_client():
    """Create a FileJiraClient fixture using test data."""
    return FileJiraClient(fixtures_path("jira"))
```

**Query Manager:**
```python
@pytest.fixture()
def query_manager(request):
    """Create a QueryManager fixture with minimal settings."""
    client = request.getfixturevalue("jira_client")
    return QueryManager(client, settings=_create_query_manager_settings())
```

**Settings:**
```python
@pytest.fixture()
def simple_cycle_settings(tmp_path):
    """Create simple cycle time settings for functional tests."""
    output_csv = tmp_path / "cycletime.csv"
    settings = _create_base_settings(cycle_time_data=[str(output_csv)])
    return settings, output_csv
```

## Test Patterns

### Calculator Testing

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
def test_calculator_with_dependencies(query_manager, base_minimal_settings):
    """Test calculator that depends on other calculators."""
    # Run prerequisite calculator first
    cycle_calc = CycleTimeCalculator(query_manager, base_minimal_settings, {})
    cycle_data = cycle_calc.run()
    
    # Now run dependent calculator
    settings = {**base_minimal_settings, "my_setting": "value"}
    my_calc = MyCalculator(query_manager, settings, {CycleTimeCalculator: cycle_data})
    result = my_calc.run()
    
    assert result is not None
```

**Empty Data Handling:**
```python
def test_calculator_with_empty_data(query_manager, base_minimal_settings):
    """Test calculator handles empty data gracefully."""
    # Create settings with no matching issues
    settings = {**base_minimal_settings, "queries": [{"jql": "project=NONEXISTENT"}]}
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
2. Regenerate expected outputs if needed
3. Run tests to verify

### Refactoring Tests

When refactoring code:
1. Run all tests first
2. Refactor incrementally
3. Run tests after each change
4. Update tests if behavior changes

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

