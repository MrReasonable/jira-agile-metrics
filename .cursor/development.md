# Development Workflow & Guidelines

## Quick Reference

| Concept | Location |
|---------|----------|
| Calculator base class API | [API Reference: Calculator Base Class](api.md#calculator-base-class) |
| Calculator pattern design | [Architecture: Calculator Pattern](architecture.md#calculator-pattern) |
| Calculator code examples | [Patterns: Calculator Patterns](patterns.md#calculator-patterns) |
| Calculator execution order | [Architecture: Calculator Dependencies](architecture.md#calculator-dependencies) |
| Configuration API | [API Reference: Configuration API](api.md#configuration-api) |
| Configuration patterns | [Patterns: Configuration Patterns](patterns.md#configuration-patterns) |
| Chart generation patterns | [Patterns: Chart Generation Patterns](patterns.md#chart-generation-patterns) |
| Chart styling API | [API Reference: Chart Styling](api.md#chart-styling) |
| Testing calculators | [Testing: Calculator Testing](testing.md#calculator-testing) |
| Test fixtures | [Testing: Test Fixtures](testing.md#test-fixtures) |

## Getting Started

### Prerequisites

- Python 3.11 or later
- pip package manager
- (Optional) Docker for containerized development

### Setup Development Environment

```bash
# Clone repository (if not already done)
git clone <repository-url>
cd jira-agile-metrics

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
make install-dev
# OR
pip install -r requirements.txt -r requirements-dev.txt
```

### Verify Installation

```bash
# Run tests
make test

# Check linting
make lint

# Run CLI help
jira-agile-metrics --help
```

## Development Workflow

### Making Changes

1. **Create a branch:**

   ```bash
   git checkout -b feature/my-feature
   ```

1. **Make changes:**

   - Write code following style guidelines
   - Add tests for new functionality
   - Update documentation as needed

1. **Test locally:**

   ```bash
   make test          # Run all tests
   make lint          # Check code quality
   make format        # Auto-format code
   ```

1. **Commit changes:**

   ```bash
   git add .
   git commit -m "Description of changes"
   ```

1. **Push and create PR:**

   ```bash
   git push origin feature/my-feature
   ```

### Code Quality Checks

**Before Committing:**

```bash
# Format code
make format

# Run linters
make lint

# Run tests
make test

# Full check (format + lint + test)
make check-full
```

**Linting Tools:**

- `ruff` - Fast Python linter (E, F, W, I rules)
- `pylint` - Comprehensive code quality checker
- `black` - Code formatter (88 char line length)

## Code Style Guidelines

### Python Style

**Line Length:**

- Maximum 88 characters (Black default)
- Break long lines appropriately

**Imports:**

- Group imports: stdlib, third-party, local
- Use absolute imports
- One import per line for clarity

**Naming:**

- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`

**Docstrings:**

- Use Google-style docstrings
- Document all public functions/classes
- Include Args, Returns, Raises sections

**Example:**

```python
def calculate_cycle_time(
    issues: List[Dict[str, Any]],
    workflow: List[Dict[str, str]]
) -> pd.DataFrame:
    """Calculate cycle time for issues.
    
    Args:
        issues: List of issue dictionaries from JIRA
        workflow: Workflow configuration with stage names
        
    Returns:
        DataFrame with cycle time data
        
    Raises:
        ValueError: If workflow is invalid
    """
    pass
```

### Calculator Development

**Related Documentation:**

- [Calculator Pattern Architecture](architecture.md#calculator-pattern) -
  System design
- [Calculator Patterns](patterns.md#calculator-patterns) - Code examples
- [Calculator Base Class API](api.md#calculator-base-class) - API reference
- [Calculator Testing](testing.md#calculator-testing) - Testing patterns

**Creating a New Calculator:**

1. **Create calculator file:**

   ```python
   # jira_agile_metrics/calculators/my_metric.py

   from ..calculator import Calculator
   import pandas as pd
   import logging

   logger = logging.getLogger(__name__)


   class MyMetricCalculator(Calculator):
       """Calculate my custom metric."""
       
       def run(self, now=None):
           """Calculate metric and return results."""
           # Get data from query manager
           cycle_data = self.get_result(CycleTimeCalculator)
           
           # Perform calculation
           result = self._calculate(cycle_data)
           
           return result
       
       def write(self):
           """Write output files."""
           result = self.get_result()
           if result is None:
               return
           
           # Write data file if configured
           if self.settings.get("my_metric_data"):
               self._write_data_file(result, self.settings["my_metric_data"])
           
           # Write chart if configured
           if self.settings.get("my_metric_chart"):
               self._write_chart_file(result, self.settings["my_metric_chart"])
       
       def _calculate(self, cycle_data):
           """Internal calculation logic."""
           # Implementation here
           pass
   ```

1. **Add to calculator list:**

   ```python
   # jira_agile_metrics/config_main.py

   from .calculators.my_metric import MyMetricCalculator

   CALCULATORS = (
       CycleTimeCalculator,
       # ... existing calculators ...
       MyMetricCalculator,  # Add here in appropriate order
   )
   ```

1. **Add configuration support:**

   ```python
   # jira_agile_metrics/config/loader.py

   # Add parsing for new config options
   if "My metric data" in output:
       options["my_metric_data"] = output["My metric data"]
   ```

1. **Write tests:**

   **Note:** For complete information on available test fixtures, see the [Test
   Fixtures](testing.md#test-fixtures) section. See also [Calculator Testing](testing.md#calculator-testing) for testing patterns.

   ```python
   # jira_agile_metrics/calculators/my_metric_test.py

   import pytest
   from jira_agile_metrics.calculators.my_metric import MyMetricCalculator

   def test_my_metric_calculator(query_manager, simple_cycle_settings):
       settings_dict, _ = simple_cycle_settings
       settings = {
           **settings_dict,
           "my_metric_data": "my_metric.csv",
       }
       
       calculator = MyMetricCalculator(query_manager, settings, {})
       result = calculator.run()
       
       assert result is not None
       # Add more assertions
   ```

**Dependency Ordering and Result Absence:**

When using `self.get_result(SomeCalculator)`, the result can be `None`
for two distinct reasons:

1. **Ordering Issue**: The dependency calculator is listed **after** this
   calculator in the `CALCULATORS` tuple, so it hasn't run yet when `get_result()` is called.
1. **No Data**: The dependency calculator ran successfully but produced no data
   (e.g., empty DataFrame, no matching issues).

To ensure dependencies are available, **always list dependent calculators
earlier** in the `CALCULATORS` tuple. The calculator execution order matches the tuple order exactly.

**Example:**

```python
# jira_agile_metrics/config_main.py

CALCULATORS = (
    CycleTimeCalculator,  # Base calculator - no dependencies
    CFDCalculator,  # Depends on CycleTimeCalculator (listed above)
    BurnupCalculator,  # Depends on CFDCalculator (listed above)
    MyMetricCalculator,  # Depends on CycleTimeCalculator (must come after it)
)
```

**Why Order Matters:**
Calculators are executed sequentially in tuple order. Each calculator's `run()`
method is called, and its result is stored in the shared results dictionary. If a calculator tries to access a dependency that hasn't run yet, `get_result()` will return `None` (or the default value), which is indistinguishable from the dependency having no data.

**Distinguishing "No Data" vs "Calculator Failed" in `write()`:**

The `write()` method needs to handle different scenarios gracefully. When
`self.get_result()` returns `None`, it could mean:

- The calculator failed to run (exception, error condition)
- The calculator ran but produced no data (valid empty result)
- The calculator hasn't run yet (ordering issue - should not happen if
  dependencies are correct)

**Recommended Patterns:**

1. **Return Empty Result Objects from `run()`:**

   ```python
   def run(self, now=None):
       """Calculate and return results."""
       cycle_data = self.get_result(CycleTimeCalculator)
       if cycle_data is None:
           logger.warning("CycleTimeCalculator result not available")
           return None  # or return pd.DataFrame() for empty DataFrame
       
       if len(cycle_data) == 0:
           logger.info("No data available for calculation")
           return pd.DataFrame()  # Return empty but valid result
       
       try:
           result = self._calculate(cycle_data)
           return result
       except Exception as e:
           logger.error("Calculation failed: %s", e, exc_info=True)
           raise  # Re-raise to indicate failure
   ```

1. **Set Explicit Failure Flags (Optional):**

   ```python
   class MyMetricCalculator(Calculator):
       def __init__(self, query_manager, settings, results):
           super().__init__(query_manager, settings, results)
           self._failed = False
       
       def run(self, now=None):
           try:
               # ... calculation logic ...
               return result
           except Exception as e:
               self._failed = True
               logger.error("Calculation failed: %s", e)
               raise
       
       def write(self):
           if self._failed:
               logger.warning("Skipping write() due to calculation failure")
               return
           
           result = self.get_result()
           if result is None:
               logger.debug("No result to write")
               return
           
           # Write outputs...
   ```

1. **Handle Empty Results in `write()`:**

   ```python
   def write(self):
       """Write output files."""
       result = self.get_result()
       
       # Distinguish between None (not run/failed) and empty (no data)
       if result is None:
           logger.debug("No result available, skipping write")
           return
       
       # For DataFrames, check if empty
       if hasattr(result, 'empty') and result.empty:
           logger.info("Result is empty, skipping file writes")
           return
       
       # For lists/dicts, check length
       if isinstance(result, (list, dict)) and len(result) == 0:
           logger.info("Result is empty, skipping file writes")
           return
       
       # Write data file if configured
       if self.settings.get("my_metric_data"):
           self._write_data_file(result, self.settings["my_metric_data"])
       
       # Write chart if configured
       if self.settings.get("my_metric_chart"):
           self._write_chart_file(result, self.settings["my_metric_chart"])
   ```

**CALCULATORS Tuple Ordering Pattern:**

Follow this recommended ordering pattern when adding new calculators:

```python
# jira_agile_metrics/config_main.py

CALCULATORS = (
    # 1. Base calculators (no dependencies)
    CycleTimeCalculator,  # Foundation for most other calculators
    
    # 2. First-level dependents (depend only on base calculators)
    CFDCalculator,  # Depends on CycleTimeCalculator
    ScatterplotCalculator,  # Depends on CycleTimeCalculator
    HistogramCalculator,  # Depends on CycleTimeCalculator
    
    # 3. Second-level dependents (depend on first-level)
    BurnupCalculator,  # Depends on CFDCalculator
    WIPChartCalculator,  # Depends on CFDCalculator
    NetFlowChartCalculator,  # Depends on CFDCalculator
    
    # 4. Third-level dependents (depend on second-level or multiple)
    BurnupForecastCalculator,  # Depends on BurnupCalculator
    
    # 5. Independent calculators (no dependencies or optional dependencies)
    ThroughputCalculator,
    ImpedimentsCalculator,
    DebtCalculator,
    DefectsCalculator,
    WasteCalculator,
    ProgressReportCalculator,
)
```

**Key Principles:**

- Dependencies must appear **before** dependents in the tuple
- Group calculators by dependency level for clarity
- Add comments explaining dependencies when not obvious
- When in doubt, place the calculator later in the tuple (it's safer
  to run after dependencies)

**Testing Dependencies:**

When writing tests for calculators with dependencies, test all three scenarios:

1. **Missing Dependency (Ordering Issue):**

   ```python
   def test_missing_dependency(query_manager, settings):
       """Test behavior when dependency calculator hasn't run."""
       results = {}  # Empty results dict simulates dependency not run
       calculator = MyMetricCalculator(query_manager, settings, results)
       
       result = calculator.run()
       
       # Should handle gracefully - either return None or empty result
       assert result is None or (hasattr(result, 'empty') and result.empty)
   ```

1. **Dependency Present but Empty:**

   ```python
   def test_empty_dependency(query_manager, settings):
       """Test behavior when dependency produced no data."""
       from jira_agile_metrics.calculators.cycletime import CycleTimeCalculator
       import pandas as pd
       
       results = {
           CycleTimeCalculator: pd.DataFrame()  # Empty DataFrame
       }
       calculator = MyMetricCalculator(query_manager, settings, results)
       
       result = calculator.run()
       
       # Should handle empty data gracefully
       assert result is None or (hasattr(result, 'empty') and result.empty)
   ```

1. **Dependency Raises Error:**

   ```python
   def test_dependency_error(query_manager, settings):
       """Test behavior when dependency calculation fails."""
       from jira_agile_metrics.calculators.cycletime import CycleTimeCalculator
       
       # Simulate dependency failure by not including it in results
       # (In real execution, failed calculators may not store results)
       results = {}
       calculator = MyMetricCalculator(query_manager, settings, results)
       
       # Should handle missing dependency gracefully
       result = calculator.run()
       assert result is None
       
       # write() should not crash
       calculator.write()  # Should complete without exception
   ```

1. **Dependency Present with Data:**

   ```python
   def test_with_valid_dependency(query_manager, settings):
       """Test normal operation with valid dependency data."""
       from jira_agile_metrics.calculators.cycletime import CycleTimeCalculator
       import pandas as pd
       
       # Create valid dependency result
       cycle_data = pd.DataFrame({
           'key': ['PROJ-1', 'PROJ-2'],
           'start': [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02')],
           'end': [pd.Timestamp('2024-01-05'), pd.Timestamp('2024-01-06')],
       })
       
       results = {
           CycleTimeCalculator: cycle_data
       }
       calculator = MyMetricCalculator(query_manager, settings, results)
       
       result = calculator.run()
       
       # Should produce valid result
       assert result is not None
       assert len(result) > 0  # or appropriate assertion for your result type
   ```

### Configuration Development

**Related Documentation:**

- [Configuration System Architecture](architecture.md#configuration-system) -
  Configuration flow
- [Configuration API](api.md#configuration-api) - API reference
- [Configuration Patterns](patterns.md#configuration-patterns) - Validation
  and inheritance examples

**Adding New Configuration Options:**

1. **Define in loader:**

   ```python
   # jira_agile_metrics/config/loader.py

   def _parse_output_section(output, options):
       # ... existing parsing ...
       
       # New option
       if "My new option" in output:
           options["my_new_option"] = output["My new option"]
   ```

1. **Add validation:**

   ```python
   # jira_agile_metrics/config/type_utils.py

   def validate_my_new_option(value):
       """Validate my new option."""
       if not isinstance(value, str):
           raise ConfigError("My new option must be a string")
       return value
   ```

1. **Document in README:**

   - Add to configuration examples
   - Document in output settings reference

### Chart Development

**Related Documentation:**

- [Chart Generation Architecture](architecture.md#chart-generation) - Static
  and interactive charts
- [Chart Generation Patterns](patterns.md#chart-generation-patterns) - Code
  examples
- [Chart Styling API](api.md#chart-styling) - `apply_chart_style` function

**Creating a New Chart:**

1. **Use matplotlib for static charts:**

   ```python
   import matplotlib.pyplot as plt
   from jira_agile_metrics.chart_styling_utils import apply_chart_style

   def create_my_chart(data, output_file, title=None):
       """Create my custom chart."""
       if data is None or len(data) == 0:
           logger.warning("Cannot draw chart with zero items")
           return
       
       fig, ax = plt.subplots(figsize=(12, 6))
       
       # Plot data
       ax.plot(data.index, data.values)
       
       # Apply styling
       apply_chart_style(fig, ax, title=title or "My Chart")
       
       # Save
       plt.savefig(output_file, dpi=150, bbox_inches="tight")
       plt.close()
   ```

1. **Use Bokeh for interactive charts:**

   ```python
   from bokeh.plotting import figure
   from bokeh.embed import components

   def create_interactive_chart(data):
       """Create interactive Bokeh chart."""
       p = figure(title="My Chart", x_axis_label="X", y_axis_label="Y")
       p.line(data.index, data.values)
       return p
   ```

## Common Tasks

### Adding a New Metric

1. Create calculator class
1. Add to `CALCULATORS` list
1. Add configuration parsing
1. Write tests
1. Update documentation
1. Add example to README

### Fixing a Bug

1. Reproduce the bug
1. Write a failing test
1. Fix the code
1. Verify test passes
1. Check for regressions
1. Update documentation if needed

### Refactoring

1. Ensure test coverage
1. Run all tests before refactoring
1. Make small, incremental changes
1. Run tests after each change
1. Update documentation

### Performance Optimization

1. Profile the code
1. Identify bottlenecks
1. Optimize incrementally
1. Measure improvements
1. Document changes

## Debugging

### Local Development

**Verbose Logging:**

```bash
# INFO level
jira-agile-metrics -v config.yml

# DEBUG level
jira-agile-metrics -vv config.yml
```

**Python Debugger:**

```python
import pdb; pdb.set_trace()  # Set breakpoint
```

**Print Debugging:**

```python
logger.debug("Variable value: %s", variable)
```

### Docker Development

**Run in Development Container:**

```bash
make docker-cli
# OR
docker-compose up
```

**Access Container Shell:**

```bash
docker exec -it jira_metrics bash
```

**View Logs:**

```bash
docker logs jira_metrics
docker logs -f jira_metrics  # Follow logs
```

## Git Workflow

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `refactor/` - Code refactoring
- `docs/` - Documentation updates
- `test/` - Test improvements

### Commit Messages

**Format:**

```text
Short summary (50 chars max)

Longer description if needed, explaining:
- What changed
- Why it changed
- Any breaking changes
```

**Examples:**

```text
Add throughput calculator

Implements weekly throughput calculation with trend line.
Adds new configuration option 'Throughput frequency' to control
aggregation period.
```

```text
Fix CFD chart empty data handling

Prevents crash when no issues match workflow criteria.
Returns early with warning log message.
```

## Documentation

### Code Documentation

**Docstrings:**

- All public functions/classes
- Google-style format
- Include examples for complex functions

**Comments:**

- Explain "why", not "what"
- Complex algorithms
- Non-obvious workarounds

### User Documentation

**README Updates:**

- New features
- Configuration options
- Usage examples
- Troubleshooting tips

**Configuration Examples:**

- Add to README
- Include in test fixtures
- Document all options
