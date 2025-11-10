# Development Workflow & Guidelines

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

2. **Make changes:**
   - Write code following style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test locally:**
   ```bash
   make test          # Run all tests
   make lint          # Check code quality
   make format        # Auto-format code
   ```

4. **Commit changes:**
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

5. **Push and create PR:**
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

2. **Add to calculator list:**
   ```python
   # jira_agile_metrics/config_main.py
   
   from .calculators.my_metric import MyMetricCalculator
   
   CALCULATORS = (
       CycleTimeCalculator,
       # ... existing calculators ...
       MyMetricCalculator,  # Add here in appropriate order
   )
   ```

3. **Add configuration support:**
   ```python
   # jira_agile_metrics/config/loader.py
   
   # Add parsing for new config options
   if "My metric data" in output:
       options["my_metric_data"] = output["My metric data"]
   ```

4. **Write tests:**
   ```python
   # jira_agile_metrics/calculators/my_metric_test.py
   
   import pytest
   from jira_agile_metrics.calculators.my_metric import MyMetricCalculator
   
   def test_my_metric_calculator(query_manager, base_minimal_settings):
       settings = {
           **base_minimal_settings,
           "my_metric_data": "my_metric.csv",
       }
       
       calculator = MyMetricCalculator(query_manager, settings, {})
       result = calculator.run()
       
       assert result is not None
       # Add more assertions
   ```

### Configuration Development

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

2. **Add validation:**
   ```python
   # jira_agile_metrics/config/type_utils.py
   
   def validate_my_new_option(value):
       """Validate my new option."""
       if not isinstance(value, str):
           raise ConfigError("My new option must be a string")
       return value
   ```

3. **Document in README:**
   - Add to configuration examples
   - Document in output settings reference

### Chart Development

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

2. **Use Bokeh for interactive charts:**
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
2. Add to `CALCULATORS` list
3. Add configuration parsing
4. Write tests
5. Update documentation
6. Add example to README

### Fixing a Bug

1. Reproduce the bug
2. Write a failing test
3. Fix the code
4. Verify test passes
5. Check for regressions
6. Update documentation if needed

### Refactoring

1. Ensure test coverage
2. Run all tests before refactoring
3. Make small, incremental changes
4. Run tests after each change
5. Update documentation

### Performance Optimization

1. Profile the code
2. Identify bottlenecks
3. Optimize incrementally
4. Measure improvements
5. Document changes

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
```
Short summary (50 chars max)

Longer description if needed, explaining:
- What changed
- Why it changed
- Any breaking changes
```

**Examples:**
```
Add throughput calculator

Implements weekly throughput calculation with trend line.
Adds new configuration option 'Throughput frequency' to control
aggregation period.
```

```
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

