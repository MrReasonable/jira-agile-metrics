# Technical Architecture

## Architecture Patterns

### Calculator Pattern

The application uses a calculator pattern where each metric is computed by a dedicated calculator class:

```python
class Calculator:
    """Base calculator interface."""
    
    def __init__(self, query_manager, settings, results):
        self.query_manager = query_manager
        self.settings = settings
        self._results = results  # Shared results dict
    
    def run(self):
        """Perform calculation and return results."""
        pass
    
    def write(self):
        """Write output files."""
        pass
```

**Key Design Decisions:**
- Calculators share results via `_results` dictionary
- Two-phase execution: `run()` then `write()`
- Calculators can depend on results from previous calculators
- Order matters - see `CALCULATORS` tuple in `config_main.py`

### Query Manager Pattern

Centralized data fetching and caching:

```python
class QueryManager:
    """Manages JIRA/Trello queries and data fetching."""
    
    def get_issues(self):
        """Fetch and cache issues."""
        
    def get_cycle_data(self):
        """Get processed cycle time data."""
```

**Responsibilities:**
- Abstract JIRA/Trello API differences
- Cache fetched data
- Transform raw API responses to internal format
- Handle pagination and rate limiting

### Configuration System

**YAML Configuration:**
- Hierarchical structure
- Support for `Extends` (configuration inheritance)
- Type coercion utilities (`force_int`, `force_float`, `force_date`)
- Validation with clear error messages

**Configuration Flow:**
1. Load YAML file
2. Apply `Extends` if present
3. Merge command-line overrides
4. Validate required fields
5. Convert to internal settings dict
6. Pass to calculators

### Data Flow

```
Config File (YAML)
    ↓
Config Loader → Settings Dict
    ↓
Query Manager → Issues Data
    ↓
Calculators (in order)
    ├─ CycleTimeCalculator (base data)
    ├─ CFDCalculator
    ├─ ScatterplotCalculator
    ├─ ... (others)
    └─ ProgressReportCalculator
    ↓
Results Dict → Output Files
```

## Calculator Implementation Details

### Base Calculator

`BaseCalculator` provides common functionality:

```python
class BaseCalculator(Calculator):
    """Shared functionality for calculators."""
    
    def check_chart_data_empty(self, chart_data, chart_name):
        """Validate chart data before plotting."""
        
    def get_active_cycle_columns(self):
        """Get columns between committed and done."""
        
    def create_monthly_breakdown_chart(self, chart_data, output_file, config):
        """Common monthly breakdown chart pattern."""
```

### Calculator Dependencies

**Execution Order (from `config_main.py`):**
1. `CycleTimeCalculator` - Must run first (base data)
2. `BottleneckChartsCalculator` - Depends on cycle time
3. `CFDCalculator` - Needed by burnup, WIP, net flow
4. `ScatterplotCalculator` - Independent
5. `HistogramCalculator` - Independent
6. `PercentilesCalculator` - Independent
7. `ThroughputCalculator` - Independent
8. `BurnupCalculator` - Depends on CFD
9. `WIPChartCalculator` - Depends on CFD
10. `NetFlowChartCalculator` - Depends on CFD
11. `AgeingWIPChartCalculator` - Independent
12. `BurnupForecastCalculator` - Depends on Burnup
13. `ImpedimentsCalculator` - Independent
14. `DebtCalculator` - Independent
15. `DefectsCalculator` - Independent
16. `WasteCalculator` - Independent
17. `ProgressReportCalculator` - Complex, depends on multiple

### Accessing Previous Calculator Results

```python
class MyCalculator(Calculator):
    def run(self):
        # Get results from CycleTimeCalculator
        cycle_data = self.get_result(CycleTimeCalculator)
        
        # Get results from CFDCalculator
        cfd_data = self.get_result(CFDCalculator)
        
        # Process and return own results
        return my_results
```

## Data Structures

### Cycle Time Data Format

```python
DataFrame with columns:
- key: Issue key (e.g., "PROJ-123")
- created: Creation date
- resolved: Resolution date
- start: Cycle start date
- end: Cycle end date
- {column_name}: Date entered each workflow column
- Blocked Days: Days flagged as impeded
- Attributes: Additional fields from config
```

### CFD Data Format

```python
DataFrame with:
- Index: Date (daily)
- Columns: Workflow stage names
- Values: Count of issues in each stage on that date
```

### Settings Dictionary Structure

```python
{
    "cycle": [
        {"name": "Backlog", "statuses": ["Backlog"]},
        {"name": "Committed", "statuses": ["Committed"]},
        # ...
    ],
    "committed_column": "Committed",
    "done_column": "Done",
    "backlog_column": "Backlog",
    "queries": [{"jql": "project=ABC"}],
    "attributes": {"Priority": "Priority"},
    "cycle_time_data": ["cycletime.csv"],
    "cfd_chart": "cfd.png",
    # ... many more output options
}
```

## Chart Generation

### Static Charts (matplotlib)

**Common Pattern:**
```python
import matplotlib.pyplot as plt
from jira_agile_metrics.chart_styling_utils import apply_chart_style

fig, ax = plt.subplots()
# ... plot data ...
apply_chart_style(fig, ax, title="Chart Title")
plt.savefig(output_file)
plt.close()
```

**Styling:**
- Use `chart_styling_utils` for consistent styling
- Handle empty data gracefully
- Provide meaningful titles and labels
- Support windowing (show only recent data)

### Interactive Charts (Bokeh)

**Web Application:**
- Bokeh charts embedded in Flask templates
- JavaScript callbacks for interactivity
- Responsive design with Bootstrap

**Template Pattern:**
```python
from bokeh.embed import components

script, div = components(chart)
return render_template('bokeh_chart.html', script=script, div=div)
```

## Error Handling Patterns

### Configuration Errors

```python
from jira_agile_metrics.config.exceptions import ConfigError

if not required_field:
    raise ConfigError(f"Missing required field: {field_name}")
```

### Data Validation

```python
if chart_data is None or len(chart_data) == 0:
    logger.warning("Cannot draw chart with zero items")
    return
```

### API Errors

```python
try:
    issues = jira_client.search_issues(jql)
except JIRAError as e:
    logger.error(f"JIRA API error: {e}")
    raise
```

## Performance Optimizations

### Caching

**Query Manager:**
- Cache issues after first fetch
- Cache cycle data processing
- Avoid redundant API calls

**Web Application:**
- In-memory cache for calculator results
- Cache key based on config hash
- Thread-safe caching

### Data Processing

**Pandas Operations:**
- Use vectorized operations
- Avoid row-by-row iteration
- Use appropriate dtypes
- Process in chunks for large datasets

**Memory Management:**
- Close matplotlib figures after saving
- Clear large DataFrames when done
- Use generators for large datasets

## Web Application Architecture

### Flask Structure

**Routes:**
- Dashboard (`/`) - List all available charts
- Individual chart routes (`/cfd`, `/burnup`, etc.)
- Configuration upload (`POST /set_query`)

**Calculator Integration:**
- Run calculators on-demand
- Cache results per configuration
- Handle long-running calculations
- Display progress/errors

### Template System

**Templates:**
- `layout.html` - Base template with Bootstrap
- `index.html` - Dashboard
- `bokeh_chart.html` - Interactive chart wrapper
- `burnup_forecast.html` - Special forecast chart
- `results.html` - Results display

**Static Assets:**
- Bootstrap CSS/JS
- Custom CSS (`metrics.css`)
- Bokeh JavaScript libraries

## Docker Architecture

### Multi-Stage Builds

**Development Image:**
- Includes dev dependencies
- Mounts source code
- Hot reload support

**Production Image:**
- Minimal dependencies
- Optimized size
- Non-root user

**Batch Image:**
- Processes multiple configs
- Writes to volume
- Logs to file

**Web Server Image:**
- nginx + uwsgi
- Production WSGI server
- Static file serving

### Volume Mounts

```bash
# Data directory
-v $PWD:/data

# Config directory (batch mode)
-v /path/to/config:/config

# Output directory
-v /path/to/output:/data
```

## Logging

### Log Levels

- **DEBUG** (`-vv`) - Detailed execution flow
- **INFO** (`-v`) - General progress messages
- **WARNING** - Non-fatal issues (missing data, etc.)
- **ERROR** - Fatal errors

### Log Format

```python
import logging
import colorlog  # Optional colored output

logger = logging.getLogger(__name__)
logger.info("Processing %d issues", len(issues))
logger.warning("No data for chart: %s", chart_name)
```

## Type Hints

### Common Types

```python
from typing import Dict, List, Optional, Any
import pandas as pd

def process_data(
    issues: List[Dict[str, Any]],
    settings: Dict[str, Any]
) -> pd.DataFrame:
    """Process issues into DataFrame."""
    pass
```

### Calculator Type Hints

```python
class Calculator:
    def __init__(
        self,
        query_manager: QueryManager,
        settings: Dict[str, Any],
        results: Dict[type, Any]
    ) -> None:
        pass
    
    def run(self, now: Optional[datetime] = None) -> Any:
        pass
```

## Code Organization Principles

1. **Single Responsibility** - Each calculator does one thing
2. **Dependency Injection** - Pass dependencies via constructor
3. **Immutability** - Don't modify input data
4. **Error Handling** - Fail fast with clear messages
5. **Logging** - Log important events and errors
6. **Testing** - Write tests for all new code
7. **Documentation** - Document public APIs

