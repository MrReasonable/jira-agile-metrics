# Project Overview

## What is Jira Agile Metrics?

Jira Agile Metrics is a Python tool for extracting Agile metrics and charts from JIRA projects and Trello boards. It provides comprehensive analytics for agile teams including cycle time analysis, cumulative flow diagrams, burnup charts, throughput metrics, and more.

## Key Features

- **Multi-Source Support**: Extract metrics from both JIRA and Trello
- **Comprehensive Metrics**: Cycle time, throughput, WIP, flow metrics, and more
- **Visual Analytics**: Generate multiple chart types (CFD, scatter plots, histograms, burnup charts)
- **Forecasting**: Monte Carlo simulation for completion date forecasting
- **Multiple Interfaces**: 
  - Command-line interface for batch processing
  - Web interface for interactive chart viewing
- **Deployment Options**: Docker support for easy deployment
- **Flexible Configuration**: YAML-based configuration with inheritance support

## Use Cases

- **Team Performance Analysis**: Track cycle time, throughput, and flow efficiency
- **Forecasting**: Predict completion dates using Monte Carlo simulation
- **Process Improvement**: Identify bottlenecks and waste in workflows
- **Reporting**: Generate progress reports and status updates
- **Historical Analysis**: Track trends over time with various chart types

## Technology Stack

- **Python 3.11+** - Core language
- **pandas** - Data manipulation
- **matplotlib** - Static chart generation
- **bokeh** - Interactive web charts
- **Flask** - Web application framework
- **jira** (pycontribs/jira) - JIRA API client
- **trello** - Trello API client
- **PyYAML** - Configuration parsing

## Project Structure

```
jira_agile_metrics/
├── cli.py                 # Command-line interface
├── calculator.py         # Calculator orchestrator
├── calculators/          # Individual metric calculators
├── config/              # Configuration loading and validation
├── jira_client.py       # JIRA API integration
├── trello.py            # Trello API integration
├── querymanager.py      # Query management
├── webapp/              # Flask web application
└── tests/               # Test suite
    ├── functional/      # Functional tests
    ├── e2e/            # End-to-end tests
    └── fixtures/       # Test data
```

## Quick Start

### Installation

```bash
pip install jira-agile-metrics
```

### Basic Usage

```bash
# Run with configuration file
jira-agile-metrics config.yml

# Start web server
jira-agile-metrics --server 5000
```

### Example Configuration

```yaml
Connection:
  Domain: https://my.jira.com
  Username: user@example.com
  Password: API_TOKEN

Query: project=ABC AND issueType=Story

Workflow:
  Backlog: Backlog
  Committed: Committed
  Build: Build
  Done: Done

Output:
  Cycle time data: cycletime.csv
  CFD chart: cfd.png
  Burnup chart: burnup.png
```

## Available Metrics

### Core Metrics
- **Cycle Time**: Time from commitment to completion
- **Throughput**: Items completed per time period
- **WIP**: Work in progress tracking
- **Flow Efficiency**: Ratio of active time to total time

### Charts
- **Cumulative Flow Diagram (CFD)**: Visualize workflow stages over time
- **Cycle Time Scatter Plot**: Distribution of cycle times
- **Burn-up Chart**: Progress toward completion
- **Burn-up Forecast**: Monte Carlo forecasting
- **Throughput Chart**: Completion rate trends
- **WIP Chart**: Work in progress over time
- **Net Flow Chart**: Arrivals vs departures
- **Ageing WIP**: Current work item ages

### Advanced Analytics
- **Progress Reports**: Epic-level forecasting with Monte Carlo
- **Defect Analysis**: Defects by priority, type, environment
- **Technical Debt**: Debt tracking and aging
- **Impediments**: Blocked work analysis
- **Waste Analysis**: Withdrawn/cancelled work

## Calculator System

The application uses a calculator pattern where each metric is computed by a dedicated calculator class. Calculators can depend on results from previous calculators, enabling complex analytics.

**Calculator Execution Order:**
1. `CycleTimeCalculator` - Base data (must run first)
2. `CFDCalculator` - Cumulative flow data
3. Other calculators depend on these base results

See `architecture.md` for detailed architecture information.

## Configuration System

Configuration is YAML-based with support for:
- **Inheritance**: Use `Extends` to reuse common settings
- **Multiple Queries**: Run different queries with attributes
- **Flexible Output**: Configure data files and charts independently
- **Workflow Mapping**: Map JIRA statuses to workflow stages

## Output Formats

- **CSV**: Spreadsheet-compatible data files
- **JSON**: Structured data for APIs
- **XLSX**: Excel-compatible spreadsheets
- **PNG**: Static chart images
- **HTML**: Interactive progress reports

## Development

See `development.md` for development guidelines and workflow.

## Testing

See `testing.md` for testing guidelines and patterns.

## API Reference

See `api.md` for detailed API documentation.

## Common Patterns

See `patterns.md` for common patterns and best practices.

