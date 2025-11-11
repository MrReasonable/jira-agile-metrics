# Cursor Rules Documentation

This directory contains comprehensive documentation for the Jira Agile Metrics
project, organized by purpose and topic.

**Last updated:** 2025-11-10

## Document Summaries

### [`overview.md`](overview.md) - Project Overview & Quick Start

**Purpose:** High-level introduction to the project, its features, and getting
started guide.

**Contents:**

- What is Jira Agile Metrics and its key features
- Technology stack and project structure
- Quick start guide with installation and basic usage
- Available metrics and calculators overview
- Configuration system introduction
- Links to detailed documentation

**Best for:** New contributors, understanding project scope, quick reference
for available features.

### [`architecture.md`](architecture.md) - Technical Architecture

**Purpose:** Deep dive into system design, data flow,
and architectural patterns.

**Contents:**

- Calculator pattern and execution order
- Query Manager pattern and data fetching
- Configuration system architecture
- Data structures and formats (Cycle Time, CFD, etc.)
- Chart generation (matplotlib and Bokeh)
- Error handling patterns
- Performance optimizations
- Web application architecture
- Docker architecture
- Logging and type hints

**Best for:** Understanding how the system works, designing new features,
debugging architecture issues.

### [`api.md`](api.md) - API Reference

**Purpose:** Complete reference for all public interfaces, classes,
and functions.

**Contents:**

- Calculator base class API
- Configuration API and exceptions
- Query Manager API
- JIRA Client API
- Calculator runner functions
- Web application routes
- CLI arguments and options
- Utility functions (chart styling, data utilities, type utilities)
- Individual calculator classes
- Usage examples

**Best for:** Looking up specific API details, understanding function
signatures, finding usage examples.

### [`development.md`](development.md) - Development Workflow

**Purpose:** Guide for setting up development environment
and common development tasks.

**Contents:**

- Development environment setup
- Code style guidelines (Python style, line length, imports, naming, docstrings)
- Creating new calculators (step-by-step guide)
- Configuration development
- Chart development patterns
- Common tasks (adding metrics, fixing bugs, refactoring)
- Debugging techniques
- Git workflow and commit conventions
- Documentation standards

**Best for:** Setting up your dev environment, writing new code, following
project conventions.

### [`patterns.md`](patterns.md) - Code Patterns & Best Practices

**Purpose:** Reusable code patterns, examples, and best practices
for common scenarios.

**Contents:**

- Calculator patterns (basic, with dependencies, empty data handling)
- Chart generation patterns (matplotlib, Bokeh, windowing)
- Data processing patterns (pandas operations, date handling, monthly breakdown)
- Configuration patterns (validation, inheritance)
- Error handling patterns (graceful degradation, validation)
- Logging patterns (structured, progress)
- Testing patterns (fixtures, calculator tests, mock APIs)
- Performance patterns (caching, batch processing, memory efficiency)
- Best practices summary

**Best for:** Copy-paste code examples, understanding common patterns,
learning best practices.

### [`testing.md`](testing.md) - Testing Guidelines

**Purpose:** Comprehensive guide to writing and running tests.

**Contents:**

- Test organization and directory structure
- Test types (unit, functional, E2E)
- Test fixtures (detailed documentation of all fixtures)
- Test patterns (calculator testing, data validation, chart testing)
- Running tests (commands, markers, coverage)
- Test best practices (do's and don'ts)
- Debugging tests
- Continuous integration
- Test maintenance

**Best for:** Writing tests, understanding test fixtures, debugging test
failures.

## Recommended Reading Order

### For New Contributors

1. **Start here:** [`overview.md`](overview.md) - Get familiar with what the
   project does
1. **Then:** [`development.md`](development.md) - Set up your development
   environment
1. **Next:** [`architecture.md`](architecture.md) - Understand the system design
1. **Reference:** [`patterns.md`](patterns.md) - Use as needed for code examples
1. **When coding:** [`api.md`](api.md) - Look up specific APIs
1. **When testing:** [`testing.md`](testing.md) - Write and run tests

### For Adding a New Calculator

1. [`development.md`](development.md#creating-a-new-calculator) - Step-by-step
   guide
1. [`patterns.md`](patterns.md#calculator-patterns) - Code examples
1. [`architecture.md`](architecture.md#calculator-pattern) - Understand the
   pattern
1. [`api.md`](api.md#calculator-base-class) - API reference
1. [`testing.md`](testing.md#calculator-testing) - Write tests

### For Understanding the System

1. [`overview.md`](overview.md) - High-level understanding
1. [`architecture.md`](architecture.md) - Deep dive into design
1. [`api.md`](api.md) - Reference for specific components
1. [`patterns.md`](patterns.md) - See how things are implemented

### For Debugging

1. [`architecture.md`](architecture.md#error-handling-patterns) - Error handling
1. [`development.md`](development.md#debugging) - Debugging techniques
1. [`api.md`](api.md) - API reference for relevant components
1. [`testing.md`](testing.md#debugging-tests) - Test debugging

## Quick Reference

**Getting Started?** → [`overview.md`](overview.md)

**Adding a new calculator?** →
[`development.md`](development.md#creating-a-new-calculator) and
[`patterns.md`](patterns.md#calculator-patterns)

**Understanding the architecture?** → [`architecture.md`](architecture.md)

**Writing tests?** → [`testing.md`](testing.md)

**Looking for API details?** → [`api.md`](api.md)

**Need code examples?** → [`patterns.md`](patterns.md)

## Maintenance

When updating these files:

- Keep examples current with actual code
- Update when architecture changes
- Add new patterns as they emerge
- Document breaking changes clearly
- Update the "Last updated" date at the top of this file when making changes

### Cursor Rules

The project uses a `.cursorrules` file at the repository root to provide quick
reference information to Cursor. This file complements the detailed
documentation in the `.cursor/` directory.

**Location:** `.cursorrules` (repository root)

**Format:** Markdown file with quick reference sections including:

- Project overview and key components
- Code patterns and examples
- Development commands
- Common patterns and best practices
- Important notes about project structure

**Example entries:**

- Calculator pattern examples
- Code style guidelines
- Testing conventions
- Development workflow steps

The `.cursorrules` file serves as a quick reference, while detailed
documentation lives in the `.cursor/` directory files. When updating project
patterns or conventions, update both the `.cursorrules` file and the relevant
detailed documentation files.

## Troubleshooting

Common issues with Cursor integration and their solutions:

1. Cursor not picking up documentation changes

   - **Fix:** Restart Cursor or reload the window
     (Cmd+Shift+P → "Developer: Reload Window")
   - **Prevention:** Ensure files are saved before expecting Cursor to use them

1. Documentation examples don't match current code

   - **Fix:** Update the relevant documentation file in `.cursor/` directory
     and the `.cursorrules` file
   - **Prevention:** Update documentation as part of code changes, not as a
     separate task

1. Cursor suggests patterns that don't match project conventions

   - **Fix:** Check that `.cursorrules` and relevant `.cursor/` files are up
     to date with current patterns
   - **Prevention:** Review and update documentation when architecture
     or patterns change

1. Missing context in Cursor suggestions

   - **Fix:** Ensure all relevant `.cursor/` documentation files exist
     and are properly formatted
   - **Prevention:** Keep documentation comprehensive and organized by topic

1. Incorrect file paths or references in documentation

   - **Fix:** Verify all file paths in documentation match actual project
     structure
   - **Prevention:** Use relative paths from repository root
     and test documentation accuracy
