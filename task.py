#!/usr/bin/env python3
"""
Python task runner for Jira Agile Metrics
Alternative to Makefile using Python's invoke library pattern
"""

import sys
import subprocess
import os
from pathlib import Path

# Colors for output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color

# Project paths
VENV = Path('.venv')
VENV_BIN = VENV / 'bin'
PYTHON = VENV_BIN / 'python'
PIP = VENV_BIN / 'pip'
PYTEST = VENV_BIN / 'pytest'
BLACK = VENV_BIN / 'black'
RUFF = VENV_BIN / 'ruff'
PYLINT = VENV_BIN / 'pylint'


def run_command(cmd, check=True, capture_output=False):
    """Run a command and return the result."""
    print(f"{GREEN}Running: {cmd}{NC}")
    result = subprocess.run(
        cmd,
        shell=True,
        check=check,
        capture_output=capture_output
    )
    return result


def check_venv():
    """Check if virtual environment exists."""
    if not VENV.exists():
        print(f"{YELLOW}Virtual environment not found. Creating...{NC}")
        subprocess.run([sys.executable, '-m', 'venv', '.venv'], check=True)
        print(f"{GREEN}Virtual environment created{NC}")


def task_help():
    """Show help information."""
    print(f"""{GREEN}Available tasks:{NC}

{YELLOW}Development:{NC}
  python task.py dev                  # Full development setup
  python task.py install              # Install dependencies
  python task.py install-dev         # Install dev dependencies

{YELLOW}Testing:{NC}
  python task.py test                # Run tests
  python task.py test-coverage       # Run tests with coverage
  python task.py test-verbose        # Run tests with verbose output

{YELLOW}Code Quality:{NC}
  python task.py lint                # Run linters
  python task.py format              # Format code
  python task.py check               # Run all checks

{YELLOW}Application:{NC}
  python task.py run                 # Run the CLI
  python task.py webapp              # Start web application

{YELLOW}Docker:{NC}
  python task.py docker-build        # Build Docker images
  python task.py docker-run          # Run with Docker

{YELLOW}Cleanup:{NC}
  python task.py clean               # Remove build artifacts
  python task.py clean-venv          # Remove virtual environment
  python task.py reset               # Reset dev environment

{YELLOW}Info:{NC}
  python task.py info                # Show environment info
  python task.py version             # Show version info
""")


def task_dev():
    """Full development setup."""
    print(f"{GREEN}Setting up development environment...{NC}")
    check_venv()
    run_command(f"{PIP} install --upgrade pip")
    run_command(f"{PIP} install -r requirements.txt")
    run_command(f"{PIP} install -r requirements-dev.txt")
    print(f"\n{GREEN}Development environment ready!{NC}")


def task_install():
    """Install production dependencies."""
    check_venv()
    print(f"{GREEN}Installing production dependencies...{NC}")
    run_command(f"{PIP} install --upgrade pip")
    run_command(f"{PIP} install -r requirements.txt")


def task_install_dev():
    """Install development dependencies."""
    task_install()
    print(f"{GREEN}Installing development dependencies...{NC}")
    run_command(f"{PIP} install -r requirements-dev.txt")


def task_clean():
    """Remove build artifacts."""
    print(f"{YELLOW}Cleaning build artifacts...{NC}")
    dirs_to_remove = [
        '__pycache__',
        '.pytest_cache',
        '.mypy_cache',
        '*.egg-info',
        'build',
        'dist',
    ]
    for pattern in dirs_to_remove:
        for path in Path('.').rglob(pattern):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                import shutil
                shutil.rmtree(path)
    print(f"{GREEN}Clean complete{NC}")


def task_clean_venv():
    """Remove virtual environment."""
    task_clean()
    print(f"{YELLOW}Removing virtual environment...{NC}")
    import shutil
    if VENV.exists():
        shutil.rmtree(VENV)
    print(f"{GREEN}Virtual environment removed{NC}")


def task_test():
    """Run tests."""
    print(f"{GREEN}Running tests...{NC}")
    run_command(f"{PYTEST} -v")


def task_test_coverage():
    """Run tests with coverage."""
    print(f"{GREEN}Running tests with coverage...{NC}")
    run_command(f"{PYTEST} --cov=jira_agile_metrics --cov-report=html --cov-report=term")


def task_test_verbose():
    """Run tests with verbose output."""
    print(f"{GREEN}Running tests with verbose output...{NC}")
    run_command(f"{PYTEST} -vv")


def task_lint():
    """Run linters."""
    print(f"{GREEN}Running ruff...{NC}")
    run_command(f"{RUFF} check .")


def task_lint_fix():
    """Run ruff with auto-fix."""
    print(f"{GREEN}Running ruff with auto-fix...{NC}")
    run_command(f"{RUFF} check . --fix")


def task_format():
    """Format code."""
    print(f"{GREEN}Formatting code...{NC}")
    run_command(f"{BLACK} .")


def task_format_check():
    """Check code formatting."""
    print(f"{GREEN}Checking code formatting...{NC}")
    run_command(f"{BLACK} . --check")


def task_check():
    """Run all checks."""
    task_format_check()
    task_lint()
    print(f"{GREEN}All checks passed!{NC}")


def task_run():
    """Run the CLI."""
    if not Path('config.yml').exists():
        print(f"{RED}Error: config.yml not found{NC}")
        sys.exit(1)
    print(f"{GREEN}Running jira-agile-metrics...{NC}")
    run_command(f"{PYTHON} -m jira_agile_metrics.cli -vv config.yml")


def task_webapp():
    """Start the web application."""
    print(f"{GREEN}Starting web application...{NC}")
    run_command(f"{PYTHON} -m jira_agile_metrics.webapp.app")


def task_info():
    """Show environment information."""
    python_version = subprocess.run(
        [sys.executable, '--version'],
        capture_output=True,
        text=True
    ).stdout.strip()
    
    print(f"{GREEN}Environment Information:{NC}")
    print(f"  Python: {python_version}")
    print(f"  Virtual env: {VENV}")
    print(f"  Venv exists: {'Yes' if VENV.exists() else 'No'}")
    
    req_count = sum(1 for _ in Path('requirements.txt').open()) if Path('requirements.txt').exists() else 0
    print(f"  Requirements: {req_count} lines")
    
    py_files = len(list(Path('jira_agile_metrics').rglob('*.py')))
    print(f"  Source files: {py_files} files")


def task_version():
    """Show version information."""
    print(f"{GREEN}Version Information:{NC}")
    if Path('setup.py').exists():
        with Path('setup.py').open() as f:
            for line in f:
                if 'version' in line and '=' in line:
                    print(f"  {line.strip()}")
                    break
    else:
        print("  setup.py not found")


def task_reset():
    """Reset development environment."""
    task_clean_venv()
    print(f"{YELLOW}Resetting development environment...{NC}")
    task_dev()
    print(f"{GREEN}Development environment reset complete{NC}")


def main():
    """Main task dispatcher."""
    if len(sys.argv) < 2:
        task_help()
        return
    
    task_name = sys.argv[1].replace('-', '_')
    
    # Map task names to functions
    tasks = {
        'help': task_help,
        'dev': task_dev,
        'install': task_install,
        'install_dev': task_install_dev,
        'clean': task_clean,
        'clean_venv': task_clean_venv,
        'test': task_test,
        'test_coverage': task_test_coverage,
        'test_verbose': task_test_verbose,
        'lint': task_lint,
        'lint_fix': task_lint_fix,
        'format': task_format,
        'format_check': task_format_check,
        'check': task_check,
        'run': task_run,
        'webapp': task_webapp,
        'info': task_info,
        'version': task_version,
        'reset': task_reset,
    }
    
    if task_name in tasks:
        try:
            tasks[task_name]()
        except subprocess.CalledProcessError as e:
            print(f"{RED}Task failed with exit code {e.returncode}{NC}")
            sys.exit(1)
    else:
        print(f"{RED}Unknown task: {sys.argv[1]}{NC}")
        task_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
