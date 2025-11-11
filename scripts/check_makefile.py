#!/usr/bin/env python3
"""
Wrapper script for makefile-checker to check Makefile script references.

This script should be run with the virtual environment's Python interpreter
(e.g., `.venv/bin/python3` or via `make validate-makefile`). The virtual
environment must have the `makefile-checker` package installed.
"""
import os
import sys
from pathlib import Path
from typing import List

try:
    from makefile_checker.checker import Checker
except ImportError:
    print(
        "Error: makefile-checker package is not installed or not available.\n"
        "Please ensure you are running this script with the virtual environment's\n"
        "Python interpreter (e.g., .venv/bin/python3) and that makefile-checker\n"
        "is installed in that environment.\n"
        "You can install it with: pip install makefile-checker",
        file=sys.stderr,
    )
    sys.exit(1)


def get_makefiles_in_current_working_directory(pattern: str = "Makefile") -> List[str]:
    """Get all Makefiles in current working directory."""
    path = os.getcwd()
    result = []
    for root, _, files in os.walk(path):
        # Skip virtual environments and common ignore directories
        ignore_dirs = ['.venv', 'venv', '.git', 'node_modules', '__pycache__']
        root_path = Path(root)
        if any(part in ignore_dirs for part in root_path.parts):
            continue
        for name in files:
            if name == pattern:
                result.append(os.path.join(root, name))
    return result


def check():
    """Check all Makefiles for missing script references."""
    makefiles = get_makefiles_in_current_working_directory()

    if not makefiles:
        print("No Makefile found in current directory.")
        return 0

    all_alerts = []
    for makefile in makefiles:
        try:
            checker = Checker(makefile)
            # Fix the bug in get_missing_scripts_for_makefile - it calls
            # read_file_contents incorrectly. Let's call the methods directly.
            makefile_contents = checker.read_file_contents()
            scripts = checker.clean_and_parse_makefile_scripts(makefile_contents)

            # Get base path
            makefile_path = Path(makefile)
            base_path = makefile_path.parent

            # Check if scripts exist
            for script in scripts:
                script_path = base_path / script.strip()
                if not script_path.is_file():
                    all_alerts.append(f"{makefile}: {script}")
        except FileNotFoundError as e:
            print(f"Warning: Makefile not found: {makefile}: {e}", file=sys.stderr)
            continue
        except (ValueError, AssertionError) as e:
            print(f"Warning: Error parsing/validating Makefile {makefile}: {e}", file=sys.stderr)
            continue
        except PermissionError as e:
            print(f"Error: Permission denied accessing {makefile}: {e}", file=sys.stderr)
            continue
        except OSError as e:
            print(f"Error: IO error accessing {makefile}: {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Error: Unexpected error checking {makefile}: {e}", file=sys.stderr)
            raise

    if len(all_alerts) > 0:
        print("Error: Missing script references found:", file=sys.stderr)
        for alert in all_alerts:
            print(f"  {alert}", file=sys.stderr)
        return 1

    print("✓ All Makefile script references are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(check())
