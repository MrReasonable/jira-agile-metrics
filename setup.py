"""Package configuration for jira-agile-metrics.

This file defines installation metadata and console entry points.
"""

import os

import setuptools

# Defer any expensive or failure-prone I/O (like reading README/requirements)
# until setup is actually executed. This avoids side effects at import time and
# supports lazy-loading goals during static analysis or tooling.


def main():
    """Entrypoint for invoking setuptools.setup with package metadata."""

    here = os.path.abspath(os.path.dirname(__file__))

    # Safely read long description
    try:
        with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
            long_description = f.read()
    except OSError:
        long_description = ""

    # Safely read install requirements (production-only)
    # Reading requirements.txt directly would include "-r requirements-dev.txt",
    # which is invalid inside install_requires. We only include prod deps here.
    try:
        with open(os.path.join(here, "requirements-prod.txt"), encoding="utf-8") as f:
            install_requires = [
                line.strip()
                for line in f.read().splitlines()
                if line.strip()
                and not line.strip().startswith("#")
                and not line.strip().startswith("-r ")
            ]
    except OSError:
        install_requires = []

    setuptools.setup(
        name="jira-agile-metrics",
        version="0.25",
        description="Agile metrics and summary data extracted from JIRA",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Martin Aspeli",
        author_email="optilude@gmail.com",
        url="https://github.com/optilude/jira-agile-metrics",
        license="MIT",
        keywords="agile jira analytics metrics",
        packages=setuptools.find_packages(exclude=["contrib", "docs", "tests*"]),
        install_requires=install_requires,
        python_requires=">=3.8",
        include_package_data=True,
        package_data={
            "jira_agile_metrics.webapp": ["templates/*.*", "static/*.*"],
            "jira_agile_metrics.calculators": ["*.html"],
        },
        entry_points={
            "console_scripts": [
                "jira-agile-metrics=jira_agile_metrics.cli:main",
            ],
        },
    )


if __name__ == "__main__":
    main()
