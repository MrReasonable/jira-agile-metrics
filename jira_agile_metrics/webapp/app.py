"""Web application for Jira Agile Metrics.

This module provides a Flask-based web interface for viewing and generating
agile metrics from JIRA data.
"""

import logging
import os
import os.path
import secrets
import threading
import time

import jinja2
from bokeh.embed import components
from bokeh.plotting import figure
from dotenv import load_dotenv
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from ..calculator import run_calculators
from ..calculators.ageingwip import AgeingWIPChartCalculator
from ..calculators.burnup import BurnupCalculator
from ..calculators.cfd import CFDCalculator
from ..calculators.cycletime import CycleTimeCalculator
from ..calculators.debt import DebtCalculator
from ..calculators.defects import DefectsCalculator
from ..calculators.forecast import BurnupForecastCalculator
from ..calculators.histogram import HistogramCalculator
from ..calculators.impediments import ImpedimentsCalculator
from ..calculators.netflow import NetFlowChartCalculator
from ..calculators.percentiles import PercentilesCalculator
from ..calculators.progressreport import ProgressReportCalculator
from ..calculators.scatterplot import ScatterplotCalculator
from ..calculators.waste import WasteCalculator
from ..config import config_to_options
from ..config_main import CALCULATORS
from ..querymanager import QueryManager
from ..utils import find_backlog_and_done_columns
from .helpers import (
    get_jira_client,
    plot_forecast_fan,
)

load_dotenv()

template_folder = os.path.join(os.path.dirname(__file__), "templates")
static_folder = os.path.join(os.path.dirname(__file__), "static")

app = Flask(
    "jira-agile-metrics",
    template_folder=template_folder,
    static_folder=static_folder,
)


# Add security headers
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    return response


app.jinja_loader = jinja2.PackageLoader("jira_agile_metrics.webapp", "templates")

logger = logging.getLogger(__name__)

# In-memory cache for calculator results (thread-safe)
results_cache = {}
results_cache_lock = threading.Lock()

# Thread lock to protect os.chdir() operations
# os.chdir() modifies global process state and is not thread-safe
_chdir_lock = threading.Lock()

app.secret_key = os.environ.get("FLASK_SECRET_KEY")
if not app.secret_key:
    logger.warning(
        "FLASK_SECRET_KEY environment variable not set. "
        "Using a random key for this session only. "
        "Set FLASK_SECRET_KEY for production use."
    )
    app.secret_key = secrets.token_hex(32)


# Helper to load config, create QueryManager, and run calculators
def get_real_results():
    """Load configuration and generate real results from JIRA data."""
    config_path = os.path.join(os.path.dirname(__file__), "../../config.yml")
    with open(config_path, encoding="utf-8") as f:
        config_data = f.read()
    options = config_to_options(config_data)
    # Only use credentials from config.yml, do not use env vars or prompt
    if (
        not options["connection"].get("username")
        or not options["connection"].get("password")
        or not options["connection"].get("domain")
    ):
        raise RuntimeError(
            "JIRA credentials (domain, username, password) must be set in config.yml "
            "under the Connection section."
        )
    # If user provided a query, override the config
    user_query = session.get("user_query")
    if user_query:
        # Try to override in both settings and connection for compatibility
        if "Query" in options["settings"]:
            options["settings"]["Query"] = user_query
        if "Query" in options["connection"]:
            options["connection"]["Query"] = user_query
        options["settings"]["queries"] = [{"jql": user_query}]
    # Use a cache key based on connection, settings, and user_query
    cache_key = str(options["connection"]) + str(options["settings"]) + str(user_query)
    now = time.time()
    with results_cache_lock:
        if cache_key in results_cache:
            results, timestamp = results_cache[cache_key]
            if now - timestamp < 86400:  # 24 hours
                return results

    # Change to output directory if specified
    # (to write files there, not in project root)
    # Default to "output/" to prevent writing files to project root
    # os.chdir() modifies global process state and is protected by a lock
    # for thread-safety in multi-threaded environments (e.g., Flask's threaded mode)
    output_dir = options.get("output_directory")
    # If no output directory specified, default to "output/" to prevent
    # writing files to the project root
    if not output_dir:
        output_dir = "output"
    original_cwd = os.getcwd()

    # Protect os.chdir() operations with a lock for thread safety
    with _chdir_lock:
        try:
            os.makedirs(output_dir, exist_ok=True)
            os.chdir(output_dir)

            jira = get_jira_client(options["connection"])
            query_manager = QueryManager(jira, options["settings"])
            results = run_calculators(CALCULATORS, query_manager, options["settings"])
        finally:
            # Always restore the original working directory
            os.chdir(original_cwd)

    with results_cache_lock:
        results_cache[cache_key] = (results, now)
    return results


@app.route("/")
def index():
    """Display the main index page."""
    return render_template("index.html")


@app.route("/burnup-forecast")
def burnup_forecast():
    """Generate burnup forecast chart with Monte Carlo forecast fans."""
    try:
        results = get_real_results()
        forecast_data = results[BurnupForecastCalculator]
        burnup_data = results[BurnupCalculator]

        if forecast_data is None or len(forecast_data.index) == 0:
            flash("No data available for Burnup Forecast Chart.", "warning")
            return render_template("burnup_forecast.html", script="", div="")

        if burnup_data is None or len(burnup_data.index) == 0:
            flash("No historical burnup data available.", "warning")
            return render_template("burnup_forecast.html", script="", div="")

        p = figure(
            title="Burnup Forecast Chart",
            x_axis_type="datetime",
            width=1000,
            height=600,
        )

        # Plot historical data
        backlog_column, done_column = find_backlog_and_done_columns(burnup_data)
        if backlog_column and backlog_column in burnup_data.columns:
            p.line(
                burnup_data.index,
                burnup_data[backlog_column],
                legend_label="Backlog (historical)",
                line_width=2,
                color="blue",
            )
        elif backlog_column:
            logger.warning(
                "Backlog column '%s' not found in burnup_data.columns. "
                "Skipping backlog plot.",
                backlog_column,
            )
        if done_column and done_column in burnup_data.columns:
            p.line(
                burnup_data.index,
                burnup_data[done_column],
                legend_label="Done (historical)",
                line_width=2,
                color="green",
            )
        elif done_column:
            logger.warning(
                "Done column '%s' not found in burnup_data.columns. "
                "Skipping done plot.",
                done_column,
            )

        # Calculate and plot forecast fan bands
        plot_forecast_fan(p, forecast_data)

        p.legend.location = "top_left"
        p.xaxis.axis_label = "Date"
        p.yaxis.axis_label = "Items"
        script, div = components(p)
        return render_template("burnup_forecast.html", script=script, div=div)
    except (ValueError, AttributeError, KeyError, ImportError) as e:
        logger.error("Error generating burnup forecast: %s", e)
        flash(str(e), "danger")
        return render_template("burnup_forecast.html", script="", div="")


@app.route("/burnup")
def burnup_chart():
    """Generate burnup chart."""
    try:
        results = get_real_results()
        chart_data = results[BurnupCalculator]
        if chart_data is None or len(chart_data.index) == 0:
            flash("No data available for Burnup Chart.", "warning")
            return render_template(
                "bokeh_chart.html", title="Burnup Chart", script="", div=""
            )
        p = figure(title="Burnup Chart", x_axis_type="datetime", width=800, height=400)
        for col in chart_data.columns:
            p.line(
                chart_data.index,
                chart_data[col],
                legend_label=col,
                line_width=2,
            )
        p.legend.location = "top_left"
        p.xaxis.axis_label = "Date"
        p.yaxis.axis_label = "Items"
        script, div = components(p)
        return render_template(
            "bokeh_chart.html", title="Burnup Chart", script=script, div=div
        )
    except (ValueError, AttributeError, KeyError, ImportError) as e:
        logger.error("Error generating burnup chart: %s", e)
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Burnup Chart", script="", div=""
        )


@app.route("/cfd")
def cfd_chart():
    """Generate CFD (Cumulative Flow Diagram) chart."""
    try:
        results = get_real_results()
        chart_data = results[CFDCalculator]
        if chart_data is None or len(chart_data.index) == 0:
            flash("No data available for CFD Chart.", "warning")
            return render_template(
                "bokeh_chart.html",
                title="Cumulative Flow Diagram (CFD)",
                script="",
                div="",
            )
        p = figure(
            title="Cumulative Flow Diagram (CFD)",
            x_axis_type="datetime",
            width=800,
            height=400,
        )
        for col in chart_data.columns:
            p.line(
                chart_data.index,
                chart_data[col],
                legend_label=col,
                line_width=2,
            )
        p.legend.location = "top_left"
        p.xaxis.axis_label = "Date"
        p.yaxis.axis_label = "Items"
        script, div = components(p)
        return render_template(
            "bokeh_chart.html",
            title="Cumulative Flow Diagram (CFD)",
            script=script,
            div=div,
        )
    except (ValueError, AttributeError, KeyError, ImportError) as e:
        logger.error("Error generating CFD chart: %s", e)
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html",
            title="Cumulative Flow Diagram (CFD)",
            script="",
            div="",
        )


@app.route("/histogram")
def histogram_chart():
    """Generate histogram chart."""
    try:
        results = get_real_results()
        chart_data = results[HistogramCalculator]
        if chart_data is None or len(chart_data.index) == 0:
            flash("No data available for Cycle Time Histogram.", "warning")
            return render_template(
                "bokeh_chart.html",
                title="Cycle Time Histogram",
                script="",
                div="",
            )
        p = figure(
            title="Cycle Time Histogram",
            x_range=list(chart_data.index),
            width=800,
            height=400,
        )
        p.vbar(x=list(chart_data.index), top=list(chart_data.values), width=0.9)
        p.xaxis.axis_label = "Cycle Time Bin"
        p.yaxis.axis_label = "Items"
        script, div = components(p)
        return render_template(
            "bokeh_chart.html",
            title="Cycle Time Histogram",
            script=script,
            div=div,
        )
    except (ValueError, AttributeError, KeyError, ImportError) as e:
        logger.error("Error generating histogram chart: %s", e)
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Cycle Time Histogram", script="", div=""
        )


@app.route("/scatterplot")
def scatterplot_chart():
    """Generate scatterplot chart."""
    try:
        results = get_real_results()
        chart_data = results[ScatterplotCalculator]
        if chart_data is None or len(chart_data.index) == 0:
            flash("No data available for Cycle Time Scatterplot.", "warning")
            return render_template(
                "bokeh_chart.html",
                title="Cycle Time Scatterplot",
                script="",
                div="",
            )
        p = figure(title="Cycle Time Scatterplot", width=800, height=400)
        p.circle(chart_data["x"], chart_data["y"], size=8, color="navy", alpha=0.5)
        p.xaxis.axis_label = "X"
        p.yaxis.axis_label = "Y"
        script, div = components(p)
        return render_template(
            "bokeh_chart.html",
            title="Cycle Time Scatterplot",
            script=script,
            div=div,
        )
    except (ValueError, AttributeError, KeyError, ImportError) as e:
        logger.error("Error generating scatterplot chart: %s", e)
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html",
            title="Cycle Time Scatterplot",
            script="",
            div="",
        )


@app.route("/netflow")
def netflow_chart():
    """Generate netflow chart."""
    try:
        results = get_real_results()
        chart_data = results[NetFlowChartCalculator]
        if chart_data is None or len(chart_data.index) == 0:
            flash("No data available for Net Flow Chart.", "warning")
            return render_template(
                "bokeh_chart.html", title="Net Flow Chart", script="", div=""
            )
        p = figure(
            title="Net Flow Chart",
            x_axis_type="datetime",
            width=800,
            height=400,
        )
        for col in chart_data.columns:
            p.line(
                chart_data.index,
                chart_data[col],
                legend_label=col,
                line_width=2,
            )
        p.legend.location = "top_left"
        p.xaxis.axis_label = "Date"
        p.yaxis.axis_label = "Items"
        script, div = components(p)
        return render_template(
            "bokeh_chart.html", title="Net Flow Chart", script=script, div=div
        )
    except (ValueError, AttributeError, KeyError, ImportError) as e:
        logger.error("Error generating netflow chart: %s", e)
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Net Flow Chart", script="", div=""
        )


@app.route("/ageingwip")
def ageingwip_chart():
    """Generate ageing WIP chart."""
    try:
        results = get_real_results()
        chart_data = results[AgeingWIPChartCalculator]
        if chart_data is None or len(chart_data.index) == 0:
            flash("No data available for Ageing WIP Chart.", "warning")
            return render_template(
                "bokeh_chart.html", title="Ageing WIP Chart", script="", div=""
            )
        p = figure(
            title="Ageing WIP Chart",
            x_range=list(chart_data["status"].astype(str)),
            width=800,
            height=400,
        )
        p.vbar(
            x=list(chart_data["status"].astype(str)),
            top=list(chart_data["age"]),
            width=0.9,
        )
        p.xaxis.axis_label = "Status"
        p.yaxis.axis_label = "Age (days)"
        script, div = components(p)
        return render_template(
            "bokeh_chart.html",
            title="Ageing WIP Chart",
            script=script,
            div=div,
        )
    except (ValueError, AttributeError, KeyError, ImportError) as e:
        logger.error("Error generating ageing WIP chart: %s", e)
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Ageing WIP Chart", script="", div=""
        )


@app.route("/debt")
def debt_chart():
    """Generate debt chart."""
    try:
        results = get_real_results()
        chart_data = results[DebtCalculator]
        if chart_data is None or len(chart_data.index) == 0:
            flash("No data available for Technical Debt Chart.", "warning")
            return render_template(
                "bokeh_chart.html",
                title="Technical Debt Chart",
                script="",
                div="",
            )
        p = figure(
            title="Technical Debt Chart",
            x_axis_type="datetime",
            width=800,
            height=400,
        )
        p.line(
            chart_data["created"],
            chart_data["age"].dt.days,
            legend_label="Debt Age",
            line_width=2,
        )
        p.xaxis.axis_label = "Created Date"
        p.yaxis.axis_label = "Age (days)"
        script, div = components(p)
        return render_template(
            "bokeh_chart.html",
            title="Technical Debt Chart",
            script=script,
            div=div,
        )
    except (ValueError, AttributeError, KeyError, ImportError) as e:
        logger.error("Error generating debt chart: %s", e)
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Technical Debt Chart", script="", div=""
        )


@app.route("/debt-age")
def debt_age_chart():
    """Generate debt age chart."""
    try:
        results = get_real_results()
        chart_data = results[DebtCalculator]
        if chart_data is None or len(chart_data.index) == 0:
            flash("No data available for Debt Age Chart.", "warning")
            return render_template(
                "bokeh_chart.html", title="Debt Age Chart", script="", div=""
            )
        p = figure(
            title="Debt Age Chart",
            x_range=list(chart_data["priority"].astype(str)),
            width=800,
            height=400,
        )
        p.vbar(
            x=list(chart_data["priority"].astype(str)),
            top=list(chart_data["age"].dt.days),
            width=0.9,
        )
        p.xaxis.axis_label = "Priority"
        p.yaxis.axis_label = "Age (days)"
        script, div = components(p)
        return render_template(
            "bokeh_chart.html", title="Debt Age Chart", script=script, div=div
        )
    except (ValueError, AttributeError, KeyError, ImportError) as e:
        logger.error("Error generating debt age chart: %s", e)
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Debt Age Chart", script="", div=""
        )


@app.route("/defects-priority")
def defects_priority_chart():
    """Generate defects priority chart."""
    try:
        results = get_real_results()
        chart_data = results[DefectsCalculator]
        if chart_data is None or len(chart_data.index) == 0:
            flash("No data available for Defects by Priority.", "warning")
            return render_template(
                "bokeh_chart.html",
                title="Defects by Priority",
                script="",
                div="",
            )
        p = figure(
            title="Defects by Priority",
            x_range=list(chart_data["priority"].astype(str)),
            width=800,
            height=400,
        )
        p.vbar(
            x=list(chart_data["priority"].astype(str)),
            top=list(chart_data["key"].value_counts()),
            width=0.9,
        )
        p.xaxis.axis_label = "Priority"
        p.yaxis.axis_label = "Count"
        script, div = components(p)
        return render_template(
            "bokeh_chart.html",
            title="Defects by Priority",
            script=script,
            div=div,
        )
    except (ValueError, AttributeError, KeyError, ImportError) as e:
        logger.error("Error generating defects priority chart: %s", e)
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Defects by Priority", script="", div=""
        )


@app.route("/defects-type")
def defects_type_chart():
    """Generate defects type chart."""
    try:
        results = get_real_results()
        chart_data = results[DefectsCalculator]
        if chart_data is None or len(chart_data.index) == 0:
            flash("No data available for Defects by Type.", "warning")
            return render_template(
                "bokeh_chart.html", title="Defects by Type", script="", div=""
            )
        p = figure(
            title="Defects by Type",
            x_range=list(chart_data["type"].astype(str)),
            width=800,
            height=400,
        )
        p.vbar(
            x=list(chart_data["type"].astype(str)),
            top=list(chart_data["key"].value_counts()),
            width=0.9,
        )
        p.xaxis.axis_label = "Type"
        p.yaxis.axis_label = "Count"
        script, div = components(p)
        return render_template(
            "bokeh_chart.html", title="Defects by Type", script=script, div=div
        )
    except (ValueError, AttributeError, KeyError, ImportError) as e:
        logger.error("Error generating defects type chart: %s", e)
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Defects by Type", script="", div=""
        )


@app.route("/defects-environment")
def defects_environment_chart():
    """Generate defects environment chart."""
    try:
        results = get_real_results()
        chart_data = results[DefectsCalculator]
        if chart_data is None or len(chart_data.index) == 0:
            flash("No data available for Defects by Environment.", "warning")
            return render_template(
                "bokeh_chart.html",
                title="Defects by Environment",
                script="",
                div="",
            )
        p = figure(
            title="Defects by Environment",
            x_range=list(chart_data["environment"].astype(str)),
            width=800,
            height=400,
        )
        p.vbar(
            x=list(chart_data["environment"].astype(str)),
            top=list(chart_data["key"].value_counts()),
            width=0.9,
        )
        p.xaxis.axis_label = "Environment"
        p.yaxis.axis_label = "Count"
        script, div = components(p)
        return render_template(
            "bokeh_chart.html",
            title="Defects by Environment",
            script=script,
            div=div,
        )
    except (ValueError, AttributeError, KeyError, ImportError) as e:
        logger.error("Error generating defects environment chart: %s", e)
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html",
            title="Defects by Environment",
            script="",
            div="",
        )


@app.route("/impediments")
def impediments_chart():
    """Generate impediments chart."""
    try:
        results = get_real_results()
        chart_data = results[ImpedimentsCalculator]
        if chart_data is None or len(chart_data.index) == 0:
            flash("No data available for Impediments Chart.", "warning")
            return render_template(
                "bokeh_chart.html",
                title="Impediments Chart",
                script="",
                div="",
            )
        p = figure(
            title="Impediments Chart",
            x_range=list(chart_data["status"].astype(str)),
            width=800,
            height=400,
        )
        p.vbar(
            x=list(chart_data["status"].astype(str)),
            top=list(chart_data["key"].value_counts()),
            width=0.9,
        )
        p.xaxis.axis_label = "Status"
        p.yaxis.axis_label = "Count"
        script, div = components(p)
        return render_template(
            "bokeh_chart.html",
            title="Impediments Chart",
            script=script,
            div=div,
        )
    except (ValueError, AttributeError, KeyError, ImportError) as e:
        logger.error("Error generating impediments chart: %s", e)
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Impediments Chart", script="", div=""
        )


@app.route("/waste")
def waste_chart():
    """Generate waste chart."""
    try:
        results = get_real_results()
        chart_data = results[WasteCalculator]
        if chart_data is None or len(chart_data.index) == 0:
            flash("No data available for Waste Chart.", "warning")
            return render_template(
                "bokeh_chart.html", title="Waste Chart", script="", div=""
            )
        p = figure(
            title="Waste Chart",
            x_range=list(chart_data["last_status"].astype(str)),
            width=800,
            height=400,
        )
        p.vbar(
            x=list(chart_data["last_status"].astype(str)),
            top=list(chart_data["key"].value_counts()),
            width=0.9,
        )
        p.xaxis.axis_label = "Last Status"
        p.yaxis.axis_label = "Count"
        script, div = components(p)
        return render_template(
            "bokeh_chart.html", title="Waste Chart", script=script, div=div
        )
    except (ValueError, AttributeError, KeyError, ImportError) as e:
        logger.error("Error generating waste chart: %s", e)
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Waste Chart", script="", div=""
        )


@app.route("/progress")
def progress_chart():
    """Generate progress chart."""
    try:
        results = get_real_results()
        chart_data = results[ProgressReportCalculator]
        if not chart_data or "teams" not in chart_data or len(chart_data["teams"]) == 0:
            flash("No data available for Progress Report Chart.", "warning")
            return render_template(
                "bokeh_chart.html",
                title="Progress Report Chart",
                script="",
                div="",
            )
        p = figure(title="Progress Report Chart", width=800, height=400)
        for team in chart_data["teams"]:
            p.line(
                [1, 2, 3], [1, 2, 3], legend_label=team.name, line_width=2
            )  # Placeholder
        p.legend.location = "top_left"
        p.xaxis.axis_label = "X"
        p.yaxis.axis_label = "Y"
        script, div = components(p)
        return render_template(
            "bokeh_chart.html",
            title="Progress Report Chart",
            script=script,
            div=div,
        )
    except (ValueError, AttributeError, KeyError, ImportError) as e:
        logger.error("Error generating progress chart: %s", e)
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html",
            title="Progress Report Chart",
            script="",
            div="",
        )


@app.route("/percentiles")
def percentiles_chart():
    """Generate percentiles chart."""
    try:
        results = get_real_results()
        chart_data = results[PercentilesCalculator]
        if chart_data is None or len(chart_data) == 0:
            flash("No data available for Percentiles Chart.", "warning")
            return render_template(
                "bokeh_chart.html",
                title="Percentiles Chart",
                script="",
                div="",
            )
        p = figure(
            title="Percentiles Chart",
            x_range=[str(q) for q in chart_data.index],
            width=800,
            height=400,
        )
        p.vbar(
            x=[str(q) for q in chart_data.index],
            top=list(chart_data.values),
            width=0.9,
        )
        p.xaxis.axis_label = "Quantile"
        p.yaxis.axis_label = "Cycle Time"
        script, div = components(p)
        return render_template(
            "bokeh_chart.html",
            title="Percentiles Chart",
            script=script,
            div=div,
        )
    except (ValueError, AttributeError, KeyError, ImportError) as e:
        logger.error("Error generating percentiles chart: %s", e)
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Percentiles Chart", script="", div=""
        )


@app.route("/cycletime")
def cycletime_chart():
    """Generate cycle time chart."""
    try:
        results = get_real_results()
        chart_data = results[CycleTimeCalculator]
        if chart_data is None or len(chart_data.index) == 0:
            flash("No data available for Cycle Time Chart.", "warning")
            return render_template(
                "bokeh_chart.html", title="Cycle Time Chart", script="", div=""
            )
        p = figure(
            title="Cycle Time Chart",
            x_range=list(chart_data["key"].astype(str)),
            width=800,
            height=400,
        )
        p.vbar(
            x=list(chart_data["key"].astype(str)),
            top=list(chart_data["cycle_time"].dt.days),
            width=0.9,
        )
        p.xaxis.axis_label = "Issue Key"
        p.yaxis.axis_label = "Cycle Time (days)"
        script, div = components(p)
        return render_template(
            "bokeh_chart.html",
            title="Cycle Time Chart",
            script=script,
            div=div,
        )
    except (ValueError, AttributeError, KeyError, ImportError) as e:
        logger.error("Error generating cycle time chart: %s", e)
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Cycle Time Chart", script="", div=""
        )


@app.route("/set_query", methods=["POST"])
def set_query():
    """Set custom JIRA query from user input."""
    user_query = request.form.get("user_query", "").strip()

    # Validate input length to prevent potential issues
    if len(user_query) > 1000:
        flash("JQL query is too long. Maximum length is 1000 characters.", "danger")
        return redirect(url_for("index"))

    # Store the user query in session
    # Note: XSS protection is handled by Flask/Jinja2 auto-escaping and the fact that
    # the query is only used server-side to construct JQL API calls to JIRA.
    if user_query:
        session["user_query"] = user_query
        flash("Custom JQL query set!", "success")
    else:
        session.pop("user_query", None)
        flash("Custom JQL query cleared. Using default from config.", "info")
    return redirect(url_for("index"))
