import contextlib
import logging
import os
import os.path
import shutil
import tempfile
import threading
import time
import zipfile

import jinja2
from bokeh.embed import components
from bokeh.plotting import figure
from dotenv import load_dotenv
from flask import (Flask, flash, redirect, render_template, request, session,
                   url_for)
from jira import JIRA
from jira.exceptions import JIRAError

from ..calculator import run_calculators
from ..config import ConfigError, config_to_options
from ..config_main import CALCULATORS
from ..querymanager import QueryManager

load_dotenv()

template_folder = os.path.join(os.path.dirname(__file__), "templates")
static_folder = os.path.join(os.path.dirname(__file__), "static")

app = Flask(
    "jira-agile-metrics",
    template_folder=template_folder,
    static_folder=static_folder,
)

app.jinja_loader = jinja2.PackageLoader(
    "jira_agile_metrics.webapp", "templates"
)

logger = logging.getLogger(__name__)

# In-memory cache for calculator results (thread-safe)
results_cache = {}
results_cache_lock = threading.Lock()

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecretkey")


# Helper to load config, create QueryManager, and run calculators
def get_real_results():
    config_path = os.path.join(os.path.dirname(__file__), "../../config.yml")
    with open(config_path) as f:
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
    cache_key = (
        str(options["connection"]) + str(options["settings"]) + str(user_query)
    )
    now = time.time()
    with results_cache_lock:
        if cache_key in results_cache:
            results, timestamp = results_cache[cache_key]
            if now - timestamp < 86400:  # 24 hours
                return results
    jira = get_jira_client(options["connection"])
    query_manager = QueryManager(jira, options["settings"])
    results = run_calculators(CALCULATORS, query_manager, options["settings"])
    with results_cache_lock:
        results_cache[cache_key] = (results, now)
    return results


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/burnup-forecast")
def burnup_forecast():
    try:
        results = get_real_results()
        from ..calculators.forecast import BurnupForecastCalculator

        chart_data = results[BurnupForecastCalculator]
        if chart_data is None or len(chart_data.index) == 0:
            flash("No data available for Burnup Forecast Chart.", "warning")
            return render_template("burnup_forecast.html", script="", div="")
        from bokeh.embed import components
        from bokeh.plotting import figure

        p = figure(
            title="Burnup Forecast Chart",
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
        return render_template("burnup_forecast.html", script=script, div=div)
    except Exception as e:
        flash(str(e), "danger")
        return render_template("burnup_forecast.html", script="", div="")


@app.route("/burnup")
def burnup_chart():
    try:
        results = get_real_results()
        from ..calculators.burnup import BurnupCalculator

        chart_data = results[BurnupCalculator]
        if chart_data is None or len(chart_data.index) == 0:
            flash("No data available for Burnup Chart.", "warning")
            return render_template(
                "bokeh_chart.html", title="Burnup Chart", script="", div=""
            )
        p = figure(
            title="Burnup Chart", x_axis_type="datetime", width=800, height=400
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
            "bokeh_chart.html", title="Burnup Chart", script=script, div=div
        )
    except Exception as e:
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Burnup Chart", script="", div=""
        )


@app.route("/cfd")
def cfd_chart():
    try:
        results = get_real_results()
        from ..calculators.cfd import CFDCalculator

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
    except Exception as e:
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html",
            title="Cumulative Flow Diagram (CFD)",
            script="",
            div="",
        )


@app.route("/histogram")
def histogram_chart():
    try:
        results = get_real_results()
        from ..calculators.histogram import HistogramCalculator

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
        p.vbar(
            x=list(chart_data.index), top=list(chart_data.values), width=0.9
        )
        p.xaxis.axis_label = "Cycle Time Bin"
        p.yaxis.axis_label = "Items"
        script, div = components(p)
        return render_template(
            "bokeh_chart.html",
            title="Cycle Time Histogram",
            script=script,
            div=div,
        )
    except Exception as e:
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Cycle Time Histogram", script="", div=""
        )


@app.route("/scatterplot")
def scatterplot_chart():
    try:
        results = get_real_results()
        from ..calculators.scatterplot import ScatterplotCalculator

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
        p.circle(
            chart_data["x"], chart_data["y"], size=8, color="navy", alpha=0.5
        )
        p.xaxis.axis_label = "X"
        p.yaxis.axis_label = "Y"
        script, div = components(p)
        return render_template(
            "bokeh_chart.html",
            title="Cycle Time Scatterplot",
            script=script,
            div=div,
        )
    except Exception as e:
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html",
            title="Cycle Time Scatterplot",
            script="",
            div="",
        )


@app.route("/netflow")
def netflow_chart():
    try:
        results = get_real_results()
        from ..calculators.netflow import NetFlowChartCalculator

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
    except Exception as e:
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Net Flow Chart", script="", div=""
        )


@app.route("/ageingwip")
def ageingwip_chart():
    try:
        results = get_real_results()
        from ..calculators.ageingwip import AgeingWIPChartCalculator

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
    except Exception as e:
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Ageing WIP Chart", script="", div=""
        )


@app.route("/debt")
def debt_chart():
    try:
        results = get_real_results()
        from ..calculators.debt import DebtCalculator

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
    except Exception as e:
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Technical Debt Chart", script="", div=""
        )


@app.route("/debt-age")
def debt_age_chart():
    try:
        results = get_real_results()
        from ..calculators.debt import DebtCalculator

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
    except Exception as e:
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Debt Age Chart", script="", div=""
        )


@app.route("/defects-priority")
def defects_priority_chart():
    try:
        results = get_real_results()
        from ..calculators.defects import DefectsCalculator

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
    except Exception as e:
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Defects by Priority", script="", div=""
        )


@app.route("/defects-type")
def defects_type_chart():
    try:
        results = get_real_results()
        from ..calculators.defects import DefectsCalculator

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
    except Exception as e:
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Defects by Type", script="", div=""
        )


@app.route("/defects-environment")
def defects_environment_chart():
    try:
        results = get_real_results()
        from ..calculators.defects import DefectsCalculator

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
    except Exception as e:
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html",
            title="Defects by Environment",
            script="",
            div="",
        )


@app.route("/impediments")
def impediments_chart():
    try:
        results = get_real_results()
        from ..calculators.impediments import ImpedimentsCalculator

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
    except Exception as e:
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Impediments Chart", script="", div=""
        )


@app.route("/waste")
def waste_chart():
    try:
        results = get_real_results()
        from ..calculators.waste import WasteCalculator

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
    except Exception as e:
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Waste Chart", script="", div=""
        )


@app.route("/progress")
def progress_chart():
    try:
        results = get_real_results()
        from ..calculators.progressreport import ProgressReportCalculator

        chart_data = results[ProgressReportCalculator]
        if (
            not chart_data
            or "teams" not in chart_data
            or len(chart_data["teams"]) == 0
        ):
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
    except Exception as e:
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html",
            title="Progress Report Chart",
            script="",
            div="",
        )


@app.route("/percentiles")
def percentiles_chart():
    try:
        results = get_real_results()
        from ..calculators.percentiles import PercentilesCalculator

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
    except Exception as e:
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Percentiles Chart", script="", div=""
        )


@app.route("/cycletime")
def cycletime_chart():
    try:
        results = get_real_results()
        from ..calculators.cycletime import CycleTimeCalculator

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
    except Exception as e:
        flash(str(e), "danger")
        return render_template(
            "bokeh_chart.html", title="Cycle Time Chart", script="", div=""
        )


@app.route("/set_query", methods=["POST"])
def set_query():
    user_query = request.form.get("user_query", "").strip()
    if user_query:
        session["user_query"] = user_query
        flash("Custom JQL query set!", "success")
    else:
        session.pop("user_query", None)
        flash("Custom JQL query cleared. Using default from config.", "info")
    return redirect(url_for("index"))


# Helpers


@contextlib.contextmanager
def capture_log(buffer, level, formatter=None):
    """Temporarily write log output to the StringIO `buffer` with log level
    threshold `level`, before returning logging to normal.
    """
    root_logger = logging.getLogger()

    old_level = root_logger.getEffectiveLevel()
    root_logger.setLevel(level)

    handler = logging.StreamHandler(buffer)

    if formatter:
        formatter = logging.Formatter(formatter)
        handler.setFormatter(formatter)

    root_logger.addHandler(handler)

    yield

    root_logger.removeHandler(handler)
    root_logger.setLevel(old_level)

    handler.flush()
    buffer.flush()


def override_options(options, form):
    """Override options from the configuration files with form data where
    applicable.
    """
    for key in options.keys():
        if key in form and form[key] != "":
            options[key] = form[key]


def get_jira_client(connection):
    """Create a JIRA client with the given connection options"""

    url = connection["domain"] or os.environ.get("JIRA_URL")
    username = connection["username"]
    if not username:
        username = os.environ.get("JIRA_USERNAME")
    password = connection["password"]
    if not password:
        password = os.environ.get("JIRA_PASSWORD")
    jira_client_options = connection["jira_client_options"]
    jira_server_version_check = connection["jira_server_version_check"]

    jira_options = {"server": url, "rest_api_version": 3}
    jira_options.update(jira_client_options)

    try:
        return JIRA(
            jira_options,
            basic_auth=(username, password),
            get_server_info=jira_server_version_check,
        )
    except JIRAError as e:
        if e.status_code == 401:
            raise ConfigError(
                (
                    "JIRA authentication failed. "
                    "Check URL and credentials, "
                    "and ensure the account is not locked."
                )
            )
        else:
            raise


def get_archive(calculators, query_manager, settings):
    """Run all calculators and write outputs to a temporary directory.
    Create a zip archive of all the files written, and return it as a bytes
    array. Remove the temporary directory on completion.
    """
    zip_data = b""

    cwd = os.getcwd()
    temp_path = tempfile.mkdtemp()

    try:
        os.chdir(temp_path)
        run_calculators(calculators, query_manager, settings)

        with zipfile.ZipFile("metrics.zip", "w", zipfile.ZIP_STORED) as z:
            for root, dirs, files in os.walk(temp_path):
                for file_name in files:
                    if file_name != "metrics.zip":
                        z.write(
                            os.path.join(root, file_name),
                            os.path.join("metrics", file_name),
                        )
        with open("metrics.zip", "rb") as metrics_zip:
            zip_data = metrics_zip.read()

    finally:
        os.chdir(cwd)
        shutil.rmtree(temp_path)

    return zip_data
