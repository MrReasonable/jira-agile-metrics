import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..calculator import Calculator
from ..utils import get_extension, set_chart_style
from .cycletime import CycleTimeCalculator

logger = logging.getLogger(__name__)


class HistogramCalculator(Calculator):
    """Build histogram data for the cycle times in `cycle_data`. Returns
    a dictionary with keys `bin_values` and `bin_edges` of numpy arrays
    """

    def run(self):
        cycle_data = self.get_result(CycleTimeCalculator)

        cycle_times = cycle_data["cycle_time"].dt.days.dropna().tolist()

        if not cycle_times:
            bins = range(11)
        else:
            bins = range(int(max(cycle_times)) + 2)

        values, edges = np.histogram(cycle_times, bins=bins, density=False)

        index = []
        for i, _ in enumerate(edges):
            if i == 0:
                continue
            index.append(
                "%.01f to %.01f"
                % (
                    edges[i - 1],
                    edges[i],
                )
            )

        return pd.Series(values, name="Items", index=index)

    def write(self):
        data = self.get_result()

        logger.debug(
            f"[DEBUG] lead_time_histogram_data: {self.settings.get('lead_time_histogram_data')}"
        )
        logger.debug(
            f"[DEBUG] lead_time_histogram_chart: {self.settings.get('lead_time_histogram_chart')}"
        )

        if self.settings["histogram_data"]:
            self.write_file(data, self.settings["histogram_data"])
        else:
            logger.debug("No output file specified for histogram data")

        if self.settings["histogram_chart"]:
            self.write_chart(data, self.settings["histogram_chart"])
        else:
            logger.debug("No output file specified for histogram chart")

        # Lead time histogram outputs
        if self.settings.get("lead_time_histogram_data"):
            self.write_lead_time_file(
                data, self.settings["lead_time_histogram_data"]
            )
        else:
            logger.debug(
                "No output file specified for lead time histogram data"
            )

        if self.settings.get("lead_time_histogram_chart"):
            for output_file in self.settings["lead_time_histogram_chart"]:
                self.write_lead_time_chart(data, output_file)
        else:
            logger.debug(
                "No output file specified for lead time histogram chart"
            )

    def write_file(self, data, output_files):
        file_data = self.get_result()

        for output_file in output_files:
            output_extension = get_extension(output_file)

            logger.info("Writing histogram data to %s", output_file)
            if output_extension == ".json":
                file_data.to_json(output_file, date_format="iso")
            elif output_extension == ".xlsx":
                file_data.to_frame(name="histogram").to_excel(
                    output_file, "Histogram", header=True
                )
            else:
                file_data.to_csv(output_file, header=True)

    def write_chart(self, data, output_file):
        cycle_data = self.get_result(CycleTimeCalculator)
        chart_data = cycle_data[["cycle_time", "completed_timestamp"]].dropna(
            subset=["cycle_time"]
        )

        # The `window` calculation and the chart output will fail if we don't
        # have at least two valid data points.
        ct_days = chart_data["cycle_time"].dt.days
        if len(ct_days.index) < 2:
            logger.warning("Need at least 2 completed items to draw histogram")
            return

        # Slice off items before the window
        window = self.settings["histogram_window"]
        if window:
            start = chart_data[
                "completed_timestamp"
            ].max().normalize() - pd.Timedelta(window, "D")
            chart_data = chart_data[chart_data.completed_timestamp >= start]

            # Re-check that we have enough data
            ct_days = chart_data["cycle_time"].dt.days
            if len(ct_days.index) < 2:
                logger.warning(
                    "Need at least 2 completed items to draw histogram"
                )
                return

        quantiles = self.settings["quantiles"]
        logger.debug(
            "Showing histogram at quantiles %s",
            ", ".join(["%.2f" % (q * 100.0) for q in quantiles]),
        )

        fig, ax = plt.subplots()
        bins = range(int(ct_days.max()) + 2)

        sns.histplot(ct_days, bins=bins, ax=ax, kde=False)
        ax.set_xlabel("Cycle time (days)")

        if self.settings["histogram_chart_title"]:
            ax.set_title(self.settings["histogram_chart_title"])

        _, right = ax.get_xlim()
        ax.set_xlim(0, right)

        # Add quantiles
        bottom, top = ax.get_ylim()
        quantile_labels = []
        for quantile, value in ct_days.quantile(quantiles).items():
            ax.vlines(
                value, bottom, top - 0.001, linestyles="--", linewidths=1
            )
            label = "%.0f%% (%.0f days)" % ((quantile * 100), value)
            quantile_labels.append(label)

        # Add quantile labels as a block of text below the chart
        if quantile_labels:
            fig.text(
                0.5,
                -0.03,
                " | ".join(quantile_labels),
                ha="center",
                va="top",
                fontsize="small",
            )

        ax.set_ylabel("Frequency")
        set_chart_style()

        # Write file
        logger.info("Writing histogram chart to %s", output_file)
        fig.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def write_lead_time_file(self, data, output_files):
        cycle_data = self.get_result(CycleTimeCalculator)
        lead_times = cycle_data["lead_time"].dt.days.dropna().tolist()
        if not lead_times:
            bins = range(11)
        else:
            bins = range(int(max(lead_times)) + 2)
        values, edges = np.histogram(lead_times, bins=bins, density=False)
        index = []
        for i, _ in enumerate(edges):
            if i == 0:
                continue
            index.append("%.01f to %.01f" % (edges[i - 1], edges[i]))
        file_data = pd.Series(values, name="Items", index=index)
        for output_file in output_files:
            output_extension = get_extension(output_file)
            logger.info("Writing lead time histogram data to %s", output_file)
            if output_extension == ".json":
                file_data.to_json(output_file, date_format="iso")
            elif output_extension == ".xlsx":
                file_data.to_frame(name="lead_time_histogram").to_excel(
                    output_file, "LeadTimeHistogram", header=True
                )
            else:
                file_data.to_csv(output_file, header=True)

    def write_lead_time_chart(self, data, output_file):
        cycle_data = self.get_result(CycleTimeCalculator)
        chart_data = cycle_data[["lead_time", "completed_timestamp"]].dropna(
            subset=["lead_time"]
        )
        lt_days = chart_data["lead_time"].dt.days
        if len(lt_days.index) < 2:
            logger.warning(
                "Need at least 2 completed items to draw lead time histogram"
            )
            return
        window = self.settings.get("histogram_window")
        if window:
            start = chart_data[
                "completed_timestamp"
            ].max().normalize() - pd.Timedelta(window, "D")
            chart_data = chart_data[chart_data.completed_timestamp >= start]
            lt_days = chart_data["lead_time"].dt.days
            if len(lt_days.index) < 2:
                logger.warning(
                    "Need at least 2 completed items to draw lead time histogram"
                )
                return
        quantiles = self.settings["quantiles"]
        logger.debug(
            "Showing lead time histogram at quantiles %s",
            ", ".join(["%.2f" % (q * 100.0) for q in quantiles]),
        )
        fig, ax = plt.subplots()
        bins = range(int(lt_days.max()) + 2)
        sns.histplot(lt_days, bins=bins, ax=ax, kde=False)
        ax.set_xlabel("Lead time (days)")
        if self.settings.get("lead_time_histogram_chart_title"):
            ax.set_title(self.settings["lead_time_histogram_chart_title"])
        _, right = ax.get_xlim()
        ax.set_xlim(0, right)
        bottom, top = ax.get_ylim()
        quantile_labels = []
        for quantile, value in lt_days.quantile(quantiles).items():
            ax.vlines(
                value, bottom, top - 0.001, linestyles="--", linewidths=1
            )
            label = "%.0f%% (%.0f days)" % ((quantile * 100), value)
            quantile_labels.append(label)
        if quantile_labels:
            fig.text(
                0.5,
                -0.03,
                " | ".join(quantile_labels),
                ha="center",
                va="top",
                fontsize="small",
            )
        ax.set_ylabel("Frequency")
        set_chart_style()
        logger.info("Writing lead time histogram chart to %s", output_file)
        fig.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close(fig)
