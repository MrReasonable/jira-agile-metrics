"""Percentiles calculator for Jira Agile Metrics.

This module provides functionality to calculate percentile metrics from cycle time data.
"""

import logging

from ..calculator import Calculator
from ..utils import get_extension
from .cycletime import CycleTimeCalculator

logger = logging.getLogger(__name__)


class PercentilesCalculator(Calculator):
    """Build percentiles for `cycle_time` in cycle data as a DataFrame"""

    def run(self):
        cycle_data = self.get_result(CycleTimeCalculator)

        quantiles = self.settings["quantiles"]
        percentile_strs = [f"{q * 100.0:.2f}" for q in quantiles]
        logger.debug("Calculating percentiles at %s", ", ".join(percentile_strs))

        return cycle_data["cycle_time"].dropna().quantile(quantiles)

    def write(self):
        output_files = self.settings["percentiles_data"]
        if not output_files:
            logger.debug("No output file specified for percentiles data")
            return

        file_data = self.get_result()

        for output_file in output_files:
            output_extension = get_extension(output_file)
            logger.info("Writing percentiles data to %s", output_file)
            if output_extension == ".json":
                file_data.to_json(output_file, date_format="iso")
            elif output_extension == ".xlsx":
                file_data.to_frame(name="percentiles").to_excel(
                    output_file, "Percentiles", header=True
                )
            else:
                # Reset index to convert it to a named column
                file_data_to_write = file_data.reset_index()
                file_data_to_write.columns = ["Quantile", "cycle_time"]
                file_data_to_write.to_csv(output_file, header=True, index=False)
