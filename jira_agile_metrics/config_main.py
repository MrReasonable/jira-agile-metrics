import logging

from .calculators.ageingwip import AgeingWIPChartCalculator
from .calculators.burnup import BurnupCalculator
from .calculators.cfd import CFDCalculator
from .calculators.cycletime import BottleneckChartsCalculator, CycleTimeCalculator
from .calculators.debt import DebtCalculator
from .calculators.defects import DefectsCalculator
from .calculators.forecast import BurnupForecastCalculator
from .calculators.histogram import HistogramCalculator
from .calculators.impediments import ImpedimentsCalculator
from .calculators.netflow import NetFlowChartCalculator
from .calculators.percentiles import PercentilesCalculator
from .calculators.progressreport import ProgressReportCalculator
from .calculators.scatterplot import ScatterplotCalculator
from .calculators.throughput import ThroughputCalculator
from .calculators.waste import WasteCalculator
from .calculators.wip import WIPChartCalculator

CALCULATORS = (
    CycleTimeCalculator,  # should come first
    BottleneckChartsCalculator,  # now included for bottleneck visualizations
    # -- others depend on results from this one
    CFDCalculator,  # needs to come before burn-up charts,
    # wip charts, and net flow charts
    ScatterplotCalculator,
    HistogramCalculator,
    PercentilesCalculator,
    ThroughputCalculator,
    BurnupCalculator,
    WIPChartCalculator,
    NetFlowChartCalculator,
    AgeingWIPChartCalculator,
    BurnupForecastCalculator,
    ImpedimentsCalculator,
    DebtCalculator,
    DefectsCalculator,
    WasteCalculator,
    ProgressReportCalculator,
)

logger = logging.getLogger(__name__)
