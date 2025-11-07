"""Type definitions for burnup chart generation."""

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class FanPlotParams:
    """Parameters for plotting fan bands."""

    forecast_dates: List[pd.Timestamp]
    fan_data: np.ndarray
    percentiles: List[int]
    style: Dict[str, Any]
