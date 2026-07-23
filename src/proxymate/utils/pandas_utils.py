# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from proxymate.lib.base_classes.data_types import Estimate


def segmented_estimates_to_dataframe(
    segmented_estimates: Dict[str, Dict[str, Estimate]] | None,
) -> pd.DataFrame:
    """Convert a nested dictionary of Estimates into a pandas DataFrame."""
    if segmented_estimates is None:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, value in segmented_estimates.items():
        for estimate_key, estimate in value.items():
            row = {
                "category": key,
                "key": estimate_key,
                "confidence_interval_level": estimate.confidence_interval_level,
                "estimate_val": estimate.estimate_val,
                "lower_bound": estimate.lower_bound,
                "upper_bound": estimate.upper_bound,
                "sample_size": estimate.sample_size,
                "standard_error": estimate.standard_error,
            }
            rows.append(row)
    return pd.DataFrame(rows)
