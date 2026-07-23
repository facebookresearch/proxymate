# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
from proxymate.lib.base_classes.df_based_estimator_base import DfBasedEstimatorBase
from proxymate.lib.base_classes.estimator_base import Estimate
from proxymate.utils.stats_utils import get_ci


class WeightedSumEstimator(DfBasedEstimatorBase):
    """
    A concrete estimator class for computing the weighted sum of a given dataset.
    """

    def __init__(
        self,
        confidence_level: float | None = 0.95,
        weight_column: str | None = None,
    ) -> None:
        confidence_level_rounded = (
            round(confidence_level, 2) if confidence_level else None
        )
        super().__init__(confidence_level_rounded)
        self.weight_column = weight_column

    def get_name(self) -> str:
        return "Weighted Sum"

    def get_additional_entities_columns(self) -> set[str]:
        return set()

    def get_additional_metric_columns(self) -> set[str]:
        return {self.weight_column} if self.weight_column else set()

    def _compute_estimator(
        self,
        input_df: pd.DataFrame,
    ) -> Estimate:
        if self.weight_column:
            input_df["weighted_metric"] = (
                input_df["metric"] * input_df[self.weight_column]
            )
        else:
            input_df["weighted_metric"] = input_df["metric"]
        estimate_val = input_df["weighted_metric"].sum()
        sample_size = len(input_df)
        weighted_sum_var = input_df["weighted_metric"].var()
        estimate_val_std_error = np.sqrt(sample_size * weighted_sum_var)
        lower_bound, upper_bound = get_ci(
            delta=estimate_val,
            se=estimate_val_std_error,
            conf=int(self._confidence_level * 100),
        )
        return Estimate(
            confidence_interval_level=self._confidence_level,
            estimate_val=estimate_val,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            sample_size=sample_size,
            standard_error=estimate_val_std_error,
        )
