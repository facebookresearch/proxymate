# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Set

import numpy as np
import pandas as pd
from proxymate.lib.base_classes.df_based_estimator_base import DfBasedEstimatorBase
from proxymate.lib.base_classes.estimator_base import Estimate
from proxymate.utils.stats_utils import get_ci


class WeightedAverageEstimator(DfBasedEstimatorBase):
    """
    A concrete estimator class for computing the weighted mean of a given dataset.
    """

    _use_winsorization: bool
    _winsorization_percentage: float
    _scale_factor: bool

    def __init__(
        self,
        confidence_level: float | None = 0.95,
        use_winsorization: bool | None = None,
        winsorization_percentage: float = 99.95,
        scale_factor: bool = True,
        weight_column: str | None = None,
    ) -> None:
        confidence_level_rounded = (
            round(confidence_level, 2) if confidence_level else None
        )
        super().__init__(confidence_level_rounded)
        self._use_winsorization = (
            use_winsorization if use_winsorization is not None else False
        )
        self._winsorization_percentage = winsorization_percentage
        self._scale_factor = scale_factor
        self.weight_column = weight_column

    def get_name(self) -> str:
        return "Weighted Average"

    def get_additional_entities_columns(self) -> Set[str]:
        return set()

    def get_additional_metric_columns(self) -> Set[str]:
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
            input_df["_unit_weight"] = 1.0
        w_col = self.weight_column if self.weight_column else "_unit_weight"
        estimate_val = input_df["weighted_metric"].sum() / input_df[w_col].sum()
        sample_size: int = len(input_df)
        residuals = (input_df["metric"] - estimate_val) * input_df[w_col]
        sum_weights = input_df[w_col].sum()
        # SE of the self-normalizing (Hajek) weighted mean:
        #   sqrt( sum_i w_i^2 (y_i - mu)^2 ) / sum_i w_i
        # The denominator must be (sum w)^2, NOT sum w^2 -- dividing by sum w^2
        # yields the per-observation weighted variance (population spread) rather
        # than the variance of the mean.
        weighted_average_var = residuals.apply(lambda x: x**2).sum() / (sum_weights**2)
        estimate_val_std_error = np.sqrt(weighted_average_var)
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
