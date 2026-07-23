# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Set

import numpy as np
import pandas as pd
from proxymate.lib.base_classes.df_based_estimator_base import DfBasedEstimatorBase
from proxymate.lib.base_classes.estimator_base import Estimate
from proxymate.utils.stats_utils import calculate_average, calculate_median, get_ci


class MedianCausalEffectEstimator(DfBasedEstimatorBase):
    """
    Estimator for the median causal effect of a given dataset.
    """

    _use_winsorization: bool
    _winsorization_percentage: float

    def __init__(
        self,
        experiment_name: str,
        control_name: str,
        test_name: str,
        confidence_level: float | None = 0.95,
        use_winsorization: bool | None = None,
        winsorization_percentage: float = 99.95,
        weight_column: str | None = None,
        variance_method: str = "kde",
    ) -> None:
        confidence_level_rounded = (
            round(confidence_level, 2) if confidence_level else None
        )
        super().__init__(confidence_level_rounded)
        self._use_winsorization = (
            use_winsorization if use_winsorization is not None else False
        )
        self._winsorization_percentage = winsorization_percentage
        self._weight_column = weight_column
        self._experiment_name = experiment_name
        self._control_name = control_name
        self._test_name = test_name
        self._variance_method = variance_method

    def get_name(self) -> str:
        return "Median Causal Effect"

    def get_additional_entities_columns(self) -> Set[str]:
        return {
            "experiment_id",
            "condition",
        }

    def get_estimator_default_confidence_level(self) -> float:
        return 0.95

    def get_additional_metric_columns(self) -> Set[str]:
        return {self._weight_column} if self._weight_column else set()

    def _compute_estimator(
        self,
        input_df: pd.DataFrame,
    ) -> Estimate:
        experiment_name = self._experiment_name

        input_df = input_df[input_df["experiment_id"] == experiment_name]
        if self._weight_column:
            input_df["weighted_metric"] = (
                input_df["metric"] * input_df[self._weight_column]
            )
        else:
            input_df["weighted_metric"] = input_df["metric"]

        list_values_control = input_df[input_df["condition"] == self._control_name][
            "weighted_metric"
        ].tolist()
        pe_control, std_control, sample_size_control = calculate_median(
            list_values_control, self._variance_method
        )

        list_values_test = input_df[input_df["condition"] == self._test_name][
            "weighted_metric"
        ].tolist()
        pe_test, std_test, _ = calculate_median(
            list_values_test, method=self._variance_method
        )

        estimate_val = pe_test - pe_control
        estimate_val_std_error = np.sqrt(std_control**2 + std_test**2)
        lower_bound, upper_bound = get_ci(
            estimate_val, estimate_val_std_error, int(self._confidence_level * 100)
        )
        return Estimate(
            confidence_interval_level=self._confidence_level,
            estimate_val=estimate_val,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            sample_size=sample_size_control,
            standard_error=estimate_val_std_error,
        )


class AverageTreatmentEffectEstimator(DfBasedEstimatorBase):
    """
    Estimator for the average treatment effect of a given dataset.
    """

    _use_winsorization: bool
    _winsorization_percentage: float

    def __init__(
        self,
        experiment_name: str,
        control_name: str,
        test_name: str,
        confidence_level: float | None = 0.95,
        use_winsorization: bool | None = None,
        winsorization_percentage: float = 99.95,
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
        self._weight_column = weight_column
        self._experiment_name = experiment_name
        self._control_name = control_name
        self._test_name = test_name

    def get_name(self) -> str:
        return "Average Treatment Effect"

    def get_additional_entities_columns(self) -> Set[str]:
        return {
            "experiment_id",
            "condition",
        }

    def get_estimator_default_confidence_level(self) -> float:
        return 0.95

    def get_additional_metric_columns(self) -> Set[str]:
        return {self._weight_column} if self._weight_column else set()

    def _compute_estimator(
        self,
        input_df: pd.DataFrame,
    ) -> Estimate:
        if "experiment_id" in input_df.columns:
            input_df = input_df[input_df["experiment_id"] == self._experiment_name]
        if self._weight_column:
            input_df = input_df.copy()
            input_df["weighted_metric"] = (
                input_df["metric"] * input_df[self._weight_column]
            )
        else:
            input_df = input_df.copy()
            input_df["weighted_metric"] = input_df["metric"]

        list_values_control = input_df[input_df["condition"] == self._control_name][
            "weighted_metric"
        ].tolist()
        pe_control, var_control, sample_size_control = calculate_average(
            list_values_control
        )

        list_values_test = input_df[input_df["condition"] == self._test_name][
            "weighted_metric"
        ].tolist()
        pe_test, var_test, sample_size_test = calculate_average(list_values_test)

        estimate_val = pe_test - pe_control
        estimate_val_std_error = np.sqrt(
            var_control / sample_size_control + var_test / sample_size_test
        )
        lower_bound, upper_bound = get_ci(
            estimate_val, estimate_val_std_error, int(self._confidence_level * 100)
        )
        return Estimate(
            confidence_interval_level=self._confidence_level,
            estimate_val=estimate_val,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            sample_size=sample_size_control,
            standard_error=estimate_val_std_error,
        )
