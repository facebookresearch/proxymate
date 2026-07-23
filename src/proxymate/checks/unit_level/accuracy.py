# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from proxymate.checks.base import CheckBase
from proxymate.checks.check_utils import (
    extract_paired_arrays,
    extract_paired_arrays_with_weights,
)
from proxymate.types.enums import OutcomeType, UnitLevelProperty, ValidationLevel
from proxymate.types.inputs import CheckInput
from scipy import stats


@dataclass
class TTestResult:
    t_statistic: float
    p_value: float
    mean_diff: float
    n_samples: int


class TTestCheck(CheckBase[TTestResult, CheckInput]):
    """Welch's t-test for mean equality: E[Ŷ - Y] = 0?

    Tests whether the proxy and primary have equal means using
    Welch's t-test (unequal variances). A low p-value indicates
    the proxy is systematically biased.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.ACCURACY
    supported_outcome_types: ClassVar[list[OutcomeType]] = [OutcomeType.CONTINUOUS]

    def get_name(self) -> str:
        return "T-Test"

    def get_description(self) -> str:
        return "Welch's t-test for mean equality between primary and proxy."

    def get_default_threshold(self) -> float:
        return 0.05

    def compute(self, input: CheckInput) -> TTestResult:
        primary, proxy, weights = extract_paired_arrays_with_weights(input)

        if len(primary) < 2:
            return TTestResult(
                t_statistic=0.0,
                p_value=1.0,
                mean_diff=0.0,
                n_samples=len(primary),
            )

        if weights is not None:
            # Weighted Welch's t-test (no statsmodels dependency)
            mu_p = np.average(primary, weights=weights)
            mu_q = np.average(proxy, weights=weights)
            mean_diff = float(mu_q - mu_p)

            sum_w = np.sum(weights)
            sum_w2 = np.sum(weights**2)
            n_eff = sum_w**2 / sum_w2

            if n_eff <= 1:
                return TTestResult(
                    t_statistic=0.0,
                    p_value=1.0,
                    mean_diff=mean_diff,
                    n_samples=len(primary),
                )

            # Weighted variances with reliability-weight Bessel correction
            var_p = np.average((primary - mu_p) ** 2, weights=weights)
            var_q = np.average((proxy - mu_q) ** 2, weights=weights)
            bessel = n_eff / (n_eff - 1)
            var_p *= bessel
            var_q *= bessel

            se2_p = var_p / n_eff
            se2_q = var_q / n_eff
            denom = np.sqrt(se2_p + se2_q)
            if denom == 0:
                return TTestResult(
                    t_statistic=0.0,
                    p_value=1.0,
                    mean_diff=mean_diff,
                    n_samples=len(primary),
                )

            t_stat = mean_diff / denom

            # Welch-Satterthwaite degrees of freedom
            df = (se2_p + se2_q) ** 2 / (
                se2_p**2 / (n_eff - 1) + se2_q**2 / (n_eff - 1)
            )
            p_value = float(2 * stats.t.sf(abs(float(t_stat)), float(df)))

            return TTestResult(
                t_statistic=float(t_stat),
                p_value=p_value,
                mean_diff=mean_diff,
                n_samples=len(primary),
            )

        result = stats.ttest_ind(primary, proxy, equal_var=False)
        return TTestResult(
            t_statistic=float(result.statistic),
            p_value=float(result.pvalue),
            mean_diff=float(np.mean(proxy) - np.mean(primary)),
            n_samples=len(primary),
        )

    def project(self, result: TTestResult) -> float:
        # High p-value = no significant difference = good proxy.
        return min(1.0, max(0.0, result.p_value))


@dataclass
class MAPEResult:
    mape: float
    n_samples: int


class MAPECheck(CheckBase[MAPEResult, CheckInput]):
    """Mean Absolute Percentage Error.

    Measures average percentage deviation of proxy from primary.
    Only valid when primary values are non-zero.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.ACCURACY
    supported_outcome_types: ClassVar[list[OutcomeType]] = [OutcomeType.CONTINUOUS]

    def get_name(self) -> str:
        return "MAPE"

    def get_description(self) -> str:
        return "Mean Absolute Percentage Error of proxy relative to primary."

    def compute(self, input: CheckInput) -> MAPEResult:
        primary, proxy = extract_paired_arrays(input)
        nonzero = primary != 0
        primary, proxy = primary[nonzero], proxy[nonzero]
        if len(primary) == 0:
            return MAPEResult(mape=float("inf"), n_samples=0)

        mape = float(np.mean(np.abs((primary - proxy) / primary)))
        return MAPEResult(mape=mape, n_samples=len(primary))

    def project(self, result: MAPEResult) -> float:
        return 1.0 / (1.0 + result.mape)
