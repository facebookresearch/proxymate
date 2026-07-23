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
from sklearn.metrics import r2_score


@dataclass
class RMSEResult:
    rmse: float
    normalized_rmse: float
    n_samples: int


class RMSECheck(CheckBase[RMSEResult, CheckInput]):
    """Root Mean Square Error, normalized by primary standard deviation.

    Measures the typical magnitude of proxy errors. Normalized RMSE
    divides by std(Y) to produce a scale-free metric.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.PRECISION
    supported_outcome_types: ClassVar[list[OutcomeType]] = [OutcomeType.CONTINUOUS]

    def get_name(self) -> str:
        return "RMSE"

    def get_description(self) -> str:
        return "Root Mean Square Error normalized by primary standard deviation."

    def compute(self, input: CheckInput) -> RMSEResult:
        primary, proxy = extract_paired_arrays(input)

        if len(primary) < 2:
            return RMSEResult(
                rmse=float("inf"), normalized_rmse=float("inf"), n_samples=len(primary)
            )

        rmse = float(np.sqrt(np.mean((primary - proxy) ** 2)))
        std_primary = float(np.std(primary, ddof=1))
        normalized = rmse / std_primary if std_primary > 0 else float("inf")
        return RMSEResult(
            rmse=rmse,
            normalized_rmse=normalized,
            n_samples=len(primary),
        )

    def project(self, result: RMSEResult) -> float:
        return 1.0 / (1.0 + result.normalized_rmse)


@dataclass
class RSquaredResult:
    r_squared: float
    n_samples: int


class RSquaredCheck(CheckBase[RSquaredResult, CheckInput]):
    """Coefficient of determination (R²) from OLS regression.

    Measures the fraction of primary variance explained by the proxy.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.PRECISION
    supported_outcome_types: ClassVar[list[OutcomeType]] = [OutcomeType.CONTINUOUS]

    def get_name(self) -> str:
        return "R Squared"

    def get_description(self) -> str:
        return "Coefficient of determination measuring variance explained by the proxy."

    def compute(self, input: CheckInput) -> RSquaredResult:
        primary, proxy, weights = extract_paired_arrays_with_weights(input)

        if len(primary) < 2:
            return RSquaredResult(r_squared=0.0, n_samples=len(primary))

        r2 = float(r2_score(primary, proxy, sample_weight=weights))
        return RSquaredResult(r_squared=r2, n_samples=len(primary))

    def project(self, result: RSquaredResult) -> float:
        return min(1.0, max(0.0, result.r_squared))


@dataclass
class PearsonCorrelationResult:
    correlation: float
    p_value: float
    n_samples: int


class PearsonCorrelationCheck(CheckBase[PearsonCorrelationResult, CheckInput]):
    """Pearson correlation coefficient between primary and proxy.

    Measures the linear association strength. Supports optional
    sample weights via weighted correlation formula.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.PRECISION
    supported_outcome_types: ClassVar[list[OutcomeType]] = [OutcomeType.CONTINUOUS]

    def get_name(self) -> str:
        return "Pearson Correlation"

    def get_description(self) -> str:
        return "Pearson correlation coefficient measuring linear association."

    def compute(self, input: CheckInput) -> PearsonCorrelationResult:
        primary, proxy, weights = extract_paired_arrays_with_weights(input)

        if len(primary) < 2:
            return PearsonCorrelationResult(
                correlation=0.0, p_value=1.0, n_samples=len(primary)
            )

        if weights is not None:
            r = _weighted_correlation(primary, proxy, weights)
            n = len(primary)
            if abs(r) < 1.0 and n > 2:
                t_stat = r * np.sqrt((n - 2) / (1 - r**2))
                p_val = float(2 * stats.t.sf(abs(t_stat), n - 2))
            else:
                p_val = 0.0
        else:
            r_val, p_val = stats.pearsonr(primary, proxy)
            r = float(r_val)
            p_val = float(p_val)

        return PearsonCorrelationResult(
            correlation=r,
            p_value=p_val,
            n_samples=len(primary),
        )

    def project(self, result: PearsonCorrelationResult) -> float:
        return min(1.0, max(0.0, result.correlation))


def _weighted_correlation(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> float:
    total = weights.sum()
    if total == 0:
        return 0.0
    w = weights / total
    mean_x = np.average(x, weights=w)
    mean_y = np.average(y, weights=w)
    cov_xy = np.sum(w * (x - mean_x) * (y - mean_y))
    std_x = np.sqrt(np.sum(w * (x - mean_x) ** 2))
    std_y = np.sqrt(np.sum(w * (y - mean_y) ** 2))
    if std_x == 0 or std_y == 0:
        return 0.0
    return float(cov_xy / (std_x * std_y))
