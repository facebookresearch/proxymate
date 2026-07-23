# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from proxymate.checks.base import CheckBase
from proxymate.checks.check_utils import extract_paired_arrays
from proxymate.types.enums import OutcomeType, UnitLevelProperty, ValidationLevel
from proxymate.types.inputs import CheckInput
from scipy import stats


@dataclass
class LinCCCResult:
    ccc: float
    pearson_rho: float
    bias_correction: float
    n_samples: int


class LinCCCCheck(CheckBase[LinCCCResult, CheckInput]):
    """Lin's Concordance Correlation Coefficient: CCC = ρ · C_b.

    Measures agreement by combining Pearson correlation (precision)
    with a bias correction factor (accuracy). CCC = 1 means perfect
    agreement on the identity line.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.AGREEMENT
    supported_outcome_types: ClassVar[list[OutcomeType]] = [OutcomeType.CONTINUOUS]

    def get_name(self) -> str:
        return "Lin's CCC"

    def get_description(self) -> str:
        return (
            "Lin's Concordance Correlation Coefficient measuring agreement "
            "on the identity line."
        )

    def compute(self, input: CheckInput) -> LinCCCResult:
        primary, proxy = extract_paired_arrays(input)
        n = len(primary)
        if n < 2:
            return LinCCCResult(
                ccc=0.0, pearson_rho=0.0, bias_correction=0.0, n_samples=n
            )

        mean_y = np.mean(primary)
        mean_yhat = np.mean(proxy)
        var_y = np.var(primary, ddof=1)
        var_yhat = np.var(proxy, ddof=1)
        # Use ddof=1 consistently: sample covariance = Σ(x-μ)(y-μ) / (n-1)
        cov = (
            float(np.sum((primary - mean_y) * (proxy - mean_yhat)) / (n - 1))
            if n > 1
            else 0.0
        )

        # CCC = 2 * cov / (var_y + var_yhat + (mean_y - mean_yhat)^2)
        denom = var_y + var_yhat + (mean_y - mean_yhat) ** 2
        ccc = float(2 * cov / denom) if denom > 0 else 0.0

        # Decompose: CCC = rho * C_b
        std_y = np.sqrt(var_y)
        std_yhat = np.sqrt(var_yhat)
        rho = float(cov / (std_y * std_yhat)) if std_y > 0 and std_yhat > 0 else 0.0
        c_b = float(ccc / rho) if abs(rho) > 1e-10 else 0.0

        return LinCCCResult(
            ccc=ccc,
            pearson_rho=rho,
            bias_correction=c_b,
            n_samples=n,
        )

    def project(self, result: LinCCCResult) -> float:
        return min(1.0, max(0.0, result.ccc))


@dataclass
class BlandAltmanResult:
    mean_diff: float
    std_diff: float
    upper_loa: float
    lower_loa: float
    percent_within_loa: float
    n_samples: int


class BlandAltmanCheck(CheckBase[BlandAltmanResult, CheckInput]):
    """Bland-Altman limits of agreement analysis.

    Computes mean difference (bias) and limits of agreement (LOA)
    between proxy and primary. Reports the percentage of observations
    falling within the LOA.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.AGREEMENT
    supported_outcome_types: ClassVar[list[OutcomeType]] = [OutcomeType.CONTINUOUS]

    def __init__(
        self,
        threshold: float | None = None,
        confidence_level: float = 0.95,
    ) -> None:
        super().__init__(threshold=threshold)
        self.confidence_level = confidence_level

    def get_name(self) -> str:
        return "Bland-Altman"

    def get_description(self) -> str:
        return "Bland-Altman limits of agreement between proxy and primary."

    def compute(self, input: CheckInput) -> BlandAltmanResult:
        primary, proxy = extract_paired_arrays(input)

        if len(primary) < 2:
            return BlandAltmanResult(
                mean_diff=0.0,
                std_diff=0.0,
                upper_loa=0.0,
                lower_loa=0.0,
                percent_within_loa=0.0,
                n_samples=len(primary),
            )
        diff = proxy - primary
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))

        z = float(stats.norm.ppf((1 + self.confidence_level) / 2))
        upper_loa = mean_diff + z * std_diff
        lower_loa = mean_diff - z * std_diff

        within = np.sum((diff >= lower_loa) & (diff <= upper_loa))
        pct = float(within / len(diff)) if len(diff) > 0 else 0.0

        return BlandAltmanResult(
            mean_diff=mean_diff,
            std_diff=std_diff,
            upper_loa=upper_loa,
            lower_loa=lower_loa,
            percent_within_loa=pct,
            n_samples=len(primary),
        )

    def project(self, result: BlandAltmanResult) -> float:
        return min(1.0, max(0.0, result.percent_within_loa))


@dataclass
class DirectionalConcordanceResult:
    concordance_rate: float
    n_concordant: int
    n_total: int


class DirectionalConcordanceCheck(CheckBase[DirectionalConcordanceResult, CheckInput]):
    """Directional concordance: fraction of units where sign(Ŷ) = sign(Y).

    Measures how often proxy and primary agree on direction (both positive
    or both negative/zero).
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.AGREEMENT
    supported_outcome_types: ClassVar[list[OutcomeType]] = [OutcomeType.CONTINUOUS]

    def get_name(self) -> str:
        return "Directional Concordance"

    def get_description(self) -> str:
        return "Fraction of units where proxy and primary agree on direction."

    def compute(self, input: CheckInput) -> DirectionalConcordanceResult:
        primary, proxy = extract_paired_arrays(input)

        both_positive = (primary > 0) & (proxy > 0)
        both_negative = (primary <= 0) & (proxy <= 0)
        concordant = int(np.sum(both_positive | both_negative))
        total = len(primary)

        rate = concordant / total if total > 0 else 0.0
        return DirectionalConcordanceResult(
            concordance_rate=rate,
            n_concordant=concordant,
            n_total=total,
        )

    def project(self, result: DirectionalConcordanceResult) -> float:
        return min(1.0, max(0.0, result.concordance_rate))
