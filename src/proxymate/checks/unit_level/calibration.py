# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from types import ModuleType
from typing import ClassVar

import numpy as np
from proxymate.checks.base import CheckBase
from proxymate.checks.check_utils import (
    extract_paired_arrays,
    extract_paired_arrays_with_weights,
)
from proxymate.types.enums import OutcomeType, UnitLevelProperty, ValidationLevel
from proxymate.types.inputs import CheckInput
from sklearn.metrics import brier_score_loss


def _load_multicalibration_metrics() -> ModuleType:
    """Import the optional ``mcgrad`` dependency on demand.

    Kept out of module scope so ``import proxymate`` works without the
    optional ``calibration`` extra installed.
    """
    try:
        import mcgrad.metrics as mc_metrics  # pyre-ignore[import-error]
    except ImportError as e:
        raise ImportError(
            "The 'mcgrad' package is required for calibration "
            "checks. Install it with: pip install proxymate[calibration]"
        ) from e
    return mc_metrics


@dataclass
class CalibrationRatioResult:
    calibration_ratio: float
    n_samples: int


class CalibrationRatioCheck(CheckBase[CalibrationRatioResult, CheckInput]):
    """Calibration ratio via multicalibration library.

    Measures how well the predicted probabilities match the observed
    rates across binned groups. Requires binary primary (Y ∈ {0,1})
    and probability proxy (Ŷ ∈ [0,1]).
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.CALIBRATION
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.BINARY_PROBABILITY,
    ]

    def get_name(self) -> str:
        return "Calibration Ratio"

    def get_description(self) -> str:
        return "Measures how well predicted probabilities match observed rates."

    def get_default_threshold(self) -> float:
        return 0.5

    def compute(self, input: CheckInput) -> CalibrationRatioResult:
        primary, proxy, weights = extract_paired_arrays_with_weights(input)

        if weights is not None:
            gtl_mean = float(np.average(primary, weights=weights))
            pred_mean = float(np.average(proxy, weights=weights))
            ratio = pred_mean / gtl_mean if gtl_mean > 0 else float("nan")
        else:
            mc_metrics = _load_multicalibration_metrics()
            ratio = float(
                mc_metrics.calibration_ratio(labels=primary, predicted_scores=proxy)
            )
        return CalibrationRatioResult(
            calibration_ratio=ratio,
            n_samples=len(primary),
        )

    def project(self, result: CalibrationRatioResult) -> float:
        return min(1.0, max(0.0, result.calibration_ratio))


@dataclass
class KuiperCalibrationResult:
    ecce_kuiper: float
    ecce_sigma: float
    normalized_kuiper: float
    n_samples: int


class KuiperCalibrationCheck(CheckBase[KuiperCalibrationResult, CheckInput]):
    """ECCE-based Kuiper calibration statistic.

    Uses the Expected Conditional Calibration Error (ECCE) and its
    sigma variant to assess calibration quality. The Kuiper statistic
    is normalized by the minority class prevalence.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.CALIBRATION
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.BINARY_PROBABILITY,
    ]

    def get_name(self) -> str:
        return "Kuiper Calibration"

    def get_description(self) -> str:
        return "ECCE-based Kuiper statistic for calibration quality assessment."

    def get_default_threshold(self) -> float:
        return 0.5

    def compute(self, input: CheckInput) -> KuiperCalibrationResult:
        primary, proxy = extract_paired_arrays(input)

        mc_metrics = _load_multicalibration_metrics()
        kuiper = float(mc_metrics.ecce(labels=primary, predicted_scores=proxy))
        sigma = float(mc_metrics.ecce_sigma(labels=primary, predicted_scores=proxy))

        prevalence = float(np.mean(primary))
        min_prev = min(prevalence, 1 - prevalence)
        normalized = kuiper / min_prev if min_prev > 0 else float("inf")

        return KuiperCalibrationResult(
            ecce_kuiper=kuiper,
            ecce_sigma=sigma,
            normalized_kuiper=normalized,
            n_samples=len(primary),
        )

    def project(self, result: KuiperCalibrationResult) -> float:
        # Lower normalized Kuiper = better calibration. Invert.
        return min(1.0, max(0.0, 1.0 - result.normalized_kuiper))


@dataclass
class ECEResult:
    ece: float
    n_bins: int
    n_samples: int


class ECECheck(CheckBase[ECEResult, CheckInput]):
    """Expected Calibration Error for probability proxies.

    Bins predicted probabilities and measures the weighted average gap
    between predicted confidence and observed accuracy:
    ECE = Σ (n_b / N) |acc_b - conf_b|  (Paper eq. 8).
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.CALIBRATION
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.BINARY_PROBABILITY,
    ]

    def __init__(
        self,
        threshold: float | None = None,
        n_bins: int = 10,
    ) -> None:
        super().__init__(threshold=threshold)
        self.n_bins = n_bins

    def get_name(self) -> str:
        return "Expected Calibration Error"

    def get_description(self) -> str:
        return (
            "Weighted average gap between predicted confidence and observed accuracy."
        )

    def get_default_threshold(self) -> float:
        return 0.5

    def compute(self, input: CheckInput) -> ECEResult:
        primary, proxy = extract_paired_arrays(input)

        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        ece = 0.0
        n = len(primary)
        for i in range(self.n_bins):
            mask = (proxy >= bin_edges[i]) & (proxy < bin_edges[i + 1])
            if i == self.n_bins - 1:
                mask = (proxy >= bin_edges[i]) & (proxy <= bin_edges[i + 1])
            n_b = int(np.sum(mask))
            if n_b == 0:
                continue
            acc_b = float(np.mean(primary[mask]))
            conf_b = float(np.mean(proxy[mask]))
            ece += (n_b / n) * abs(acc_b - conf_b)

        return ECEResult(ece=ece, n_bins=self.n_bins, n_samples=n)

    def project(self, result: ECEResult) -> float:
        # Lower ECE = better. Project as 1 - ECE.
        return min(1.0, max(0.0, 1.0 - result.ece))


@dataclass
class BrierScoreResult:
    brier_score: float
    reliability: float
    resolution: float
    uncertainty: float
    n_samples: int


class BrierScoreCheck(CheckBase[BrierScoreResult, CheckInput]):
    """Brier Score with decomposition for probability proxies.

    BS = (1/N) Σ (Y*_i - Y_i)² ∈ [0, 1].
    Decomposed as BS = Reliability - Resolution + Uncertainty.
    Lower is better.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.CALIBRATION
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.BINARY_PROBABILITY,
    ]

    def __init__(
        self,
        threshold: float | None = None,
        n_bins: int = 10,
    ) -> None:
        super().__init__(threshold=threshold)
        self.n_bins = n_bins

    def get_name(self) -> str:
        return "Brier Score"

    def get_description(self) -> str:
        return "Mean squared error between predicted probabilities and binary outcomes."

    def get_default_threshold(self) -> float:
        return 0.5

    def compute(self, input: CheckInput) -> BrierScoreResult:
        primary, proxy = extract_paired_arrays(input)
        n = len(primary)

        bs = float(brier_score_loss(primary, proxy))

        # Murphy decomposition: BS = Reliability - Resolution + Uncertainty
        base_rate = float(np.mean(primary))
        uncertainty = base_rate * (1.0 - base_rate)

        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        reliability = 0.0
        resolution = 0.0
        for i in range(self.n_bins):
            mask = (proxy >= bin_edges[i]) & (proxy < bin_edges[i + 1])
            if i == self.n_bins - 1:
                mask = (proxy >= bin_edges[i]) & (proxy <= bin_edges[i + 1])
            n_b = int(np.sum(mask))
            if n_b == 0:
                continue
            acc_b = float(np.mean(primary[mask]))
            conf_b = float(np.mean(proxy[mask]))
            reliability += (n_b / n) * (conf_b - acc_b) ** 2
            resolution += (n_b / n) * (acc_b - base_rate) ** 2

        return BrierScoreResult(
            brier_score=bs,
            reliability=reliability,
            resolution=resolution,
            uncertainty=uncertainty,
            n_samples=n,
        )

    def project(self, result: BrierScoreResult) -> float:
        # Lower BS = better. Project as 1 - BS.
        return min(1.0, max(0.0, 1.0 - result.brier_score))


@dataclass
class BrierSkillScoreResult:
    bss: float
    brier_score: float
    brier_ref: float
    n_samples: int


class BrierSkillScoreCheck(CheckBase[BrierSkillScoreResult, CheckInput]):
    """Brier Skill Score: improvement over climatological forecast.

    BSS = 1 - BS / BS_ref where BS_ref = p̄(1 - p̄) is the
    Brier score of always predicting the base rate.
    BSS = 1 is perfect, BSS = 0 is no skill, BSS < 0 is worse than climatology.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.CALIBRATION
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.BINARY_PROBABILITY,
    ]

    def get_name(self) -> str:
        return "Brier Skill Score"

    def get_description(self) -> str:
        return "Skill improvement over base-rate prediction."

    def get_default_threshold(self) -> float:
        return 0.5

    def compute(self, input: CheckInput) -> BrierSkillScoreResult:
        primary, proxy = extract_paired_arrays(input)

        bs = float(brier_score_loss(primary, proxy))
        base_rate = float(np.mean(primary))
        bs_ref = base_rate * (1.0 - base_rate)

        bss = 1.0 - bs / bs_ref if bs_ref > 0 else 0.0

        return BrierSkillScoreResult(
            bss=bss,
            brier_score=bs,
            brier_ref=bs_ref,
            n_samples=len(primary),
        )

    def project(self, result: BrierSkillScoreResult) -> float:
        # BSS in (-inf, 1]. Clamp to [0, 1].
        return min(1.0, max(0.0, result.bss))
