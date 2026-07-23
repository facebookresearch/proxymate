# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import ClassVar

from proxymate.checks.base import CheckBase
from proxymate.checks.check_utils import (
    extract_paired_arrays,
    extract_paired_arrays_with_weights,
)
from proxymate.types.enums import OutcomeType, UnitLevelProperty, ValidationLevel
from proxymate.types.inputs import CheckInput
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --- Continuous outcome: rank-based checks ---


@dataclass
class SpearmanCorrelationResult:
    correlation: float
    p_value: float
    n_samples: int


class SpearmanCorrelationCheck(CheckBase[SpearmanCorrelationResult, CheckInput]):
    """Spearman rank correlation between primary and proxy.

    Measures monotonic association — whether the proxy preserves
    the relative ordering of the primary.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.DISCRIMINATION
    supported_outcome_types: ClassVar[list[OutcomeType]] = [OutcomeType.CONTINUOUS]

    def get_name(self) -> str:
        return "Spearman Correlation"

    def get_description(self) -> str:
        return "Spearman rank correlation measuring monotonic association."

    def compute(self, input: CheckInput) -> SpearmanCorrelationResult:
        primary, proxy = extract_paired_arrays(input)

        r, p = stats.spearmanr(primary, proxy)
        return SpearmanCorrelationResult(
            correlation=float(r),
            p_value=float(p),
            n_samples=len(primary),
        )

    def project(self, result: SpearmanCorrelationResult) -> float:
        return min(1.0, max(0.0, result.correlation))


@dataclass
class KendallTauResult:
    tau: float
    p_value: float
    n_samples: int


class KendallTauCheck(CheckBase[KendallTauResult, CheckInput]):
    """Kendall's τ rank correlation between primary and proxy.

    A non-parametric measure of ordinal association.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.DISCRIMINATION
    supported_outcome_types: ClassVar[list[OutcomeType]] = [OutcomeType.CONTINUOUS]

    def get_name(self) -> str:
        return "Kendall's Tau"

    def get_description(self) -> str:
        return "Kendall's tau rank correlation for ordinal association."

    def compute(self, input: CheckInput) -> KendallTauResult:
        primary, proxy = extract_paired_arrays(input)

        tau, p = stats.kendalltau(primary, proxy)
        return KendallTauResult(
            tau=float(tau),
            p_value=float(p),
            n_samples=len(primary),
        )

    def project(self, result: KendallTauResult) -> float:
        return min(1.0, max(0.0, result.tau))


# --- Binary probability outcome: ROC / PR AUC ---


@dataclass
class ROCAUCResult:
    roc_auc: float
    n_samples: int


class ROCAUCCheck(CheckBase[ROCAUCResult, CheckInput]):
    """ROC AUC: area under the receiver operating characteristic curve.

    Requires binary primary (Y ∈ {0,1}) and probability proxy (Ŷ ∈ [0,1]).
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.DISCRIMINATION
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.BINARY_PROBABILITY,
    ]

    def get_name(self) -> str:
        return "ROC AUC"

    def get_description(self) -> str:
        return "Area under the receiver operating characteristic curve."

    def get_default_threshold(self) -> float:
        return 0.5

    def compute(self, input: CheckInput) -> ROCAUCResult:
        primary, proxy, weights = extract_paired_arrays_with_weights(input)

        auc = float(roc_auc_score(primary, proxy, sample_weight=weights))
        return ROCAUCResult(roc_auc=auc, n_samples=len(primary))

    def project(self, result: ROCAUCResult) -> float:
        return result.roc_auc


@dataclass
class PrecisionRecallAUCResult:
    pr_auc: float
    n_samples: int


class PrecisionRecallAUCCheck(CheckBase[PrecisionRecallAUCResult, CheckInput]):
    """Precision-Recall AUC for binary classification with probability proxy."""

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.DISCRIMINATION
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.BINARY_PROBABILITY,
    ]

    def get_name(self) -> str:
        return "Precision-Recall AUC"

    def get_description(self) -> str:
        return "Area under the precision-recall curve."

    def get_default_threshold(self) -> float:
        return 0.5

    def compute(self, input: CheckInput) -> PrecisionRecallAUCResult:
        from sklearn.metrics import auc, precision_recall_curve

        primary, proxy, weights = extract_paired_arrays_with_weights(input)

        precision_arr, recall_arr, _ = precision_recall_curve(
            primary, proxy, sample_weight=weights
        )
        pr_auc = float(auc(recall_arr, precision_arr))
        return PrecisionRecallAUCResult(pr_auc=pr_auc, n_samples=len(primary))

    def project(self, result: PrecisionRecallAUCResult) -> float:
        return result.pr_auc


# --- Binary-binary outcome: classification metrics ---


@dataclass
class ClassificationPrecisionResult:
    precision: float
    n_samples: int


class ClassificationPrecisionCheck(
    CheckBase[ClassificationPrecisionResult, CheckInput]
):
    """Classification precision: TP / (TP + FP)."""

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.DISCRIMINATION
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.BINARY_BINARY,
    ]

    def get_name(self) -> str:
        return "Precision"

    def get_description(self) -> str:
        return "Classification precision: TP / (TP + FP)."

    def get_default_threshold(self) -> float:
        return 0.7

    def compute(self, input: CheckInput) -> ClassificationPrecisionResult:
        primary, proxy = extract_paired_arrays(input)

        prec = float(precision_score(primary, proxy, zero_division=0))
        return ClassificationPrecisionResult(precision=prec, n_samples=len(primary))

    def project(self, result: ClassificationPrecisionResult) -> float:
        return min(1.0, max(0.0, result.precision))


@dataclass
class ClassificationAccuracyResult:
    accuracy: float
    n_samples: int


class ClassificationAccuracyCheck(CheckBase[ClassificationAccuracyResult, CheckInput]):
    """Classification accuracy: (TP + TN) / N."""

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.DISCRIMINATION
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.BINARY_BINARY,
    ]

    def get_name(self) -> str:
        return "Accuracy"

    def get_description(self) -> str:
        return "Classification accuracy: (TP + TN) / N."

    def get_default_threshold(self) -> float:
        return 0.7

    def compute(self, input: CheckInput) -> ClassificationAccuracyResult:
        primary, proxy, weights = extract_paired_arrays_with_weights(input)

        acc = float(accuracy_score(primary, proxy, sample_weight=weights))
        return ClassificationAccuracyResult(accuracy=acc, n_samples=len(primary))

    def project(self, result: ClassificationAccuracyResult) -> float:
        return min(1.0, max(0.0, result.accuracy))


@dataclass
class SensitivityResult:
    sensitivity: float
    n_samples: int


class SensitivityCheck(CheckBase[SensitivityResult, CheckInput]):
    """Sensitivity (recall): TP / (TP + FN)."""

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.DISCRIMINATION
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.BINARY_BINARY,
    ]

    def get_name(self) -> str:
        return "Sensitivity"

    def get_description(self) -> str:
        return "Sensitivity (recall): TP / (TP + FN)."

    def get_default_threshold(self) -> float:
        return 0.7

    def compute(self, input: CheckInput) -> SensitivityResult:
        primary, proxy, weights = extract_paired_arrays_with_weights(input)

        sens = float(
            recall_score(primary, proxy, zero_division=0, sample_weight=weights)
        )
        return SensitivityResult(sensitivity=sens, n_samples=len(primary))

    def project(self, result: SensitivityResult) -> float:
        return min(1.0, max(0.0, result.sensitivity))


@dataclass
class SpecificityResult:
    specificity: float
    n_samples: int


class SpecificityCheck(CheckBase[SpecificityResult, CheckInput]):
    """Specificity: TN / (TN + FP)."""

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.DISCRIMINATION
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.BINARY_BINARY,
    ]

    def get_name(self) -> str:
        return "Specificity"

    def get_description(self) -> str:
        return "Specificity: TN / (TN + FP)."

    def get_default_threshold(self) -> float:
        return 0.7

    def compute(self, input: CheckInput) -> SpecificityResult:
        primary, proxy, weights = extract_paired_arrays_with_weights(input)

        cm = confusion_matrix(primary, proxy, labels=[0, 1], sample_weight=weights)
        tn, fp = float(cm[0, 0]), float(cm[0, 1])
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return SpecificityResult(specificity=spec, n_samples=len(primary))

    def project(self, result: SpecificityResult) -> float:
        return min(1.0, max(0.0, result.specificity))


@dataclass
class ConfusionMatrixResult:
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    n_samples: int


class ConfusionMatrixCheck(CheckBase[ConfusionMatrixResult, CheckInput]):
    """Full confusion matrix for binary classification."""

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.DISCRIMINATION
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.BINARY_BINARY,
    ]

    def get_name(self) -> str:
        return "Confusion Matrix"

    def get_description(self) -> str:
        return "Full confusion matrix for binary classification."

    def get_default_threshold(self) -> float:
        return 0.0

    def compute(self, input: CheckInput) -> ConfusionMatrixResult:
        primary, proxy = extract_paired_arrays(input)

        cm = confusion_matrix(primary, proxy, labels=[0, 1])
        return ConfusionMatrixResult(
            true_negatives=int(cm[0, 0]),
            false_positives=int(cm[0, 1]),
            false_negatives=int(cm[1, 0]),
            true_positives=int(cm[1, 1]),
            n_samples=len(primary),
        )

    def project(self, result: ConfusionMatrixResult) -> float:
        return 1.0


@dataclass
class MCCResult:
    mcc: float
    n_samples: int


class MCCCheck(CheckBase[MCCResult, CheckInput]):
    """Matthews Correlation Coefficient for binary classification.

    MCC ranges from -1 (total disagreement) through 0 (random) to +1
    (perfect agreement). It is considered a balanced measure even when
    classes are imbalanced.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.AGREEMENT
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.BINARY_BINARY,
    ]

    def get_name(self) -> str:
        return "MCC"

    def get_description(self) -> str:
        return "Matthews Correlation Coefficient for binary classification."

    def get_default_threshold(self) -> float:
        return 0.7

    def compute(self, input: CheckInput) -> MCCResult:
        primary, proxy = extract_paired_arrays(input)

        mcc = float(matthews_corrcoef(primary, proxy))
        return MCCResult(mcc=mcc, n_samples=len(primary))

    def project(self, result: MCCResult) -> float:
        # MCC ranges [-1, 1]; map to [0, 1] via (mcc + 1) / 2
        return min(1.0, max(0.0, (result.mcc + 1.0) / 2.0))


@dataclass
class NPVResult:
    npv: float
    n_samples: int


class NPVCheck(CheckBase[NPVResult, CheckInput]):
    """Negative Predictive Value: TN / (TN + FN)."""

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.DISCRIMINATION
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.BINARY_BINARY,
    ]

    def get_name(self) -> str:
        return "NPV"

    def get_description(self) -> str:
        return "Negative Predictive Value: TN / (TN + FN)."

    def get_default_threshold(self) -> float:
        return 0.7

    def compute(self, input: CheckInput) -> NPVResult:
        primary, proxy = extract_paired_arrays(input)

        cm = confusion_matrix(primary, proxy, labels=[0, 1])
        tn, fn = int(cm[0, 0]), int(cm[1, 0])
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        return NPVResult(npv=npv, n_samples=len(primary))

    def project(self, result: NPVResult) -> float:
        return min(1.0, max(0.0, result.npv))
