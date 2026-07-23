# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
from proxymate.checks.base import CheckBase
from proxymate.types.enums import EstimateLevelProperty, OutcomeType, ValidationLevel
from proxymate.types.inputs import AggregateInput, BootstrappedEstimatesInput


@dataclass
class AggregateBiasResult:
    primary_estimate: float
    proxy_estimate: float
    absolute_error: float
    relative_absolute_error: float


class AggregateBiasCheck(CheckBase[AggregateBiasResult, AggregateInput]):
    """Relative absolute error between primary and proxy aggregate estimates.

    Computes |θ_Y - θ_Ŷ| / |θ_Y| to measure how far the proxy aggregate
    estimate is from the primary. A value near 0 means the proxy recovers
    the same aggregate quantity as the primary.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.ESTIMATE_LEVEL
    sub_property: ClassVar[EstimateLevelProperty] = EstimateLevelProperty.AGGREGATE_BIAS
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.CONTINUOUS,
        OutcomeType.BINARY_BINARY,
        OutcomeType.BINARY_PROBABILITY,
    ]

    def get_name(self) -> str:
        return "Aggregate Bias"

    def get_description(self) -> str:
        return "Relative absolute error between primary and proxy aggregate estimates."

    def get_default_threshold(self) -> float:
        return 0.95

    def compute(self, input: AggregateInput) -> AggregateBiasResult:
        primary_pe = (
            float("nan")
            if pd.isna(input.primary_estimate.estimate_val)
            else float(input.primary_estimate.estimate_val)
        )
        proxy_pe = (
            float("nan")
            if pd.isna(input.proxy_estimate.estimate_val)
            else float(input.proxy_estimate.estimate_val)
        )

        absolute_error = abs(primary_pe - proxy_pe)
        relative_absolute_error = (
            absolute_error / abs(primary_pe)
            if primary_pe != 0 and not np.isnan(primary_pe)
            else float("inf")
        )

        return AggregateBiasResult(
            primary_estimate=primary_pe,
            proxy_estimate=proxy_pe,
            absolute_error=absolute_error,
            relative_absolute_error=relative_absolute_error,
        )

    def project(self, result: AggregateBiasResult) -> float:
        # Lower relative error = better. Project as 1 - RAE, clamped to [0, 1].
        return min(1.0, max(0.0, 1.0 - result.relative_absolute_error))


@dataclass
class InferentialValidityResult:
    coverage_rate: float
    primary_estimate: float
    n_bootstrap_iterations: int


class InferentialValidityCheck(
    CheckBase[InferentialValidityResult, BootstrappedEstimatesInput]
):
    """Confidence interval coverage: does CI_Ŷ cover θ_Y?

    Checks what fraction of bootstrapped proxy confidence intervals
    contain the primary point estimate. High coverage means the proxy's
    uncertainty quantification is well-calibrated relative to the primary.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.ESTIMATE_LEVEL
    sub_property: ClassVar[EstimateLevelProperty] = (
        EstimateLevelProperty.INFERENTIAL_VALIDITY
    )
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.CONTINUOUS,
        OutcomeType.BINARY_BINARY,
        OutcomeType.BINARY_PROBABILITY,
    ]

    def get_name(self) -> str:
        return "Inferential Validity"

    def get_description(self) -> str:
        return "Fraction of bootstrapped proxy CIs covering the primary estimate."

    def get_default_threshold(self) -> float:
        return 0.05

    def compute(self, input: BootstrappedEstimatesInput) -> InferentialValidityResult:
        primary_pe = input.primary_estimate.estimate_val
        bootstrap_estimates = input.proxy_bootstrapped_estimates

        if len(bootstrap_estimates) == 0:
            return InferentialValidityResult(
                coverage_rate=0.0,
                primary_estimate=primary_pe,
                n_bootstrap_iterations=0,
            )

        n_covered = sum(
            1
            for est in bootstrap_estimates
            if est.lower_bound <= primary_pe <= est.upper_bound
        )
        coverage_rate = n_covered / len(bootstrap_estimates)

        return InferentialValidityResult(
            coverage_rate=coverage_rate,
            primary_estimate=primary_pe,
            n_bootstrap_iterations=len(bootstrap_estimates),
        )

    def project(self, result: InferentialValidityResult) -> float:
        return min(1.0, max(0.0, result.coverage_rate))


@dataclass
class UnbiasednessResult:
    bootstrap_mean: float
    bootstrap_se: float
    primary_estimate: float
    normalized_bias: float
    n_bootstrap_iterations: int


class UnbiasednessCheck(CheckBase[UnbiasednessResult, BootstrappedEstimatesInput]):
    """Bootstrap unbiasedness: E[θ̂_Ŷ] = θ_Y?

    Checks whether the mean of bootstrapped proxy estimates is close
    to the primary estimate, normalized by the bootstrap standard error.
    A normalized bias near 0 means the proxy estimator is unbiased.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.ESTIMATE_LEVEL
    sub_property: ClassVar[EstimateLevelProperty] = EstimateLevelProperty.UNBIASEDNESS
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.CONTINUOUS,
        OutcomeType.BINARY_BINARY,
        OutcomeType.BINARY_PROBABILITY,
    ]

    def get_name(self) -> str:
        return "Unbiasedness"

    def get_description(self) -> str:
        return "Whether the mean of bootstrapped proxy estimates matches the primary."

    def get_default_threshold(self) -> float:
        return 0.5

    def compute(self, input: BootstrappedEstimatesInput) -> UnbiasednessResult:
        if len(input.proxy_bootstrapped_estimates) == 0:
            return UnbiasednessResult(
                bootstrap_mean=0.0,
                bootstrap_se=0.0,
                primary_estimate=input.primary_estimate.estimate_val,
                normalized_bias=float("inf"),
                n_bootstrap_iterations=0,
            )
        primary_pe = input.primary_estimate.estimate_val
        point_estimates = np.array(
            [est.estimate_val for est in input.proxy_bootstrapped_estimates]
        )

        bootstrap_mean = float(np.mean(point_estimates))
        bootstrap_se = float(np.std(point_estimates, ddof=1))

        normalized_bias = (
            abs(bootstrap_mean - primary_pe) / bootstrap_se
            if bootstrap_se > 0
            else float("inf")
        )

        return UnbiasednessResult(
            bootstrap_mean=bootstrap_mean,
            bootstrap_se=bootstrap_se,
            primary_estimate=primary_pe,
            normalized_bias=normalized_bias,
            n_bootstrap_iterations=len(input.proxy_bootstrapped_estimates),
        )

    def project(self, result: UnbiasednessResult) -> float:
        # Lower normalized bias = better. Projection from old code:
        # min(1, max(0, 0.75 - normalized_bias))
        return min(1.0, max(0.0, 0.75 - result.normalized_bias))


@dataclass
class CICoverageResult:
    """Result of CI coverage check."""

    coverage_rate: float
    n_bootstrap_iterations: int
    confidence_level: float


class CICoverageCheck(CheckBase[CICoverageResult, BootstrappedEstimatesInput]):
    """Bootstrap CI coverage: do proxy CIs capture the primary estimate?

    Computes the fraction of bootstrapped proxy confidence intervals
    that contain the primary point estimate. A well-calibrated proxy
    should have coverage close to the nominal confidence level.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.ESTIMATE_LEVEL
    sub_property: ClassVar[EstimateLevelProperty] = EstimateLevelProperty.CI_COVERAGE
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.CONTINUOUS,
        OutcomeType.BINARY_BINARY,
        OutcomeType.BINARY_PROBABILITY,
    ]

    def get_name(self) -> str:
        return "CI Coverage"

    def get_description(self) -> str:
        return (
            "Fraction of bootstrapped proxy CIs that contain "
            "the primary point estimate."
        )

    def get_default_threshold(self) -> float:
        return 0.5

    def compute(self, input: BootstrappedEstimatesInput) -> CICoverageResult:
        if len(input.proxy_bootstrapped_estimates) == 0:
            return CICoverageResult(
                coverage_rate=0.0,
                n_bootstrap_iterations=0,
                confidence_level=input.primary_estimate.confidence_interval_level,
            )
        primary_pe = input.primary_estimate.estimate_val
        covered = sum(
            1
            for est in input.proxy_bootstrapped_estimates
            if est.lower_bound <= primary_pe <= est.upper_bound
        )
        n = len(input.proxy_bootstrapped_estimates)
        return CICoverageResult(
            coverage_rate=covered / n,
            n_bootstrap_iterations=n,
            confidence_level=input.primary_estimate.confidence_interval_level,
        )

    def project(self, result: CICoverageResult) -> float:
        return result.coverage_rate


@dataclass
class CIOverlapResult:
    primary_lower: float
    primary_upper: float
    proxy_lower: float
    proxy_upper: float
    overlap_coefficient: float


class CIOverlapCheck(CheckBase[CIOverlapResult, AggregateInput]):
    """Confidence interval overlap between primary and proxy estimates.

    Computes the overlap coefficient: the length of the intersection of
    the two CIs divided by the length of the shorter CI. A value of 1.0
    means one CI is entirely contained in the other; 0.0 means no overlap.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.ESTIMATE_LEVEL
    sub_property: ClassVar[EstimateLevelProperty] = EstimateLevelProperty.CI_OVERLAP
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.CONTINUOUS,
        OutcomeType.BINARY_BINARY,
        OutcomeType.BINARY_PROBABILITY,
    ]

    def get_name(self) -> str:
        return "CI Overlap"

    def get_description(self) -> str:
        return "Overlap coefficient between primary and proxy confidence intervals."

    def get_default_threshold(self) -> float:
        return 0.5

    def compute(self, input: AggregateInput) -> CIOverlapResult:
        p_lo = input.primary_estimate.lower_bound
        p_hi = input.primary_estimate.upper_bound
        q_lo = input.proxy_estimate.lower_bound
        q_hi = input.proxy_estimate.upper_bound

        overlap_lo = max(p_lo, q_lo)
        overlap_hi = min(p_hi, q_hi)
        overlap_len = max(0.0, overlap_hi - overlap_lo)

        shorter_ci = min(p_hi - p_lo, q_hi - q_lo)
        overlap_coeff = overlap_len / shorter_ci if shorter_ci > 0 else 0.0

        return CIOverlapResult(
            primary_lower=p_lo,
            primary_upper=p_hi,
            proxy_lower=q_lo,
            proxy_upper=q_hi,
            overlap_coefficient=overlap_coeff,
        )

    def project(self, result: CIOverlapResult) -> float:
        return min(1.0, max(0.0, result.overlap_coefficient))


@dataclass
class RelativePrecisionResult:
    primary_se: float
    proxy_se: float
    se_ratio: float


class RelativePrecisionCheck(CheckBase[RelativePrecisionResult, AggregateInput]):
    """Relative precision: ratio of proxy SE to primary SE.

    A ratio near 1.0 means the proxy and primary have comparable
    precision. A ratio >> 1 means the proxy is much noisier.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.ESTIMATE_LEVEL
    sub_property: ClassVar[EstimateLevelProperty] = (
        EstimateLevelProperty.RELATIVE_PRECISION
    )
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.CONTINUOUS,
        OutcomeType.BINARY_BINARY,
        OutcomeType.BINARY_PROBABILITY,
    ]

    def get_name(self) -> str:
        return "Relative Precision"

    def get_description(self) -> str:
        return "Ratio of proxy standard error to primary standard error."

    def get_default_threshold(self) -> float:
        return 0.5

    def compute(self, input: AggregateInput) -> RelativePrecisionResult:
        primary_se = input.primary_estimate.standard_error
        proxy_se = input.proxy_estimate.standard_error

        se_ratio = proxy_se / primary_se if primary_se > 0 else float("inf")

        return RelativePrecisionResult(
            primary_se=primary_se,
            proxy_se=proxy_se,
            se_ratio=se_ratio,
        )

    def project(self, result: RelativePrecisionResult) -> float:
        # Project as 1 / (1 + |log(ratio)|) — symmetric around ratio=1
        if result.se_ratio <= 0 or np.isinf(result.se_ratio):
            return 0.0
        return 1.0 / (1.0 + abs(np.log(result.se_ratio)))


@dataclass
class SignAgreementResult:
    primary_estimate: float
    proxy_estimate: float
    signs_agree: bool


class SignAgreementCheck(CheckBase[SignAgreementResult, AggregateInput]):
    """Directional agreement: do primary and proxy have the same sign?

    Projects as 1.0 if signs agree (or both are zero), 0.0 otherwise.
    This is important when the direction of an effect matters more than
    the exact magnitude.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.ESTIMATE_LEVEL
    sub_property: ClassVar[EstimateLevelProperty] = EstimateLevelProperty.SIGN_AGREEMENT
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.CONTINUOUS,
        OutcomeType.BINARY_BINARY,
        OutcomeType.BINARY_PROBABILITY,
    ]

    def get_name(self) -> str:
        return "Sign Agreement"

    def get_description(self) -> str:
        return "Whether primary and proxy estimates have the same sign."

    def get_default_threshold(self) -> float:
        return 0.8

    def compute(self, input: AggregateInput) -> SignAgreementResult:
        p = input.primary_estimate.estimate_val
        q = input.proxy_estimate.estimate_val

        if p == 0 and q == 0:
            agree = True
        elif p == 0 or q == 0:
            agree = False
        else:
            agree = (p > 0) == (q > 0)

        return SignAgreementResult(
            primary_estimate=p,
            proxy_estimate=q,
            signs_agree=agree,
        )

    def project(self, result: SignAgreementResult) -> float:
        return 1.0 if result.signs_agree else 0.0


@dataclass
class EffectSizeAgreementResult:
    primary_in_proxy_ci: bool
    proxy_in_primary_ci: bool
    agreement_score: float


class EffectSizeAgreementCheck(CheckBase[EffectSizeAgreementResult, AggregateInput]):
    """Bidirectional CI coverage: are estimates within each other's CIs?

    Checks whether the primary estimate falls within the proxy CI and
    vice versa. Full agreement (both covered) scores 1.0, one-way
    coverage scores 0.5, no coverage scores 0.0.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.ESTIMATE_LEVEL
    sub_property: ClassVar[EstimateLevelProperty] = (
        EstimateLevelProperty.EFFECT_SIZE_AGREEMENT
    )
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.CONTINUOUS,
        OutcomeType.BINARY_BINARY,
        OutcomeType.BINARY_PROBABILITY,
    ]

    def get_name(self) -> str:
        return "Effect Size Agreement"

    def get_description(self) -> str:
        return "Bidirectional CI coverage: are estimates within each other's CIs?"

    def get_default_threshold(self) -> float:
        return 0.5

    def compute(self, input: AggregateInput) -> EffectSizeAgreementResult:
        p_val = input.primary_estimate.estimate_val
        p_lo = input.primary_estimate.lower_bound
        p_hi = input.primary_estimate.upper_bound

        q_val = input.proxy_estimate.estimate_val
        q_lo = input.proxy_estimate.lower_bound
        q_hi = input.proxy_estimate.upper_bound

        primary_in_proxy = q_lo <= p_val <= q_hi
        proxy_in_primary = p_lo <= q_val <= p_hi

        score = (float(primary_in_proxy) + float(proxy_in_primary)) / 2.0

        return EffectSizeAgreementResult(
            primary_in_proxy_ci=primary_in_proxy,
            proxy_in_primary_ci=proxy_in_primary,
            agreement_score=score,
        )

    def project(self, result: EffectSizeAgreementResult) -> float:
        return result.agreement_score
