# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from proxymate.checks.base import CheckBase
from proxymate.types.enums import DomainLevelProperty, OutcomeType, ValidationLevel
from proxymate.types.inputs import MetaAnalysisInput
from scipy import stats


def _extract_arrays(
    input: MetaAnalysisInput,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Extract primary and proxy estimate arrays from MetaAnalysisInput."""
    primary = np.array([p.estimate_val for p, _ in input.domain_estimates])
    proxy = np.array([q.estimate_val for _, q in input.domain_estimates])
    return primary, proxy


def _extract_ses(
    input: MetaAnalysisInput,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Extract primary and proxy SE arrays."""
    primary_ses = np.array([p.standard_error for p, _ in input.domain_estimates])
    proxy_ses = np.array([q.standard_error for _, q in input.domain_estimates])
    return primary_ses, proxy_ses


def _compute_weights(
    input: MetaAnalysisInput,
) -> npt.NDArray[np.floating] | None:
    """Precision weights w_k = 1/σ²_{Y,k} when primary SEs are available.

    Follows the DerSimonian-Laird convention where the primary (outcome)
    variable determines the precision weights. Returns None when any
    primary SE is zero or unavailable, signalling OLS fallback.
    """
    primary_ses = np.array([p.standard_error for p, _ in input.domain_estimates])
    if bool(np.all(primary_ses > 0)):
        return np.asarray(1.0 / primary_ses**2)
    return None


def _residual_sampling_variance(
    primary_ses: npt.NDArray[np.floating],
    proxy_ses: npt.NDArray[np.floating],
    covariances: npt.NDArray[np.floating],
    slope: float,
) -> npt.NDArray[np.floating]:
    """Sampling variance of residual r_k = ŷ_k - γ₀ - γ₁x̂_k.
    ŷ_k = θ̂_{Y,k} = the observed primary estimate in domain k
    x̂_k = θ̂*_{Y*,k} = the observed proxy estimate in domain k
    γ₀, γ₁ = intercept and slope of the calibration regression

    Var(r_k | τ²=0) = σ²_{Y,k} + γ₁²σ²_{*,k} - 2γ₁ρ_k
    """
    return np.asarray(
        primary_ses**2 + slope**2 * proxy_ses**2 - 2.0 * slope * covariances
    )


@dataclass
class _WLSResult:
    slope: float
    intercept: float
    slope_se: float
    residuals: npt.NDArray[np.floating]
    tau_squared: float = 0.0


def _wls_fit(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating] | None = None,
) -> _WLSResult:
    """Fit y = a + b*x via WLS (or OLS when weights is None)."""
    n = len(x)
    if weights is None:
        result = stats.linregress(x, y)
        residuals = y - (result.intercept + result.slope * x)
        return _WLSResult(
            slope=float(result.slope),
            intercept=float(result.intercept),
            slope_se=float(result.stderr),
            residuals=residuals,
        )

    sw = float(np.sum(weights))
    x_bar = float(np.sum(weights * x) / sw)
    y_bar = float(np.sum(weights * y) / sw)

    sxx = float(np.sum(weights * (x - x_bar) ** 2))
    sxy = float(np.sum(weights * (x - x_bar) * (y - y_bar)))

    if sxx <= 0:
        residuals = y - y_bar
        return _WLSResult(slope=0.0, intercept=y_bar, slope_se=0.0, residuals=residuals)

    slope = sxy / sxx
    intercept = y_bar - slope * x_bar
    residuals = y - (intercept + slope * x)

    df = n - 2
    if df > 0:
        sigma2 = float(np.sum(weights * residuals**2) / df)
        slope_se = float(np.sqrt(sigma2 / sxx))
    else:
        slope_se = 0.0

    return _WLSResult(
        slope=slope,
        intercept=intercept,
        slope_se=slope_se,
        residuals=residuals,
    )


def _gls_fit(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    primary_ses: npt.NDArray[np.floating],
    proxy_ses: npt.NDArray[np.floating],
    covariances: npt.NDArray[np.floating],
    max_iter: int = 50,
    tol: float = 1e-6,
) -> _WLSResult:
    """Iterative GLS fit accounting for measurement error in both variables.

    The observed residual r_k = ŷ_k - γ₀ - γ₁x̂_k has variance:

        Var(r_k) = σ²_{Y,k} + γ₁²σ²_{*,k} - 2γ₁ρ_k + τ²

    The procedure iterates: estimate (γ₀, γ₁) via WLS with current
    weights, then re-estimate τ² via DerSimonian-Laird, until convergence.
    """
    n = len(x)
    slope = 1.0
    tau2 = 0.0

    for _ in range(max_iter):
        v_k = _residual_sampling_variance(primary_ses, proxy_ses, covariances, slope)
        v_k_total = np.maximum(v_k + tau2, 1e-12)
        w = 1.0 / v_k_total

        fit = _wls_fit(x, y, w)
        residuals = fit.residuals

        df = n - 2
        if df > 0:
            q_stat = float(np.sum(w * residuals**2))
            sw = float(np.sum(w))
            c = sw - float(np.sum(w**2)) / sw
            tau2_new = max(0.0, (q_stat - df) / c) if c > 0 else 0.0
        else:
            tau2_new = 0.0

        if abs(fit.slope - slope) < tol and abs(tau2_new - tau2) < tol:
            slope = fit.slope
            tau2 = tau2_new
            break
        slope = fit.slope
        tau2 = tau2_new

    v_k = _residual_sampling_variance(primary_ses, proxy_ses, covariances, slope)
    v_k_total = np.maximum(v_k + tau2, 1e-12)
    w = 1.0 / v_k_total
    final_fit = _wls_fit(x, y, w)
    final_fit.tau_squared = tau2
    return final_fit


def _meta_regression(
    input: MetaAnalysisInput,
) -> _WLSResult:
    """Dispatcher: GLS when covariances are available, WLS/OLS otherwise."""
    primary, proxy = _extract_arrays(input)
    primary_ses, proxy_ses = _extract_ses(input)

    has_ses = bool(np.all(primary_ses > 0)) and bool(np.all(proxy_ses > 0))

    if input.domain_covariances is not None and has_ses:
        covs = np.asarray(input.domain_covariances, dtype=np.float64)
        return _gls_fit(proxy, primary, primary_ses, proxy_ses, covs)

    weights = _compute_weights(input)
    return _wls_fit(proxy, primary, weights)


@dataclass
class AggregateAssociationResult:
    r_squared: float
    slope: float
    intercept: float
    n_domains: int
    weighted: bool = False


class AggregateAssociationCheck(
    CheckBase[AggregateAssociationResult, MetaAnalysisInput]
):
    """Aggregate association: R² from regression across domains.

    Fits θ_Y_k = α + β·θ_Ŷ_k across K aggregate estimate pairs and returns R².
    Uses precision-weighted WLS (w_k = 1/σ²_{Y,k}) when domain-level primary
    standard errors are available; falls back to OLS otherwise.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.DOMAIN_LEVEL
    sub_property: ClassVar[DomainLevelProperty] = (
        DomainLevelProperty.AGGREGATE_ASSOCIATION
    )
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.CONTINUOUS,
        OutcomeType.BINARY_BINARY,
        OutcomeType.BINARY_PROBABILITY,
    ]

    def get_name(self) -> str:
        return "Aggregate Association"

    def get_description(self) -> str:
        return (
            "R² from regression of primary on proxy across domains "
            "(WLS when SEs available)."
        )

    def compute(self, input: MetaAnalysisInput) -> AggregateAssociationResult:
        if len(input.domain_estimates) < 2:
            return AggregateAssociationResult(
                r_squared=0.0,
                slope=0.0,
                intercept=0.0,
                n_domains=len(input.domain_estimates),
            )
        primary, proxy = _extract_arrays(input)
        fit = _meta_regression(input)

        weights = _compute_weights(input)
        if weights is not None:
            sw = float(np.sum(weights))
            y_bar = float(np.sum(weights * primary) / sw)
            ss_res = float(np.sum(weights * fit.residuals**2))
            ss_tot = float(np.sum(weights * (primary - y_bar) ** 2))
        else:
            ss_res = float(np.sum(fit.residuals**2))
            ss_tot = float(np.sum((primary - np.mean(primary)) ** 2))

        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return AggregateAssociationResult(
            r_squared=r_squared,
            slope=fit.slope,
            intercept=fit.intercept,
            n_domains=len(input.domain_estimates),
            weighted=weights is not None or input.domain_covariances is not None,
        )

    def project(self, result: AggregateAssociationResult) -> float:
        return min(1.0, max(0.0, result.r_squared))


@dataclass
class EffectSensitivityResult:
    slope: float
    intercept: float
    slope_se: float
    n_domains: int
    weighted: bool = False


class EffectSensitivityCheck(CheckBase[EffectSensitivityResult, MetaAnalysisInput]):
    """Effect sensitivity: slope β ≈ 1?

    Fits θ_Y_k = α + β·θ_Ŷ_k and checks whether β is close to 1.
    Uses precision-weighted WLS (w_k = 1/σ²_{Y,k}) when domain-level primary
    standard errors are available; falls back to OLS otherwise.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.DOMAIN_LEVEL
    sub_property: ClassVar[DomainLevelProperty] = DomainLevelProperty.EFFECT_SENSITIVITY
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.CONTINUOUS,
        OutcomeType.BINARY_BINARY,
        OutcomeType.BINARY_PROBABILITY,
    ]

    def get_name(self) -> str:
        return "Effect Sensitivity"

    def get_description(self) -> str:
        return (
            "Whether the regression slope of primary on proxy is close to 1 "
            "(WLS when SEs available)."
        )

    def compute(self, input: MetaAnalysisInput) -> EffectSensitivityResult:
        if len(input.domain_estimates) < 2:
            return EffectSensitivityResult(
                slope=0.0,
                intercept=0.0,
                slope_se=0.0,
                n_domains=len(input.domain_estimates),
            )
        fit = _meta_regression(input)

        return EffectSensitivityResult(
            slope=fit.slope,
            intercept=fit.intercept,
            slope_se=fit.slope_se,
            n_domains=len(input.domain_estimates),
            weighted=_compute_weights(input) is not None
            or input.domain_covariances is not None,
        )

    def project(self, result: EffectSensitivityResult) -> float:
        return min(1.0, max(0.0, 1.0 - abs(1.0 - result.slope)))


@dataclass
class DirectionalAlignmentResult:
    concordance_rate: float
    n_concordant: int
    n_domains: int


class DirectionalAlignmentCheck(
    CheckBase[DirectionalAlignmentResult, MetaAnalysisInput]
):
    """Directional alignment: sign(θ_Ŷ) = sign(θ_Y)?

    Measures the fraction of domains where the proxy and primary
    aggregate estimates have the same sign. Important for ensuring
    the proxy doesn't flip the direction of the effect.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.DOMAIN_LEVEL
    sub_property: ClassVar[DomainLevelProperty] = (
        DomainLevelProperty.DIRECTIONAL_ALIGNMENT
    )
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.CONTINUOUS,
        OutcomeType.BINARY_BINARY,
        OutcomeType.BINARY_PROBABILITY,
    ]

    def get_name(self) -> str:
        return "Directional Alignment"

    def get_description(self) -> str:
        return "Fraction of domains where proxy and primary have the same sign."

    def compute(self, input: MetaAnalysisInput) -> DirectionalAlignmentResult:
        if len(input.domain_estimates) == 0:
            return DirectionalAlignmentResult(
                concordance_rate=0.0,
                n_concordant=0,
                n_domains=0,
            )
        primary, proxy = _extract_arrays(input)
        concordant = np.sign(primary) == np.sign(proxy)
        n_concordant = int(np.sum(concordant))

        return DirectionalAlignmentResult(
            concordance_rate=n_concordant / len(input.domain_estimates),
            n_concordant=n_concordant,
            n_domains=len(input.domain_estimates),
        )

    def project(self, result: DirectionalAlignmentResult) -> float:
        return min(1.0, max(0.0, result.concordance_rate))


@dataclass
class PredictionAccuracyResult:
    loo_mse: float
    loo_mae: float
    n_domains: int
    weighted: bool = False


class PredictionAccuracyCheck(CheckBase[PredictionAccuracyResult, MetaAnalysisInput]):
    """Prediction accuracy: LOO cross-validated prediction error.

    Uses leave-one-out (LOO) cross-validation on the linear model
    θ_Y_k = α + β·θ_Ŷ_k to assess out-of-sample predictive accuracy.
    Uses precision-weighted WLS (w_k = 1/σ²_{Y,k}) when domain-level primary
    standard errors are available; falls back to OLS otherwise.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.DOMAIN_LEVEL
    sub_property: ClassVar[DomainLevelProperty] = (
        DomainLevelProperty.PREDICTION_ACCURACY
    )
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.CONTINUOUS,
        OutcomeType.BINARY_BINARY,
        OutcomeType.BINARY_PROBABILITY,
    ]

    def get_name(self) -> str:
        return "Prediction Accuracy"

    def get_description(self) -> str:
        return (
            "Leave-one-out cross-validated prediction error across domains "
            "(WLS when SEs available)."
        )

    def compute(self, input: MetaAnalysisInput) -> PredictionAccuracyResult:
        primary, proxy = _extract_arrays(input)
        n = len(primary)

        if n < 3:
            return PredictionAccuracyResult(
                loo_mse=float("inf"),
                loo_mae=float("inf"),
                n_domains=n,
            )

        primary_ses, proxy_ses = _extract_ses(input)
        has_ses = bool(np.all(primary_ses > 0)) and bool(np.all(proxy_ses > 0))
        has_covs = input.domain_covariances is not None and has_ses
        covs = (
            np.asarray(input.domain_covariances, dtype=np.float64) if has_covs else None
        )
        weights = _compute_weights(input)
        loo_errors = np.empty(n)

        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            if covs is not None:
                fold_fit = _gls_fit(
                    proxy[mask],
                    primary[mask],
                    primary_ses[mask],
                    proxy_ses[mask],
                    covs[mask],
                )
            else:
                fold_weights = weights[mask] if weights is not None else None
                fold_fit = _wls_fit(proxy[mask], primary[mask], fold_weights)
            predicted = fold_fit.intercept + fold_fit.slope * proxy[i]
            loo_errors[i] = primary[i] - predicted

        loo_mse = float(np.mean(loo_errors**2))
        loo_mae = float(np.mean(np.abs(loo_errors)))

        return PredictionAccuracyResult(
            loo_mse=loo_mse,
            loo_mae=loo_mae,
            n_domains=n,
            weighted=weights is not None or has_covs,
        )

    def project(self, result: PredictionAccuracyResult) -> float:
        return 1.0 / (1.0 + result.loo_mae)


@dataclass
class HeterogeneityResult:
    i_squared: float
    tau_squared: float
    q_statistic: float
    q_p_value: float
    n_domains: int
    weighted: bool = False


class HeterogeneityCheck(CheckBase[HeterogeneityResult, MetaAnalysisInput]):
    """Heterogeneity: is the proxy-primary relationship stable across domains?

    Computes Cochran's Q, I², and τ² on the domain-level residuals from
    the regression θ_Y_k = α + β·θ_Ŷ_k. Low heterogeneity (high I² inverted)
    means the relationship is consistent across experimental contexts.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.DOMAIN_LEVEL
    sub_property: ClassVar[DomainLevelProperty] = DomainLevelProperty.HETEROGENEITY
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.CONTINUOUS,
        OutcomeType.BINARY_BINARY,
        OutcomeType.BINARY_PROBABILITY,
    ]

    def get_name(self) -> str:
        return "Heterogeneity"

    def get_description(self) -> str:
        return "Stability of the proxy-primary relationship across domains (I², τ²)."

    def compute(self, input: MetaAnalysisInput) -> HeterogeneityResult:
        primary, proxy = _extract_arrays(input)
        n = len(primary)

        if n <= 2:
            return HeterogeneityResult(
                i_squared=0.0,
                tau_squared=0.0,
                q_statistic=0.0,
                q_p_value=1.0,
                n_domains=n,
            )

        primary_ses, proxy_ses = _extract_ses(input)
        has_ses = bool(np.all(primary_ses > 0)) and bool(np.all(proxy_ses > 0))
        use_gls = input.domain_covariances is not None and has_ses

        fit = _meta_regression(input)

        if use_gls:
            covs = np.asarray(input.domain_covariances, dtype=np.float64)
            v_k = _residual_sampling_variance(primary_ses, proxy_ses, covs, fit.slope)
            v_k_total = np.maximum(v_k + fit.tau_squared, 1e-12)
            w = 1.0 / v_k_total
            q_stat = float(np.sum(fit.residuals**2 / v_k_total))
            tau_squared = fit.tau_squared
        elif _compute_weights(input) is not None:
            weights = _compute_weights(input)
            assert weights is not None
            q_stat = float(np.sum(weights * fit.residuals**2))
            sw = float(np.sum(weights))
            c = sw - float(np.sum(weights**2)) / sw
            tau_squared = max(0.0, (q_stat - (n - 2)) / c) if c > 0 else 0.0
            w = weights
        else:
            residual_var = float(np.var(fit.residuals, ddof=2)) if n > 2 else 1.0
            q_stat = (
                float(np.sum(fit.residuals**2) / residual_var)
                if residual_var > 0
                else 0.0
            )
            tau_squared = (
                max(0.0, float(np.var(fit.residuals, ddof=2))) if n > 2 else 0.0
            )
            w = None

        df = n - 2
        q_p_value = float(1.0 - stats.chi2.cdf(q_stat, df)) if df > 0 else 1.0
        i_squared = max(0.0, (q_stat - df) / q_stat) if q_stat > 0 else 0.0

        return HeterogeneityResult(
            i_squared=i_squared,
            tau_squared=tau_squared,
            q_statistic=q_stat,
            q_p_value=q_p_value,
            n_domains=n,
            weighted=w is not None,
        )

    def project(self, result: HeterogeneityResult) -> float:
        # Lower I² = less heterogeneity = better. Project as 1 - I².
        return min(1.0, max(0.0, 1.0 - result.i_squared))


@dataclass
class AutocorrelationResult:
    durbin_watson: float
    n_domains: int
    autocorrelation_detected: bool


class ResidualAutocorrelationCheck(CheckBase[AutocorrelationResult, MetaAnalysisInput]):
    """Residual autocorrelation: are regression residuals independent?

    Computes the Durbin-Watson statistic on the residuals from the
    OLS regression θ_Y_k = α + β·θ_Ŷ_k. A DW near 2 indicates no
    autocorrelation; values near 0 or 4 indicate positive or negative
    autocorrelation respectively. Domain order is assumed to be the
    order in MetaAnalysisInput.domain_estimates (practitioners should
    sort chronologically before passing data).
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.DOMAIN_LEVEL
    sub_property: ClassVar[DomainLevelProperty] = DomainLevelProperty.HETEROGENEITY
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.CONTINUOUS,
        OutcomeType.BINARY_BINARY,
        OutcomeType.BINARY_PROBABILITY,
    ]

    def get_name(self) -> str:
        return "Residual Autocorrelation"

    def get_description(self) -> str:
        return (
            "Durbin-Watson test for independence of regression "
            "residuals across domains."
        )

    def get_default_threshold(self) -> float:
        return 0.5

    def compute(self, input: MetaAnalysisInput) -> AutocorrelationResult:
        primary, proxy = _extract_arrays(input)
        n = len(primary)

        if n < 3:
            return AutocorrelationResult(
                durbin_watson=2.0,
                n_domains=n,
                autocorrelation_detected=False,
            )

        result = stats.linregress(proxy, primary)
        residuals = primary - (result.intercept + result.slope * proxy)

        diffs = np.diff(residuals)
        ss_diff = float(np.sum(diffs**2))
        ss_res = float(np.sum(residuals**2))
        dw = ss_diff / ss_res if ss_res > 0 else 2.0

        return AutocorrelationResult(
            durbin_watson=dw,
            n_domains=n,
            autocorrelation_detected=(dw < 1.5 or dw > 2.5),
        )

    def project(self, result: AutocorrelationResult) -> float:
        return min(1.0, max(0.0, 1.0 - abs(result.durbin_watson - 2.0) / 2.0))
