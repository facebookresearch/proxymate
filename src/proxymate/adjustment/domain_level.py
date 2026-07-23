# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from proxymate.types.data import Estimate
from proxymate.types.inputs import MetaAnalysisInput


def _em_latent_normal(
    Y: npt.NDArray[np.floating],
    sigma: npt.NDArray[np.floating],
    max_iter: int = 10000,
    tol: float = 1e-12,
) -> tuple[float, float]:
    """EM for latent normal model: Y_i = delta_i + eps_i, delta_i ~ N(rho, gamma^2).

    Returns (rho, gamma^2).
    """
    Y_arr = np.asarray(Y, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    mask = ~(np.isnan(Y_arr) | np.isnan(sigma_arr))
    Y_arr, sigma_arr = Y_arr[mask], sigma_arr[mask]

    rho = float(np.mean(Y_arr))
    gamma2 = max(
        float(np.mean((Y_arr - rho) ** 2) - np.mean(sigma_arr**2)),
        1e-5,
    )

    for _ in range(max_iter):
        inv_s2 = 1.0 / sigma_arr**2
        inv_g2 = 1.0 / gamma2
        v = 1.0 / (inv_g2 + inv_s2)
        m = v * (rho * inv_g2 + Y_arr * inv_s2)
        rho_new = float(np.mean(m))
        gamma2_new = float(np.mean(v + (m - rho_new) ** 2))
        if abs(rho_new - rho) < tol and abs(gamma2_new - gamma2) < tol:
            break
        rho, gamma2 = rho_new, gamma2_new

    return rho, gamma2


@dataclass
class RegressionCalibrationResult:
    """Result of regression calibration."""

    intercept: float
    slope: float
    r_squared: float
    calibrated_estimate: Estimate


class RegressionCalibrator:
    """Regression calibration using meta-analysis data.

    Fits a linear model θ_Y = α + β · θ_Ŷ across K domains, then uses it
    to predict the primary estimate from a new proxy estimate.

    Uses precision-weighted WLS (w_k = 1/σ²_{Y,k}) when domain-level
    primary standard errors are available; falls back to OLS otherwise.
    The prediction interval incorporates both the proxy's sampling error
    and the residual cross-domain heterogeneity τ² (Proposition 3).
    """

    def __init__(self) -> None:
        self._intercept: float = 0.0
        self._slope: float = 1.0
        self._r_squared: float = 0.0
        self._tau_squared: float = 0.0
        self._fitted: bool = False

    def fit(self, meta_input: MetaAnalysisInput) -> "RegressionCalibrator":
        """Fit θ_Y = α + β · θ_Ŷ from paired domain estimates.

        When ``domain_covariances`` are available, uses iterative GLS that
        accounts for measurement error in both variables and their
        within-domain covariance.
        """
        from proxymate.checks.domain_level.meta_analysis import _meta_regression

        fit = _meta_regression(meta_input)
        self._slope = fit.slope
        self._intercept = fit.intercept
        self._tau_squared = fit.tau_squared

        primary_vals = np.array(
            [p.estimate_val for p, _ in meta_input.domain_estimates]
        )
        proxy_vals = np.array([p.estimate_val for _, p in meta_input.domain_estimates])
        predicted = self._intercept + self._slope * proxy_vals
        residuals = primary_vals - predicted

        primary_ses = np.array(
            [p.standard_error for p, _ in meta_input.domain_estimates]
        )
        has_weights = bool(np.all(primary_ses > 0))
        if has_weights:
            w: npt.NDArray[np.floating] = np.asarray(1.0 / primary_ses**2)
            sw = float(np.sum(w))
            y_bar = float(np.sum(w * primary_vals) / sw)
            ss_res = float(np.sum(w * residuals**2))
            ss_tot = float(np.sum(w * (primary_vals - y_bar) ** 2))
        else:
            ss_res = float(np.sum(residuals**2))
            ss_tot = float(np.sum((primary_vals - np.mean(primary_vals)) ** 2))

        self._r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        if fit.tau_squared == 0.0 and not has_weights:
            n = len(primary_vals)
            self._tau_squared = ss_res / (n - 2) if n > 2 else 0.0

        self._fitted = True
        return self

    def predict(self, proxy_estimate: Estimate) -> RegressionCalibrationResult:
        """Predict the primary estimate from a proxy estimate.

        The calibrated CI follows Proposition 3:
            I_cal = [γ̂₀ + γ̂₁θ̂* ± z · √(γ̂₁²σ²* + τ̂²)]
        """
        if not self._fitted:
            raise RuntimeError("RegressionCalibrator has not been fitted yet.")

        calibrated_val = self._intercept + self._slope * proxy_estimate.estimate_val

        proxy_se_contribution = self._slope**2 * proxy_estimate.standard_error**2
        total_se = float(np.sqrt(proxy_se_contribution + self._tau_squared))

        z = 1.96
        calibrated = Estimate(
            confidence_interval_level=0.95,
            estimate_val=calibrated_val,
            lower_bound=calibrated_val - z * total_se,
            upper_bound=calibrated_val + z * total_se,
            sample_size=proxy_estimate.sample_size,
            standard_error=total_se,
        )

        return RegressionCalibrationResult(
            intercept=self._intercept,
            slope=self._slope,
            r_squared=self._r_squared,
            calibrated_estimate=calibrated,
        )


@dataclass
class EstimateLevelAdjustmentResult:
    """Result of estimate-level adjustment."""

    rho: float
    gamma_squared: float
    adjusted_estimate: Estimate


class EstimateLevelAdjustment:
    """Estimate-level adjustment for residual proxy bias.

    Models the proxy-primary discrepancy across K historical domains as
    φ_k ~ N(ρ, γ²) and estimates (ρ, γ²) via EM. The adjusted interval
    for a target domain proxy estimate is:

        I*_K = [θ̂*_K - ρ̂  ±  z · √(σ²_{*,K} + γ̂²)]

    This can be layered on top of any proxy estimator (including PPI),
    adjusting for whatever residual bias remains after the initial
    correction. Requires only aggregate estimates from historical
    domains, not individual-level data.

    Reference: Wilkins-Reeves, Darmon & Sinha (2026).
    """

    def __init__(self) -> None:
        self._rho: float = 0.0
        self._gamma_squared: float = 0.0
        self._fitted: bool = False

    def fit(self, meta_input: MetaAnalysisInput) -> "EstimateLevelAdjustment":
        """Estimate (ρ, γ²) from paired domain estimates.

        Computes d_k = θ̂*_k - θ̂_k with known SE. When domain covariances
        are available, the SE accounts for the within-domain correlation::

            SE(d_k) = √(σ²_{Y,k} + σ²_{Ŷ,k} - 2ρ_k)

        Otherwise falls back to the independence assumption::

            SE(d_k) = √(σ²_{Y,k} + σ²_{Ŷ,k})
        """
        differences = np.array(
            [q.estimate_val - p.estimate_val for p, q in meta_input.domain_estimates]
        )

        if meta_input.domain_covariances is not None:
            covs = np.asarray(meta_input.domain_covariances, dtype=np.float64)
            ses = np.array(
                [
                    np.sqrt(
                        max(0.0, p.standard_error**2 + q.standard_error**2 - 2.0 * c)
                    )
                    for (p, q), c in zip(meta_input.domain_estimates, covs)
                ]
            )
        else:
            ses = np.array(
                [
                    np.sqrt(p.standard_error**2 + q.standard_error**2)
                    for p, q in meta_input.domain_estimates
                ]
            )

        self._rho, self._gamma_squared = _em_latent_normal(differences, ses)
        self._fitted = True
        return self

    def adjust(self, proxy_estimate: Estimate) -> EstimateLevelAdjustmentResult:
        """Adjust a target-domain proxy estimate.

        Subtracts the estimated mean bias ρ̂ and inflates the CI
        by γ̂² to account for cross-domain heterogeneity.
        """
        if not self._fitted:
            raise RuntimeError("EstimateLevelAdjustment has not been fitted yet.")

        adjusted_val = proxy_estimate.estimate_val - self._rho
        total_se = float(
            np.sqrt(proxy_estimate.standard_error**2 + self._gamma_squared)
        )

        z = 1.96
        adjusted = Estimate(
            confidence_interval_level=0.95,
            estimate_val=adjusted_val,
            lower_bound=adjusted_val - z * total_se,
            upper_bound=adjusted_val + z * total_se,
            sample_size=proxy_estimate.sample_size,
            standard_error=total_se,
        )

        return EstimateLevelAdjustmentResult(
            rho=self._rho,
            gamma_squared=self._gamma_squared,
            adjusted_estimate=adjusted,
        )
