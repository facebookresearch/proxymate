# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from proxymate.checks.base import CheckBase
from proxymate.types.enums import OutcomeType, RepresentativityProperty, ValidationLevel
from proxymate.types.inputs import CheckInput
from scipy import stats


def _get_target_proxy_split(
    input: CheckInput,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split CheckInput into target and proxy populations.

    Default: target = all rows, proxy = rows where proxy_col is not null.
    If population_col is set, proxy = rows where population_col is True.
    """
    target_df = input.df
    if input.population_col is not None:
        proxy_df = input.df[input.df[input.population_col].astype(bool)]
    else:
        proxy_df = input.df[input.df[input.proxy_col].notna()]
    return target_df, proxy_df


def _get_covariate_columns(
    input: CheckInput,
    target_df: pd.DataFrame,
    proxy_df: pd.DataFrame,
) -> list[str]:
    """Determine which columns to use for covariate comparison."""
    if input.covariate_cols is not None:
        return input.covariate_cols
    target_cols = set(target_df.columns)
    proxy_cols = set(proxy_df.columns)
    # Exclude primary, proxy, weight, and population columns from auto-detection
    exclude = {input.primary_col, input.proxy_col}
    if input.weight_col is not None:
        exclude.add(input.weight_col)
    if input.population_col is not None:
        exclude.add(input.population_col)
    return sorted((target_cols & proxy_cols) - exclude)


@dataclass
class CoverageResult:
    target_size: int
    proxy_size: int
    coverage_rate: float


class CoverageCheck(CheckBase[CoverageResult, CheckInput]):
    """R.1: Sample size coverage rate N_P / N_T.

    Measures what fraction of the target population is covered by the
    proxy/labeled population. Low coverage may indicate the proxy
    validation results don't generalize to the full target.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.REPRESENTATIVITY
    sub_property: ClassVar[RepresentativityProperty] = RepresentativityProperty.COVERAGE
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.CONTINUOUS,
        OutcomeType.BINARY_BINARY,
        OutcomeType.BINARY_PROBABILITY,
        OutcomeType.CATEGORICAL,
    ]

    def get_name(self) -> str:
        return "Coverage"

    def get_description(self) -> str:
        return "Sample size coverage rate of proxy population over target."

    def compute(self, input: CheckInput) -> CoverageResult:
        target_df, proxy_df = _get_target_proxy_split(input)
        target_size = len(target_df)
        proxy_size = len(proxy_df)
        coverage_rate = proxy_size / target_size if target_size > 0 else 0.0

        return CoverageResult(
            target_size=target_size,
            proxy_size=proxy_size,
            coverage_rate=coverage_rate,
        )

    def project(self, result: CoverageResult) -> float:
        return min(1.0, max(0.0, result.coverage_rate))


@dataclass
class SampleRepresentativityResult:
    covariate_stats: dict[str, dict[str, float]]
    max_smd: float
    max_ks_statistic: float
    n_covariates: int


class SampleRepresentativityCheck(CheckBase[SampleRepresentativityResult, CheckInput]):
    """R.2: Covariate distribution comparison between target and proxy populations.

    For each covariate, computes:
    - Standardized mean difference (SMD) for continuous covariates
    - KS statistic for distributional comparison

    High SMD or KS values indicate the proxy population differs
    systematically from the target, suggesting selection bias.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.REPRESENTATIVITY
    sub_property: ClassVar[RepresentativityProperty] = (
        RepresentativityProperty.SAMPLE_REPRESENTATIVITY
    )
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.CONTINUOUS,
        OutcomeType.BINARY_BINARY,
        OutcomeType.BINARY_PROBABILITY,
        OutcomeType.CATEGORICAL,
    ]

    def get_name(self) -> str:
        return "Sample Representativity"

    def get_description(self) -> str:
        return "Covariate distribution comparison between target and proxy populations."

    def get_default_threshold(self) -> float:
        return 0.9

    def compute(self, input: CheckInput) -> SampleRepresentativityResult:
        target_df, proxy_df = _get_target_proxy_split(input)
        columns = _get_covariate_columns(input, target_df, proxy_df)
        covariate_stats: dict[str, dict[str, float]] = {}
        max_smd = 0.0
        max_ks = 0.0

        for col in columns:
            target_vals = target_df[col].dropna().values
            proxy_vals = proxy_df[col].dropna().values

            if len(target_vals) == 0 or len(proxy_vals) == 0:
                continue

            # Skip non-numeric columns (e.g. categorical/string covariates)
            target_vals = np.asarray(target_vals)
            proxy_vals = np.asarray(proxy_vals)
            if not np.issubdtype(target_vals.dtype, np.number):
                try:
                    target_vals = target_vals.astype(float)
                    proxy_vals = proxy_vals.astype(float)
                except (ValueError, TypeError):
                    continue

            # Standardized mean difference
            target_mean = float(np.mean(target_vals))
            proxy_mean = float(np.mean(proxy_vals))
            pooled_std = float(
                np.sqrt((np.var(target_vals, ddof=1) + np.var(proxy_vals, ddof=1)) / 2)
            )
            smd = abs(target_mean - proxy_mean) / pooled_std if pooled_std > 0 else 0.0

            # KS statistic
            ks_result = stats.ks_2samp(target_vals, proxy_vals)
            ks_stat = float(ks_result.statistic)

            covariate_stats[col] = {
                "smd": smd,
                "ks_statistic": ks_stat,
                "ks_p_value": float(ks_result.pvalue),
                "target_mean": target_mean,
                "proxy_mean": proxy_mean,
            }

            max_smd = max(max_smd, smd)
            max_ks = max(max_ks, ks_stat)

        return SampleRepresentativityResult(
            covariate_stats=covariate_stats,
            max_smd=max_smd,
            max_ks_statistic=max_ks,
            n_covariates=len(columns),
        )

    def project(self, result: SampleRepresentativityResult) -> float:
        # SMD < 0.1 is considered well-balanced. Project as 1 - max_smd, clamped.
        return min(1.0, max(0.0, 1.0 - result.max_smd))


@dataclass
class EffectiveSampleSizeResult:
    effective_n: float
    actual_n: int
    efficiency: float


class EffectiveSampleSizeCheck(CheckBase[EffectiveSampleSizeResult, CheckInput]):
    """R.3: Effective sample size after weighting.

    Computes Kish's effective sample size: (Σw)² / Σw².
    Low efficiency (effective_n / actual_n) means the weights are
    highly variable and the weighted estimates are unstable.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.REPRESENTATIVITY
    sub_property: ClassVar[RepresentativityProperty] = (
        RepresentativityProperty.EFFECTIVE_SAMPLE_SIZE
    )
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.CONTINUOUS,
        OutcomeType.BINARY_BINARY,
        OutcomeType.BINARY_PROBABILITY,
        OutcomeType.CATEGORICAL,
    ]

    def get_name(self) -> str:
        return "Effective Sample Size"

    def get_description(self) -> str:
        return "Kish's effective sample size after weighting."

    def get_default_threshold(self) -> float:
        return 0.5

    def compute(self, input: CheckInput) -> EffectiveSampleSizeResult:
        _, proxy_df = _get_target_proxy_split(input)

        if input.weight_col is None:
            actual_n = len(proxy_df)
            return EffectiveSampleSizeResult(
                effective_n=float(actual_n),
                actual_n=actual_n,
                efficiency=1.0,
            )

        weights: npt.NDArray[np.floating] = np.array(
            proxy_df[input.weight_col].values, dtype=float
        )
        actual_n = len(weights)
        sum_w = float(np.sum(weights))
        sum_w2 = float(np.sum(weights**2))

        effective_n = (sum_w**2) / sum_w2 if sum_w2 > 0 else 0.0
        efficiency = effective_n / actual_n if actual_n > 0 else 0.0

        return EffectiveSampleSizeResult(
            effective_n=effective_n,
            actual_n=actual_n,
            efficiency=efficiency,
        )

    def project(self, result: EffectiveSampleSizeResult) -> float:
        return min(1.0, max(0.0, result.efficiency))


@dataclass
class WeightExtremeResult:
    max_weight: float
    min_weight: float
    weight_ratio: float
    cv: float
    n_weights: int


class WeightExtremeCheck(CheckBase[WeightExtremeResult, CheckInput]):
    """R.4: Weight extremity diagnostics.

    Examines the distribution of importance/sampling weights for
    extreme values. High max/min ratio or high coefficient of variation
    indicates a few observations dominate the weighted analysis.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.REPRESENTATIVITY
    sub_property: ClassVar[RepresentativityProperty] = (
        RepresentativityProperty.WEIGHT_EXTREMITY
    )
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.CONTINUOUS,
        OutcomeType.BINARY_BINARY,
        OutcomeType.BINARY_PROBABILITY,
        OutcomeType.CATEGORICAL,
    ]

    def get_name(self) -> str:
        return "Weight Extremity"

    def get_description(self) -> str:
        return "Diagnostics for extreme importance/sampling weights."

    def get_default_threshold(self) -> float:
        return 0.5

    def compute(self, input: CheckInput) -> WeightExtremeResult:
        _, proxy_df = _get_target_proxy_split(input)

        if input.weight_col is None:
            return WeightExtremeResult(
                max_weight=1.0,
                min_weight=1.0,
                weight_ratio=1.0,
                cv=0.0,
                n_weights=len(proxy_df),
            )

        weights: npt.NDArray[np.floating] = np.array(
            proxy_df[input.weight_col].values, dtype=float
        )

        if len(weights) == 0:
            return WeightExtremeResult(
                max_weight=0.0,
                min_weight=0.0,
                weight_ratio=0.0,
                cv=0.0,
                n_weights=0,
            )
        max_w = float(np.max(weights))
        min_w = float(np.min(weights))
        weight_ratio = max_w / min_w if min_w > 0 else float("inf")

        mean_w = float(np.mean(weights))
        std_w = float(np.std(weights, ddof=1)) if len(weights) > 1 else 0.0
        cv = std_w / mean_w if mean_w > 0 else 0.0

        return WeightExtremeResult(
            max_weight=max_w,
            min_weight=min_w,
            weight_ratio=weight_ratio,
            cv=cv,
            n_weights=len(weights),
        )

    def project(self, result: WeightExtremeResult) -> float:
        # Lower CV = less extreme weights = better. Use 1 / (1 + CV).
        return 1.0 / (1.0 + result.cv)


@dataclass
class WeightingEffectivenessResult:
    target_mean: float
    proxy_mean_unweighted: float
    proxy_mean_weighted: float
    gap_unweighted: float
    gap_weighted: float
    gap_reduction: float
    per_covariate_gap_reduction: dict[str, dict[str, float]]


class WeightingEffectivenessCheck(CheckBase[WeightingEffectivenessResult, CheckInput]):
    """R.5: Weighting effectiveness — does IW actually reduce the bias?

    After applying importance weights, the weighted proxy mean should
    converge toward the target mean. This check measures:

        gap_reduction = 1 - |weighted_gap| / |unweighted_gap|

    where gap = proxy_mean - target_mean on the primary outcome.

    A gap_reduction of 1.0 means perfect correction; 0.0 means no
    improvement; negative means weighting made things worse.

    Also reports per-covariate gap reduction so users can identify
    which covariates remain imbalanced after reweighting.

    Requires weight_col to be set in the input.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.REPRESENTATIVITY
    sub_property: ClassVar[RepresentativityProperty] = (
        RepresentativityProperty.WEIGHTING_EFFECTIVENESS
    )
    supported_outcome_types: ClassVar[list[OutcomeType]] = [
        OutcomeType.CONTINUOUS,
        OutcomeType.BINARY_BINARY,
        OutcomeType.BINARY_PROBABILITY,
        OutcomeType.CATEGORICAL,
    ]

    def get_name(self) -> str:
        return "Weighting Effectiveness"

    def get_description(self) -> str:
        return (
            "Validates that IW weights reduce the gap between "
            "proxy and target population means."
        )

    def get_default_threshold(self) -> float:
        return 0.5

    def compute(self, input: CheckInput) -> WeightingEffectivenessResult:
        target_df, proxy_df = _get_target_proxy_split(input)

        # Primary outcome means
        target_vals: npt.NDArray[np.floating] = np.array(
            target_df[input.primary_col].values, dtype=float
        )
        proxy_vals: npt.NDArray[np.floating] = np.array(
            proxy_df[input.primary_col].values, dtype=float
        )

        target_mean = float(np.mean(target_vals))
        proxy_mean_unweighted = float(np.mean(proxy_vals))
        gap_unweighted = abs(proxy_mean_unweighted - target_mean)

        # Weighted proxy mean
        if input.weight_col is not None and input.weight_col in proxy_df.columns:
            weights: npt.NDArray[np.floating] = np.array(
                proxy_df[input.weight_col].values, dtype=float
            )
            proxy_mean_weighted = float(np.average(proxy_vals, weights=weights))
        else:
            # No weights — weighted == unweighted
            proxy_mean_weighted = proxy_mean_unweighted

        gap_weighted = abs(proxy_mean_weighted - target_mean)
        gap_reduction = (
            1.0 - gap_weighted / gap_unweighted if gap_unweighted > 0 else 0.0
        )

        # Per-covariate gap reduction
        columns = _get_covariate_columns(input, target_df, proxy_df)
        per_cov: dict[str, dict[str, float]] = {}
        for col in columns:
            t_vals = target_df[col].dropna().values.astype(float)
            p_vals = proxy_df[col].dropna().values.astype(float)
            if len(t_vals) == 0 or len(p_vals) == 0:
                continue

            t_mean = float(np.mean(t_vals))
            p_mean = float(np.mean(p_vals))
            cov_gap_unweighted = abs(p_mean - t_mean)

            if input.weight_col is not None and input.weight_col in proxy_df.columns:
                p_col_vals = proxy_df[col].values.astype(float)
                w = np.array(proxy_df[input.weight_col].values, dtype=float)
                # Handle NaN: use non-NaN mask
                valid = ~np.isnan(p_col_vals)
                p_wmean = float(np.average(p_col_vals[valid], weights=w[valid]))
            else:
                p_wmean = p_mean

            cov_gap_weighted = abs(p_wmean - t_mean)
            cov_gap_red = (
                1.0 - cov_gap_weighted / cov_gap_unweighted
                if cov_gap_unweighted > 0
                else 0.0
            )

            per_cov[col] = {
                "target_mean": t_mean,
                "proxy_mean_unweighted": p_mean,
                "proxy_mean_weighted": p_wmean,
                "gap_reduction": cov_gap_red,
            }

        return WeightingEffectivenessResult(
            target_mean=target_mean,
            proxy_mean_unweighted=proxy_mean_unweighted,
            proxy_mean_weighted=proxy_mean_weighted,
            gap_unweighted=gap_unweighted,
            gap_weighted=gap_weighted,
            gap_reduction=gap_reduction,
            per_covariate_gap_reduction=per_cov,
        )

    def project(self, result: WeightingEffectivenessResult) -> float:
        # gap_reduction of 1.0 = perfect, 0.0 = no improvement, <0 = worse.
        return min(1.0, max(0.0, result.gap_reduction))
