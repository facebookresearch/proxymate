# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field

import pandas as pd
from proxymate.lib.base_classes.estimator_base import EstimatorBase
from proxymate.types.data import Estimate


@dataclass
class CheckInput:
    """Single (primary, proxy) pair that unit-level checks operate on.

    This is the input type for all unit-level checks. Each check receives
    exactly one primary column and one proxy column. The orchestrator
    creates these from a UnitLevelInput by iterating over proxy_cols.
    """

    df: pd.DataFrame
    primary_col: str = "primary_outcome"
    proxy_col: str = "proxy_outcome"
    weight_col: str | None = None
    covariate_cols: list[str] | None = None
    population_col: str | None = None


@dataclass
class UnitLevelInput:
    """D1: Unit-level paired data — user-facing input for validation.

    The DataFrame must contain the primary outcome column and one or more proxy
    columns. When multiple proxy_cols are provided, the validator runs each check
    independently per proxy and produces a comparison summary.

    Use ``iter_check_inputs()`` to get one ``CheckInput`` per proxy.
    """

    df: pd.DataFrame
    primary_col: str = "primary_outcome"
    proxy_cols: list[str] = field(default_factory=lambda: ["proxy_outcome"])
    weight_col: str | None = None
    covariate_cols: list[str] | None = None
    population_col: str | None = None

    def iter_check_inputs(self) -> list[tuple[str, CheckInput]]:
        """Return a list of (proxy_col_name, CheckInput) for each proxy."""
        return [
            (
                proxy_col,
                CheckInput(
                    df=self.df,
                    primary_col=self.primary_col,
                    proxy_col=proxy_col,
                    weight_col=self.weight_col,
                    covariate_cols=self.covariate_cols,
                    population_col=self.population_col,
                ),
            )
            for proxy_col in self.proxy_cols
        ]


@dataclass
class AggregateInput:
    """D2: Two aggregate estimates to compare.

    Used by estimate-level single-domain checks (e.g., AggregateBiasCheck).
    """

    primary_estimate: Estimate
    proxy_estimate: Estimate


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap-based estimate-level checks.

    Optional — when bootstrap_checks are passed to validate(), the validator
    uses the top-level ``estimator`` parameter with default n_iterations=100
    and seed=None. Pass a BootstrapConfig only to override these defaults.

    The validator runs the bootstrap loop:
      UnitLevelInput → resample N times → run estimator → N Estimates
      → feed into estimate-level checks (UnbiasednessCheck, InferentialValidityCheck)

    When multiple proxies are present (proxy_cols has > 1 entry), the validator
    generates the resampled row IDs ONCE per iteration and reuses them across
    all proxies. This ensures bootstrap variance estimates are comparable and
    proxy comparisons are not confounded by sampling noise.

    Attributes:
        n_iterations: Number of bootstrap iterations.
        confidence_level: Confidence level for bootstrap CIs.
        seed: Random seed for reproducibility.
        estimator: Overrides the top-level estimator if set.
    """

    n_iterations: int = 100
    confidence_level: float = 0.95
    seed: int | None = None
    estimator: EstimatorBase | None = None


@dataclass
class BootstrappedEstimatesInput:
    """D2 variant: Primary and proxy estimates + bootstrapped vectors.

    Used by estimate-level bootstrap-based checks (InferentialValidityCheck,
    UnbiasednessCheck). The orchestrator produces this by running the bootstrap
    loop specified by BootstrapConfig on unit-level data.

    Attributes:
        primary_estimate: True primary estimate on full (non-resampled) data.
        proxy_estimate: True proxy estimate on full (non-resampled) data.
        proxy_bootstrapped_estimates: Proxy estimates from each bootstrap iteration.
        primary_bootstrapped_estimates: Primary estimates from each bootstrap
            iteration. Present when both primary and proxy are bootstrapped
            with shared indices. Enables paired-vector checks (e.g., comparing
            primary vs proxy bootstrap distributions).
    """

    primary_estimate: Estimate
    proxy_bootstrapped_estimates: list[Estimate]
    proxy_estimate: Estimate | None = None
    primary_bootstrapped_estimates: list[Estimate] = field(default_factory=list)


@dataclass
class MetaAnalysisInput:
    """D3: Paired estimates across K domains (experiments, segments, time
    periods, etc.).

    Used by domain-level meta-analysis checks (e.g., AggregateAssociationCheck).

    When ``domain_covariances`` is provided, domain-level checks use a GLS
    regression that accounts for measurement error in both the primary and
    proxy estimates and their within-domain covariance. This follows the
    two-stage bivariate approach (Buyse & Molenberghs, 2000): per-domain
    estimates are computed in Stage 1, and the covariance is carried forward
    into the Stage 2 meta-regression.

    For a two-arm experiment with ATE estimands::

        cov(ATE_primary, ATE_proxy) = cov(Y, Y*)/n_treat + cov(Y, Y*)/n_control

    where cov(Y, Y*) is estimated from paired data within each arm.
    """

    domain_estimates: list[tuple[Estimate, Estimate]]  # [(primary_k, proxy_k)]
    domain_names: list[str] | None = None
    domain_covariances: list[float] | None = None
