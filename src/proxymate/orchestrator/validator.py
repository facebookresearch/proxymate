# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from proxymate.adjustment.domain_level import (
    EstimateLevelAdjustment,
    RegressionCalibrator,
)
from proxymate.adjustment.estimate_level import PPIEstimator
from proxymate.adjustment.unit_level import AdditiveBiasCorrector, PlattRecalibrator
from proxymate.checks.base import CheckBase
from proxymate.data.data_handler import DataHandler, sql_where_to_pandas_query
from proxymate.lib.base_classes.df_based_estimator_base import (
    DfBasedEstimatorBase,
    DfBasedEstimatorBaseInput,
)
from proxymate.lib.base_classes.estimator_base import EstimatorBase
from proxymate.lib.base_classes.transformation_base import TransformationBase
from proxymate.lib.segment import Segment, SegmentGroup
from proxymate.orchestrator.results import (
    ProxyComparison,
    ProxyValidationResult,
    SegmentValidationResult,
    ValidationReport,
)
from proxymate.types.data import CheckResult, Estimate
from proxymate.types.enums import (
    DomainLevelProperty,
    EstimateLevelProperty,
    OutcomeType,
    UnitLevelProperty,
    ValidationLevel,
)
from proxymate.types.inputs import (
    AggregateInput,
    BootstrapConfig,
    BootstrappedEstimatesInput,
    CheckInput,
    MetaAnalysisInput,
    UnitLevelInput,
)

_logger: logging.Logger = logging.getLogger(__name__)

_MIN_SEGMENT_ROWS = 2

EstimatorOrMap = EstimatorBase | dict[str, EstimatorBase]


def _resolve_estimator(
    estimator: EstimatorOrMap,
    domain_name: str,
) -> EstimatorBase:
    """Look up the estimator for a domain. Single estimator → use as-is."""
    if isinstance(estimator, dict):
        if domain_name not in estimator:
            raise ValueError(
                f"No estimator for domain '{domain_name}'. "
                f"Available: {sorted(estimator.keys())}"
            )
        return estimator[domain_name]
    return estimator


def _get_single_estimator(
    estimator: EstimatorOrMap | None,
) -> EstimatorBase | None:
    """Extract a single estimator for non-domain paths. Returns None for dicts."""
    if estimator is None or isinstance(estimator, dict):
        return None
    return estimator


@dataclass
class AdjustmentResult:
    """Result of the diagnostic-to-adjustment bridge.

    Records which adjustments were applied to address failed checks
    and which were skipped (with the reason).
    """

    applied: list[tuple[str, str, Any]] = field(
        default_factory=list
    )  # (failed_check, adjustment_name, result)
    skipped: list[tuple[str, str]] = field(
        default_factory=list
    )  # (failed_check, reason)


def _route_result(
    result: CheckResult[Any],
    proxy_result: ProxyValidationResult,
) -> None:
    """Route a check result into the appropriate sub-property bucket."""
    if result.validation_level == ValidationLevel.UNIT_LEVEL:
        prop = result.sub_property
        if not isinstance(prop, UnitLevelProperty):
            return
        proxy_result.unit_level_results.setdefault(prop, []).append(result)

    elif result.validation_level == ValidationLevel.ESTIMATE_LEVEL:
        prop = result.sub_property
        if not isinstance(prop, EstimateLevelProperty):
            return
        proxy_result.estimate_level_results.setdefault(prop, []).append(result)

    elif result.validation_level == ValidationLevel.DOMAIN_LEVEL:
        prop = result.sub_property
        if not isinstance(prop, DomainLevelProperty):
            return
        proxy_result.domain_level_results.setdefault(prop, []).append(result)

    elif result.validation_level == ValidationLevel.REPRESENTATIVITY:
        proxy_result.representativity_results.append(result)


def _build_comparison(
    per_proxy: dict[str, ProxyValidationResult],
) -> ProxyComparison | None:
    """Build a side-by-side comparison across proxies."""
    if len(per_proxy) < 2:
        return None

    ranking: dict[str, dict[str, float]] = {}
    best_proxy_per_check: dict[str, str] = {}

    for proxy_col, proxy_result in per_proxy.items():
        for results in proxy_result.unit_level_results.values():
            for r in results:
                ranking.setdefault(r.check_name, {})[proxy_col] = r.projection_result
        for results in proxy_result.estimate_level_results.values():
            for r in results:
                ranking.setdefault(r.check_name, {})[proxy_col] = r.projection_result
        for results in proxy_result.domain_level_results.values():
            for r in results:
                ranking.setdefault(r.check_name, {})[proxy_col] = r.projection_result
        for r in proxy_result.representativity_results:
            ranking.setdefault(r.check_name, {})[proxy_col] = r.projection_result

    for check_name, proxy_scores in ranking.items():
        best_proxy_per_check[check_name] = max(  # pyre-ignore[no-matching-overload]
            proxy_scores, key=proxy_scores.get
        )

    return ProxyComparison(
        ranking=ranking,
        best_proxy_per_check=best_proxy_per_check,
    )


def _compute_aggregate_input(
    df: pd.DataFrame,
    primary_col: str,
    proxy_col: str,
    estimator: EstimatorBase,
    primary_estimate: Estimate | None = None,
    data_handler: DataHandler | None = None,
    segment: Segment | None = None,
) -> tuple[AggregateInput, Estimate]:
    """Use the estimator to compute aggregate estimates from unit-level data.

    Uses a ``DfBasedEstimatorBase`` for in-memory computation.

    If ``primary_estimate`` is provided, it is reused instead of recomputed.
    Returns the primary_estimate for caching across proxy iterations.
    """
    if isinstance(estimator, DfBasedEstimatorBase):
        # Include additional columns required by the estimator (e.g. weight column)
        extra_cols = sorted(
            estimator.get_additional_metric_columns()
            | estimator.get_additional_entities_columns()
        )
        extra_cols = [c for c in extra_cols if c in df.columns]

        # Compute primary estimate once, reuse on subsequent calls
        if primary_estimate is None:
            primary_df = (
                df[[primary_col] + extra_cols]
                .rename(columns={primary_col: "metric"})
                .dropna(subset=["metric"])
            )
            primary_estimate = estimator.compute_estimator(
                DfBasedEstimatorBaseInput(df=primary_df)
            )

        # Proxy estimate
        proxy_df = (
            df[[proxy_col] + extra_cols]
            .rename(columns={proxy_col: "metric"})
            .dropna(subset=["metric"])
        )
        proxy_estimate = estimator.compute_estimator(
            DfBasedEstimatorBaseInput(df=proxy_df)
        )
    else:
        raise TypeError(
            f"estimator must be a DfBasedEstimatorBase, got {type(estimator).__name__}"
        )

    return (
        AggregateInput(
            primary_estimate=primary_estimate,
            proxy_estimate=proxy_estimate,
        ),
        primary_estimate,
    )


def _run_unit_level_checks(
    unit_level_checks: list[CheckBase[Any, CheckInput]],
    unit_level_input: UnitLevelInput,
    per_proxy: dict[str, ProxyValidationResult],
) -> None:
    """Run unit-level checks for each proxy column."""
    for proxy_col, check_input in unit_level_input.iter_check_inputs():
        proxy_result = per_proxy.setdefault(proxy_col, ProxyValidationResult())
        for check in unit_level_checks:
            result = check.run(check_input)
            _route_result(result, proxy_result)


def _run_aggregate_checks_with_estimator(
    aggregate_checks: list[CheckBase[Any, AggregateInput]],
    unit_level_input: UnitLevelInput,
    estimator: EstimatorBase,
    per_proxy: dict[str, ProxyValidationResult],
    data_handler: DataHandler | None = None,
    segment: Segment | None = None,
) -> dict[str, AggregateInput]:
    """Compute estimates from unit-level data and run aggregate checks.

    Returns the computed AggregateInput per proxy for storage on the report.
    """
    agg_inputs: dict[str, AggregateInput] = {}
    cached_primary: Estimate | None = None
    for proxy_col in unit_level_input.proxy_cols:
        agg_input, cached_primary = _compute_aggregate_input(
            df=unit_level_input.df,
            primary_col=unit_level_input.primary_col,
            proxy_col=proxy_col,
            estimator=estimator,
            primary_estimate=cached_primary,
            data_handler=data_handler,
            segment=segment,
        )
        agg_inputs[proxy_col] = agg_input
        proxy_result = per_proxy.setdefault(proxy_col, ProxyValidationResult())
        for check in aggregate_checks:
            result = check.run(agg_input)
            _route_result(result, proxy_result)
    return agg_inputs


def _run_aggregate_checks(
    aggregate_checks: list[CheckBase[Any, AggregateInput]] | None,
    unit_level_input: UnitLevelInput | None,
    aggregate_input: AggregateInput | None,
    estimator: EstimatorBase | None,
    per_proxy: dict[str, ProxyValidationResult],
    default_key: str,
    data_handler: DataHandler | None = None,
) -> dict[str, AggregateInput]:
    """Dispatch aggregate checks to the appropriate runner.

    Uses the estimator path when unit-level data and an estimator are available;
    falls back to the pre-computed ``aggregate_input`` otherwise.
    """
    if not aggregate_checks:
        return {}
    if (
        estimator is not None
        and unit_level_input is not None
        and aggregate_input is None
    ):
        return _run_aggregate_checks_with_estimator(
            aggregate_checks,
            unit_level_input,
            estimator,
            per_proxy,
            data_handler=data_handler,
        )
    if aggregate_input is not None:
        proxy_result = per_proxy.setdefault(default_key, ProxyValidationResult())
        for check in aggregate_checks:
            result = check.run(aggregate_input)
            _route_result(result, proxy_result)
    return {}


def _run_per_domain_aggregate_checks(
    aggregate_checks: list[CheckBase[Any, AggregateInput]],
    data_handler: DataHandler,
    domain_col: str,
    estimator: EstimatorOrMap,
) -> dict[str, dict[str, list[CheckResult[Any]]]]:
    """Run aggregate checks per domain when domain_col is set.

    Groups unit-level data by domain_col values, computes per-domain
    (primary, proxy) estimates using the estimator, and runs each
    aggregate check on every domain.

    Returns {proxy_col: {domain_name: [CheckResult, ...]}}.
    """
    df = data_handler.df
    primary_col = data_handler.primary_col
    proxy_cols = data_handler.proxy_cols

    domain_vals = sorted(df[domain_col].unique())
    results: dict[str, dict[str, list[CheckResult[Any]]]] = {}

    for proxy_col in proxy_cols:
        proxy_domain_results: dict[str, list[CheckResult[Any]]] = {}
        cached_primary: Estimate | None = None

        for domain_val in domain_vals:
            domain_name = str(domain_val)
            domain_est = _resolve_estimator(estimator, domain_name)

            segment = Segment(
                domain_name,
                f"{domain_col} = '{domain_val}'",
                [domain_col],
            )
            filtered = data_handler.filter_segment(segment)

            if len(filtered.df) < _MIN_SEGMENT_ROWS:
                continue

            agg_input, cached_primary = _compute_aggregate_input(
                df=filtered.df,
                primary_col=primary_col,
                proxy_col=proxy_col,
                estimator=domain_est,
                primary_estimate=None,
                data_handler=data_handler,
                segment=segment,
            )

            domain_check_results: list[CheckResult[Any]] = []
            for check in aggregate_checks:
                domain_check_results.append(check.run(agg_input))
            proxy_domain_results[domain_name] = domain_check_results

        results[proxy_col] = proxy_domain_results

    return results


def _get_proxy_estimate_from_bootstrap(
    proxy_bootstrap_input: BootstrappedEstimatesInput,
) -> Estimate:
    """Derive a point estimate for a proxy from its bootstrap distribution."""
    boot_vals = np.array(
        [e.estimate_val for e in proxy_bootstrap_input.proxy_bootstrapped_estimates]
    )
    proxy_mean = float(np.mean(boot_vals))
    proxy_se = float(np.std(boot_vals, ddof=1))
    ci_level = proxy_bootstrap_input.primary_estimate.confidence_interval_level
    alpha = 1.0 - ci_level
    lo = float(np.percentile(boot_vals, 100 * alpha / 2))
    hi = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))
    return Estimate(
        confidence_interval_level=ci_level,
        estimate_val=proxy_mean,
        lower_bound=lo,
        upper_bound=hi,
        sample_size=len(boot_vals),
        standard_error=proxy_se,
    )


def _store_proxy_aggregate_estimate(
    proxy_col: str,
    proxy_bootstrap_input: BootstrappedEstimatesInput,
    data_handler: DataHandler,
    aggregate_inputs: dict[str, AggregateInput],
) -> None:
    """Store a true or bootstrap-derived point estimate for ``proxy_col``."""
    try:
        proxy_est = data_handler.get_proxy_estimate(proxy_col)
    except (ValueError, AttributeError):
        proxy_est = _get_proxy_estimate_from_bootstrap(proxy_bootstrap_input)
    aggregate_inputs[proxy_col] = AggregateInput(
        primary_estimate=proxy_bootstrap_input.primary_estimate,
        proxy_estimate=proxy_est,
    )


def _run_bootstrap_checks_from_handler(
    bootstrap_checks: list[CheckBase[Any, BootstrappedEstimatesInput]],
    bootstrap_config: BootstrapConfig | None,
    estimator: EstimatorBase | None,
    data_handler: DataHandler,
    unit_level_input: UnitLevelInput | None,
    per_proxy: dict[str, ProxyValidationResult],
    aggregate_inputs: dict[str, AggregateInput],
) -> None:
    """Run bootstrap checks using a DataHandler (auto-bootstrap path)."""
    bootstrap_estimator = (
        bootstrap_config.estimator
        if bootstrap_config is not None and bootstrap_config.estimator is not None
        else estimator
    )
    if bootstrap_estimator is None:
        raise ValueError(
            "Bootstrap checks require an estimator. Pass estimator= to "
            "validate() or set estimator on BootstrapConfig."
        )
    seed = bootstrap_config.seed if bootstrap_config is not None else None
    n_iters = bootstrap_config.n_iterations if bootstrap_config is not None else 100

    if not data_handler.is_bootstrapped_with(seed, n_iters):
        data_handler.run_bootstrap(
            estimator=bootstrap_estimator,
            n_iterations=n_iters,
            seed=seed,
        )

    proxy_cols = (
        unit_level_input.proxy_cols
        if unit_level_input is not None
        else data_handler.proxy_cols
    )
    for proxy_col in proxy_cols:
        proxy_bootstrap_input = data_handler.get_bootstrapped_estimates_input(proxy_col)
        proxy_result = per_proxy.setdefault(proxy_col, ProxyValidationResult())
        for check in bootstrap_checks:
            result = check.run(proxy_bootstrap_input)
            _route_result(result, proxy_result)

        if proxy_col not in aggregate_inputs:
            _store_proxy_aggregate_estimate(
                proxy_col, proxy_bootstrap_input, data_handler, aggregate_inputs
            )


def _run_bootstrap_checks(
    bootstrap_checks: list[CheckBase[Any, BootstrappedEstimatesInput]],
    bootstrap_input: BootstrappedEstimatesInput | None,
    bootstrap_config: BootstrapConfig | None,
    estimator: EstimatorBase | None,
    data_handler: DataHandler | None,
    unit_level_input: UnitLevelInput | None,
    per_proxy: dict[str, ProxyValidationResult],
    default_key: str,
    aggregate_inputs: dict[str, AggregateInput],
) -> None:
    """Run bootstrap-based aggregate checks.

    If a DataHandler is available and no manual bootstrap input was given,
    the bootstrap loop is run automatically via ``_run_bootstrap_checks_from_handler``.
    Otherwise the pre-computed ``bootstrap_input`` is used directly.
    """
    if bootstrap_input is None and data_handler is not None:
        _run_bootstrap_checks_from_handler(
            bootstrap_checks,
            bootstrap_config,
            estimator,
            data_handler,
            unit_level_input,
            per_proxy,
            aggregate_inputs,
        )
    elif bootstrap_input is not None:
        proxy_result = per_proxy.setdefault(default_key, ProxyValidationResult())
        for check in bootstrap_checks:
            result = check.run(bootstrap_input)
            _route_result(result, proxy_result)


def _run_per_domain_bootstrap_checks(
    bootstrap_checks: list[CheckBase[Any, BootstrappedEstimatesInput]],
    bootstrap_config: BootstrapConfig | None,
    estimator: EstimatorOrMap,
    data_handler: DataHandler,
    domain_col: str,
) -> dict[str, dict[str, list[CheckResult[Any]]]]:
    """Run bootstrap checks per domain in Case 4.

    For each domain, filters the data, runs bootstrap with the
    domain-specific estimator, and runs all bootstrap checks.

    Returns {proxy_col: {domain_name: [CheckResult, ...]}}.
    """
    df = data_handler.df
    domain_vals = sorted(df[domain_col].dropna().unique())
    results: dict[str, dict[str, list[CheckResult[Any]]]] = {}

    seed = bootstrap_config.seed if bootstrap_config is not None else None
    n_iters = bootstrap_config.n_iterations if bootstrap_config is not None else 100

    for domain_val in domain_vals:
        domain_name = str(domain_val)
        domain_est = _resolve_estimator(estimator, domain_name)

        bootstrap_est = (
            bootstrap_config.estimator
            if bootstrap_config is not None and bootstrap_config.estimator is not None
            else domain_est
        )
        if bootstrap_est is None:
            continue

        segment = Segment(
            domain_name,
            f"{domain_col} = '{domain_val}'",
            [domain_col],
        )
        domain_handler = data_handler.filter_segment(segment)

        if len(domain_handler.df) < _MIN_SEGMENT_ROWS:
            continue

        domain_handler.run_bootstrap(
            estimator=bootstrap_est,
            n_iterations=n_iters,
            seed=seed,
        )

        for proxy_col in domain_handler.proxy_cols:
            proxy_domain_results = results.setdefault(proxy_col, {})
            boot_input = domain_handler.get_bootstrapped_estimates_input(proxy_col)

            domain_check_results: list[CheckResult[Any]] = []
            for check in bootstrap_checks:
                domain_check_results.append(check.run(boot_input))
            proxy_domain_results[domain_name] = domain_check_results

    return results


def _run_meta_analysis_checks(
    meta_analysis_checks: list[CheckBase[Any, MetaAnalysisInput]],
    meta_analysis_input: MetaAnalysisInput,
    per_proxy: dict[str, ProxyValidationResult],
    default_key: str,
) -> None:
    """Run meta-analysis checks against the pre-computed MetaAnalysisInput."""
    proxy_result = per_proxy.setdefault(default_key, ProxyValidationResult())
    for check in meta_analysis_checks:
        result = check.run(meta_analysis_input)
        _route_result(result, proxy_result)


def _run_meta_analysis_checks_per_proxy(
    meta_analysis_checks: list[CheckBase[Any, MetaAnalysisInput]],
    meta_analysis_inputs: dict[str, MetaAnalysisInput],
    per_proxy: dict[str, ProxyValidationResult],
) -> None:
    """Run meta-analysis checks for each proxy's MetaAnalysisInput."""
    for proxy_col, ma_input in meta_analysis_inputs.items():
        proxy_result = per_proxy.setdefault(proxy_col, ProxyValidationResult())
        for check in meta_analysis_checks:
            result = check.run(ma_input)
            _route_result(result, proxy_result)


def _run_representativity_checks(
    representativity_checks: list[CheckBase[Any, CheckInput]],
    unit_level_input: UnitLevelInput,
    per_proxy: dict[str, ProxyValidationResult],
) -> None:
    """Run representativity checks against the first proxy's CheckInput.

    Representativity checks operate on population-level properties, not
    per-proxy, so they run once using the first proxy's data.
    """
    check_inputs = unit_level_input.iter_check_inputs()
    if not check_inputs:
        return
    first_proxy_col, first_check_input = check_inputs[0]
    proxy_result = per_proxy.setdefault(first_proxy_col, ProxyValidationResult())
    for check in representativity_checks:
        result = check.run(first_check_input)
        _route_result(result, proxy_result)


def _resolve_bootstrap_input(
    bootstrap_input: BootstrappedEstimatesInput | None,
    data_handler: DataHandler | None,
) -> BootstrappedEstimatesInput | None:
    """Return the bootstrap input to attach to the ValidationReport.

    Uses the first proxy's bootstrap data from the handler when no manual
    input was provided and the handler has already run a bootstrap loop.
    """
    if (
        bootstrap_input is not None
        or data_handler is None
        or not data_handler.is_bootstrapped
    ):
        return bootstrap_input
    first_proxy = data_handler.proxy_cols[0] if data_handler.proxy_cols else None
    if first_proxy is None:
        return None
    return data_handler.get_bootstrapped_estimates_input(first_proxy)


def _run_domain_pipeline(
    domain_handler: DataHandler,
    domain_estimator: EstimatorBase,
    unit_level_checks: list[CheckBase[Any, CheckInput]],
    aggregate_checks: list[CheckBase[Any, AggregateInput]],
    bootstrap_checks: list[CheckBase[Any, BootstrappedEstimatesInput]],
    representativity_checks: list[CheckBase[Any, CheckInput]],
    bootstrap_config: BootstrapConfig | None,
) -> dict[str, ProxyValidationResult]:
    """Run the full Case 2 pipeline for a single domain.

    Representativity → Unit Level → Estimate Level (aggregate + bootstrap),
    returning a ProxyValidationResult per proxy column.
    """
    unit_level_input = domain_handler.to_unit_level_input()
    per_proxy: dict[str, ProxyValidationResult] = {}

    # 1. Representativity checks
    if representativity_checks:
        _run_representativity_checks(
            representativity_checks, unit_level_input, per_proxy
        )

    # 2. Unit-level checks
    if unit_level_checks:
        _run_unit_level_checks(unit_level_checks, unit_level_input, per_proxy)

    # 3. Estimate-level: aggregate checks
    aggregate_inputs: dict[str, AggregateInput] = {}
    if aggregate_checks:
        aggregate_inputs = _run_aggregate_checks_with_estimator(
            aggregate_checks,
            unit_level_input,
            domain_estimator,
            per_proxy,
            data_handler=domain_handler,
        )

    # 4. Estimate-level: bootstrap checks
    if bootstrap_checks:
        _run_bootstrap_checks(
            bootstrap_checks,
            bootstrap_input=None,
            bootstrap_config=bootstrap_config,
            estimator=domain_estimator,
            data_handler=domain_handler,
            unit_level_input=unit_level_input,
            per_proxy=per_proxy,
            default_key=(
                unit_level_input.proxy_cols[0]
                if unit_level_input.proxy_cols
                else "_aggregate"
            ),
            aggregate_inputs=aggregate_inputs,
        )

    return per_proxy


class Validator:
    """Runs validation checks and organizes results by the four-level framework.

    Accepts any combination of unit-level, estimate-level, domain-level, and
    representativity checks. Automatically groups results by level and
    sub-property. Supports multi-proxy comparison when UnitLevelInput has
    multiple proxy_cols.

    Usage::

        validator = Validator()
        report = validator.validate(
            unit_level_checks=[TTestCheck(), RMSECheck()],
            unit_level_input=unit_input,
        )
        print(report.per_proxy_results["proxy_outcome"].summary())

    With DataHandler (standardized two-table input)::

        handler = DataHandler.from_dataframes(
            metrics_df=metrics_df,
            entities_df=entities_df,
            primary_outcome_name="primary",
            proxy_outcome_names=["proxy"],
        )
        report = validator.validate(
            unit_level_checks=[TTestCheck(), RMSECheck()],
            data_handler=handler,
        )

    With segments::

        from proxymate.lib.segment import Segment, SegmentGroup
        segment_groups = [
            SegmentGroup("country", [
                Segment("US", "country = 'US'", ["country"]),
                Segment("CA", "country = 'CA'", ["country"]),
            ]),
        ]
        report = validator.validate(
            unit_level_checks=[TTestCheck()],
            data_handler=handler,
            segment_groups=segment_groups,
        )
        # report.segment_results contains per-segment results

    With estimator (auto-compute aggregate estimates)::

        from proxymate.estimators.averages_estimator import WeightedAverageEstimator
        report = validator.validate(
            unit_level_checks=[TTestCheck()],
            aggregate_checks=[AggregateBiasCheck()],
            data_handler=handler,
            estimator=WeightedAverageEstimator(),
        )
    """

    def validate(
        self,
        unit_level_checks: list[CheckBase[Any, CheckInput]] | None = None,
        aggregate_checks: list[CheckBase[Any, AggregateInput]] | None = None,
        bootstrap_checks: (
            list[CheckBase[Any, BootstrappedEstimatesInput]] | None
        ) = None,
        meta_analysis_checks: list[CheckBase[Any, MetaAnalysisInput]] | None = None,
        representativity_checks: list[CheckBase[Any, CheckInput]] | None = None,
        unit_level_input: UnitLevelInput | None = None,
        aggregate_input: AggregateInput | None = None,
        bootstrap_input: BootstrappedEstimatesInput | None = None,
        bootstrap_config: BootstrapConfig | None = None,
        meta_analysis_input: MetaAnalysisInput | None = None,
        data_handler: DataHandler | None = None,
        segment_groups: list[SegmentGroup] | None = None,
        estimator: EstimatorOrMap | None = None,
        outcome_type: OutcomeType | None = None,
        transformations: list[TransformationBase] | None = None,
        transformation_suffix: str | None = None,
        domain_col: str | None = None,
        treatment_col: str | None = None,
    ) -> ValidationReport:
        """Run all provided checks against the appropriate data.

        Returns a ValidationReport grouped by proxy, then by level/sub-property.

        Args:
            data_handler: If provided and ``unit_level_input`` is None, the
                handler's merged DataFrame is used to build a UnitLevelInput.
            segment_groups: If provided, runs unit-level (and optionally
                estimate-level) checks on each segment. Results are stored
                in ``report.segment_results``.
            estimator: If provided along with unit-level data, the
                estimator is used to compute aggregate estimates from the
                unit-level data. Aggregate checks are then run against these
                auto-computed estimates. Also used as the default estimator
                for bootstrap checks when ``data_handler`` is provided.
            bootstrap_config: Optional overrides for the bootstrap loop
                (n_iterations, seed, estimator). When omitted, bootstrap
                checks use the top-level ``estimator`` with n_iterations=100
                and seed=None.
            transformations: Optional list of transformations to apply to
                proxy columns. Creates new transformed proxy columns
                alongside the originals, enabling multi-proxy comparison
                between original and transformed data. Requires
                ``data_handler``.
            transformation_suffix: Optional custom suffix for transformed
                column names. Defaults to the concatenated transformation
                names.
            domain_col: If provided, automatically builds per-proxy
                ``MetaAnalysisInput`` by grouping unit-level data on this
                column. Requires ``data_handler`` and ``estimator``.
                Cannot be combined with a manual ``meta_analysis_input``.
            treatment_col: If provided alongside ``domain_col``, computes
                per-arm covariances for ATE estimands. The column must be
                binary (0/1 or bool). When ``None``, assumes a simple mean
                estimand for covariance computation.
        """
        # --- Apply transformations to data_handler if provided ---
        if transformations and data_handler is not None:
            data_handler = data_handler.apply_transformations(
                transformations, suffix=transformation_suffix
            )

        # --- Resolve unit_level_input from data_handler if needed ---
        if unit_level_input is None and data_handler is not None:
            unit_level_input = data_handler.to_unit_level_input()

        per_proxy: dict[str, ProxyValidationResult] = {}

        # --- Unit-level checks: one run per proxy column ---
        if unit_level_checks and unit_level_input is not None:
            _run_unit_level_checks(unit_level_checks, unit_level_input, per_proxy)

        # For aggregate/bootstrap/meta/repr checks, use a default proxy key
        # since these operate on pre-aggregated data, not per-proxy columns.
        default_key = "_aggregate"
        if unit_level_input is not None and unit_level_input.proxy_cols:
            default_key = unit_level_input.proxy_cols[0]

        single_estimator = _get_single_estimator(estimator)
        aggregate_inputs: dict[str, AggregateInput] = {}
        domain_results: dict[str, dict[str, list[CheckResult[Any]]]] = {}
        per_domain_proxy_results: dict[str, dict[str, ProxyValidationResult]] = {}

        if (
            domain_col is not None
            and data_handler is not None
            and estimator is not None
        ):
            # --- Case 4: full pipeline per domain ---
            df = data_handler.df
            domain_vals = sorted(df[domain_col].dropna().unique())

            for domain_val in domain_vals:
                domain_name = str(domain_val)
                try:
                    domain_est = _resolve_estimator(estimator, domain_name)
                except ValueError:
                    _logger.warning(
                        "No estimator for domain '%s' — skipping.", domain_name
                    )
                    continue

                segment = Segment(
                    domain_name,
                    f"{domain_col} = '{domain_val}'",
                    [domain_col],
                )
                domain_handler = data_handler.filter_segment(segment)
                if len(domain_handler.df) < _MIN_SEGMENT_ROWS:
                    continue

                domain_proxy_results = _run_domain_pipeline(
                    domain_handler=domain_handler,
                    domain_estimator=domain_est,
                    unit_level_checks=unit_level_checks or [],
                    aggregate_checks=aggregate_checks or [],
                    bootstrap_checks=bootstrap_checks or [],
                    representativity_checks=representativity_checks or [],
                    bootstrap_config=bootstrap_config,
                )
                per_domain_proxy_results[domain_name] = domain_proxy_results
        else:
            # --- Cases 1/2: single-domain checks ---
            if representativity_checks and unit_level_input is not None:
                _run_representativity_checks(
                    representativity_checks, unit_level_input, per_proxy
                )

            aggregate_inputs = _run_aggregate_checks(
                aggregate_checks,
                unit_level_input,
                aggregate_input,
                single_estimator,
                per_proxy,
                default_key,
                data_handler=data_handler,
            )
            if bootstrap_checks:
                _run_bootstrap_checks(
                    bootstrap_checks,
                    bootstrap_input,
                    bootstrap_config,
                    single_estimator,
                    data_handler,
                    unit_level_input,
                    per_proxy,
                    default_key,
                    aggregate_inputs,
                )

        # --- Meta-analysis checks ---
        meta_analysis_inputs: dict[str, MetaAnalysisInput] = {}
        if meta_analysis_checks:
            if domain_col is not None:
                # Auto-build from domain column
                if meta_analysis_input is not None:
                    raise ValueError(
                        "Cannot specify both 'domain_col' and "
                        "'meta_analysis_input'. Use one or the other."
                    )
                if data_handler is None:
                    raise ValueError(
                        "'domain_col' requires 'data_handler' to be provided."
                    )
                if estimator is None:
                    raise ValueError(
                        "'domain_col' requires 'estimator' to be provided."
                    )
                meta_analysis_inputs = data_handler.build_meta_analysis_input(
                    domain_col=domain_col,
                    estimator=estimator,
                    treatment_col=treatment_col,
                )
                _run_meta_analysis_checks_per_proxy(
                    meta_analysis_checks, meta_analysis_inputs, per_proxy
                )
            elif meta_analysis_input is not None:
                # Manual single-input path
                _run_meta_analysis_checks(
                    meta_analysis_checks, meta_analysis_input, per_proxy, default_key
                )
                meta_analysis_inputs = {default_key: meta_analysis_input}

        # --- Segment-level checks ---
        segment_results: list[SegmentValidationResult] = []
        if segment_groups and unit_level_input is not None:
            segment_results = self._run_segmented_checks(
                unit_level_checks=unit_level_checks or [],
                aggregate_checks=aggregate_checks or [],
                unit_level_input=unit_level_input,
                data_handler=data_handler,
                segment_groups=segment_groups,
                estimator=single_estimator,
            )

        return ValidationReport(
            per_proxy_results=per_proxy,
            comparison=_build_comparison(per_proxy),
            segment_results=segment_results,
            unit_level_input=unit_level_input,
            aggregate_inputs=aggregate_inputs,
            bootstrap_input=_resolve_bootstrap_input(bootstrap_input, data_handler),
            meta_analysis_inputs=meta_analysis_inputs,
            domain_results=domain_results,
            per_domain_proxy_results=per_domain_proxy_results,
            outcome_type=outcome_type,
        )

    def _run_segmented_checks(
        self,
        unit_level_checks: list[CheckBase[Any, CheckInput]],
        aggregate_checks: list[CheckBase[Any, AggregateInput]],
        unit_level_input: UnitLevelInput,
        data_handler: DataHandler | None,
        segment_groups: list[SegmentGroup],
        estimator: EstimatorBase | None,
    ) -> list[SegmentValidationResult]:
        """Run checks on each segment within each segment group."""
        results: list[SegmentValidationResult] = []

        for group in segment_groups:
            for segment in group.segments:
                # Filter data to this segment
                if data_handler is not None:
                    seg_handler = data_handler.filter_segment(segment)
                    seg_input = seg_handler.to_unit_level_input()
                else:
                    pandas_query = sql_where_to_pandas_query(segment.get_where_clause())
                    filtered_df = unit_level_input.df.query(pandas_query).reset_index(
                        drop=True
                    )
                    seg_input = UnitLevelInput(
                        df=filtered_df,
                        primary_col=unit_level_input.primary_col,
                        proxy_cols=unit_level_input.proxy_cols,
                        weight_col=unit_level_input.weight_col,
                        covariate_cols=unit_level_input.covariate_cols,
                    )

                seg_per_proxy: dict[str, ProxyValidationResult] = {}

                # Skip segments with insufficient data
                if len(seg_input.df) < _MIN_SEGMENT_ROWS:
                    _logger.warning(
                        "Segment '%s/%s' has only %d row(s) — skipping checks.",
                        group.name,
                        segment.name,
                        len(seg_input.df),
                    )
                    results.append(
                        SegmentValidationResult(
                            segment_name=segment.name,
                            group_name=group.name,
                            proxy_results=seg_per_proxy,
                        )
                    )
                    continue

                # Run unit-level checks on the segment
                if unit_level_checks:
                    _run_unit_level_checks(unit_level_checks, seg_input, seg_per_proxy)

                # Run aggregate checks with estimator on the segment
                seg_agg_inputs: dict[str, AggregateInput] = {}
                if aggregate_checks and estimator is not None:
                    seg_agg_inputs = _run_aggregate_checks_with_estimator(
                        aggregate_checks,
                        seg_input,
                        estimator,
                        seg_per_proxy,
                        data_handler=data_handler,
                        segment=segment,
                    )

                results.append(
                    SegmentValidationResult(
                        segment_name=segment.name,
                        group_name=group.name,
                        proxy_results=seg_per_proxy,
                        aggregate_inputs=seg_agg_inputs,
                    )
                )

        return results

    def adjust(self, report: ValidationReport) -> AdjustmentResult:
        """Apply adjustments for failed checks in a ValidationReport.

        Implements the diagnostic-to-adjustment bridge (Table 2 of the
        proxymate paper). Iterates over all per-proxy results, identifies
        failed checks (``thresholding_result == False``), and applies the
        corresponding adjustment when the required data is available on
        the report.

        This is a first version: adjustments are applied independently
        and checks are **not** re-run after adjustment.

        Args:
            report: A completed ``ValidationReport`` from ``validate()``.

        Returns:
            An ``AdjustmentResult`` listing which adjustments were applied
            and which were skipped.
        """
        result = AdjustmentResult()

        # Mapping from check_name to the adjustment logic.  Each entry
        # is handled explicitly so the conditions and data requirements
        # are clear.
        _CHECK_NAME_TTEST = "T-Test"
        _CHECK_NAME_ECE = "Expected Calibration Error"
        _CHECK_NAME_AGG_BIAS = "Aggregate Bias"
        _CHECK_NAME_AGG_ASSOC = "Aggregate Association"
        _CHECK_NAME_EFFECT_SENS = "Effect Sensitivity"
        _CHECK_NAME_HETEROGENEITY = "Heterogeneity"

        for proxy_col, proxy_result in report.per_proxy_results.items():
            # Build a set of passing check names for the gating rule
            # (Theorem 1): skip adjustment when diagnostics indicate
            # the proxy is already adequate.
            passing_checks: set[str] = set()
            for results_list in proxy_result.estimate_level_results.values():
                for r in results_list:
                    if r.thresholding_result:
                        passing_checks.add(r.check_name)

            # Collect all failed check results across levels
            failed_checks: list[CheckResult[Any]] = []
            for results_list in proxy_result.unit_level_results.values():
                failed_checks.extend(
                    r for r in results_list if not r.thresholding_result
                )
            for results_list in proxy_result.estimate_level_results.values():
                failed_checks.extend(
                    r for r in results_list if not r.thresholding_result
                )
            for results_list in proxy_result.domain_level_results.values():
                failed_checks.extend(
                    r for r in results_list if not r.thresholding_result
                )

            for check_result in failed_checks:
                name = check_result.check_name

                # --- T-Test (accuracy) → AdditiveBiasCorrector ---
                if name == _CHECK_NAME_TTEST:
                    if report.unit_level_input is None:
                        result.skipped.append((name, "unit-level data not available"))
                        continue
                    check_inputs = report.unit_level_input.iter_check_inputs()
                    check_input = next(
                        (ci for col, ci in check_inputs if col == proxy_col),
                        None,
                    )
                    if check_input is None:
                        result.skipped.append(
                            (name, f"no CheckInput for proxy '{proxy_col}'")
                        )
                        continue
                    try:
                        corrector = AdditiveBiasCorrector()
                        corrector.fit(check_input)
                        _logger.info(
                            "Applied AdditiveBiasCorrector for '%s' on proxy "
                            "'%s' (bias=%.6f).",
                            name,
                            proxy_col,
                            corrector.bias,
                        )
                        result.applied.append(
                            (name, "AdditiveBiasCorrector", corrector)
                        )
                    except Exception as e:
                        reason = f"AdditiveBiasCorrector failed: {e}"
                        _logger.warning("Skipped adjustment for '%s': %s", name, reason)
                        result.skipped.append((name, reason))

                # --- ECE (calibration) → PlattRecalibrator ---
                elif name == _CHECK_NAME_ECE:
                    if report.unit_level_input is None:
                        result.skipped.append((name, "unit-level data not available"))
                        continue
                    check_inputs = report.unit_level_input.iter_check_inputs()
                    check_input = next(
                        (ci for col, ci in check_inputs if col == proxy_col),
                        None,
                    )
                    if check_input is None:
                        result.skipped.append(
                            (name, f"no CheckInput for proxy '{proxy_col}'")
                        )
                        continue
                    try:
                        recalibrator = PlattRecalibrator()
                        recalibrator.fit(check_input)
                        _logger.info(
                            "Applied PlattRecalibrator for '%s' on proxy '%s' "
                            "(a=%.6f, b=%.6f).",
                            name,
                            proxy_col,
                            recalibrator.a,
                            recalibrator.b,
                        )
                        result.applied.append((name, "PlattRecalibrator", recalibrator))
                    except Exception as e:
                        reason = f"PlattRecalibrator failed: {e}"
                        _logger.warning("Skipped adjustment for '%s': %s", name, reason)
                        result.skipped.append((name, reason))

                # --- Aggregate Bias → PPIEstimator (with Theorem 1 gating) ---
                elif name == _CHECK_NAME_AGG_BIAS:
                    # Theorem 1 gating: if Inferential Validity passes,
                    # the proxy CI already covers the primary — applying
                    # PPI would add variance without reducing bias.
                    _CHECK_NAME_INF_VAL = "Inferential Validity"
                    if _CHECK_NAME_INF_VAL in passing_checks:
                        reason = (
                            "Skipped PPI: Inferential Validity passes, "
                            "proxy is already adequate (Theorem 1)"
                        )
                        _logger.info(
                            "Gating rule applied for '%s' on proxy '%s': %s",
                            name,
                            proxy_col,
                            reason,
                        )
                        result.skipped.append((name, reason))
                        continue
                    if report.unit_level_input is None:
                        result.skipped.append(
                            (name, "unit-level data not available for PPI")
                        )
                        continue
                    check_inputs = report.unit_level_input.iter_check_inputs()
                    check_input = next(
                        (ci for col, ci in check_inputs if col == proxy_col),
                        None,
                    )
                    if check_input is None:
                        result.skipped.append(
                            (name, f"no CheckInput for proxy '{proxy_col}'")
                        )
                        continue
                    # PPI needs unlabeled proxy data; use the full proxy
                    # column (including rows without primary labels) as
                    # the unlabeled set.
                    proxy_col_name = check_input.proxy_col
                    full_proxy = (
                        report.unit_level_input.df[proxy_col_name]
                        .dropna()
                        .values.astype(float)
                    )
                    try:
                        ppi = PPIEstimator()
                        ppi_result = ppi.estimate(check_input, full_proxy)
                        _logger.info(
                            "Applied PPIEstimator for '%s' on proxy '%s' "
                            "(correction=%.6f).",
                            name,
                            proxy_col,
                            ppi_result.correction,
                        )
                        result.applied.append((name, "PPIEstimator", ppi_result))
                    except Exception as e:
                        reason = f"PPIEstimator failed: {e}"
                        _logger.warning("Skipped adjustment for '%s': %s", name, reason)
                        result.skipped.append((name, reason))

                # --- Aggregate Association / Effect Sensitivity →
                #     RegressionCalibrator ---
                elif name in (_CHECK_NAME_AGG_ASSOC, _CHECK_NAME_EFFECT_SENS):
                    ma_input = report.meta_analysis_inputs.get(proxy_col)
                    if ma_input is None:
                        result.skipped.append(
                            (
                                name,
                                "MetaAnalysisInput not available for proxy "
                                f"'{proxy_col}'",
                            )
                        )
                        continue
                    try:
                        calibrator = RegressionCalibrator()
                        calibrator.fit(ma_input)
                        _logger.info(
                            "Applied RegressionCalibrator for '%s' on proxy '%s'.",
                            name,
                            proxy_col,
                        )
                        result.applied.append(
                            (name, "RegressionCalibrator", calibrator)
                        )
                    except Exception as e:
                        reason = f"RegressionCalibrator failed: {e}"
                        _logger.warning("Skipped adjustment for '%s': %s", name, reason)
                        result.skipped.append((name, reason))

                # --- Heterogeneity → EstimateLevelAdjustment ---
                elif name == _CHECK_NAME_HETEROGENEITY:
                    ma_input = report.meta_analysis_inputs.get(proxy_col)
                    if ma_input is None:
                        result.skipped.append(
                            (
                                name,
                                "MetaAnalysisInput not available for proxy "
                                f"'{proxy_col}'",
                            )
                        )
                        continue
                    try:
                        adjustment = EstimateLevelAdjustment()
                        adjustment.fit(ma_input)
                        _logger.info(
                            "Applied EstimateLevelAdjustment for '%s' on "
                            "proxy '%s' (rho=%.6f, gamma²=%.6f).",
                            name,
                            proxy_col,
                            adjustment._rho,
                            adjustment._gamma_squared,
                        )
                        result.applied.append(
                            (name, "EstimateLevelAdjustment", adjustment)
                        )
                    except Exception as e:
                        reason = f"EstimateLevelAdjustment failed: {e}"
                        _logger.warning("Skipped adjustment for '%s': %s", name, reason)
                        result.skipped.append((name, reason))

        _logger.info(
            "Adjustment complete: %d applied, %d skipped.",
            len(result.applied),
            len(result.skipped),
        )
        return result
