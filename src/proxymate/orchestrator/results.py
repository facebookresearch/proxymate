# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from proxymate.types.data import CheckResult
from proxymate.types.enums import (
    DomainLevelProperty,
    EstimateLevelProperty,
    OutcomeType,
    UnitLevelProperty,
)
from proxymate.types.inputs import (
    AggregateInput,
    BootstrappedEstimatesInput,
    MetaAnalysisInput,
    UnitLevelInput,
)


@dataclass
class ProxyValidationResult:
    """Results for a single proxy, grouped by sub-property."""

    unit_level_results: dict[UnitLevelProperty, list[CheckResult[Any]]] = field(
        default_factory=dict
    )
    estimate_level_results: dict[EstimateLevelProperty, list[CheckResult[Any]]] = field(
        default_factory=dict
    )
    domain_level_results: dict[DomainLevelProperty, list[CheckResult[Any]]] = field(
        default_factory=dict
    )
    representativity_results: list[CheckResult[Any]] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        """Pass rates per sub-property."""
        rates: dict[str, float] = {}
        for prop, results in self.unit_level_results.items():
            if results:
                rates[prop.value] = sum(
                    1 for r in results if r.thresholding_result
                ) / len(results)
        for prop, results in self.estimate_level_results.items():
            if results:
                rates[prop.value] = sum(
                    1 for r in results if r.thresholding_result
                ) / len(results)
        for prop, results in self.domain_level_results.items():
            if results:
                rates[prop.value] = sum(
                    1 for r in results if r.thresholding_result
                ) / len(results)
        if self.representativity_results:
            rates["representativity"] = sum(
                1 for r in self.representativity_results if r.thresholding_result
            ) / len(self.representativity_results)
        return rates

    def to_dataframe(self) -> pd.DataFrame:
        """Flat table with level, sub_property, check_name, description,
        projection, pass/fail."""
        rows: list[dict[str, object]] = []
        for prop, results in self.unit_level_results.items():
            for result in results:
                rows.append(
                    {
                        "level": "unit_level",
                        "sub_property": prop.value,
                        "check_name": result.check_name,
                        "description": result.description,
                        "projection": result.projection_result,
                        "threshold": result.threshold,
                        "passed": result.thresholding_result,
                    }
                )
        for prop, results in self.estimate_level_results.items():
            for result in results:
                rows.append(
                    {
                        "level": "estimate_level",
                        "sub_property": prop.value,
                        "check_name": result.check_name,
                        "description": result.description,
                        "projection": result.projection_result,
                        "threshold": result.threshold,
                        "passed": result.thresholding_result,
                    }
                )
        for prop, results in self.domain_level_results.items():
            for result in results:
                rows.append(
                    {
                        "level": "domain_level",
                        "sub_property": prop.value,
                        "check_name": result.check_name,
                        "description": result.description,
                        "projection": result.projection_result,
                        "threshold": result.threshold,
                        "passed": result.thresholding_result,
                    }
                )
        for result in self.representativity_results:
            rows.append(
                {
                    "level": "representativity",
                    "sub_property": "representativity",
                    "check_name": result.check_name,
                    "description": result.description,
                    "projection": result.projection_result,
                    "threshold": result.threshold,
                    "passed": result.thresholding_result,
                }
            )
        return pd.DataFrame(rows)


@dataclass
class SegmentValidationResult:
    """Results for a single segment within a segment group."""

    segment_name: str
    group_name: str
    proxy_results: dict[str, ProxyValidationResult] = field(default_factory=dict)
    aggregate_inputs: dict[str, AggregateInput] = field(default_factory=dict)


@dataclass
class ProxyComparison:
    """Side-by-side comparison when multiple proxies are validated."""

    ranking: dict[str, dict[str, float]]  # check_name → {proxy_col → projection}
    best_proxy_per_check: dict[str, str]  # check_name → best proxy_col


@dataclass
class ValidationReport:
    """Results organized by the four-level framework, supporting multiple proxies.

    The input fields (unit_level_input, aggregate_inputs, bootstrap_input,
    meta_analysis_inputs) are stored after validation to support plotting. They
    may contain large DataFrames. Call strip_data() before persisting or
    sharing a report if you do not need the raw inputs.
    """

    # When single proxy: one entry keyed by proxy_col name
    # When multiple proxies: one entry per proxy + comparison
    per_proxy_results: dict[str, ProxyValidationResult] = field(default_factory=dict)
    comparison: ProxyComparison | None = None
    segment_results: list[SegmentValidationResult] = field(default_factory=list)
    # Stored inputs for plotting convenience — call strip_data() before pickling
    unit_level_input: UnitLevelInput | None = None
    aggregate_inputs: dict[str, AggregateInput] = field(default_factory=dict)
    bootstrap_input: BootstrappedEstimatesInput | None = None
    meta_analysis_inputs: dict[str, MetaAnalysisInput] = field(default_factory=dict)
    domain_results: dict[str, dict[str, list[CheckResult[Any]]]] = field(
        default_factory=dict
    )
    per_domain_proxy_results: dict[str, dict[str, ProxyValidationResult]] = field(
        default_factory=dict
    )
    outcome_type: OutcomeType | None = None

    def strip_data(self) -> None:
        """Remove stored raw input data from this report.

        The input fields are retained after validation for plotting convenience,
        but they may hold large DataFrames. Call this method before persisting
        or sharing a report to avoid writing sensitive raw data to disk or
        keeping it in memory longer than necessary.
        """
        self.unit_level_input = None
        self.aggregate_inputs = {}
        self.bootstrap_input = None
        self.meta_analysis_inputs = {}
