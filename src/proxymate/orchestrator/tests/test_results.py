# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from unittest import TestCase

import pandas as pd
from proxymate.lib.base_classes.data_types import Estimate
from proxymate.orchestrator.results import (
    ProxyComparison,
    ProxyValidationResult,
    ValidationReport,
)
from proxymate.types.data import CheckResult
from proxymate.types.enums import (
    EstimateLevelProperty,
    UnitLevelProperty,
    ValidationLevel,
)
from proxymate.types.inputs import (
    AggregateInput,
    BootstrappedEstimatesInput,
    UnitLevelInput,
)


def _make_estimate() -> Estimate:
    return Estimate(
        confidence_interval_level=0.95,
        estimate_val=0.1,
        lower_bound=0.05,
        upper_bound=0.15,
        sample_size=100,
        standard_error=0.02,
    )


def _make_check_result(
    name: str,
    projection: float,
    threshold: float,
    sub_property: UnitLevelProperty = UnitLevelProperty.ACCURACY,
) -> CheckResult[float]:
    return CheckResult(
        check_name=name,
        description=f"Test {name}",
        validation_level=ValidationLevel.UNIT_LEVEL,
        sub_property=sub_property,
        check_result=projection,
        projection_result=projection,
        threshold=threshold,
        thresholding_result=projection >= threshold,
    )


class TestProxyValidationResult(TestCase):
    def test_summary_empty(self) -> None:
        result = ProxyValidationResult()
        self.assertEqual(result.summary(), {})

    def test_summary_with_results(self) -> None:
        result = ProxyValidationResult()
        result.unit_level_results[UnitLevelProperty.ACCURACY] = [
            _make_check_result("check1", 0.9, 0.8),
            _make_check_result("check2", 0.5, 0.8),
        ]
        summary = result.summary()
        # 1 pass out of 2
        self.assertAlmostEqual(summary["accuracy"], 0.5)

    def test_to_dataframe(self) -> None:
        result = ProxyValidationResult()
        result.unit_level_results[UnitLevelProperty.ACCURACY] = [
            _make_check_result("check1", 0.9, 0.8),
        ]
        df = result.to_dataframe()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["check_name"], "check1")
        self.assertEqual(df.iloc[0]["level"], "unit_level")
        self.assertTrue(df.iloc[0]["passed"])

    def test_to_dataframe_with_aggregate(self) -> None:
        result = ProxyValidationResult()
        result.unit_level_results[UnitLevelProperty.ACCURACY] = [
            _make_check_result("check1", 0.9, 0.8),
        ]
        agg_result: CheckResult[float] = CheckResult(
            check_name="agg_check",
            description="Test agg",
            validation_level=ValidationLevel.ESTIMATE_LEVEL,
            sub_property=EstimateLevelProperty.AGGREGATE_BIAS,
            check_result=0.85,
            projection_result=0.85,
            threshold=0.5,
            thresholding_result=True,
        )
        result.estimate_level_results[EstimateLevelProperty.AGGREGATE_BIAS] = [
            agg_result,
        ]
        df = result.to_dataframe()
        self.assertEqual(len(df), 2)

    def test_to_dataframe_representativity(self) -> None:
        result = ProxyValidationResult()
        result.representativity_results = [
            _make_check_result("repr_check", 0.7, 0.5),
        ]
        df = result.to_dataframe()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["level"], "representativity")


class TestValidationReport(TestCase):
    def test_empty_report(self) -> None:
        report = ValidationReport()
        self.assertEqual(len(report.per_proxy_results), 0)
        self.assertIsNone(report.comparison)

    def test_single_proxy(self) -> None:
        pvr = ProxyValidationResult()
        pvr.unit_level_results[UnitLevelProperty.ACCURACY] = [
            _make_check_result("check1", 0.9, 0.8),
        ]
        report = ValidationReport(per_proxy_results={"proxy_1": pvr})
        self.assertIn("proxy_1", report.per_proxy_results)

    def test_with_comparison(self) -> None:
        comparison = ProxyComparison(
            ranking={"check1": {"proxy_1": 0.9, "proxy_2": 0.7}},
            best_proxy_per_check={"check1": "proxy_1"},
        )
        report = ValidationReport(comparison=comparison)
        self.assertIsNotNone(report.comparison)
        self.assertEqual(report.comparison.best_proxy_per_check["check1"], "proxy_1")

    def test_strip_data_clears_inputs(self) -> None:
        est = _make_estimate()
        report = ValidationReport(
            unit_level_input=UnitLevelInput(df=pd.DataFrame()),
            aggregate_inputs={
                "proxy_1": AggregateInput(primary_estimate=est, proxy_estimate=est)
            },
            bootstrap_input=BootstrappedEstimatesInput(
                primary_estimate=est,
                proxy_bootstrapped_estimates=[est],
            ),
        )
        self.assertIsNotNone(report.unit_level_input)
        self.assertNotEqual(report.aggregate_inputs, {})
        self.assertIsNotNone(report.bootstrap_input)

        report.strip_data()

        self.assertIsNone(report.unit_level_input)
        self.assertEqual(report.aggregate_inputs, {})
        self.assertIsNone(report.bootstrap_input)
        self.assertEqual(report.meta_analysis_inputs, {})

    def test_strip_data_idempotent(self) -> None:
        report = ValidationReport()
        report.strip_data()
        self.assertIsNone(report.unit_level_input)
        self.assertEqual(report.aggregate_inputs, {})
        self.assertIsNone(report.bootstrap_input)
        self.assertEqual(report.meta_analysis_inputs, {})
