# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import ClassVar
from unittest import TestCase

import numpy as np
import pandas as pd
from proxymate.checks.base import CheckBase
from proxymate.lib.base_classes.data_types import Estimate
from proxymate.lib.base_classes.df_based_estimator_base import DfBasedEstimatorBase
from proxymate.lib.segment import Segment, SegmentGroup
from proxymate.orchestrator.validator import (
    _build_comparison,
    _compute_aggregate_input,
    _route_result,
    Validator,
)
from proxymate.types.data import CheckResult
from proxymate.types.enums import (
    EstimateLevelProperty,
    OutcomeType,
    UnitLevelProperty,
    ValidationLevel,
)
from proxymate.types.inputs import AggregateInput, CheckInput, UnitLevelInput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _SimpleResult:
    value: float


class _UnitLevelCheck(CheckBase[_SimpleResult, CheckInput]):
    """Minimal unit-level check that returns correlation as projection."""

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.ACCURACY
    supported_outcome_types: ClassVar[list[OutcomeType]] = [OutcomeType.CONTINUOUS]

    def get_name(self) -> str:
        return "Unit Accuracy Check"

    def get_description(self) -> str:
        return "Test unit-level accuracy check."

    def compute(self, input: CheckInput) -> _SimpleResult:
        corr = input.df[[input.primary_col, input.proxy_col]].corr().iloc[0, 1]
        return _SimpleResult(value=float(corr))

    def project(self, result: _SimpleResult) -> float:
        return max(0.0, result.value)


class _AggregateBiasCheck(CheckBase[_SimpleResult, AggregateInput]):
    """Minimal estimate-level check that measures bias."""

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.ESTIMATE_LEVEL
    sub_property: ClassVar[EstimateLevelProperty] = EstimateLevelProperty.UNBIASEDNESS
    supported_outcome_types: ClassVar[list[OutcomeType]] = [OutcomeType.CONTINUOUS]

    def get_name(self) -> str:
        return "Aggregate Bias Check"

    def get_description(self) -> str:
        return "Test aggregate bias check."

    def compute(self, input: AggregateInput) -> _SimpleResult:
        bias = abs(
            input.primary_estimate.estimate_val - input.proxy_estimate.estimate_val
        )
        return _SimpleResult(value=bias)

    def project(self, result: _SimpleResult) -> float:
        return max(0.0, 1.0 - result.value)


class _MeanEstimator(DfBasedEstimatorBase):
    """Simple mean estimator for testing."""

    def get_name(self) -> str:
        return "MeanEstimator"

    def _compute_estimator(self, input_df: pd.DataFrame) -> Estimate:
        values = input_df["metric"].dropna()
        mean_val = float(values.mean())
        se = (
            float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        )
        return Estimate(
            confidence_interval_level=0.95,
            estimate_val=mean_val,
            lower_bound=mean_val - 1.96 * se,
            upper_bound=mean_val + 1.96 * se,
            sample_size=len(values),
            standard_error=se,
        )


def _make_check_result(
    check_name: str,
    validation_level: ValidationLevel,
    sub_property: UnitLevelProperty | EstimateLevelProperty,
    projection: float,
    threshold: float = 0.8,
) -> CheckResult[_SimpleResult]:
    return CheckResult(
        check_name=check_name,
        description="test",
        validation_level=validation_level,
        sub_property=sub_property,
        check_result=_SimpleResult(value=projection),
        projection_result=projection,
        threshold=threshold,
        thresholding_result=(projection >= threshold),
    )


def _make_unit_level_input(
    n: int = 50,
    noise: float = 0.1,
    seed: int = 42,
) -> UnitLevelInput:
    rng = np.random.default_rng(seed)
    primary = rng.normal(0, 1, n)
    proxy = primary + rng.normal(0, noise, n)
    df = pd.DataFrame(
        {
            "primary_outcome": primary,
            "proxy_outcome": proxy,
        }
    )
    return UnitLevelInput(df=df)


def _make_multi_proxy_input(
    n: int = 50,
    seed: int = 42,
) -> UnitLevelInput:
    rng = np.random.default_rng(seed)
    primary = rng.normal(0, 1, n)
    proxy_a = primary + rng.normal(0, 0.1, n)  # Good proxy
    proxy_b = primary + rng.normal(0, 2.0, n)  # Noisy proxy
    df = pd.DataFrame(
        {
            "primary_outcome": primary,
            "proxy_a": proxy_a,
            "proxy_b": proxy_b,
        }
    )
    return UnitLevelInput(df=df, proxy_cols=["proxy_a", "proxy_b"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class RouteResultTest(TestCase):
    def test_unit_level_result_routed_correctly(self) -> None:
        from proxymate.orchestrator.results import ProxyValidationResult

        result = _make_check_result(
            "test", ValidationLevel.UNIT_LEVEL, UnitLevelProperty.ACCURACY, 0.9
        )
        proxy_result = ProxyValidationResult()
        _route_result(result, proxy_result)
        self.assertIn(UnitLevelProperty.ACCURACY, proxy_result.unit_level_results)
        self.assertEqual(
            len(proxy_result.unit_level_results[UnitLevelProperty.ACCURACY]), 1
        )

    def test_estimate_level_result_routed_correctly(self) -> None:
        from proxymate.orchestrator.results import ProxyValidationResult

        result = _make_check_result(
            "test",
            ValidationLevel.ESTIMATE_LEVEL,
            EstimateLevelProperty.UNBIASEDNESS,
            0.85,
        )
        proxy_result = ProxyValidationResult()
        _route_result(result, proxy_result)
        self.assertIn(
            EstimateLevelProperty.UNBIASEDNESS, proxy_result.estimate_level_results
        )

    def test_multiple_results_same_property(self) -> None:
        from proxymate.orchestrator.results import ProxyValidationResult

        proxy_result = ProxyValidationResult()
        for i in range(3):
            result = _make_check_result(
                f"check_{i}",
                ValidationLevel.UNIT_LEVEL,
                UnitLevelProperty.ACCURACY,
                0.9,
            )
            _route_result(result, proxy_result)
        self.assertEqual(
            len(proxy_result.unit_level_results[UnitLevelProperty.ACCURACY]), 3
        )


class BuildComparisonTest(TestCase):
    def test_no_comparison_single_proxy(self) -> None:
        from proxymate.orchestrator.results import ProxyValidationResult

        per_proxy = {"proxy_a": ProxyValidationResult()}
        self.assertIsNone(_build_comparison(per_proxy))

    def test_comparison_two_proxies(self) -> None:
        from proxymate.orchestrator.results import ProxyValidationResult

        proxy_a = ProxyValidationResult()
        proxy_a.unit_level_results[UnitLevelProperty.ACCURACY] = [
            _make_check_result(
                "corr", ValidationLevel.UNIT_LEVEL, UnitLevelProperty.ACCURACY, 0.95
            )
        ]
        proxy_b = ProxyValidationResult()
        proxy_b.unit_level_results[UnitLevelProperty.ACCURACY] = [
            _make_check_result(
                "corr", ValidationLevel.UNIT_LEVEL, UnitLevelProperty.ACCURACY, 0.70
            )
        ]

        comparison = _build_comparison({"proxy_a": proxy_a, "proxy_b": proxy_b})
        self.assertIsNotNone(comparison)
        assert comparison is not None
        self.assertIn("corr", comparison.ranking)
        self.assertEqual(comparison.best_proxy_per_check["corr"], "proxy_a")


class ComputeAggregateInputTest(TestCase):
    def test_basic(self) -> None:
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "primary": rng.normal(5, 1, 100),
                "proxy": rng.normal(5.2, 1, 100),
            }
        )
        estimator = _MeanEstimator()
        agg, primary_est = _compute_aggregate_input(df, "primary", "proxy", estimator)
        self.assertIsInstance(agg, AggregateInput)
        self.assertIsInstance(primary_est, Estimate)
        self.assertAlmostEqual(
            agg.primary_estimate.estimate_val, df["primary"].mean(), places=5
        )
        self.assertAlmostEqual(
            agg.proxy_estimate.estimate_val, df["proxy"].mean(), places=5
        )


class ValidatorUnitLevelTest(TestCase):
    def test_single_proxy(self) -> None:
        validator = Validator()
        unit_input = _make_unit_level_input()
        report = validator.validate(
            unit_level_checks=[_UnitLevelCheck()],
            unit_level_input=unit_input,
        )
        self.assertIn("proxy_outcome", report.per_proxy_results)
        proxy_result = report.per_proxy_results["proxy_outcome"]
        self.assertIn(UnitLevelProperty.ACCURACY, proxy_result.unit_level_results)
        results = proxy_result.unit_level_results[UnitLevelProperty.ACCURACY]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].check_name, "Unit Accuracy Check")
        # High-corr data should produce high projection
        self.assertGreater(results[0].projection_result, 0.9)

    def test_multi_proxy(self) -> None:
        validator = Validator()
        unit_input = _make_multi_proxy_input()
        report = validator.validate(
            unit_level_checks=[_UnitLevelCheck()],
            unit_level_input=unit_input,
        )
        self.assertIn("proxy_a", report.per_proxy_results)
        self.assertIn("proxy_b", report.per_proxy_results)
        # proxy_a is a better proxy (less noise)
        proj_a = (
            report.per_proxy_results["proxy_a"]
            .unit_level_results[UnitLevelProperty.ACCURACY][0]
            .projection_result
        )
        proj_b = (
            report.per_proxy_results["proxy_b"]
            .unit_level_results[UnitLevelProperty.ACCURACY][0]
            .projection_result
        )
        self.assertGreater(proj_a, proj_b)

    def test_multi_proxy_comparison(self) -> None:
        validator = Validator()
        unit_input = _make_multi_proxy_input()
        report = validator.validate(
            unit_level_checks=[_UnitLevelCheck()],
            unit_level_input=unit_input,
        )
        self.assertIsNotNone(report.comparison)
        assert report.comparison is not None
        self.assertEqual(
            report.comparison.best_proxy_per_check["Unit Accuracy Check"], "proxy_a"
        )


class ValidatorAggregateWithEstimatorTest(TestCase):
    def test_aggregate_checks_with_estimator(self) -> None:
        validator = Validator()
        unit_input = _make_unit_level_input()
        estimator = _MeanEstimator()
        report = validator.validate(
            unit_level_checks=[_UnitLevelCheck()],
            aggregate_checks=[_AggregateBiasCheck()],
            unit_level_input=unit_input,
            estimator=estimator,
        )
        proxy_result = report.per_proxy_results["proxy_outcome"]
        self.assertIn(
            EstimateLevelProperty.UNBIASEDNESS,
            proxy_result.estimate_level_results,
        )
        agg_results = proxy_result.estimate_level_results[
            EstimateLevelProperty.UNBIASEDNESS
        ]
        self.assertEqual(len(agg_results), 1)
        self.assertEqual(agg_results[0].check_name, "Aggregate Bias Check")

    def test_aggregate_checks_precomputed_input(self) -> None:
        validator = Validator()
        agg_input = AggregateInput(
            primary_estimate=Estimate(
                confidence_interval_level=0.95,
                estimate_val=5.0,
                lower_bound=4.5,
                upper_bound=5.5,
                sample_size=100,
                standard_error=0.25,
            ),
            proxy_estimate=Estimate(
                confidence_interval_level=0.95,
                estimate_val=5.1,
                lower_bound=4.6,
                upper_bound=5.6,
                sample_size=100,
                standard_error=0.25,
            ),
        )
        report = validator.validate(
            aggregate_checks=[_AggregateBiasCheck()],
            aggregate_input=agg_input,
        )
        # With no unit-level input, results go under default "_aggregate" key
        self.assertTrue(len(report.per_proxy_results) > 0)
        first_key = list(report.per_proxy_results.keys())[0]
        proxy_result = report.per_proxy_results[first_key]
        self.assertIn(
            EstimateLevelProperty.UNBIASEDNESS,
            proxy_result.estimate_level_results,
        )


class ValidatorSegmentedTest(TestCase):
    def test_segmented_validation(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        primary = rng.normal(0, 1, n)
        proxy = primary + rng.normal(0, 0.1, n)
        country = np.array(["US"] * 60 + ["CA"] * 40)
        df = pd.DataFrame(
            {
                "primary_outcome": primary,
                "proxy_outcome": proxy,
                "country": country,
            }
        )
        unit_input = UnitLevelInput(df=df)

        segments = [
            Segment("us", "country = 'US'", ["country"]),
            Segment("ca", "country = 'CA'", ["country"]),
        ]
        group = SegmentGroup("country", segments)

        validator = Validator()
        report = validator.validate(
            unit_level_checks=[_UnitLevelCheck()],
            unit_level_input=unit_input,
            segment_groups=[group],
        )
        self.assertEqual(len(report.segment_results), 2)
        self.assertEqual(report.segment_results[0].segment_name, "us")
        self.assertEqual(report.segment_results[0].group_name, "country")
        self.assertEqual(report.segment_results[1].segment_name, "ca")

        # Each segment should have results for the proxy
        for seg_result in report.segment_results:
            self.assertIn("proxy_outcome", seg_result.proxy_results)

    def test_segment_too_small_skipped(self) -> None:
        """Segments with < 2 rows should be skipped gracefully."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "primary_outcome": rng.normal(0, 1, 10),
                "proxy_outcome": rng.normal(0, 1, 10),
                "group": ["big"] * 9 + ["tiny"],
            }
        )
        unit_input = UnitLevelInput(df=df)

        segments = [
            Segment("big", "group = 'big'", ["group"]),
            Segment("tiny", "group = 'tiny'", ["group"]),
        ]
        group = SegmentGroup("size", segments)

        validator = Validator()
        report = validator.validate(
            unit_level_checks=[_UnitLevelCheck()],
            unit_level_input=unit_input,
            segment_groups=[group],
        )
        self.assertEqual(len(report.segment_results), 2)
        # "tiny" segment has only 1 row → no checks run → empty proxy_results
        tiny_result = report.segment_results[1]
        self.assertEqual(tiny_result.segment_name, "tiny")
        self.assertEqual(len(tiny_result.proxy_results), 0)


class ValidatorEmptyChecksTest(TestCase):
    def test_no_checks_returns_empty_report(self) -> None:
        validator = Validator()
        report = validator.validate()
        self.assertEqual(len(report.per_proxy_results), 0)
        self.assertIsNone(report.comparison)
        self.assertEqual(len(report.segment_results), 0)

    def test_unit_input_stored_on_report(self) -> None:
        validator = Validator()
        unit_input = _make_unit_level_input()
        report = validator.validate(
            unit_level_checks=[_UnitLevelCheck()],
            unit_level_input=unit_input,
        )
        self.assertIs(report.unit_level_input, unit_input)
