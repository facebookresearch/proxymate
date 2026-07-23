# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from unittest import TestCase

from proxymate.checks.estimate_level.single_domain import (
    AggregateBiasCheck,
    CIOverlapCheck,
    EffectSizeAgreementCheck,
    InferentialValidityCheck,
    RelativePrecisionCheck,
    SignAgreementCheck,
    UnbiasednessCheck,
)
from proxymate.lib.base_classes.data_types import Estimate
from proxymate.types.enums import EstimateLevelProperty, ValidationLevel
from proxymate.types.inputs import AggregateInput, BootstrappedEstimatesInput


def _make_estimate(
    val: float,
    se: float = 0.1,
    ci_level: float = 0.95,
    n: int = 100,
) -> Estimate:
    return Estimate(
        confidence_interval_level=ci_level,
        estimate_val=val,
        lower_bound=val - 1.96 * se,
        upper_bound=val + 1.96 * se,
        sample_size=n,
        standard_error=se,
    )


# ---- AggregateBiasCheck ----


class TestAggregateBiasCheck(TestCase):
    def test_no_bias(self) -> None:
        check = AggregateBiasCheck()
        inp = AggregateInput(
            primary_estimate=_make_estimate(10.0),
            proxy_estimate=_make_estimate(10.0),
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.absolute_error, 0.0)
        self.assertAlmostEqual(result.relative_absolute_error, 0.0)

    def test_small_bias(self) -> None:
        check = AggregateBiasCheck()
        inp = AggregateInput(
            primary_estimate=_make_estimate(10.0),
            proxy_estimate=_make_estimate(11.0),
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.absolute_error, 1.0)
        self.assertAlmostEqual(result.relative_absolute_error, 0.1)

    def test_project(self) -> None:
        check = AggregateBiasCheck()
        inp = AggregateInput(
            primary_estimate=_make_estimate(10.0),
            proxy_estimate=_make_estimate(11.0),
        )
        result = check.compute(inp)
        proj = check.project(result)
        # 1 - 0.1 = 0.9
        self.assertAlmostEqual(proj, 0.9)

    def test_metadata(self) -> None:
        self.assertEqual(
            AggregateBiasCheck.validation_level,
            ValidationLevel.ESTIMATE_LEVEL,
        )
        self.assertEqual(
            AggregateBiasCheck.sub_property,
            EstimateLevelProperty.AGGREGATE_BIAS,
        )


# ---- InferentialValidityCheck ----


class TestInferentialValidityCheck(TestCase):
    def test_full_coverage(self) -> None:
        check = InferentialValidityCheck()
        primary = _make_estimate(5.0)
        # All bootstrap CIs contain 5.0
        bootstrap = [_make_estimate(5.0 + i * 0.01, se=1.0) for i in range(10)]
        inp = BootstrappedEstimatesInput(
            primary_estimate=primary,
            proxy_bootstrapped_estimates=bootstrap,
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.coverage_rate, 1.0)

    def test_no_coverage(self) -> None:
        check = InferentialValidityCheck()
        primary = _make_estimate(100.0)
        # Bootstrap CIs around 0 won't contain 100
        bootstrap = [_make_estimate(0.0, se=0.01) for _ in range(10)]
        inp = BootstrappedEstimatesInput(
            primary_estimate=primary,
            proxy_bootstrapped_estimates=bootstrap,
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.coverage_rate, 0.0)

    def test_project(self) -> None:
        check = InferentialValidityCheck()
        primary = _make_estimate(5.0)
        bootstrap = [_make_estimate(5.0, se=1.0) for _ in range(10)]
        inp = BootstrappedEstimatesInput(
            primary_estimate=primary,
            proxy_bootstrapped_estimates=bootstrap,
        )
        result = check.compute(inp)
        self.assertAlmostEqual(check.project(result), 1.0)


# ---- UnbiasednessCheck ----


class TestUnbiasednessCheck(TestCase):
    def test_unbiased(self) -> None:
        check = UnbiasednessCheck()
        primary = _make_estimate(5.0)
        # Bootstrap estimates centered around 5.0
        bootstrap = [_make_estimate(5.0 + 0.01 * i) for i in range(-5, 6)]
        inp = BootstrappedEstimatesInput(
            primary_estimate=primary,
            proxy_bootstrapped_estimates=bootstrap,
        )
        result = check.compute(inp)
        # Normalized bias should be small
        self.assertLess(result.normalized_bias, 1.0)

    def test_biased(self) -> None:
        check = UnbiasednessCheck()
        primary = _make_estimate(5.0)
        # Bootstrap centered far from primary
        bootstrap = [_make_estimate(50.0 + 0.01 * i) for i in range(-5, 6)]
        inp = BootstrappedEstimatesInput(
            primary_estimate=primary,
            proxy_bootstrapped_estimates=bootstrap,
        )
        result = check.compute(inp)
        self.assertGreater(result.normalized_bias, 1.0)


# ---- CIOverlapCheck ----


class TestCIOverlapCheck(TestCase):
    def test_identical_cis(self) -> None:
        check = CIOverlapCheck()
        est = _make_estimate(5.0, se=1.0)
        inp = AggregateInput(primary_estimate=est, proxy_estimate=est)
        result = check.compute(inp)
        self.assertAlmostEqual(result.overlap_coefficient, 1.0)

    def test_no_overlap(self) -> None:
        check = CIOverlapCheck()
        inp = AggregateInput(
            primary_estimate=_make_estimate(0.0, se=0.01),
            proxy_estimate=_make_estimate(100.0, se=0.01),
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.overlap_coefficient, 0.0)

    def test_partial_overlap(self) -> None:
        check = CIOverlapCheck()
        inp = AggregateInput(
            primary_estimate=_make_estimate(0.0, se=1.0),
            proxy_estimate=_make_estimate(2.0, se=1.0),
        )
        result = check.compute(inp)
        self.assertGreater(result.overlap_coefficient, 0.0)
        self.assertLess(result.overlap_coefficient, 1.0)


# ---- RelativePrecisionCheck ----


class TestRelativePrecisionCheck(TestCase):
    def test_equal_precision(self) -> None:
        check = RelativePrecisionCheck()
        inp = AggregateInput(
            primary_estimate=_make_estimate(5.0, se=1.0),
            proxy_estimate=_make_estimate(5.0, se=1.0),
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.se_ratio, 1.0)

    def test_noisier_proxy(self) -> None:
        check = RelativePrecisionCheck()
        inp = AggregateInput(
            primary_estimate=_make_estimate(5.0, se=1.0),
            proxy_estimate=_make_estimate(5.0, se=3.0),
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.se_ratio, 3.0)

    def test_project_equal(self) -> None:
        check = RelativePrecisionCheck()
        inp = AggregateInput(
            primary_estimate=_make_estimate(5.0, se=1.0),
            proxy_estimate=_make_estimate(5.0, se=1.0),
        )
        result = check.compute(inp)
        # ratio=1 => log(1)=0 => 1/(1+0)=1
        self.assertAlmostEqual(check.project(result), 1.0)


# ---- SignAgreementCheck ----


class TestSignAgreementCheck(TestCase):
    def test_same_positive(self) -> None:
        check = SignAgreementCheck()
        inp = AggregateInput(
            primary_estimate=_make_estimate(5.0),
            proxy_estimate=_make_estimate(3.0),
        )
        result = check.compute(inp)
        self.assertTrue(result.signs_agree)

    def test_different_signs(self) -> None:
        check = SignAgreementCheck()
        inp = AggregateInput(
            primary_estimate=_make_estimate(5.0),
            proxy_estimate=_make_estimate(-3.0),
        )
        result = check.compute(inp)
        self.assertFalse(result.signs_agree)

    def test_both_zero(self) -> None:
        check = SignAgreementCheck()
        inp = AggregateInput(
            primary_estimate=_make_estimate(0.0),
            proxy_estimate=_make_estimate(0.0),
        )
        result = check.compute(inp)
        self.assertTrue(result.signs_agree)

    def test_project(self) -> None:
        check = SignAgreementCheck()
        inp = AggregateInput(
            primary_estimate=_make_estimate(5.0),
            proxy_estimate=_make_estimate(3.0),
        )
        result = check.compute(inp)
        self.assertAlmostEqual(check.project(result), 1.0)


# ---- EffectSizeAgreementCheck ----


class TestEffectSizeAgreementCheck(TestCase):
    def test_full_agreement(self) -> None:
        check = EffectSizeAgreementCheck()
        # Both estimates within each other's CIs
        inp = AggregateInput(
            primary_estimate=_make_estimate(5.0, se=1.0),
            proxy_estimate=_make_estimate(5.1, se=1.0),
        )
        result = check.compute(inp)
        self.assertTrue(result.primary_in_proxy_ci)
        self.assertTrue(result.proxy_in_primary_ci)
        self.assertAlmostEqual(result.agreement_score, 1.0)

    def test_no_agreement(self) -> None:
        check = EffectSizeAgreementCheck()
        # Neither estimate within the other's CI
        inp = AggregateInput(
            primary_estimate=_make_estimate(0.0, se=0.01),
            proxy_estimate=_make_estimate(100.0, se=0.01),
        )
        result = check.compute(inp)
        self.assertFalse(result.primary_in_proxy_ci)
        self.assertFalse(result.proxy_in_primary_ci)
        self.assertAlmostEqual(result.agreement_score, 0.0)


# ---- End-to-end run() ----


class TestEndToEnd(TestCase):
    def test_aggregate_bias_run(self) -> None:
        check = AggregateBiasCheck(threshold=0.8)
        inp = AggregateInput(
            primary_estimate=_make_estimate(10.0),
            proxy_estimate=_make_estimate(10.5),
        )
        result = check.run(inp)
        self.assertEqual(result.check_name, "Aggregate Bias")
        self.assertTrue(result.thresholding_result)

    def test_sign_agreement_run_fails(self) -> None:
        check = SignAgreementCheck(threshold=0.8)
        inp = AggregateInput(
            primary_estimate=_make_estimate(5.0),
            proxy_estimate=_make_estimate(-3.0),
        )
        result = check.run(inp)
        self.assertEqual(result.check_name, "Sign Agreement")
        self.assertFalse(result.thresholding_result)
