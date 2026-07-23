# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from unittest import TestCase

import numpy as np
import pandas as pd
from proxymate.adjustment.domain_level import (
    EstimateLevelAdjustment,
    RegressionCalibrator,
)
from proxymate.adjustment.estimate_level import PPIEstimator
from proxymate.adjustment.unit_level import (
    AdditiveBiasCorrector,
    IsotonicRecalibrator,
    PlattRecalibrator,
)
from proxymate.lib.base_classes.data_types import Estimate
from proxymate.types.inputs import CheckInput, MetaAnalysisInput


def _make_check_input(
    primary: list[float],
    proxy: list[float],
) -> CheckInput:
    return CheckInput(
        df=pd.DataFrame({"primary": primary, "proxy": proxy}),
        primary_col="primary",
        proxy_col="proxy",
    )


def _make_estimate(
    val: float,
    se: float = 0.1,
    n: int = 100,
) -> Estimate:
    return Estimate(
        confidence_interval_level=0.95,
        estimate_val=val,
        lower_bound=val - 1.96 * se,
        upper_bound=val + 1.96 * se,
        sample_size=n,
        standard_error=se,
    )


# ---- Unit-level: PlattRecalibrator ----


class TestPlattRecalibrator(TestCase):
    def test_fit_and_transform(self) -> None:
        recal = PlattRecalibrator()
        # Perfectly calibrated data
        inp = _make_check_input(
            [0, 0, 0, 1, 1, 1] * 10,
            [0.1, 0.2, 0.3, 0.7, 0.8, 0.9] * 10,
        )
        recal.fit(inp)
        self.assertTrue(recal._fitted)

        scores = np.array([0.1, 0.5, 0.9])
        calibrated = recal.transform(scores)
        self.assertEqual(len(calibrated), 3)
        # Calibrated scores should be in [0, 1]
        self.assertTrue(np.all(calibrated >= 0))
        self.assertTrue(np.all(calibrated <= 1))

    def test_transform_without_fit_raises(self) -> None:
        recal = PlattRecalibrator()
        with self.assertRaises(RuntimeError):
            recal.transform(np.array([0.5]))

    def test_fit_transform(self) -> None:
        recal = PlattRecalibrator()
        inp = _make_check_input(
            [0, 0, 1, 1] * 10,
            [0.2, 0.3, 0.7, 0.8] * 10,
        )
        result = recal.fit_transform(inp)
        self.assertEqual(len(result), 40)


# ---- Unit-level: IsotonicRecalibrator ----


class TestIsotonicRecalibrator(TestCase):
    def test_fit_and_transform(self) -> None:
        recal = IsotonicRecalibrator()
        inp = _make_check_input(
            [0, 0, 0, 1, 1, 1] * 10,
            [0.1, 0.2, 0.3, 0.7, 0.8, 0.9] * 10,
        )
        recal.fit(inp)
        self.assertTrue(recal._fitted)

        scores = np.array([0.1, 0.5, 0.9])
        calibrated = recal.transform(scores)
        self.assertEqual(len(calibrated), 3)

    def test_transform_without_fit_raises(self) -> None:
        recal = IsotonicRecalibrator()
        with self.assertRaises(RuntimeError):
            recal.transform(np.array([0.5]))


# ---- Unit-level: AdditiveBiasCorrector ----


class TestAdditiveBiasCorrector(TestCase):
    def test_no_bias(self) -> None:
        corrector = AdditiveBiasCorrector()
        inp = _make_check_input(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        )
        corrector.fit(inp)
        self.assertAlmostEqual(corrector.bias, 0.0)

    def test_constant_bias(self) -> None:
        corrector = AdditiveBiasCorrector()
        inp = _make_check_input(
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],  # proxy = primary + 1
        )
        corrector.fit(inp)
        self.assertAlmostEqual(corrector.bias, 1.0)

        corrected = corrector.transform(np.array([5.0, 6.0]))
        np.testing.assert_array_almost_equal(corrected, [4.0, 5.0])

    def test_transform_without_fit_raises(self) -> None:
        corrector = AdditiveBiasCorrector()
        with self.assertRaises(RuntimeError):
            corrector.transform(np.array([5.0]))


# ---- Aggregate-level: PPIEstimator ----


class TestPPIEstimator(TestCase):
    def test_no_bias(self) -> None:
        estimator = PPIEstimator()
        labeled = _make_check_input(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        )
        unlabeled = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        result = estimator.estimate(labeled, unlabeled)
        # correction = 0 => ppi_estimate = proxy_all_mean
        self.assertAlmostEqual(result.correction, 0.0)
        self.assertAlmostEqual(result.ppi_estimate, 4.0)

    def test_with_bias(self) -> None:
        estimator = PPIEstimator()
        labeled = _make_check_input(
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],  # proxy = primary + 1
        )
        unlabeled = np.array([5.0, 6.0, 7.0, 8.0])
        result = estimator.estimate(labeled, unlabeled)
        # correction = primary_mean - proxy_mean = 2.5 - 3.5 = -1.0
        self.assertAlmostEqual(result.correction, -1.0)
        # ppi = 6.5 + (-1.0) = 5.5
        self.assertAlmostEqual(result.ppi_estimate, 5.5)


# ---- Aggregate-level: RegressionCalibrator ----


class TestRegressionCalibrator(TestCase):
    def test_identity_fit(self) -> None:
        cal = RegressionCalibrator()
        meta = MetaAnalysisInput(
            domain_estimates=[
                (_make_estimate(1.0), _make_estimate(1.0)),
                (_make_estimate(2.0), _make_estimate(2.0)),
                (_make_estimate(3.0), _make_estimate(3.0)),
            ]
        )
        cal.fit(meta)
        self.assertAlmostEqual(cal._slope, 1.0, places=5)
        self.assertAlmostEqual(cal._intercept, 0.0, places=5)
        self.assertAlmostEqual(cal._r_squared, 1.0, places=5)

    def test_predict(self) -> None:
        cal = RegressionCalibrator()
        meta = MetaAnalysisInput(
            domain_estimates=[
                (_make_estimate(2.0), _make_estimate(1.0)),
                (_make_estimate(4.0), _make_estimate(2.0)),
                (_make_estimate(6.0), _make_estimate(3.0)),
            ]
        )
        cal.fit(meta)
        # primary = 2 * proxy, so slope=2, intercept=0
        self.assertAlmostEqual(cal._slope, 2.0, places=5)

        result = cal.predict(_make_estimate(5.0, se=0.1))
        self.assertAlmostEqual(result.calibrated_estimate.estimate_val, 10.0, places=5)

    def test_predict_without_fit_raises(self) -> None:
        cal = RegressionCalibrator()
        with self.assertRaises(RuntimeError):
            cal.predict(_make_estimate(5.0))


# ---- Aggregate-level: EstimateLevelAdjustment ----


class TestEstimateLevelAdjustment(TestCase):
    def test_no_bias(self) -> None:
        adj = EstimateLevelAdjustment()
        meta = MetaAnalysisInput(
            domain_estimates=[
                (_make_estimate(v), _make_estimate(v))
                for v in [1.0, 2.0, 3.0, 4.0, 5.0]
            ]
        )
        adj.fit(meta)
        result = adj.adjust(_make_estimate(5.0, se=0.1))
        self.assertAlmostEqual(result.rho, 0.0, places=2)

    def test_constant_bias(self) -> None:
        adj = EstimateLevelAdjustment()
        meta = MetaAnalysisInput(
            domain_estimates=[
                (_make_estimate(v), _make_estimate(v + 0.5, se=0.01))
                for v in [1.0, 2.0, 3.0, 4.0, 5.0]
            ]
        )
        adj.fit(meta)
        result = adj.adjust(_make_estimate(5.0, se=0.1))
        self.assertAlmostEqual(result.rho, 0.5, places=1)

    def test_adjust_subtracts_rho(self) -> None:
        adj = EstimateLevelAdjustment()
        meta = MetaAnalysisInput(
            domain_estimates=[
                (_make_estimate(v, se=0.01), _make_estimate(v + 1.0, se=0.01))
                for v in [1.0, 2.0, 3.0, 4.0, 5.0]
            ]
        )
        adj.fit(meta)
        result = adj.adjust(_make_estimate(10.0, se=0.1))
        self.assertAlmostEqual(
            result.adjusted_estimate.estimate_val, 10.0 - result.rho, places=3
        )

    def test_adjust_inflates_ci(self) -> None:
        np.random.seed(42)  # pyre-ignore[bad-argument-type]
        adj = EstimateLevelAdjustment()
        meta = MetaAnalysisInput(
            domain_estimates=[
                (
                    _make_estimate(v, se=0.1),
                    _make_estimate(v + np.random.randn() * 0.3, se=0.1),
                )
                for v in [1.0, 2.0, 3.0, 4.0, 5.0]
            ]
        )
        adj.fit(meta)
        proxy_est = _make_estimate(5.0, se=0.1)
        result = adj.adjust(proxy_est)
        self.assertGreater(
            result.adjusted_estimate.standard_error, proxy_est.standard_error
        )

    def test_adjust_without_fit_raises(self) -> None:
        adj = EstimateLevelAdjustment()
        with self.assertRaises(RuntimeError):
            adj.adjust(_make_estimate(5.0))
