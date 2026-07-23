# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from unittest import TestCase

import numpy as np
import pandas as pd
from proxymate.checks.unit_level.accuracy import MAPECheck, TTestCheck
from proxymate.checks.unit_level.agreement import (
    BlandAltmanCheck,
    DirectionalConcordanceCheck,
    LinCCCCheck,
)
from proxymate.checks.unit_level.precision import (
    PearsonCorrelationCheck,
    RMSECheck,
    RSquaredCheck,
)
from proxymate.types.enums import UnitLevelProperty, ValidationLevel
from proxymate.types.inputs import CheckInput


def _make_input(
    primary: list[float],
    proxy: list[float],
    weight: list[float] | None = None,
) -> CheckInput:
    data: dict[str, list[float]] = {
        "primary_outcome": primary,
        "proxy_outcome": proxy,
    }
    weight_col = None
    if weight is not None:
        data["weight"] = weight
        weight_col = "weight"
    df = pd.DataFrame(data)
    return CheckInput(df=df, weight_col=weight_col)


class TestTTestCheck(TestCase):
    def test_identical_data(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = _make_input(data, data)
        check = TTestCheck()
        result = check.compute(ci)
        # Identical data → no difference → high p-value
        self.assertAlmostEqual(result.mean_diff, 0.0)
        self.assertGreater(result.p_value, 0.9)

    def test_shifted_data(self) -> None:
        primary = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        proxy = [x + 10.0 for x in primary]
        ci = _make_input(primary, proxy)
        check = TTestCheck()
        result = check.compute(ci)
        self.assertAlmostEqual(result.mean_diff, 10.0)
        self.assertLess(result.p_value, 0.01)

    def test_metadata(self) -> None:
        self.assertEqual(TTestCheck.validation_level, ValidationLevel.UNIT_LEVEL)
        self.assertEqual(TTestCheck.sub_property, UnitLevelProperty.ACCURACY)

    def test_run_end_to_end(self) -> None:
        ci = _make_input([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        check = TTestCheck()
        result = check.run(ci)
        self.assertEqual(result.check_name, "T-Test")
        self.assertIsNotNone(result.check_result)

    def test_small_sample(self) -> None:
        ci = _make_input([1.0], [2.0])
        check = TTestCheck()
        result = check.compute(ci)
        self.assertEqual(result.n_samples, 1)
        self.assertAlmostEqual(result.p_value, 1.0)

    def test_uniform_weights_match_unweighted(self) -> None:
        primary = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        proxy = [x + 10.0 for x in primary]
        ci_unw = _make_input(primary, proxy)
        ci_w = _make_input(primary, proxy, weight=[1.0] * len(primary))
        check = TTestCheck()
        r_unw = check.compute(ci_unw)
        r_w = check.compute(ci_w)
        self.assertAlmostEqual(r_unw.mean_diff, r_w.mean_diff, places=5)
        self.assertAlmostEqual(r_unw.p_value, r_w.p_value, places=2)

    def test_weights_shift_mean_diff(self) -> None:
        # primary=[0, 0, 10], proxy=[0, 0, 10]
        # Without weights: mean_diff = 0
        # With weights=[1,1,100]: heavily weights the last sample
        # mean_diff should still be ~0 since primary and proxy match
        ci = _make_input(
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 10.0],
            weight=[1.0, 1.0, 100.0],
        )
        check = TTestCheck()
        result = check.compute(ci)
        self.assertAlmostEqual(result.mean_diff, 0.0)
        self.assertGreater(result.p_value, 0.9)

    def test_weights_detect_bias(self) -> None:
        # primary and proxy differ; weights amplify the difference
        primary = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        proxy = [x + 10.0 for x in primary]
        ci = _make_input(primary, proxy, weight=[1.0] * len(primary))
        check = TTestCheck()
        result = check.compute(ci)
        self.assertAlmostEqual(result.mean_diff, 10.0)
        self.assertLess(result.p_value, 0.01)

    def test_concentrated_weights_returns_default(self) -> None:
        # One sample has almost all the weight → n_eff ≈ 1 → should return
        # default (p=1.0) to avoid division by zero
        ci = _make_input(
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            weight=[1e12, 1e-6, 1e-6],
        )
        check = TTestCheck()
        result = check.compute(ci)
        self.assertAlmostEqual(result.p_value, 1.0)


class TestMAPECheck(TestCase):
    def test_perfect_proxy(self) -> None:
        ci = _make_input([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        check = MAPECheck()
        result = check.compute(ci)
        self.assertAlmostEqual(result.mape, 0.0)

    def test_known_mape(self) -> None:
        # 10% error on each
        ci = _make_input([100.0, 200.0], [110.0, 220.0])
        check = MAPECheck()
        result = check.compute(ci)
        self.assertAlmostEqual(result.mape, 0.1)

    def test_project(self) -> None:
        check = MAPECheck()
        from proxymate.checks.unit_level.accuracy import MAPEResult

        self.assertAlmostEqual(check.project(MAPEResult(mape=0.0, n_samples=10)), 1.0)
        self.assertAlmostEqual(check.project(MAPEResult(mape=1.0, n_samples=10)), 0.5)

    def test_all_zero_primary(self) -> None:
        ci = _make_input([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
        check = MAPECheck()
        result = check.compute(ci)
        self.assertEqual(result.mape, float("inf"))
        self.assertEqual(result.n_samples, 0)


class TestRMSECheck(TestCase):
    def test_perfect_proxy(self) -> None:
        ci = _make_input([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
        check = RMSECheck()
        result = check.compute(ci)
        self.assertAlmostEqual(result.rmse, 0.0)

    def test_known_rmse(self) -> None:
        ci = _make_input([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
        check = RMSECheck()
        result = check.compute(ci)
        self.assertAlmostEqual(result.rmse, 1.0)

    def test_metadata(self) -> None:
        self.assertEqual(RMSECheck.sub_property, UnitLevelProperty.PRECISION)

    def test_small_sample(self) -> None:
        ci = _make_input([1.0], [2.0])
        check = RMSECheck()
        result = check.compute(ci)
        self.assertEqual(result.rmse, float("inf"))


class TestRSquaredCheck(TestCase):
    def test_perfect_proxy(self) -> None:
        ci = _make_input([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
        check = RSquaredCheck()
        result = check.compute(ci)
        self.assertAlmostEqual(result.r_squared, 1.0)

    def test_poor_proxy(self) -> None:
        rng = np.random.RandomState(42)  # pyre-ignore[bad-argument-type]
        primary = rng.randn(100).tolist()
        proxy = rng.randn(100).tolist()
        ci = _make_input(primary, proxy)
        check = RSquaredCheck()
        result = check.compute(ci)
        self.assertLess(result.r_squared, 0.1)

    def test_small_sample(self) -> None:
        ci = _make_input([1.0], [2.0])
        check = RSquaredCheck()
        result = check.compute(ci)
        self.assertAlmostEqual(result.r_squared, 0.0)

    def test_with_weights(self) -> None:
        ci = _make_input(
            [1.0, 2.0, 3.0, 4.0],
            [1.1, 2.1, 3.1, 4.1],
            weight=[1.0, 1.0, 1.0, 1.0],
        )
        check = RSquaredCheck()
        result = check.compute(ci)
        self.assertGreater(result.r_squared, 0.95)

    def test_metadata(self) -> None:
        self.assertEqual(RSquaredCheck.sub_property, UnitLevelProperty.PRECISION)


class TestPearsonCorrelationCheck(TestCase):
    def test_perfect_correlation(self) -> None:
        ci = _make_input([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
        check = PearsonCorrelationCheck()
        result = check.compute(ci)
        self.assertAlmostEqual(result.correlation, 1.0)
        self.assertLess(result.p_value, 0.05)

    def test_no_correlation(self) -> None:
        rng = np.random.RandomState(42)  # pyre-ignore[bad-argument-type]
        primary = rng.randn(200).tolist()
        proxy = rng.randn(200).tolist()
        ci = _make_input(primary, proxy)
        check = PearsonCorrelationCheck()
        result = check.compute(ci)
        self.assertLess(abs(result.correlation), 0.2)

    def test_small_sample(self) -> None:
        ci = _make_input([1.0], [2.0])
        check = PearsonCorrelationCheck()
        result = check.compute(ci)
        self.assertAlmostEqual(result.correlation, 0.0)
        self.assertAlmostEqual(result.p_value, 1.0)

    def test_with_weights(self) -> None:
        ci = _make_input(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.1, 2.2, 2.9, 4.1, 5.0],
            weight=[1.0, 1.0, 1.0, 1.0, 1.0],
        )
        check = PearsonCorrelationCheck()
        result = check.compute(ci)
        self.assertGreater(result.correlation, 0.95)

    def test_run_end_to_end(self) -> None:
        ci = _make_input([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        check = PearsonCorrelationCheck()
        result = check.run(ci)
        self.assertEqual(result.check_name, "Pearson Correlation")
        self.assertTrue(result.thresholding_result)

    def test_metadata(self) -> None:
        self.assertEqual(
            PearsonCorrelationCheck.sub_property, UnitLevelProperty.PRECISION
        )


class TestLinCCCCheck(TestCase):
    def test_perfect_agreement(self) -> None:
        # Use large N so ddof=0 vs ddof=1 difference is negligible
        data = [float(x) for x in range(1, 101)]
        ci = _make_input(data, data)
        check = LinCCCCheck()
        result = check.compute(ci)
        self.assertGreater(result.ccc, 0.98)

    def test_no_agreement(self) -> None:
        rng = np.random.RandomState(42)  # pyre-ignore[bad-argument-type]
        primary = rng.randn(200).tolist()
        proxy = rng.randn(200).tolist()
        ci = _make_input(primary, proxy)
        check = LinCCCCheck()
        result = check.compute(ci)
        self.assertLess(abs(result.ccc), 0.3)

    def test_small_sample(self) -> None:
        ci = _make_input([1.0], [2.0])
        check = LinCCCCheck()
        result = check.compute(ci)
        self.assertAlmostEqual(result.ccc, 0.0)
        self.assertEqual(result.n_samples, 1)

    def test_metadata(self) -> None:
        self.assertEqual(LinCCCCheck.sub_property, UnitLevelProperty.AGREEMENT)

    def test_project_bounds(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0]
        ci = _make_input(data, data)
        check = LinCCCCheck()
        result = check.run(ci)
        self.assertGreaterEqual(result.projection_result, 0.0)
        self.assertLessEqual(result.projection_result, 1.0)


class TestBlandAltmanCheck(TestCase):
    def test_identical_data(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = _make_input(data, data)
        check = BlandAltmanCheck()
        result = check.compute(ci)
        self.assertAlmostEqual(result.mean_diff, 0.0)
        self.assertAlmostEqual(result.std_diff, 0.0)
        self.assertAlmostEqual(result.percent_within_loa, 1.0)

    def test_shifted_data(self) -> None:
        primary = [1.0, 2.0, 3.0, 4.0, 5.0]
        proxy = [x + 2.0 for x in primary]
        ci = _make_input(primary, proxy)
        check = BlandAltmanCheck()
        result = check.compute(ci)
        self.assertAlmostEqual(result.mean_diff, 2.0)
        # All diffs are identical so all within LOA
        self.assertAlmostEqual(result.percent_within_loa, 1.0)

    def test_small_sample(self) -> None:
        ci = _make_input([1.0], [2.0])
        check = BlandAltmanCheck()
        result = check.compute(ci)
        self.assertEqual(result.n_samples, 1)
        self.assertAlmostEqual(result.mean_diff, 0.0)

    def test_run_end_to_end(self) -> None:
        ci = _make_input([1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 2.9, 4.1])
        check = BlandAltmanCheck()
        result = check.run(ci)
        self.assertEqual(result.check_name, "Bland-Altman")
        self.assertIsNotNone(result.check_result)

    def test_metadata(self) -> None:
        self.assertEqual(BlandAltmanCheck.sub_property, UnitLevelProperty.AGREEMENT)


class TestDirectionalConcordanceCheck(TestCase):
    def test_same_signs(self) -> None:
        ci = _make_input([1.0, 2.0, -1.0, -2.0], [3.0, 4.0, -3.0, -4.0])
        check = DirectionalConcordanceCheck()
        result = check.compute(ci)
        self.assertAlmostEqual(result.concordance_rate, 1.0)
        self.assertEqual(result.n_concordant, 4)

    def test_opposite_signs(self) -> None:
        ci = _make_input([1.0, 2.0, -1.0, -2.0], [-3.0, -4.0, 3.0, 4.0])
        check = DirectionalConcordanceCheck()
        result = check.compute(ci)
        self.assertAlmostEqual(result.concordance_rate, 0.0)

    def test_mixed_signs(self) -> None:
        ci = _make_input([1.0, -1.0, 1.0, -1.0], [1.0, -1.0, -1.0, 1.0])
        check = DirectionalConcordanceCheck()
        result = check.compute(ci)
        self.assertAlmostEqual(result.concordance_rate, 0.5)

    def test_empty_input(self) -> None:
        ci = _make_input([], [])
        check = DirectionalConcordanceCheck()
        result = check.compute(ci)
        self.assertAlmostEqual(result.concordance_rate, 0.0)
        self.assertEqual(result.n_total, 0)

    def test_run_end_to_end(self) -> None:
        ci = _make_input([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        check = DirectionalConcordanceCheck()
        result = check.run(ci)
        self.assertEqual(result.check_name, "Directional Concordance")
        self.assertTrue(result.thresholding_result)

    def test_metadata(self) -> None:
        self.assertEqual(
            DirectionalConcordanceCheck.sub_property, UnitLevelProperty.AGREEMENT
        )
