# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from unittest import TestCase

from proxymate.checks.domain_level.meta_analysis import (
    AggregateAssociationCheck,
    DirectionalAlignmentCheck,
    EffectSensitivityCheck,
    HeterogeneityCheck,
    PredictionAccuracyCheck,
    ResidualAutocorrelationCheck,
)
from proxymate.lib.base_classes.data_types import Estimate
from proxymate.types.enums import DomainLevelProperty, ValidationLevel
from proxymate.types.inputs import MetaAnalysisInput


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


def _make_meta_input(
    primary_vals: list[float],
    proxy_vals: list[float],
    se: float = 0.1,
    primary_ses: list[float] | None = None,
    proxy_ses: list[float] | None = None,
) -> MetaAnalysisInput:
    pairs = []
    for i, (p, q) in enumerate(zip(primary_vals, proxy_vals)):
        p_se = primary_ses[i] if primary_ses is not None else se
        q_se = proxy_ses[i] if proxy_ses is not None else se
        pairs.append((_make_estimate(p, se=p_se), _make_estimate(q, se=q_se)))
    return MetaAnalysisInput(domain_estimates=pairs)


# ---- AggregateAssociationCheck ----


class TestAggregateAssociationCheck(TestCase):
    def test_perfect_linear(self) -> None:
        check = AggregateAssociationCheck()
        # Perfect linear: proxy = 2*primary + 1
        inp = _make_meta_input(
            [3.0, 5.0, 7.0, 9.0, 11.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.r_squared, 1.0, places=5)
        self.assertEqual(result.n_domains, 5)

    def test_no_association(self) -> None:
        check = AggregateAssociationCheck()
        # Constant primary => R² = 0
        inp = _make_meta_input(
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.r_squared, 0.0)

    def test_metadata(self) -> None:
        self.assertEqual(
            AggregateAssociationCheck.validation_level,
            ValidationLevel.DOMAIN_LEVEL,
        )
        self.assertEqual(
            AggregateAssociationCheck.sub_property,
            DomainLevelProperty.AGGREGATE_ASSOCIATION,
        )


# ---- EffectSensitivityCheck ----


class TestEffectSensitivityCheck(TestCase):
    def test_perfect_sensitivity(self) -> None:
        check = EffectSensitivityCheck()
        # Slope = 1: primary = proxy
        inp = _make_meta_input(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.slope, 1.0, places=5)

    def test_half_sensitivity(self) -> None:
        check = EffectSensitivityCheck()
        # primary = 0.5 * proxy => slope = 0.5
        inp = _make_meta_input(
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.slope, 0.5, places=5)

    def test_project(self) -> None:
        check = EffectSensitivityCheck()
        inp = _make_meta_input(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        result = check.compute(inp)
        # Slope=1 => project = 1 - |1-1| = 1.0
        self.assertAlmostEqual(check.project(result), 1.0, places=5)


# ---- DirectionalAlignmentCheck ----


class TestDirectionalAlignmentCheck(TestCase):
    def test_all_concordant(self) -> None:
        check = DirectionalAlignmentCheck()
        inp = _make_meta_input(
            [1.0, 2.0, -1.0, -2.0],
            [0.5, 1.0, -0.5, -1.0],
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.concordance_rate, 1.0)
        self.assertEqual(result.n_concordant, 4)

    def test_half_concordant(self) -> None:
        check = DirectionalAlignmentCheck()
        inp = _make_meta_input(
            [1.0, 2.0, -1.0, -2.0],
            [0.5, 1.0, 0.5, 1.0],  # last two flip sign
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.concordance_rate, 0.5)


# ---- PredictionAccuracyCheck ----


class TestPredictionAccuracyCheck(TestCase):
    def test_perfect_prediction(self) -> None:
        check = PredictionAccuracyCheck()
        # Perfect linear relationship => LOO error should be very small
        inp = _make_meta_input(
            [2.0, 4.0, 6.0, 8.0, 10.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.loo_mse, 0.0, places=5)
        self.assertAlmostEqual(result.loo_mae, 0.0, places=5)

    def test_k2_returns_inf(self) -> None:
        check = PredictionAccuracyCheck()
        inp = _make_meta_input([1.0, 2.0], [1.0, 2.0])
        result = check.compute(inp)
        self.assertEqual(result.loo_mse, float("inf"))
        self.assertEqual(result.loo_mae, float("inf"))
        self.assertEqual(result.n_domains, 2)

    def test_project(self) -> None:
        check = PredictionAccuracyCheck()
        inp = _make_meta_input(
            [2.0, 4.0, 6.0, 8.0, 10.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        result = check.compute(inp)
        # Near-zero MAE => project ~ 1/(1+0) = 1
        self.assertGreater(check.project(result), 0.99)


# ---- HeterogeneityCheck ----


class TestHeterogeneityCheck(TestCase):
    def test_no_heterogeneity(self) -> None:
        check = HeterogeneityCheck()
        # Perfect linear => no residual variation
        inp = _make_meta_input(
            [2.0, 4.0, 6.0, 8.0, 10.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.i_squared, 0.0, places=3)

    def test_k2_returns_zeros(self) -> None:
        check = HeterogeneityCheck()
        inp = _make_meta_input([1.0, 2.0], [1.0, 2.0])
        result = check.compute(inp)
        self.assertAlmostEqual(result.i_squared, 0.0)
        self.assertAlmostEqual(result.tau_squared, 0.0)
        self.assertAlmostEqual(result.q_statistic, 0.0)
        self.assertAlmostEqual(result.q_p_value, 1.0)
        self.assertEqual(result.n_domains, 2)

    def test_project(self) -> None:
        check = HeterogeneityCheck()
        inp = _make_meta_input(
            [2.0, 4.0, 6.0, 8.0, 10.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        result = check.compute(inp)
        # I² ~ 0 => project ~ 1
        self.assertGreater(check.project(result), 0.99)


# ---- End-to-end run() ----


class TestEndToEnd(TestCase):
    def test_association_run(self) -> None:
        check = AggregateAssociationCheck(threshold=0.5)
        inp = _make_meta_input(
            [2.0, 4.0, 6.0, 8.0, 10.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        result = check.run(inp)
        self.assertEqual(result.check_name, "Aggregate Association")
        self.assertTrue(result.thresholding_result)

    def test_directional_alignment_run(self) -> None:
        check = DirectionalAlignmentCheck(threshold=0.8)
        inp = _make_meta_input(
            [1.0, 2.0, -1.0, -2.0],
            [0.5, 1.0, 0.5, 1.0],
        )
        result = check.run(inp)
        self.assertEqual(result.check_name, "Directional Alignment")
        self.assertFalse(result.thresholding_result)


# ---- WLS vs OLS tests ----


class TestWLSWeighting(TestCase):
    def test_association_weighted_flag_with_ses(self) -> None:
        check = AggregateAssociationCheck()
        inp = _make_meta_input([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], se=0.1)
        result = check.compute(inp)
        self.assertTrue(result.weighted)

    def test_association_unweighted_flag_with_zero_se(self) -> None:
        check = AggregateAssociationCheck()
        inp = _make_meta_input([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], se=0.0)
        result = check.compute(inp)
        self.assertFalse(result.weighted)

    def test_sensitivity_weighted_flag(self) -> None:
        check = EffectSensitivityCheck()
        inp = _make_meta_input([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], se=0.5)
        result = check.compute(inp)
        self.assertTrue(result.weighted)

    def test_sensitivity_unweighted_flag(self) -> None:
        check = EffectSensitivityCheck()
        inp = _make_meta_input([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], se=0.0)
        result = check.compute(inp)
        self.assertFalse(result.weighted)

    def test_prediction_weighted_flag(self) -> None:
        check = PredictionAccuracyCheck()
        inp = _make_meta_input([2.0, 4.0, 6.0, 8.0], [1.0, 2.0, 3.0, 4.0], se=0.2)
        result = check.compute(inp)
        self.assertTrue(result.weighted)

    def test_prediction_unweighted_flag(self) -> None:
        check = PredictionAccuracyCheck()
        inp = _make_meta_input([2.0, 4.0, 6.0, 8.0], [1.0, 2.0, 3.0, 4.0], se=0.0)
        result = check.compute(inp)
        self.assertFalse(result.weighted)

    def test_wls_downweights_noisy_domain(self) -> None:
        """WLS should give less influence to a domain with large SEs."""
        check = EffectSensitivityCheck()
        # 4 precise domains with slope=1, one noisy outlier that would pull OLS
        primary = [1.0, 2.0, 3.0, 4.0, 10.0]
        proxy = [1.0, 2.0, 3.0, 4.0, 2.0]
        # Precise SEs for first 4, very large SE for the outlier
        p_ses = [0.05, 0.05, 0.05, 0.05, 5.0]
        x_ses = [0.05, 0.05, 0.05, 0.05, 5.0]
        wls_inp = _make_meta_input(primary, proxy, primary_ses=p_ses, proxy_ses=x_ses)
        ols_inp = _make_meta_input(primary, proxy, se=0.0)

        wls_result = check.compute(wls_inp)
        ols_result = check.compute(ols_inp)

        # WLS slope should be closer to 1.0 than OLS because the outlier is
        # downweighted
        self.assertLess(
            abs(1.0 - wls_result.slope),
            abs(1.0 - ols_result.slope),
        )


# ---- ResidualAutocorrelationCheck ----


class TestResidualAutocorrelationCheck(TestCase):
    def test_independent_residuals(self) -> None:
        check = ResidualAutocorrelationCheck()
        inp = _make_meta_input(
            [2.0, 4.0, 6.0, 8.0, 10.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.durbin_watson, 2.0, places=1)
        self.assertFalse(result.autocorrelation_detected)

    def test_positive_autocorrelation(self) -> None:
        check = ResidualAutocorrelationCheck()
        # Residuals have a block pattern (low then high) -> positive autocorrelation
        primary = [1.0, 1.1, 1.2, 1.3, 1.4, 5.0, 5.1, 5.2, 5.3, 5.4]
        proxy = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
        inp = _make_meta_input(primary, proxy, se=0.0)
        result = check.compute(inp)
        self.assertLess(result.durbin_watson, 1.5)
        self.assertTrue(result.autocorrelation_detected)

    def test_k2_returns_default(self) -> None:
        check = ResidualAutocorrelationCheck()
        inp = _make_meta_input([1.0, 2.0], [1.0, 2.0])
        result = check.compute(inp)
        self.assertAlmostEqual(result.durbin_watson, 2.0)
        self.assertFalse(result.autocorrelation_detected)

    def test_project(self) -> None:
        check = ResidualAutocorrelationCheck()
        inp = _make_meta_input(
            [2.0, 4.0, 6.0, 8.0, 10.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        result = check.compute(inp)
        projection = check.project(result)
        self.assertGreater(projection, 0.9)
