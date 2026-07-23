# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from unittest import TestCase

import pandas as pd
from proxymate.types.data import CheckResult
from proxymate.types.enums import (
    DomainLevelProperty,
    EstimateLevelProperty,
    OutcomeType,
    ProxyChallenge,
    RepresentativityProperty,
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


class TestValidationLevel(TestCase):
    def test_values(self) -> None:
        self.assertEqual(ValidationLevel.UNIT_LEVEL.value, "unit_level")
        self.assertEqual(ValidationLevel.ESTIMATE_LEVEL.value, "estimate_level")
        self.assertEqual(ValidationLevel.DOMAIN_LEVEL.value, "domain_level")
        self.assertEqual(ValidationLevel.REPRESENTATIVITY.value, "representativity")

    def test_all_members(self) -> None:
        self.assertEqual(len(ValidationLevel), 4)


class TestUnitLevelProperty(TestCase):
    def test_all_properties(self) -> None:
        expected = {
            "accuracy",
            "precision",
            "agreement",
            "calibration",
            "discrimination",
            "distributional_fidelity",
        }
        actual = {p.value for p in UnitLevelProperty}
        self.assertEqual(actual, expected)


class TestEstimateLevelProperty(TestCase):
    def test_all_properties(self) -> None:
        expected = {
            "aggregate_bias",
            "inferential_validity",
            "unbiasedness",
            "ci_coverage",
            "ci_overlap",
            "relative_precision",
            "sign_agreement",
            "effect_size_agreement",
        }
        actual = {p.value for p in EstimateLevelProperty}
        self.assertEqual(actual, expected)


class TestDomainLevelProperty(TestCase):
    def test_all_properties(self) -> None:
        expected = {
            "aggregate_association",
            "effect_sensitivity",
            "directional_alignment",
            "prediction_accuracy",
            "heterogeneity",
        }
        actual = {p.value for p in DomainLevelProperty}
        self.assertEqual(actual, expected)


class TestOutcomeType(TestCase):
    def test_all_types(self) -> None:
        self.assertEqual(len(OutcomeType), 4)
        self.assertEqual(OutcomeType.CONTINUOUS.value, "continuous")
        self.assertEqual(OutcomeType.BINARY_PROBABILITY.value, "binary_prob")


class TestProxyChallenge(TestCase):
    def test_combinable_flags(self) -> None:
        combined = ProxyChallenge.LONG_TERM_PRIMARY | ProxyChallenge.RESTRICTED_LABELS
        self.assertTrue(combined & ProxyChallenge.LONG_TERM_PRIMARY)
        self.assertTrue(combined & ProxyChallenge.RESTRICTED_LABELS)
        self.assertFalse(combined & ProxyChallenge.LOW_DETECTABILITY)


class TestRepresentativityProperty(TestCase):
    def test_all_properties(self) -> None:
        expected = {
            "coverage",
            "sample_representativity",
            "effective_sample_size",
            "weight_extremity",
            "weighting_effectiveness",
        }
        actual = {p.value for p in RepresentativityProperty}
        self.assertEqual(actual, expected)


class TestCheckResult(TestCase):
    def test_construction(self) -> None:
        result: CheckResult[float] = CheckResult(
            check_name="test_check",
            description="A test check",
            validation_level=ValidationLevel.UNIT_LEVEL,
            sub_property=UnitLevelProperty.ACCURACY,
            check_result=0.95,
            projection_result=0.95,
            threshold=0.8,
            thresholding_result=True,
        )
        self.assertEqual(result.check_name, "test_check")
        self.assertEqual(result.validation_level, ValidationLevel.UNIT_LEVEL)
        self.assertEqual(result.sub_property, UnitLevelProperty.ACCURACY)
        self.assertTrue(result.thresholding_result)

    def test_failed_result(self) -> None:
        result: CheckResult[float] = CheckResult(
            check_name="failed_check",
            description="A failed check",
            validation_level=ValidationLevel.ESTIMATE_LEVEL,
            sub_property=EstimateLevelProperty.AGGREGATE_BIAS,
            check_result=0.3,
            projection_result=0.3,
            threshold=0.8,
            thresholding_result=False,
        )
        self.assertFalse(result.thresholding_result)
        self.assertLess(result.projection_result, result.threshold)


class TestCheckInput(TestCase):
    def test_defaults(self) -> None:
        df = pd.DataFrame({"primary_outcome": [1.0], "proxy_outcome": [1.1]})
        ci = CheckInput(df=df)
        self.assertEqual(ci.primary_col, "primary_outcome")
        self.assertEqual(ci.proxy_col, "proxy_outcome")
        self.assertIsNone(ci.weight_col)
        self.assertIsNone(ci.covariate_cols)

    def test_custom_columns(self) -> None:
        df = pd.DataFrame({"y": [1.0], "y_hat": [1.1], "w": [0.5]})
        ci = CheckInput(df=df, primary_col="y", proxy_col="y_hat", weight_col="w")
        self.assertEqual(ci.primary_col, "y")
        self.assertEqual(ci.proxy_col, "y_hat")
        self.assertEqual(ci.weight_col, "w")


class TestUnitLevelInput(TestCase):
    def test_iter_check_inputs_single_proxy(self) -> None:
        df = pd.DataFrame({"primary_outcome": [1.0, 2.0], "proxy_outcome": [1.1, 2.1]})
        uli = UnitLevelInput(df=df)
        check_inputs = uli.iter_check_inputs()
        self.assertEqual(len(check_inputs), 1)
        proxy_col, ci = check_inputs[0]
        self.assertEqual(proxy_col, "proxy_outcome")
        self.assertEqual(ci.primary_col, "primary_outcome")
        self.assertEqual(ci.proxy_col, "proxy_outcome")

    def test_iter_check_inputs_multiple_proxies(self) -> None:
        df = pd.DataFrame(
            {
                "primary": [1.0, 2.0],
                "proxy_a": [1.1, 2.1],
                "proxy_b": [0.9, 1.9],
            }
        )
        uli = UnitLevelInput(
            df=df,
            primary_col="primary",
            proxy_cols=["proxy_a", "proxy_b"],
        )
        check_inputs = uli.iter_check_inputs()
        self.assertEqual(len(check_inputs), 2)
        self.assertEqual(check_inputs[0][0], "proxy_a")
        self.assertEqual(check_inputs[1][0], "proxy_b")
        # Both should share the same DataFrame
        self.assertIs(check_inputs[0][1].df, check_inputs[1][1].df)

    def test_weight_and_covariates_propagated(self) -> None:
        df = pd.DataFrame(
            {
                "primary": [1.0],
                "proxy": [1.1],
                "w": [0.5],
                "cov1": ["a"],
            }
        )
        uli = UnitLevelInput(
            df=df,
            primary_col="primary",
            proxy_cols=["proxy"],
            weight_col="w",
            covariate_cols=["cov1"],
        )
        _, ci = uli.iter_check_inputs()[0]
        self.assertEqual(ci.weight_col, "w")
        self.assertEqual(ci.covariate_cols, ["cov1"])


class TestBootstrapConfig(TestCase):
    def test_defaults(self) -> None:
        config = BootstrapConfig()
        self.assertEqual(config.n_iterations, 100)
        self.assertEqual(config.confidence_level, 0.95)
        self.assertIsNone(config.seed)
        self.assertIsNone(config.estimator)

    def test_custom_values(self) -> None:
        config = BootstrapConfig(n_iterations=500, seed=42)
        self.assertEqual(config.n_iterations, 500)
        self.assertEqual(config.seed, 42)


class TestAggregateInput(TestCase):
    def test_construction(self) -> None:
        from proxymate.types.data import Estimate

        primary = Estimate(
            confidence_interval_level=0.95,
            estimate_val=10.0,
            lower_bound=9.0,
            upper_bound=11.0,
            sample_size=100,
            standard_error=0.5,
        )
        proxy = Estimate(
            confidence_interval_level=0.95,
            estimate_val=10.5,
            lower_bound=9.5,
            upper_bound=11.5,
            sample_size=100,
            standard_error=0.5,
        )
        agg_input = AggregateInput(primary_estimate=primary, proxy_estimate=proxy)
        self.assertEqual(agg_input.primary_estimate.estimate_val, 10.0)
        self.assertEqual(agg_input.proxy_estimate.estimate_val, 10.5)


class TestBootstrappedEstimatesInput(TestCase):
    def test_construction(self) -> None:
        from proxymate.types.data import Estimate

        primary = Estimate(
            confidence_interval_level=0.95,
            estimate_val=10.0,
            lower_bound=9.0,
            upper_bound=11.0,
            sample_size=100,
            standard_error=0.5,
        )
        boot_estimates = [
            Estimate(
                confidence_interval_level=0.95,
                estimate_val=10.0 + i * 0.1,
                lower_bound=9.0,
                upper_bound=11.0,
                sample_size=100,
                standard_error=0.5,
            )
            for i in range(5)
        ]
        bei = BootstrappedEstimatesInput(
            primary_estimate=primary,
            proxy_bootstrapped_estimates=boot_estimates,
        )
        self.assertEqual(len(bei.proxy_bootstrapped_estimates), 5)
        self.assertIsNone(bei.proxy_estimate)
        self.assertEqual(len(bei.primary_bootstrapped_estimates), 0)


class TestMetaAnalysisInput(TestCase):
    def test_construction(self) -> None:
        from proxymate.types.data import Estimate

        def _make_estimate(val: float) -> Estimate:
            return Estimate(
                confidence_interval_level=0.95,
                estimate_val=val,
                lower_bound=val - 1.0,
                upper_bound=val + 1.0,
                sample_size=100,
                standard_error=0.5,
            )

        domain_estimates = [
            (_make_estimate(1.0), _make_estimate(1.2)),
            (_make_estimate(2.0), _make_estimate(2.3)),
        ]
        mai = MetaAnalysisInput(
            domain_estimates=domain_estimates,
            domain_names=["exp1", "exp2"],
        )
        self.assertEqual(len(mai.domain_estimates), 2)
        self.assertEqual(mai.domain_names, ["exp1", "exp2"])

    def test_domain_names_optional(self) -> None:
        from proxymate.types.data import Estimate

        est = Estimate(
            confidence_interval_level=0.95,
            estimate_val=1.0,
            lower_bound=0.0,
            upper_bound=2.0,
            sample_size=100,
            standard_error=0.5,
        )
        mai = MetaAnalysisInput(domain_estimates=[(est, est)])
        self.assertIsNone(mai.domain_names)
