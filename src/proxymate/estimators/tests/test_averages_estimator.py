# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from unittest import TestCase

import numpy as np
import pandas as pd
from proxymate.estimators.averages_estimator import WeightedAverageEstimator
from proxymate.lib.base_classes.df_based_estimator_base import DfBasedEstimatorBaseInput


class WeightedAverageEstimatorTest(TestCase):
    def test_average_without_weights(self) -> None:
        df = pd.DataFrame({"metric": [1.0, 2.0, 3.0, 4.0, 5.0]})
        estimator = WeightedAverageEstimator(confidence_level=0.95)

        result = estimator.compute_estimator(DfBasedEstimatorBaseInput(df=df))

        self.assertEqual(result.estimate_val, 3.0)
        self.assertEqual(result.sample_size, 5)
        self.assertEqual(result.confidence_interval_level, 0.95)

    def test_average_with_weights(self) -> None:
        df = pd.DataFrame({"metric": [1.0, 2.0, 3.0], "weight": [2.0, 3.0, 4.0]})
        estimator = WeightedAverageEstimator(
            confidence_level=0.95, weight_column="weight"
        )

        result = estimator.compute_estimator(DfBasedEstimatorBaseInput(df=df))

        # sum(metric * weight) / sum(weight) = (2+6+12) / (2+3+4) = 20/9
        expected_mean = 20.0 / 9.0
        self.assertAlmostEqual(result.estimate_val, expected_mean, places=4)
        self.assertEqual(result.sample_size, 3)

    def test_standard_error_without_weights(self) -> None:
        df = pd.DataFrame({"metric": [1.0, 2.0, 3.0, 4.0, 5.0]})
        estimator = WeightedAverageEstimator(confidence_level=0.95)

        result = estimator.compute_estimator(DfBasedEstimatorBaseInput(df=df))

        # SE of the mean = sqrt(sum(residuals^2)) / sum(w); denominator is
        # (sum w)^2, not sum(w^2). Unit weights: residuals = metric - 3.0 =
        # [-2,-1,0,1,2], sum(r^2) = 10, sum(w) = 5 => var = 10 / 5^2 = 0.4
        # (equivalently SD / sqrt(N)).
        est = 3.0
        residuals = (df["metric"] - est) * 1.0
        expected_std_error = np.sqrt((residuals**2).sum() / 5.0**2)
        self.assertAlmostEqual(result.standard_error, expected_std_error, places=4)

    def test_standard_error_with_weights(self) -> None:
        df = pd.DataFrame({"metric": [1.0, 2.0, 3.0], "weight": [2.0, 3.0, 4.0]})
        estimator = WeightedAverageEstimator(
            confidence_level=0.95, weight_column="weight"
        )

        result = estimator.compute_estimator(DfBasedEstimatorBaseInput(df=df))

        # Hajek SE = sqrt(sum_i w_i^2 (y_i - mu)^2) / sum_i w_i
        # estimate_val = sum(metric*weight)/sum(weight) = 20/9
        est = 20.0 / 9.0
        weights = pd.Series([2.0, 3.0, 4.0])
        residuals = (pd.Series([1.0, 2.0, 3.0]) - est) * weights
        expected_std_error = np.sqrt((residuals**2).sum() / weights.sum() ** 2)
        self.assertAlmostEqual(result.standard_error, expected_std_error, places=4)

    def test_standard_error_is_variance_of_mean_not_population_spread(self) -> None:
        # Regression test: the weighted-mean SE must be the variance of the
        # MEAN (Hajek/linearization), not the per-observation weighted spread.
        # A prior bug divided by sum(w^2) instead of (sum w)^2, so for binary
        # prevalence data over ~1e5 rows it returned ~the population SD (~0.17)
        # and CIs that spilled below 0.
        rng = np.random.default_rng(0)
        n = 100_000
        y = (rng.random(n) < 0.03).astype(float)  # binary, prevalence ~3%
        w = 1.0 / np.clip(rng.beta(0.5, 3, n), 1e-3, 1.0)  # heavy-tailed IW
        df = pd.DataFrame({"metric": y, "weight": w})
        estimator = WeightedAverageEstimator(weight_column="weight")

        result = estimator.compute_estimator(DfBasedEstimatorBaseInput(df=df))

        mu = float(np.average(y, weights=w))
        expected_se = float(np.sqrt(np.sum((w * (y - mu)) ** 2)) / w.sum())
        self.assertAlmostEqual(result.estimate_val, mu, places=10)
        self.assertAlmostEqual(result.standard_error, expected_se, places=10)
        # SE of the mean is orders of magnitude below the population SD (~0.17)...
        self.assertLess(result.standard_error, 0.01)
        # ...and the 95% CI stays inside [0, 1] for a bounded {0, 1} outcome.
        self.assertGreaterEqual(result.lower_bound, 0.0)
        self.assertLessEqual(result.upper_bound, 1.0)

    def test_confidence_intervals(self) -> None:
        df = pd.DataFrame({"metric": [10.0, 20.0, 30.0]})
        estimator = WeightedAverageEstimator(confidence_level=0.95)

        result = estimator.compute_estimator(DfBasedEstimatorBaseInput(df=df))

        self.assertLess(result.lower_bound, result.estimate_val)
        self.assertGreater(result.upper_bound, result.estimate_val)

    def test_get_name(self) -> None:
        estimator = WeightedAverageEstimator()

        self.assertEqual(estimator.get_name(), "Weighted Average")

    def test_get_additional_entities_columns(self) -> None:
        estimator = WeightedAverageEstimator()

        self.assertEqual(estimator.get_additional_entities_columns(), set())

    def test_get_additional_metric_columns_with_weight(self) -> None:
        estimator = WeightedAverageEstimator(weight_column="weight")

        self.assertEqual(estimator.get_additional_metric_columns(), {"weight"})

    def test_get_additional_metric_columns_without_weight(self) -> None:
        estimator = WeightedAverageEstimator()

        self.assertEqual(estimator.get_additional_metric_columns(), set())

    def test_confidence_level_rounding(self) -> None:
        estimator = WeightedAverageEstimator(confidence_level=0.956)

        self.assertEqual(estimator._confidence_level, 0.96)

    def test_single_value(self) -> None:
        df = pd.DataFrame({"metric": [42.0]})
        estimator = WeightedAverageEstimator(confidence_level=0.95)

        result = estimator.compute_estimator(DfBasedEstimatorBaseInput(df=df))

        self.assertEqual(result.estimate_val, 42.0)
        self.assertEqual(result.sample_size, 1)
        # residual = (42-42)*1 = 0, so SE = 0
        self.assertEqual(result.standard_error, 0.0)

    def test_negative_values(self) -> None:
        df = pd.DataFrame({"metric": [-1.0, -2.0, 3.0]})
        estimator = WeightedAverageEstimator(confidence_level=0.95)

        result = estimator.compute_estimator(DfBasedEstimatorBaseInput(df=df))

        self.assertAlmostEqual(result.estimate_val, 0.0, places=4)
        self.assertEqual(result.sample_size, 3)
