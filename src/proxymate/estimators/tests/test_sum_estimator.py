# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from unittest import TestCase

import numpy as np
import pandas as pd
from proxymate.estimators.sum_estimator import WeightedSumEstimator
from proxymate.lib.base_classes.df_based_estimator_base import DfBasedEstimatorBaseInput


class TestWeightedSumEstimator(TestCase):
    def test_sum_estimate_without_weights(self) -> None:
        # Setup: create a simple dataframe without weights
        df = pd.DataFrame({"metric": [1.0, 2.0, 3.0, 4.0, 5.0]})
        estimator = WeightedSumEstimator(confidence_level=0.95)

        # Execute: compute the estimator
        result = estimator.compute_estimator(DfBasedEstimatorBaseInput(df=df))

        # Assert: verify sum is correct (1+2+3+4+5 = 15)
        self.assertEqual(result.estimate_val, 15.0)
        self.assertEqual(result.sample_size, 5)
        self.assertEqual(result.confidence_interval_level, 0.95)

    def test_sum_estimate_with_weights(self) -> None:
        # Setup: create a dataframe with weights
        df = pd.DataFrame({"metric": [1.0, 2.0, 3.0], "weight": [2.0, 3.0, 4.0]})
        estimator = WeightedSumEstimator(confidence_level=0.95, weight_column="weight")

        # Execute: compute the estimator
        result = estimator.compute_estimator(DfBasedEstimatorBaseInput(df=df))

        # Assert: verify weighted sum is correct (1*2 + 2*3 + 3*4 = 2 + 6 + 12 = 20)
        self.assertEqual(result.estimate_val, 20.0)
        self.assertEqual(result.sample_size, 3)

    def test_standard_error_without_weights(self) -> None:
        # Setup: create a dataframe with known variance
        df = pd.DataFrame({"metric": [1.0, 2.0, 3.0, 4.0, 5.0]})
        estimator = WeightedSumEstimator(confidence_level=0.95)

        # Execute: compute the estimator
        result = estimator.compute_estimator(DfBasedEstimatorBaseInput(df=df))

        # Assert: verify standard error calculation
        # For sum estimator: std_error = sqrt(n * var(metric))
        # var([1, 2, 3, 4, 5]) = 2.5 (sample variance)
        # std_error = sqrt(5 * 2.5) = sqrt(12.5) ≈ 3.5355
        expected_variance = df["metric"].var()
        expected_std_error = np.sqrt(5 * expected_variance)
        self.assertAlmostEqual(result.standard_error, expected_std_error, places=4)

    def test_standard_error_with_weights(self) -> None:
        # Setup: create a dataframe with weights
        df = pd.DataFrame({"metric": [1.0, 2.0, 3.0], "weight": [2.0, 3.0, 4.0]})
        estimator = WeightedSumEstimator(confidence_level=0.95, weight_column="weight")

        # Execute: compute the estimator
        result = estimator.compute_estimator(DfBasedEstimatorBaseInput(df=df))

        # Assert: verify standard error calculation
        # weighted_metric = [1*2, 2*3, 3*4] = [2, 6, 12]
        # var([2, 6, 12]) = 26.0 (sample variance)
        # std_error = sqrt(3 * 26.0) = sqrt(78.0) ≈ 8.8318
        weighted_metrics = df["metric"] * df["weight"]
        expected_variance = weighted_metrics.var()
        expected_std_error = np.sqrt(3 * expected_variance)
        self.assertAlmostEqual(result.standard_error, expected_std_error, places=4)

    def test_confidence_intervals_calculated(self) -> None:
        # Setup: create a dataframe
        df = pd.DataFrame({"metric": [10.0, 20.0, 30.0]})
        estimator = WeightedSumEstimator(confidence_level=0.95)

        # Execute: compute the estimator
        result = estimator.compute_estimator(DfBasedEstimatorBaseInput(df=df))

        # Assert: verify confidence intervals are calculated
        # The lower_bound should be less than estimate_val
        # The upper_bound should be greater than estimate_val
        self.assertIsNotNone(result.lower_bound)
        self.assertIsNotNone(result.upper_bound)
        self.assertLess(result.lower_bound, result.estimate_val)
        self.assertGreater(result.upper_bound, result.estimate_val)

    def test_empty_dataframe(self) -> None:
        # Setup: create an empty dataframe
        df = pd.DataFrame({"metric": []})
        estimator = WeightedSumEstimator(confidence_level=0.95)

        # Execute: compute the estimator
        result = estimator.compute_estimator(DfBasedEstimatorBaseInput(df=df))

        # Assert: verify sum is 0 for empty dataframe
        self.assertEqual(result.estimate_val, 0.0)
        self.assertEqual(result.sample_size, 0)

    def test_single_value(self) -> None:
        # Setup: create a dataframe with a single value
        df = pd.DataFrame({"metric": [42.0]})
        estimator = WeightedSumEstimator(confidence_level=0.95)

        # Execute: compute the estimator
        result = estimator.compute_estimator(DfBasedEstimatorBaseInput(df=df))

        # Assert: verify sum is correct and variance is 0 for single value
        self.assertEqual(result.estimate_val, 42.0)
        self.assertEqual(result.sample_size, 1)
        # For a single value, variance is NaN, so std_error should be NaN
        self.assertTrue(np.isnan(result.standard_error))

    def test_zero_weights(self) -> None:
        # Setup: create a dataframe with zero weights
        df = pd.DataFrame({"metric": [1.0, 2.0, 3.0], "weight": [0.0, 0.0, 0.0]})
        estimator = WeightedSumEstimator(confidence_level=0.95, weight_column="weight")

        # Execute: compute the estimator
        result = estimator.compute_estimator(DfBasedEstimatorBaseInput(df=df))

        # Assert: verify weighted sum is 0 when all weights are 0
        self.assertEqual(result.estimate_val, 0.0)
        self.assertEqual(result.sample_size, 3)

    def test_negative_values(self) -> None:
        # Setup: create a dataframe with negative values
        df = pd.DataFrame({"metric": [-1.0, -2.0, 3.0]})
        estimator = WeightedSumEstimator(confidence_level=0.95)

        # Execute: compute the estimator
        result = estimator.compute_estimator(DfBasedEstimatorBaseInput(df=df))

        # Assert: verify sum handles negative values correctly (-1 + -2 + 3 = 0)
        self.assertEqual(result.estimate_val, 0.0)
        self.assertEqual(result.sample_size, 3)
