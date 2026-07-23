# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
Tests for the common transformation implementations.
"""

from unittest import TestCase

import numpy as np
import pandas as pd
from proxymate.lib.base_classes.tests.utils.transformation_examples import (
    create_test_dataframe,
    create_test_dataframe_for_scaling,
    create_test_dataframe_with_outliers,
)
from proxymate.lib.base_classes.transformation_base import TransformationPipeline
from proxymate.transformations.common_transformations import (
    ClipTransformation,
    IdentityTransformation,
    LogTransformation,
    MinMaxScaleTransformation,
    StandardizeTransformation,
    ThresholdTransformation,
    WinsorizeTransformation,
)


class TestIdentityTransformation(TestCase):
    def test_returns_unchanged_copy(self) -> None:
        df = create_test_dataframe()
        transformation = IdentityTransformation()

        output = transformation.apply(df)

        pd.testing.assert_frame_equal(output, df)
        # Should be a copy, not the same object
        self.assertIsNot(output, df)

    def test_get_name(self) -> None:
        transformation = IdentityTransformation()
        self.assertEqual(transformation.get_name(), "Identity Transformation")


class TestLogTransformation(TestCase):
    def test_applies_log_with_default_offset(self) -> None:
        df = create_test_dataframe()
        transformation = LogTransformation()

        output = transformation.apply(df)

        # log(1.0 + 1.0) = log(2.0) ≈ 0.693
        self.assertAlmostEqual(output["primary_outcome"].iloc[0], np.log(2.0), places=5)
        # log(1.1 + 1.0) = log(2.1) ≈ 0.742
        self.assertAlmostEqual(output["model_outcome"].iloc[0], np.log(2.1), places=5)

    def test_applies_log_with_custom_offset(self) -> None:
        df = create_test_dataframe()
        transformation = LogTransformation(offset=0.5)

        output = transformation.apply(df)

        # log(1.0 + 0.5) = log(1.5)
        self.assertAlmostEqual(output["primary_outcome"].iloc[0], np.log(1.5), places=5)

    def test_applies_to_primary_only(self) -> None:
        df = create_test_dataframe()
        original_model = df["model_outcome"].iloc[0]
        transformation = LogTransformation(apply_to_primary=True, apply_to_proxy=False)

        output = transformation.apply(df)

        # Primary should be transformed
        self.assertAlmostEqual(output["primary_outcome"].iloc[0], np.log(2.0), places=5)
        # Model should be unchanged
        self.assertAlmostEqual(output["model_outcome"].iloc[0], original_model)

    def test_applies_to_proxy_only(self) -> None:
        df = create_test_dataframe()
        original_primary = df["primary_outcome"].iloc[0]
        transformation = LogTransformation(apply_to_primary=False, apply_to_proxy=True)

        output = transformation.apply(df)

        # Primary should be unchanged
        self.assertAlmostEqual(output["primary_outcome"].iloc[0], original_primary)
        # Model should be transformed
        self.assertAlmostEqual(output["model_outcome"].iloc[0], np.log(2.1), places=5)


class TestClipTransformation(TestCase):
    def test_clips_with_absolute_bounds(self) -> None:
        df = create_test_dataframe_with_outliers()
        # This test expects both primary_outcome and model_outcome to be clipped
        transformation = ClipTransformation(
            lower_bound=0.0,
            upper_bound=10.0,
            apply_to_primary=True,
            apply_to_proxy=True,
        )

        output = transformation.apply(df)

        # 100.0 should be clipped to 10.0
        self.assertEqual(output["primary_outcome"].max(), 10.0)
        # -50.0 should be clipped to 0.0
        self.assertEqual(output["model_outcome"].min(), 0.0)

    def test_clips_primary_only(self) -> None:
        df = create_test_dataframe_with_outliers()
        original_model_max = df["model_outcome"].max()
        transformation = ClipTransformation(
            upper_bound=10.0, apply_to_primary=True, apply_to_proxy=False
        )

        output = transformation.apply(df)

        # Primary should be clipped
        self.assertEqual(output["primary_outcome"].max(), 10.0)
        # Model should be unchanged
        self.assertEqual(output["model_outcome"].max(), original_model_max)


class TestWinsorizeTransformation(TestCase):
    def test_winsorizes_with_percentile_bounds(self) -> None:
        df = create_test_dataframe_with_outliers()
        # Winsorize at 5th and 95th percentile
        transformation = WinsorizeTransformation(
            percentile_lower=5.0, percentile_upper=95.0
        )

        output = transformation.apply(df)

        # Values should be within bounds
        self.assertLessEqual(
            output["primary_outcome"].max(), df["primary_outcome"].max()
        )
        self.assertGreaterEqual(
            output["model_outcome"].min(), df["model_outcome"].min()
        )


class TestStandardizeTransformation(TestCase):
    def test_standardizes_to_mean_zero_std_one(self) -> None:
        df = create_test_dataframe()
        transformation = StandardizeTransformation()

        output = transformation.apply(df)

        # Mean should be approximately 0
        self.assertAlmostEqual(output["primary_outcome"].mean(), 0.0, places=10)
        self.assertAlmostEqual(output["model_outcome"].mean(), 0.0, places=10)

        # Std should be approximately 1
        self.assertAlmostEqual(output["primary_outcome"].std(), 1.0, places=10)
        self.assertAlmostEqual(output["model_outcome"].std(), 1.0, places=10)

    def test_standardizes_primary_only(self) -> None:
        df = create_test_dataframe()
        original_model_mean = df["model_outcome"].mean()
        transformation = StandardizeTransformation(
            apply_to_primary=True, apply_to_proxy=False
        )

        output = transformation.apply(df)

        # Primary should be standardized
        self.assertAlmostEqual(output["primary_outcome"].mean(), 0.0, places=10)
        # Model should be unchanged
        self.assertAlmostEqual(
            output["model_outcome"].mean(), original_model_mean, places=10
        )

    def test_handles_zero_std(self) -> None:
        df = pd.DataFrame(
            {
                "primary_outcome": [5.0, 5.0, 5.0],
                "model_outcome": [3.0, 3.0, 3.0],
            }
        )
        transformation = StandardizeTransformation()

        output = transformation.apply(df)

        # With zero std, values should remain unchanged
        self.assertEqual(output["primary_outcome"].iloc[0], 5.0)
        self.assertEqual(output["model_outcome"].iloc[0], 3.0)


class TestMinMaxScaleTransformation(TestCase):
    def test_scales_to_0_1_by_default(self) -> None:
        df = create_test_dataframe_for_scaling()
        transformation = MinMaxScaleTransformation()

        output = transformation.apply(df)

        # Min should be 0
        self.assertEqual(output["primary_outcome"].min(), 0.0)
        self.assertEqual(output["model_outcome"].min(), 0.0)
        # Max should be 1
        self.assertEqual(output["primary_outcome"].max(), 1.0)
        self.assertEqual(output["model_outcome"].max(), 1.0)

    def test_scales_to_custom_range(self) -> None:
        df = create_test_dataframe_for_scaling()
        transformation = MinMaxScaleTransformation(min_val=-1.0, max_val=1.0)

        output = transformation.apply(df)

        # Min should be -1
        self.assertEqual(output["primary_outcome"].min(), -1.0)
        # Max should be 1
        self.assertEqual(output["primary_outcome"].max(), 1.0)

    def test_handles_constant_values(self) -> None:
        df = pd.DataFrame(
            {
                "primary_outcome": [5.0, 5.0, 5.0],
                "model_outcome": [3.0, 3.0, 3.0],
            }
        )
        transformation = MinMaxScaleTransformation()

        output = transformation.apply(df)

        # With constant values, should remain unchanged
        self.assertEqual(output["primary_outcome"].iloc[0], 5.0)


class TestThresholdTransformation(TestCase):
    def test_binarizes_with_default_threshold(self) -> None:
        df = pd.DataFrame(
            {
                "primary_outcome": [0.2, 0.5, 0.8],
                "model_outcome": [0.3, 0.6, 0.9],
            }
        )
        transformation = ThresholdTransformation()

        output = transformation.apply(df)

        # Values > 0.5 become 1, values <= 0.5 become 0
        expected_primary = [0.0, 0.0, 1.0]
        expected_model = [0.0, 1.0, 1.0]

        for i in range(3):
            self.assertEqual(output["primary_outcome"].iloc[i], expected_primary[i])
            self.assertEqual(output["model_outcome"].iloc[i], expected_model[i])

    def test_binarizes_with_custom_threshold(self) -> None:
        df = pd.DataFrame(
            {
                "primary_outcome": [0.2, 0.5, 0.8],
                "model_outcome": [0.3, 0.6, 0.9],
            }
        )
        transformation = ThresholdTransformation(threshold=0.3)

        output = transformation.apply(df)

        # Values > 0.3 become 1
        expected_primary = [0.0, 1.0, 1.0]

        for i in range(3):
            self.assertEqual(output["primary_outcome"].iloc[i], expected_primary[i])

    def test_binarizes_primary_only(self) -> None:
        df = pd.DataFrame(
            {
                "primary_outcome": [0.2, 0.8],
                "model_outcome": [0.3, 0.7],
            }
        )
        transformation = ThresholdTransformation(
            apply_to_primary=True, apply_to_proxy=False
        )

        output = transformation.apply(df)

        # Primary should be binarized
        self.assertEqual(output["primary_outcome"].iloc[0], 0.0)
        self.assertEqual(output["primary_outcome"].iloc[1], 1.0)
        # Model should be unchanged
        self.assertEqual(output["model_outcome"].iloc[0], 0.3)
        self.assertEqual(output["model_outcome"].iloc[1], 0.7)


class TestTransformationIntegration(TestCase):
    """Integration tests combining multiple transformations."""

    def test_clip_then_log(self) -> None:
        """Common pattern: clip outliers then apply log."""
        df = create_test_dataframe_with_outliers()

        pipeline = TransformationPipeline(
            [
                ClipTransformation(lower_bound=0.5, upper_bound=10.0),
                LogTransformation(offset=0.0),
            ]
        )

        output = pipeline.apply(df)

        # All values should be log-transformed and finite
        self.assertTrue(np.all(np.isfinite(output["primary_outcome"])))
        self.assertTrue(np.all(np.isfinite(output["model_outcome"])))

    def test_standardize_then_threshold(self) -> None:
        """Pattern: standardize then binarize at mean."""
        df = create_test_dataframe()

        pipeline = TransformationPipeline(
            [
                StandardizeTransformation(),
                ThresholdTransformation(threshold=0.0),
            ]
        )

        output = pipeline.apply(df)

        # Values above mean become 1, below become 0
        unique_values = output["primary_outcome"].unique()
        self.assertTrue(all(v in [0.0, 1.0] for v in unique_values))


class TestEstimatorDataFormat(TestCase):
    """Tests for transformations working with estimator data format (metric column).

    These tests verify that transformations work correctly when applied to
    DataFrames from the estimator flow which use 'metric' column instead of
    'primary_outcome' and 'model_outcome'.
    """

    def test_clip_transformation_on_metric_column(self) -> None:
        """Test that ClipTransformation works on metric column."""
        df = pd.DataFrame(
            {
                "metric": [1.0, 2.0, 100.0, 4.0, -50.0],
                "model_name": ["model1"] * 5,
                "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
                "row_id": ["a", "b", "c", "d", "e"],
            }
        )
        # For estimator data (metric column), only apply to primary
        transformation = ClipTransformation(
            lower_bound=0.0, upper_bound=10.0, apply_to_proxy=False
        )

        output = transformation.apply(df)

        # Outlier 100.0 should be clipped to 10.0
        self.assertEqual(output["metric"].max(), 10.0)
        # Outlier -50.0 should be clipped to 0.0
        self.assertEqual(output["metric"].min(), 0.0)
        # Other columns should be unchanged
        self.assertEqual(list(output["model_name"]), ["model1"] * 5)

    def test_winsorize_transformation_with_percentiles_on_metric(self) -> None:
        """Test WinsorizeTransformation with percentile bounds on metric column."""
        df = pd.DataFrame(
            {
                "metric": [1.0, 2.0, 3.0, 4.0, 100.0],
                "weight": [1.0] * 5,
            }
        )
        # For estimator data (metric column), only apply to primary
        transformation = WinsorizeTransformation(
            percentile_lower=10.0, percentile_upper=90.0, apply_to_proxy=False
        )

        output = transformation.apply(df)

        # After clipping at percentiles, max should be reduced
        self.assertLess(output["metric"].max(), 100.0)

    def test_log_transformation_on_metric_column(self) -> None:
        """Test that LogTransformation works on metric column."""
        df = pd.DataFrame(
            {
                "metric": [1.0, 2.0, 3.0],
                "weight": [1.0, 1.0, 1.0],
            }
        )
        # For estimator data (metric column), only apply to primary
        transformation = LogTransformation(offset=1.0, apply_to_proxy=False)

        output = transformation.apply(df)

        # log(1.0 + 1.0) = log(2.0)
        self.assertAlmostEqual(output["metric"].iloc[0], np.log(2.0), places=5)
        # log(2.0 + 1.0) = log(3.0)
        self.assertAlmostEqual(output["metric"].iloc[1], np.log(3.0), places=5)

    def test_standardize_transformation_on_metric_column(self) -> None:
        """Test that StandardizeTransformation works on metric column."""
        df = pd.DataFrame(
            {
                "metric": [1.0, 2.0, 3.0, 4.0, 5.0],
                "weight": [1.0] * 5,
            }
        )
        # For estimator data (metric column), only apply to primary
        transformation = StandardizeTransformation(apply_to_proxy=False)

        output = transformation.apply(df)

        # Mean should be approximately 0
        self.assertAlmostEqual(output["metric"].mean(), 0.0, places=10)
        # Std should be approximately 1
        self.assertAlmostEqual(output["metric"].std(), 1.0, places=10)

    def test_minmax_transformation_on_metric_column(self) -> None:
        """Test that MinMaxScaleTransformation works on metric column."""
        df = pd.DataFrame(
            {
                "metric": [0.0, 25.0, 50.0, 75.0, 100.0],
                "weight": [1.0] * 5,
            }
        )
        # For estimator data (metric column), only apply to primary
        transformation = MinMaxScaleTransformation(
            min_val=0.0, max_val=1.0, apply_to_proxy=False
        )

        output = transformation.apply(df)

        # Min should be 0
        self.assertEqual(output["metric"].min(), 0.0)
        # Max should be 1
        self.assertEqual(output["metric"].max(), 1.0)
        # Middle value (50) should be 0.5
        self.assertEqual(output["metric"].iloc[2], 0.5)

    def test_threshold_transformation_on_metric_column(self) -> None:
        """Test that ThresholdTransformation works on metric column."""
        df = pd.DataFrame(
            {
                "metric": [0.2, 0.5, 0.8],
                "weight": [1.0] * 3,
            }
        )
        # For estimator data (metric column), only apply to primary
        transformation = ThresholdTransformation(threshold=0.5, apply_to_proxy=False)

        output = transformation.apply(df)

        # Values > 0.5 become 1, values <= 0.5 become 0
        self.assertEqual(output["metric"].iloc[0], 0.0)
        self.assertEqual(output["metric"].iloc[1], 0.0)
        self.assertEqual(output["metric"].iloc[2], 1.0)

    def test_identity_transformation_on_metric_column(self) -> None:
        """Test that IdentityTransformation works on metric column."""
        df = pd.DataFrame(
            {
                "metric": [1.0, 2.0, 3.0],
                "weight": [1.0] * 3,
            }
        )
        transformation = IdentityTransformation()

        output = transformation.apply(df)

        # Values should be unchanged
        pd.testing.assert_frame_equal(output, df)
        # But should be a copy
        self.assertIsNot(output, df)

    def test_pipeline_on_metric_column(self) -> None:
        """Test that TransformationPipeline works on metric column."""
        df = pd.DataFrame(
            {
                "metric": [1.0, 2.0, 100.0, 4.0, 5.0],
                "weight": [1.0] * 5,
            }
        )

        # Clip then standardize (for estimator data, only apply to primary)
        pipeline = TransformationPipeline(
            [
                ClipTransformation(
                    lower_bound=0.0, upper_bound=10.0, apply_to_proxy=False
                ),
                StandardizeTransformation(apply_to_proxy=False),
            ]
        )

        output = pipeline.apply(df)

        # After clipping and standardizing, mean should be 0
        self.assertAlmostEqual(output["metric"].mean(), 0.0, places=10)

    def test_transformation_does_not_affect_other_estimator_columns(self) -> None:
        """Test that transformations only modify metric column."""
        df = pd.DataFrame(
            {
                "metric": [1.0, 2.0, 100.0],
                "model_name": ["model1", "model2", "model3"],
                "weight": [0.5, 1.0, 1.5],
                "row_id": ["a", "b", "c"],
            }
        )
        # For distribution check data with only primary_outcome, only apply to primary
        transformation = ClipTransformation(
            lower_bound=0.0, upper_bound=10.0, apply_to_proxy=False
        )

        output = transformation.apply(df)

        # metric should be clipped
        self.assertEqual(output["metric"].max(), 10.0)
        # Other columns should be unchanged
        self.assertEqual(list(output["model_name"]), ["model1", "model2", "model3"])
        self.assertEqual(list(output["weight"]), [0.5, 1.0, 1.5])
        self.assertEqual(list(output["row_id"]), ["a", "b", "c"])

    def test_no_validation_error_on_estimator_data(self) -> None:
        """Test that transformations don't raise validation errors for
        estimator data."""
        df = pd.DataFrame(
            {
                "metric": [1.0, 2.0, 3.0],
                "model_name": ["model1"] * 3,
                "weight": [1.0] * 3,
            }
        )

        # These transformations should work with estimator data when
        # apply_to_proxy=False
        transformations = [
            ClipTransformation(lower_bound=0.0, upper_bound=10.0, apply_to_proxy=False),
            WinsorizeTransformation(
                percentile_lower=5.0, percentile_upper=95.0, apply_to_proxy=False
            ),
            LogTransformation(apply_to_proxy=False),
            StandardizeTransformation(apply_to_proxy=False),
            MinMaxScaleTransformation(apply_to_proxy=False),
            ThresholdTransformation(apply_to_proxy=False),
            IdentityTransformation(),
        ]

        for transformation in transformations:
            # Should not raise ValueError
            output = transformation.apply(df)
            self.assertIsNotNone(output)


class TestMixedColumnData(TestCase):
    """Tests for edge cases with mixed or missing columns."""

    def test_transformation_with_only_primary_outcome(self) -> None:
        """Test transformation when only primary_outcome is present."""
        df = pd.DataFrame(
            {
                "primary_outcome": [1.0, 2.0, 100.0],
                "other_col": ["a", "b", "c"],
            }
        )
        # For distribution check data with only primary_outcome, only apply to primary
        transformation = ClipTransformation(
            lower_bound=0.0,
            upper_bound=10.0,
            apply_to_primary=True,
            apply_to_proxy=False,
        )

        output = transformation.apply(df)

        # primary_outcome should be clipped
        self.assertEqual(output["primary_outcome"].max(), 10.0)
        # model_outcome should not be added
        self.assertNotIn("model_outcome", output.columns)

    def test_transformation_with_only_model_outcome(self) -> None:
        """Test transformation when only model_outcome is present."""
        df = pd.DataFrame(
            {
                "model_outcome": [1.0, 2.0, 100.0],
                "other_col": ["a", "b", "c"],
            }
        )
        # For distribution check data with only model_outcome, only apply to proxy
        transformation = ClipTransformation(
            lower_bound=0.0,
            upper_bound=10.0,
            apply_to_primary=False,
            apply_to_proxy=True,
        )

        output = transformation.apply(df)

        # model_outcome should be clipped
        self.assertEqual(output["model_outcome"].max(), 10.0)
        # primary_outcome should not be added
        self.assertNotIn("primary_outcome", output.columns)

    def test_transformation_with_no_applicable_columns(self) -> None:
        """Test transformation when no applicable columns are present."""
        df = pd.DataFrame(
            {
                "other_col1": [1.0, 2.0, 3.0],
                "other_col2": ["a", "b", "c"],
            }
        )
        # Since no outcome columns exist, this should raise an error with strict
        # validation
        transformation = ClipTransformation(lower_bound=0.0, upper_bound=10.0)

        with self.assertRaises(ValueError) as context:
            transformation.apply(df)

        # Verify the error message mentions the missing columns
        self.assertIn("primary_outcome", str(context.exception))
        self.assertIn("metric", str(context.exception))

    def test_transformation_with_all_column_types(self) -> None:
        """Test transformation when all column types are present."""
        df = pd.DataFrame(
            {
                "primary_outcome": [1.0, 2.0, 100.0],
                "model_outcome": [1.1, 2.1, -50.0],
                "metric": [1.5, 2.5, 200.0],
            }
        )
        transformation = ClipTransformation(lower_bound=0.0, upper_bound=10.0)

        output = transformation.apply(df)

        # All applicable columns should be clipped
        self.assertEqual(output["primary_outcome"].max(), 10.0)
        self.assertEqual(output["model_outcome"].min(), 0.0)
        self.assertEqual(output["metric"].max(), 10.0)
