# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
Tests for the transformation base classes and helper functions.
"""

from typing import List
from unittest import TestCase

import pandas as pd
from proxymate.lib.base_classes.tests.utils.transformation_examples import (
    AddOffsetTransformation,
    create_test_dataframe,
    create_test_dataframe_with_custom_column,
    CustomColumnTransformation,
    DoubleValuesTransformation,
)
from proxymate.lib.base_classes.transformation_base import (
    apply_transformations,
    get_transformations_for_segment,
    TransformationBase,
    TransformationPipeline,
    TransformationsType,
)


class TestTransformationBase(TestCase):
    def test_validate_input_with_missing_columns(self) -> None:
        df = pd.DataFrame({"primary_outcome": [1.0, 2.0, 3.0]})

        transformation = DoubleValuesTransformation()

        with self.assertRaises(ValueError) as context:
            transformation.validate_input(df)

        self.assertIn("model_outcome", str(context.exception))

    def test_validate_input_with_all_columns_present(self) -> None:
        df = create_test_dataframe()
        transformation = DoubleValuesTransformation()

        # Should not raise
        transformation.validate_input(df)

    def test_validate_input_with_custom_columns(self) -> None:
        df = create_test_dataframe()  # Missing custom_col

        transformation = CustomColumnTransformation(custom_column="custom_col")

        with self.assertRaises(ValueError) as context:
            transformation.validate_input(df)

        self.assertIn("custom_col", str(context.exception))

    def test_apply_validates_and_transforms(self) -> None:
        df = create_test_dataframe()
        transformation = DoubleValuesTransformation()

        output = transformation.apply(df)

        # Values should be doubled
        self.assertEqual(output["primary_outcome"].iloc[0], 2.0)
        self.assertEqual(output["model_outcome"].iloc[0], 2.2)

    def test_get_name(self) -> None:
        transformation = DoubleValuesTransformation()
        self.assertEqual(transformation.get_name(), "Double Values Transformation")

    def test_get_required_columns_default(self) -> None:
        transformation = DoubleValuesTransformation()
        required_cols = transformation.get_required_columns()

        self.assertIn("primary_outcome", required_cols)
        self.assertIn("model_outcome", required_cols)

    def test_get_additional_columns(self) -> None:
        transformation = CustomColumnTransformation(custom_column="my_column")
        additional_cols = transformation.get_additional_columns()

        self.assertIn("my_column", additional_cols)


class TestTransformationPipeline(TestCase):
    def test_empty_pipeline_is_identity(self) -> None:
        pipeline = TransformationPipeline()
        df = create_test_dataframe()

        output = pipeline.apply(df)

        # Should return a copy with same values
        pd.testing.assert_frame_equal(output, df)
        # But should be a copy, not the same object
        self.assertIsNot(output, df)

    def test_empty_pipeline_name(self) -> None:
        pipeline = TransformationPipeline()
        self.assertEqual(pipeline.get_name(), "Identity (No Transformation)")

    def test_single_transformation_pipeline(self) -> None:
        pipeline = TransformationPipeline([DoubleValuesTransformation()])
        df = create_test_dataframe()

        output = pipeline.apply(df)

        # Values should be doubled
        self.assertEqual(output["primary_outcome"].iloc[0], 2.0)
        self.assertEqual(output["model_outcome"].iloc[0], 2.2)

    def test_chained_transformations(self) -> None:
        # Double values, then add 10
        pipeline = TransformationPipeline(
            [
                DoubleValuesTransformation(),
                AddOffsetTransformation(offset=10.0),
            ]
        )
        df = create_test_dataframe()

        output = pipeline.apply(df)

        # 1.0 * 2 + 10 = 12.0
        self.assertEqual(output["primary_outcome"].iloc[0], 12.0)
        # 1.1 * 2 + 10 = 12.2
        self.assertAlmostEqual(output["model_outcome"].iloc[0], 12.2, places=5)

    def test_pipeline_name_with_transformations(self) -> None:
        pipeline = TransformationPipeline(
            [
                DoubleValuesTransformation(),
                AddOffsetTransformation(offset=10.0),
            ]
        )

        name = pipeline.get_name()

        self.assertIn("Double Values Transformation", name)
        self.assertIn("Add Offset", name)
        self.assertIn("->", name)

    def test_add_transformation(self) -> None:
        pipeline = TransformationPipeline()
        self.assertEqual(len(pipeline), 0)

        pipeline.add_transformation(DoubleValuesTransformation())
        self.assertEqual(len(pipeline), 1)

        pipeline.add_transformation(AddOffsetTransformation())
        self.assertEqual(len(pipeline), 2)

    def test_len_and_iter(self) -> None:
        transformations: List[TransformationBase] = [
            DoubleValuesTransformation(),
            AddOffsetTransformation(),
        ]
        pipeline = TransformationPipeline(transformations)

        self.assertEqual(len(pipeline), 2)

        iterated = list(pipeline)
        self.assertEqual(len(iterated), 2)
        self.assertIsInstance(iterated[0], DoubleValuesTransformation)
        self.assertIsInstance(iterated[1], AddOffsetTransformation)


class TestApplyTransformations(TestCase):
    def test_with_none_transformations(self) -> None:
        df = create_test_dataframe()

        result = apply_transformations(df, None)

        pd.testing.assert_frame_equal(result, df)
        self.assertIsNot(result, df)  # Should be a copy

    def test_with_empty_list(self) -> None:
        df = create_test_dataframe()

        result = apply_transformations(df, [])

        pd.testing.assert_frame_equal(result, df)
        self.assertIsNot(result, df)

    def test_with_single_transformation(self) -> None:
        df = create_test_dataframe()

        result = apply_transformations(df, [DoubleValuesTransformation()])

        self.assertEqual(result["primary_outcome"].iloc[0], 2.0)

    def test_with_multiple_transformations(self) -> None:
        df = create_test_dataframe()

        result = apply_transformations(
            df,
            [
                DoubleValuesTransformation(),
                AddOffsetTransformation(offset=5.0),
            ],
        )

        # 1.0 * 2 + 5 = 7.0
        self.assertEqual(result["primary_outcome"].iloc[0], 7.0)


class TestGetTransformationsForSegment(TestCase):
    def test_with_none_returns_none(self) -> None:
        result = get_transformations_for_segment(None, "group1", "segment1")
        self.assertIsNone(result)

    def test_with_list_returns_same_for_all_segments(self) -> None:
        transformations: List[TransformationBase] = [DoubleValuesTransformation()]

        result1 = get_transformations_for_segment(transformations, "group1", "segment1")
        result2 = get_transformations_for_segment(transformations, "group2", "segment2")

        self.assertEqual(result1, transformations)
        self.assertEqual(result2, transformations)

    def test_with_dict_returns_segment_specific(self) -> None:
        transform1: TransformationBase = DoubleValuesTransformation()
        transform2: TransformationBase = AddOffsetTransformation(offset=10.0)

        per_segment_transforms: TransformationsType = {
            "country": {
                "US": [transform1],
                "CA": [transform2],
            },
            "platform": {
                "iOS": [transform1, transform2],
            },
        }

        result_us = get_transformations_for_segment(
            per_segment_transforms, "country", "US"
        )
        result_ca = get_transformations_for_segment(
            per_segment_transforms, "country", "CA"
        )
        result_ios = get_transformations_for_segment(
            per_segment_transforms, "platform", "iOS"
        )

        self.assertEqual(result_us, [transform1])
        self.assertEqual(result_ca, [transform2])
        self.assertEqual(result_ios, [transform1, transform2])

    def test_with_dict_returns_none_for_missing_segment_group(self) -> None:
        per_segment_transforms: TransformationsType = {
            "country": {
                "US": [DoubleValuesTransformation()],
            },
        }

        result = get_transformations_for_segment(
            per_segment_transforms, "unknown_group", "US"
        )

        self.assertIsNone(result)

    def test_with_dict_returns_none_for_missing_segment(self) -> None:
        per_segment_transforms: TransformationsType = {
            "country": {
                "US": [DoubleValuesTransformation()],
            },
        }

        result = get_transformations_for_segment(
            per_segment_transforms, "country", "CA"
        )

        self.assertIsNone(result)


class TestCustomColumnTransformation(TestCase):
    def test_transform_with_custom_column(self) -> None:
        df = create_test_dataframe_with_custom_column()
        transformation = CustomColumnTransformation(custom_column="custom_col")

        output = transformation.apply(df)

        # 1.0 + 0.1 = 1.1
        self.assertAlmostEqual(output["primary_outcome"].iloc[0], 1.1, places=5)
        # 1.1 + 0.1 = 1.2
        self.assertAlmostEqual(output["model_outcome"].iloc[0], 1.2, places=5)
