# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
Example transformation implementations for testing the transformation base classes.
"""

from typing import Set

import pandas as pd
from proxymate.lib.base_classes.transformation_base import TransformationBase


class DoubleValuesTransformation(TransformationBase):
    """
    A simple test transformation that doubles the primary and model outcomes.
    """

    def get_name(self) -> str:
        return "Double Values Transformation"

    def get_required_columns(self) -> Set[str]:
        return {"primary_outcome", "model_outcome"}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["primary_outcome"] = result["primary_outcome"] * 2
        result["model_outcome"] = result["model_outcome"] * 2
        return result


class AddOffsetTransformation(TransformationBase):
    """
    A test transformation that adds an offset to outcomes.
    """

    _offset: float

    def __init__(self, offset: float = 10.0) -> None:
        self._offset = offset

    def get_name(self) -> str:
        return f"Add Offset ({self._offset}) Transformation"

    def get_required_columns(self) -> Set[str]:
        return {"primary_outcome", "model_outcome"}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["primary_outcome"] = result["primary_outcome"] + self._offset
        result["model_outcome"] = result["model_outcome"] + self._offset
        return result


class CustomColumnTransformation(TransformationBase):
    """
    A test transformation that requires a custom column.
    """

    _custom_column: str

    def __init__(self, custom_column: str = "custom_col") -> None:
        self._custom_column = custom_column

    def get_name(self) -> str:
        return "Custom Column Transformation"

    def get_additional_columns(self) -> Set[str]:
        return {self._custom_column}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        # Add the custom column value to the outcomes
        result["primary_outcome"] = (
            result["primary_outcome"] + result[self._custom_column]
        )
        result["model_outcome"] = result["model_outcome"] + result[self._custom_column]
        return result


def create_test_dataframe() -> pd.DataFrame:
    """
    Creates a simple test DataFrame with standard columns.
    """
    return pd.DataFrame(
        {
            "primary_outcome": [1.0, 2.0, 3.0, 4.0, 5.0],
            "model_outcome": [1.1, 2.1, 3.1, 4.1, 5.1],
            "unit_id": ["a", "b", "c", "d", "e"],
        }
    )


def create_test_dataframe_with_custom_column() -> pd.DataFrame:
    """
    Creates a test DataFrame with a custom column.
    """
    df = create_test_dataframe()
    df["custom_col"] = [0.1, 0.2, 0.3, 0.4, 0.5]
    return df


def create_test_dataframe_with_outliers() -> pd.DataFrame:
    """
    Creates a test DataFrame with outliers.
    """
    return pd.DataFrame(
        {
            "primary_outcome": [1.0, 2.0, 3.0, 100.0, 5.0],
            "model_outcome": [1.1, 2.1, 3.1, -50.0, 5.1],
            "unit_id": ["a", "b", "c", "d", "e"],
        }
    )


def create_test_dataframe_for_scaling() -> pd.DataFrame:
    """
    Creates a test DataFrame suitable for scaling transformations.
    """
    return pd.DataFrame(
        {
            "primary_outcome": [0.0, 25.0, 50.0, 75.0, 100.0],
            "model_outcome": [0.0, 50.0, 100.0, 150.0, 200.0],
            "unit_id": ["a", "b", "c", "d", "e"],
        }
    )


def create_estimator_dataframe() -> pd.DataFrame:
    """
    Creates a test DataFrame with estimator data format (metric column).
    This simulates the data structure used by DfBasedEstimator.
    """
    return pd.DataFrame(
        {
            "metric": [1.0, 2.0, 3.0, 4.0, 5.0],
            "model_name": ["model1"] * 5,
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
            "row_id": ["a", "b", "c", "d", "e"],
        }
    )


def create_estimator_dataframe_with_outliers() -> pd.DataFrame:
    """
    Creates an estimator-format DataFrame with outliers in the metric column.
    """
    return pd.DataFrame(
        {
            "metric": [1.0, 2.0, 100.0, 4.0, -50.0],
            "model_name": ["model1"] * 5,
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
            "row_id": ["a", "b", "c", "d", "e"],
        }
    )
