# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
Common transformations for unit level data.

This module provides a set of commonly used transformations that can be applied
to unit level data before aggregation in the proxymate validation framework.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from proxymate.lib.base_classes.transformation_base import TransformationBase


class IdentityTransformation(TransformationBase):
    """
    An identity transformation that returns the input unchanged.

    Useful as a default/no-op transformation when a transformation is required
    but no actual transformation should be applied.
    """

    def get_name(self) -> str:
        return "Identity Transformation"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()


class LogTransformation(TransformationBase):
    """
    Applies log transformation to outcomes.

    Useful for skewed data or when working with ratio/percentage changes.
    Handles zero values by adding an offset before taking the log.
    Supports both distribution check data (primary_outcome/model_outcome columns)
    and estimator data (metric column).
    """

    _offset: float
    _apply_to_primary: bool
    _apply_to_proxy: bool

    def __init__(
        self,
        offset: float = 1.0,
        apply_to_primary: bool = True,
        apply_to_proxy: bool = True,
    ) -> None:
        """
        Initialize the log transformation.

        Args:
            offset: Value to add before taking log (prevents log(0))
            apply_to_primary: Whether to apply to primary_outcome (or metric column)
            apply_to_proxy: Whether to apply to model_outcome
        """
        self._offset = offset
        self._apply_to_primary = apply_to_primary
        self._apply_to_proxy = apply_to_proxy

    def get_name(self) -> str:
        return "Log Transformation"

    def validate_input(self, df: pd.DataFrame) -> None:
        self._validate_columns_for_flags(
            df, self._apply_to_primary, self._apply_to_proxy
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._transform_columns(
            df,
            lambda s: np.log(s + self._offset),
            self._apply_to_primary,
            self._apply_to_proxy,
        )


class ClipTransformation(TransformationBase):
    """
    Clips outcomes to specified absolute bounds.

    Useful for constraining values to a known valid range.
    Supports both distribution check data (primary_outcome/model_outcome columns)
    and estimator data (metric column).
    """

    _lower_bound: Optional[float]
    _upper_bound: Optional[float]
    _apply_to_primary: bool
    _apply_to_proxy: bool

    def __init__(
        self,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        apply_to_primary: bool = True,
        apply_to_proxy: bool = True,
    ) -> None:
        """
        Initialize the clip transformation.

        Args:
            lower_bound: Absolute lower bound for clipping
            upper_bound: Absolute upper bound for clipping
            apply_to_primary: Whether to apply to primary_outcome (or metric column)
            apply_to_proxy: Whether to apply to model_outcome
        """
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._apply_to_primary = apply_to_primary
        self._apply_to_proxy = apply_to_proxy

    def get_name(self) -> str:
        return "Clip Transformation"

    def validate_input(self, df: pd.DataFrame) -> None:
        self._validate_columns_for_flags(
            df, self._apply_to_primary, self._apply_to_proxy
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._transform_columns(
            df,
            lambda s: s.clip(lower=self._lower_bound, upper=self._upper_bound),
            self._apply_to_primary,
            self._apply_to_proxy,
        )


class WinsorizeTransformation(TransformationBase):
    """
    Winsorizes outcomes by clipping to percentile-based bounds.

    Computes clipping thresholds from the data distribution using the specified
    percentiles, then clips values to those bounds. Useful for handling outliers
    without requiring prior knowledge of the data range.
    Supports both distribution check data (primary_outcome/model_outcome columns)
    and estimator data (metric column).
    """

    _percentile_lower: Optional[float]
    _percentile_upper: Optional[float]
    _apply_to_primary: bool
    _apply_to_proxy: bool

    def __init__(
        self,
        percentile_lower: Optional[float] = None,
        percentile_upper: Optional[float] = None,
        apply_to_primary: bool = True,
        apply_to_proxy: bool = True,
    ) -> None:
        """
        Initialize the winsorize transformation.

        Args:
            percentile_lower: Lower percentile for clipping (0-100)
            percentile_upper: Upper percentile for clipping (0-100)
            apply_to_primary: Whether to apply to primary_outcome (or metric column)
            apply_to_proxy: Whether to apply to model_outcome
        """
        self._percentile_lower = percentile_lower
        self._percentile_upper = percentile_upper
        self._apply_to_primary = apply_to_primary
        self._apply_to_proxy = apply_to_proxy

    def get_name(self) -> str:
        return "Winsorize Transformation"

    def validate_input(self, df: pd.DataFrame) -> None:
        self._validate_columns_for_flags(
            df, self._apply_to_primary, self._apply_to_proxy
        )

    def _get_bounds(self, series: pd.Series) -> Tuple[Optional[float], Optional[float]]:
        """Calculate bounds from percentiles of the data distribution."""
        lower = None
        upper = None

        if self._percentile_lower is not None:
            lower = float(np.percentile(series.dropna(), self._percentile_lower))

        if self._percentile_upper is not None:
            upper = float(np.percentile(series.dropna(), self._percentile_upper))

        return lower, upper

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        def winsorize(series: pd.Series) -> pd.Series:
            lower, upper = self._get_bounds(series)
            return series.clip(lower=lower, upper=upper)

        return self._transform_columns(
            df, winsorize, self._apply_to_primary, self._apply_to_proxy
        )


class StandardizeTransformation(TransformationBase):
    """
    Standardizes outcomes to have mean 0 and standard deviation 1.

    Useful for normalizing data before comparison.
    Supports both distribution check data (primary_outcome/model_outcome columns)
    and estimator data (metric column).
    """

    _apply_to_primary: bool
    _apply_to_proxy: bool

    def __init__(
        self,
        apply_to_primary: bool = True,
        apply_to_proxy: bool = True,
    ) -> None:
        self._apply_to_primary = apply_to_primary
        self._apply_to_proxy = apply_to_proxy

    def get_name(self) -> str:
        return "Standardize Transformation"

    def validate_input(self, df: pd.DataFrame) -> None:
        self._validate_columns_for_flags(
            df, self._apply_to_primary, self._apply_to_proxy
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        def standardize(series: pd.Series) -> pd.Series:
            mean = series.mean()
            std = series.std()
            if std > 0:
                return (series - mean) / std
            return series

        return self._transform_columns(
            df, standardize, self._apply_to_primary, self._apply_to_proxy
        )


class MinMaxScaleTransformation(TransformationBase):
    """
    Scales outcomes to a specified range (default [0, 1]).

    Useful for normalizing data to a consistent scale.
    Supports both distribution check data (primary_outcome/model_outcome columns)
    and estimator data (metric column).
    """

    _min_val: float
    _max_val: float
    _apply_to_primary: bool
    _apply_to_proxy: bool

    def __init__(
        self,
        min_val: float = 0.0,
        max_val: float = 1.0,
        apply_to_primary: bool = True,
        apply_to_proxy: bool = True,
    ) -> None:
        self._min_val = min_val
        self._max_val = max_val
        self._apply_to_primary = apply_to_primary
        self._apply_to_proxy = apply_to_proxy

    def get_name(self) -> str:
        return "Min-Max Scale Transformation"

    def validate_input(self, df: pd.DataFrame) -> None:
        self._validate_columns_for_flags(
            df, self._apply_to_primary, self._apply_to_proxy
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        def scale(series: pd.Series) -> pd.Series:
            s_min = series.min()
            s_max = series.max()
            # If all values are constant, return unchanged to avoid division by zero
            if s_max - s_min > 0:
                return (series - s_min) / (s_max - s_min) * (
                    self._max_val - self._min_val
                ) + self._min_val
            return series

        return self._transform_columns(
            df, scale, self._apply_to_primary, self._apply_to_proxy
        )


class ThresholdTransformation(TransformationBase):
    """
    Binarizes outcomes based on a threshold.

    Values above the threshold become 1, values at or below become 0.
    Useful for converting continuous outcomes to binary.
    Supports both distribution check data (primary_outcome/model_outcome columns)
    and estimator data (metric column).
    """

    _threshold: float
    _apply_to_primary: bool
    _apply_to_proxy: bool

    def __init__(
        self,
        threshold: float = 0.5,
        apply_to_primary: bool = True,
        apply_to_proxy: bool = True,
    ) -> None:
        self._threshold = threshold
        self._apply_to_primary = apply_to_primary
        self._apply_to_proxy = apply_to_proxy

    def get_name(self) -> str:
        return "Threshold Transformation"

    def validate_input(self, df: pd.DataFrame) -> None:
        self._validate_columns_for_flags(
            df, self._apply_to_primary, self._apply_to_proxy
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._transform_columns(
            df,
            lambda s: (s > self._threshold).astype(float),
            self._apply_to_primary,
            self._apply_to_proxy,
        )
