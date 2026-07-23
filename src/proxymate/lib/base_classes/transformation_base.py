# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterator, List, Optional, Set, Union

import pandas as pd

# Type alias for transformations that can be applied uniformly or per-segment
# - List[TransformationBase]: Same transformations applied to all segments
# - Dict[str, Dict[str, List[TransformationBase]]]: Different transformations per
#   segment group and segment. Structure:
#   {segment_group_name: {segment_name: [transformations]}}
TransformationsType = Union[
    List["TransformationBase"],
    Dict[str, Dict[str, List["TransformationBase"]]],
    None,
]


class TransformationBase(ABC):
    """
    Abstract base class for defining transformations on unit level data.

    Transformations are applied to unit level data before aggregation.
    They can modify the primary_outcome, model_outcome, or other columns
    in the DataFrame.

    Similar to EstimatorBase, this class provides a template for implementing
    specific transformations that can be applied to the data.

    The column names used for primary and proxy outcomes are configurable via
    ``apply()``. By default they use v1 column names (``primary_outcome`` /
    ``model_outcome``), but v2 callers can pass arbitrary column names so
    that transformations work with the dynamic names from ``DataHandler``.
    """

    _primary_col: str = "primary_outcome"
    _proxy_col: str = "model_outcome"

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the name of the transformation.
        """
        pass

    def _get_columns_to_transform(
        self, apply_to_primary: bool, apply_to_proxy: bool
    ) -> List[str]:
        """
        Helper method to determine which columns to transform based on flags.

        Uses ``self._primary_col`` and ``self._proxy_col`` which are set by
        ``apply()`` before ``transform()`` is called.

        Args:
            apply_to_primary: Whether to apply to primary outcome or metric column
            apply_to_proxy: Whether to apply to proxy outcome column

        Returns:
            List of column names to transform
        """
        columns = []
        if apply_to_primary:
            columns.append(self._primary_col)
            # Also include "metric" when using v1 default column names
            if self._primary_col == "primary_outcome":
                columns.append("metric")
        if apply_to_proxy:
            columns.append(self._proxy_col)
        return columns

    def _validate_columns_for_flags(
        self, df: pd.DataFrame, apply_to_primary: bool, apply_to_proxy: bool
    ) -> None:
        """
        Validates that required columns exist based on apply_to_primary and
        apply_to_proxy flags.

        This enforces strict validation: if a flag is True, at least one of the
        corresponding columns must exist in the DataFrame.

        Args:
            df: DataFrame to validate
            apply_to_primary: Whether transformation should apply to primary outcome
            apply_to_proxy: Whether transformation should apply to proxy outcome

        Raises:
            ValueError: If required columns are missing based on the flags
        """
        if apply_to_primary:
            if self._primary_col == "primary_outcome":
                # v1 compatibility: accept either primary_outcome or metric
                has_primary = "primary_outcome" in df.columns or "metric" in df.columns
                if not has_primary:
                    raise ValueError(
                        f"Transformation '{self.get_name()}' has "
                        "apply_to_primary=True but neither 'primary_outcome' nor "
                        "'metric' column is present in the DataFrame."
                    )
            else:
                has_primary = self._primary_col in df.columns
                if not has_primary:
                    raise ValueError(
                        f"Transformation '{self.get_name()}' has "
                        "apply_to_primary=True but "
                        f"'{self._primary_col}' column is not present in the DataFrame."
                    )

        if apply_to_proxy:
            if self._proxy_col not in df.columns:
                raise ValueError(
                    f"Transformation '{self.get_name()}' has apply_to_proxy=True but "
                    f"'{self._proxy_col}' column is not present in the DataFrame."
                )

    def _transform_columns(
        self,
        df: pd.DataFrame,
        transform_fn: Callable[[pd.Series], pd.Series],
        apply_to_primary: bool,
        apply_to_proxy: bool,
    ) -> pd.DataFrame:
        """
        Apply a transformation function to the appropriate columns.

        This helper eliminates the repeated pattern of copying the DataFrame,
        determining target columns, and applying a function to each.

        Args:
            df: Input DataFrame.
            transform_fn: Function to apply to each target column's Series.
            apply_to_primary: Whether to apply to primary_outcome/metric columns.
            apply_to_proxy: Whether to apply to model_outcome column.

        Returns:
            Transformed DataFrame.
        """
        result = df.copy()
        for col in self._get_columns_to_transform(apply_to_primary, apply_to_proxy):
            if col in result.columns:
                result[col] = transform_fn(result[col])
        return result

    def get_required_columns(self) -> Set[str]:
        """
        Returns the set of columns required for this transformation.
        Default is an empty set, allowing transformations to dynamically
        adapt to available columns.

        Override this method if your transformation requires specific columns
        to be present.

        Note: Returns a set to avoid duplicates when combining required columns
        from multiple transformations.
        """
        return set()

    def get_additional_columns(self) -> Set[str]:
        """
        Returns additional columns needed beyond the required columns.
        Override this method if your transformation needs extra columns.

        Note: Returns a set to avoid duplicates when combining additional columns
        from multiple transformations.
        """
        return set()

    def validate_input(self, df: pd.DataFrame) -> None:
        """
        Validates that the input DataFrame contains required columns.

        Args:
            df: DataFrame to validate.

        Raises:
            ValueError: If required columns are missing from the DataFrame.
        """
        required_cols = self.get_required_columns() | self.get_additional_columns()
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Transformation '{self.get_name()}' requires columns {missing_cols} "
                f"which are missing from the input DataFrame."
            )

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the transformation to the input data.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed DataFrame.
        """
        pass

    def apply(
        self,
        df: pd.DataFrame,
        primary_col: str = "primary_outcome",
        proxy_col: str = "model_outcome",
    ) -> pd.DataFrame:
        """
        Validate input and apply the transformation.

        This is the main entry point for applying transformations.
        It sets column names on ``self`` so that helpers like
        ``_get_columns_to_transform`` and ``_validate_columns_for_flags``
        use the correct names, then validates and transforms.

        Args:
            df: DataFrame to transform.
            primary_col: Name of the primary outcome column.
            proxy_col: Name of the proxy outcome column.

        Returns:
            Transformed DataFrame.
        """
        self._primary_col = primary_col
        self._proxy_col = proxy_col
        self.validate_input(df)
        return self.transform(df)


class TransformationPipeline(TransformationBase):
    """
    A pipeline that chains multiple transformations together.

    Transformations are applied in the order they are provided.
    If no transformations are provided, an identity transformation is used.
    """

    _transformations: List[TransformationBase]

    def __init__(
        self, transformations: Optional[List[TransformationBase]] = None
    ) -> None:
        """
        Initialize the transformation pipeline.

        Args:
            transformations: List of transformations to apply in order.
                           If None or empty, no transformations are applied.
        """
        self._transformations = transformations or []

    def get_name(self) -> str:
        if not self._transformations:
            return "Identity (No Transformation)"
        names = [t.get_name() for t in self._transformations]
        return " -> ".join(names)

    def get_required_columns(self) -> Set[str]:
        """Returns union of all required columns from child transformations."""
        return {col for t in self._transformations for col in t.get_required_columns()}

    def get_additional_columns(self) -> Set[str]:
        """Returns union of all additional columns from child transformations."""
        return {
            col for t in self._transformations for col in t.get_additional_columns()
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformations in sequence.

        Forwards ``self._primary_col`` and ``self._proxy_col`` to each
        child transformation so column names propagate through the pipeline.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed DataFrame.
        """
        result = df.copy()
        for transformation in self._transformations:
            result = transformation.apply(
                result,
                primary_col=self._primary_col,
                proxy_col=self._proxy_col,
            )
        return result

    def add_transformation(self, transformation: TransformationBase) -> None:
        """
        Add a transformation to the end of the pipeline.

        Args:
            transformation: Transformation to add.
        """
        self._transformations.append(transformation)

    def __len__(self) -> int:
        return len(self._transformations)

    def __iter__(self) -> Iterator[TransformationBase]:
        return iter(self._transformations)


def apply_transformations(
    df: pd.DataFrame,
    transformations: Optional[List[TransformationBase]] = None,
    primary_col: str = "primary_outcome",
    proxy_col: str = "model_outcome",
) -> pd.DataFrame:
    """
    Apply a list of transformations to a DataFrame.

    This is a convenience function for applying transformations in the
    validator.py flow. If transformations is None or empty,
    returns a copy of the input DataFrame (identity transformation).

    Args:
        df: Input DataFrame to transform.
        transformations: Optional list of transformations to apply in order.
        primary_col: Name of the primary outcome column.
        proxy_col: Name of the proxy outcome column.

    Returns:
        Transformed DataFrame (or copy of input if no transformations).
    """
    if not transformations:
        return df.copy()

    pipeline = TransformationPipeline(transformations)
    return pipeline.apply(df, primary_col=primary_col, proxy_col=proxy_col)


def get_transformations_for_segment(
    transformations: TransformationsType,
    segment_group_name: str,
    segment_name: str,
) -> Optional[List[TransformationBase]]:
    """
    Get the list of transformations to apply for a specific segment.

    This supports both uniform transformations (same for all segments)
    and per-segment transformations (different for each segment group/segment).

    Args:
        transformations: Either:
            - None: No transformations
            - List[TransformationBase]: Same transformations for all segments
            - Dict[str, Dict[str, List[TransformationBase]]]:
              Per-segment transformations
              with structure {segment_group_name: {segment_name: [transformations]}}
        segment_group_name: The name of the segment group.
        segment_name: The name of the segment within the group.

    Returns:
        List of transformations to apply, or None if no transformations.
    """
    if transformations is None:
        return None

    if isinstance(transformations, dict):
        # Per-segment transformations - navigate the nested structure
        segment_group_transforms = transformations.get(segment_group_name, {})
        if isinstance(segment_group_transforms, dict):
            return segment_group_transforms.get(segment_name, None)
        return None
    else:
        # Uniform transformations - same list for all segments
        return transformations
