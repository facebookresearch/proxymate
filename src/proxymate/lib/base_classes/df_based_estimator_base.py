# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from dataclasses import dataclass
from typing import final

import pandas as pd
from proxymate.lib.base_classes.estimator_base import (
    Estimate,
    EstimatorBase,
    EstimatorInput,
)


@dataclass
class DfBasedEstimatorBaseInput(EstimatorInput):
    df: pd.DataFrame


class DfBasedEstimatorBase(EstimatorBase[DfBasedEstimatorBaseInput]):
    @abstractmethod
    def _compute_estimator(
        self,
        input_df: pd.DataFrame,
    ) -> Estimate:
        pass

    @final
    def compute_estimator(
        self,
        input: DfBasedEstimatorBaseInput,
    ) -> Estimate:
        """
        Compute the estimator on the given input
        Args:
            input pd.DataFrame: Containg fileds: value, weight
            and additional fields specified in additional_fields classes
        Returns:
            Estimate
        """
        input_df = input.df.copy()
        return self._compute_estimator(input_df)
