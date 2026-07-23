# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Set, TypeVar

from proxymate.lib.base_classes.data_types import Estimate


@dataclass
class EstimatorInput:
    pass


TEstimatorInput = TypeVar("TEstimatorInput", bound=EstimatorInput)


class EstimatorBase(Generic[TEstimatorInput], ABC):
    """
    Abstract Base base class for defining estimators.
    This class provides a template for implementing the computation of estimators.
    """

    def __init__(
        self,
        confidence_level: float | None = None,
    ) -> None:
        self._confidence_level: float = (
            confidence_level or self.get_estimator_default_confidence_level()
        )
        if self._confidence_level > 1 or self._confidence_level < 0:
            raise ValueError(
                f"Supplied confidence_interval {self._confidence_level} "
                "is out of range [0,1]"
            )

    def get_additional_entities_columns(self) -> Set[str]:
        return set()

    def get_additional_metric_columns(self) -> Set[str]:
        return set()

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the name of the estimator.
        """
        pass

    def get_estimator_default_confidence_level(self) -> float:
        """
        get the default confidence level for the check (to be used in case no
        confidence level is provided by the user).
        """
        return 0.95

    @abstractmethod
    def compute_estimator(
        self,
        input: TEstimatorInput,
    ) -> Estimate:
        pass
