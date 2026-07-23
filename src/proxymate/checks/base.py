# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from abc import ABC, abstractmethod
from typing import cast, ClassVar, final, Generic, TypeVar

from proxymate.types.data import CheckResult
from proxymate.types.enums import (
    DomainLevelProperty,
    EstimateLevelProperty,
    OutcomeType,
    RepresentativityProperty,
    UnitLevelProperty,
    ValidationLevel,
)

_logger: logging.Logger = logging.getLogger(__name__)

TResult = TypeVar("TResult")
TInput = TypeVar("TInput")


class CheckBase(ABC, Generic[TResult, TInput]):
    """Root base class for the redesigned validation checks.

    Every check declares its position in the four-level proxy validation framework via
    class-level attributes. The orchestrator uses these to group results,
    recommend checks, and validate input compatibility.

    Subclasses must:
    1. Set the four ClassVar attributes (validation_level, sub_property, etc.)
    2. Implement get_name(), compute(), and project()
    """

    # --- Framework metadata (set in each concrete subclass) ---
    validation_level: ClassVar[ValidationLevel]
    sub_property: ClassVar[
        UnitLevelProperty
        | EstimateLevelProperty
        | DomainLevelProperty
        | RepresentativityProperty
    ]
    supported_outcome_types: ClassVar[list[OutcomeType]]

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold: float = (
            threshold if threshold is not None else self.get_default_threshold()
        )

    # --- Check interface (override in subclass) ---

    @abstractmethod
    def get_name(self) -> str:
        """Returns a human-readable name for the check."""
        ...

    @abstractmethod
    def get_description(self) -> str:
        """Returns a short description of what this check measures."""
        ...

    @abstractmethod
    def compute(self, input: TInput) -> TResult:
        """Pure computation. Returns the raw check result."""
        ...

    @abstractmethod
    def project(self, result: TResult) -> float:
        """Normalize result to [0, 1] where 1 = best."""
        ...

    def get_default_threshold(self) -> float:
        return 0.8

    # --- Orchestration (not overridden) ---

    @final
    def run(
        self, input: TInput, threshold: float | None = None
    ) -> CheckResult[TResult]:
        """Compute, project, and threshold. Returns structured result.

        If ``compute()`` or ``project()`` raise an exception (e.g. due to
        insufficient data in a segment), a failed result is returned with
        ``projection_result=0.0`` and ``thresholding_result=False`` instead
        of letting the error propagate and crash the pipeline.
        """
        effective_threshold = threshold if threshold is not None else self.threshold
        try:
            result = self.compute(input)
            projection = self.project(result)
        except (ValueError, ArithmeticError) as exc:
            _logger.warning(
                "Check '%s' failed: %s. Returning a failed result.",
                self.get_name(),
                type(exc).__name__,
            )
            return cast(
                "CheckResult[TResult]",
                CheckResult(
                    check_name=self.get_name(),
                    description=self.get_description(),
                    validation_level=self.validation_level,
                    sub_property=self.sub_property,
                    check_result=None,
                    projection_result=0.0,
                    threshold=effective_threshold,
                    thresholding_result=False,
                ),
            )
        return CheckResult(
            check_name=self.get_name(),
            description=self.get_description(),
            validation_level=self.validation_level,
            sub_property=self.sub_property,
            check_result=result,
            projection_result=projection,
            threshold=effective_threshold,
            thresholding_result=(projection >= effective_threshold),
        )
