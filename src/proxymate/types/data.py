# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Generic, TypeVar

from proxymate.lib.base_classes.data_types import Estimate as Estimate  # noqa: F401
from proxymate.types.enums import (
    DomainLevelProperty,
    EstimateLevelProperty,
    RepresentativityProperty,
    UnitLevelProperty,
    ValidationLevel,
)

TCheckResultType = TypeVar("TCheckResultType")


@dataclass
class CheckResult(Generic[TCheckResultType]):
    """Result of a single check, enriched with framework metadata."""

    check_name: str
    description: str
    validation_level: ValidationLevel
    sub_property: (
        UnitLevelProperty
        | EstimateLevelProperty
        | DomainLevelProperty
        | RepresentativityProperty
    )
    check_result: TCheckResultType
    projection_result: float
    threshold: float
    thresholding_result: bool
