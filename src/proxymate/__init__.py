# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""proxymate: A four-level framework for proxy metric validation."""

__version__ = "0.1.0"

from proxymate.data.data_handler import DataHandler
from proxymate.orchestrator.validator import Validator as Validator
from proxymate.planner.validation_planner import (
    build_default_planner,
    ValidationPlanner,
)
from proxymate.types.data import CheckResult, Estimate
from proxymate.types.inputs import (
    AggregateInput,
    CheckInput,
    MetaAnalysisInput,
    UnitLevelInput,
)

__all__ = [
    "Validator",
    "DataHandler",
    "ValidationPlanner",
    "build_default_planner",
    "Estimate",
    "CheckResult",
    "CheckInput",
    "UnitLevelInput",
    "AggregateInput",
    "MetaAnalysisInput",
]
