# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import ClassVar
from unittest import TestCase

from proxymate.checks.base import CheckBase
from proxymate.types.data import CheckResult
from proxymate.types.enums import OutcomeType, UnitLevelProperty, ValidationLevel


@dataclass
class _DummyResult:
    value: float


class _DummyCheck(CheckBase[_DummyResult, float]):
    """Minimal concrete check for testing the base class."""

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.ACCURACY
    supported_outcome_types: ClassVar[list[OutcomeType]] = [OutcomeType.CONTINUOUS]

    def get_name(self) -> str:
        return "Dummy Check"

    def get_description(self) -> str:
        return "A dummy check for testing."

    def compute(self, input: float) -> _DummyResult:
        return _DummyResult(value=input)

    def project(self, result: _DummyResult) -> float:
        return result.value


class _FailingCheck(CheckBase[_DummyResult, float]):
    """Check that always raises during compute."""

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = UnitLevelProperty.PRECISION
    supported_outcome_types: ClassVar[list[OutcomeType]] = [OutcomeType.CONTINUOUS]

    def get_name(self) -> str:
        return "Failing Check"

    def get_description(self) -> str:
        return "A check that always fails."

    def compute(self, input: float) -> _DummyResult:
        raise ValueError("Intentional failure")

    def project(self, result: _DummyResult) -> float:
        return result.value


class TestCheckBase(TestCase):
    def test_run_passing(self) -> None:
        check = _DummyCheck(threshold=0.5)
        result = check.run(0.9)

        self.assertIsInstance(result, CheckResult)
        self.assertEqual(result.check_name, "Dummy Check")
        self.assertEqual(result.validation_level, ValidationLevel.UNIT_LEVEL)
        self.assertEqual(result.sub_property, UnitLevelProperty.ACCURACY)
        self.assertAlmostEqual(result.projection_result, 0.9)
        self.assertEqual(result.threshold, 0.5)
        self.assertTrue(result.thresholding_result)

    def test_run_failing_threshold(self) -> None:
        check = _DummyCheck(threshold=0.95)
        result = check.run(0.5)

        self.assertAlmostEqual(result.projection_result, 0.5)
        self.assertFalse(result.thresholding_result)

    def test_run_with_override_threshold(self) -> None:
        check = _DummyCheck(threshold=0.5)
        result = check.run(0.7, threshold=0.8)

        self.assertEqual(result.threshold, 0.8)
        self.assertFalse(result.thresholding_result)

    def test_default_threshold(self) -> None:
        check = _DummyCheck()
        self.assertAlmostEqual(check.threshold, 0.8)

    def test_zero_threshold(self) -> None:
        check = _DummyCheck(threshold=0.0)
        self.assertAlmostEqual(check.threshold, 0.0)
        result = check.run(0.0)
        self.assertTrue(result.thresholding_result)

    def test_error_handling(self) -> None:
        check = _FailingCheck()
        result = check.run(0.9)

        self.assertEqual(result.check_name, "Failing Check")
        self.assertAlmostEqual(result.projection_result, 0.0)
        self.assertFalse(result.thresholding_result)

    def test_class_metadata(self) -> None:
        self.assertEqual(_DummyCheck.validation_level, ValidationLevel.UNIT_LEVEL)
        self.assertEqual(_DummyCheck.sub_property, UnitLevelProperty.ACCURACY)
        self.assertIn(OutcomeType.CONTINUOUS, _DummyCheck.supported_outcome_types)
