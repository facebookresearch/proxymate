# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import importlib.util
from unittest import skipUnless, TestCase

import numpy as np
import pandas as pd
from proxymate.checks.unit_level.calibration import (
    BrierScoreCheck,
    BrierSkillScoreCheck,
    CalibrationRatioCheck,
    ECECheck,
    KuiperCalibrationCheck,
)
from proxymate.checks.unit_level.discrimination import (
    ClassificationAccuracyCheck,
    ClassificationPrecisionCheck,
    ConfusionMatrixCheck,
    KendallTauCheck,
    PrecisionRecallAUCCheck,
    ROCAUCCheck,
    SensitivityCheck,
    SpearmanCorrelationCheck,
    SpecificityCheck,
)
from proxymate.types.enums import UnitLevelProperty, ValidationLevel
from proxymate.types.inputs import CheckInput

_HAS_MCGRAD = importlib.util.find_spec("mcgrad") is not None


def _make_input(
    primary: list[float],
    proxy: list[float],
    weight: list[float] | None = None,
) -> CheckInput:
    data = {"primary": primary, "proxy": proxy}
    if weight is not None:
        data["weight"] = weight
    return CheckInput(
        df=pd.DataFrame(data),
        primary_col="primary",
        proxy_col="proxy",
        weight_col="weight" if weight is not None else None,
    )


# ---- Calibration checks ----


@skipUnless(_HAS_MCGRAD, "requires the optional 'mcgrad' package")
class TestCalibrationRatioCheck(TestCase):
    def test_well_calibrated(self) -> None:
        # Scores that match the observed rate well
        # 50% positive rate, scores centered around 0.5
        check = CalibrationRatioCheck()
        inp = _make_input(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9],
        )
        result = check.compute(inp)
        # Calibration ratio should be close to 1.0 for well-calibrated scores
        self.assertGreater(result.calibration_ratio, 0.5)
        self.assertEqual(result.n_samples, 10)

    def test_project_bounds(self) -> None:
        check = CalibrationRatioCheck()
        inp = _make_input(
            [0, 0, 1, 1],
            [0.1, 0.4, 0.6, 0.9],
        )
        result = check.compute(inp)
        proj = check.project(result)
        self.assertGreaterEqual(proj, 0.0)
        self.assertLessEqual(proj, 1.0)

    def test_default_threshold(self) -> None:
        check = CalibrationRatioCheck()
        self.assertAlmostEqual(check.threshold, 0.5)

    def test_metadata(self) -> None:
        self.assertEqual(
            CalibrationRatioCheck.sub_property,
            UnitLevelProperty.CALIBRATION,
        )
        self.assertEqual(
            CalibrationRatioCheck.validation_level,
            ValidationLevel.UNIT_LEVEL,
        )

    def test_run_end_to_end(self) -> None:
        check = CalibrationRatioCheck()
        inp = _make_input(
            [0, 0, 0, 1, 1, 1],
            [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        )
        result = check.run(inp)
        self.assertEqual(result.check_name, "Calibration Ratio")
        self.assertIsNotNone(result.check_result)

    def test_uniform_weights_close_to_unweighted(self) -> None:
        check = CalibrationRatioCheck()
        inp_unw = _make_input(
            [0, 0, 0, 1, 1, 1],
            [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        )
        inp_w = _make_input(
            [0, 0, 0, 1, 1, 1],
            [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
            weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        r_unw = check.compute(inp_unw)
        r_w = check.compute(inp_w)
        # Weighted with uniform weights uses np.average (= simple mean)
        # while unweighted uses multicalibration's calibration_ratio.
        # Both should be close for well-calibrated data.
        self.assertAlmostEqual(r_w.calibration_ratio, 1.0, places=0)
        self.assertAlmostEqual(r_unw.calibration_ratio, 1.0, places=0)

    def test_weights_shift_calibration_ratio(self) -> None:
        check = CalibrationRatioCheck()
        # primary=[0,0,1,1], proxy=[0.2,0.3,0.8,0.9]
        # With w=[10,1,1,1]:
        #   weighted primary mean = (0*10+0+1+1)/13 = 2/13
        #   weighted proxy mean = (0.2*10+0.3+0.8+0.9)/13 = 4.0/13
        #   ratio = 4.0/2.0 = 2.0
        inp = _make_input(
            [0, 0, 1, 1],
            [0.2, 0.3, 0.8, 0.9],
            weight=[10, 1, 1, 1],
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.calibration_ratio, 2.0, places=2)


@skipUnless(_HAS_MCGRAD, "requires the optional 'mcgrad' package")
class TestKuiperCalibrationCheck(TestCase):
    def test_well_calibrated(self) -> None:
        # Well-calibrated scores should have low Kuiper statistic
        check = KuiperCalibrationCheck()
        inp = _make_input(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9],
        )
        result = check.compute(inp)
        self.assertGreaterEqual(result.ecce_kuiper, 0.0)
        self.assertEqual(result.n_samples, 10)

    def test_miscalibrated(self) -> None:
        # Constant score of 0.9 for 50/50 labels → poor calibration
        check = KuiperCalibrationCheck()
        inp = _make_input(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        )
        result = check.compute(inp)
        # Miscalibrated scores should have higher normalized Kuiper
        self.assertGreater(result.normalized_kuiper, 0.0)

    def test_project_inverts(self) -> None:
        check = KuiperCalibrationCheck()
        inp = _make_input(
            [0, 0, 0, 1, 1, 1],
            [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        )
        result = check.compute(inp)
        proj = check.project(result)
        # Lower normalized Kuiper → higher projection
        self.assertGreaterEqual(proj, 0.0)
        self.assertLessEqual(proj, 1.0)

    def test_default_threshold(self) -> None:
        check = KuiperCalibrationCheck()
        self.assertAlmostEqual(check.threshold, 0.5)

    def test_metadata(self) -> None:
        self.assertEqual(
            KuiperCalibrationCheck.sub_property,
            UnitLevelProperty.CALIBRATION,
        )

    def test_run_end_to_end(self) -> None:
        check = KuiperCalibrationCheck()
        inp = _make_input(
            [0, 0, 0, 1, 1, 1],
            [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        )
        result = check.run(inp)
        self.assertEqual(result.check_name, "Kuiper Calibration")
        self.assertIsNotNone(result.check_result)


# ---- ECE ----


class TestECECheck(TestCase):
    def test_well_calibrated(self) -> None:
        check = ECECheck(n_bins=5)
        # Scores match labels perfectly: 0s get low scores, 1s get high scores
        inp = _make_input(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0.05, 0.1, 0.15, 0.2, 0.25, 0.75, 0.8, 0.85, 0.9, 0.95],
        )
        result = check.compute(inp)
        # Well-calibrated => low ECE
        self.assertLess(result.ece, 0.3)
        self.assertEqual(result.n_samples, 10)
        self.assertEqual(result.n_bins, 5)

    def test_miscalibrated(self) -> None:
        check = ECECheck(n_bins=5)
        # All 0 labels but all high predicted probabilities
        inp = _make_input(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        )
        result = check.compute(inp)
        # Very miscalibrated => high ECE
        self.assertGreater(result.ece, 0.5)

    def test_project_bounds(self) -> None:
        check = ECECheck()
        inp = _make_input(
            [0, 0, 1, 1],
            [0.1, 0.4, 0.6, 0.9],
        )
        result = check.compute(inp)
        proj = check.project(result)
        self.assertGreaterEqual(proj, 0.0)
        self.assertLessEqual(proj, 1.0)

    def test_default_threshold(self) -> None:
        check = ECECheck()
        self.assertAlmostEqual(check.threshold, 0.5)

    def test_metadata(self) -> None:
        self.assertEqual(ECECheck.sub_property, UnitLevelProperty.CALIBRATION)
        self.assertEqual(ECECheck.validation_level, ValidationLevel.UNIT_LEVEL)

    def test_run_end_to_end(self) -> None:
        check = ECECheck()
        inp = _make_input(
            [0, 0, 0, 1, 1, 1],
            [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        )
        result = check.run(inp)
        self.assertEqual(result.check_name, "Expected Calibration Error")
        self.assertIsNotNone(result.check_result)


# ---- Brier Score ----


class TestBrierScoreCheck(TestCase):
    def test_perfect_predictions(self) -> None:
        check = BrierScoreCheck()
        inp = _make_input(
            [0, 0, 0, 1, 1, 1],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.brier_score, 0.0)
        self.assertEqual(result.n_samples, 6)

    def test_worst_predictions(self) -> None:
        check = BrierScoreCheck()
        inp = _make_input(
            [0, 0, 0, 1, 1, 1],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.brier_score, 1.0)

    def test_decomposition_sums(self) -> None:
        check = BrierScoreCheck()
        inp = _make_input(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9],
        )
        result = check.compute(inp)
        # BS ≈ Reliability - Resolution + Uncertainty
        decomposition = result.reliability - result.resolution + result.uncertainty
        self.assertAlmostEqual(result.brier_score, decomposition, places=1)

    def test_uncertainty(self) -> None:
        check = BrierScoreCheck()
        # 50% base rate => uncertainty = 0.25
        inp = _make_input(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0.5] * 10,
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.uncertainty, 0.25)

    def test_project(self) -> None:
        check = BrierScoreCheck()
        inp = _make_input(
            [0, 0, 0, 1, 1, 1],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        result = check.compute(inp)
        proj = check.project(result)
        self.assertAlmostEqual(proj, 1.0)

    def test_metadata(self) -> None:
        self.assertEqual(BrierScoreCheck.sub_property, UnitLevelProperty.CALIBRATION)

    def test_run_end_to_end(self) -> None:
        check = BrierScoreCheck()
        inp = _make_input(
            [0, 0, 0, 1, 1, 1],
            [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        )
        result = check.run(inp)
        self.assertEqual(result.check_name, "Brier Score")
        self.assertIsNotNone(result.check_result)


# ---- Brier Skill Score ----


class TestBrierSkillScoreCheck(TestCase):
    def test_perfect_predictions(self) -> None:
        check = BrierSkillScoreCheck()
        inp = _make_input(
            [0, 0, 0, 1, 1, 1],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        result = check.compute(inp)
        # Perfect => BS=0, BS_ref=0.25 => BSS=1
        self.assertAlmostEqual(result.bss, 1.0)
        self.assertAlmostEqual(result.brier_score, 0.0)

    def test_no_skill(self) -> None:
        check = BrierSkillScoreCheck()
        # Predicting base rate (0.5) for balanced data => BSS = 0
        inp = _make_input(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0.5] * 10,
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.bss, 0.0, places=1)

    def test_negative_skill(self) -> None:
        check = BrierSkillScoreCheck()
        # Inverted predictions => worse than climatology
        inp = _make_input(
            [0, 0, 0, 1, 1, 1],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        )
        result = check.compute(inp)
        self.assertLess(result.bss, 0.0)

    def test_project_clamps(self) -> None:
        check = BrierSkillScoreCheck()
        inp = _make_input(
            [0, 0, 0, 1, 1, 1],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        )
        result = check.compute(inp)
        proj = check.project(result)
        # Negative BSS clamped to 0
        self.assertAlmostEqual(proj, 0.0)

    def test_metadata(self) -> None:
        self.assertEqual(
            BrierSkillScoreCheck.sub_property, UnitLevelProperty.CALIBRATION
        )

    def test_run_end_to_end(self) -> None:
        check = BrierSkillScoreCheck()
        inp = _make_input(
            [0, 0, 0, 1, 1, 1],
            [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        )
        result = check.run(inp)
        self.assertEqual(result.check_name, "Brier Skill Score")
        self.assertIsNotNone(result.check_result)


# ---- Continuous: Spearman, Kendall ----


class TestSpearmanCorrelationCheck(TestCase):
    def test_perfect_monotonic(self) -> None:
        check = SpearmanCorrelationCheck()
        inp = _make_input([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
        result = check.compute(inp)
        self.assertAlmostEqual(result.correlation, 1.0)
        self.assertEqual(result.n_samples, 5)

    def test_negative_correlation(self) -> None:
        check = SpearmanCorrelationCheck()
        inp = _make_input([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
        result = check.compute(inp)
        self.assertAlmostEqual(result.correlation, -1.0)

    def test_project_clamps(self) -> None:
        check = SpearmanCorrelationCheck()
        inp = _make_input([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
        result = check.compute(inp)
        proj = check.project(result)
        self.assertGreaterEqual(proj, 0.0)
        self.assertLessEqual(proj, 1.0)

    def test_metadata(self) -> None:
        self.assertEqual(
            SpearmanCorrelationCheck.sub_property,
            UnitLevelProperty.DISCRIMINATION,
        )
        self.assertEqual(
            SpearmanCorrelationCheck.validation_level,
            ValidationLevel.UNIT_LEVEL,
        )


class TestKendallTauCheck(TestCase):
    def test_perfect_concordance(self) -> None:
        check = KendallTauCheck()
        inp = _make_input([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        result = check.compute(inp)
        self.assertAlmostEqual(result.tau, 1.0)
        self.assertEqual(result.n_samples, 5)

    def test_perfect_discordance(self) -> None:
        check = KendallTauCheck()
        inp = _make_input([1, 2, 3, 4, 5], [10, 8, 6, 4, 2])
        result = check.compute(inp)
        self.assertAlmostEqual(result.tau, -1.0)


# ---- Binary Probability: ROC AUC, PR AUC ----


class TestROCAUCCheck(TestCase):
    def test_perfect_separation(self) -> None:
        check = ROCAUCCheck()
        inp = _make_input(
            [0, 0, 0, 1, 1, 1],
            [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.roc_auc, 1.0)
        self.assertEqual(result.n_samples, 6)

    def test_random_predictions(self) -> None:
        check = ROCAUCCheck()
        np.random.seed(42)  # pyre-ignore[bad-argument-type]
        n = 200
        labels = np.array([0] * (n // 2) + [1] * (n // 2), dtype=float)
        scores = np.random.rand(n).tolist()
        inp = _make_input(labels.tolist(), scores)
        result = check.compute(inp)
        # Random predictions should give AUC ~ 0.5
        self.assertAlmostEqual(result.roc_auc, 0.5, places=0)

    def test_with_weights(self) -> None:
        check = ROCAUCCheck()
        inp = _make_input(
            [0, 0, 1, 1],
            [0.1, 0.4, 0.6, 0.9],
            weight=[1.0, 1.0, 1.0, 1.0],
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.roc_auc, 1.0)

    def test_default_threshold(self) -> None:
        check = ROCAUCCheck()
        self.assertAlmostEqual(check.threshold, 0.5)

    def test_project(self) -> None:
        check = ROCAUCCheck()
        inp = _make_input(
            [0, 0, 1, 1],
            [0.1, 0.4, 0.6, 0.9],
        )
        result = check.compute(inp)
        proj = check.project(result)
        self.assertAlmostEqual(proj, 1.0)


class TestPrecisionRecallAUCCheck(TestCase):
    def test_perfect_separation(self) -> None:
        check = PrecisionRecallAUCCheck()
        inp = _make_input(
            [0, 0, 0, 1, 1, 1],
            [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.pr_auc, 1.0)
        self.assertEqual(result.n_samples, 6)


# ---- Binary-Binary: Classification metrics ----


class TestClassificationPrecisionCheck(TestCase):
    def test_perfect_precision(self) -> None:
        check = ClassificationPrecisionCheck()
        inp = _make_input([1, 1, 0, 0], [1, 1, 0, 0])
        result = check.compute(inp)
        self.assertAlmostEqual(result.precision, 1.0)

    def test_half_precision(self) -> None:
        check = ClassificationPrecisionCheck()
        # Predicted 1 for all: 2 TP, 2 FP => precision = 0.5
        inp = _make_input([1, 1, 0, 0], [1, 1, 1, 1])
        result = check.compute(inp)
        self.assertAlmostEqual(result.precision, 0.5)

    def test_default_threshold(self) -> None:
        check = ClassificationPrecisionCheck()
        self.assertAlmostEqual(check.threshold, 0.7)


class TestClassificationAccuracyCheck(TestCase):
    def test_perfect_accuracy(self) -> None:
        check = ClassificationAccuracyCheck()
        inp = _make_input([1, 1, 0, 0], [1, 1, 0, 0])
        result = check.compute(inp)
        self.assertAlmostEqual(result.accuracy, 1.0)

    def test_half_accuracy(self) -> None:
        check = ClassificationAccuracyCheck()
        inp = _make_input([1, 1, 0, 0], [0, 0, 0, 0])
        result = check.compute(inp)
        self.assertAlmostEqual(result.accuracy, 0.5)

    def test_uniform_weights_match_unweighted(self) -> None:
        check = ClassificationAccuracyCheck()
        inp_unw = _make_input([1, 1, 0, 0], [1, 0, 0, 0])
        inp_w = _make_input([1, 1, 0, 0], [1, 0, 0, 0], weight=[1.0, 1.0, 1.0, 1.0])
        r_unw = check.compute(inp_unw)
        r_w = check.compute(inp_w)
        self.assertAlmostEqual(r_unw.accuracy, r_w.accuracy)

    def test_weights_shift_accuracy(self) -> None:
        check = ClassificationAccuracyCheck()
        # Correct: sample 0 (w=1), sample 2 (w=1), sample 3 (w=10) = 12
        # Wrong: sample 1 (w=1) = 1
        # Weighted accuracy = 12/13
        inp = _make_input([1, 1, 0, 0], [1, 0, 0, 0], weight=[1, 1, 1, 10])
        result = check.compute(inp)
        self.assertAlmostEqual(result.accuracy, 12.0 / 13.0, places=3)


class TestSensitivityCheck(TestCase):
    def test_perfect_sensitivity(self) -> None:
        check = SensitivityCheck()
        inp = _make_input([1, 1, 0, 0], [1, 1, 0, 0])
        result = check.compute(inp)
        self.assertAlmostEqual(result.sensitivity, 1.0)

    def test_zero_sensitivity(self) -> None:
        check = SensitivityCheck()
        inp = _make_input([1, 1, 0, 0], [0, 0, 0, 0])
        result = check.compute(inp)
        self.assertAlmostEqual(result.sensitivity, 0.0)

    def test_uniform_weights_match_unweighted(self) -> None:
        check = SensitivityCheck()
        inp_unw = _make_input([1, 1, 0, 0], [1, 0, 0, 0])
        inp_w = _make_input([1, 1, 0, 0], [1, 0, 0, 0], weight=[1.0, 1.0, 1.0, 1.0])
        r_unw = check.compute(inp_unw)
        r_w = check.compute(inp_w)
        self.assertAlmostEqual(r_unw.sensitivity, r_w.sensitivity)

    def test_weights_shift_sensitivity(self) -> None:
        check = SensitivityCheck()
        # TP at sample 0 (primary=1,proxy=1), FN at sample 1 (primary=1,proxy=0)
        # Unweighted: sens = 1/2 = 0.5
        # With w=[10,1,1,1]: weighted TP=10, FN=1 → sens = 10/11
        inp = _make_input([1, 1, 0, 0], [1, 0, 0, 0], weight=[10, 1, 1, 1])
        result = check.compute(inp)
        self.assertAlmostEqual(result.sensitivity, 10.0 / 11.0, places=3)


class TestSpecificityCheck(TestCase):
    def test_perfect_specificity(self) -> None:
        check = SpecificityCheck()
        inp = _make_input([1, 1, 0, 0], [1, 1, 0, 0])
        result = check.compute(inp)
        self.assertAlmostEqual(result.specificity, 1.0)

    def test_zero_specificity(self) -> None:
        check = SpecificityCheck()
        inp = _make_input([1, 1, 0, 0], [1, 1, 1, 1])
        result = check.compute(inp)
        self.assertAlmostEqual(result.specificity, 0.0)

    def test_uniform_weights_match_unweighted(self) -> None:
        check = SpecificityCheck()
        inp_unw = _make_input([1, 1, 0, 0], [1, 1, 1, 0])
        inp_w = _make_input([1, 1, 0, 0], [1, 1, 1, 0], weight=[1.0, 1.0, 1.0, 1.0])
        r_unw = check.compute(inp_unw)
        r_w = check.compute(inp_w)
        self.assertAlmostEqual(r_unw.specificity, r_w.specificity)

    def test_weights_shift_specificity(self) -> None:
        check = SpecificityCheck()
        # TN at sample 3 (primary=0,proxy=0), FP at sample 2 (primary=0,proxy=1)
        # Unweighted: spec = 1/2 = 0.5
        # With w=[1,1,1,10]: weighted TN=10, FP=1 → spec = 10/11
        inp = _make_input([1, 1, 0, 0], [1, 1, 1, 0], weight=[1, 1, 1, 10])
        result = check.compute(inp)
        self.assertAlmostEqual(result.specificity, 10.0 / 11.0, places=3)


class TestConfusionMatrixCheck(TestCase):
    def test_known_matrix(self) -> None:
        check = ConfusionMatrixCheck()
        # primary=[1,1,0,0], proxy=[1,0,1,0]
        # TP=1, FN=1, FP=1, TN=1
        inp = _make_input([1, 1, 0, 0], [1, 0, 1, 0])
        result = check.compute(inp)
        self.assertEqual(result.true_positives, 1)
        self.assertEqual(result.false_negatives, 1)
        self.assertEqual(result.false_positives, 1)
        self.assertEqual(result.true_negatives, 1)
        self.assertEqual(result.n_samples, 4)

    def test_project_always_one(self) -> None:
        check = ConfusionMatrixCheck()
        inp = _make_input([1, 0], [1, 0])
        result = check.compute(inp)
        self.assertAlmostEqual(check.project(result), 1.0)


# ---- End-to-end run() ----


class TestEndToEnd(TestCase):
    def test_spearman_run(self) -> None:
        check = SpearmanCorrelationCheck(threshold=0.5)
        inp = _make_input([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        result = check.run(inp)
        self.assertEqual(result.check_name, "Spearman Correlation")
        self.assertAlmostEqual(result.projection_result, 1.0)
        self.assertTrue(result.thresholding_result)

    def test_roc_auc_run(self) -> None:
        check = ROCAUCCheck(threshold=0.8)
        inp = _make_input(
            [0, 0, 0, 1, 1, 1],
            [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        )
        result = check.run(inp)
        self.assertEqual(result.check_name, "ROC AUC")
        self.assertAlmostEqual(result.projection_result, 1.0)
        self.assertTrue(result.thresholding_result)

    def test_classification_accuracy_run_fails_threshold(self) -> None:
        check = ClassificationAccuracyCheck(threshold=0.9)
        inp = _make_input([1, 1, 0, 0], [1, 0, 0, 0])
        result = check.run(inp)
        self.assertEqual(result.check_name, "Accuracy")
        self.assertAlmostEqual(result.projection_result, 0.75)
        self.assertFalse(result.thresholding_result)
