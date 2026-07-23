# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from unittest import TestCase

import numpy as np
import pandas as pd
from proxymate.checks.representativity_level.representativity import (
    CoverageCheck,
    EffectiveSampleSizeCheck,
    SampleRepresentativityCheck,
    WeightExtremeCheck,
    WeightingEffectivenessCheck,
)
from proxymate.checks.unit_level.distributional_fidelity import (
    KSDistCheck,
    TVDistCheck,
    WassersteinDistCheck,
)
from proxymate.types.enums import (
    RepresentativityProperty,
    UnitLevelProperty,
    ValidationLevel,
)
from proxymate.types.inputs import CheckInput


def _make_input(
    primary: list[float],
    proxy: list[float],
    weight: list[float] | None = None,
    population: list[bool] | None = None,
    covariates: dict[str, list[float]] | None = None,
    covariate_cols: list[str] | None = None,
) -> CheckInput:
    data: dict[str, list[float] | list[bool]] = {
        "primary": primary,
        "proxy": proxy,
    }
    if weight is not None:
        data["weight"] = weight
    if population is not None:
        data["population"] = population
    if covariates is not None:
        data.update(covariates)
    return CheckInput(
        df=pd.DataFrame(data),
        primary_col="primary",
        proxy_col="proxy",
        weight_col="weight" if weight is not None else None,
        population_col="population" if population is not None else None,
        covariate_cols=covariate_cols,
    )


# ---- Distributional Fidelity ----


class TestTVDistCheck(TestCase):
    def test_identical_distributions(self) -> None:
        check = TVDistCheck()
        vals = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        inp = _make_input(vals, vals)
        result = check.compute(inp)
        self.assertAlmostEqual(result.tv_distance, 0.0, places=2)
        self.assertEqual(result.n_samples, 100)

    def test_different_distributions(self) -> None:
        check = TVDistCheck()
        np.random.seed(42)  # pyre-ignore[bad-argument-type]
        # pyre-ignore[missing-attribute]
        primary = np.random.normal(0, 1, 500).tolist()
        # pyre-ignore[missing-attribute]
        proxy = np.random.normal(5, 1, 500).tolist()
        inp = _make_input(primary, proxy)
        result = check.compute(inp)
        # Very different distributions should have high TV distance
        self.assertGreater(result.tv_distance, 0.5)

    def test_project_inverts(self) -> None:
        check = TVDistCheck()
        vals = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        inp = _make_input(vals, vals)
        result = check.compute(inp)
        proj = check.project(result)
        # Identical distributions => TV ~ 0 => project ~ 1
        self.assertGreater(proj, 0.9)

    def test_custom_bins(self) -> None:
        check = TVDistCheck(n_bins=10)
        self.assertEqual(check.n_bins, 10)

    def test_metadata(self) -> None:
        self.assertEqual(
            TVDistCheck.sub_property,
            UnitLevelProperty.DISTRIBUTIONAL_FIDELITY,
        )
        self.assertEqual(
            TVDistCheck.validation_level,
            ValidationLevel.UNIT_LEVEL,
        )


class TestKSDistCheck(TestCase):
    def test_identical_distributions(self) -> None:
        check = KSDistCheck()
        vals = list(range(100))
        fvals = [float(v) for v in vals]
        inp = _make_input(fvals, fvals)
        result = check.compute(inp)
        self.assertAlmostEqual(result.ks_statistic, 0.0)
        self.assertEqual(result.n_samples, 100)

    def test_different_distributions(self) -> None:
        check = KSDistCheck()
        np.random.seed(42)  # pyre-ignore[bad-argument-type]
        # pyre-ignore[missing-attribute]
        primary = np.random.normal(0, 1, 200).tolist()
        # pyre-ignore[missing-attribute]
        proxy = np.random.normal(5, 1, 200).tolist()
        inp = _make_input(primary, proxy)
        result = check.compute(inp)
        self.assertGreater(result.ks_statistic, 0.5)
        self.assertLess(result.p_value, 0.05)

    def test_project(self) -> None:
        check = KSDistCheck()
        vals = list(range(100))
        fvals = [float(v) for v in vals]
        inp = _make_input(fvals, fvals)
        result = check.compute(inp)
        proj = check.project(result)
        self.assertAlmostEqual(proj, 1.0)


class TestWassersteinDistCheck(TestCase):
    def test_identical_distributions(self) -> None:
        check = WassersteinDistCheck()
        vals = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        inp = _make_input(vals, vals)
        result = check.compute(inp)
        self.assertAlmostEqual(result.wasserstein_distance, 0.0)
        self.assertAlmostEqual(result.normalized_distance, 0.0)
        self.assertEqual(result.n_samples, 100)

    def test_shifted_distributions(self) -> None:
        check = WassersteinDistCheck()
        np.random.seed(42)  # pyre-ignore[bad-argument-type]
        # pyre-ignore[missing-attribute]
        primary = np.random.normal(0, 1, 500).tolist()
        # pyre-ignore[missing-attribute]
        proxy = np.random.normal(3, 1, 500).tolist()
        inp = _make_input(primary, proxy)
        result = check.compute(inp)
        # Shift of 3 with SD ~1 => W1 ~ 3, normalized ~ 3
        self.assertGreater(result.wasserstein_distance, 2.0)
        self.assertGreater(result.normalized_distance, 1.0)

    def test_project_identical(self) -> None:
        check = WassersteinDistCheck()
        vals = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        inp = _make_input(vals, vals)
        result = check.compute(inp)
        proj = check.project(result)
        # Identical => distance = 0 => exp(0) = 1
        self.assertAlmostEqual(proj, 1.0)

    def test_project_shifted(self) -> None:
        check = WassersteinDistCheck()
        np.random.seed(42)  # pyre-ignore[bad-argument-type]
        # pyre-ignore[missing-attribute]
        primary = np.random.normal(0, 1, 500).tolist()
        # pyre-ignore[missing-attribute]
        proxy = np.random.normal(5, 1, 500).tolist()
        inp = _make_input(primary, proxy)
        result = check.compute(inp)
        proj = check.project(result)
        # Very shifted => low projection
        self.assertLess(proj, 0.1)

    def test_metadata(self) -> None:
        self.assertEqual(
            WassersteinDistCheck.sub_property,
            UnitLevelProperty.DISTRIBUTIONAL_FIDELITY,
        )
        self.assertEqual(
            WassersteinDistCheck.validation_level,
            ValidationLevel.UNIT_LEVEL,
        )

    def test_run_end_to_end(self) -> None:
        check = WassersteinDistCheck(threshold=0.5)
        vals = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        inp = _make_input(vals, vals)
        result = check.run(inp)
        self.assertEqual(result.check_name, "Wasserstein-1 Distance")
        self.assertTrue(result.thresholding_result)


# ---- Representativity: Coverage ----


class TestCoverageCheck(TestCase):
    def test_full_coverage(self) -> None:
        check = CoverageCheck()
        # All rows have non-null proxy => coverage = 1.0
        inp = _make_input(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        )
        result = check.compute(inp)
        self.assertEqual(result.target_size, 4)
        self.assertEqual(result.proxy_size, 4)
        self.assertAlmostEqual(result.coverage_rate, 1.0)

    def test_partial_coverage_with_population_col(self) -> None:
        check = CoverageCheck()
        inp = _make_input(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            population=[True, True, False, False],
        )
        result = check.compute(inp)
        self.assertEqual(result.target_size, 4)
        self.assertEqual(result.proxy_size, 2)
        self.assertAlmostEqual(result.coverage_rate, 0.5)

    def test_project(self) -> None:
        check = CoverageCheck()
        inp = _make_input(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            population=[True, False, False, False],
        )
        result = check.compute(inp)
        proj = check.project(result)
        self.assertAlmostEqual(proj, 0.25)

    def test_metadata(self) -> None:
        self.assertEqual(
            CoverageCheck.validation_level,
            ValidationLevel.REPRESENTATIVITY,
        )
        self.assertEqual(
            CoverageCheck.sub_property,
            RepresentativityProperty.COVERAGE,
        )


# ---- Representativity: Sample Representativity ----


class TestSampleRepresentativityCheck(TestCase):
    def test_identical_populations(self) -> None:
        check = SampleRepresentativityCheck()
        np.random.seed(42)  # pyre-ignore[bad-argument-type]
        # pyre-ignore[missing-attribute]
        vals = np.random.normal(0, 1, 100).tolist()
        inp = _make_input(
            vals,
            vals,
            covariates={"age": vals},
            covariate_cols=["age"],
        )
        result = check.compute(inp)
        # Same population => SMD ~ 0
        self.assertAlmostEqual(result.max_smd, 0.0, places=1)

    def test_different_populations(self) -> None:
        check = SampleRepresentativityCheck()
        np.random.seed(42)  # pyre-ignore[bad-argument-type]
        n = 100
        # Target has age ~ N(30, 5), proxy subset has age ~ N(50, 5)
        proxy_vals = [1.0] * n
        # Use population_col to split: first half = proxy
        population = [True] * (n // 2) + [False] * (n // 2)
        # Construct so proxy population has different ages
        # pyre-ignore[missing-attribute]
        age_proxy = np.random.normal(50, 5, n // 2).tolist()
        # pyre-ignore[missing-attribute]
        age_target = np.random.normal(30, 5, n // 2).tolist()
        combined_age = age_proxy + age_target

        inp = _make_input(
            [0.0] * n,
            proxy_vals,
            population=population,
            covariates={"age": combined_age},
            covariate_cols=["age"],
        )
        result = check.compute(inp)
        # Different means => high SMD
        self.assertGreater(result.max_smd, 1.0)

    def test_project(self) -> None:
        check = SampleRepresentativityCheck()
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        inp = _make_input(
            vals,
            vals,
            covariates={"x": vals},
            covariate_cols=["x"],
        )
        result = check.compute(inp)
        proj = check.project(result)
        # Same population => SMD ~ 0 => project ~ 1
        self.assertGreater(proj, 0.9)


# ---- Representativity: Effective Sample Size ----


class TestEffectiveSampleSizeCheck(TestCase):
    def test_equal_weights(self) -> None:
        check = EffectiveSampleSizeCheck()
        inp = _make_input(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            weight=[1.0, 1.0, 1.0, 1.0],
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.effective_n, 4.0)
        self.assertEqual(result.actual_n, 4)
        self.assertAlmostEqual(result.efficiency, 1.0)

    def test_no_weights(self) -> None:
        check = EffectiveSampleSizeCheck()
        inp = _make_input(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.efficiency, 1.0)

    def test_unequal_weights(self) -> None:
        check = EffectiveSampleSizeCheck()
        # One very large weight dominates
        inp = _make_input(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            weight=[100.0, 1.0, 1.0, 1.0],
        )
        result = check.compute(inp)
        # Effective n should be much less than actual n
        self.assertLess(result.effective_n, 4.0)
        self.assertLess(result.efficiency, 0.5)

    def test_project(self) -> None:
        check = EffectiveSampleSizeCheck()
        inp = _make_input(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            weight=[1.0, 1.0, 1.0, 1.0],
        )
        result = check.compute(inp)
        self.assertAlmostEqual(check.project(result), 1.0)

    def test_metadata(self) -> None:
        self.assertEqual(
            EffectiveSampleSizeCheck.sub_property,
            RepresentativityProperty.EFFECTIVE_SAMPLE_SIZE,
        )


# ---- Representativity: Weight Extremity ----


class TestWeightExtremeCheck(TestCase):
    def test_equal_weights(self) -> None:
        check = WeightExtremeCheck()
        inp = _make_input(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            weight=[1.0, 1.0, 1.0, 1.0],
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.weight_ratio, 1.0)
        self.assertAlmostEqual(result.cv, 0.0)

    def test_no_weights(self) -> None:
        check = WeightExtremeCheck()
        inp = _make_input(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.weight_ratio, 1.0)
        self.assertAlmostEqual(result.cv, 0.0)

    def test_extreme_weights(self) -> None:
        check = WeightExtremeCheck()
        inp = _make_input(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            weight=[100.0, 1.0, 1.0, 1.0],
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.max_weight, 100.0)
        self.assertAlmostEqual(result.min_weight, 1.0)
        self.assertAlmostEqual(result.weight_ratio, 100.0)
        self.assertGreater(result.cv, 1.0)

    def test_project_equal_weights(self) -> None:
        check = WeightExtremeCheck()
        inp = _make_input(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            weight=[1.0, 1.0, 1.0, 1.0],
        )
        result = check.compute(inp)
        # CV = 0 => project = 1 / (1 + 0) = 1.0
        self.assertAlmostEqual(check.project(result), 1.0)

    def test_project_extreme_weights(self) -> None:
        check = WeightExtremeCheck()
        inp = _make_input(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            weight=[100.0, 1.0, 1.0, 1.0],
        )
        result = check.compute(inp)
        # High CV => project < 0.5
        self.assertLess(check.project(result), 0.5)


# ---- End-to-end run() ----


class TestEndToEnd(TestCase):
    def test_tv_dist_run(self) -> None:
        check = TVDistCheck(threshold=0.5)
        vals = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        inp = _make_input(vals, vals)
        result = check.run(inp)
        self.assertEqual(result.check_name, "TV Distance")
        self.assertTrue(result.thresholding_result)

    def test_coverage_run(self) -> None:
        check = CoverageCheck(threshold=0.3)
        inp = _make_input(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            population=[True, False, False, False],
        )
        result = check.run(inp)
        self.assertEqual(result.check_name, "Coverage")
        self.assertFalse(result.thresholding_result)


# ---- Representativity: Weighting Effectiveness ----


class TestWeightingEffectivenessCheck(TestCase):
    def test_weights_shift_proxy_mean(self) -> None:
        """Weights shift the proxy mean closer to target."""
        check = WeightingEffectivenessCheck()
        inp = _make_input(
            # 4 target + 4 proxy rows
            [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 11.0, 12.0],
            [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 11.0, 12.0],
            population=[False, False, False, False, True, True, True, True],
            weight=[1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1],
        )
        result = check.compute(inp)
        # Target mean = 6.25 (all 8 rows), proxy unweighted = 10.0
        self.assertGreater(result.gap_unweighted, 0)
        self.assertEqual(result.proxy_mean_unweighted, 10.0)
        # Weights are uniform across proxy rows so weighted == unweighted
        # (equal weights don't change the mean)
        self.assertAlmostEqual(result.proxy_mean_weighted, 10.0)

    def test_perfect_correction(self) -> None:
        """gap_reduction = 1.0 when weighted proxy mean equals target mean."""
        check = WeightingEffectivenessCheck()
        # Target = all 5 rows: [0, 0, 0, 0, 10], mean = 2.0
        # Proxy = population=True rows: [0, 10], unweighted mean = 5.0
        # Gap unweighted = |5 - 2| = 3.0
        # Need weighted proxy mean = 2.0:
        #   (0*w1 + 10*w2)/(w1+w2) = 2 => w1 = 4*w2
        # Use w1=4, w2=1: weighted mean = 10/5 = 2.0, gap_reduction = 1.0
        inp = _make_input(
            [0.0, 0.0, 0.0, 0.0, 10.0],
            [0.0, 0.0, 0.0, 0.0, 10.0],
            population=[False, False, False, True, True],
            weight=[1.0, 1.0, 1.0, 4.0, 1.0],
        )
        result = check.compute(inp)
        self.assertAlmostEqual(result.gap_reduction, 1.0, places=5)

    def test_no_weights(self) -> None:
        """Without weights, gap_reduction = 0."""
        check = WeightingEffectivenessCheck()
        inp = _make_input(
            [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 11.0, 12.0],
            [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 11.0, 12.0],
            population=[False, False, False, False, True, True, True, True],
        )
        result = check.compute(inp)
        # No weights => weighted == unweighted => gap_reduction = 0
        self.assertAlmostEqual(result.gap_reduction, 0.0)
        self.assertAlmostEqual(result.proxy_mean_weighted, result.proxy_mean_unweighted)

    def test_weights_reduce_gap(self) -> None:
        """Weights bring proxy closer to target."""
        check = WeightingEffectivenessCheck()
        # Target rows: values [1, 2, 3, 4], mean=2.5
        # Proxy rows: values [10, 20, 30, 40], mean=25.0
        # With weights [10, 1, 1, 1]: weighted mean = (100+20+30+40)/13 ≈ 14.6
        # Gap: 25-2.5=22.5 unweighted, 14.6-2.5=12.1 weighted
        # gap_reduction = 1 - 12.1/22.5 ≈ 0.46
        inp = _make_input(
            [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            population=[False, False, False, False, True, True, True, True],
            weight=[1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0],
        )
        result = check.compute(inp)
        self.assertGreater(result.gap_reduction, 0.0)
        self.assertLess(result.gap_weighted, result.gap_unweighted)

    def test_per_covariate_gap_reduction(self) -> None:
        """Per-covariate gap reduction is computed."""
        check = WeightingEffectivenessCheck()
        inp = _make_input(
            [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            population=[False, False, False, False, True, True, True, True],
            weight=[1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0],
            covariates={
                "age": [25.0, 30.0, 35.0, 40.0, 50.0, 55.0, 60.0, 65.0],
            },
            covariate_cols=["age"],
        )
        result = check.compute(inp)
        self.assertIn("age", result.per_covariate_gap_reduction)
        age_stats = result.per_covariate_gap_reduction["age"]
        self.assertIn("gap_reduction", age_stats)
        self.assertIn("target_mean", age_stats)
        self.assertIn("proxy_mean_unweighted", age_stats)
        self.assertIn("proxy_mean_weighted", age_stats)

    def test_project_clamps(self) -> None:
        """Project clamps to [0, 1]."""
        check = WeightingEffectivenessCheck()
        # No weights => gap_reduction = 0 => project = 0
        inp = _make_input(
            [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            population=[False, False, False, False, True, True, True, True],
        )
        result = check.compute(inp)
        proj = check.project(result)
        self.assertGreaterEqual(proj, 0.0)
        self.assertLessEqual(proj, 1.0)

    def test_metadata(self) -> None:
        self.assertEqual(
            WeightingEffectivenessCheck.validation_level,
            ValidationLevel.REPRESENTATIVITY,
        )
        self.assertEqual(
            WeightingEffectivenessCheck.sub_property,
            RepresentativityProperty.WEIGHTING_EFFECTIVENESS,
        )

    def test_run_end_to_end(self) -> None:
        """End-to-end run with threshold."""
        check = WeightingEffectivenessCheck(threshold=0.3)
        inp = _make_input(
            [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            population=[False, False, False, False, True, True, True, True],
            weight=[1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0],
        )
        result = check.run(inp)
        self.assertEqual(result.check_name, "Weighting Effectiveness")
        # gap_reduction > 0.3 threshold => should pass
        self.assertTrue(result.thresholding_result)
