# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from unittest import TestCase

from proxymate.planner.validation_planner import ValidationPlanner
from proxymate.types.enums import ProxyChallenge


class TestValidationPlanner(TestCase):
    def test_recommend_names_long_term(self) -> None:
        planner = ValidationPlanner()
        names = planner.recommend_names(ProxyChallenge.LONG_TERM_PRIMARY)
        self.assertIn("effect_sensitivity", names)
        self.assertIn("directional_alignment", names)
        self.assertIn("aggregate_association", names)
        self.assertIn("heterogeneity", names)

    def test_recommend_names_low_detectability(self) -> None:
        planner = ValidationPlanner()
        names = planner.recommend_names(ProxyChallenge.LOW_DETECTABILITY)
        self.assertIn("precision", names)
        self.assertIn("agreement", names)
        self.assertIn("inferential_validity", names)
        self.assertIn("aggregate_bias", names)

    def test_recommend_names_restricted_labels(self) -> None:
        planner = ValidationPlanner()
        names = planner.recommend_names(ProxyChallenge.RESTRICTED_LABELS)
        self.assertIn("coverage", names)
        self.assertIn("sample_representativity", names)
        self.assertIn("calibration", names)

    def test_recommend_names_combined_challenges(self) -> None:
        planner = ValidationPlanner()
        combined = ProxyChallenge.LONG_TERM_PRIMARY | ProxyChallenge.LOW_DETECTABILITY
        names = planner.recommend_names(combined)
        self.assertIn("effect_sensitivity", names)
        self.assertIn("precision", names)

    def test_explain_with_results(self) -> None:
        planner = ValidationPlanner()
        explanation = planner.explain(ProxyChallenge.LONG_TERM_PRIMARY)
        self.assertIn("Recommended checks", explanation)
        self.assertIn("effect_sensitivity", explanation)

    def test_recommend_returns_registered(self) -> None:
        planner = ValidationPlanner()
        checks = planner.recommend(ProxyChallenge.LONG_TERM_PRIMARY)
        self.assertEqual(checks, [])
