# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any

from proxymate.checks.base import CheckBase
from proxymate.types.enums import ProxyChallenge, ValidationLevel

_CHALLENGE_ROUTING: dict[ProxyChallenge, list[str]] = {
    ProxyChallenge.LONG_TERM_PRIMARY: [
        "aggregate_association",
        "effect_sensitivity",
        "directional_alignment",
        "heterogeneity",
        "prediction_accuracy",
    ],
    ProxyChallenge.LOW_DETECTABILITY: [
        "precision",
        "agreement",
        "inferential_validity",
        "aggregate_bias",
    ],
    ProxyChallenge.RESTRICTED_LABELS: [
        "coverage",
        "sample_representativity",
        "effective_sample_size",
        "weight_extremity",
        "weighting_effectiveness",
        "calibration",
        "discrimination",
        "aggregate_bias",
    ],
    ProxyChallenge.LOW_SENSITIVITY: [
        "directional_alignment",
        "effect_sensitivity",
        "discrimination",
        "accuracy",
    ],
}


class ValidationPlanner:
    """Maps ProxyChallenge to recommended check names.

    Encodes the routing table from the research formalization to help
    users select appropriate checks based on their proxy validation
    challenge. The validator auto-detects the data scenario (Cases 1-4)
    from the inputs provided.
    """

    _registry: dict[str, CheckBase[Any, Any]]

    def __init__(
        self,
        check_registry: dict[str, CheckBase[Any, Any]] | None = None,
    ) -> None:
        self._registry = check_registry or {}

    def register(self, check: CheckBase[Any, Any]) -> None:
        """Register a check instance for lookup by sub-property name."""
        self._registry[check.sub_property.value] = check

    def recommend_names(
        self,
        challenge: ProxyChallenge,
    ) -> list[str]:
        """Return check names appropriate for this challenge."""
        results: list[str] = []

        for single_challenge in ProxyChallenge:
            if not (challenge & single_challenge):
                continue

            for check_name in _CHALLENGE_ROUTING.get(single_challenge, []):
                if check_name not in results:
                    results.append(check_name)

        return results

    def recommend(
        self,
        challenge: ProxyChallenge,
    ) -> list[CheckBase[Any, Any]]:
        """Return instantiated checks appropriate for this challenge.

        Only returns checks that have been registered. Unregistered check names
        are silently skipped — call ``recommend_names`` to see all recommendations
        regardless of registration.
        """
        names = self.recommend_names(challenge)
        return [self._registry[name] for name in names if name in self._registry]

    def explain(
        self,
        challenge: ProxyChallenge,
    ) -> str:
        """Human-readable explanation of recommended checks."""
        names = self.recommend_names(challenge)
        if not names:
            return (
                f"No checks recommended for challenge={challenge.name}. "
                f"The routing table may not cover this challenge."
            )

        lines = [
            f"Recommended checks for {challenge.name}:",
        ]
        for i, name in enumerate(names, 1):
            registered = name in self._registry
            status = "" if registered else " (not registered)"
            lines.append(f"  {i}. {name}{status}")

        return "\n".join(lines)

    def auto_checks(
        self,
        challenge: ProxyChallenge,
    ) -> dict[str, list[CheckBase[Any, Any]]]:
        """Return checks grouped by category for use with Validator.validate().

        Returns a dict with keys matching the validate() parameter names:
        'unit_level_checks', 'aggregate_checks', 'bootstrap_checks',
        'meta_analysis_checks', 'representativity_checks'.
        """
        checks = self.recommend(challenge)

        grouped: dict[str, list[CheckBase[Any, Any]]] = {
            "unit_level_checks": [],
            "aggregate_checks": [],
            "bootstrap_checks": [],
            "meta_analysis_checks": [],
            "representativity_checks": [],
        }

        for check in checks:
            level = check.validation_level
            if level == ValidationLevel.REPRESENTATIVITY:
                grouped["representativity_checks"].append(check)
            elif level == ValidationLevel.ESTIMATE_LEVEL:
                grouped["aggregate_checks"].append(check)
            elif level == ValidationLevel.DOMAIN_LEVEL:
                grouped["meta_analysis_checks"].append(check)
            else:
                grouped["unit_level_checks"].append(check)

        return {k: v for k, v in grouped.items() if v}


def build_default_planner() -> ValidationPlanner:
    """Create a planner pre-loaded with all standard proxymate checks."""
    from proxymate.checks.domain_level.meta_analysis import (
        AggregateAssociationCheck,
        DirectionalAlignmentCheck,
        EffectSensitivityCheck,
        HeterogeneityCheck,
        PredictionAccuracyCheck,
    )
    from proxymate.checks.estimate_level.single_domain import (
        AggregateBiasCheck,
        CICoverageCheck,
        CIOverlapCheck,
        EffectSizeAgreementCheck,
        InferentialValidityCheck,
        RelativePrecisionCheck,
        SignAgreementCheck,
        UnbiasednessCheck,
    )
    from proxymate.checks.representativity_level.representativity import (
        CoverageCheck,
        EffectiveSampleSizeCheck,
        SampleRepresentativityCheck,
        WeightExtremeCheck,
        WeightingEffectivenessCheck,
    )
    from proxymate.checks.unit_level.accuracy import MAPECheck, TTestCheck
    from proxymate.checks.unit_level.agreement import (
        BlandAltmanCheck,
        DirectionalConcordanceCheck,
        LinCCCCheck,
    )
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
        KendallTauCheck,
        MCCCheck,
        NPVCheck,
        PrecisionRecallAUCCheck,
        ROCAUCCheck,
        SensitivityCheck,
        SpearmanCorrelationCheck,
        SpecificityCheck,
    )
    from proxymate.checks.unit_level.distributional_fidelity import (
        KSDistCheck,
        TVDistCheck,
        WassersteinDistCheck,
    )
    from proxymate.checks.unit_level.precision import (
        PearsonCorrelationCheck,
        RMSECheck,
        RSquaredCheck,
    )

    planner = ValidationPlanner()
    all_checks: list[CheckBase[Any, Any]] = [
        # Unit level — Accuracy
        TTestCheck(),
        MAPECheck(),
        # Unit level — Precision
        RMSECheck(),
        RSquaredCheck(),
        PearsonCorrelationCheck(),
        # Unit level — Agreement
        LinCCCCheck(),
        BlandAltmanCheck(),
        DirectionalConcordanceCheck(),
        MCCCheck(),
        # Unit level — Calibration
        ECECheck(),
        BrierScoreCheck(),
        BrierSkillScoreCheck(),
        CalibrationRatioCheck(),
        KuiperCalibrationCheck(),
        # Unit level — Discrimination
        ROCAUCCheck(),
        PrecisionRecallAUCCheck(),
        KendallTauCheck(),
        SpearmanCorrelationCheck(),
        ClassificationAccuracyCheck(),
        ClassificationPrecisionCheck(),
        SensitivityCheck(),
        SpecificityCheck(),
        NPVCheck(),
        # Unit level — Distributional fidelity
        KSDistCheck(),
        TVDistCheck(),
        WassersteinDistCheck(),
        # Representativity
        CoverageCheck(),
        SampleRepresentativityCheck(),
        EffectiveSampleSizeCheck(),
        WeightExtremeCheck(),
        WeightingEffectivenessCheck(),
        # Estimate level
        AggregateBiasCheck(),
        InferentialValidityCheck(),
        UnbiasednessCheck(),
        CICoverageCheck(),
        CIOverlapCheck(),
        RelativePrecisionCheck(),
        SignAgreementCheck(),
        EffectSizeAgreementCheck(),
        # Domain level
        AggregateAssociationCheck(),
        EffectSensitivityCheck(),
        DirectionalAlignmentCheck(),
        PredictionAccuracyCheck(),
        HeterogeneityCheck(),
    ]
    for check in all_checks:
        planner.register(check)

    return planner
