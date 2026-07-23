# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from enum import auto, Enum, Flag


class ValidationLevel(Enum):
    """Primary organizing axis: representativity, unit-level, estimate-level,
    and domain-level."""

    UNIT_LEVEL = "unit_level"
    ESTIMATE_LEVEL = "estimate_level"
    DOMAIN_LEVEL = "domain_level"
    REPRESENTATIVITY = "representativity"


class UnitLevelProperty(Enum):
    """Sub-properties of unit-level (measurement quality) validation.

    Each property answers a different question about how well
    the proxy Ŷ_i tracks the primary Y_i at the individual unit level.
    """

    ACCURACY = "accuracy"  # E[Ŷ - Y] = 0?
    PRECISION = "precision"  # Var(Ŷ - Y) small?
    AGREEMENT = "agreement"  # CCC = ρ · C_b
    CALIBRATION = "calibration"  # E[Y | Ŷ = q] = g(q)?
    DISCRIMINATION = "discrimination"  # Preserves relative ordering?
    DISTRIBUTIONAL_FIDELITY = "distributional_fidelity"  # F_Ŷ ≈ F_Y?


class EstimateLevelProperty(Enum):
    """Sub-properties of estimate-level (decision validity) validation.

    Each property answers a different question about whether
    the proxy-based aggregate estimate θ_Ŷ tracks the primary θ_Y
    within a single domain.
    """

    AGGREGATE_BIAS = "aggregate_bias"
    INFERENTIAL_VALIDITY = "inferential_validity"
    UNBIASEDNESS = "unbiasedness"
    CI_COVERAGE = "ci_coverage"
    CI_OVERLAP = "ci_overlap"
    RELATIVE_PRECISION = "relative_precision"
    SIGN_AGREEMENT = "sign_agreement"
    EFFECT_SIZE_AGREEMENT = "effect_size_agreement"


class DomainLevelProperty(Enum):
    """Sub-properties of domain-level (cross-domain transportability) validation.

    Each property answers a different question about whether
    the proxy-primary relationship is stable across domains.
    """

    AGGREGATE_ASSOCIATION = "aggregate_association"
    EFFECT_SENSITIVITY = "effect_sensitivity"
    DIRECTIONAL_ALIGNMENT = "directional_alignment"
    PREDICTION_ACCURACY = "prediction_accuracy"
    HETEROGENEITY = "heterogeneity"


class OutcomeType(Enum):
    """Primary/proxy outcome type combinations."""

    CONTINUOUS = "continuous"  # Y ∈ ℝ, Ŷ ∈ ℝ
    BINARY_BINARY = "binary_binary"  # Y ∈ {0,1}, Ŷ ∈ {0,1}
    BINARY_PROBABILITY = "binary_prob"  # Y ∈ {0,1}, Ŷ ∈ [0,1]
    CATEGORICAL = "categorical"  # Y ∈ C, Ŷ ∈ C


class ProxyChallenge(Flag):
    """Combinable flags for proxy validation challenges.

    A proxy can face multiple challenges simultaneously.
    Usage:
        challenge = ProxyChallenge.LONG_TERM_PRIMARY | ProxyChallenge.RESTRICTED_LABELS
    """

    LONG_TERM_PRIMARY = auto()  # Primary takes long to materialize
    LOW_DETECTABILITY = auto()  # High variance, low base rate
    RESTRICTED_LABELS = auto()  # Limited labeled data
    LOW_SENSITIVITY = auto()  # Primary too blunt


class RepresentativityProperty(Enum):
    """Sub-properties of representativity (population gap) diagnostics.

    Each property answers a different question about how well
    the proxy/labeled population P represents the target population T.
    """

    COVERAGE = "coverage"  # N_P / N_T coverage rate
    SAMPLE_REPRESENTATIVITY = "sample_representativity"  # Covariate distribution gaps
    EFFECTIVE_SAMPLE_SIZE = "effective_sample_size"  # (Σw)² / Σw²
    WEIGHT_EXTREMITY = "weight_extremity"  # max(w)/min(w), weight distribution
    WEIGHTING_EFFECTIVENESS = "weighting_effectiveness"  # Gap reduction after IW
