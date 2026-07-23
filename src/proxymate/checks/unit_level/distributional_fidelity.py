# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from proxymate.checks.base import CheckBase
from proxymate.checks.check_utils import extract_paired_arrays
from proxymate.types.enums import OutcomeType, UnitLevelProperty, ValidationLevel
from proxymate.types.inputs import CheckInput
from scipy import stats


@dataclass
class TVDistResult:
    tv_distance: float
    n_samples: int


class TVDistCheck(CheckBase[TVDistResult, CheckInput]):
    """Total Variation distance between primary and proxy distributions.

    Discretizes both distributions into equal-width bins and computes
    TV = 0.5 * Σ |p_i - q_i|. TV = 0 means identical distributions,
    TV = 1 means no overlap.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = (
        UnitLevelProperty.DISTRIBUTIONAL_FIDELITY
    )
    supported_outcome_types: ClassVar[list[OutcomeType]] = [OutcomeType.CONTINUOUS]

    def __init__(
        self,
        threshold: float | None = None,
        n_bins: int = 50,
    ) -> None:
        super().__init__(threshold=threshold)
        self.n_bins = n_bins

    def get_name(self) -> str:
        return "TV Distance"

    def get_description(self) -> str:
        return "Total Variation distance between primary and proxy distributions."

    def compute(self, input: CheckInput) -> TVDistResult:
        primary, proxy = extract_paired_arrays(input)

        all_vals = np.concatenate([primary, proxy])
        val_min, val_max = all_vals.min(), all_vals.max()
        if val_min == val_max:
            return TVDistResult(tv_distance=0.0, n_samples=len(primary))

        bins = np.linspace(val_min, val_max, self.n_bins + 1)

        hist_p, _ = np.histogram(primary, bins=bins, density=True)
        hist_q, _ = np.histogram(proxy, bins=bins, density=True)

        bin_width = bins[1] - bins[0]
        tv = float(0.5 * np.sum(np.abs(hist_p - hist_q)) * bin_width)
        return TVDistResult(tv_distance=tv, n_samples=len(primary))

    def project(self, result: TVDistResult) -> float:
        # Lower TV = better. Project as 1 - TV.
        return min(1.0, max(0.0, 1.0 - result.tv_distance))


@dataclass
class KSDistResult:
    ks_statistic: float
    p_value: float
    n_samples: int


class KSDistCheck(CheckBase[KSDistResult, CheckInput]):
    """Kolmogorov-Smirnov distance between primary and proxy distributions.

    Uses the two-sample KS test to measure the maximum difference
    between the empirical CDFs of primary and proxy.
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = (
        UnitLevelProperty.DISTRIBUTIONAL_FIDELITY
    )
    supported_outcome_types: ClassVar[list[OutcomeType]] = [OutcomeType.CONTINUOUS]

    def get_name(self) -> str:
        return "KS Distance"

    def get_description(self) -> str:
        return "Kolmogorov-Smirnov distance between primary and proxy distributions."

    def compute(self, input: CheckInput) -> KSDistResult:
        primary, proxy = extract_paired_arrays(input)

        result = stats.ks_2samp(primary, proxy)
        return KSDistResult(
            ks_statistic=float(result.statistic),
            p_value=float(result.pvalue),
            n_samples=len(primary),
        )

    def project(self, result: KSDistResult) -> float:
        # Lower KS stat = better. Project as 1 - KS.
        return min(1.0, max(0.0, 1.0 - result.ks_statistic))


@dataclass
class WassersteinResult:
    wasserstein_distance: float
    normalized_distance: float
    n_samples: int


class WassersteinDistCheck(CheckBase[WassersteinResult, CheckInput]):
    """Wasserstein-1 (Earth Mover's) distance between primary and proxy distributions.

    Computes W_1(F_Y, F_{Ŷ}) = ∫|F_Y(x) - F_{Ŷ}(x)| dx.
    Normalized by SD(Y) for scale-free comparison (Paper eq. 11).
    """

    validation_level: ClassVar[ValidationLevel] = ValidationLevel.UNIT_LEVEL
    sub_property: ClassVar[UnitLevelProperty] = (
        UnitLevelProperty.DISTRIBUTIONAL_FIDELITY
    )
    supported_outcome_types: ClassVar[list[OutcomeType]] = [OutcomeType.CONTINUOUS]

    def get_name(self) -> str:
        return "Wasserstein-1 Distance"

    def get_description(self) -> str:
        return "Earth Mover's Distance between primary and proxy distributions."

    def compute(self, input: CheckInput) -> WassersteinResult:
        primary, proxy = extract_paired_arrays(input)

        w1 = float(stats.wasserstein_distance(primary, proxy))
        sd_y = float(np.std(primary, ddof=1)) if len(primary) > 1 else 1.0
        normalized = w1 / sd_y if sd_y > 0 else 0.0

        return WassersteinResult(
            wasserstein_distance=w1,
            normalized_distance=normalized,
            n_samples=len(primary),
        )

    def project(self, result: WassersteinResult) -> float:
        # Lower normalized distance = better. Use exp(-d) for smooth [0,1] mapping.
        return min(1.0, max(0.0, np.exp(-result.normalized_distance)))
