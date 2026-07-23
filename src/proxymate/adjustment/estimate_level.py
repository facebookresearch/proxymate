# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from proxymate.types.inputs import CheckInput


@dataclass
class PPIResult:
    """Result of Prediction-Powered Inference estimation."""

    ppi_estimate: float
    proxy_only_estimate: float
    correction: float
    standard_error: float


class PPIEstimator:
    """Prediction-Powered Inference for mean estimation.

    Combines a large unlabeled dataset (proxy-only) with a small labeled
    dataset (primary + proxy) to produce a de-biased estimate:

        θ̂_PPI = θ̂_proxy_all + (θ̂_primary_labeled - θ̂_proxy_labeled)

    The correction term (θ̂_primary_labeled - θ̂_proxy_labeled) removes
    the systematic bias of the proxy, while leveraging the proxy's
    full-population coverage for variance reduction.

    Addresses L2.1 (aggregate bias) when labeled data is scarce but
    proxy predictions are available for the entire population.
    """

    def estimate(
        self,
        labeled_input: CheckInput,
        unlabeled_proxy: npt.NDArray[np.floating],
    ) -> PPIResult:
        """Compute the PPI estimate.

        Args:
            labeled_input: Labeled data with both primary and proxy columns.
            unlabeled_proxy: Proxy predictions for the full (unlabeled) population.
        """
        complete = labeled_input.df[
            [labeled_input.primary_col, labeled_input.proxy_col]
        ].dropna()
        primary_labeled = complete[labeled_input.primary_col].values.astype(float)
        proxy_labeled = complete[labeled_input.proxy_col].values.astype(float)

        proxy_all_mean = float(np.mean(unlabeled_proxy))
        primary_labeled_mean = float(np.mean(primary_labeled))
        proxy_labeled_mean = float(np.mean(proxy_labeled))

        correction = primary_labeled_mean - proxy_labeled_mean
        ppi_estimate = proxy_all_mean + correction

        # Standard error: SE of the correction term
        n_labeled = len(primary_labeled)
        residuals = primary_labeled - proxy_labeled
        se_correction = float(np.std(residuals, ddof=1) / np.sqrt(n_labeled))

        return PPIResult(
            ppi_estimate=ppi_estimate,
            proxy_only_estimate=proxy_all_mean,
            correction=correction,
            standard_error=se_correction,
        )
