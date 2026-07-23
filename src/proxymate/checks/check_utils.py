# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import numpy as np
import numpy.typing as npt
from proxymate.types.inputs import CheckInput


def extract_paired_arrays(
    input: CheckInput,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Extract finite-valued primary and proxy arrays from a CheckInput.

    Converts both columns to numpy arrays, masks out rows where either
    value is non-finite (NaN/Inf), and returns the aligned pair.
    """
    primary = input.df[input.primary_col].to_numpy()
    proxy = input.df[input.proxy_col].to_numpy()
    mask = np.isfinite(primary) & np.isfinite(proxy)
    return primary[mask], proxy[mask]


def extract_paired_arrays_with_weights(
    input: CheckInput,
) -> Tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating] | None,
]:
    """Extract finite-valued primary, proxy, and optional weight arrays.

    Like extract_paired_arrays but also handles the weight column.
    Returns (primary, proxy, weights) where weights is None if no
    weight_col is set.
    """
    primary = input.df[input.primary_col].to_numpy()
    proxy = input.df[input.proxy_col].to_numpy()
    mask = np.isfinite(primary) & np.isfinite(proxy)

    weights = None
    if input.weight_col is not None:
        weights = input.df[input.weight_col].to_numpy()
        mask &= np.isfinite(weights)
        weights = weights[mask]

    return primary[mask], proxy[mask], weights
