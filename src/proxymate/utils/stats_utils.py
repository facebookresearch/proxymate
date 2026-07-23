# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Tuple

import numpy as np
import scipy.stats as st


def alpha_to_z(alpha: float) -> float:
    z = st.norm.ppf(1 - (alpha * 0.5))
    return z


def conf_to_z(conf: int) -> float:
    alpha = (100 - conf) / 100
    return alpha_to_z(alpha)


def get_ci(delta: float, se: float, conf: int = 90) -> Tuple[float, float]:
    ci = (delta - conf_to_z(conf) * se, delta + conf_to_z(conf) * se)
    return ci


def h(list_values: List[float]) -> float:
    sigma = float(np.std(list_values))
    iqr_value = float(st.iqr(list_values))
    n = len(list_values)
    return 0.9 * min(sigma, iqr_value / 1.34) * n ** (-1 / 5)


def K(y: float) -> float:
    return 1 / np.sqrt(2 * np.pi) * np.exp(-((y) ** 2))


def calculate_median(
    list_values: List[float], method: str = "kde"
) -> Tuple[float, float, int]:
    estimate_sample_size = len(list_values)
    estimate_val = np.median(list_values)
    if method == "woodruff":
        list_values_sorted = sorted(list_values, reverse=False)
        estimate_var = (
            0.5
            * (
                list_values_sorted[
                    int(0.5 * (estimate_sample_size + np.sqrt(estimate_sample_size)))
                ]
                - list_values_sorted[
                    int(0.5 * (estimate_sample_size - np.sqrt(estimate_sample_size)))
                ]
            )
        ) ** 2
    elif method == "kde":
        h_value = h(list_values)
        estimate_var = 1 / (
            4
            * estimate_sample_size
            * (
                np.sum(K((estimate_val - np.array(list_values)) / h_value))
                / (estimate_sample_size * h_value)
            )
            ** 2
        )
    else:
        estimate_var = None
        raise ValueError("method must be either 'woodruff' or 'kde'")

    return estimate_val, estimate_var, estimate_sample_size


def calculate_average(list_values: List[float]) -> Tuple[float, float, int]:
    estimate_sample_size = len(list_values)
    estimate_val = np.mean(list_values)
    estimate_var = np.var(list_values)
    return estimate_val, estimate_var, estimate_sample_size


def is_binary(arr: np.ndarray) -> bool:
    """Check if an array contains only 0 and 1 values."""
    unique = np.unique(arr[np.isfinite(arr)])
    return bool(len(unique) <= 2 and set(unique.tolist()).issubset({0.0, 1.0}))


def is_probability(arr: np.ndarray) -> bool:
    """Check if an array contains values in [0, 1] (but is not purely binary)."""
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return False
    return bool(float(np.min(finite)) >= 0.0 and float(np.max(finite)) <= 1.0)
