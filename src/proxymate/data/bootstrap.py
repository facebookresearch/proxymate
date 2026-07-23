# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import pandas as pd


class BootstrapSampler:
    """Pure bootstrap sampling logic using numpy RNG.

    Seeded RNG ensures identical samples across proxies and across runs.
    No I/O — only generates row index mappings.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def sample_indices(self, n: int) -> list[int]:
        """Return a list of n indices sampled with replacement from range(n)."""
        if n <= 0:
            raise ValueError("n must be positive")
        return self._rng.integers(0, n, size=n).tolist()

    def generate_samples(self, n: int, n_iterations: int) -> np.ndarray:
        """Pre-generate all N bootstrap index samples upfront.

        Returns an int32 array of shape (n_iterations, n). Keeping indices
        as a numpy array avoids the ~7x memory overhead of nested Python
        lists for large sample counts.

        All samples share the same RNG sequence, so with a fixed seed
        the results are deterministic and reproducible.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        return self._rng.integers(
            0, n, size=(n_iterations, n), dtype=np.dtype(np.int32)
        )


def resample_df(
    df: pd.DataFrame,
    indices: list[int] | np.ndarray,
) -> pd.DataFrame:
    """Resample a DataFrame by row indices (with replacement).

    Args:
        df: Source DataFrame.
        indices: Row indices to select (may contain duplicates).
            Accepts a Python list or a numpy array.

    Returns:
        Resampled DataFrame with reset index.
    """
    return df.iloc[indices].reset_index(drop=True)
