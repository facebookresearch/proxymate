# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from unittest import TestCase

import numpy as np
import pandas as pd
from proxymate.data.bootstrap import BootstrapSampler, resample_df


class TestBootstrapSampler(TestCase):
    def test_sample_indices_length(self) -> None:
        sampler = BootstrapSampler(seed=42)
        indices = sampler.sample_indices(10)
        self.assertEqual(len(indices), 10)

    def test_sample_indices_range(self) -> None:
        sampler = BootstrapSampler(seed=42)
        indices = sampler.sample_indices(5)
        for idx in indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, 5)

    def test_deterministic_with_seed(self) -> None:
        s1 = BootstrapSampler(seed=123)
        s2 = BootstrapSampler(seed=123)
        self.assertEqual(s1.sample_indices(10), s2.sample_indices(10))

    def test_different_seeds_differ(self) -> None:
        s1 = BootstrapSampler(seed=1)
        s2 = BootstrapSampler(seed=2)
        # With n=100, different seeds should produce different samples
        self.assertNotEqual(s1.sample_indices(100), s2.sample_indices(100))

    def test_generate_samples_shape(self) -> None:
        sampler = BootstrapSampler(seed=42)
        samples = sampler.generate_samples(n=10, n_iterations=5)
        self.assertIsInstance(samples, np.ndarray)
        self.assertEqual(samples.shape, (5, 10))
        self.assertEqual(samples.dtype, np.int32)

    def test_invalid_n(self) -> None:
        sampler = BootstrapSampler()
        with self.assertRaises(ValueError):
            sampler.sample_indices(0)
        with self.assertRaises(ValueError):
            sampler.sample_indices(-1)


class TestResampleDf(TestCase):
    def test_basic_resample(self) -> None:
        df = pd.DataFrame({"a": [10, 20, 30], "b": ["x", "y", "z"]})
        resampled = resample_df(df, [0, 0, 2])
        self.assertEqual(len(resampled), 3)
        self.assertEqual(list(resampled["a"]), [10, 10, 30])
        self.assertEqual(list(resampled["b"]), ["x", "x", "z"])

    def test_index_reset(self) -> None:
        df = pd.DataFrame({"a": [10, 20, 30]})
        resampled = resample_df(df, [2, 0, 1])
        self.assertEqual(list(resampled.index), [0, 1, 2])

    def test_duplicates(self) -> None:
        df = pd.DataFrame({"v": [1, 2, 3, 4]})
        resampled = resample_df(df, [1, 1, 1, 1])
        self.assertTrue(all(v == 2 for v in resampled["v"]))
