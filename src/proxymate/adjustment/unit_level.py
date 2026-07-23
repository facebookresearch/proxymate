# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from proxymate.types.inputs import CheckInput


class Recalibrator(ABC):
    """Base class for unit-level recalibration methods.

    Recalibrators learn a mapping from proxy scores to calibrated scores
    using labeled data, then apply that mapping to new proxy scores.
    Addresses unit-level calibration (L1.4) failures.
    """

    @abstractmethod
    def fit(self, input: CheckInput) -> "Recalibrator":
        """Fit the recalibration model on labeled (primary, proxy) pairs."""
        ...

    @abstractmethod
    def transform(
        self, proxy_scores: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Apply the learned recalibration to new proxy scores."""
        ...

    def fit_transform(self, input: CheckInput) -> npt.NDArray[np.floating]:
        """Fit on the input data and return the transformed proxy scores."""
        self.fit(input)
        complete = input.df[[input.proxy_col, input.primary_col]].dropna()
        return self.transform(complete[input.proxy_col].values.astype(float))


@dataclass
class PlattRecalibrator(Recalibrator):
    """Platt scaling: logistic regression P(Y=1 | Ŷ) = σ(a·Ŷ + b).

    Fits a logistic regression with the proxy score as the single feature
    and the binary primary outcome as the target. The fitted model maps
    uncalibrated proxy scores to calibrated probabilities.

    Appropriate when the proxy produces scores in [0, 1] and the primary
    is binary. Fixes L1.4 (calibration) when the calibration curve is
    approximately sigmoidal.
    """

    a: float = 0.0
    b: float = 0.0
    _fitted: bool = False

    def fit(self, input: CheckInput) -> "PlattRecalibrator":
        from scipy.optimize import minimize

        complete = input.df[[input.proxy_col, input.primary_col]].dropna()
        proxy_arr: npt.NDArray[np.floating] = complete[input.proxy_col].values.astype(
            float
        )
        primary_arr: npt.NDArray[np.floating] = complete[
            input.primary_col
        ].values.astype(float)

        def neg_log_likelihood(params: npt.NDArray[np.floating]) -> float:
            a, b = params[0], params[1]
            p = 1.0 / (1.0 + np.exp(-(a * proxy_arr + b)))
            p = np.clip(p, 1e-12, 1.0 - 1e-12)
            return -float(
                np.sum(primary_arr * np.log(p) + (1 - primary_arr) * np.log(1 - p))
            )

        result = minimize(
            neg_log_likelihood, x0=np.array([1.0, 0.0]), method="L-BFGS-B"
        )
        self.a = float(result.x[0])
        self.b = float(result.x[1])
        self._fitted = True
        return self

    def transform(
        self, proxy_scores: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        if not self._fitted:
            raise RuntimeError("PlattRecalibrator has not been fitted yet.")
        result: npt.NDArray[np.floating] = np.true_divide(
            1.0, 1.0 + np.exp(-(self.a * proxy_scores + self.b))
        )
        return result


@dataclass
class IsotonicRecalibrator(Recalibrator):
    """Isotonic regression recalibration.

    Fits a non-parametric, monotonically increasing function from proxy
    scores to primary outcomes. More flexible than Platt scaling — does
    not assume a sigmoidal relationship.

    Fixes L1.4 (calibration) for any monotone miscalibration shape.
    """

    _isotonic: Any = None  # sklearn.isotonic.IsotonicRegression, lazily imported
    _fitted: bool = False

    def fit(self, input: CheckInput) -> "IsotonicRecalibrator":
        from sklearn.isotonic import IsotonicRegression

        complete = input.df[[input.proxy_col, input.primary_col]].dropna()
        proxy = complete[input.proxy_col].values.astype(float)
        primary = complete[input.primary_col].values.astype(float)

        self._isotonic = IsotonicRegression(out_of_bounds="clip")
        self._isotonic.fit(proxy, primary)
        self._fitted = True
        return self

    def transform(
        self, proxy_scores: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        if not self._fitted or self._isotonic is None:
            raise RuntimeError("IsotonicRecalibrator has not been fitted yet.")
        return self._isotonic.transform(proxy_scores)


@dataclass
class AdditiveBiasCorrector(Recalibrator):
    """Additive bias correction: Ŷ_corrected = Ŷ - mean(Ŷ - Y).

    Estimates the systematic additive bias from labeled data and subtracts
    it from proxy scores. Fixes L1.1 (accuracy) when the proxy has a
    constant offset from the primary.
    """

    bias: float = 0.0
    _fitted: bool = False

    def fit(self, input: CheckInput) -> "AdditiveBiasCorrector":
        complete = input.df[[input.proxy_col, input.primary_col]].dropna()
        proxy = complete[input.proxy_col].values.astype(float)
        primary = complete[input.primary_col].values.astype(float)

        self.bias = float(np.mean(proxy - primary))
        self._fitted = True
        return self

    def transform(
        self, proxy_scores: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        if not self._fitted:
            raise RuntimeError("AdditiveBiasCorrector has not been fitted yet.")
        return proxy_scores - self.bias


@dataclass
class MCGradRecalibrator(Recalibrator):
    """MCGrad multicalibration for subgroup-aware recalibration.

    Fits a LightGBM-based multicalibration model that ensures calibration
    holds within data-driven subgroups, not just globally. More flexible
    than Platt or isotonic scaling when miscalibration varies across
    subpopulations.

    Requires the ``mcgrad`` package (optional dependency:
    ``pip install proxymate[calibration]``).

    Fixes Unit Level calibration failures, especially when per-segment
    calibration is poor.
    """

    _mcgrad: Any = None
    _fitted: bool = False
    _categorical_feature_cols: list[str] | None = None
    _numerical_feature_cols: list[str] | None = None

    def fit(self, input: CheckInput) -> "MCGradRecalibrator":
        import pandas as pd
        from mcgrad.methods import MCGrad  # pyre-ignore[import-error]

        complete = input.df[[input.proxy_col, input.primary_col]].dropna()
        df_train = pd.DataFrame(
            {
                "prediction": complete[input.proxy_col].values.astype(float),
                "label": complete[input.primary_col].values.astype(float),
            }
        )

        cat_cols: list[str] | None = None
        num_cols: list[str] | None = None
        if input.covariate_cols:
            valid_covs = [c for c in input.covariate_cols if c in input.df.columns]
            if valid_covs:
                cov_data = input.df.loc[complete.index, valid_covs]
                for col in valid_covs:
                    df_train[col] = cov_data[col].values
                cat_cols = [
                    c
                    for c in valid_covs
                    if df_train[c].dtype == object
                    or df_train[c].dtype.name == "category"
                ]
                num_cols = [c for c in valid_covs if c not in (cat_cols or [])]
                cat_cols = cat_cols or None
                num_cols = num_cols or None

        self._categorical_feature_cols = cat_cols
        self._numerical_feature_cols = num_cols

        self._mcgrad = MCGrad()
        self._mcgrad.fit(
            df_train=df_train,
            prediction_column_name="prediction",
            label_column_name="label",
            categorical_feature_column_names=cat_cols,
            numerical_feature_column_names=num_cols,
        )
        self._fitted = True
        return self

    def transform(
        self, proxy_scores: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        if not self._fitted or self._mcgrad is None:
            raise RuntimeError("MCGradRecalibrator has not been fitted yet.")

        import pandas as pd

        df_pred = pd.DataFrame({"prediction": proxy_scores.astype(float)})
        return self._mcgrad.predict(
            df=df_pred,
            prediction_column_name="prediction",
        )
