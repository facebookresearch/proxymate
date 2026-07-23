---
id: adjustments
title: Adjustments Reference
sidebar_position: 3
---

# Adjustments Reference

Adjustments are applied when diagnostic checks fail. Each adjustment targets a specific failure mode.

## Representativity Level Adjustments

Before any proxy-quality checks run, the **Representativity Level** decides whether the validation data must be reweighted to represent the target population.

### Importance weighting (IW)

When the validation sample is not representative of the domain population (for example, a review queue that over-represents flagged content), importance weights $w(x) = p_k(x) / p_{\mathcal{V}_k}(x)$ make any estimate computed on the sample target the population $P_k$. IW is a **sample-level** correction: it is supplied through the `weight_col` of `DataHandler` (or estimated propensity weights) and flows into every downstream Unit-, Estimate-, and Domain-level check — there is no separate corrector class.

```python
handler = DataHandler.from_dataframes(
    metrics_df=metrics_df,
    entities_df=entities_df,
    primary_outcome_name="primary",
    proxy_outcome_names=["surrogate"],
    weight_col="weight",   # importance / sampling weights
)
```

If the weights are too unstable to close the gap (low effective sample size), the Representativity checks flag the validation as unreliable rather than applying IW.

**Targets:** Labeling gap (Representativity failure)

## Unit Level Adjustments

**Import:** `from proxymate.adjustment.unit_level import ...`

:::note Auto-selection
`Validator.adjust()` applies **Platt scaling** by default for `Expected Calibration Error` failures. `IsotonicRecalibrator` and `MCGradRecalibrator` are available for manual use when miscalibration is non-monotone or varies across subgroups.
:::

### `AdditiveBiasCorrector`

Subtracts the estimated mean difference $E[Y^* - Y]$ from proxy scores.

```python
corrector = AdditiveBiasCorrector()
corrector.fit(check_input)           # estimate bias from paired data
corrected = corrector.transform(proxy_scores)  # apply correction
```

**Targets:** T-test failure (systematic unit bias)

### `PlattRecalibrator`

Fits logistic regression $P(Y=1 \mid Y^*) = \sigma(aY^* + b)$ for probability calibration.

```python
recal = PlattRecalibrator()
recal.fit(check_input)
calibrated = recal.transform(proxy_probabilities)
```

**Targets:** ECE failure (miscalibration)

### `IsotonicRecalibrator`

Fits a non-parametric monotone calibration function. More flexible than Platt but needs more data.

```python
recal = IsotonicRecalibrator()
recal.fit(check_input)
calibrated = recal.transform(proxy_probabilities)
```

**Targets:** ECE failure when miscalibration is non-monotone

### `MCGradRecalibrator`

Fits a LightGBM-based multicalibration model that enforces calibration *within data-driven subgroups*, not just globally. More flexible than Platt or isotonic scaling when miscalibration varies across subpopulations.

```python
recal = MCGradRecalibrator()
recal.fit(check_input)                        # uses covariate_cols for subgroups
calibrated = recal.transform(proxy_probabilities)
```

Requires the optional `mcgrad` package (install the `calibration` extra: `pip install "proxymate[calibration]"`). It is imported lazily, so `import proxymate` works without it.

**Targets:** Unit-level calibration failure, especially poor per-segment calibration

## Estimate Level Adjustments

**Import:** `from proxymate.adjustment.estimate_level import ...`

### `PPIEstimator`

Prediction-Powered Inference. Combines full-population proxy estimates with a small labeled sample to produce a debiased estimate:

$$\hat\theta^{\text{PPI}} = \hat\theta^*_{\text{all}} + (\hat\theta_{Y,\mathcal{V}} - \hat\theta^*_{Y^*,\mathcal{V}})$$

```python
ppi = PPIEstimator()
result = ppi.estimate(labeled_check_input, unlabeled_proxy_array)

print(result.ppi_estimate)      # debiased estimate
print(result.correction)        # bias adjustment term
print(result.standard_error)    # SE of the correction
```

**Targets:** Aggregate bias failure

:::warning Theorem 1 Gating

PPI is only applied when the proxy bias exceeds the correction's variance cost: $|\phi| > \sigma_\Delta / \sqrt{n}$. The `adjust()` method checks this automatically.

:::

## Domain Level Adjustments

**Import:** `from proxymate.adjustment.domain_level import ...`

### `RegressionCalibrator`

Fits a linear calibration model $\theta_Y = \gamma_0 + \gamma_1 \theta^*_{Y^*}$ across historical domains, then predicts the primary estimate in a target domain.

```python
calibrator = RegressionCalibrator()
calibrator.fit(meta_analysis_input)

target_proxy = Estimate(estimate_val=3.0, standard_error=0.2, ...)
result = calibrator.predict(target_proxy)

print(result.calibrated_estimate)   # calibrated point estimate + CI
print(result.slope, result.intercept, result.r_squared)
```

The calibrated CI incorporates both the proxy's sampling error and the residual cross-domain heterogeneity $\tau^2$:

$$I^{\text{cal}} = \left[\hat\gamma_0 + \hat\gamma_1 \hat\theta^* \pm z \sqrt{\hat\gamma_1^2 \sigma^2_* + \hat\tau^2}\right]$$

When `domain_covariances` are provided in the `MetaAnalysisInput`, the fit uses iterative GLS.

**Targets:** Aggregate association or effect sensitivity failure

### `EstimateLevelAdjustment`

Models residual proxy bias as a random effect $\phi_k \sim \mathcal{N}(\rho, \gamma^2)$ and estimates $(\rho, \gamma^2)$ via EM. Can be layered on top of any proxy estimator (including PPI).

```python
adj = EstimateLevelAdjustment()
adj.fit(meta_analysis_input)

result = adj.adjust(proxy_estimate)

print(result.adjusted_estimate)   # bias-corrected + CI inflated
print(result.rho)                 # estimated mean bias
print(result.gamma_squared)       # estimated bias variance
```

The adjusted interval subtracts the mean bias and inflates for heterogeneity:

$$I^* = \left[\hat\theta^* - \hat\rho \pm z \sqrt{\sigma^2_* + \hat\gamma^2}\right]$$

When `domain_covariances` are available, the SE of the within-domain difference accounts for the correlation: $\text{SE}(d_k) = \sqrt{\sigma^2_Y + \sigma^2_{Y^*} - 2\rho_k}$.

**Targets:** Heterogeneity failure, residual bias after other adjustments
