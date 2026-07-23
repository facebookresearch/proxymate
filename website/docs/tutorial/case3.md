---
id: case3
title: "Case 3: Aggregate-Only Domains"
sidebar_position: 4
---

# Case 3: Aggregate-Only Domains

**Scenario:** You have pre-computed paired estimates $(\hat\theta_k, \hat\theta^*_k)$ from $K$ domains (e.g., experiments). No unit-level data is available — only summary statistics. You want to assess cross-domain transportability.

## Setup

```python
import numpy as np
from proxymate.types.data import Estimate
from proxymate.types.inputs import MetaAnalysisInput

np.random.seed(42)
K = 15

# Simulate: primary ATE ≈ 0.8 * proxy ATE + noise
proxy_ates = np.random.normal(2.0, 1.5, K)
primary_ates = 0.8 * proxy_ates + np.random.normal(0, 0.4, K)

def make_estimate(val, se=0.3, n=500):
    return Estimate(
        confidence_interval_level=0.95,
        estimate_val=float(val),
        lower_bound=float(val - 1.96 * se),
        upper_bound=float(val + 1.96 * se),
        sample_size=n,
        standard_error=se,
    )

meta_input = MetaAnalysisInput(
    domain_estimates=[
        (make_estimate(p), make_estimate(x))
        for p, x in zip(primary_ates, proxy_ates)
    ],
    domain_names=[f"exp_{k:03d}" for k in range(K)],
)
```

### With Covariances (Optional)

If you know the within-domain covariance between the primary and proxy ATEs:

```python
# For a two-arm experiment:
# cov(ATE_primary, ATE_proxy) = cov(Y, Y*)/n_treat + cov(Y, Y*)/n_control
covariances = [0.05] * K  # example: constant covariance

meta_input = MetaAnalysisInput(
    domain_estimates=[(make_estimate(p), make_estimate(x)) for p, x in zip(primary_ates, proxy_ates)],
    domain_names=[f"exp_{k:03d}" for k in range(K)],
    domain_covariances=covariances,
)
```

When covariances are provided, the meta-regression uses iterative GLS that accounts for measurement error in both variables.

## Run Domain Level Checks

```python
from proxymate.orchestrator.validator import Validator
from proxymate.checks.domain_level.meta_analysis import (
    AggregateAssociationCheck,
    EffectSensitivityCheck,
    DirectionalAlignmentCheck,
    PredictionAccuracyCheck,
    HeterogeneityCheck,
)

report = Validator().validate(
    meta_analysis_checks=[
        AggregateAssociationCheck(),
        EffectSensitivityCheck(),
        DirectionalAlignmentCheck(),
        PredictionAccuracyCheck(),
        HeterogeneityCheck(),
    ],
    meta_analysis_input=meta_input,
)
```

## Interpret Results

```python
for proxy_col, result in report.per_proxy_results.items():
    df = result.to_dataframe()
    print(df[["check_name", "projection", "threshold", "passed"]].to_string(index=False))
```

Key things to look for:

- **R² high** → proxy effect predicts primary effect well
- **Slope ≈ 1** → one-to-one relationship (no systematic scaling)
- **I² low** → relationship is stable across domains
- **I² high** → relationship varies, prediction intervals should be widened

## Apply Domain-Level Adjustments

```python
from proxymate.adjustment.domain_level import RegressionCalibrator, EstimateLevelAdjustment

# Regression calibration: correct slope and intercept
calibrator = RegressionCalibrator()
calibrator.fit(meta_input)

new_proxy = make_estimate(3.0, se=0.2)
calibrated = calibrator.predict(new_proxy)
print(f"Raw proxy: {new_proxy.estimate_val:.2f}")
print(f"Calibrated: {calibrated.calibrated_estimate.estimate_val:.2f}")
print(f"CI: [{calibrated.calibrated_estimate.lower_bound:.2f}, {calibrated.calibrated_estimate.upper_bound:.2f}]")

# Estimate-level adjustment: model residual bias as random effect
adj = EstimateLevelAdjustment()
adj.fit(meta_input)
adjusted = adj.adjust(new_proxy)
print(f"Adjusted: {adjusted.adjusted_estimate.estimate_val:.2f} (ρ̂={adjusted.rho:.3f}, γ̂²={adjusted.gamma_squared:.4f})")
```
